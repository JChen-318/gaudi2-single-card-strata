// The device Q8_1 activation quantizer (hpu/tpc/activation_quant.c) against three authorities:
//   (1) an independent transcription of the pinned CUDA law - native_mmvq.cu:155
//       native_quantize_q8_1_kernel with q8_1_finite.hpp:18 (the finite clamp), :23 (round half away
//       from zero, the int8 clamp, the all-zero block) and :29 (both metadata fields stored as fp16),
//       taking the conversion from the pinned include/strata/kernels/f16_bits.hpp and never from the
//       kernel's own copy;
//   (2) this backend's host quantizer (packed_quant.cpp quantize_q8_activation), which feeds the graph
//       today and is already dual-authority tested;
//   (3) the consumer: the bytes and metadata the kernel writes are handed to the real packed dot,
//       which must return bit-identical results to the host-produced activation. That proves the
//       layout contract ({columns,batch} U8 codes, {2,columns/32,batch} F32 metadata) instead of
//       asserting it.
// The kernel's fp16 converter is pinned over the whole fp16 domain plus the overflow, subnormal and
// exact-tie cases, because a transcription of that converter is the failure mode the pinned header
// documents: a private copy conflated a finite-but-too-large value with an inf/NaN, and no O(1)
// fixture could see it. Ties, the fp32-vs-fp16 scale distinction, a block whose raw sum exceeds the
// largest finite half, an all-zero block, and the inputs the host gate refuses before upload are each
// covered on purpose.
// Host emulator only: no ISA execution, no device, no performance claim.
#include "strata/hpu/packed_quant.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "packed_emulator.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>
namespace actq_emulator {
using int5=std::array<int,5>;
int5 offset{},extent{};
int5 get_index_space_offset(){return offset;}
int5 get_index_space_size(){return extent;}
int5 operator+(int5 a,const int5& b){for(size_t i=0;i<5;++i) a[i]+=b[i];return a;}
struct tensor {void* data;std::vector<uint64_t> dims;uint32_t element;};
void* gen_addr(const int5& p,const tensor& t) {
    uint64_t index=0,stride=1;
    for(size_t i=0;i<5;++i) {
        const uint64_t size=i<t.dims.size()?t.dims[i]:1;
        if(p[i]<0 || uint64_t(p[i])>=size) throw std::out_of_range("activation TPC scalar access outside tensor");
        index+=uint64_t(p[i])*stride;stride*=size;
    }
    return static_cast<uint8_t*>(t.data)+index*t.element;
}
float s_f32_ld_g(float* p){return *p;}
void s_f32_st_g(float* p,float value){*p=value;}
void s_u8_st_g(unsigned char* p,unsigned char value){*p=value;}
#define __global__
#define main activation_quant_kernel
#include "../tpc/activation_quant.c"
#undef main
#undef __global__
}
using namespace strata::hpu;
namespace {
constexpr uint32_t values=32;
uint32_t cases=0,ties=0,fields=0,converter_checks=0,dot_checks=0;
void require(bool ok,const char* why){if(!ok) throw std::runtime_error(why);}
template<class F> void rejects(F f){bool refused=false;try{f();}catch(const std::exception&){refused=true;}
    require(refused,"input the host gate must refuse reached the quantizer");}
float widen(uint16_t bits){return strata::kernels::f32_from_f16(bits);}
uint16_t narrow(float value){return strata::kernels::f16_from_f32(value);}
float q8_1_finite(float value){return std::abs(value)>65504.0f?std::copysign(65504.0f,value):value;}
struct Law { std::vector<uint8_t> codes; std::vector<float> metadata; };
bool same(const Law& a,const Law& b){return a.codes==b.codes && a.metadata==b.metadata;}
// Authority 1, written from the CUDA text; the conversion is the pinned one, not the kernel's.
Law cuda_law(std::span<const float> input) {
    if(input.empty() || input.size()%values) throw std::invalid_argument("law input is not whole blocks");
    Law out;out.codes.resize(input.size());out.metadata.resize(input.size()/16);
    for(size_t base=0;base<input.size();base+=values) {
        float amax=0;std::array<float,values> lanes;
        for(uint32_t lane=0;lane<values;++lane) {lanes[lane]=input[base+lane];amax=std::max(amax,std::abs(lanes[lane]));}
        for(uint32_t offset=values/2;offset;offset>>=1) {
            const auto before=lanes;
            for(uint32_t lane=0;lane<values;++lane) lanes[lane]=before[lane]+before[lane^offset];   // warp_sum
        }
        const float scale=amax/127.0f;
        out.metadata[base/16]=widen(narrow(q8_1_finite(scale)));
        out.metadata[base/16+1]=widen(narrow(q8_1_finite(lanes[0])));
        for(uint32_t lane=0;lane<values;++lane) {
            const float q=amax==0.0f?0.0f:std::round(input[base+lane]/scale);   // divides by the fp32 scale
            out.codes[base+lane]=static_cast<uint8_t>(static_cast<int8_t>(q>127.0f?127.0f:q<-127.0f?-127.0f:q));
        }
    }
    return out;
}
// Authority 3's runner: one work item per (block, token), as the graph's index space would issue.
Law device_law(std::span<const float> input,uint32_t columns,uint32_t batch) {
    Law out;out.codes.assign(uint64_t(columns)*batch,uint8_t(0));out.metadata.assign(uint64_t(batch)*columns/16,0.0f);
    std::vector<float> source(input.begin(),input.end());
    using namespace actq_emulator;
    tensor in{source.data(),{columns,batch},4},codes{out.codes.data(),{columns,batch},1},
        metadata{out.metadata.data(),{2,columns/values,batch},4};
    for(uint32_t block=0;block<columns/values;++block) for(uint32_t token=0;token<batch;++token) {
        offset={int(block),int(token),0,0,0};extent={1,1,1,1,1};
        activation_quant_kernel(in,codes,metadata,int(columns),int(batch));
    }
    return out;
}
std::vector<float> sweep(uint32_t columns,uint32_t batch,uint32_t seed) {
    std::vector<float> input(uint64_t(columns)*batch);
    for(size_t i=0;i<input.size();++i) {
        const uint32_t step=uint32_t(i*37+seed*11)%997;
        input[i]=std::ldexp(float(int(step%21)-10),int(step%7)-3)*((i%5)==0?-1.0f:1.0f);
    }
    return input;
}
void pin_converter() {
    for(uint32_t bits=0;bits<65536;++bits) {
        const uint16_t pattern=uint16_t(bits);
        // Compared as bits: the fp16 NaN patterns widen to f32 NaNs, and a NaN never compares equal,
        // so a float comparison here would silently skip the payload the dot might read.
        require(std::bit_cast<uint32_t>(actq_emulator::actq_f32(pattern))==std::bit_cast<uint32_t>(widen(pattern)),
            "the kernel's fp16->fp32 widening differs from the pinned converter");
        // Every fp16 value is exact in fp32, so narrowing it must return its own bits - except the
        // inf/NaN patterns, which the pinned converter maps onto one canonical form by design.
        if(((pattern>>10)&0x1fu)!=0x1fu) require(actq_emulator::actq_f16_bits(widen(pattern))==pattern,
            "the kernel's converter does not round-trip an fp16 value");
        else require(actq_emulator::actq_f16_bits(widen(pattern))==narrow(widen(pattern)),
            "the kernel's converter maps an fp16 inf/NaN differently from the pinned one");
        ++converter_checks;
    }
    const std::vector<float> edges{0.0f,-0.0f,0.5f,-0.5f,1.0f,-1.0f,1.5f,-1.5f,2.5f,-3.5f,65504.0f,-65504.0f,
        65505.0f,65520.0f,65536.0f,1e30f,-1e30f,std::numeric_limits<float>::max(),std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN(),
        std::ldexp(1.0f,-14),std::ldexp(1.0f,-15),std::ldexp(1.0f,-24),std::ldexp(1.0f,-25),std::ldexp(1.5f,-24),
        std::ldexp(1.0f,-26),std::ldexp(3.0f,-25),std::nextafter(65504.0f,1e9f)};
    for(float value:edges) {
        require(actq_emulator::actq_f16_bits(value)==narrow(value),"the kernel's fp16 rounding differs at an edge value");
        require(std::bit_cast<uint32_t>(actq_emulator::actq_f32(actq_emulator::actq_f16_bits(value)))
            ==std::bit_cast<uint32_t>(widen(narrow(value))),"the kernel's fp16 round trip differs at an edge value");
        require(std::bit_cast<uint32_t>(actq_emulator::actq_metadata(value))
            ==std::bit_cast<uint32_t>(widen(narrow(q8_1_finite(value)))),
            "the kernel's stored metadata field differs from the pinned clamp plus conversion");
        ++converter_checks;
    }
}
void run_shape(uint32_t columns,uint32_t batch,uint32_t seed) {
    const auto input=sweep(columns,batch,seed);
    const auto host=quantize_q8_activation(std::span<const float>(input),columns);
    const Law reference{host.codes,host.metadata};
    require(same(cuda_law(std::span<const float>(input)),reference),"the host quantizer differs from the transcribed CUDA law");
    const auto device=device_law(std::span<const float>(input),columns,batch);
    require(same(device,reference),"the device quantizer differs from the bytes the host path produces");
    for(uint8_t byte:device.codes) require(byte!=0x80u,"a quantized code became -128, outside the range the CUDA clamps to");
    for(size_t i=0;i<device.metadata.size();++i) require(std::isfinite(device.metadata[i]),"a stored activation field is not finite");
    ++cases;
}
// A tie must actually execute, otherwise away-from-zero is only asserted. With amax = 127 * 2^e the
// fp32 scale is exactly 2^e, so (k + 0.5) * 2^e divides exactly in float and lands on the boundary.
void tie_rounding() {
    constexpr uint32_t columns=64,exponent=3;
    std::vector<float> input(columns,0.0f);
    for(uint32_t block=0;block<2;++block) {
        const uint32_t at=block*32;
        input[at]=std::ldexp(127.0f,exponent);
        for(uint32_t k=0;k<16;++k) {
            const float half=std::ldexp(float(2*int(k)+1),exponent)*0.5f;   // (k + 0.5) * 2^e
            input[at+1+k]=((k%2)==0)?half:-half;
            // the quotient must be exactly k + 0.5 in float, i.e. a real boundary case
            const float quotient=input[at+1+k]/(std::ldexp(127.0f,exponent)/127.0f);
            ties+=std::abs(quotient-std::floor(quotient))==0.5f?1u:0u;
        }
        for(uint32_t k=17;k<32;++k) input[at+k]=std::ldexp(float(int(k)-16),exponent);   // exact integers
    }
    const auto host=quantize_q8_activation(std::span<const float>(input),columns);
    const auto device=device_law(std::span<const float>(input),columns,1);
    require(same(device,Law{host.codes,host.metadata}),"the tie case differs from the host path");
    require(same(cuda_law(std::span<const float>(input)),Law{host.codes,host.metadata}),"the tie case differs from the CUDA law");
    for(uint32_t block=0;block<2;++block) {
        const uint32_t at=block*32;
        require(int8_t(host.codes[at])==127,"the block maximum must quantize to the range edge");
        for(uint32_t k=0;k<16;++k) {
            const int expected=int(k)+1;    // half away from zero, both signs
            const int actual=int(int8_t(host.codes[at+1+k]));
            require(actual==((k%2)==0?expected:-expected),"a half-way value did not round away from zero");
        }
        for(uint32_t k=17;k<32;++k) require(int8_t(host.codes[at+k])==int8_t(int(k)-16),"an exact integer quotient must quantize to itself");
    }
    require(ties==32,"the tie construction must produce exactly one boundary value per tie lane");
    ++cases;
}
void zero_block() {
    constexpr uint32_t columns=96;
    std::vector<float> input(columns,0.0f);
    const auto host=quantize_q8_activation(std::span<const float>(input),columns);
    const auto device=device_law(std::span<const float>(input),columns,1);
    require(same(device,Law{host.codes,host.metadata}),"an all-zero block differs from the host path");
    for(uint8_t byte:host.codes) require(byte==0,"an all-zero block produced a nonzero code");
    for(float value:host.metadata) require(value==0.0f,"an all-zero block produced a nonzero metadata field");
    ++cases;
}
void sum_clamp() {
    // 32 lanes of 5000 sum to 160000, past the largest finite half: q8_1_finite pulls it back so the
    // dot never multiplies an inf by zero, and the codes stay in range because they divide by amax/127.
    constexpr uint32_t columns=32;
    std::vector<float> input(columns,5000.0f);
    const auto host=quantize_q8_activation(std::span<const float>(input),columns);
    const auto device=device_law(std::span<const float>(input),columns,1);
    require(same(device,Law{host.codes,host.metadata}),"a clamped block sum differs from the host path");
    require(host.metadata[1]==65504.0f,"the overflowed sum was not clamped to the largest finite half");
    for(uint8_t byte:host.codes) require(int8_t(byte)==127,"a constant block must quantize every lane to the range edge");
    ++cases;
}
void unrounded_scale() {
    // amax small enough that the fp16 metadata flushes to zero while the fp32 scale does not: the
    // codes must still divide by the fp32 scale, so they are nonzero while the stored field is zero.
    constexpr uint32_t columns=32;
    std::vector<float> input(columns,0.0f);
    input[0]=std::ldexp(1.0f,-20);                                  // scale = 2^-20/127, below half of fp16's minimum
    for(uint32_t lane=1;lane<columns;++lane) input[lane]=std::ldexp(float(int(lane)%9)-4,-24);
    const auto host=quantize_q8_activation(std::span<const float>(input),columns);
    const auto device=device_law(std::span<const float>(input),columns,1);
    require(same(device,Law{host.codes,host.metadata}),"a flushing fp16 scale differs from the host path");
    require(host.metadata[0]==0.0f,"the stored scale was expected to flush to zero");
    require(host.codes[0]==uint8_t(127),"the code did not use the unrounded fp32 scale");
    bool any=false;for(uint32_t lane=1;lane<columns;++lane) any=any||host.codes[lane]!=0;
    require(any,"the block produced no codes at all, so the case proved nothing");
    ++cases;
}
void overflowing_quotient() {
    // The scale is clamped to the largest finite half while the value is not, so the quotient leaves the
    // int8 range by orders of magnitude (float max / 65504 is about 5e33). Clamping the quotient BEFORE
    // converting it is what keeps the kernel away from an out-of-range int conversion, and it must land
    // on the same 127 the host's clamp-then-cast produces.
    constexpr uint32_t columns=32;
    std::vector<float> input(columns,1.0f);
    input[0]=std::numeric_limits<float>::max();
    const auto host=quantize_q8_activation(std::span<const float>(input),columns);
    const auto device=device_law(std::span<const float>(input),columns,1);
    require(same(device,Law{host.codes,host.metadata}),"an overflowing quotient differs from the host path");
    require(host.codes[0]==uint8_t(127),"the clamped maximum did not quantize to the range edge");
    for(uint32_t lane=1;lane<columns;++lane) require(host.codes[lane]==0,"a value far below the clamped scale must quantize to zero");
    require(host.metadata[0]==65504.0f,"the stored scale was expected to sit at the largest finite half");
    ++cases;
}
void refused_inputs() {
    rejects([]{ std::vector<float> input(32,std::numeric_limits<float>::infinity());
        quantize_q8_activation(std::span<const float>(input),32); });
    rejects([]{ std::vector<float> input(32,1.0f); input[0]=std::numeric_limits<float>::quiet_NaN();
        quantize_q8_activation(std::span<const float>(input),32); });
    rejects([]{ std::vector<float> input(32,0.0f); input[0]=std::ldexp(1.0f,-149);   // amax nonzero, scale flushes
        quantize_q8_activation(std::span<const float>(input),32); });
    rejects([]{ std::vector<float> input(31,1.0f); quantize_q8_activation(std::span<const float>(input),32); });
    ++cases;
}
// The consumer contract, with a fixture the weight validator itself accepts.
void feeds_the_dot() {
    constexpr uint32_t columns=256,rows=4,batch=3;
    std::vector<float> input(uint64_t(columns)*batch);
    for(size_t i=0;i<input.size();++i) input[i]=float(int(i%199)-99)*0.25f;
    const auto host=quantize_q8_activation(std::span<const float>(input),columns);
    const auto device=device_law(std::span<const float>(input),columns,batch);
    const uint64_t row_bytes=packed_row_bytes(8,columns);
    std::vector<uint8_t> weights(uint64_t(row_bytes)*rows,0);
    const uint16_t bits=narrow(0.00390625f);
    for(uint32_t row=0;row<rows;++row) for(uint32_t block=0;block<columns/values;++block) {
        const uint64_t at=(uint64_t(row)*row_bytes)+uint64_t(block)*34;
        weights[at]=uint8_t(bits&0xffu);weights[at+1]=uint8_t(bits>>8);
        for(uint32_t lane=0;lane<values;++lane) weights[at+2+lane]=uint8_t((row*31+block*17+lane*7)%251);
    }
    validate_packed_rows(std::span<const uint8_t>(weights),8,columns,rows);
    auto dot=[&](const Law& activation) {
        std::vector<float> result(uint64_t(rows)*batch,NAN);
        auto bytes=weights;auto codes=activation.codes;auto metadata=activation.metadata;
        using namespace packed_emulator;
        tensor w{bytes.data(),{row_bytes,rows},1},a{codes.data(),{columns,batch},1},
            m{metadata.data(),{2,columns/values,batch},4},o{result.data(),{rows,batch},4};
        for(uint32_t token=0;token<batch;++token) for(uint32_t row=0;row<rows;++row) {
            offset={int(row),int(token),0,0,0};extent={1,1,1,1,1};
            packed_kernel(w,a,m,o,int(columns),int(rows),int(batch),8,0);
        }
        return result;
    };
    const auto from_host=dot(Law{host.codes,host.metadata});
    const auto from_device=dot(device);
    require(from_host.size()==from_device.size(),"the dot changed shape between the two activation sources");
    for(size_t i=0;i<from_host.size();++i) {
        require(std::isfinite(from_host[i])&&std::isfinite(from_device[i]),"the packed dot became non-finite");
        require(std::bit_cast<uint32_t>(from_host[i])==std::bit_cast<uint32_t>(from_device[i]),
            "the packed dot reads a different result from the kernel's buffers than from the host's");
        ++dot_checks;
    }
    auto poisoned=device.codes;poisoned[3]^=0x10;
    require(dot(Law{poisoned,device.metadata})!=from_host,"the packed dot ignored the activation bytes");
    // Field 0 is the block scale, which every dot path reads; field 1 is the raw sum, which the
    // affine correction paths read and the per-format dot tests pin. Perturbing the scale must move
    // the result, which proves the kernel wrote it at the offset the consumer expects.
    auto shifted=device.metadata;shifted[0]+=1.0f;
    require(dot(Law{device.codes,shifted})!=from_host,"the packed dot ignored the activation scale field");
    ++cases;
}
void per_field_checks() {
    for(uint32_t columns:{uint32_t(32),uint32_t(96),uint32_t(256),uint32_t(2560)})
        for(uint32_t batch:{1u,2u,5u,16u}) {
            const auto input=sweep(columns,batch,columns*7+batch);
            const auto host=quantize_q8_activation(std::span<const float>(input),columns);
            const auto device=device_law(std::span<const float>(input),columns,batch);
            require(host.metadata.size()==device.metadata.size(),"metadata length differs");
            for(size_t i=0;i<host.metadata.size();++i) {
                require(std::bit_cast<uint32_t>(device.metadata[i])==std::bit_cast<uint32_t>(host.metadata[i]),
                    "a metadata field differs bit for bit");
                ++fields;
            }
            run_shape(columns,batch,columns+batch);
        }
}
}
int main(){try {
    // Exercise the selector used by ModelHead, not just either quantizer in
    // isolation. RoutedQ8 must consume the same bytes/half metadata as Q8_1.
    {
        std::array<float,32> input{};
        for(size_t i=0;i<input.size();++i) input[i]=float(int(i)-13)*.13f+.0123f;
        const auto q8=quantize_q8_activation(input,32),s2=quantize_s2_activation(input,32);
        require(q8.metadata!=s2.metadata,"quantizer-selection fixture does not distinguish S2 and Q8_1");
        for(auto mode:{PackedActivation::Q8_1,PackedActivation::RoutedQ8}) {
            const auto selected=quantize_packed_activation(input,32,mode);
            require(selected.codes==q8.codes && selected.metadata==q8.metadata,"runtime routed/shared Q8 quantizer selection changed contract");
        }
        const auto selected=quantize_packed_activation(input,32,PackedActivation::RoutedS2);
        require(selected.codes==s2.codes && selected.metadata==s2.metadata,"runtime S2 quantizer selection changed contract");
        bool rejected=false;try{(void)quantize_packed_activation(input,32,static_cast<PackedActivation>(3));}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"unknown runtime quantizer mode was accepted");
    }
    pin_converter();
    per_field_checks();
    tie_rounding();
    zero_block();
    sum_clamp();
    unrounded_scale();
    overflowing_quotient();
    refused_inputs();
    feeds_the_dot();
    std::cout<<"PASS device activation quantizer "<<cases<<" cases: the kernel's codes and metadata are identical to the "
        "host path AND to a transcription of native_mmvq's quantize_q8_1 law, "+std::to_string(converter_checks)+
        " converter pins over all 65536 fp16 patterns plus the overflow/subnormal/tie edges, "+std::to_string(fields)+
        " metadata fields compared bit for bit, "+std::to_string(ties)+" exact half-way ties held away from zero in both "
        "signs, all-zero / clamped-sum / flushing-fp16-scale blocks, the inputs the host gate refuses before upload, and "+
        std::to_string(dot_checks)+" packed-dot results that read the kernel's own buffers bit for bit; no device\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
