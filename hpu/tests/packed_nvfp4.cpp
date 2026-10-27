// Native NVFP4 (GGML type 40) 64-value block dot for the shared/head and routed q8_1
// contracts. NVFP4 is the format Strata must pack that pinned upstream does NOT route through
// native_mmvq: a block is four UE4M3 sub-block scales plus 32 packed E2M1 nibble pairs, the
// value is kvalues_fp4[nibble] * ggml_ue4m3_to_fp32(byte), and the per-expert global F32 scale
// stays outside the block. The pinned helper (ggml/src/ggml-impl.h:514) folds a final half
// into every factor precisely because kvalues_fp4 stores DOUBLED e2m1 magnitudes, so the dot
// can stay integer; both authorities below are therefore transcriptions of pinned code, and the
// kernel is compared with the independent analytical oracle AND with the unmodified pinned
// ggml NVFP4 row decoder (ggml/src/ggml-quants.c:589). All 16 nibbles x 256 scale encodings are
// pinned through the decoder, and rows always span several 64-value blocks because a
// per-block indexing bug is invisible in a single-block row.
// Host only: no ISA execution, no device, no performance claim.
#include "strata/hpu/packed_quant.hpp"
#include "strata/hpu/expert_format.hpp"
#include "strata/hpu/packed_expert_flow.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "packed_emulator.hpp"
#include <ggml.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <span>
#include <vector>
using namespace strata::hpu;
namespace {
void require(bool ok,const char* why){if(!ok) throw std::runtime_error(why);}
template<class F> void rejects(F f){bool bad=false;try{f();}catch(const std::exception&){bad=true;}require(bad,"invalid NVFP4 payload or policy accepted");}
int signed_code(uint8_t b){return b<128?b:int(b)-256;}
constexpr uint32_t per_block=64,sub_blocks=4,sub_values=16,bytes_per_block=36,q8_bytes=34;
constexpr uint32_t scales_at=0,qs_at=4;
// Transcribed from the pinned table kvalues_fp4 (ggml-common.h:1126, "e2m1 values (doubled)"),
// independently of the generated include the kernel uses.
constexpr std::array<int,16> doubled_e2m1{{0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12}};
struct Oracle { long double value,magnitude; };
// ggml-impl.h:514: unsigned 4-bit exponent with bias 7, 3-bit mantissa, and a closing half.
long double ue4m3(uint8_t byte) {
    if(byte==0x00 || byte==0x7f) return 0.0L;
    const int exp=(byte>>3)&15,man=byte&7;
    if(exp==0) return std::ldexp(static_cast<long double>(man),-9)*0.5L;
    return std::ldexp(1.0L+static_cast<long double>(man)/8.0L,exp-7)*0.5L;
}
const ggml_type_traits* traits() {
    const auto* t=ggml_get_type_traits(GGML_TYPE_NVFP4);
    require(t && t->blck_size==int(per_block) && t->type_size==bytes_per_block && t->to_float,"pinned ggml lacks the NVFP4 row decoder or its block size differs");
    return t;
}
uint32_t blocks(uint32_t columns){return columns/per_block;}
std::vector<float> decode_row(const std::vector<uint8_t>& bytes,uint32_t columns,uint32_t row) {
    std::vector<float> weights(columns);
    traits()->to_float(bytes.data()+uint64_t(row)*packed_row_bytes(40,columns),weights.data(),int64_t(columns));
    return weights;
}
struct Row {
    uint32_t columns;
    std::vector<uint8_t> nibble;                  // [blocks][64] one E2M1 nibble per value
    std::vector<uint8_t> scale;                   // [blocks][4] UE4M3 factor per 16-value sub-block
};
Row x_logical(uint32_t columns,uint32_t row) {
    const auto count=blocks(columns);
    Row w{columns,std::vector<uint8_t>(uint64_t(count)*per_block),std::vector<uint8_t>(uint64_t(count)*sub_blocks)};
    for(uint32_t b=0;b<count;++b) {
        for(uint32_t v=0;v<per_block;++v) w.nibble[uint64_t(b)*per_block+v]=uint8_t((v*7+b*5+row*3)%16);
        for(uint32_t s=0;s<sub_blocks;++s) w.scale[uint64_t(b)*sub_blocks+s]=uint8_t((s*41+b*17+row*11)%255u+1u); // never 0/0x7f: real blocks keep a usable factor
    }
    return w;
}
std::vector<uint8_t> x_encode(const std::vector<Row>& rows) {
    const auto columns=rows.front().columns;
    std::vector<uint8_t> bytes(uint64_t(rows.size())*blocks(columns)*bytes_per_block,0);
    for(size_t row=0;row<rows.size();++row) {const auto& w=rows[row];
        for(uint32_t b=0;b<blocks(columns);++b) {
            const size_t at=(uint64_t(row)*blocks(columns)+b)*bytes_per_block;
            for(uint32_t s=0;s<sub_blocks;++s) {
                bytes[at+scales_at+s]=w.scale[uint64_t(b)*sub_blocks+s];
                for(uint32_t j=0;j<8;++j)
                    bytes[at+qs_at+s*8+j]=uint8_t(w.nibble[uint64_t(b)*per_block+s*16+j]
                        | (w.nibble[uint64_t(b)*per_block+s*16+8+j]<<4));
            }
        }
    }
    return bytes;
}
uint32_t nibble_pins=0,scale_checks=0;
void pin_alphabet() {
    // For every UE4M3 encoding, put all sixteen nibbles in sub-block 0 in order and require the
    // decoder to return exactly doubled_e2m1[n] * ue4m3(byte). This pins the whole 16x256
    // composite grid, so neither the value table nor the scale helper can be misread, and the
    // same numbers must be what the kernel's generated include produces.
    for(uint32_t byte=0;byte<256;++byte) {
        std::vector<uint8_t> block(bytes_per_block,0);
        block[scales_at]=uint8_t(byte);
        for(uint32_t j=0;j<8;++j) block[qs_at+j]=uint8_t(j | ((j+8)<<4));
        std::array<float,per_block> decoded{};
        traits()->to_float(block.data(),decoded.data(),int(per_block));
        const long double factor=ue4m3(uint8_t(byte));
        for(uint32_t n=0;n<16;++n) {
            const uint32_t value=n;   // low nibble of byte n, or high nibble of byte n-8
            const long double expected=static_cast<long double>(doubled_e2m1[n])*factor;
            require(std::abs(static_cast<long double>(decoded[value])-expected)<=std::max(1e-9L,std::abs(expected)*1e-6L),
                "the pinned NVFP4 decoder disagrees with the transcribed nibble/scale law");
            ++nibble_pins;
        }
    }
    // The kernel's own generated table must be the same transcription, entry by entry.
    for(uint32_t n=0;n<16;++n) require(packed_emulator::packed_kvalues_fp4[n]==doubled_e2m1[n],
        "the FP4 table compiled into the TPC kernel differs from the pinned transcription");
    // ... and so must the kernel's scale conversion be, for every encoding including the
    // subnormal branch where the pinned helper's closing half changes the exponent.
    for(uint32_t byte=0;byte<256;++byte) {
        const long double expected=ue4m3(uint8_t(byte)),actual=static_cast<long double>(packed_emulator::packed_ue4m3(uint8_t(byte)));
        require(std::abs(actual-expected)<=std::abs(expected)*1e-7L+1e-12L,
            "the UE4M3 factor compiled into the TPC kernel differs from the pinned helper");
        ++scale_checks;
    }
}
void scale_rule() {
    // 0x00 and 0x7F both zero the sub-block, the top bit is ignored, and the exponent doubles
    // the factor: measured against the decoder, not assumed.
    auto build=[&](uint8_t byte) {
        std::vector<uint8_t> block(bytes_per_block,0);
        block[scales_at]=byte;
        for(uint32_t j=0;j<8;++j) block[qs_at+j]=uint8_t(0x02 | (0x02<<4));
        std::array<float,per_block> out{};
        traits()->to_float(block.data(),out.data(),int(per_block));
        return out;
    };
    const auto zero=build(0x00),reserved=build(0x7f),half=build(0x38),whole=build(0x40),mirrored=build(0xb8);
    for(uint32_t v=0;v<per_block;++v) {
        require(zero[v]==0.0f && reserved[v]==0.0f,"an NVFP4 zero/reserved scale did not zero its sub-block");
        ++scale_checks;
    }
    for(uint32_t v=0;v<sub_values;++v) {
        require(std::abs(half[v]*2.0f-whole[v])<1e-6f,"the NVFP4 exponent step did not double the factor");
        require(half[v]==mirrored[v],"the NVFP4 scale ignored something other than its top bit");
        ++scale_checks;
    }
    for(uint32_t v=sub_values;v<per_block;++v) require(half[v]==0.0f && whole[v]==0.0f,"an NVFP4 sub-block scale leaked past its 16 values");
    // Every mantissa step is 1/8 of the exponent band: 0x38,0x39,0x3a..0x3f ratio 1+man/8.
    for(uint32_t man=0;man<8;++man) {
        const auto stepped=build(uint8_t(0x38u|man));
        const long double expected=static_cast<long double>(half[0])*(1.0L+static_cast<long double>(man)/8.0L);
        require(std::abs(static_cast<long double>(stepped[0])-expected)<=std::abs(expected)*1e-6L+1e-9L,
            "the NVFP4 mantissa did not step by eighths of the exponent band");
        ++scale_checks;
    }
}
// Independent transcription of the pinned law: value = doubled_e2m1[nibble] * ue4m3(scale),
// summed against the quantized activation codes with the sub-block's activation d8.
Oracle x_oracle(const Row& w,const Q8Activation& q,uint32_t token) {
    Oracle out{0,0};
    for(uint32_t b=0;b<blocks(w.columns);++b) for(uint32_t s=0;s<sub_blocks;++s) {
        const uint64_t base=uint64_t(b)*per_block+uint64_t(s)*sub_values;   // this sub-block's first value
        long double dot=0;
        for(uint32_t v=0;v<sub_values;++v)
            dot+=static_cast<long double>(doubled_e2m1[w.nibble[base+v]])
                *signed_code(q.codes[uint64_t(token)*w.columns+base+v]);
        const long double term=dot*ue4m3(w.scale[uint64_t(b)*sub_blocks+s])
            *q.metadata[(uint64_t(token)*w.columns/32+b*2+s/2)*2];
        out.value+=term;out.magnitude+=std::abs(term);
    }
    return out;
}
Oracle x_decoded(const std::vector<uint8_t>& bytes,uint32_t columns,uint32_t row,const Q8Activation& q,uint32_t token) {
    const auto weights=decode_row(bytes,columns,row);
    Oracle out{0,0};
    for(uint32_t block=0;block<columns/32;++block) {
        const long double d8=q.metadata[(uint64_t(token)*columns/32+block)*2];long double part=0;
        for(uint32_t v=0;v<32;++v) part+=static_cast<long double>(weights[block*32+v])
            *signed_code(q.codes[uint64_t(token)*columns+block*32+v]);
        const long double term=d8*part;out.value+=term;out.magnitude+=std::abs(term);
    }
    return out;
}
std::vector<float> tpc(const std::vector<uint8_t>& bytes,uint32_t type,uint32_t columns,uint32_t rows,const Q8Activation& q,uint32_t batch,PackedActivation mode) {
    std::vector<float> result(uint64_t(rows)*batch,NAN);
    using namespace packed_emulator;
    auto& w_bytes=const_cast<std::vector<uint8_t>&>(bytes);auto& codes=const_cast<std::vector<uint8_t>&>(q.codes);auto& meta=const_cast<std::vector<float>&>(q.metadata);
    tensor w{w_bytes.data(),{packed_row_bytes(type,columns),rows},1},a{codes.data(),{columns,batch},1},
        m{meta.data(),{2,columns/32,batch},4},o{result.data(),{rows,batch},4};
    for(uint32_t token=0;token<batch;++token) for(uint32_t row=0;row<rows;++row) {
        offset={int(row),int(token),0,0,0};extent={1,1,1,1,1};
        packed_kernel(w,a,m,o,int(columns),int(rows),int(batch),int(type),int(mode));
    }
    return result;
}
bool close(float actual,const Oracle& expected) {
    if(!std::isfinite(actual)) return false;
    return std::abs(static_cast<long double>(actual)-expected.value)
        <=std::max<long double>(2e-3L,expected.magnitude*2e-5L);
}
std::vector<float> source(uint64_t values,uint32_t seed) {
    std::vector<float> out(values);for(uint64_t i=0;i<values;++i) out[i]=float(int((i*23+seed)%127)-63)*.03125f+.0140625f;return out;
}
uint32_t cases_count=0;
void cases(uint32_t columns,uint32_t batch) {
    constexpr uint32_t rows=3;std::vector<Row> logical_rows;
    for(uint32_t row=0;row<rows;++row) logical_rows.push_back(x_logical(columns,row));
    const auto bytes=x_encode(logical_rows);
    validate_packed_rows(bytes,40,columns,rows);
    const auto values=source(uint64_t(batch)*columns,batch);
    const auto q=quantize_q8_activation(values,columns);
    const auto actual=tpc(bytes,40,columns,rows,q,batch,PackedActivation::Q8_1);
    const auto routed=tpc(bytes,40,columns,rows,q,batch,PackedActivation::RoutedQ8);
    for(uint32_t token=0;token<batch;++token) for(uint32_t row=0;row<rows;++row) {
        const auto index=uint64_t(token)*rows+row;
        require(close(actual[index],x_oracle(logical_rows[row],q,token)),"NVFP4 TPC differs from the independent pinned-law oracle");
        require(close(actual[index],x_decoded(bytes,columns,row,q,token)),"NVFP4 TPC differs from the unmodified pinned ggml NVFP4 decoder");
        require(routed[index]==actual[index],"NVFP4 dot differs between the routed and shared/head q8_1 contracts");
        std::vector<float> one(values.begin()+uint64_t(token)*columns,values.begin()+uint64_t(token)*columns+columns);
        const auto isolated=tpc(bytes,40,columns,rows,quantize_q8_activation(one,columns),1,PackedActivation::Q8_1);
        require(isolated[row]==actual[index],"batched NVFP4 changed single-row arithmetic or crossed tokens");
        ++cases_count;
    }
}
void counterexamples() {
    constexpr uint32_t columns=512;                                     // eight 64-value blocks
    const auto w=x_logical(columns,0);const auto bytes=x_encode({w});
    const auto values=source(columns,7);const auto q=quantize_q8_activation(values,columns);
    const auto before=tpc(bytes,40,columns,1,q,1,PackedActivation::Q8_1)[0];
    require(close(before,x_oracle(w,q,0)) && close(before,x_decoded(bytes,columns,0,q,0)),"NVFP4 baseline disagrees with its authorities");
    auto raw=q;for(uint32_t block=0;block<columns/32;++block) const_cast<Q8Activation&>(raw).metadata[block*2+1]=1234.5f;
    require(tpc(bytes,40,columns,1,raw,1,PackedActivation::Q8_1)[0]==before,"NVFP4 output moved when only raw sums changed");
    uint32_t moved=0;
    for(uint32_t v=0;v<columns && moved<4;++v) {
        if(!signed_code(q.codes[v])) continue;
        ++moved;
        auto altered=w;const uint32_t block=v/per_block,inside=v%per_block;
        altered.nibble[block*per_block+inside]=uint8_t((altered.nibble[block*per_block+inside]+1)%16);
        const auto altered_bytes=x_encode({altered});validate_packed_rows(altered_bytes,40,columns,1);
        const auto result=tpc(altered_bytes,40,columns,1,q,1,PackedActivation::Q8_1)[0];
        require(close(result,x_oracle(altered,q,0)) && close(result,x_decoded(altered_bytes,columns,0,q,0)),
            "the changed NVFP4 nibble lost its oracle or decoder match");
    }
    require(moved==4,"too few observable NVFP4 lanes to pin a single nibble");
    // The nibble's sign bit is bit 3 of the value, not part of the magnitude: n and n+8 negate.
    auto negated=w;negated.nibble[0]^=8u;
    const auto negated_bytes=x_encode({negated});validate_packed_rows(negated_bytes,40,columns,1);
    require(close(tpc(negated_bytes,40,columns,1,q,1,PackedActivation::Q8_1)[0],x_oracle(negated,q,0)),
        "flipping the NVFP4 sign bit left the independent oracle");
    // One scale byte owns exactly its sub-block: change sub-block 2 and only its values move.
    auto scaled=w;scaled.scale[2]=uint8_t((scaled.scale[2]+0x08u)&0xffu);
    const auto scaled_bytes=x_encode({scaled});validate_packed_rows(scaled_bytes,40,columns,1);
    const auto scaled_result=tpc(scaled_bytes,40,columns,1,q,1,PackedActivation::Q8_1)[0];
    require(close(scaled_result,x_oracle(scaled,q,0)) && close(scaled_result,x_decoded(scaled_bytes,columns,0,q,0)),
        "the changed NVFP4 sub-block scale left an authority");
    const auto decoded_before=decode_row(bytes,columns,0),decoded_scaled=decode_row(scaled_bytes,columns,0);
    uint32_t outside=0;
    for(uint32_t v=0;v<columns;++v) {
        const bool third=(v%per_block)>=32 && (v%per_block)<48;
        if(decoded_before[v]!=decoded_scaled[v] && !third) ++outside;
    }
    require(outside==0,"an NVFP4 sub-block scale moved values outside its own sub-block");
    // A zeroed scale must zero its whole sub-block's contribution, and the reserved 0x7F must
    // behave identically: the block's other sub-blocks keep working.
    auto muted=w;muted.scale[0]=0x7fu;
    const auto muted_bytes=x_encode({muted});validate_packed_rows(muted_bytes,40,columns,1);
    auto reserved=w;reserved.scale[0]=0x00u;
    const auto reserved_bytes=x_encode({reserved});validate_packed_rows(reserved_bytes,40,columns,1);
    require(tpc(muted_bytes,40,columns,1,q,1,PackedActivation::Q8_1)[0]==tpc(reserved_bytes,40,columns,1,q,1,PackedActivation::Q8_1)[0],
        "NVFP4 treated the reserved 0x7F scale differently from zero");
    require(tpc(muted_bytes,40,columns,1,q,1,PackedActivation::Q8_1)[0]!=before,"zeroing an NVFP4 sub-block scale changed nothing");
    // Multi-block indexing: only the LAST block's scale changed, so every earlier block must be
    // byte-identical and its decoded values unchanged - the failure mode a single-block row cannot
    // see (this is exactly what caught the IQ1_M encoder bug).
    const uint32_t last_block=columns/per_block-1;
    auto tail=w;tail.scale[last_block*sub_blocks]=0x40u;
    const auto tail_bytes=x_encode({tail});validate_packed_rows(tail_bytes,40,columns,1);
    for(uint32_t b=0;b<last_block;++b) for(uint32_t i=0;i<bytes_per_block;++i)
        require(tail_bytes[size_t(b)*bytes_per_block+i]==bytes[size_t(b)*bytes_per_block+i],"an NVFP4 tail-block write leaked into an earlier block");
    require(tail_bytes[last_block*bytes_per_block]!=bytes[last_block*bytes_per_block],"the NVFP4 tail block did not change at all");
    require(close(tpc(tail_bytes,40,columns,1,q,1,PackedActivation::Q8_1)[0],x_oracle(tail,q,0)),
        "the NVFP4 tail block did not follow the independent oracle, so per-block indexing is wrong");
}
void policy_types() {
    for(uint32_t type:{2,3,6,7,8,11,12,13,14,16,17,18,20,21,22,23,29,40,42}) require(packed_supported(type),"a packed GGML type with a working dot was rejected");
    for(uint32_t type:{19,30,31}) require(!packed_supported(type),"a still-decoded GGML type was accepted as packed");
}
void guards() {
    policy_types();
    const auto w=x_logical(256,0);const auto bytes=x_encode({w});
    validate_packed_rows(bytes,40,256,1);
    auto wild=bytes;for(size_t i=0;i<wild.size();++i) wild[i]=0xffu;      // every UE4M3 byte is finite
    validate_packed_rows(wild,40,256,1);
    rejects([&]{(void)packed_row_bytes(40,300);});rejects([&]{(void)packed_row_bytes(40,0);});
    require(packed_row_bytes(40,640)==360 && packed_row_bytes(40,2560)==1440,"NVFP4 row charge wrong");
    require(packed_supported(40) && packed_values_per_block(40)==per_block && packed_block_bytes(40)==bytes_per_block,"NVFP4 geometry wrong");
    require(packed_layout_supported(40,640) && packed_layout_supported(40,2560),"NVFP4 refused a length it divides");
    require(!packed_layout_supported(40,66),"NVFP4 accepted a partial 64-value block");
    require(packed_activation_supported(40,PackedActivation::Q8_1) && packed_activation_supported(40,PackedActivation::RoutedQ8)
        && !packed_activation_supported(40,PackedActivation::RoutedS2),"NVFP4 policy wrong");
    const auto routed=expert_format(2560,640,0,{40,40,8});
    require(routed.packed && routed.activation==PackedActivation::RoutedQ8
        && routed.bytes==uint64_t(1440)*640*2+uint64_t(34)*20*2560,"routed NVFP4 expert charge wrong");
    const auto shared=expert_format(2560,640,-1,{40,29,8});
    require(shared.packed && shared.activation==PackedActivation::Q8_1
        && shared.bytes==uint64_t(1440)*640+uint64_t(560)*640+uint64_t(34)*20*2560,"shared NVFP4 expert was not packed or is mischarged");
    const auto tail=expert_format(2560,640,0,{8,8,40});
    require(tail.packed && tail.bytes==uint64_t(34)*20*2560*2+uint64_t(360)*2560,"NVFP4 down role was not packed at a 640-column projection");
}
struct Flow {
    uint32_t type,columns,rows;
    std::vector<uint8_t> bytes;
    std::vector<Row> xs;
    std::vector<int> q8_codes;
    std::vector<float> q8_d;
};
Flow flow_role(uint32_t type,uint32_t columns,uint32_t rows,uint32_t seed) {
    Flow r{type,columns,rows,{},{},{},{}};
    if(type==8) {
        r.q8_codes.resize(uint64_t(rows)*columns);r.q8_d.resize(uint64_t(rows)*columns/32);
        r.bytes.assign(uint64_t(rows)*(columns/32)*q8_bytes,0);
        for(uint32_t row=0;row<rows;++row) for(uint32_t b=0;b<columns/32;++b) {
            const size_t at=(uint64_t(row)*columns/32+b)*q8_bytes;
            const float d=float(int((b*3+row*5+seed)%9)-4)/16.f;
            const auto bits=strata::kernels::f16_from_f32(d);
            r.bytes[at]=uint8_t(bits);r.bytes[at+1]=uint8_t(bits>>8);r.q8_d[uint64_t(row)*columns/32+b]=d;
            for(uint32_t j=0;j<32;++j) {
                const int code=int((b*32+j)*13+row*11+seed*7+3)%249-124;
                r.q8_codes[uint64_t(row)*columns+b*32+j]=code;r.bytes[at+2+j]=uint8_t(code);
            }
        }
        validate_packed_rows(r.bytes,8,columns,rows);return r;
    }
    for(uint32_t row=0;row<rows;++row) r.xs.push_back(x_logical(columns,row+seed*1000));
    r.bytes=x_encode(r.xs);validate_packed_rows(r.bytes,40,columns,rows);return r;
}
std::vector<Oracle> flow_oracle(const Flow& r,const Q8Activation& q,uint32_t batch) {
    std::vector<Oracle> out(uint64_t(r.rows)*batch);
    for(uint32_t token=0;token<batch;++token) for(uint32_t row=0;row<r.rows;++row) {
        if(r.type==8) {
            long double total=0,magnitude=0;
            for(uint32_t b=0;b<r.columns/32;++b) {
                long double dot=0;
                for(uint32_t j=0;j<32;++j) dot+=static_cast<long double>(r.q8_codes[uint64_t(row)*r.columns+b*32+j])*signed_code(q.codes[uint64_t(token)*r.columns+b*32+j]);
                const long double term=r.q8_d[uint64_t(row)*r.columns/32+b]*q.metadata[(uint64_t(token)*r.columns/32+b)*2]*dot;
                total+=term;magnitude+=std::abs(term);
            }
            out[uint64_t(token)*r.rows+row]=Oracle{total,magnitude};continue;
        }
        auto o=x_oracle(r.xs[row],q,token);
        out[uint64_t(token)*r.rows+row]=o;
    }
    return out;
}
uint32_t flows=0;
void shared_flows() {
    constexpr uint32_t hidden=512,intermediate=256;
    // Rows here are NVFP4-shaped, so only NVFP4 and Q8_0 roles appear; mixed packed formats are
    // exercised by tests/packed_expert.cpp's real layout source. The per-expert global scale is
    // part of the checked contract: it applies to the down projection's F32 result AFTER its own
    // activation quantization and dot, so packing the weights must not change that.
    for(auto types:{std::array<uint32_t,3>{40,40,40},std::array<uint32_t,3>{40,40,8},std::array<uint32_t,3>{40,8,40},std::array<uint32_t,3>{8,40,40}})
    for(uint32_t batch:{1u,3u}) {
        const auto format=expert_format(hidden,intermediate,-1,types);
        require(format.packed && format.activation==PackedActivation::Q8_1,"shared NVFP4 expert lost the head contract");
        std::array<Flow,3> role{flow_role(types[0],hidden,intermediate,0),flow_role(types[1],hidden,intermediate,1),flow_role(types[2],intermediate,hidden,2)};
        const uint64_t charge=packed_row_bytes(types[0],hidden)*intermediate+packed_row_bytes(types[1],hidden)*intermediate
            +packed_row_bytes(types[2],intermediate)*hidden;
        require(format.bytes==charge,"shared NVFP4 expert payload mischarged");
        std::vector<float> input(uint64_t(batch)*hidden);
        for(size_t i=0;i<input.size();++i) input[i]=float(int((i*29+batch)%113)-56)*.0625f;
        std::array<uint32_t,3> calls{0,0,0};
        auto run=[&](const std::array<Flow,3>& roles,float scale)->std::vector<float> {
            calls={0,0,0};
            const std::array<float,3> scales{1.0f,1.0f,scale};
            return execute_packed_expert(std::span<const float>(input),hidden,intermediate,[&](uint32_t which,std::span<const float> values) {
                ++calls[which];
                const auto mode=expert_projection_activation(format,which);
                require(mode==PackedActivation::Q8_1,"shared NVFP4 projection inherited a routed activation contract");
                const auto tokens=uint32_t(values.size()/roles[which].columns);
                const auto q=quantize_q8_activation(values,roles[which].columns);
                // Callbacks return unweighted projections: execute_packed_expert applies the global
                // scale afterwards, which the linearity check below verifies separately.
                const auto expected=flow_oracle(roles[which],q,tokens);
                const auto actual=tpc(roles[which].bytes,roles[which].type,roles[which].columns,roles[which].rows,q,tokens,mode);
                for(size_t i=0;i<actual.size();++i) {
                    require(std::isfinite(actual[i]),"non-finite shared NVFP4 projection");
                    const auto tolerance=std::max({.02,std::abs(static_cast<double>(expected[i].value))*2e-3,
                        static_cast<double>(expected[i].magnitude)*8e-7});
                    require(std::abs(double(actual[i])-double(expected[i].value))<=tolerance,
                        "shared expert NVFP4 projection differs from the independent oracle");
                }
                return actual;
            },scales);
        };
        const auto output=run(role,0.125f);
        require(calls==std::array<uint32_t,3>{1,1,1},"shared NVFP4 expert flow did not run each projection once");
        require(output.size()==input.size(),"shared NVFP4 expert flow output shape");
        for(float value:output) require(std::isfinite(value),"non-finite shared NVFP4 expert output");
        // The global scale must be the last thing applied to the down result, and exactly linear.
        const auto unscaled=run(role,1.0f);
        require(unscaled.size()==output.size(),"the NVFP4 global-scale comparison changed the output shape");
        for(size_t i=0;i<output.size();++i) {
            require(std::abs(double(output[i])-double(unscaled[i]*0.125f))<=std::max(1e-5,std::abs(unscaled[i])*0.125f*1e-5),
                "the packed NVFP4 expert did not apply its global scale linearly to the down result");
            ++scale_checks;
        }
        if(types[0]==types[1] && types[0]==40) {
            auto swapped=role;std::swap(swapped[0],swapped[1]);
            const auto other=run(swapped,0.125f);bool differs=false;
            for(size_t i=0;i<output.size();++i) differs=differs||output[i]!=other[i];
            require(differs,"exchanging gate and up NVFP4 weights left the shared expert output unchanged");
        }
        ++flows;
    }
}
}
int main(){try {
    pin_alphabet();scale_rule();
    for(uint32_t columns:{64u,512u,2560u}) for(uint32_t batch:{1u,3u,8u}) cases(columns,batch);
    counterexamples();guards();shared_flows();
    std::cout<<"PASS NVFP4 TPC "<<cases_count<<" cases against the independent pinned-law oracle (doubled e2m1 nibbles times the UE4M3 factor with its closing half) AND the unmodified pinned ggml NVFP4 row decoder, "
        <<nibble_pins<<" nibble/scale composite pins over all 16 values and 256 encodings plus the 16 kernel table entries, "
        <<scale_checks<<" scale-rule checks (0x00 and 0x7F zero, top bit ignored, exponent doubles, mantissa steps by eighths), "
        <<scale_checks<<" global-scale linearity checks on the flow result, "
        "sub-block/sign/multi-block counterexamples, "<<flows<<" production shared-expert flows with the per-expert global scale in the checked contract, and policy guards; no device\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
