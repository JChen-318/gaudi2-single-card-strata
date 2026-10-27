// Native IQ1_M (GGML type 29) 256-value superblock dot for the shared/head and routed q8_1
// contracts - the last format pinned native_mmvq_supported admits. IQ1_M is structurally
// unlike every format handled here: the block carries NO d of its own, its fp16 super-block
// scale is assembled from the top nibbles of its own four scale words, an 11-bit index into
// the ternary grid {-1, 0, +1} is split between a qs byte and three bits of a qh byte, each
// 8-value lane has its own +/-0.125 delta bit, and two 3-bit fields scale the two 16-value
// halves of a 32-value group. The pinned dot (src/kernels/cuda/iq_kernels.cu:241, summing at
// :271) is mathematically the decoder's (ggml/src/ggml-quants.c:2675) term by term - unlike
// the IQ2 family there is no truncation - so the kernel is compared with an independent
// transcription of the CUDA accumulation AND with the decoder inside a plain float-order
// band. All 2048 grid entries are recovered through the decoder and the codebook compiled
// into the kernel is pinned byte for byte against those readings.
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
template<class F> void rejects(F f){bool bad=false;try{f();}catch(const std::exception&){bad=true;}require(bad,"invalid IQ1_M payload or policy accepted");}
int signed_code(uint8_t b){return b<128?b:int(b)-256;}
constexpr uint32_t super=256,groups=8,lanes=4,lane_values=8,xs_bytes=56,grid_entries=2048,q8_bytes=34;
constexpr uint32_t qs_at=0,qh_at=32,scales_at=48;
constexpr float iq1_delta=0.125f;
struct Oracle { long double value,magnitude,bound; };
struct Row {
    uint32_t columns;
    std::vector<uint16_t> index;                  // [blocks][32] 11-bit ternary grid index per 8-value lane
    std::vector<uint8_t> delta;                   // [blocks][32] one +/-0.125 bit per lane
    std::vector<uint16_t> word;                   // [blocks][4] scale words: two 3-bit fields per group, bits 0..11
    std::vector<uint16_t> d;                      // [blocks] the assembled fp16 bits
};
const ggml_type_traits* traits() {
    const auto* t=ggml_get_type_traits(GGML_TYPE_IQ1_M);
    require(t && t->blck_size==int(super) && t->type_size==xs_bytes && t->to_float,"pinned ggml lacks the IQ1_M row decoder or its block size differs");
    return t;
}
std::vector<float> decode_row(const std::vector<uint8_t>& bytes,uint32_t columns,uint32_t row) {
    std::vector<float> weights(columns);
    traits()->to_float(bytes.data()+uint64_t(row)*packed_row_bytes(29,columns),weights.data(),int64_t(columns));
    return weights;
}
uint32_t blocks(uint32_t columns){return columns/super;}
// The measured map: qs[32] at 0 (byte k owns values 8k..8k+7), qh[16] at 32 (byte 2g+l/2 holds
// index bits 8..10 of lane l of group g at 4*(l%2) and that lane's delta bit one higher), and
// scales[8] at 48 as four 16-bit words whose bits 0..11 carry the two 3-bit fields of their
// two groups and whose top nibbles assemble the block's fp16 scale.
std::vector<uint8_t> x_encode(const std::vector<Row>& rows) {
    const auto columns=rows.front().columns;
    std::vector<uint8_t> bytes(uint64_t(rows.size())*blocks(columns)*xs_bytes,0);
    for(size_t row=0;row<rows.size();++row) {const auto& w=rows[row];
        for(uint32_t sb=0;sb<blocks(columns);++sb) {
            const size_t at=(uint64_t(row)*blocks(columns)+sb)*xs_bytes;
            for(uint32_t lane=0;lane<32;++lane) {
                bytes[at+qs_at+lane]=uint8_t(w.index[uint64_t(sb)*32+lane]&255u);
                bytes[at+qh_at+lane/2]|=uint8_t((uint32_t(w.index[uint64_t(sb)*32+lane]>>8)&7u)<<(4*(lane%2)));
                if(w.delta[uint64_t(sb)*32+lane]) bytes[at+qh_at+lane/2]|=uint8_t(1u<<(4*(lane%2)+3));
            }
            for(uint32_t i=0;i<4;++i) {
                const uint16_t nibble=uint16_t((w.d[sb]>>(4*i))&15u);
                bytes[at+scales_at+2*i+1]=uint8_t(bytes[at+scales_at+2*i+1]|(nibble<<4));
                bytes[at+scales_at+2*i]=uint8_t(w.word[sb*4+i]&255u);
                bytes[at+scales_at+2*i+1]=uint8_t(bytes[at+scales_at+2*i+1]|(uint8_t(w.word[sb*4+i]>>8)&0x0fu));
            }
        }
    }
    return bytes;
}
uint16_t assembled(const std::vector<uint8_t>& bytes,size_t at) {
    const uint32_t w0=uint16_t(bytes[at+48])|(uint16_t(bytes[at+49])<<8);
    const uint32_t w1=uint16_t(bytes[at+50])|(uint16_t(bytes[at+51])<<8);
    const uint32_t w2=uint16_t(bytes[at+52])|(uint16_t(bytes[at+53])<<8);
    const uint32_t w3=uint16_t(bytes[at+54])|(uint16_t(bytes[at+55])<<8);
    return uint16_t((w0>>12)|((w1>>8)&0xf0u)|((w2>>4)&0xf00u)|(w3&0xf000u));
}
// Eight ternary values of one 11-bit index, read back through the decoder with the scale set
// to exactly 1.0, every field clear (so the factor is 2*0+1 = 1) and the delta bit of lane 0
// clear, which leaves value j = grid[j] + 0.125.
std::array<int,8> pattern_of(uint32_t index) {
    std::vector<uint8_t> bytes(xs_bytes,0);
    const uint16_t one=strata::kernels::f16_from_f32(1.0f);
    bytes[qs_at]=uint8_t(index&255u);bytes[qh_at]=uint8_t((index>>8)&7u);
    for(uint32_t i=0;i<4;++i) bytes[scales_at+2*i+1]=uint8_t(uint16_t((one>>(4*i))&15u)<<4);
    const auto weights=decode_row(bytes,super,0);
    std::array<int,8> out{};
    for(uint32_t j=0;j<lane_values;++j) {
        const double value=double(weights[j])-0.125;   // the clear delta bit adds exactly +0.125
        require(std::abs(value-std::round(value))<1e-3,"the IQ1_M grid is not integral through the decoder");
        out[j]=int(std::round(value));
        require(out[j]>=-1 && out[j]<=1,"an IQ1_M grid value left the ternary alphabet");
    }
    return out;
}
const std::array<std::array<int,8>,grid_entries>& grid_table() {
    static const std::array<std::array<int,8>,grid_entries> table=[]{
        std::array<std::array<int,8>,grid_entries> built{};
        for(uint32_t index=0;index<grid_entries;++index) built[index]=pattern_of(index);
        return built;
    }();
    return table;
}
uint32_t grid_pins=0,table_bytes=0;
void pin_grid() {
    std::vector<std::array<int,8>> seen(grid_entries);
    for(uint32_t index=0;index<grid_entries;++index) {
        const auto m=grid_table()[index];
        seen[index]=m;++grid_pins;
    }
    uint32_t collisions=0;
    for(uint32_t a=0;a<grid_entries;++a) for(uint32_t b=a+1;b<grid_entries;++b) if(seen[a]==seen[b]) ++collisions;
    require(collisions==0,"IQ1_M grid entries are not distinguishable through the decoder");
    // The three qh index bits must be readable: index 0, 256, 512 ... 1792 are all different.
    require(grid_table()[0]!=grid_table()[1024] && grid_table()[0]!=grid_table()[2047],
        "the IQ1_M index bits above 8 are not observable through the decoder");
    uint32_t neighbours=0;
    for(uint32_t index=0;index+1<grid_entries;++index) if(grid_table()[index]!=grid_table()[index+1]) ++neighbours;
    require(neighbours==grid_entries-1,"an IQ1_M grid index has an indistinguishable neighbour");
    // Pin the codebook compiled into the kernel, byte for byte, against those decoder readings.
    for(uint32_t index=0;index<grid_entries;++index) {
        const auto m=grid_table()[index];
        const unsigned int entry=packed_emulator::packed_iq1s_grid_gpu[index];
        for(uint32_t j=0;j<lane_values;++j) {
            const unsigned int nibble=(entry>>(8*(j<4?j:j-4)+(j<4?0:4)))&15u;
            require(int(nibble)==m[j]+1,"the IQ1_M codebook compiled into the TPC kernel differs from the pinned decoder");
            require(nibble<=2u,"an IQ1_M kernel grid nibble is outside the ternary range");
            ++table_bytes;
        }
    }
}
uint32_t delta_checks=0;
void delta_rule() {
    // Each lane's delta bit is independent and plain: setting it subtracts exactly 0.25 of the
    // group factor from all eight values of that lane - no parity fold, no shared sign.
    for(uint32_t lane=0;lane<3;++lane) {
        std::vector<uint8_t> clear(xs_bytes,0),set(xs_bytes,0);
        const uint16_t one=strata::kernels::f16_from_f32(1.0f);
        for(std::vector<uint8_t>* v:{&clear,&set}) {
            (*v)[qs_at+lane]=1;
            for(uint32_t i=0;i<4;++i) (*v)[scales_at+2*i+1]=uint8_t(uint16_t((one>>(4*i))&15u)<<4);
        }
        set[qh_at+lane/2]|=uint8_t(1u<<(4*(lane%2)+3));
        const auto a=decode_row(clear,super,0),b=decode_row(set,super,0);
        for(uint32_t j=0;j<lane_values;++j) {
            const uint32_t v=lane*8+j;
            require(std::abs(double(b[v])-double(a[v])+0.25)<1e-4,"an IQ1_M delta bit did not shift its own lane by -0.25");
            ++delta_checks;
        }
        for(uint32_t other=0;other<super;++other) if(other<lane*8 || other>=lane*8+8)
            require(a[other]==b[other],"an IQ1_M delta bit moved values outside its lane");
    }
}
Row x_logical(uint32_t columns,uint32_t row) {
    const auto count=blocks(columns);
    Row w{columns,std::vector<uint16_t>(uint64_t(count)*32),std::vector<uint8_t>(uint64_t(count)*32),
        std::vector<uint16_t>(uint64_t(count)*4),std::vector<uint16_t>(count)};
    for(uint32_t sb=0;sb<count;++sb) {
        const int signed_d=int((sb*7+row*3+4)%11)-5;
        w.d[sb]=strata::kernels::f16_from_f32(float(signed_d)/8.f);
        for(uint32_t lane=0;lane<32;++lane) {
            w.index[uint64_t(sb)*32+lane]=uint16_t((lane*37+sb*11+row*5)%grid_entries);
            w.delta[uint64_t(sb)*32+lane]=uint8_t((lane*13+sb*5+row)%2);
        }
        for(uint32_t g=0;g<groups;++g) {
            const uint32_t f0=uint32_t((g*3+row)%8),f1=uint32_t((g*5+sb*2+row)%8);
            w.word[uint64_t(sb)*4+g/2]|=uint16_t((f0|(f1<<3))<<(6*(g%2)));
        }
    }
    return w;
}
int field(const Row& w,uint32_t sb,uint32_t group,bool second) {
    const uint32_t tmp=uint32_t(w.word[sb*4+group/2]>>(6*(group%2)));
    return int((tmp>>(second?3:0))&7u);
}
// Independent transcription of the pinned CUDA accumulation: per 16-value half,
// (sumi + delta*sumy) * (2*field+1) with delta = -1 + 0.125 - (bit ? 0.25 : 0).
Oracle x_oracle(const Row& w,const Q8Activation& q,uint32_t token) {
    Oracle out{0,0,0};
    for(uint32_t sb=0;sb<blocks(w.columns);++sb) {
        const float d=strata::kernels::f32_from_f16(w.d[sb]);
        for(uint32_t group=0;group<groups;++group) {
            int sumi[2]={0,0};float sumf[2]={0.f,0.f};
            for(uint32_t lane=0;lane<lanes;++lane) {
                const uint32_t code=group*4+lane;
                const auto m=grid_table()[w.index[uint64_t(sb)*32+code]];
                const float delta=-1.0f+iq1_delta-(w.delta[uint64_t(sb)*32+code]?2*iq1_delta:0.f);
                int nibble_dot=0,code_sum=0;
                for(uint32_t j=0;j<lane_values;++j) {
                    const int value=signed_code(q.codes[uint64_t(token)*w.columns+sb*super+group*32+lane*8+j]);
                    nibble_dot+=(m[j]+1)*value;code_sum+=value;
                }
                sumi[lane/2]+=nibble_dot;sumf[lane/2]+=delta*(float)code_sum;
            }
            const float factor=d*q.metadata[(uint64_t(token)*w.columns/32+sb*groups+group)*2];
            const long double term=static_cast<long double>(factor)*
                ((static_cast<long double>(sumi[0])+sumf[0])*(1+2*field(w,sb,group,false))
                +(static_cast<long double>(sumi[1])+sumf[1])*(1+2*field(w,sb,group,true)));
            out.value+=term;out.magnitude+=std::abs(term);
            out.bound+=std::abs(static_cast<long double>(factor))*(1+2*std::max(field(w,sb,group,false),field(w,sb,group,true)));
        }
    }
    return out;
}
// The kernel's own float32 accumulation, replicated term by term, so a flow can tell a
// cancellation band apart from an arithmetic disagreement.
float x_accumulated(const Row& w,const Q8Activation& q,uint32_t token) {
    float total=0;
    for(uint32_t sb=0;sb<blocks(w.columns);++sb) for(uint32_t group=0;group<groups;++group) {
        int sumi[2]={0,0};float sumf[2]={0.f,0.f};
        for(uint32_t lane=0;lane<lanes;++lane) {
            const uint32_t code=group*4+lane;
            const auto m=grid_table()[w.index[uint64_t(sb)*32+code]];
            const float delta=-1.0f+iq1_delta-(w.delta[uint64_t(sb)*32+code]?2*iq1_delta:0.f);
            for(uint32_t j=0;j<lane_values;++j) {
                const int value=signed_code(q.codes[uint64_t(token)*w.columns+sb*super+group*32+lane*8+j]);
                sumi[lane/2]+=(m[j]+1)*value;sumf[lane/2]+=delta*(float)value;
            }
        }
        const float d=strata::kernels::f32_from_f16(w.d[sb]);
        const float d8=q.metadata[(uint64_t(token)*w.columns/32+sb*groups+group)*2];
        total+=(d*d8)*(((float)sumi[0]+sumf[0])*(float)(1+2*field(w,sb,group,false))
            +((float)sumi[1]+sumf[1])*(float)(1+2*field(w,sb,group,true)));
    }
    return total;
}
Oracle x_decoded(const std::vector<uint8_t>& bytes,uint32_t columns,uint32_t row,const Q8Activation& q,uint32_t token) {
    const auto weights=decode_row(bytes,columns,row);
    Oracle out{0,0,0};
    for(uint32_t block=0;block<columns/32;++block) {
        const long double d8=q.metadata[(uint64_t(token)*columns/32+block)*2];long double part=0;
        for(uint32_t v=0;v<32;++v) part+=static_cast<long double>(weights[block*32+v])*signed_code(q.codes[uint64_t(token)*columns+block*32+v]);
        const long double term=d8*part;out.value+=term;out.magnitude+=std::abs(term);
        const size_t at=uint64_t(row)*packed_row_bytes(29,columns)+uint64_t(block/8)*xs_bytes;
        out.bound+=std::abs(static_cast<long double>(strata::kernels::f32_from_f16(assembled(bytes,at)))*d8);
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
bool close_exact(float actual,long double expected) {
    return std::isfinite(actual) && std::abs(static_cast<long double>(actual)-expected)<=std::max<long double>(2e-3L,std::abs(expected)*2e-5L);
}
// The two authorities agree in value; only float order separates them, and both sums share the
// same group order, so the band is the size of the terms, not of the cancelled result.
bool close_float(float actual,const Oracle& expected) {
    if(!std::isfinite(actual)) return false;
    const auto difference=std::abs(static_cast<long double>(actual)-expected.value);
    return difference<=std::max<long double>(2e-3L,expected.magnitude*2e-5L);
}
std::vector<float> source(uint64_t values,uint32_t seed) {
    std::vector<float> out(values);for(uint64_t i=0;i<values;++i) out[i]=float(int((i*23+seed)%127)-63)*.03125f+.0140625f;return out;
}
uint32_t cases_count=0;
void cases(uint32_t columns,uint32_t batch) {
    constexpr uint32_t rows=3;std::vector<Row> logical_rows;
    for(uint32_t row=0;row<rows;++row) logical_rows.push_back(x_logical(columns,row));
    const auto bytes=x_encode(logical_rows);
    validate_packed_rows(bytes,29,columns,rows);
    const auto values=source(uint64_t(batch)*columns,batch);
    const auto q=quantize_q8_activation(values,columns);
    const auto actual=tpc(bytes,29,columns,rows,q,batch,PackedActivation::Q8_1);
    const auto routed=tpc(bytes,29,columns,rows,q,batch,PackedActivation::RoutedQ8);
    for(uint32_t token=0;token<batch;++token) for(uint32_t row=0;row<rows;++row) {
        const auto index=uint64_t(token)*rows+row;
        require(close_exact(actual[index],x_oracle(logical_rows[row],q,token).value),
            "IQ1_M TPC differs from the independent CUDA-form oracle");
        require(close_float(actual[index],x_decoded(bytes,columns,row,q,token)),
            "IQ1_M TPC differs from the unmodified pinned ggml decoder");
        require(routed[index]==actual[index],"IQ1_M dot differs between the routed and shared/head q8_1 contracts");
        std::vector<float> one(values.begin()+uint64_t(token)*columns,values.begin()+uint64_t(token)*columns+columns);
        const auto isolated=tpc(bytes,29,columns,rows,quantize_q8_activation(one,columns),1,PackedActivation::Q8_1);
        require(isolated[row]==actual[index],"batched IQ1_M changed single-row arithmetic or crossed tokens");
        ++cases_count;
    }
}
void counterexamples() {
    constexpr uint32_t columns=512;
    const auto w=x_logical(columns,0);const auto bytes=x_encode({w});
    const auto values=source(columns,7);const auto q=quantize_q8_activation(values,columns);
    const auto before=tpc(bytes,29,columns,1,q,1,PackedActivation::Q8_1)[0];
    require(close_exact(before,x_oracle(w,q,0).value) && close_float(before,x_decoded(bytes,columns,0,q,0)),"IQ1_M baseline disagrees with its authorities");
    auto raw=q;for(uint32_t block=0;block<columns/32;++block) const_cast<Q8Activation&>(raw).metadata[block*2+1]=1234.5f;
    require(tpc(bytes,29,columns,1,raw,1,PackedActivation::Q8_1)[0]==before,"IQ1_M output moved when only raw sums changed");
    uint32_t observable=0,moved=0;
    for(uint32_t v=0;v<columns && observable<4;++v) {
        if(!signed_code(q.codes[v]) || !q.metadata[(v/32)*2]) continue;
        ++observable;
        auto altered=w;const uint32_t sb=v/super,lane=(v%super)/lane_values;
        altered.index[sb*32+lane]=uint16_t((altered.index[sb*32+lane]+1)%grid_entries);
        const auto altered_bytes=x_encode({altered});validate_packed_rows(altered_bytes,29,columns,1);
        const auto result=tpc(altered_bytes,29,columns,1,q,1,PackedActivation::Q8_1)[0];
        require(close_exact(result,x_oracle(altered,q,0).value) && close_float(result,x_decoded(altered_bytes,columns,0,q,0)),
            "the changed IQ1_M index lost its oracle or decoder match");
        // A ternary pattern can coincide with its neighbour on the values this activation
        // weights, so the counterexample counts the lanes that actually moved.
        if(result!=before) ++moved;
    }
    require(observable==4,"too few observable IQ1_M lanes to pin a single grid index");
    require(moved>=1,"changing IQ1_M grid indices left the dot unchanged on every lane");
    // The three index bits above 8 live in qh: flipping them must select a different pattern.
    uint32_t raised=0;
    for(uint32_t lane=0;lane<16 && raised<2;++lane) {
        auto lifted=w;lifted.index[lane]^=uint16_t(0x400u);
        const auto lifted_bytes=x_encode({lifted});validate_packed_rows(lifted_bytes,29,columns,1);
        const auto result=tpc(lifted_bytes,29,columns,1,q,1,PackedActivation::Q8_1)[0];
        require(close_exact(result,x_oracle(lifted,q,0).value) && close_float(result,x_decoded(lifted_bytes,columns,0,q,0)),
            "flipping an IQ1_M high index bit left an authority");
        if(result!=before) ++raised;
    }
    require(raised>=1,"flipping IQ1_M index bits 8..10 changed nothing, so qh is unread");
    // The delta bit is per lane: flipping one lane must move only that lane's eight values.
    auto flipped=w;flipped.delta[0]^=1u;
    const auto flipped_bytes=x_encode({flipped});validate_packed_rows(flipped_bytes,29,columns,1);
    const auto flipped_result=tpc(flipped_bytes,29,columns,1,q,1,PackedActivation::Q8_1)[0];
    require(flipped_result!=before && close_exact(flipped_result,x_oracle(flipped,q,0).value)
        && close_float(flipped_result,x_decoded(flipped_bytes,columns,0,q,0)),
        "flipping an IQ1_M delta bit did nothing or lost an authority");
    // The two 3-bit fields of one scale word are independent: change only the odd half.
    auto high=w;high.word[0]^=uint16_t(7u<<3);
    const auto high_bytes=x_encode({high});validate_packed_rows(high_bytes,29,columns,1);
    require(close_exact(tpc(high_bytes,29,columns,1,q,1,PackedActivation::Q8_1)[0],x_oracle(high,q,0).value),
        "the IQ1_M second field of a scale word did not follow the independent oracle");
    // Every field cleared leaves the factor 2*0+1 = 1, which must change the dot.
    auto zeroed=w;for(auto& value:zeroed.word) value=0;
    const auto zeroed_bytes=x_encode({zeroed});validate_packed_rows(zeroed_bytes,29,columns,1);
    const auto zeroed_result=tpc(zeroed_bytes,29,columns,1,q,1,PackedActivation::Q8_1)[0];
    require(zeroed_result!=before && close_exact(zeroed_result,x_oracle(zeroed,q,0).value)
        && close_float(zeroed_result,x_decoded(zeroed_bytes,columns,0,q,0)),
        "clearing the IQ1_M fields left the dot, the oracle or the decoder inconsistent");
    // Negating the assembled scale negates the dot: the nibble split must not lose the sign.
    auto negated=w;for(auto& value:negated.d) value=strata::kernels::f16_from_f32(-strata::kernels::f32_from_f16(value));
    const auto neg_bytes=x_encode({negated});validate_packed_rows(neg_bytes,29,columns,1);
    require(tpc(neg_bytes,29,columns,1,q,1,PackedActivation::Q8_1)[0]==-before,"negating the IQ1_M assembled scale did not negate the dot");
    const auto neg_decoded=decode_row(neg_bytes,columns,0);
    const auto plain=decode_row(bytes,columns,0);
    uint32_t mirrored=0;for(uint32_t v=0;v<columns;++v) mirrored+=int(std::abs(neg_decoded[v]+plain[v])<1e-9);
    require(mirrored==columns,"the IQ1_M decoder did not mirror the negated assembled scale");
}
// The packed type set is asserted as a whole: adding a format cannot leave a stale
// negative assertion behind in an older test.
void policy_types() {
    for(uint32_t type:{2,3,6,7,8,11,12,13,14,16,17,18,20,21,22,23,29,40,42}) require(packed_supported(type),"a packed GGML type with a working dot was rejected");
    for(uint32_t type:{19,30,31}) require(!packed_supported(type),"a still-decoded GGML type was accepted as packed");
}
void guards() {
    policy_types();
    const auto w=x_logical(256,0);auto bytes=x_encode({w});
    // 0x7c00 assembled from the nibbles (w3 top = 7, w2 top = c) is +inf and must be refused.
    bytes[scales_at+7]=0x70;bytes[scales_at+5]=uint8_t((bytes[scales_at+5]&0x0fu)|0xc0u);
    rejects([&]{validate_packed_rows(bytes,29,256,1);});
    require(assembled(bytes,0)==0x7c00,"the IQ1_M guard did not assemble the scale it checks");
    const auto clean=x_encode({w});validate_packed_rows(clean,29,256,1);
    auto wild=w;for(uint32_t lane=0;lane<32;++lane) {wild.index[lane]=uint16_t((lane*255)%grid_entries);wild.delta[lane]=1u;}
    for(auto& value:wild.word) value=0xffffu&uint16_t(value|0x0fffu);
    const auto wild_bytes=x_encode({wild});
    require(assembled(wild_bytes,0)==strata::kernels::f16_from_f32(float(int((0*7+0*3+4)%11)-5)/8.f),
        "the IQ1_M field bits leaked into the assembled scale");
    rejects([&]{(void)packed_row_bytes(29,640);});rejects([&]{(void)packed_row_bytes(29,300);});
    require(packed_supported(29) && packed_values_per_block(29)==super && packed_block_bytes(29)==xs_bytes
        && packed_row_bytes(29,2560)==560 && packed_row_bytes(29,256)==xs_bytes,"IQ1_M geometry wrong");
    require(packed_activation_supported(29,PackedActivation::Q8_1) && packed_activation_supported(29,PackedActivation::RoutedQ8)
        && !packed_activation_supported(29,PackedActivation::RoutedS2),"IQ1_M policy wrong");
    const auto routed=expert_format(2560,640,0,{29,29,8});
    require(routed.packed && routed.activation==PackedActivation::RoutedQ8
        && routed.bytes==uint64_t(560)*640*2+uint64_t(34)*20*2560,"routed IQ1_M expert charge wrong");
    const auto shared=expert_format(2560,640,-1,{29,22,8});
    require(shared.packed && shared.activation==PackedActivation::Q8_1
        && shared.bytes==uint64_t(560)*640+uint64_t(820)*640+uint64_t(34)*20*2560,"shared IQ1_M expert was not packed or is mischarged");
    const auto down=expert_format(2560,640,0,{8,8,29});
    require(!down.packed && down.bytes==uint64_t(2560)*640*6,"IQ1_M down projection with 640 columns was not rejected by layout");
    // The grid is 8 KiB of kernel constants: 2048 four-byte entries, so the packed row must be
    // 56 bytes per 256 values and never a fraction of a block.
    require(packed_layout_supported(29,256) && !packed_layout_supported(29,257),"IQ1_M accepted a partial superblock");
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
    r.bytes=x_encode(r.xs);validate_packed_rows(r.bytes,29,columns,rows);return r;
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
            out[uint64_t(token)*r.rows+row]=Oracle{total,magnitude,0};continue;
        }
        out[uint64_t(token)*r.rows+row]=x_oracle(r.xs[row],q,token);
    }
    return out;
}
uint32_t flows=0;
void shared_flows() {
    constexpr uint32_t hidden=512,intermediate=256;
    // These rows are IQ1_M-shaped, so only IQ1_M and Q8_0 roles can appear here; experts that
    // mix several packed formats are exercised by tests/packed_expert.cpp's real layout source.
    for(auto types:{std::array<uint32_t,3>{29,29,29},std::array<uint32_t,3>{29,29,8},std::array<uint32_t,3>{29,8,29},std::array<uint32_t,3>{8,29,29}})
    for(uint32_t batch:{1u,3u}) {
        const auto format=expert_format(hidden,intermediate,-1,types);
        require(format.packed && format.activation==PackedActivation::Q8_1,"shared IQ1_M expert lost the head contract");
        std::array<Flow,3> role{flow_role(types[0],hidden,intermediate,0),flow_role(types[1],hidden,intermediate,1),flow_role(types[2],intermediate,hidden,2)};
        const uint64_t charge=packed_row_bytes(types[0],hidden)*intermediate+packed_row_bytes(types[1],hidden)*intermediate
            +packed_row_bytes(types[2],intermediate)*hidden;
        require(format.bytes==charge,"shared IQ1_M expert payload mischarged");
        std::vector<float> input(uint64_t(batch)*hidden);
        for(size_t i=0;i<input.size();++i) input[i]=float(int((i*29+batch)%113)-56)*.0625f;
        std::array<uint32_t,3> calls{0,0,0};
        auto run=[&](const std::array<Flow,3>& roles)->std::vector<float> {
            calls={0,0,0};
            return execute_packed_expert(std::span<const float>(input),hidden,intermediate,[&](uint32_t which,std::span<const float> values) {
                ++calls[which];
                const auto mode=expert_projection_activation(format,which);
                require(mode==PackedActivation::Q8_1,"shared IQ1_M projection inherited a routed activation contract");
                const auto tokens=uint32_t(values.size()/roles[which].columns);
                const auto q=quantize_q8_activation(values,roles[which].columns);
                const auto expected=flow_oracle(roles[which],q,tokens);
                const auto actual=tpc(roles[which].bytes,roles[which].type,roles[which].columns,roles[which].rows,q,tokens,mode);
                for(size_t i=0;i<actual.size();++i) {
                    require(std::isfinite(actual[i]),"non-finite shared IQ1_M projection");
                    const auto tolerance=std::max({.02,std::abs(static_cast<double>(expected[i].value))*2e-3,
                        static_cast<double>(expected[i].magnitude)*8e-7});
                    require(std::abs(double(actual[i])-double(expected[i].value))<=tolerance,
                        "shared expert IQ1_M projection differs from the independent oracle");
                    if(roles[which].type==29) {
                        const uint32_t row=uint32_t(i%roles[which].rows),token=uint32_t(i/roles[which].rows);
                        require(std::abs(double(actual[i])-double(x_accumulated(roles[which].xs[row],q,token)))
                            <=std::max(1e-6,double(std::abs(actual[i]))*1e-6),
                            "the shared IQ1_M projection did not follow the documented float32 term order");
                    }
                }
                return actual;
            });
        };
        const auto output=run(role);
        require(calls==std::array<uint32_t,3>{1,1,1},"shared IQ1_M expert flow did not run each projection once");
        require(output.size()==input.size(),"shared IQ1_M expert flow output shape");
        for(float value:output) require(std::isfinite(value),"non-finite shared IQ1_M expert output");
        if(types[0]==types[1] && types[0]==29) {
            auto swapped=role;std::swap(swapped[0],swapped[1]);
            const auto other=run(swapped);bool differs=false;
            for(size_t i=0;i<output.size();++i) differs=differs||output[i]!=other[i];
            require(differs,"exchanging gate and up IQ1_M weights left the shared expert output unchanged");
        }
        ++flows;
    }
}
}
int main(){try {
    pin_grid();delta_rule();
    for(uint32_t columns:{256u,512u,2560u}) for(uint32_t batch:{1u,3u,8u}) cases(columns,batch);
    counterexamples();guards();shared_flows();
    std::cout<<"PASS IQ1_M TPC "<<cases_count<<" cases against the independent CUDA-form oracle ((sumi + delta*sumy)*(2*field+1), no truncation) AND the unmodified pinned ggml IQ1_M row decoder, "
        <<grid_pins<<" 11-bit ternary grid patterns recovered through the decoder and shown distinguishable including every neighbour pair, "
        <<table_bytes<<" codebook nibbles compiled into the kernel matched against those decoder readings, "
        <<delta_checks<<" per-lane delta checks, "<<flows<<" production shared-expert flows, index/high-bit/delta/field-independence/negation counterexamples and policy guards; no device\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
