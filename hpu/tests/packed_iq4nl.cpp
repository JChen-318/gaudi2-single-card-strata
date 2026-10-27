// Native IQ4_NL (GGML type 20) packed dot for the shared/head and routed q8_1
// contracts. IQ4_NL is nonlinear: a four-bit field is an INDEX into a fixed sixteen
// entry codebook, so the layout authority alone is not enough and every index is
// pinned individually. Two independent authorities are used, a logical-codebook
// long-double oracle and the UNMODIFIED pinned ggml row decoder, which decides the
// nibble positions and the codebook itself. Host only: no ISA execution, no device,
// no performance claim.
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
template<class F> void rejects(F f){bool bad=false;try{f();}catch(const std::exception&){bad=true;}require(bad,"invalid IQ4_NL payload or policy accepted");}
int signed_code(uint8_t b){return b<128?b:int(b)-256;}
constexpr uint32_t block_values=32,iq4_bytes=18,q8_bytes=34;
uint32_t blocks(uint32_t columns){return columns/block_values;}
// Transcribed twice independently: pinned ggml src/ggml-common.h:1120 (kvalues_iq4nl)
// and the fixed CUDA native_mmvq.cu:449 (iq4nl_values) agree on these sixteen int8
// values. codebook_pins() below proves the kernel uses exactly this ordering rather
// than trusting the transcription.
constexpr std::array<int,16> codebook{{-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113}};
struct Oracle { long double value,magnitude; };
struct IQRow {
    uint32_t columns;
    std::vector<int> codes;                     // [columns] 0..15 codebook indices
    std::vector<float> d;                       // [blocks] one signed block scale
};
IQRow iq4_logical(uint32_t columns,uint32_t row) {
    const auto count=blocks(columns);
    IQRow w{columns,std::vector<int>(columns),std::vector<float>(count)};
    for(uint32_t b=0;b<count;++b) w.d[b]=float(int((b*7+row*3+4)%11)-5)/16.f;  // both signs, never zero
    for(uint32_t v=0;v<columns;++v) w.codes[v]=int((v*13+(v>>4)*7+(v/block_values)*5+row*3)%16);
    // A purely linear stride would repeat every 16 values and give the two nibbles of
    // a byte identical indices, so the (v>>4) term keeps low and high halves apart.
    require(std::any_of(w.codes.begin(),w.codes.end(),[](int c){return c==15;}),"IQ4_NL fixture never reached index 15");
    for(uint32_t v=0;v+16<columns;++v) if(v%32<16) require(w.codes[v]!=w.codes[v+16],"IQ4_NL fixture gave a nibble pair identical indices");
    require(std::all_of(w.codes.begin(),w.codes.end(),[](int c){return c>=0 && c<16;}),"IQ4_NL fixture index out of range");
    return w;
}
// Layout: ggml_half d at 0, then qs[16] at 2 where the low nibble of qs[j] is value
// j and the high nibble of qs[j] is value j+16 (dequantize_row_iq4_nl).
std::vector<uint8_t> iq4_encode(const std::vector<IQRow>& rows) {
    const auto columns=rows.front().columns;
    std::vector<uint8_t> bytes(uint64_t(rows.size())*blocks(columns)*iq4_bytes,0);
    const auto put=[&](size_t at,float value){const auto bits=strata::kernels::f16_from_f32(value);bytes[at]=uint8_t(bits);bytes[at+1]=uint8_t(bits>>8);};
    for(size_t row=0;row<rows.size();++row) {const auto& w=rows[row];
        for(uint32_t b=0;b<blocks(columns);++b) {
            const size_t at=(uint64_t(row)*blocks(columns)+b)*iq4_bytes;
            for(uint32_t j=0;j<16;++j) {
                bytes[at+2+j]|=uint8_t(uint32_t(w.codes[b*block_values+j])&15u);
                bytes[at+2+j]|=uint8_t(uint32_t(w.codes[b*block_values+j+16]&15u)<<4);
            }
            put(at,w.d[b]);
        }
    }
    return bytes;
}
// Logical-codebook oracle in the upstream shape: d = w->d * activation d8, then the
// single integer dot. The raw-input sum stored by Q8_1 never appears.
Oracle iq4_oracle(const std::vector<uint8_t>& bytes,const IQRow& w,uint32_t row,const Q8Activation& q,uint32_t token,
                  const std::array<int,16>& table=codebook) {
    Oracle out{0,0};
    const size_t base=uint64_t(row)*packed_row_bytes(20,w.columns);
    for(uint32_t b=0;b<blocks(w.columns);++b) {
        const size_t at=base+uint64_t(b)*iq4_bytes;
        const float d=strata::kernels::f32_from_f16(uint16_t(bytes[at])|(uint16_t(bytes[at+1])<<8));
        require(std::abs(double(d)-double(w.d[b]))<=1e-9,"IQ4_NL fixture d disagrees with its own encoded half");
        int dot=0;
        for(uint32_t v=0;v<block_values;++v) {
            const int code=w.codes[b*block_values+v];
            dot+=table[code]*signed_code(q.codes[uint64_t(token)*w.columns+b*block_values+v]);
        }
        const long double term=static_cast<long double>(d)*q.metadata[(uint64_t(token)*w.columns/32+b)*2]*dot;
        out.value+=term;out.magnitude+=std::abs(term);
    }
    return out;
}
// Byte-layout and codebook authority: decode with the unmodified pinned ggml row
// decoder, then dot with the values the q8_1 activation represents.
Oracle iq4_decoded(const std::vector<uint8_t>& bytes,uint32_t columns,uint32_t row,const Q8Activation& q,uint32_t token) {
    const auto* traits=ggml_get_type_traits(GGML_TYPE_IQ4_NL);
    require(traits && traits->blck_size==int(block_values) && traits->to_float,"pinned ggml lacks the IQ4_NL row decoder");
    require(traits->type_size==iq4_bytes,"pinned ggml IQ4_NL block size differs from the packed charge");
    std::vector<float> weights(columns);
    traits->to_float(bytes.data()+uint64_t(row)*packed_row_bytes(20,columns),weights.data(),int64_t(columns));
    Oracle out{0,0};
    for(uint32_t b=0;b<blocks(columns);++b) {
        const long double d8=q.metadata[(uint64_t(token)*columns/32+b)*2];long double part=0;
        for(uint32_t v=0;v<block_values;++v) part+=static_cast<long double>(weights[b*block_values+v])*signed_code(q.codes[uint64_t(token)*columns+b*block_values+v]);
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
    return std::abs(static_cast<long double>(actual)-expected.value)<=std::max<long double>(.02L,expected.magnitude*2e-4L);
}
std::vector<float> source(uint64_t values,uint32_t seed) {
    std::vector<float> out(values);for(uint64_t i=0;i<values;++i) out[i]=float(int((i*23+seed)%127)-63)*.03125f+.0140625f;return out;
}
// A payload whose every index is one value, so its closed form pins that codebook
// entry: result == table[index] * d * d8 * sum(activation codes).
std::vector<uint8_t> uniform(uint32_t columns,uint32_t index,float d) {
    IQRow w{columns,std::vector<int>(columns,int(index)),std::vector<float>(blocks(columns),d)};
    return iq4_encode({w});
}
void cases(uint32_t columns,uint32_t batch,uint32_t& count) {
    constexpr uint32_t rows=3;std::vector<IQRow> logical_rows;
    for(uint32_t row=0;row<rows;++row) logical_rows.push_back(iq4_logical(columns,row));
    const auto bytes=iq4_encode(logical_rows);
    validate_packed_rows(bytes,20,columns,rows);
    const auto values=source(uint64_t(batch)*columns,batch);
    const auto q=quantize_q8_activation(values,columns);
    const auto actual=tpc(bytes,20,columns,rows,q,batch,PackedActivation::Q8_1);
    const auto routed=tpc(bytes,20,columns,rows,q,batch,PackedActivation::RoutedQ8);
    for(uint32_t token=0;token<batch;++token) for(uint32_t row=0;row<rows;++row) {
        const auto index=uint64_t(token)*rows+row;
        require(close(actual[index],iq4_oracle(bytes,logical_rows[row],row,q,token)),"IQ4_NL TPC differs from the independent codebook oracle");
        require(close(actual[index],iq4_decoded(bytes,columns,row,q,token)),"IQ4_NL bytes disagree with the unmodified pinned ggml decoder");
        require(routed[index]==actual[index],"IQ4_NL dot differs between the routed and shared/head q8_1 contracts");
        std::vector<float> one(values.begin()+uint64_t(token)*columns,values.begin()+uint64_t(token)*columns+columns);
        const auto isolated=tpc(bytes,20,columns,rows,quantize_q8_activation(one,columns),1,PackedActivation::Q8_1);
        require(isolated[row]==actual[index],"batched IQ4_NL changed single-row arithmetic or crossed tokens");
        ++count;
    }
}
// Every one of the sixteen indices, pinned on its own against both authorities, and
// each pin falsified by changing only that codebook entry.
void codebook_pins(uint32_t& count) {
    constexpr uint32_t columns=256,rows=1;constexpr float scale=.25f;
    const auto values=source(columns,5);const auto q=quantize_q8_activation(values,columns);
    std::vector<long double> lane(blocks(columns));
    for(uint32_t b=0;b<blocks(columns);++b) {long double sum=0;for(uint32_t v=0;v<block_values;++v) sum+=signed_code(q.codes[b*block_values+v]);lane[b]=sum;}
    auto prediction=[&](const std::array<int,16>& table,uint32_t index) {
        long double total=0;for(uint32_t b=0;b<blocks(columns);++b) total+=static_cast<long double>(table[index])*scale*q.metadata[b*2]*lane[b];
        return total;};
    require(std::abs(double(prediction(codebook,0)))>1.0,"IQ4_NL pin fixture cancels every block, so no index could be pinned");
    for(uint32_t index=0;index<16;++index) {
        const auto bytes=uniform(columns,index,scale);
        validate_packed_rows(bytes,20,columns,rows);
        const auto result=tpc(bytes,20,columns,rows,q,1,PackedActivation::Q8_1)[0];
        const auto exact=prediction(codebook,index);
        require(close(result,{exact,std::abs(exact)}),"IQ4_NL codebook index does not match its pinned table entry");
        require(close(result,iq4_decoded(bytes,columns,0,q,0)),"IQ4_NL codebook index disagrees with the pinned ggml decoder");
        // The pin must be falsifiable: perturbing only this entry, or transposing the
        // adjacent pair, must move the prediction off the kernel result.
        auto perturbed=codebook;perturbed[index]=codebook[index]+7;
        const auto wrong=prediction(perturbed,index);
        require(std::abs(double(result)-double(wrong))>std::max(.02,double(std::abs(wrong))*.01),"IQ4_NL output was insensitive to its own codebook entry");
        // Transposing this entry with its neighbour (mod 16, so 15 pairs with 0) must
        // move the prediction off the kernel result for every index, not just one.
        auto pair=codebook;std::swap(pair[index],pair[(index+1)%16]);
        const auto transpose=prediction(pair,index);
        require(std::abs(double(result)-double(transpose))>std::max(.02,double(std::abs(transpose))*.01),
            "IQ4_NL output was insensitive to swapping two adjacent codebook entries");
        // A wrong index must change the value: the neighbouring index is a different
        // table entry, so the same payload shape must give a different result.
        const auto shifted=tpc(uniform(columns,(index+1)%16,scale),20,columns,rows,q,1,PackedActivation::Q8_1)[0];
        require(shifted!=result,"a one-step IQ4_NL codebook index change left the dot unchanged");
        ++count;
    }
}
void counterexamples() {
    const auto w=iq4_logical(256,0);const auto bytes=iq4_encode({w});
    const auto values=source(256,7);const auto q=quantize_q8_activation(values,256);
    const auto before=tpc(bytes,20,256,1,q,1,PackedActivation::Q8_1)[0];
    require(close(before,iq4_oracle(bytes,w,0,q,0)) && close(before,iq4_decoded(bytes,256,0,q,0)),"IQ4_NL baseline disagrees with its oracles");
    auto raw=q;for(uint32_t b=0;b<256/32;++b) const_cast<Q8Activation&>(raw).metadata[b*2+1]=1234.5f;
    require(tpc(bytes,20,256,1,raw,1,PackedActivation::Q8_1)[0]==before,"IQ4_NL output moved when only raw sums changed");
    // A single index step changes the value wherever the lane is observable: the
    // activation int8 code and the block d8 must both be non-zero.
    uint32_t stepped=0;
    for(uint32_t v=0;v<256 && stepped<6;++v) {
        if(!signed_code(q.codes[v]) || !q.metadata[(v/block_values)*2]) continue;
        ++stepped;
        auto advanced=w;advanced.codes[v]=(advanced.codes[v]+1)%16;
        const auto advanced_bytes=iq4_encode({advanced});validate_packed_rows(advanced_bytes,20,256,1);
        const auto moved=tpc(advanced_bytes,20,256,1,q,1,PackedActivation::Q8_1)[0];
        require(moved!=before,"an IQ4_NL index step was absorbed by the block scale");
        require(close(moved,iq4_oracle(advanced_bytes,advanced,0,q,0)) && close(moved,iq4_decoded(advanced_bytes,256,0,q,0)),
            "the stepped IQ4_NL index lost its oracle or decoder match");
    }
    require(stepped==6,"too few observable IQ4_NL lanes to pin a single index step");
    // Low and high nibbles are different values, 16 apart: exchanging them in one byte
    // must change the row and stay decoder-consistent.
    auto exchanged=w;std::swap(exchanged.codes[3],exchanged.codes[19]);
    require(exchanged.codes[3]!=exchanged.codes[19],"the fixture put identical indices in a nibble pair");
    const auto exchange_bytes=iq4_encode({exchanged});validate_packed_rows(exchange_bytes,20,256,1);
    require(tpc(exchange_bytes,20,256,1,q,1,PackedActivation::Q8_1)[0]!=before,"exchanging an IQ4_NL low/high nibble pair changed nothing");
    // Zero d annihilates the block; negating d negates the result.
    auto zeroed=w;for(auto& value:zeroed.d) value=0.f;
    const auto zero_bytes=iq4_encode({zeroed});validate_packed_rows(zero_bytes,20,256,1);
    require(tpc(zero_bytes,20,256,1,q,1,PackedActivation::Q8_1)[0]==0.f,"a zero IQ4_NL block scale did not zero its contribution");
    auto negated=w;for(auto& value:negated.d) value=-value;
    const auto neg_bytes=iq4_encode({negated});validate_packed_rows(neg_bytes,20,256,1);
    require(tpc(neg_bytes,20,256,1,q,1,PackedActivation::Q8_1)[0]==-before,"negating the IQ4_NL block scale did not negate the dot");
    require(close(tpc(neg_bytes,20,256,1,q,1,PackedActivation::Q8_1)[0],iq4_decoded(neg_bytes,256,0,q,0)),"negated IQ4_NL scales stopped matching the decoder");
    // Permuting the codebook order inside the weight row (index 15 -> 0) is a data
    // change the decoder sees too: it must move the result.
    auto folded=w;for(auto& code:folded.codes) if(code==15) code=0;
    const auto fold_bytes=iq4_encode({folded});validate_packed_rows(fold_bytes,20,256,1);
    const auto folded_result=tpc(fold_bytes,20,256,1,q,1,PackedActivation::Q8_1)[0];
    require(folded_result!=before,"collapsing IQ4_NL index 15 to 0 left the dot unchanged");
    require(close(folded_result,iq4_oracle(fold_bytes,folded,0,q,0)) && close(folded_result,iq4_decoded(fold_bytes,256,0,q,0)),
        "the collapsed IQ4_NL row lost its oracle or decoder match");
}
// The packed type set is asserted as a whole: adding a format cannot leave a stale
// negative assertion behind in an older test. The IQ families and BF16 must stay on
// the decoded side until each has its own dot and dual-authority test.
void policy_types() {
    for(uint32_t type:{2,3,6,7,8,11,12,13,14,16,17,18,20,21,22,23,29,40,42}) require(packed_supported(type),"a packed GGML type with a working dot was rejected");
    for(uint32_t type:{19,30,31}) require(!packed_supported(type),"a still-decoded GGML type was accepted as packed");
}
void guards() {
    policy_types();
    const auto w=iq4_logical(32,0);auto bytes=iq4_encode({w});
    bytes[0]=0;bytes[1]=0x7c;rejects([&]{validate_packed_rows(bytes,20,32,1);});
    bytes[0]=0;bytes[1]=0xfc;rejects([&]{validate_packed_rows(bytes,20,32,1);});
    bytes[0]=0;bytes[1]=0x3c;validate_packed_rows(bytes,20,32,1);
    // Any nibble pattern is legal code data.
    bytes[2]=0xff;bytes[3]=0x00;validate_packed_rows(bytes,20,32,1);
    rejects([&]{(void)packed_row_bytes(20,300);});rejects([&]{(void)packed_row_bytes(20,33);});
    require(packed_supported(20) && packed_values_per_block(20)==block_values && packed_block_bytes(20)==iq4_bytes
        && packed_row_bytes(20,2560)==1440 && packed_row_bytes(20,640)==360 && packed_row_bytes(20,32)==iq4_bytes,"IQ4_NL geometry wrong");
    require(packed_activation_supported(20,PackedActivation::Q8_1) && packed_activation_supported(20,PackedActivation::RoutedQ8)
        && !packed_activation_supported(20,PackedActivation::RoutedS2),
        "IQ4_NL policy or still-missing IQ format set wrong");
    const auto routed=expert_format(2560,640,0,{20,20,8});
    require(routed.packed && routed.activation==PackedActivation::RoutedQ8
        && routed.bytes==uint64_t(1440)*640*2+uint64_t(34)*20*2560,"routed IQ4_NL expert charge wrong");
    const auto shared=expert_format(2560,640,-1,{20,14,8});
    require(shared.packed && shared.activation==PackedActivation::Q8_1
        && shared.bytes==uint64_t(1440)*640+uint64_t(2100)*640+uint64_t(34)*20*2560,"shared IQ4_NL expert was not packed or is mischarged");
    // A 256-value superblock still cannot serve a 640-column down projection, but a
    // 32-value IQ4_NL role can sit beside it only when every role divides.
    const auto down=expert_format(2560,640,0,{20,20,14});
    require(!down.packed && down.bytes==uint64_t(2560)*640*6,"mixed IQ4_NL/Q6_K expert was not rejected by layout");
    require(expert_projection_activation(routed,0)==PackedActivation::RoutedQ8
        && expert_projection_activation(routed,2)==PackedActivation::RoutedQ8,"IQ4_NL routed role inherited the wrong activation contract");
    const auto all=expert_format(2560,640,0,{20,20,20});
    require(all.packed && expert_projection_activation(all,2)==PackedActivation::RoutedQ8,"IQ4_NL down projection lost the routed contract");
}
struct Flow {
    uint32_t type,columns,rows;
    std::vector<uint8_t> bytes;
    std::vector<IQRow> iq4;
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
    for(uint32_t row=0;row<rows;++row) r.iq4.push_back(iq4_logical(columns,row+seed*1000));
    r.bytes=iq4_encode(r.iq4);validate_packed_rows(r.bytes,20,columns,rows);return r;
}
std::vector<float> flow_oracle(const Flow& r,const Q8Activation& q,uint32_t batch) {
    std::vector<float> out(uint64_t(r.rows)*batch);
    for(uint32_t token=0;token<batch;++token) for(uint32_t row=0;row<r.rows;++row) {
        if(r.type==8) {
            long double total=0;
            for(uint32_t b=0;b<r.columns/32;++b) {
                long double dot=0;
                for(uint32_t j=0;j<32;++j) dot+=static_cast<long double>(r.q8_codes[uint64_t(row)*r.columns+b*32+j])*signed_code(q.codes[uint64_t(token)*r.columns+b*32+j]);
                total+=r.q8_d[uint64_t(row)*r.columns/32+b]*q.metadata[(uint64_t(token)*r.columns/32+b)*2]*dot;
            }
            out[uint64_t(token)*r.rows+row]=float(total);continue;
        }
        out[uint64_t(token)*r.rows+row]=float(iq4_oracle(r.bytes,r.iq4[row],row,q,token).value);
    }
    return out;
}
void shared_flows(uint32_t& count) {
    constexpr uint32_t hidden=512,intermediate=256;
    for(auto types:{std::array<uint32_t,3>{20,20,20},std::array<uint32_t,3>{20,20,8},std::array<uint32_t,3>{20,8,20},std::array<uint32_t,3>{8,20,20}})
    for(uint32_t batch:{1u,3u}) {
        const auto format=expert_format(hidden,intermediate,-1,types);
        require(format.packed && format.activation==PackedActivation::Q8_1,"shared IQ4_NL expert lost the head contract");
        std::array<Flow,3> role{flow_role(types[0],hidden,intermediate,0),flow_role(types[1],hidden,intermediate,1),flow_role(types[2],intermediate,hidden,2)};
        const uint64_t charge=packed_row_bytes(types[0],hidden)*intermediate+packed_row_bytes(types[1],hidden)*intermediate
            +packed_row_bytes(types[2],intermediate)*hidden;
        require(format.bytes==charge,"shared IQ4_NL expert payload mischarged");
        std::vector<float> input(uint64_t(batch)*hidden);
        for(size_t i=0;i<input.size();++i) input[i]=float(int((i*29+batch)%113)-56)*.0625f;
        std::array<uint32_t,3> calls{0,0,0};
        auto run=[&](const std::array<Flow,3>& roles)->std::vector<float> {
            calls={0,0,0};
            return execute_packed_expert(std::span<const float>(input),hidden,intermediate,[&](uint32_t which,std::span<const float> values) {
                ++calls[which];
                const auto mode=expert_projection_activation(format,which);
                require(mode==PackedActivation::Q8_1,"shared IQ4_NL projection inherited a routed activation contract");
                const auto tokens=uint32_t(values.size()/roles[which].columns);
                const auto q=quantize_q8_activation(values,roles[which].columns);
                const auto expected=flow_oracle(roles[which],q,tokens);
                const auto actual=tpc(roles[which].bytes,roles[which].type,roles[which].columns,roles[which].rows,q,tokens,mode);
                for(size_t i=0;i<actual.size();++i) {
                    require(std::isfinite(actual[i]),"non-finite shared IQ4_NL projection");
                    require(std::abs(double(actual[i])-double(expected[i]))<=std::max(.02,double(std::abs(expected[i]))*2e-3),
                        "shared expert IQ4_NL projection differs from the independent oracle");
                }
                return actual;
            });
        };
        const auto output=run(role);
        require(calls==std::array<uint32_t,3>{1,1,1},"shared IQ4_NL expert flow did not run each projection once");
        require(output.size()==input.size(),"shared IQ4_NL expert flow output shape");
        for(float value:output) require(std::isfinite(value),"non-finite shared IQ4_NL expert output");
        if(types[0]==types[1] && types[0]==20) {
            auto swapped=role;std::swap(swapped[0],swapped[1]);
            const auto other=run(swapped);bool differs=false;
            for(size_t i=0;i<output.size();++i) differs=differs||output[i]!=other[i];
            require(differs,"exchanging gate and up IQ4_NL weights left the shared expert output unchanged");
        }
        ++count;
    }
}
}
int main(){try {uint32_t count=0,pins=0,flows=0;
    for(uint32_t columns:{32u,256u,2560u}) for(uint32_t batch:{1u,3u,8u}) cases(columns,batch,count);
    codebook_pins(pins);counterexamples();guards();shared_flows(flows);
    std::cout<<"PASS IQ4_NL TPC "<<count<<" cases and "<<pins<<" per-index codebook pins against an independent logical-codebook oracle and the unmodified pinned ggml IQ4_NL row decoder, plus "<<flows<<" production shared-expert flows, low/high nibble split, raw-sum insensitivity and policy guards; no device\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
