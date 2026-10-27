// Native IQ3_XXS (GGML type 18) 256-value codebook superblock dot for the shared/head and
// routed q8_1 contracts. As with IQ2_XXS the two authorities differ by construction: the
// pinned CUDA dot (iq_kernels.cu:124) applies the group scale as the INTEGER
// (scale*sumi + sumi/2)/2 while ggml's row decoder multiplies the float d*(0.5+scale)*0.5.
// The kernel is therefore compared EXACTLY against an independent transcription of the CUDA
// integer form and against the pinned decoder within the bound that the truncation provably
// introduces. Grid magnitudes and the sign parity rule are recovered through the decoder, so
// nothing here trusts the table the kernel was compiled with. Host only: no ISA, no device.
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
template<class F> void rejects(F f){bool bad=false;try{f();}catch(const std::exception&){bad=true;}require(bad,"invalid IQ3_XXS payload or policy accepted");}
int signed_code(uint8_t b){return b<128?b:int(b)-256;}
constexpr uint32_t super=256,groups=8,lane_values=8,magnitudes_per_byte=4,xs_bytes=98,grid_bytes=64,q8_bytes=34;
uint32_t blocks(uint32_t columns){return columns/super;}
struct Oracle { long double value,magnitude,bound; };
struct Row {                                   // the logical view the oracle and fixture share
    uint32_t columns;
    std::vector<uint8_t> grid;                 // [blocks][64] codebook indices
    std::vector<uint32_t> word;                // [blocks][8] sign fields + 4-bit scale
    std::vector<float> d;                      // [blocks] signed
};
const ggml_type_traits* traits() {
    const auto* t=ggml_get_type_traits(GGML_TYPE_IQ3_XXS);
    require(t && t->blck_size==int(super) && t->type_size==xs_bytes && t->to_float,"pinned ggml lacks the IQ3_XXS row decoder or its block size differs");
    return t;
}
std::vector<float> decode_row(const std::vector<uint8_t>& bytes,uint32_t columns,uint32_t row) {
    std::vector<float> weights(columns);
    traits()->to_float(bytes.data()+uint64_t(row)*packed_row_bytes(18,columns),weights.data(),int64_t(columns));
    return weights;
}
// Four magnitudes of one grid index, read back through the decoder: d=1, scale nibble 0 so
// db = 1*(0.5+0)*0.5 = 0.25, signs clear, and grid byte 0 of group 0 carries the index.
std::array<int,4> magnitudes_of(uint32_t index) {
    std::vector<uint8_t> bytes(xs_bytes,0);
    bytes[0]=0x00;bytes[1]=0x3c;bytes[2]=uint8_t(index);
    const auto weights=decode_row(bytes,super,0);
    std::array<int,4> out{};
    for(uint32_t j=0;j<magnitudes_per_byte;++j) {
        const double value=double(weights[j])*4.0;
        require(std::abs(value-std::round(value))<1e-3,"IQ3_XXS grid magnitude is not integral");
        out[j]=int(std::round(value));
    }
    return out;
}
const std::array<std::array<int,4>,256>& grid_table() {
    static const std::array<std::array<int,4>,256> table=[]{
        std::array<std::array<int,4>,256> built{};
        for(uint32_t index=0;index<256;++index) built[index]=magnitudes_of(index);
        return built;
    }();
    return table;
}
uint32_t grid_pins=0;
void pin_grid() {
    std::vector<std::array<int,4>> seen(256);
    for(uint32_t index=0;index<256;++index) {
        const auto m=grid_table()[index];
        require(std::all_of(m.begin(),m.end(),[](int value){return value>=4 && value<=62;}),"an IQ3_XXS grid byte is outside the pinned magnitude range");
        seen[index]=m;++grid_pins;
    }
    uint32_t collisions=0;
    for(uint32_t a=0;a<256;++a) for(uint32_t b=a+1;b<256;++b) if(seen[a]==seen[b]) ++collisions;
    require(collisions==0,"IQ3_XXS grid entries are not distinguishable through the decoder");
}
uint32_t sign_rule_checks=0;
void sign_rule() {
    // The parity-fold rule the kernel uses, verified against the decoder for every 7-bit
    // field: value j of the lane is negative exactly when bits ^ ((popcount&1)<<7) has bit j.
    for(uint32_t bits=0;bits<128;++bits) {
        uint32_t parity=0;for(uint32_t bit=0;bit<7;++bit) if(bits&(1u<<bit)) ++parity;
        const uint32_t pattern=bits^((parity&1u)<<7);
        std::vector<uint8_t> bytes(xs_bytes,0);
        bytes[0]=0x00;bytes[1]=0x3c;bytes[2]=0x01;bytes[3]=0x02;   // lane 0 of group 0
        bytes[66]=uint8_t(bits);
        const auto weights=decode_row(bytes,super,0);
        const auto g1=grid_table()[1],g2=grid_table()[2];
        for(uint32_t j=0;j<4;++j) {
            require(std::abs(double(weights[j])-double(0.25f*g1[j]*((pattern&(1u<<j))?-1.0f:1.0f)))<1e-4,
                "the IQ3_XXS low sign parity rule does not match the decoder");
            require(std::abs(double(weights[j+4])-double(0.25f*g2[j]*((pattern&(1u<<(j+4)))?-1.0f:1.0f)))<1e-4,
                "the IQ3_XXS high sign parity rule does not match the decoder");
            sign_rule_checks+=2;
        }
    }
}
Row x_logical(uint32_t columns,uint32_t row) {
    const auto count=blocks(columns);
    Row w{columns,std::vector<uint8_t>(uint64_t(count)*grid_bytes),std::vector<uint32_t>(uint64_t(count)*groups),std::vector<float>(count)};
    for(uint32_t sb=0;sb<count;++sb) {
        w.d[sb]=float(int((sb*7+row*3+4)%11)-5)/8.f;
        for(uint32_t g=0;g<grid_bytes;++g) w.grid[uint64_t(sb)*grid_bytes+g]=uint8_t((g*37+sb*11+row*5)%256);
        for(uint32_t ib=0;ib<groups;++ib) {
            uint32_t word=uint32_t((ib*3+sb*5+row)%16)<<28;
            for(uint32_t lane=0;lane<4;++lane) word|=uint32_t((ib*13+lane*7+row)%128)<<(7*lane);
            w.word[uint64_t(sb)*groups+ib]=word;
        }
    }
    return w;
}
// The measured map: half d at 0, 64 grid bytes from 2 (byte 8*ib+2*lane covers values
// ib*32+lane*8+j and the next one values +4+j), and eight sign/scale words from 66.
std::vector<uint8_t> x_encode(const std::vector<Row>& rows) {
    const auto columns=rows.front().columns;
    std::vector<uint8_t> bytes(uint64_t(rows.size())*blocks(columns)*xs_bytes,0);
    const auto put=[&](size_t at,float value){const auto bits=strata::kernels::f16_from_f32(value);bytes[at]=uint8_t(bits);bytes[at+1]=uint8_t(bits>>8);};
    for(size_t row=0;row<rows.size();++row) {const auto& w=rows[row];
        for(uint32_t sb=0;sb<blocks(columns);++sb) {
            const size_t at=(uint64_t(row)*blocks(columns)+sb)*xs_bytes;
            for(uint32_t g=0;g<grid_bytes;++g) bytes[at+2+g]=w.grid[uint64_t(sb)*grid_bytes+g];
            for(uint32_t ib=0;ib<groups;++ib) {
                const uint32_t word=w.word[uint64_t(sb)*groups+ib];
                for(uint32_t b=0;b<4;++b) bytes[at+66+ib*4+b]=uint8_t(word>>(8*b));
            }
            put(at,w.d[sb]);
        }
    }
    return bytes;
}
uint32_t popcount7(uint32_t value){uint32_t count=0;for(uint32_t bit=0;bit<7;++bit) if(value&(1u<<bit)) ++count;return count;}
// Independent transcription of the pinned CUDA integer form.
Oracle x_oracle(const Row& w,const Q8Activation& q,uint32_t token) {
    Oracle out{0,0,0};
    for(uint32_t sb=0;sb<blocks(w.columns);++sb) for(uint32_t ib=0;ib<groups;++ib) {
        const uint32_t word=w.word[uint64_t(sb)*groups+ib];
        const int ls=int(word>>28);
        int sumi=0;
        for(uint32_t lane=0;lane<4;++lane) {
            const auto low=grid_table()[w.grid[uint64_t(sb)*grid_bytes+ib*8+lane*2]];
            const auto high=grid_table()[w.grid[uint64_t(sb)*grid_bytes+ib*8+lane*2+1]];
            const uint32_t bits=(word>>(7*lane))&127u;
            const uint32_t pattern=bits^((popcount7(bits)&1u)<<7);
            for(uint32_t j=0;j<4;++j) {
                const size_t value=uint64_t(token)*w.columns+sb*super+ib*32+lane*8+j;
                sumi+=((pattern&(1u<<j))?-low[j]:low[j])*signed_code(q.codes[value]);
                const size_t other=value+4;
                sumi+=((pattern&(1u<<(j+4)))?-high[j]:high[j])*signed_code(q.codes[other]);
            }
        }
        const int scaled=(ls*sumi+sumi/2)/2;                          // C truncation, as upstream
        const double factor=double(w.d[sb])*q.metadata[(uint64_t(token)*w.columns/32+sb*groups+ib)*2];
        out.value+=static_cast<long double>(factor)*scaled;
        out.magnitude+=std::abs(static_cast<long double>(factor)*scaled);
        out.bound+=std::abs(static_cast<long double>(factor));        // at most one integer unit per group
    }
    return out;
}
Oracle x_decoded(const std::vector<uint8_t>& bytes,uint32_t columns,uint32_t row,const Q8Activation& q,uint32_t token) {
    const auto weights=decode_row(bytes,columns,row);
    const size_t base=uint64_t(row)*packed_row_bytes(18,columns);
    Oracle out{0,0,0};
    for(uint32_t block=0;block<columns/32;++block) {
        const long double d8=q.metadata[(uint64_t(token)*columns/32+block)*2];long double part=0;
        for(uint32_t v=0;v<32;++v) part+=static_cast<long double>(weights[block*32+v])*signed_code(q.codes[uint64_t(token)*columns+block*32+v]);
        const long double term=d8*part;out.value+=term;out.magnitude+=std::abs(term);
        const size_t at=base+uint64_t(block/8)*xs_bytes;
        const float d=strata::kernels::f32_from_f16(uint16_t(bytes[at])|(uint16_t(bytes[at+1])<<8));
        out.bound+=std::abs(static_cast<long double>(d)*d8);
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
bool within_bound(float actual,const Oracle& decoder) {
    const auto difference=std::abs(static_cast<long double>(actual)-decoder.value);
    return std::isfinite(actual) && difference<=decoder.bound*1.02L+0.02L;
}
std::vector<float> source(uint64_t values,uint32_t seed) {
    std::vector<float> out(values);for(uint64_t i=0;i<values;++i) out[i]=float(int((i*23+seed)%127)-63)*.03125f+.0140625f;return out;
}
uint32_t cases_count=0;
void cases(uint32_t columns,uint32_t batch) {
    constexpr uint32_t rows=3;std::vector<Row> logical_rows;
    for(uint32_t row=0;row<rows;++row) logical_rows.push_back(x_logical(columns,row));
    const auto bytes=x_encode(logical_rows);
    validate_packed_rows(bytes,18,columns,rows);
    const auto values=source(uint64_t(batch)*columns,batch);
    const auto q=quantize_q8_activation(values,columns);
    const auto actual=tpc(bytes,18,columns,rows,q,batch,PackedActivation::Q8_1);
    const auto routed=tpc(bytes,18,columns,rows,q,batch,PackedActivation::RoutedQ8);
    for(uint32_t token=0;token<batch;++token) for(uint32_t row=0;row<rows;++row) {
        const auto index=uint64_t(token)*rows+row;
        require(close_exact(actual[index],x_oracle(logical_rows[row],q,token).value),
            "IQ3_XXS TPC differs from the independent CUDA integer-form oracle");
        require(within_bound(actual[index],x_decoded(bytes,columns,row,q,token)),
            "IQ3_XXS TPC differs from the pinned decoder by more than the derived truncation bound");
        require(routed[index]==actual[index],"IQ3_XXS dot differs between the routed and shared/head q8_1 contracts");
        std::vector<float> one(values.begin()+uint64_t(token)*columns,values.begin()+uint64_t(token)*columns+columns);
        const auto isolated=tpc(bytes,18,columns,rows,quantize_q8_activation(one,columns),1,PackedActivation::Q8_1);
        require(isolated[row]==actual[index],"batched IQ3_XXS changed single-row arithmetic or crossed tokens");
        ++cases_count;
    }
}
void counterexamples() {
    constexpr uint32_t columns=512;
    const auto w=x_logical(columns,0);const auto bytes=x_encode({w});
    const auto values=source(columns,7);const auto q=quantize_q8_activation(values,columns);
    const auto before=tpc(bytes,18,columns,1,q,1,PackedActivation::Q8_1)[0];
    require(close_exact(before,x_oracle(w,q,0).value) && within_bound(before,x_decoded(bytes,columns,0,q,0)),"IQ3_XXS baseline disagrees with its authorities");
    auto raw=q;for(uint32_t block=0;block<columns/32;++block) const_cast<Q8Activation&>(raw).metadata[block*2+1]=1234.5f;
    require(tpc(bytes,18,columns,1,raw,1,PackedActivation::Q8_1)[0]==before,"IQ3_XXS output moved when only raw sums changed");
    uint32_t moved=0;
    for(uint32_t v=0;v<columns && moved<4;++v) {
        if(!signed_code(q.codes[v]) || !q.metadata[(v/32)*2]) continue;
        ++moved;
        auto altered=w;altered.grid[(v/256)*grid_bytes+(v%256)/4]=uint8_t((altered.grid[(v/256)*grid_bytes+(v%256)/4]+1)%256);
        const auto altered_bytes=x_encode({altered});validate_packed_rows(altered_bytes,18,columns,1);
        const auto result=tpc(altered_bytes,18,columns,1,q,1,PackedActivation::Q8_1)[0];
        require(result!=before,"an IQ3_XXS grid index change was absorbed by the group scale");
        require(close_exact(result,x_oracle(altered,q,0).value) && within_bound(result,x_decoded(altered_bytes,columns,0,q,0)),
            "the changed IQ3_XXS index lost its oracle or decoder match");
    }
    require(moved==4,"too few observable IQ3_XXS lanes to pin a single grid index");
    // Clearing every sign field must leave all magnitudes positive and both authorities on.
    auto cleared=w;for(auto& value:cleared.word) value=value&0xf0000000u;
    const auto cleared_bytes=x_encode({cleared});validate_packed_rows(cleared_bytes,18,columns,1);
    require(cleared_bytes!=bytes && close_exact(tpc(cleared_bytes,18,columns,1,q,1,PackedActivation::Q8_1)[0],x_oracle(cleared,q,0).value),
        "clearing the IQ3_XXS sign fields changed nothing or lost the oracle");
    // The 4-bit scale is the only magnitude multiplier per group: flattening it moves the dot.
    auto minimal=w;for(auto& value:minimal.word) value=value&0x0fffffffu;
    const auto minimal_bytes=x_encode({minimal});validate_packed_rows(minimal_bytes,18,columns,1);
    require(tpc(minimal_bytes,18,columns,1,q,1,PackedActivation::Q8_1)[0]!=before,"flattening the IQ3_XXS group scales did not change the dot");
    auto negated=w;for(auto& value:negated.d) value=-value;
    const auto neg_bytes=x_encode({negated});validate_packed_rows(neg_bytes,18,columns,1);
    require(tpc(neg_bytes,18,columns,1,q,1,PackedActivation::Q8_1)[0]==-before,"negating the IQ3_XXS super-block d did not negate the dot");
    // Every group owns its own word: nulling group 0 must leave the other seven intact.
    auto muted=bytes;for(uint32_t b=0;b<4;++b) muted[66+b]=0;
    validate_packed_rows(muted,18,columns,1);
    auto muted_row=w;muted_row.word[0]=0;
    const auto result=tpc(muted,18,columns,1,q,1,PackedActivation::Q8_1)[0];
    require(close_exact(result,x_oracle(muted_row,q,0).value) && within_bound(result,x_decoded(muted,columns,0,q,0)),
        "muting one IQ3_XXS sign/scale word did not remove exactly that group");
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
    bytes[0]=0;bytes[1]=0x7c;rejects([&]{validate_packed_rows(bytes,18,256,1);});
    bytes[0]=0;bytes[1]=0x3c;validate_packed_rows(bytes,18,256,1);
    for(uint32_t at:{2,66,67,97}) { bytes[at]=0xff; } validate_packed_rows(bytes,18,256,1);
    rejects([&]{(void)packed_row_bytes(18,640);});rejects([&]{(void)packed_row_bytes(18,300);});
    require(packed_supported(18) && packed_values_per_block(18)==super && packed_block_bytes(18)==xs_bytes
        && packed_row_bytes(18,2560)==980 && packed_row_bytes(18,256)==xs_bytes,"IQ3_XXS geometry wrong");
    require(packed_activation_supported(18,PackedActivation::Q8_1) && packed_activation_supported(18,PackedActivation::RoutedQ8)
        && !packed_activation_supported(18,PackedActivation::RoutedS2),"IQ3_XXS policy wrong");
    const auto routed=expert_format(2560,640,0,{18,18,8});
    require(routed.packed && routed.activation==PackedActivation::RoutedQ8
        && routed.bytes==uint64_t(980)*640*2+uint64_t(34)*20*2560,"routed IQ3_XXS expert charge wrong");
    const auto shared=expert_format(2560,640,-1,{18,16,8});
    require(shared.packed && shared.activation==PackedActivation::Q8_1
        && shared.bytes==uint64_t(980)*640+uint64_t(660)*640+uint64_t(34)*20*2560,"shared IQ3_XXS expert was not packed or is mischarged");
    const auto down=expert_format(2560,640,0,{8,8,18});
    require(!down.packed && down.bytes==uint64_t(2560)*640*6,"IQ3_XXS down projection with 640 columns was not rejected by layout");
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
    r.bytes=x_encode(r.xs);validate_packed_rows(r.bytes,18,columns,rows);return r;
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
        out[uint64_t(token)*r.rows+row]=float(x_oracle(r.xs[row],q,token).value);
    }
    return out;
}
uint32_t flows=0;
void shared_flows() {
    constexpr uint32_t hidden=512,intermediate=256;
    for(auto types:{std::array<uint32_t,3>{18,18,18},std::array<uint32_t,3>{18,18,8},std::array<uint32_t,3>{18,8,18},std::array<uint32_t,3>{8,18,18}})
    for(uint32_t batch:{1u,3u}) {
        const auto format=expert_format(hidden,intermediate,-1,types);
        require(format.packed && format.activation==PackedActivation::Q8_1,"shared IQ3_XXS expert lost the head contract");
        std::array<Flow,3> role{flow_role(types[0],hidden,intermediate,0),flow_role(types[1],hidden,intermediate,1),flow_role(types[2],intermediate,hidden,2)};
        const uint64_t charge=packed_row_bytes(types[0],hidden)*intermediate+packed_row_bytes(types[1],hidden)*intermediate
            +packed_row_bytes(types[2],intermediate)*hidden;
        require(format.bytes==charge,"shared IQ3_XXS expert payload mischarged");
        std::vector<float> input(uint64_t(batch)*hidden);
        for(size_t i=0;i<input.size();++i) input[i]=float(int((i*29+batch)%113)-56)*.0625f;
        std::array<uint32_t,3> calls{0,0,0};
        auto run=[&](const std::array<Flow,3>& roles)->std::vector<float> {
            calls={0,0,0};
            return execute_packed_expert(std::span<const float>(input),hidden,intermediate,[&](uint32_t which,std::span<const float> values) {
                ++calls[which];
                const auto mode=expert_projection_activation(format,which);
                require(mode==PackedActivation::Q8_1,"shared IQ3_XXS projection inherited a routed activation contract");
                const auto tokens=uint32_t(values.size()/roles[which].columns);
                const auto q=quantize_q8_activation(values,roles[which].columns);
                const auto expected=flow_oracle(roles[which],q,tokens);
                const auto actual=tpc(roles[which].bytes,roles[which].type,roles[which].columns,roles[which].rows,q,tokens,mode);
                for(size_t i=0;i<actual.size();++i) {
                    require(std::isfinite(actual[i]),"non-finite shared IQ3_XXS projection");
                    require(std::abs(double(actual[i])-double(expected[i]))<=std::max(.02,double(std::abs(expected[i]))*2e-3),
                        "shared expert IQ3_XXS projection differs from the independent oracle");
                }
                return actual;
            });
        };
        const auto output=run(role);
        require(calls==std::array<uint32_t,3>{1,1,1},"shared IQ3_XXS expert flow did not run each projection once");
        require(output.size()==input.size(),"shared IQ3_XXS expert flow output shape");
        for(float value:output) require(std::isfinite(value),"non-finite shared IQ3_XXS expert output");
        if(types[0]==types[1] && types[0]==18) {
            auto swapped=role;std::swap(swapped[0],swapped[1]);
            const auto other=run(swapped);bool differs=false;
            for(size_t i=0;i<output.size();++i) differs=differs||output[i]!=other[i];
            require(differs,"exchanging gate and up IQ3_XXS weights left the shared expert output unchanged");
        }
        ++flows;
    }
}
}
int main(){try {
    pin_grid();sign_rule();
    for(uint32_t columns:{256u,512u,2560u}) for(uint32_t batch:{1u,3u,8u}) cases(columns,batch);
    counterexamples();guards();shared_flows();
    std::cout<<"PASS IQ3_XXS TPC "<<cases_count<<" cases exact against the pinned CUDA integer form ((sc*sumi+sumi/2)/2), within the derived one-unit truncation bound of the unmodified pinned ggml decoder, "
        <<grid_pins<<" grid magnitudes recovered through the decoder and shown distinguishable, "<<sign_rule_checks<<" sign-parity checks over all 128 fields, "
        <<flows<<" production shared-expert flows, raw-sum insensitivity and policy guards; no device\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
