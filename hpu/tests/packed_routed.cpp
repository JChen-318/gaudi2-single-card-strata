// Independent logical weights and integer/long-double oracle against actual
// production TPC C. Host only; no ISA, device reduction or performance claim.
#include "strata/hpu/packed_quant.hpp"
#include "strata/hpu/expert_format.hpp"
#include "strata/hpu/packed_expert_flow.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "packed_emulator.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
using namespace strata::hpu;
namespace {
void require(bool ok,const char* why){if(!ok) throw std::runtime_error(why);}
template<class F> void rejects(F f){bool bad=false;try{f();}catch(const std::exception&){bad=true;}require(bad,"invalid routed policy accepted");}
int signed_code(uint8_t b){return b<128?b:int(b)-256;}
struct Weight {
    uint32_t type,columns,rows;
    std::vector<uint8_t> bytes;
    std::vector<int> logical;
    std::vector<float> delta,minimum;
};
Weight encode(uint32_t type,uint32_t columns,uint32_t rows) {
    const uint32_t block=type==42?64:32,size=packed_block_bytes(type),blocks=columns/block;
    Weight w{type,columns,rows,std::vector<uint8_t>(uint64_t(rows)*blocks*size),std::vector<int>(uint64_t(rows)*columns),std::vector<float>(rows*blocks),std::vector<float>(rows*blocks)};
    auto half=[&](size_t at,float f){auto b=strata::kernels::f16_from_f32(f);w.bytes[at]=uint8_t(b);w.bytes[at+1]=uint8_t(b>>8);};
    for(uint32_t row=0;row<rows;++row) for(uint32_t b=0;b<blocks;++b) {
        const size_t at=(uint64_t(row)*blocks+b)*size;const bool affine=type==3 || type==7,high=type==6 || type==7;
        w.delta[row*blocks+b]=float(int((row+b)%7)-3)/128;w.minimum[row*blocks+b]=affine?float(int((row+b)%3)-1)/1024:0;
        half(at,w.delta[row*blocks+b]);if(affine) half(at+2,w.minimum[row*blocks+b]);
        const auto qh=at+(affine?4:2),qs=qh+(high?4:0);
        for(uint32_t j=0;j<block;++j) {
            const int code=type==8?int((j*37+row*11+b*3)%255)-127:type==42?int((j+row+b)%4):int((j*7+row*3+b)%uint32_t(high?32:16));
            w.logical[uint64_t(row)*columns+b*block+j]=code;
            if(type==8) w.bytes[at+2+j]=uint8_t(code);
            else if(type==42) w.bytes[at+2+j/4]|=uint8_t(code<<((j%4)*2));
            else {w.bytes[qs+j%16]|=uint8_t((code&15)<<(j<16?0:4));if(high && code>=16) w.bytes[qh+j/8]|=uint8_t(1u<<(j%8));}
        }
    }
    validate_packed_rows(w.bytes,type,columns,rows);return w;
}
std::vector<float> project(Weight& w,Q8Activation& q,PackedActivation mode) {
    const auto batch=uint32_t(q.codes.size()/w.columns);std::vector<float> result(uint64_t(batch)*w.rows,NAN);
    using namespace packed_emulator;
    tensor weights{w.bytes.data(),{packed_row_bytes(w.type,w.columns),w.rows},1},a{q.codes.data(),{w.columns,batch},1},m{q.metadata.data(),{2,w.columns/32,batch},4},out{result.data(),{w.rows,batch},4};
    // Separate row/token index spaces exercise partial launches and sentinels.
    for(uint32_t t=0;t<batch;++t) for(uint32_t r=0;r<w.rows;++r) {
        offset={int(r),int(t),0,0,0};extent={1,1,1,1,1};packed_kernel(weights,a,m,out,w.columns,w.rows,batch,w.type,int(mode));
    }
    return result;
}
std::vector<float> independent(const Weight& w,const Q8Activation& q,PackedActivation mode) {
    const uint32_t batch=uint32_t(q.codes.size()/w.columns),block=w.type==42?64:32,blocks=w.columns/block;
    std::vector<float> out(uint64_t(batch)*w.rows);
    for(uint32_t t=0;t<batch;++t) for(uint32_t r=0;r<w.rows;++r) {
        long double total=0;
        for(uint32_t b=0;b<blocks;++b) for(uint32_t sub=0;sub<block/32;++sub) {
            int dot=0,sum=0;const auto metadata=(uint64_t(t)*(w.columns/32)+b*(block/32)+sub)*2;
            for(uint32_t j=0;j<32;++j) {
                const int a=signed_code(q.codes[uint64_t(t)*w.columns+b*block+sub*32+j]);
                int code=w.logical[uint64_t(r)*w.columns+b*block+sub*32+j];
                if(w.type==2 && mode==PackedActivation::RoutedQ8) code-=8;
                if(w.type==42) code-=1;
                dot+=code*a;sum+=a;
            }
            const long double d=w.delta[r*blocks+b],m=w.minimum[r*blocks+b],scale=q.metadata[metadata],raw=q.metadata[metadata+1];
            total+=d*scale*dot+m*scale*sum;
            if(w.type==6) total-=d*16*raw;
            if(w.type==2 && mode==PackedActivation::Q8_1) total-=d*8*raw;
        }
        out[uint64_t(t)*w.rows+r]=float(total);
    }
    return out;
}
void compare(std::span<const float> actual,std::span<const float> expected) {
    require(actual.size()==expected.size(),"routed result shape");
    for(size_t i=0;i<actual.size();++i) require(std::isfinite(actual[i]) && std::abs(actual[i]-expected[i])<=std::max(.001f,std::abs(expected[i])*.00005f),"routed TPC differs from independent logical-code/correction oracle");
}
void dots(uint32_t type,uint32_t columns,uint32_t batch) {
    auto w=encode(type,columns,3);std::vector<float> input(uint64_t(columns)*batch);
    for(size_t i=0;i<input.size();++i) input[i]=float(int((i*19+batch)%127)-63)*.03125f+.0140625f;
    auto q=quantize_q8_activation(input,columns);compare(project(w,q,PackedActivation::RoutedQ8),independent(w,q,PackedActivation::RoutedQ8));
}
void correction_roles() {
    auto w=encode(2,32,1);std::fill(w.bytes.begin()+2,w.bytes.end(),uint8_t(0x88));std::fill(w.logical.begin(),w.logical.end(),8);
    w.bytes[0]=0;w.bytes[1]=0x3c;w.delta[0]=1;
    Q8Activation q;q.codes.assign(32,1);q.metadata={.5f,5};
    require(project(w,q,PackedActivation::RoutedQ8)[0]==0,"routed Q4 used raw-input correction");
    require(project(w,q,PackedActivation::Q8_1)[0]==88,"shared/head Q4 lost raw-input correction");
    q.metadata[1]=123;require(project(w,q,PackedActivation::RoutedQ8)[0]==0,"routed centered dot depends on unused raw sum");
    require(project(w,q,PackedActivation::Q8_1)[0]==-856,"shared raw sum perturbation was ignored");
    rejects([]{(void)expert_projection_activation(expert_format(128,64,0,{2,23,8}),0);});
    rejects([]{(void)expert_projection_activation(expert_format(128,64,0,{2,3,8}),3);});
    require(!packed_activation_supported(42,PackedActivation::RoutedQ8) && !packed_activation_supported(2,PackedActivation::RoutedS2) && packed_activation_supported(23,PackedActivation::Q8_1) && !packed_activation_supported(19,PackedActivation::Q8_1) && !packed_activation_supported(2,static_cast<PackedActivation>(3)),"invalid graph policy admitted");
}
void policies() {
    constexpr std::array<uint32_t,6> types{2,3,6,7,8,42};uint32_t count=0;
    for(auto gate:types) for(auto up:types) for(auto down:types) {
        const std::array<uint32_t,3> source{gate,up,down};const auto routed=expert_format(128,64,0,source),shared=expert_format(128,64,-1,source);
        const auto bytes=64*(packed_row_bytes(gate,128)+packed_row_bytes(up,128))+128*packed_row_bytes(down,64);
        require(routed.packed && shared.packed && routed.bytes==bytes && shared.bytes==bytes,"mixed source payload is decoded/undercharged");
        for(size_t role=0;role<3;++role) {
            require(expert_projection_activation(routed,role)==(source[role]==42?PackedActivation::RoutedS2:PackedActivation::RoutedQ8),"mixed routed projection quantizer policy");
            require(expert_projection_activation(shared,role)==PackedActivation::Q8_1,"shared source inherited routed policy");
        }
        ++count;
    }
    require(count==216,"policy Cartesian coverage");
    // Independent persistent slots: activation U8 + metadata F32 + logits F32,
    // two H->FF and one FF->H, all four buckets and three contexts/layer.
    const uint64_t row_slots=2*(128+128/32*8+64*4)+(64+64/32*8+128*4);
    const auto quantizer_inputs=4*(2*128+64)*169;
    const auto codebooks=packed_codebook_bytes()*3*3*4;
    require(packed_expert_mutable_bytes(128,64,49)==(row_slots*169*3+quantizer_inputs+codebooks)*49,"target+MTP mixed context reserve omitted slots");
}
void flows() {
    constexpr uint32_t h=128,ff=64;uint32_t cases=0;
    for(auto types:{std::array<uint32_t,3>{2,2,2},{3,6,7},{8,8,8},{42,2,8},{3,42,7},{6,8,42},{42,42,42},{42,3,42},{7,42,42}}) for(uint32_t batch:{1u,3u}) {
        const auto format=expert_format(h,ff,0,types);
        std::array<Weight,3> weights{encode(types[0],h,ff),encode(types[1],h,ff),encode(types[2],ff,h)};
        std::vector<float> input(batch*h);for(size_t i=0;i<input.size();++i) input[i]=float(int((i*17)%63)-31)/32;
        auto run=[&](std::span<const float> rows,bool oracle) {
            return execute_packed_expert(rows,h,ff,[&](uint32_t role,auto values) {
                const auto mode=expert_projection_activation(format,role);auto& w=weights[role];
                auto q=mode==PackedActivation::RoutedS2?quantize_s2_activation(values,w.columns):quantize_q8_activation(values,w.columns);
                return oracle?independent(w,q,mode):project(w,q,mode);
            });
        };
        const auto actual=run(input,false);compare(actual,run(input,true));
        for(uint32_t t=0;t<batch;++t) {const auto single=run(std::span<const float>(input).subspan(t*h,h),false);require(!std::memcmp(single.data(),actual.data()+t*h,h*4),"mixed routed expert crossed token rows");}
        ++cases;
    }
    require(cases==18,"mixed flow coverage");
}
}
int main(){try{uint32_t count=0;for(uint32_t type:{2u,3u,6u,7u,8u}) for(uint32_t columns:{32u,64u,128u,640u,2560u}) for(uint32_t batch:{1u,3u,7u,8u,32u,128u}) {dots(type,columns,batch);++count;}correction_roles();policies();flows();std::cout<<"PASS routed TPC "<<count<<" logical-code cases, 216 mixed projection policies, 18 independent expert flows, centered-vs-raw Q4 counterexamples, singleton rows and three-context budgets; no device\n";}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
