#include "strata/hpu/moe.hpp"
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
using namespace strata::hpu;
namespace {
constexpr uint32_t H=64,FF=32,E=4;
float round_bf16(float x) {return from_bf16(to_bf16(x));}
void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
template<class F> void rejects(F&& f) {bool rejected=false;try {f();} catch(const std::exception&) {rejected=true;}require(rejected,"invalid expert/cache operation accepted");}
template<class T> void write(std::ofstream& f,T value) {f.write(reinterpret_cast<const char*>(&value),sizeof(value));}
void string(std::ofstream& f,const std::string& value) {write<uint64_t>(f,value.size());f.write(value.data(),value.size());}
struct Fixture {
    std::filesystem::path path=std::filesystem::temp_directory_path()/("strata-hpu-moe-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".gguf");
    std::array<std::vector<float>,3> weights;
    Fixture() {
        for(uint32_t role=0;role<3;++role) {
            weights[role].resize(E*H*FF);
            for(size_t i=0;i<weights[role].size();++i) weights[role][i]=float(int((i*13+role*7+i/(H*FF)*11)%23)-11)/128;
        }
        std::ofstream f(path,std::ios::binary);write<uint32_t>(f,0x46554747);write<uint32_t>(f,3);write<uint64_t>(f,3);write<uint64_t>(f,0);
        for(uint32_t role=0;role<3;++role) {
            string(f,role==0?"gate":role==1?"up":"down");write<uint32_t>(f,3);
            write<uint64_t>(f,role==2?FF:H);write<uint64_t>(f,role==2?H:FF);write<uint64_t>(f,E);write<uint32_t>(f,30);
            write<uint64_t>(f,uint64_t(role)*E*H*FF*2);
        }
        while(uint64_t(f.tellp())%32) f.put(0);
        for(const auto& tensor:weights) for(float value:tensor) write<uint16_t>(f,to_bf16(value));
        require(bool(f),"cannot write MoE fixture");
    }
    ~Fixture() {std::error_code error;std::filesystem::remove(path,error);}
    std::vector<float> expert(std::span<const float> x,uint32_t expert) const {
        std::vector<float> y(H),activation(FF);
        for(uint32_t row=0;row<FF;++row) {
            double gate=0,up=0;
            for(uint32_t col=0;col<H;++col) {
                gate+=double(round_bf16(x[col]))*weights[0][expert*H*FF+row*H+col];
                up+=double(round_bf16(x[col]))*weights[1][expert*H*FF+row*H+col];
            }
            const auto g=round_bf16(float(gate)),u=round_bf16(float(up));
            const auto sig=round_bf16(1.0f/(1.0f+std::exp(-g)));activation[row]=round_bf16(round_bf16(g*sig)*u);
        }
        for(uint32_t row=0;row<H;++row) {
            double dot=0;for(uint32_t col=0;col<FF;++col) dot+=double(activation[col])*weights[2][expert*H*FF+row*FF+col];y[row]=round_bf16(float(dot));
        }
        return y;
    }
};
void compare(const std::vector<float>& actual,const std::vector<float>& expected,uint32_t rows) {
    double error=0,signal=0,peak=0;
    require(actual.size()==expected.size(),"MoE output size mismatch");
    for(size_t i=0;i<actual.size();++i) {
        require(std::isfinite(actual[i]),"non-finite MoE result");const double delta=double(actual[i])-expected[i];
        error+=delta*delta;signal+=double(expected[i])*expected[i];peak=std::max(peak,std::abs(delta));
    }
    const auto relative=std::sqrt(error/std::max(signal,1e-24));require(peak<0.002 && relative<0.03,"MoE complete path parity failed");
    std::cout<<"PASS real HPU native router/grouped cached experts/shared/combine T="<<rows<<" max_abs_error="<<peak<<" relative_l2="<<relative<<'\n';
}
}
int main() {
    try {
        Fixture fixture;WeightSource source(fixture.path.string());Device device;
        strata::core::ModelGeometry g;g.n_embd=H;g.n_ff=FF;g.n_expert=E;g.n_expert_used=2;g.n_layers=1;
        ExpertCache cache(device,g,[source](uint32_t layer,int32_t expert) {
            if(layer) throw std::logic_error("fixture layer");
            const uint32_t id=expert==-1?0:uint32_t(expert);
            return ExpertSlices{source.expert("gate",id),source.expert("up",id),source.expert("down",id)};
        },6*H*FF,4*H); // Exactly one resident expert, only one row per pinned slot.
        {
            auto first=cache.acquire(0,0),hit=cache.acquire(0,0);require(first.generation()==hit.generation(),"cache hit changed generation");
            rejects([&]{(void)cache.acquire(0,1);});require(cache.stats().hits==1 && cache.stats().loads==1,"cache hit/busy counters");
        }
        Stream copy(device);HostBuffer hr(device,E*H*2),hs(device,H*2);DeviceBuffer dr(device,hr.size()),ds(device,hs.size());
        auto* router=static_cast<uint16_t*>(hr.data());auto* shared=static_cast<uint16_t*>(hs.data());
        std::fill_n(router,E*H,0);std::fill_n(shared,H,0);
        for(uint32_t e=0;e<E;++e) router[e*H+e]=to_bf16(float(e+1)/2);
        shared[7]=to_bf16(1);
        upload(copy,dr,hr,hr.size());upload(copy,ds,hs,hs.size());Event ready(device);ready.record(copy);ready.synchronize();
        MoeBlock block(device,g,0,cache,{"w_router",&dr},{"w_shared_gate",&ds});
        for(uint32_t rows:{1,5,8,11}) {
            std::vector<float> x(rows*H),logits(rows*E),expected(rows*H);
            for(uint32_t row=0;row<rows;++row) for(uint32_t d=0;d<H;++d) x[row*H+d]=row==0?0:float(int((row*11+d*7)%29)-14)/32;
            for(uint32_t row=0;row<rows;++row) for(uint32_t e=0;e<E;++e) logits[row*E+e]=round_bf16(x[row*H+e])*float(e+1)/2;
            const auto routes=route_reference(logits,rows,E,2);
            for(uint32_t row=0;row<rows;++row) {
                const auto activation=std::span<const float>(x).subspan(row*H,H);
                for(uint32_t rank=0;rank<2;++rank) {
                    const auto index=row*2+rank;const auto result=fixture.expert(activation,routes.ids[index]);
                    for(uint32_t d=0;d<H;++d) expected[row*H+d]+=result[d]*routes.weights[index];
                }
                const auto result=fixture.expert(activation,0);const float gate=float(1/(1+std::exp(-double(round_bf16(x[row*H+7])))));
                for(uint32_t d=0;d<H;++d) expected[row*H+d]+=result[d]*gate;
            }
            compare(block.run(x),expected,rows);
            const auto stats=cache.stats();require(stats.resident_bytes==6*H*FF && stats.peak_bytes==6*H*FF,"single expert HBM payload budget violated");
        }
        const auto stats=cache.stats();require(stats.evictions>0,"test failed to exercise expert eviction");
        rejects([&]{(void)cache.acquire(1,0);});rejects([&]{(void)cache.acquire(0,E);});
        std::cout<<"PASS real HPU expert cache resident_bytes="<<stats.resident_bytes<<" loads="<<stats.loads<<" evictions="<<stats.evictions<<" hits="<<stats.hits<<'\n';
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
