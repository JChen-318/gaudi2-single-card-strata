// Explicit single-process hardware diagnostic for wide resident prefill.
#include "strata/hpu/model.hpp"
#include "strata/hpu/ple_io_mode.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
using namespace strata::hpu;
using Clock=std::chrono::steady_clock;
namespace {
std::vector<std::vector<int32_t>> read_cases(const std::filesystem::path& path) {
    std::ifstream in(path);if(!in) throw std::runtime_error("missing cases");
    std::vector<std::vector<int32_t>> cases;std::string line;
    while(std::getline(in,line)) {std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);int32_t id;std::vector<int32_t> ids;while(row>>id) ids.push_back(id);if(ids.empty()) throw std::runtime_error("empty case");cases.push_back(std::move(ids));}
    return cases;
}
void exact(std::span<const float> a,std::span<const float> b) {
    if(a.size()!=b.size() || std::memcmp(a.data(),b.data(),a.size_bytes())) throw std::runtime_error("state replay differs");
}
std::vector<float> prefill(Model& model,std::span<const int32_t> ids,uint32_t chunk) {
    std::vector<float> logits;
    for(size_t begin=0;begin<ids.size();begin+=chunk) {
        const auto count=uint32_t(std::min<size_t>(chunk,ids.size()-begin));uint32_t calls=0;
        model.prefill_window(ids.subspan(begin,count),[&](uint32_t committed,std::span<const float> out) {
            if(committed!=count || ++calls!=1) throw std::runtime_error("wide prefill silently split a batch");
            logits.assign(out.begin(),out.end());
        });
        if(calls!=1) throw std::runtime_error("wide batch failed to publish");
    }
    return logits;
}
double elapsed(Clock::time_point begin) {return std::chrono::duration<double,std::milli>(Clock::now()-begin).count();}
}
int main(int argc,char** argv) {
    if(argc!=6) return 2;
    try {
        ModelOptions o;o.fp8_experts=true;o.memory.bf16_head=true;o.memory.context=4096;
        o.memory.expert_cache=2048ull<<20;o.memory.host_limit=49152ull<<20;o.memory.prefill_rows=128;
        o.ple_io=resolve_ple_io("ram",o.ple_io);Model model(argv[1],o);
        if(model.plan().prefill_rows!=128 || model.plan().window_rows!=128) throw std::runtime_error("wide plan not admitted");
        std::cout<<"READY wide resident rows=128 hbm_required="<<model.plan().hbm_required<<" host_required="<<model.plan().host_required<<'\n'<<std::flush;
        const auto cases=read_cases(argv[2]),bench=read_cases(argv[5]);
        if(cases.size()!=5 || bench.size()!=5) throw std::runtime_error("incomplete diagnostic matrix");
        const std::filesystem::path root(argv[3]),reference(argv[4]);
        for(uint32_t chunk:{8u,32u,64u,128u}) {
            const auto dir=root/("r"+std::to_string(chunk));std::filesystem::create_directories(dir);
            for(uint32_t c=0;c<cases.size();++c) {
                model.reset();const auto start=Clock::now();auto logits=prefill(model,cases[c],chunk);const auto ms=elapsed(start);
                std::ofstream out(dir/(std::to_string(c)+".f32"),std::ios::binary);
                auto write=[&](const std::vector<float>& row) {out.write(reinterpret_cast<const char*>(row.data()),std::streamsize(row.size()*4));};write(logits);
                std::ifstream suffix(reference/(std::to_string(c)+".suffix.txt"));int32_t token;uint32_t count=0;
                while(suffix>>token) {write(model.decode(token));++count;}
                if(count!=16 || !out) throw std::runtime_error("missing continuation/logits");
                std::cout<<"QUALITY rows="<<chunk<<" case="<<c<<" prompt="<<cases[c].size()<<" prompt_ms="<<ms<<" logits_rows=17\n"<<std::flush;
            }
            // Warmup separate; reset gives no prefix reuse. Both passes use distinct inputs.
            for(uint32_t b=0;b<bench.size();++b) {
                model.reset();const auto start=Clock::now();auto logits=prefill(model,bench[b],chunk);const auto ms=elapsed(start);
                const auto decode_start=Clock::now();
                for(uint32_t i=0;i<32;++i) {const auto token=int32_t(std::max_element(logits.begin(),logits.end())-logits.begin());logits=model.decode(token);}
                std::cout<<"BENCH rows="<<chunk<<" item="<<b<<" prompt="<<bench[b].size()<<" prompt_ms="<<ms<<" decode_ms="<<elapsed(decode_start)<<" reused=0\n"<<std::flush;
            }
        }
        model.reset();(void)prefill(model,std::span<const int32_t>(cases[0]).first(129),128);
        if(model.size()!=129) throw std::runtime_error("128+1 tail failed");
        const auto point=model.snapshot();const auto before=model.logits();const auto window=std::span<const int32_t>(cases[0]).subspan(129,128);
        uint32_t total=0;model.prefill_window(window,[](uint32_t,std::span<const float>){},[&]{++total;return false;});model.restore(point);
        bool stopped=false;uint32_t polls=0;
        try {model.prefill_window(window,[](uint32_t,std::span<const float>){throw std::runtime_error("STOP published rows");},[&]{return ++polls==total;});}
        catch(const Cancelled&) {stopped=true;}
        if(!stopped || model.size()!=129 || !model.healthy()) throw std::runtime_error("late STOP accepted/poisoned batch");
        exact(model.logits(),before);
        const auto different=std::span<const int32_t>(cases[0]).subspan(130,128);const auto retry=prefill(model,different,128);model.restore(point);
        exact(prefill(model,different,128),retry);model.restore(point);
        for(uint32_t tail:{1u,7u,31u,63u,127u}) {
            const auto expected=prefill(model,std::span<const int32_t>(cases[0]).subspan(129,tail),128);model.restore(point);
            exact(prefill(model,std::span<const int32_t>(cases[0]).subspan(129,tail),128),expected);model.restore(point);
        }
        const auto save=(root/"state.strata").string();model.save(save);const auto next=model.decode(cases[0][0]);model.load(save);exact(model.decode(cases[0][0]),next);std::filesystem::remove(save);
        std::cout<<"PASS late_stop=exact different_retry=exact tails=1,7,31,63,127 disk_restore=exact\n"<<std::flush;
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
