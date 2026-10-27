// Explicit hardware diagnostic, never registered as a CTest. Writes complete
// vocabulary rows at identical teacher-forced prefixes for cross-build A/B.
#include "strata/hpu/model.hpp"
#include "strata/hpu/performance_modes.hpp"
#include "strata/hpu/prefill_schedule.hpp"
#include "strata/hpu/ple_io_mode.hpp"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
using namespace strata::hpu;
int main(int argc,char** argv) {
    if(argc!=5) {std::cerr<<"model.gguf cases.txt output-directory prefill-rows\n";return 2;}
    std::unique_ptr<Model> observed;
    try {
        ModelOptions o;o.fp8_experts=true;o.memory.bf16_head=true;o.memory.context=4096;
        o.memory.expert_cache=2048ull<<20;o.memory.host_limit=49152ull<<20;
        o.memory.prefill_rows=uint32_t(std::stoul(argv[4]));o.ple_io=resolve_ple_io("ram",o.ple_io);
        observed=std::make_unique<Model>(argv[1],o);Model& model=*observed;std::cout<<"READY real-model quality probe\n"<<std::flush;std::ifstream cases(argv[2]);
        if(!cases) throw std::runtime_error("cannot read cases");
        std::filesystem::create_directories(argv[3]);std::string line;uint32_t case_id=0;
        while(std::getline(cases,line)) {
            std::replace(line.begin(),line.end(),',',' ');std::istringstream input(line);
            std::vector<int32_t> ids;int32_t id;while(input>>id) ids.push_back(id);
            if(ids.empty() || ids.size()+32>o.memory.context) throw std::runtime_error("invalid case length");
            model.reset();std::vector<float> logits;uint32_t notified=0;
            model.prefill_window(ids,[&](uint32_t done,std::span<const float> x) {
                notified=done;logits.assign(x.begin(),x.end());
            });
            if(notified!=ids.size() || logits.size()!=model.vocabulary()) throw std::runtime_error("lost prefill prefix");
            std::ofstream out(std::filesystem::path(argv[3])/(std::to_string(case_id)+".f32"),std::ios::binary);
            const auto write=[&](std::span<const float> row) {out.write(reinterpret_cast<const char*>(row.data()),std::streamsize(row.size_bytes()));};
            write(logits);const auto point=model.snapshot();
            if(o.memory.prefill_rows>=2) {
                const auto count=o.memory.prefill_rows;
                uint32_t polls=0;bool stopped=false;
                try {
                    model.prefill_window(std::span<const int32_t>(ids).first(count),
                        [](uint32_t,std::span<const float>){throw std::runtime_error("cancelled window published rows");},
                        [&]{return ++polls==2;}); // after all 48 layers ran, before commit
                } catch(const Cancelled&) {stopped=true;}
                if(!stopped || model.size()!=ids.size() || model.logits()!=logits || !model.healthy())
                    throw std::runtime_error("STOP exposed an uncommitted resident window");
                std::vector<float> retry;
                const auto different=std::span<const int32_t>(ids).subspan(7,count);
                model.prefill_window(different,[&](uint32_t,std::span<const float> x){retry.assign(x.begin(),x.end());});
                model.restore(point);
                model.prefill_window(different,[&](uint32_t,std::span<const float> x){
                    if(x.size()!=retry.size() || std::memcmp(x.data(),retry.data(),x.size_bytes()))
                        throw std::runtime_error("STOP retry differs from clean restored window");
                });
                model.restore(point);
                // A complete native window followed by a scalar tail must equal the
                // same committed window followed by Model::decode of that tail.
                std::vector<float> with_tail;
                model.prefill_window(std::span<const int32_t>(ids).first(count+1),[&](uint32_t,std::span<const float> x){with_tail.assign(x.begin(),x.end());});
                model.restore(point);
                model.prefill_window(std::span<const int32_t>(ids).first(count),[](uint32_t,std::span<const float>){});
                const auto split_tail=model.decode(ids[count]);
                if(split_tail.size()!=with_tail.size() || std::memcmp(split_tail.data(),with_tail.data(),split_tail.size()*4))
                    throw std::runtime_error("native window + scalar tail differs from explicit split");
                model.restore(point);
            }
            std::vector<std::vector<float>> suffix;
            for(uint32_t j=0;j<16;++j) {suffix.push_back(model.decode(ids[(j*17)%ids.size()]));write(suffix.back());}
            // Restore and overwrite the abandoned branch with different cells,
            // then restore once more and replay. Catches a stale HBM mirror.
            model.restore(point);for(uint32_t j=0;j<3;++j) (void)model.decode(ids[(j*31+7)%ids.size()]);
            model.restore(point);
            for(uint32_t j=0;j<16;++j) {
                const auto replay=model.decode(ids[(j*17)%ids.size()]);
                if(replay.size()!=suffix[j].size() || std::memcmp(replay.data(),suffix[j].data(),replay.size()*4))
                    throw std::runtime_error("restore/branch replay differs");
            }
            // Disk save/load follows the same invalidation law.
            const auto path=(std::filesystem::path(argv[3])/"state.strata").string();model.save(path);
            const auto next=model.decode(ids[0]);model.load(path);const auto loaded=model.decode(ids[0]);
            if(next.size()!=loaded.size() || std::memcmp(next.data(),loaded.data(),next.size()*4))
                throw std::runtime_error("SAVE/RESTORE differs");
            std::filesystem::remove(path);
            if(!out) throw std::runtime_error("logits write failed");
            if(prefill_qsa32_enabled()) {
                // Exercise the actual --short-read64 scheduler and the canonical32 reference
                // in the same admitted model. Each replay starts from an empty state.
                const auto served=[&](uint32_t chunk,const std::string& suffix,bool canonical) {
                    model.reset();std::vector<float> row;
                    for(uint32_t at=0;at<ids.size();) {
                        const uint32_t remaining=uint32_t(ids.size())-at;const bool window=remaining>64;
                        const uint32_t count=window?(canonical?canonical_prefill_count(remaining,chunk,64,32):std::min(chunk,remaining)):remaining;
                        model.prefill_window(std::span<const int32_t>(ids).subspan(at,count),[&](uint32_t,std::span<const float> x){row.assign(x.begin(),x.end());},{},{},!window);
                        at+=count;
                    }
                    std::ofstream file(std::filesystem::path(argv[3])/(std::to_string(case_id)+suffix),std::ios::binary);
                    const auto save=[&](std::span<const float> x){file.write(reinterpret_cast<const char*>(x.data()),std::streamsize(x.size_bytes()));};
                    save(row);for(uint32_t j=0;j<16;++j) save(model.decode(ids[(j*17)%ids.size()]));
                    if(!file) throw std::runtime_error("served logits write failed");
                };
                served(o.memory.prefill_rows,".serve.f32",true);served(32,".serve32.f32",false);
            }
            std::cout<<"PASS case="<<case_id++<<" prompt="<<ids.size()<<" logits_rows=17 vocab="<<model.vocabulary()
                     <<" snapshot_branch_replay=exact disk_restore=exact stop_window="
                     <<(o.memory.prefill_rows>=2?"exact":"not_windowed")<<"\n"<<std::flush;
        }
        if(!case_id) throw std::runtime_error("no quality cases");
    } catch(const std::exception& e) {if(observed) {const auto h=observed->hbm_allocations(),p=observed->pinned_allocations();std::cerr<<"ALLOC hbm_used="<<h.used<<" hbm_peak="<<h.peak<<" hbm_limit="<<h.limit<<" pinned_used="<<p.used<<" pinned_peak="<<p.peak<<" pinned_limit="<<p.limit<<"\n";}std::cerr<<e.what()<<'\n';return 1;}
}
