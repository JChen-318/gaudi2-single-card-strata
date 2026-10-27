// One acquired Model, same options and prefixes: forced scalar vs batched prefill.
// Natural continuations come from the scalar reference and are teacher-forced
// into the candidate. Also isolates a single 8-row window at shared snapshots.
#include "strata/hpu/model.hpp"
#include "strata/hpu/ple_io_mode.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
using namespace strata::hpu;
namespace {
void write(const std::filesystem::path& path,const std::vector<std::vector<float>>& rows) {
    std::ofstream out(path,std::ios::binary);
    for(const auto& row:rows) out.write(reinterpret_cast<const char*>(row.data()),std::streamsize(row.size()*4));
    if(!out) throw std::runtime_error("failed writing logits");
}
std::vector<float> prefill(Model& model,std::span<const int32_t> ids,bool scalar) {
    std::vector<float> out;
    model.prefill_window(ids,[&](uint32_t,std::span<const float> row){out.assign(row.begin(),row.end());},{},{},scalar);
    return out;
}
}
int main(int argc,char** argv) {
    if(argc!=4) return 2;
    try {
        ModelOptions o;o.fp8_experts=true;o.memory.bf16_head=true;o.memory.context=4096;
        o.memory.expert_cache=2048ull<<20;o.memory.host_limit=49152ull<<20;o.memory.prefill_rows=8;
        o.ple_io=resolve_ple_io("ram",o.ple_io);
        Model model(argv[1],o);std::cout<<"READY shared-model window probe\n"<<std::flush;
        const std::filesystem::path root(argv[3]);
        for(const char* mode:{"scalar","batch","isolated"}) std::filesystem::create_directories(root/mode);
        std::ifstream cases(argv[2]);if(!cases) throw std::runtime_error("cases file missing");
        uint32_t case_id=0;std::string line;
        while(std::getline(cases,line)) {
            std::replace(line.begin(),line.end(),',',' ');std::istringstream input(line);
            std::vector<int32_t> ids;int32_t id;while(input>>id) ids.push_back(id);
            if(ids.size()<16 || ids.size()+32>o.memory.context) throw std::runtime_error("case length");
            model.reset();
            // Same scalar prefix on both sides of each isolated window. Resetting
            // the prefix between comparisons prevents accumulated drift hiding a
            // position/causality defect in the window itself.
            for(uint32_t position:{0u,128u,2040u,2048u,2304u}) {
                if(position+8>ids.size()) continue;
                if(model.size()<position) (void)prefill(model,std::span<const int32_t>(ids).subspan(model.size(),position-model.size()),true);
                const auto point=model.snapshot();
                const auto window=std::span<const int32_t>(ids).subspan(position,8);
                const auto scalar=prefill(model,window,true);model.restore(point);
                const auto batch=prefill(model,window,false);model.restore(point);
                write(root/"isolated"/(std::to_string(case_id)+"-"+std::to_string(position)+".f32"),{scalar,batch});
            }
            if(model.size()<ids.size()) (void)prefill(model,std::span<const int32_t>(ids).subspan(model.size()),true);
            std::vector<std::vector<float>> scalar{model.logits()},batch;
            std::vector<int32_t> suffix;
            for(uint32_t step=0;step<16;++step) {
                const auto& row=scalar.back();const auto token=int32_t(std::max_element(row.begin(),row.end())-row.begin());
                suffix.push_back(token);scalar.push_back(model.decode(token));
            }
            model.reset();batch.push_back(prefill(model,ids,false));
            for(const auto token:suffix) batch.push_back(model.decode(token));
            write(root/"scalar"/(std::to_string(case_id)+".f32"),scalar);
            write(root/"batch"/(std::to_string(case_id)+".f32"),batch);
            std::ofstream forced(root/(std::to_string(case_id)+".suffix.txt"));
            for(const auto token:suffix) forced<<token<<' ';
            if(!forced) throw std::runtime_error("failed writing suffix");
            std::cout<<"PASS case="<<case_id++<<" prompt="<<ids.size()<<" scalar_greedy_teacher_forced=16 vocab="<<model.vocabulary()<<'\n'<<std::flush;
        }
        if(!case_id) throw std::runtime_error("no cases");
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
