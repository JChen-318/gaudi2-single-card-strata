#include "strata/hpu/model_plan.hpp"
#include "strata/hpu/plan_json.hpp"
#include <charconv>
#include <iostream>
int main(int argc,char** argv) {
    if(argc<2 || argc>4) {std::cerr<<"usage: strata-hpu-plan any-shard.gguf [context [external-ple.gguf]]\n";return 2;}
    try {
        strata::hpu::ModelMemoryOptions options;
        if(argc>=3) {
            const std::string text(argv[2]);const auto [end,error]=std::from_chars(text.data(),text.data()+text.size(),options.context);
            if(error!=std::errc{} || end!=text.data()+text.size()) throw std::invalid_argument("invalid context integer");
        }
        const strata::hpu::WeightTable table(argv[1]);const auto plan=strata::hpu::plan_model(table,options,argc==4?argv[3]:"");
        strata::hpu::write_plan_json(std::cout,plan);
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
