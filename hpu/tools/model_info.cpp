#include "strata/hpu/weights.hpp"
#include <iostream>
int main(int argc,char** argv) {
    if(argc!=2) {std::cerr<<"usage: strata-hpu-model-info any-shard.gguf\n";return 2;}
    try {
        strata::hpu::WeightTable table(argv[1]);const auto& g=table.geometry();
        std::cout<<"qwen4exp shards="<<table.shard_count()<<" layers="<<g.n_layers<<" hidden="<<g.n_embd<<" experts="<<g.n_expert<<" top_k="<<g.n_expert_used<<'\n';
        uint64_t bytes=0;
        for(const auto& t:table.inventory()) {
            if(t.source_bytes>UINT64_MAX-bytes) throw std::overflow_error("model payload total overflow");
            bytes+=t.source_bytes;
            std::cout<<t.name<<" shard="<<table.shard(t.name)<<" type="<<t.ggml_type<<" shape=";
            for(auto d:t.gguf_shape) std::cout<<d<<',';
            std::cout<<" payload="<<t.source_bytes<<" bf16_reference="<<t.bf16_conversion_supported<<'\n';
        }
        std::cout<<"payload_bytes="<<bytes<<" complete_shard_family geometry_checked mmap_only no_device_acquired\n"
                   "Model catalog only; full layer tensor schema, tokenizer and generation are not validated by this tool.\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
