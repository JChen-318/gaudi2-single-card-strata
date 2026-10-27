#include "strata/hpu/native_pack.hpp"

namespace strata::hpu {
namespace {
NativePackCatalog catalog(const WeightTable& table) {
    const auto& g=table.geometry();NativePackCatalog out;out.experts=uint32_t(g.n_expert);out.layers.resize(g.n_layers);out.shards=table.shard_files();
    constexpr std::array<const char*,3> roles{"gate","up","down"};
    for(uint32_t l=0;l<out.layers.size();++l) {
        auto& layer=out.layers[l];layer.scales.assign(out.experts,{1,1,1});
        for(size_t r=0;r<3;++r) {
            const auto stem="blk."+std::to_string(l)+".ffn_"+roles[r]+"_exps";
            const auto& info=table.info(stem+".weight");
            const uint64_t columns=r<2?g.n_embd:g.n_ff,rows=r<2?g.n_ff:g.n_embd;
            if(info.gguf_shape!=std::vector<uint64_t>{columns,rows,out.experts} || info.source_bytes%out.experts)
                throw std::invalid_argument("native expert pack main GGUF role shape mismatch");
            const auto location=table.location(stem+".weight");
            layer.roles[r]={info.ggml_type,columns,rows,location.offset,info.source_bytes/out.experts,location.path};
            if(info.ggml_type==40) {
                const auto& scale=table.info(stem+".scale");
                if(scale.ggml_type!=0 || scale.gguf_shape!=std::vector<uint64_t>{out.experts}) throw std::invalid_argument("native expert pack requires F32 per-expert NVFP4 scales");
                const auto values=table.constant(stem+".scale",out.experts,4096);
                for(uint32_t e=0;e<out.experts;++e) layer.scales[e][r]=values[e];
            }
        }
    }
    out.gguf_expert=[table](uint32_t l,uint32_t e) {
        const auto prefix="blk."+std::to_string(l)+".ffn_";
        return std::array<WeightSlice,3>{table.expert(prefix+"gate_exps.weight",e),table.expert(prefix+"up_exps.weight",e),table.expert(prefix+"down_exps.weight",e)};
    };
    return out;
}
}
NativeExpertPack::NativeExpertPack(const std::string& directory,const WeightTable& table):NativeExpertPack(directory,catalog(table)) {}
}
