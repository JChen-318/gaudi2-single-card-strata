#include "strata/hpu/expert_scales.hpp"
namespace strata::hpu {
std::array<float,3> expert_global_scales(const WeightTable& table,uint32_t layer,int32_t expert) {
    const auto& g=table.geometry();
    if(layer>=uint64_t(g.n_layers) || expert< -1 || expert>=g.n_expert) throw std::out_of_range("expert scale layer/ID");
    std::array<uint32_t,3> types{};std::array<WeightSlice,3> owners;std::array<std::span<const uint8_t>,3> payloads;
    const char* roles[]{"gate","up","down"};const uint32_t count=expert==-1?1:uint32_t(g.n_expert);
    for(size_t role=0;role<3;++role) {
        const auto stem="blk."+std::to_string(layer)+".ffn_"+roles[role]+(expert==-1?"_shexp":"_exps");
        types[role]=table.info(stem+".weight").ggml_type;
        if(types[role]!=40) continue;
        const auto& info=table.info(stem+".scale");
        if(info.ggml_type!=0 || info.gguf_shape!=std::vector<uint64_t>{count}) throw std::invalid_argument("NVFP4 .scale must be F32 with one value per expert: "+stem);
        owners[role]=table.vector(stem+".scale");payloads[role]=owners[role].bytes();
    }
    return decode_expert_scales(types,payloads,expert==-1?0:uint32_t(expert),count);
}
void validate_expert_global_scales(const WeightTable& table,uint32_t layer) {
    // decode_expert_scales validates every value, not only the requested ID.
    (void)expert_global_scales(table,layer,0);(void)expert_global_scales(table,layer,-1);
}
}
