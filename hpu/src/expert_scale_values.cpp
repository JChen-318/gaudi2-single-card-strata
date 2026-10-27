#include "strata/hpu/expert_scales.hpp"
#include <bit>
#include <cmath>
#include <cstring>
namespace strata::hpu {
static_assert(std::endian::native==std::endian::little,"native scale tails require little endian");
std::array<float,3> decode_expert_scales(std::array<uint32_t,3> types,const std::array<std::span<const uint8_t>,3>& payloads,uint32_t expert,uint32_t count) {
    if(!count || count>4096 || expert>=count) throw std::invalid_argument("invalid NVFP4 scale index/count");
    std::array<float,3> scales{1,1,1};
    for(size_t role=0;role<3;++role) {
        if(types[role]!=40) {if(!payloads[role].empty()) throw std::invalid_argument("global scale attached to a non-NVFP4 projection");continue;}
        if(payloads[role].size()!=uint64_t(count)*4) throw std::invalid_argument("NVFP4 scale payload shape mismatch");
        for(uint32_t i=0;i<count;++i) {
            float value;std::memcpy(&value,payloads[role].data()+i*4,4);
            if(!std::isfinite(value) || value<=0) throw std::invalid_argument("NVFP4 global scales must be positive finite F32");
            if(i==expert) scales[role]=value;
        }
    }
    return scales;
}
std::vector<uint8_t> native_expert_blob(const std::array<WeightSlice,3>& slices,uint64_t up,uint64_t down,uint64_t total,uint64_t tail,std::array<float,3> scales) {
    const auto end=tail?tail:total;
    if(!up || slices[0].ggml_type()!=slices[1].ggml_type() || down!=checked_product(up,2) || end<down || slices[0].source_bytes()!=up || slices[1].source_bytes()!=up ||
       slices[2].source_bytes()!=end-down || (tail && (tail>total || total-tail!=16)) || !total)
        throw std::invalid_argument("native expert blob byte layout mismatch");
    const std::array<uint32_t,3> types{slices[0].ggml_type(),slices[1].ggml_type(),slices[2].ggml_type()};
    const bool needs_tail=types[0]==40 || types[1]==40 || types[2]==40;
    if(needs_tail!=bool(tail)) throw std::invalid_argument("native expert NVFP4 tail presence mismatch");
    for(size_t role=0;role<3;++role) if(!std::isfinite(scales[role]) || scales[role]<=0 || (types[role]!=40 && scales[role]!=1))
        throw std::invalid_argument("invalid native expert projection scale");
    std::vector<uint8_t> blob(total);
    std::memcpy(blob.data(),slices[0].bytes().data(),up);std::memcpy(blob.data()+up,slices[1].bytes().data(),up);
    std::memcpy(blob.data()+down,slices[2].bytes().data(),end-down);
    if(tail) {const std::array<float,4> values{scales[0],scales[1],scales[2],0};std::memcpy(blob.data()+tail,values.data(),16);}
    return blob;
}
}
