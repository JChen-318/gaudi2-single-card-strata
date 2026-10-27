#include "strata/hpu/native_pack.hpp"
#include "strata/hpu/expert_scales.hpp"
#include <iostream>
#include <algorithm>

using namespace strata::hpu;
int main(int argc,char** argv) {
    if(argc!=3) return 2;
    try {
        const WeightTable table(argv[1]);const NativeExpertPack pack(argv[2],table);
        for(uint32_t l:{0u,1u,47u}) for(uint32_t e:{0u,1u,511u}) {
            const auto slices=pack.slices(l,e);const auto prefix="blk."+std::to_string(l)+".ffn_";
            const std::array<WeightSlice,3> original{table.expert(prefix+"gate_exps.weight",e),table.expert(prefix+"up_exps.weight",e),table.expert(prefix+"down_exps.weight",e)};
            uint64_t offset=0;const auto blob=pack.native_blob(l,e);
            for(size_t r=0;r<3;++r) {
                if(slices[r].ggml_type()!=original[r].ggml_type() || slices[r].source_bytes()!=original[r].source_bytes() ||
                   !std::equal(slices[r].bytes().begin(),slices[r].bytes().end(),original[r].bytes().begin()) ||
                   !std::equal(slices[r].bytes().begin(),slices[r].bytes().end(),blob.begin()+offset)) throw std::runtime_error("native pack differs from actual main GGUF expert");
                offset+=slices[r].source_bytes();
            }
            if(pack.scales(l,e)!=expert_global_scales(table,l,int32_t(e))) throw std::runtime_error("native pack actual main GGUF scale binding mismatch");
        }
        std::cout<<"PASS actual full-GGUF catalog, absolute source offsets, original pack producer, CPU blob/HPU slices and scale binding; no device\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
