#include "strata/hpu/fp8_batch.hpp"
#include "strata/hpu/model_abi.hpp"
#include <perf_lib_layer_params.h>
namespace strata::hpu {
void define_u4_fp8_batch(GraphBuilder& g,TensorId a,TensorId codes,TensorId out,
                         uint64_t batch,uint64_t rows,uint64_t k,uint64_t n,
                         uint32_t terms,const std::string& prefix) {
    if(terms!=1 && terms!=2) throw std::invalid_argument("FP8 batch needs one or two activation terms");
    const auto t=[&](const char* suffix,DType dtype,std::vector<uint64_t> shape) {
        return g.tensor({prefix+suffix,dtype,std::move(shape),false});
    };
    const auto x=t("_x",DType::F32,{batch,rows,k});g.cast(a,x);
    const auto stacked=t("_stacked",DType::F8,{batch,terms*rows,k}),scale=t("_scale",DType::F32,{batch,rows,1});
    abi::Fp8Batch encode{int32_t(k),int32_t(rows),int32_t(terms)};
    {const TensorId in[]={x},o[]={stacked,scale};g.op("strata_fp8_batch_encode_f32",in,o,&encode,sizeof(encode));}
    const auto expanded=t("_weight",DType::F8,{batch,n,k});ns_CastKernel::ParamsV3 cast{};
    {const TensorId in[]={codes},o[]={expanded};g.op("cast_u4_to_hf8",in,o,&cast,sizeof(cast));}
    const auto both=t("_both",DType::F32,{batch,terms*rows,n});g.batch_gemm(stacked,expanded,both,true);
    abi::Fp8Batch finish{int32_t(n),int32_t(rows),int32_t(terms)};
    {const TensorId in[]={both,scale},o[]={out};g.op("strata_fp8_batch_finish_f32",in,o,&finish,sizeof(finish));}
}
}
