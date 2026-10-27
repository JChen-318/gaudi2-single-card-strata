#include "strata/hpu/rope_config.hpp"
namespace strata::hpu {
strata::kernels::RopeScaling resolve_rope(const WeightTable& table,const RopeOptions& options) {
    using Type=strata::kernels::RopeScalingType;strata::kernels::RopeScaling result;
    for(const auto& value:{options.factor,options.freq_base,options.freq_scale,options.orig_ctx,options.ext_factor,options.attn_factor,options.beta_fast,options.beta_slow})
        if(value && !std::isfinite(*value)) throw std::invalid_argument("RoPE options must be finite");
    if(options.type && *options.type!=Type::None && *options.type!=Type::Linear && *options.type!=Type::YaRN) throw std::invalid_argument("unknown RoPE scaling type");
    const auto type=table.text("qwen4exp.rope.scaling.type");
    if(type) {
        if(*type=="none") result.type=Type::None;
        else if(*type=="linear") result.type=Type::Linear;
        else if(*type=="yarn") result.type=Type::YaRN;
        else throw std::invalid_argument("unsupported GGUF rope.scaling.type");
    }
    if(options.type) result.type=*options.type;
    if(const auto base=table.number("qwen4exp.rope.freq_base")) result.freq_base=*base;
    if(const auto factor=table.number("qwen4exp.rope.scaling.factor")) result.factor=*factor;
    if(const auto original=table.number("qwen4exp.rope.scaling.original_context_length")) result.orig_ctx=*original;
    auto assign=[](double& field,const std::optional<double>& value) {if(value) field=*value;};
    assign(result.freq_base,options.freq_base);assign(result.factor,options.factor);assign(result.orig_ctx,options.orig_ctx);
    assign(result.freq_scale_in,options.freq_scale);assign(result.attn_factor,options.attn_factor);
    assign(result.beta_fast,options.beta_fast);assign(result.beta_slow,options.beta_slow);
    result.ext_factor=result.type==Type::YaRN?1:0;assign(result.ext_factor,options.ext_factor);
    if(result.type==Type::None) {const auto base=result.freq_base;result={};result.freq_base=base;}
    if(const auto* error=strata::kernels::rope_scaling_invalid(result)) throw std::invalid_argument(error);
    return result;
}
}
