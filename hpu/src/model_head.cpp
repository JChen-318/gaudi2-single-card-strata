#include "strata/hpu/model_head.hpp"
#include "strata/hpu/session_flow.hpp"
#include "strata/hpu/packed_codebooks.hpp"
#include "strata/hpu/packed_quant.hpp"
#include "strata/hpu/linear_plan.hpp"
#include "strata/hpu/window_limits.hpp"
#include <array>
#include <cmath>
namespace strata::hpu {
ModelHead::ModelHead(const Device& device,Binding weight,uint32_t hidden,uint32_t vocabulary,uint32_t type,uint32_t batch,PackedActivation mode)
    :device_(device),weight_(std::move(weight)),hidden_(hidden),rows_(vocabulary),type_(type),batch_(batch),mode_(mode) {
    if(weight_.name!="weight") throw std::invalid_argument("output head binding name mismatch");
    prepare(batch);
}
void ModelHead::prepare(uint32_t batch) {
    // RoutedS2 keeps the host quantizer because the device kernel implements the Q8_1 law only.
    const bool device_quantizer=mode_==PackedActivation::Q8_1;
    const auto budget=linear_pinned_bytes(hidden_,rows_,batch,type_,mode_,device_quantizer);
    if(recipe_ && recipe_batch_==batch) return;
    recipe_.reset();recipe_batch_=0;
    std::unique_ptr<Recipe> graph;
    if(packed_activation_supported(type_,mode_)) {GraphBuilder builder(device_);define_packed_matmul(builder,hidden_,rows_,batch,type_,mode_,device_quantizer);graph=builder.compile("strata_packed_linear_"+std::to_string(type_)+"_"+std::to_string(hidden_)+"_"+std::to_string(rows_)+"_"+std::to_string(uint32_t(mode_))+"_"+std::to_string(uint32_t(device_quantizer))+"_"+std::to_string(batch));}
    else {if(mode_!=PackedActivation::Q8_1) throw std::invalid_argument("BF16 projection has no scaled S2 contract");graph=build_model_head(device_,hidden_,rows_,batch);}
    const std::array<Binding,1> weights{weight_};recipe_=std::make_unique<HostRecipe>(device_,std::move(graph),weights,std::vector<std::string>{"logits"},budget);recipe_batch_=batch;
    // The codebook slot belongs to this recipe, so it is uploaded whenever a recipe is rebuilt; an
    // uninitialized input slot would otherwise stop the first run.
    if(packed_activation_supported(type_,mode_)) recipe_->set_bytes("codebook",{packed_codebook_blob(),packed_codebook_bytes()});
}
std::vector<float> ModelHead::run(std::span<const float> input) {
    return execute(input,batch_);
}
std::vector<float> ModelHead::run_rows(std::span<const float> input) {
    if(input.empty() || input.size()%hidden_ || input.size()/hidden_>kResidentWindowRows) throw std::invalid_argument("linear window requires1.."+std::to_string(kResidentWindowRows)+" complete rows");
    return execute(input,uint32_t(input.size()/hidden_));
}
std::vector<float> ModelHead::execute(std::span<const float> input,uint32_t batch) {
    if(input.size()!=uint64_t(hidden_)*batch) throw std::invalid_argument("output head activation shape mismatch");
    prepare(batch);
    // The same rule prepare() used to build the graph: the quantizer lives in the graph for the
    // Q8_1 contract, so the caller hands it FP32 instead of pre-quantized bytes.
    const bool device_quantizer=mode_==PackedActivation::Q8_1;
    if(packed_activation_supported(type_,mode_)) {
        if(device_quantizer) recipe_->set("activation_input",input);
        else {const auto activation=quantize_packed_activation(input,hidden_,mode_);recipe_->set_bytes("activation",activation.codes);recipe_->set("metadata",activation.metadata);}
    } else recipe_->set("x",input);
    recipe_->run();auto logits=recipe_->values("logits");
    if(!all_finite(logits)) throw NonFiniteOutput("non-finite output head logits");
    return logits;
}
void ModelHead::bind(Binding binding) {
    if(binding.name!="weight" || !binding.buffer || binding.buffer->state()!=device_.state() || (binding.buffer->address()+binding.offset)%128)
        throw std::invalid_argument("invalid linear rebind weight");
    const auto bytes=checked_product(rows_,packed_activation_supported(type_,mode_)?packed_row_bytes(type_,hidden_):checked_product(hidden_,2));
    check_range(binding.buffer->size(),binding.offset,bytes);
    if(recipe_) recipe_->bind_external(binding);
    weight_=std::move(binding);
}
}
