#pragma once
#include "strata/hpu/recipe.hpp"
#include "strata/hpu/host_recipe.hpp"
namespace strata::hpu {
void define_model_head(GraphBuilder&,uint32_t hidden,uint32_t vocabulary,uint32_t batch=1);
std::unique_ptr<Recipe> build_model_head(const Device&,uint32_t hidden,uint32_t vocabulary,uint32_t batch=1);
void define_packed_matmul(GraphBuilder&,uint32_t columns,uint32_t rows,uint32_t batch,uint32_t type,PackedActivation=PackedActivation::Q8_1,bool device_quantizer=false);
// Borrows ready immutable weights. Packed Q2/Q4/Q5/Q8 stay packed in HBM; other
// formats use the existing BF16 upload/MME reference. Serialized by Model/MTP.
// Also used for the MTP FC/QKV/O linear projections; no head-specific norm.
class ModelHead {
public:
    ModelHead(const Device&,Binding weight,uint32_t hidden,uint32_t vocabulary,uint32_t source_type,uint32_t batch=1,PackedActivation=PackedActivation::Q8_1);
    std::vector<float> run(std::span<const float>);
    // Dynamic1..128 rows; preserves the original constructor batch for run().
    // Holds only the current recipe, releasing it before a geometry change.
    std::vector<float> run_rows(std::span<const float>);
    void bind(Binding);
private:
    Device device_;
    Binding weight_;
    std::unique_ptr<HostRecipe> recipe_;
    uint32_t hidden_,rows_,type_,batch_,recipe_batch_=0;
    PackedActivation mode_;
    void prepare(uint32_t);
    std::vector<float> execute(std::span<const float>,uint32_t);
};
}
