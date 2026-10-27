#pragma once
#include "strata/hpu/control_vector.hpp"
#include "strata/hpu/recipe.hpp"
namespace strata::hpu {
void define_control_vector(GraphBuilder&,uint32_t hidden,uint32_t streams,ControlVectorMode);
}
