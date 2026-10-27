#pragma once
// Test-only logical tensor graph. Only tests link its implementation; no SDK
// runtime/recipe implementation or libSynapse is linked. Not an ISA simulator.
#include "strata/hpu/tensor.hpp"
#include <functional>
#include <span>
namespace oracle {
using namespace strata::hpu;
struct Slot {TensorDesc desc;std::vector<float> values;};
struct Graph {
    std::vector<Slot> slots;std::vector<std::function<void()>> nodes;
    Slot& get(const std::string& name) {for(auto& slot:slots) if(slot.desc.name==name) return slot;throw std::runtime_error("unknown interpreted tensor");}
    void set(const std::string& name,std::span<const float> values) {
        auto& slot=get(name);if(values.size()!=slot.values.size()) throw std::runtime_error("interpreted input shape mismatch");
        for(size_t i=0;i<values.size();++i) slot.values[i]=slot.desc.dtype==DType::BF16?from_bf16(to_bf16(values[i])):values[i];
    }
    void run() {for(const auto& node:nodes) node();}
};
extern Graph* current;
void require(bool,const char*);
void close(std::span<const float>,std::span<const float>);
}
