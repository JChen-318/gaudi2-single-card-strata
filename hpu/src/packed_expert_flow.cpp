#include "strata/hpu/packed_expert_flow.hpp"
#include <cmath>
#include <stdexcept>
namespace strata::hpu {
namespace {
void check(std::span<const float> values,size_t count) {
    if(values.size()!=count) throw std::invalid_argument("packed expert projection shape mismatch");
    for(float value:values) if(!std::isfinite(value)) throw std::runtime_error("non-finite packed expert value");
}
}
std::vector<float> execute_packed_expert(std::span<const float> rows,uint32_t h,uint32_t ff,const ExpertProjection& project,std::array<float,3> scales) {
    if(!h || !ff || h>65536 || ff>65536 || rows.empty() || rows.size()%h || rows.size()/h>128 || !project) throw std::invalid_argument("invalid packed expert flow geometry/callback");
    check(rows,rows.size());const auto count=rows.size()/h;
    for(float scale:scales) if(!std::isfinite(scale) || scale<=0) throw std::invalid_argument("expert output scales must be positive finite");
    auto gate=project(0,rows),up=project(1,rows);check(gate,count*ff);check(up,count*ff);
    for(size_t i=0;i<gate.size();++i) {gate[i]*=scales[0];up[i]*=scales[1];}
    check(gate,count*ff);check(up,count*ff);
    for(size_t i=0;i<gate.size();++i) gate[i]=(gate[i]/(1.0f+std::exp(-gate[i])))*up[i];
    check(gate,count*ff);auto result=project(2,gate);check(result,rows.size());
    for(float& value:result) {value*=scales[2];}
    check(result,rows.size());return result;
}
}
