#include "strata/hpu/gdn_window_flow.hpp"
#include "strata/hpu/session_flow.hpp"
#include <algorithm>
#include <cmath>
namespace strata::hpu {
namespace {
void valid(std::span<const float> values,size_t count) {
    if(values.size()!=count) throw std::invalid_argument("GDN window result shape mismatch");
    for(auto value:values) if(!std::isfinite(value)) throw std::runtime_error("non-finite GDN window input/output/state");
}
}
GdnWindowResult execute_gdn_window(std::span<const float> input,uint32_t hidden,std::span<const float> state,
                                 std::span<const float> conv,const GdnWindowRunner& runner,const std::function<bool()>& cancelled) {
    if(!hidden || input.empty() || input.size()%hidden || input.size()/hidden>128 || state.empty() || conv.empty() || !runner)
        throw std::invalid_argument("GDN window requires1..128 rows and complete state/runner");
    valid(input,input.size());valid(state,state.size());valid(conv,conv.size());
    auto poll=[&] {if(cancelled && cancelled()) throw Cancelled();};poll();
    GdnWindowResult result;result.output.resize(input.size());result.state.assign(state.begin(),state.end());result.conv.assign(conv.begin(),conv.end());
    const auto rows=input.size()/hidden;
    for(size_t row=0;row<rows;) {
        poll();const auto n=std::min<size_t>(kGdnWindowRows,rows-row);
        auto next=runner(input.subspan(row*hidden,n*hidden),result.state,result.conv);
        valid(next.output,n*hidden);valid(next.state,state.size());valid(next.conv,conv.size());poll();
        std::copy(next.output.begin(),next.output.end(),result.output.begin()+row*hidden);
        result.state=std::move(next.state);result.conv=std::move(next.conv);row+=n;
    }
    poll();return result;
}
}
