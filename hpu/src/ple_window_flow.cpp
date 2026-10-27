#include "strata/hpu/ple_window_flow.hpp"
#include "strata/hpu/tensor.hpp"
#include "strata/hpu/session_flow.hpp"
#include <algorithm>
#include <cmath>
namespace strata::hpu {
namespace {
void valid(std::span<const float> values,uint64_t count) {
    if(values.size()!=count) throw std::invalid_argument("PLE window tensor shape mismatch");
    for(float value:values) if(!std::isfinite(value)) throw std::runtime_error("non-finite PLE window tensor");
}
uint64_t sum(uint64_t a,uint64_t b) {if(b>UINT64_MAX-a) throw std::overflow_error("PLE window byte overflow");return a+b;}
}
uint64_t ple_window_pinned_bytes(uint32_t hidden,uint32_t streams,uint32_t rows) {
    if(!hidden || !streams || !rows || rows>kPleWindowRows) throw std::invalid_argument("PLE native slot geometry mismatch");
    const auto n=checked_product(hidden,streams);
    return sum(checked_product(n,sum(72,checked_product(rows,8))),checked_product(checked_product(rows,hidden),2));
}
PleWindowResult execute_ple_window(std::span<const float> input,std::span<const float> embedding,std::span<const float> history,
                                   uint32_t hidden,uint32_t streams,const PleWindowRunner& runner,const std::function<bool()>& cancelled) {
    if(!hidden || !streams || !runner) throw std::invalid_argument("incomplete PLE window geometry/runner");
    const auto n=checked_product(hidden,streams);
    if(input.empty() || input.size()%n || input.size()/n>128) throw std::invalid_argument("PLE window requires1..128 complete rows");
    const auto rows=input.size()/n;valid(input,input.size());valid(embedding,checked_product(rows,hidden));valid(history,checked_product(9,n));
    auto poll=[&] {if(cancelled && cancelled()) throw Cancelled();};poll();
    PleWindowResult result;result.output.resize(input.size());result.history.assign(history.begin(),history.end());
    for(size_t row=0;row<rows;) {
        poll();const auto count=std::min<size_t>(kPleWindowRows,rows-row);
        auto next=runner(input.subspan(row*n,count*n),embedding.subspan(row*hidden,count*hidden),result.history);
        valid(next.output,count*n);valid(next.history,history.size());poll();
        std::copy(next.output.begin(),next.output.end(),result.output.begin()+row*n);result.history=std::move(next.history);row+=count;
    }
    poll();return result;
}
}
