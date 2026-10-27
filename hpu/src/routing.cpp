#include "strata/hpu/routing.hpp"
#include "strata/platform/aux_cpus.hpp"
#include <future>
#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>

namespace strata::hpu {
Routes route_reference(std::span<const float> logits, uint32_t tokens, uint32_t experts, uint32_t top_k) {
    if (!tokens || !experts || experts > 4096 || !top_k || top_k > experts ||
        logits.size() != checked_product(tokens, experts)) throw std::invalid_argument("invalid router geometry");
    for (auto value : logits) if (!std::isfinite(value)) throw std::invalid_argument("router requires finite logits");
    Routes result{tokens, experts, top_k, {}, {}};
    result.ids.resize(checked_product(tokens, top_k));
    result.weights.resize(result.ids.size());
    std::vector<double> exponentials(experts);
    std::vector<float> probabilities(experts);
    std::vector<int32_t> order(experts);
    for (uint32_t token = 0; token < tokens; ++token) {
        const auto row = logits.subspan(uint64_t(token) * experts, experts);
        const auto maximum = *std::max_element(row.begin(), row.end());
        double denominator = 0;
        for (uint32_t expert = 0; expert < experts; ++expert) {
            exponentials[expert] = std::exp(double(row[expert]) - double(maximum));
            denominator += exponentials[expert];
        }
        const float inverse = float(1.0 / denominator);
        for (uint32_t expert = 0; expert < experts; ++expert) probabilities[expert] = float(exponentials[expert] * inverse);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + top_k, order.end(), [&](int32_t a, int32_t b) {
            return probabilities[a] > probabilities[b] || (probabilities[a] == probabilities[b] && a < b);
        });
        double selected_sum = 0;
        for (uint32_t rank = 0; rank < top_k; ++rank) selected_sum += double(probabilities[order[rank]]);
        const double scale = std::max(selected_sum, 0x1p-14);
        for (uint32_t rank = 0; rank < top_k; ++rank) {
            const auto offset = uint64_t(token) * top_k + rank;
            result.ids[offset] = order[rank];
            result.weights[offset] = float(double(probabilities[order[rank]]) / scale);
        }
    }
    return result;
}
RoutedParts::RoutedParts(Routes routes, uint32_t hidden) : routes_(std::move(routes)), hidden_(hidden) {
    const auto count = checked_product(routes_.tokens, routes_.top_k);
    if (!count || !hidden || !routes_.experts || routes_.experts > 4096 || routes_.top_k > routes_.experts ||
        routes_.ids.size() != count || routes_.weights.size() != count)
        throw std::invalid_argument("invalid routed-parts geometry");
    std::map<int32_t, std::vector<uint64_t>> grouped;
    for (uint64_t index = 0; index < count; ++index) {
        const auto id = routes_.ids[index];
        const auto weight = routes_.weights[index];
        if (!std::isfinite(weight) || weight < 0 || id < -1 || id >= int64_t(routes_.experts) ||
            (id == -1 && weight != 0)) throw std::invalid_argument("invalid routed expert or weight");
        for (auto prior = index - index % routes_.top_k; id != -1 && prior < index; ++prior)
            if (routes_.ids[prior] == id) throw std::invalid_argument("duplicate expert within one token");
        if (id != -1) grouped[id].push_back(index);
    }
    for (auto& [id, indices] : grouped) groups_.push_back({id, std::move(indices)});
    parts_.resize(checked_product(count, hidden));
    filled_.resize(count);
}
void RoutedParts::scatter_group(size_t group_index, std::span<const float> rows) {
    if (group_index >= groups_.size()) throw std::out_of_range("expert group index");
    const auto& indices = groups_[group_index].routed_indices;
    if (rows.size() != checked_product(indices.size(), hidden_)) throw std::invalid_argument("expert row count mismatch");
    for (auto index : indices) if (filled_[index]) throw std::logic_error("expert result supplied twice");
    for (auto value : rows) if (!std::isfinite(value)) throw std::invalid_argument("non-finite expert result");
    for (size_t row = 0; row < indices.size(); ++row) {
        const auto index = indices[row];
        std::copy_n(rows.data() + row * hidden_, hidden_, parts_.data() + index * hidden_);
        filled_[index] = 1;
    }
}
std::vector<float> RoutedParts::combine() const {
    for (size_t index = 0; index < routes_.ids.size(); ++index)
        if (routes_.ids[index] != -1 && !filled_[index]) throw std::logic_error("missing expert result");
    std::vector<float> output(checked_product(routes_.tokens, hidden_));
    for (uint32_t token = 0; token < routes_.tokens; ++token) {
        for (uint32_t rank = 0; rank < routes_.top_k; ++rank) {
            const auto index = uint64_t(token) * routes_.top_k + rank;
            if (routes_.ids[index] == -1) continue;
            for (uint32_t col = 0; col < hidden_; ++col)
                output[uint64_t(token) * hidden_ + col] += parts_[index * hidden_ + col] * routes_.weights[index];
        }
    }
    return output;
}
std::vector<float> dispatch_moe(Routes routes,std::span<const float> x,uint32_t h,std::span<const float> gates,const RoutedExecutor& execute) {
    if(!h || !execute || x.size()!=checked_product(routes.tokens,h) || gates.size()!=routes.tokens)
        throw std::invalid_argument("MoE dispatch input/gate/executor shape mismatch");
    for(float value:x) if(!std::isfinite(value)) throw std::invalid_argument("MoE input must be finite");
    for(float value:gates) if(!std::isfinite(value) || value<0 || value>1) throw std::invalid_argument("MoE shared gate outside [0,1]");
    const auto top_k=routes.top_k,tokens=routes.tokens;
    RoutedParts parts(std::move(routes),h);const auto groups=parts.groups();
    for(size_t group=0;group<groups.size();++group) {
        const auto& selected=groups[group];std::vector<float> gathered(checked_product(selected.routed_indices.size(),h));
        for(size_t row=0;row<selected.routed_indices.size();++row) {
            const auto token=selected.routed_indices[row]/top_k;
            std::copy_n(x.begin()+token*h,h,gathered.begin()+row*h);
        }
        const auto result=execute(selected.expert_id,gathered);parts.scatter_group(group,result);
    }
    auto output=parts.combine();const auto shared=execute(-1,x);
    if(shared.size()!=x.size()) throw std::invalid_argument("shared expert row shape mismatch");
    for(float value:shared) if(!std::isfinite(value)) throw std::invalid_argument("shared expert result must be finite");
    for(uint32_t row=0;row<tokens;++row) for(uint32_t col=0;col<h;++col) {
        output[uint64_t(row)*h+col]+=shared[uint64_t(row)*h+col]*gates[row];
        if(!std::isfinite(output[uint64_t(row)*h+col])) throw std::runtime_error("non-finite MoE combined output");
    }
    return output;
}
std::vector<float> dispatch_hybrid_moe(Routes routes,std::span<const float> x,uint32_t h,std::span<const float> gates,
    const std::function<ExpertPlacement(int32_t)>& place,const RoutedExecutor& hpu,const RoutedExecutor& cpu) {
    return dispatch_hybrid_moe(std::move(routes),x,h,gates,place,hpu,cpu,{});
}
std::vector<float> dispatch_hybrid_moe(Routes routes,std::span<const float> x,uint32_t h,std::span<const float> gates,
    const std::function<ExpertPlacement(int32_t)>& place,const RoutedExecutor& hpu,const RoutedExecutor& cpu,
    const RoutedGroupExecutor& hpu_group) {
    if(!h || !place || !hpu || x.size()!=checked_product(routes.tokens,h) || gates.size()!=routes.tokens)
        throw std::invalid_argument("hybrid MoE input/callback shape mismatch");
    for(float value:x) if(!std::isfinite(value)) throw std::invalid_argument("hybrid MoE input must be finite");
    for(float value:gates) if(!std::isfinite(value) || value<0 || value>1) throw std::invalid_argument("hybrid shared gate outside [0,1]");
    const auto tokens=routes.tokens,top_k=routes.top_k;RoutedParts parts(std::move(routes),h);const auto groups=parts.groups();
    std::vector<ExpertPlacement> destinations;destinations.reserve(groups.size());bool has_cpu=false;
    for(const auto& group:groups) {
        const auto destination=place(group.expert_id);
        if(destination!=ExpertPlacement::Cpu && destination!=ExpertPlacement::ResidentHpu && destination!=ExpertPlacement::LoadHpu) throw std::invalid_argument("unknown expert placement");
        has_cpu|=destination==ExpertPlacement::Cpu;destinations.push_back(destination);
    }
    if(has_cpu && !cpu) throw std::invalid_argument("hybrid CPU callback missing");
    auto gather=[&](size_t group) {
        const auto& selected=groups[group];std::vector<float> rows(checked_product(selected.routed_indices.size(),h));
        for(size_t row=0;row<selected.routed_indices.size();++row) std::copy_n(x.begin()+selected.routed_indices[row]/top_k*h,h,rows.begin()+row*h);
        return rows;
    };
    using Completed=std::vector<std::pair<size_t,std::vector<float>>>;
    // Declared after every borrowed input; async future's destructor joins on
    // exceptions before those inputs, callbacks or leases can be destroyed.
    std::future<Completed> pending;
    if(has_cpu) pending=std::async(std::launch::async,[&] {
        strata::aux_cpus::pin_current_thread();
        Completed completed;
        for(size_t group=0;group<groups.size();++group) if(destinations[group]==ExpertPlacement::Cpu) {
            const auto input=gather(group);completed.emplace_back(group,cpu(groups[group].expert_id,input));
        }
        return completed;
    });
    std::vector<float> shared;
    if(hpu_group) {
        std::vector<std::pair<int32_t,std::vector<float>>> calls;std::vector<size_t> owners;
        for(const auto destination:{ExpertPlacement::ResidentHpu,ExpertPlacement::LoadHpu})
            for(size_t group=0;group<groups.size();++group) if(destinations[group]==destination) {
                calls.emplace_back(groups[group].expert_id,gather(group));owners.push_back(group);
            }
        calls.emplace_back(-1,std::vector<float>(x.begin(),x.end()));
        auto outputs=hpu_group(calls);
        if(outputs.size()!=calls.size()) throw std::invalid_argument("hybrid grouped HPU result count mismatch");
        for(size_t i=0;i<owners.size();++i) parts.scatter_group(owners[i],outputs[i]);
        shared=std::move(outputs.back());
    } else {
        for(const auto destination:{ExpertPlacement::ResidentHpu,ExpertPlacement::LoadHpu})
            for(size_t group=0;group<groups.size();++group) if(destinations[group]==destination) {
                const auto input=gather(group);const auto output=hpu(groups[group].expert_id,input);parts.scatter_group(group,output);
            }
        shared=hpu(-1,x);
    }
    if(shared.size()!=x.size()) throw std::invalid_argument("hybrid shared expert row shape mismatch");
    for(float value:shared) if(!std::isfinite(value)) throw std::invalid_argument("hybrid shared output must be finite");
    if(pending.valid()) for(const auto& [group,output]:pending.get()) parts.scatter_group(group,output);
    auto output=parts.combine();
    for(uint32_t row=0;row<tokens;++row) for(uint32_t col=0;col<h;++col) {
        auto& value=output[uint64_t(row)*h+col];value+=shared[uint64_t(row)*h+col]*gates[row];
        if(!std::isfinite(value)) throw std::runtime_error("non-finite hybrid combined output");
    }
    return output;
}
} // namespace strata::hpu
