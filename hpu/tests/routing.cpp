#include "strata/hpu/routing.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <atomic>
#include <chrono>
#include <future>
#include <thread>

using namespace strata::hpu;
namespace {
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
template<class F> void rejects(F&& call) {
    bool rejected = false;
    try { call(); } catch (const std::exception&) { rejected = true; }
    require(rejected, "invalid route operation accepted");
}
}
int main() {
    try {
        std::vector<float> logits(2 * 512, 0);
        logits[512 + 511] = 10000;
        logits[512 + 1] = -1;
        const auto routes = route_reference(logits, 2, 512, 10);
        for (int rank = 0; rank < 10; ++rank) {
            require(routes.ids[rank] == rank, "equal logits must select ascending IDs");
            require(std::abs(routes.weights[rank] - 0.1f) < 1e-7f, "uniform route normalization");
        }
        require(routes.ids[10] == 511 && routes.weights[10] == 1, "extreme router logit");
        for (int rank = 1; rank < 10; ++rank)
            require(routes.ids[10 + rank] == rank - 1 && routes.weights[10 + rank] == 0,
                    "underflowed probabilities must tie by ID, independent of original logits");
        rejects([&] { auto bad = logits; bad[2] = std::numeric_limits<float>::quiet_NaN(); (void)route_reference(bad, 2, 512, 10); });
        rejects([&] { (void)route_reference(logits, 2, 512, 513); });
        const std::vector<float> known{0, std::log(2.0f), std::log(3.0f), std::log(4.0f)};
        const auto known_route = route_reference(known, 1, 4, 2);
        require(known_route.ids == std::vector<int32_t>{3, 2} &&
                std::abs(known_route.weights[0] - 4.0f / 7) < 1e-6 &&
                std::abs(known_route.weights[1] - 3.0f / 7) < 1e-6, "known softmax probabilities");
        // Two tokens, shared experts, a padded route, and deliberately different
        // weights make loss of routed index or double weighting visible.
        Routes selected{2, 4, 3, {2, 0, -1, 0, 3, 2}, {0.25f, 0.75f, 0, 0.125f, 0.375f, 0.5f}};
        RoutedParts parts(selected, 2);
        rejects([&] { (void)parts.combine(); });
        const auto groups = parts.groups();
        require(groups.size() == 3 && groups[0].expert_id == 0 && groups[1].expert_id == 2 && groups[2].expert_id == 3,
                "expert grouping");
        // Complete HPU/CPU groups in reverse scheduling order. Rows identify the
        // source token and expert and are never weighted by the executor.
        for (size_t i = groups.size(); i-- > 0;) {
            std::vector<float> values;
            for (auto index : groups[i].routed_indices) {
                values.push_back(float(10 * (index / 3) + groups[i].expert_id));
                values.push_back(float(100 + 10 * (index / 3) + groups[i].expert_id));
            }
            parts.scatter_group(i, values);
            rejects([&] { parts.scatter_group(i, values); });
        }
        const auto combined = parts.combine();
        const std::vector<float> expected{0.5f, 100.5f, 12.125f, 112.125f};
        require(combined == expected, "routed order/weight-once combine mismatch");
        rejects([&] { auto bad = selected; bad.ids[2] = 999; (void)RoutedParts(bad, 2); });
        rejects([&] { auto bad = selected; bad.weights[2] = 0.5f; (void)RoutedParts(bad, 2); });
        rejects([&] { auto bad = selected; bad.weights[0] = -1; (void)RoutedParts(bad, 2); });
        rejects([&] { auto bad = selected; bad.ids[0] = 0; (void)RoutedParts(bad, 2); });
        // Same scheduler used by MoeBlock: known token/expert functions expose
        // wrong activation gathering, double route weighting or shared mixing.
        const std::vector<float> activation{2,3,5,7},shared_gate{0.25f,0.75f};
        int calls=0;
        const auto executor=[&](int32_t expert,std::span<const float> input) {
            ++calls;std::vector<float> out(input.begin(),input.end());
            for(auto& value:out) value=value*float(expert==-1?10:expert+1)+float(expert==-1?1:expert);
            return out;
        };
        const auto dispatched=dispatch_moe(selected,activation,2,shared_gate,executor);
        require(calls==4,"dispatch must call each grouped expert and shared expert once");
        std::vector<float> direct(4);
        for(uint32_t token=0;token<2;++token) for(uint32_t d=0;d<2;++d) {
            for(uint32_t rank=0;rank<3;++rank) {
                const auto index=token*3+rank;
                const auto id=selected.ids[index];
                if(id!=-1) direct[token*2+d]+=(activation[token*2+d]*float(id+1)+float(id))*selected.weights[index];
            }
            direct[token*2+d]+=(activation[token*2+d]*10+1)*shared_gate[token];
        }
        require(dispatched==direct,"MoE scheduler gather/weight/shared combine mismatch");
        rejects([&]{(void)dispatch_moe(selected,activation,2,std::vector<float>{0,2},executor);});
        rejects([&]{(void)dispatch_moe(selected,activation,2,shared_gate,[](int32_t,std::span<const float>){return std::vector<float>{};});});
        require(calls==4,"invalid shared gate must fail before any execution");
        PlacementPolicy hybrid(1);
        require(hybrid.choose(5,true)==ExpertPlacement::ResidentHpu && hybrid.loads()==0,"resident hit consumed upload budget");
        require(hybrid.choose(-1,false)==ExpertPlacement::LoadHpu && hybrid.loads()==0,"shared expert changed routed policy");
        require(hybrid.choose(2,false)==ExpertPlacement::LoadHpu && hybrid.loads()==1,"first cold HPU expert");
        require(hybrid.choose(3,false)==ExpertPlacement::Cpu && hybrid.choose(2,true)==ExpertPlacement::ResidentHpu,"CPU miss fallback/resident HPU reuse");
        PlacementPolicy cpu_routed(0);require(cpu_routed.choose(0,false)==ExpertPlacement::Cpu && cpu_routed.choose(-1,false)==ExpertPlacement::LoadHpu,"zero routed upload budget");
        // A handshake proves both dispatchers are live at the same time; it
        // does not infer overlap from wall-clock timing or artificial speedups.
        std::promise<void> cpu_started,hpu_started;auto cpu_ready=cpu_started.get_future();auto hpu_ready=hpu_started.get_future();
        const auto main_thread=std::this_thread::get_id();std::vector<int32_t> hpu_order;
        auto pure=[](int32_t expert,std::span<const float> input) {
            std::vector<float> out(input.begin(),input.end());for(auto& value:out) value=value*float(expert==-1?10:expert+1)+float(expert==-1?1:expert);
            return out;
        };
        auto mixed=dispatch_hybrid_moe(selected,activation,2,shared_gate,[](int32_t id){return id==0?ExpertPlacement::Cpu:id==3?ExpertPlacement::ResidentHpu:ExpertPlacement::LoadHpu;},
            [&](int32_t id,std::span<const float> input) {
                hpu_order.push_back(id);
                if(id==3) {require(cpu_ready.wait_for(std::chrono::seconds(3))==std::future_status::ready,"CPU work did not start beside HPU work");hpu_started.set_value();}
                return pure(id,input);
            },[&](int32_t id,std::span<const float> input) {
                require(std::this_thread::get_id()!=main_thread,"CPU groups ran on the HPU dispatch thread");cpu_started.set_value();
                require(hpu_ready.wait_for(std::chrono::seconds(3))==std::future_status::ready,"HPU work did not start beside CPU work");return pure(id,input);
            });
        require(mixed==direct && hpu_order==std::vector<int32_t>({3,2,-1}),"hybrid gather/combine or resident-before-upload ordering");
        std::atomic<bool> drained=false;std::promise<void> begin_cpu,release_cpu;auto began=begin_cpu.get_future();auto release=release_cpu.get_future();
        rejects([&]{(void)dispatch_hybrid_moe(selected,activation,2,shared_gate,[](int32_t id){return id==0?ExpertPlacement::Cpu:ExpertPlacement::LoadHpu;},
            [&](int32_t,std::span<const float>)->std::vector<float> {
                require(began.wait_for(std::chrono::seconds(3))==std::future_status::ready,"CPU failure test never started");release_cpu.set_value();throw std::runtime_error("HPU failure");
            },[&](int32_t id,std::span<const float> input) {
                begin_cpu.set_value();require(release.wait_for(std::chrono::seconds(3))==std::future_status::ready,"CPU release missing");drained=true;return pure(id,input);
            });});
        require(drained,"HPU exception escaped while CPU still held activation inputs");
        std::cout << "PASS host router [2,512] top10 stable probability ties and finite-input contract\n"
                     "PASS mixed completion order, original routed indices, padding, exactly-once combine\n"
                     "PASS native MoE scheduler activation gather, grouped dispatch, shared expert and gate validation\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
