#include "strata/hpu/prefill_plan.hpp"
#include "strata/hpu/window_limits.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>
using namespace strata::hpu;
namespace {
void require(bool ok) {if(!ok) throw std::runtime_error("prefill admission violated fit/maximality/fallback contract");}
template<class F> void rejects(F&& fn) {try {fn();} catch(const std::exception&) {return;}throw std::runtime_error("invalid admission geometry accepted");}
uint32_t exhaustive(const strata::core::ModelGeometry& g,uint32_t context,uint64_t host,uint64_t hbm) {
    uint32_t result=0;
    for(uint32_t rows=1;rows<=std::min(kResidentWindowRows,context);++rows) {
        const auto p=plan_prefill_window(g,rows,context);
        if(p.host_bytes<=host && p.hbm_bytes<=hbm) result=rows;
    }
    return result;
}
}
int main() {
    try {
        const auto& g=strata::core::kCompiledGeometry;uint32_t cases=0;
        for(const uint32_t context:{1u,3u,8u,17u,127u,128u,32768u}) {
            const auto cap=std::min(kResidentWindowRows,context);
            for(uint32_t rows=1;rows<=cap;++rows) {
                const auto p=plan_prefill_window(g,rows,context);
                // Exact fit and one-byte misses in either resource, with the
                // other resource unlimited, plus two-resource admission.
                for(const auto& budget:{std::pair{p.host_bytes,p.hbm_bytes},
                                      std::pair{p.host_bytes-1,UINT64_MAX},
                                      std::pair{UINT64_MAX,p.hbm_bytes-1},
                                      std::pair{p.host_bytes-1,p.hbm_bytes-1}}) {
                    const auto selected=select_prefill_window(g,context,budget.first,budget.second);
                    require(selected.rows==exhaustive(g,context,budget.first,budget.second));
                    require(selected.host_bytes<=budget.first && selected.hbm_bytes<=budget.second);
                    require(selected.rows || (!selected.host_bytes && !selected.hbm_bytes));++cases;
                }
            }
            require(select_prefill_window(g,context,UINT64_MAX,UINT64_MAX).rows==cap);
            require(!select_prefill_window(g,context,0,UINT64_MAX).rows);
            require(!select_prefill_window(g,context,UINT64_MAX,0).rows);
        }
        rejects([&]{(void)select_prefill_window(g,0,0,0);});
        auto bad=g;bad.n_embd=INT64_MAX;rejects([&]{(void)select_prefill_window(bad,128,0,0);});
        bad=g;bad.ssm_d_conv=1;rejects([&]{(void)select_prefill_window(bad,128,0,0);});
        require(!plan_candidate_window(g,0,128).host_bytes);
        for(uint32_t context:{1u,3u,8u,128u,32768u}) for(uint32_t rows=1;rows<=7;++rows) {
            const auto p=plan_candidate_window(g,rows,context);require(p.rows==std::min(rows,context));
            require(p.logits_bytes==uint64_t(p.rows)*uint64_t(g.n_vocab)*4);
            require(p.host_bytes==3*p.logits_bytes+uint64_t(g.n_vocab)*16+uint64_t(context)*4);
            require(p.head_pinned_bytes==uint64_t(p.rows)*(uint64_t(g.n_embd)*2+uint64_t(g.n_vocab)*4));
            require(p.hbm_bytes==p.head_pinned_bytes+(64ull<<20));
        }
        rejects([&]{(void)plan_candidate_window(g,8,128);});rejects([&]{(void)plan_candidate_window(g,1,0);});
        bad=g;bad.n_vocab=INT64_MAX;rejects([&]{(void)plan_candidate_window(bad,7,128);});
        std::cout<<"PASS resource-admitted prefill auto: "<<cases<<" exact/one-byte budget boundary cases vs exhaustive planner, both resources, context caps, singleton fallback and checked geometry; no device\n";
        std::cout<<"PASS candidate slab/sampling/history budgets35 cases, context clamping, disabled/invalid/overflow geometry\n";
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
