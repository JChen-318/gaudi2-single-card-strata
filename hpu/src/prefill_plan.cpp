#include "strata/hpu/prefill_plan.hpp"
#include "strata/hpu/tensor.hpp"
#include "strata/hpu/performance_modes.hpp"
#include "strata/hpu/gdn_window_flow.hpp"
#include "strata/hpu/qsa_window_flow.hpp"
#include "strata/hpu/ple_window_flow.hpp"
#include "strata/hpu/linear_plan.hpp"
#include <algorithm>
namespace strata::hpu {
CandidateMemoryPlan plan_candidate_window(const core::ModelGeometry& g,uint32_t requested,uint32_t context) {
    if(requested>7 || !context || context>INT32_MAX || g.n_vocab<=0 || g.n_vocab>INT32_MAX)
        throw std::invalid_argument("invalid candidate memory geometry");
    if(!requested) return {};
    CandidateMemoryPlan out;out.rows=std::min(requested,context);
    out.logits_bytes=checked_product(checked_product(out.rows,g.n_vocab),sizeof(float));
    // Candidate slab plus native output and an owned-copy allowance, sampling
    // distributions and history. Global pinned limit is charged by ModelPlan.
    out.host_bytes=checked_product(out.logits_bytes,3)+checked_product(g.n_vocab,16)+checked_product(context,4);
    if(g.n_embd<=0 || g.n_embd>65536 || g.n_vocab>1048576) throw std::invalid_argument("invalid candidate head geometry");
    out.head_pinned_bytes=linear_pinned_bytes(uint32_t(g.n_embd),uint32_t(g.n_vocab),out.rows,30);
    out.hbm_bytes=out.head_pinned_bytes+(64ull<<20); // non-weight slots + compiler allowance
    return out;
}
PrefillWindowTerms plan_prefill_window_terms(const core::ModelGeometry& g,uint32_t rows,uint32_t context) {
    if(!rows || !context || g.n_layers<=0) throw std::invalid_argument("prefill window terms need rows, context and layers");
    auto sum=[](uint64_t a,uint64_t b) {if(b>UINT64_MAX-a) throw std::overflow_error("prefill memory sum overflow");return a+b;};
    const uint64_t h=g.n_embd,ff=g.n_ff,e=g.n_expert,n=checked_product(h,g.hc);
    PrefillWindowTerms t;
    // Session current/next rows, layer residual/readback/FFN staging, PLE and
    // MTP passthrough plus grouped routing/expert CPU scratch. KV and rewind
    // images are accounted separately by the main plan; logits are last-only.
    const auto row_host=sum(sum(checked_product(n,12),checked_product(h,8)),sum(checked_product(ff,8),checked_product(e,4)));
    t.staging_host=checked_product(checked_product(rows,row_host),4);
    const auto row_hbm=sum(sum(checked_product(h,16),checked_product(ff,16)),checked_product(e,4));
    // Priced once per layer because the layer-major path staged a layer's residual, readback and FFN scratch
    // between layers, with an 8 MiB compiler allowance each. The resident chain shares one graph across layers, so
    // that multiplication is exactly the term a wide window over-pays for -- and at 1024 rows it was 10.4 GiB of
    // the 13.7 GiB that made "resident IQ4 model plan exceeds HBM memory policy". Resident IQ4 is the only path
    // that runs a wide window (model.cpp refuses >128 rows without it), so price one copy there.
    const uint64_t layer_copies=resident_iq4_enabled()?1:uint64_t(g.n_layers);
    const auto per_layer=sum(checked_product(checked_product(rows,row_hbm),4),8ull<<20);
    t.staging_hbm=checked_product(per_layer,layer_copies);
    t.staging_allowance_hbm=checked_product(8ull<<20,layer_copies);
    // One Model-shared HC read/write pair, not one pair per layer. Include
    // persistent transfer slots, normalized/gate intermediates and allowance.
    const auto hc_row=sum(sum(checked_product(n,6),checked_product(h,4)),checked_product(uint64_t(g.hc),4));
    t.hc_hbm=sum(checked_product(checked_product(rows,hc_row),4),16ull<<20);
    const auto state=checked_product(checked_product(g.ssm_state_size,g.ssm_v_heads),g.ssm_state_size);
    const auto conv=checked_product(g.ssm_d_conv-1,g.ssm_conv_channels),values=checked_product(g.ssm_v_heads,g.ssm_state_size);
    const auto subrows=std::min(rows,kResidentWindowRows);
    // One shared native GDN recipe, not36 copies. Conservatively charge every
    // unrolled intermediate state (compiler liveness reuse is not assumed),
    // projection/body tensors and compiler workspace. Host transfers and staged
    // whole-window results use a single shared allowance, in addition to rows.
    const auto cells=sum(state,conv);
    const auto gdn_row=sum(sum(checked_product(h,4),checked_product(g.ssm_conv_channels,4)),sum(checked_product(values,12),checked_product(g.ssm_v_heads,12)));
    // A final-state window (STRATA_HPU_PREFILL_FINAL) keeps only the window's last state, not one per row.
    const uint64_t state_rows=prefill_final_enabled()?1:subrows;
    t.gdn_hbm=sum(checked_product(sum(checked_product(cells,state_rows+2),checked_product(gdn_row,subrows)),4),32ull<<20);
    t.gdn_host=checked_product(sum(checked_product(cells,8),checked_product(subrows,gdn_row)),4);
    const auto qsa=qsa_config(g,context);const auto qrows=std::min(rows,kQsaWindowRows);
    const uint64_t b=qsa.selection_width(context),hq=qsa.query_heads,hk=qsa.kv_heads,d=qsa.width;
    const auto q=checked_product(hq,d),kv=checked_product(checked_product(hk,b),d);
    const auto projections=sum(sum(checked_product(q,2),checked_product(checked_product(hk,d),2)),sum(checked_product(qsa.index_heads,qsa.index_width),qsa.index_width));
    const auto base=sum(sum(checked_product(h,8),checked_product(projections,8)),sum(checked_product(q,16),checked_product(checked_product(hk,b),4)));
    // One Model-shared QSA triple. Include projection transfers, both batched
    // KV tensors, per-row/head masks, score/softmax/context intermediates and
    // owned host copies/gather staging. No T*context or T*vocabulary slab.
    const auto qsa_hbm_row=sum(base,sum(checked_product(kv,4),checked_product(checked_product(hq,b),8)));
    const auto qsa_host_row=sum(base,sum(checked_product(kv,8),checked_product(b,4)));
    t.qsa_hbm=sum(checked_product(checked_product(qrows,qsa_hbm_row),4),32ull<<20);
    t.qsa_host=sum(checked_product(checked_product(qrows,qsa_host_row),4),8ull<<20);
    const auto prows=std::min(rows,kPleWindowRows);const auto ple_history=checked_product(9,n);
    // Exactly one layer1 PLE recipe. Charge all unrolled causal histories even
    // if the SDK can reuse them, plus projection/body and owned transfer slots.
    const auto ple_hbm=sum(checked_product(ple_history,prows+2),checked_product(prows,sum(checked_product(n,12),checked_product(h,4))));
    const auto ple_host=sum(checked_product(ple_history,6),checked_product(prows,sum(checked_product(n,4),checked_product(h,4))));
    t.ple_hbm=sum(checked_product(ple_hbm,4),16ull<<20);
    t.ple_host=checked_product(ple_host,4);
    return t;
}
PrefillMemoryPlan plan_prefill_window(const core::ModelGeometry& g,uint32_t requested,uint32_t context) {
    if(requested>kResidentWindowRows || !context || g.n_embd<=0 || g.hc<=0 || g.n_ff<=0 || g.n_expert<=0 || g.n_layers<=0 ||
       g.ssm_state_size<=0 || g.ssm_v_heads<=0 || g.ssm_conv_channels<=0 || g.ssm_d_conv<2)
        throw std::invalid_argument("invalid grouped prefill memory geometry");
    if(!requested) return {};
    PrefillMemoryPlan out;out.rows=std::min(requested,context);
    const auto t=plan_prefill_window_terms(g,out.rows,context);
    auto add=[](uint64_t total,uint64_t bytes) {if(bytes>UINT64_MAX-total) throw std::overflow_error("prefill memory sum overflow");return total+bytes;};
    out.hbm_bytes=add(add(add(add(t.staging_hbm,t.hc_hbm),t.gdn_hbm),t.qsa_hbm),t.ple_hbm);
    out.host_bytes=add(add(add(t.staging_host,t.gdn_host),t.qsa_host),t.ple_host);
    return out;
}
PrefillMemoryPlan select_prefill_window(const core::ModelGeometry& g,uint32_t context,uint64_t host,uint64_t hbm) {
    // Evaluate row1 even with no headroom, so invalid geometry is not hidden
    // by the fallback. Checked arithmetic in the planner must still fail.
    auto selected=plan_prefill_window(g,1,context);
    if(selected.host_bytes>host || selected.hbm_bytes>hbm) return {};
    uint32_t low=1,high=std::min(kResidentWindowRows,context);
    while(low<high) {
        const auto mid=low+(high-low+1)/2;
        const auto candidate=plan_prefill_window(g,mid,context);
        if(candidate.host_bytes<=host && candidate.hbm_bytes<=hbm) {low=mid;selected=candidate;}
        else high=mid-1;
    }
    return selected;
}
}
