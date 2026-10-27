#include "strata/hpu/mtp_window_plan.hpp"
#include "strata/hpu/mtp_window_flow.hpp"
#include "strata/hpu/dense_window_flow.hpp"
#include "strata/hpu/linear_plan.hpp"
#include <algorithm>
namespace strata::hpu {
MtpWindowMemoryPlan plan_mtp_window(const core::ModelGeometry& g,uint32_t requested,uint32_t context,uint32_t window) {
    if(requested>128 || !context || context>INT32_MAX || window>context || g.n_embd<=0 || g.n_embd>65536 || g.hc<=0 || g.hc>32 || g.n_ff<=0 || g.n_expert<=0 || g.hc_lr<=0)
        throw std::invalid_argument("invalid MTP window memory geometry");
    if(!requested) return {};
    const auto c=qsa_config(g,context);const uint64_t h=g.n_embd,r=g.hc,n=checked_product(h,r),q=checked_product(c.query_heads,c.width),k=checked_product(c.kv_heads,c.width);
    auto add=[](uint64_t a,uint64_t b){if(b>UINT64_MAX-a) throw std::overflow_error("MTP window memory overflow");return a+b;};
    MtpWindowMemoryPlan out;out.rows=std::min({requested,context,kMtpWindowRows});const auto t=out.rows;
    uint64_t projections=linear_pinned_bytes(uint32_t(h),uint32_t(h),t,8);
    projections=add(projections,linear_pinned_bytes(uint32_t(h),uint32_t(h),uint32_t(std::min<uint64_t>(128,t*r)),8));
    projections=add(projections,linear_pinned_bytes(uint32_t(h),uint32_t(q*2),t,8));
    projections=add(projections,checked_product(linear_pinned_bytes(uint32_t(h),uint32_t(k),t,8),2));
    projections=add(projections,linear_pinned_bytes(uint32_t(q),uint32_t(h),t,8));
    auto dense=c;dense.top_k=context;const auto b=dense_attention_bucket(context,context,window);
    const auto attention=qsa_window_attention_pinned_bytes(dense,t,b);
    const auto hc=add(checked_product(t,add(checked_product(n,3),add(checked_product(h,2),checked_product(r,2))))*4,4);
    out.pinned_bytes=add(add(projections,attention),hc);
    const auto scores=checked_product(checked_product(checked_product(t,c.query_heads),b),16); // four F32 score tensors
    const auto hc_temporary=checked_product(checked_product(t,add(checked_product(n,8),checked_product(g.hc_lr,8))),4);
    out.hbm_bytes=add(add(add(out.pinned_bytes,scores),hc_temporary),192ull<<20);
    out.hbm_bytes=add(out.hbm_bytes,checked_product(checked_product(t,q),24));
    // MME may need independent key/value layouts in addition to transfer slots.
    out.hbm_bytes=add(out.hbm_bytes,checked_product(checked_product(checked_product(t,k),b),8));
    const auto row=add(add(checked_product(n,16),checked_product(h,16)),add(checked_product(g.n_ff,8),checked_product(g.n_expert,8)));
    out.host_bytes=checked_product(checked_product(std::min(requested,context),row),4);
    out.host_bytes=add(out.host_bytes,add(checked_product(attention,3),add(checked_product(projections,2),8ull<<20)));
    return out;
}
ModelWindowMemoryPlan select_model_prefill_window(const core::ModelGeometry& g,uint32_t context,uint64_t host,uint64_t hbm,uint64_t pinned,bool mtp,uint32_t attention_window,const WindowHostCost& additional_host,uint32_t minimum_rows) {
    (void)plan_prefill_window(g,0,context);if(mtp) (void)plan_mtp_window(g,0,context,attention_window);
    if(minimum_rows>std::min(128u,context)) throw std::invalid_argument("invalid minimum automatic prefill window");
    ModelWindowMemoryPlan out;bool found=false;
    for(uint32_t rows=minimum_rows;rows<=std::min(128u,context);++rows) {
        const auto target=plan_prefill_window(g,rows,context);
        const auto draft=mtp?plan_mtp_window(g,rows,context,attention_window):MtpWindowMemoryPlan{};
        const auto extra=additional_host?additional_host(rows):0;
        if(target.host_bytes<=host && draft.host_bytes<=host-target.host_bytes && extra<=host-target.host_bytes-draft.host_bytes && target.hbm_bytes<=hbm && draft.hbm_bytes<=hbm-target.hbm_bytes && draft.pinned_bytes<=pinned) {
            out={target,draft,extra};found=true;
        }
    }
    if(!found) throw std::invalid_argument("automatic prefill and requested verify/diagnostic features exceed host/HBM/pinned memory policy");
    return out;
}
}
