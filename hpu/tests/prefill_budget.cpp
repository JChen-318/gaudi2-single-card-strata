// Offline prefill-window memory table. Pure arithmetic over strata::core::kCompiledGeometry: no Synapse, no device,
// no model file, so it runs anywhere and every number is checkable against prefill_plan.cpp by hand.
// Why this exists: kResidentWindowRows caps the window at 256 rows, and the reason it can just be raised or not is
// which of the window's terms hits the memory policy first. The table names each term at each width and, given the
// bytes a running model has already committed, the widest window still admissible.
// Usage: prefill-budget-test [committed-hbm-mib] [committed-host-mib]   (defaults 0 0)
#include "strata/hpu/prefill_plan.hpp"
#include "strata/core/geometry.hpp"
#include "strata/hpu/window_limits.hpp"
#include "strata/hpu/qsa_window_flow.hpp"
#include "strata/hpu/gdn_window_flow.hpp"
#include "strata/hpu/ple_window_flow.hpp"
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
using namespace strata::hpu;
namespace {
constexpr double kMib=1048576.0;
// The production run's policy (work/round31 harness: --hbm-limit-mib 90112 --host-limit-mib 49152).
constexpr uint64_t kHbmLimitMib=90112,kHostLimitMib=49152;
const strata::core::ModelGeometry& g=strata::core::kCompiledGeometry;

struct Row {
    uint32_t rows;PrefillWindowTerms t;uint64_t hbm,host;
};
Row measure(uint32_t rows,uint32_t context) {
    Row r{rows,plan_prefill_window_terms(g,rows,context),0,0};
    r.hbm=r.t.staging_hbm+r.t.hc_hbm+r.t.gdn_hbm+r.t.qsa_hbm+r.t.ple_hbm;
    r.host=r.t.staging_host+r.t.gdn_host+r.t.qsa_host+r.t.ple_host;
    return r;
}
// Widest window whose terms still fit beside an already-committed model, ignoring the compile-time cap.
uint32_t widest(uint32_t context,uint64_t hbm_budget,uint64_t host_budget) {
    uint32_t best=0;
    for(uint32_t rows=32;rows<=8192;rows+=rows<128?32:64) {
        const auto r=measure(rows,context);
        if(r.hbm>hbm_budget||r.host>host_budget) break;
        best=rows;
    }
    return best;
}

// What the resident chain actually declares per window, read off define_resident_moe
// (src/resident_experts.cpp:145-228) with the production flag set: Q8_PACKED, MOE_GROUP, the fused
// gate/up->activation quantizer and MOE_TAIL_FUSED, GEMV parts 2 (gate/up) and 1 (down).
// One recipe is shared by all 48 layers, so none of these multiplies by the layer count.
struct ResidentSlab {uint64_t bytes=0;std::vector<std::pair<const char*,uint64_t>> parts;};
ResidentSlab resident_slab(uint32_t rows) {
    const uint64_t h=g.n_embd,ff=g.n_ff,experts=g.n_expert,top_k=g.n_expert_used,per_row=top_k+1,r=rows,slots=per_row*r;
    const uint64_t gu_parts=2*slots*2*ff*4;         // gu_parts {2, slots, 2*ff} F32
    const uint64_t act_codes=slots*(ff/16)*256;     // activation_codes {256, ff/16, slots} U8
    const uint64_t act_scales=slots*(ff/32)*64*4;   // activation_scales {64, ff/32, slots} F32
    const uint64_t expert_parts=1*slots*h*4;        // expert_parts {1, slots, h} F32
    const uint64_t weighted=slots*h*4;              // weighted_rows {slots, h} F32
    const uint64_t block=2*r*h*4;                   // expert_sum {r,1,h} + expert_block {r,h} F32
    const uint64_t x_codes=r*(h/16)*256, x_scales=r*(h/32)*64*4;
    const uint64_t route=slots*(4+4+4)+r*4;         // ids, route_weights, route_column, group_order + shared_gate
    const uint64_t group=(experts+1)*4;             // group_starts
    ResidentSlab s;
    auto add=[&](const char* name,uint64_t bytes) {s.parts.emplace_back(name,bytes);s.bytes+=bytes;};
    add("gu_parts",gu_parts);add("act_codes",act_codes);add("act_scales",act_scales);add("expert_parts",expert_parts);
    add("weighted",weighted);add("block",block);add("x_codes",x_codes);add("x_scales",x_scales);
    add("route",route);add("group_starts",group);
    return s;
}

void line(const char* label,double value,double per_row) {
    std::printf("  %-22s %10.1f MiB %10.3f MiB/row\n",label,value,per_row);
}
}

int main(int argc,char** argv) {
    const uint64_t committed_hbm=argc>1?std::stoull(argv[1]):0,committed_host=argc>2?std::stoull(argv[2]):0;
    const uint32_t context=argc>3?uint32_t(std::stoul(argv[3])):4096;
    std::printf("geometry n_embd=%lld ff=%lld experts=%lld top_k=%lld hc=%lld layers=%lld vocab=%lld context=%u\n",
                (long long)g.n_embd,(long long)g.n_ff,(long long)g.n_expert,(long long)g.n_expert_used,(long long)g.hc,
                (long long)g.n_layers,(long long)g.n_vocab,(unsigned)context);
    std::printf("compile-time caps kResidentWindowRows=%u kQsaWindowRows=%u kPleWindowRows=%u kGdnWindowRows=%u; "
                "the MoE slot cap is kResidentMoeSlots in resident_experts.hpp (needs the Synapse include path, so it is not read here)\n",
                kResidentWindowRows,kQsaWindowRows,kPleWindowRows,kGdnWindowRows);
    for(const uint32_t rows:{64u,128u,256u,512u,1024u,2048u}) {
        const auto r=measure(rows,context);
        const auto slab=resident_slab(rows);
        std::printf("\nrows=%u slots=%u\n",rows,rows*unsigned(g.n_expert_used+1));
        line("layer-major staging",r.t.staging_hbm/kMib,double(r.t.staging_hbm-r.t.staging_allowance_hbm)/kMib/rows);
        line("  of which allowance",r.t.staging_allowance_hbm/kMib,0.0);
        line("HC read/write pair",r.t.hc_hbm/kMib,r.t.hc_hbm/kMib/rows);
        line("GDN recipe",r.t.gdn_hbm/kMib,r.t.gdn_hbm/kMib/rows);
        line("QSA triple",r.t.qsa_hbm/kMib,r.t.qsa_hbm/kMib/rows);
        line("PLE recipe",r.t.ple_hbm/kMib,r.t.ple_hbm/kMib/rows);
        line("TOTAL plan HBM",r.hbm/kMib,r.hbm/kMib/rows);
        line("TOTAL plan host",r.host/kMib,r.host/kMib/rows);
        std::printf("  %-22s %10.1f MiB %10.3f MiB/row  (sum of the resident graph's own tensors)\n",
                    "resident MoE slab",double(slab.bytes)/kMib,double(slab.bytes)/kMib/rows);
        for(const auto& p:slab.parts) std::printf("      %-18s %8.1f MiB\n",p.first,double(p.second)/kMib);
        // These have different owners: the generic plan includes HC/GDN/QSA/PLE
        // while this slab is only MoE. Their ratio does not measure overcharging;
        // Model replaces the generic reservation on the resident-chain path.
        std::printf("  generic window total and resident MoE slab have different scopes; use Model::preflight for resident admission\n");
        if(committed_hbm||committed_host) std::printf("  beside %llu MiB committed HBM: window HBM %s (limit %llu MiB)\n",
            (unsigned long long)committed_hbm,r.hbm/kMib+double(committed_hbm)<=double(kHbmLimitMib)?"fits":"NO",
            (unsigned long long)kHbmLimitMib);
    }
    if(committed_hbm<kHbmLimitMib&&committed_host<kHostLimitMib) {
        std::printf("\nwidest admissible window beside %llu MiB committed HBM / %llu MiB committed host: %u rows\n",
                    (unsigned long long)committed_hbm,(unsigned long long)committed_host,
                    widest(context,(kHbmLimitMib-committed_hbm)*1048576,(kHostLimitMib-committed_host)*1048576));
    }
    std::printf("PASS prefill_budget\n");
    return 0;
}
