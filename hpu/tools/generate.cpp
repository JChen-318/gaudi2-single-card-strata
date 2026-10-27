#include "strata/hpu/model.hpp"
#include "strata/hpu/performance_modes.hpp"
#include "strata/hpu/stop_tokens.hpp"
#include "strata/hpu/families.hpp"
#include "strata/hpu/aux_cpus.hpp"
#include "strata/hpu/cpu_experts.hpp"
#include "strata/hpu/protocol.hpp"
#include "strata/hpu/state_io.hpp"
#include "strata/hpu/plan_json.hpp"
#include "strata/hpu/window_limits.hpp"
#include "strata/core/coupled_draft.hpp"
#include "strata/core/spec_prob.hpp"
#include "strata/core/conversation_memory.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/hpu/kv_residency.hpp"
#include "strata/hpu/kv_mode.hpp"
#include "strata/hpu/stage_stats.hpp"
#include "strata/hpu/ple_io_mode.hpp"
#include "strata/hpu/ple_mode.hpp"
#include "strata/hpu/expert_residency.hpp"
#include "strata/hpu/spec_draft_source.hpp"
#include "strata/hpu/batch_slots.hpp"
#include "strata/hpu/slot_session.hpp"
#include "strata/hpu/sampler.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include <charconv>
#include <climits>
#include <cmath>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <iostream>
#include <filesystem>
#include <cstdlib>
#include <unistd.h>

using namespace strata::hpu;
namespace {
template<class T> T number(const std::string& text) {
    T value{};const auto [end,error]=std::from_chars(text.data(),text.data()+text.size(),value);
    if(error!=std::errc{} || end!=text.data()+text.size()) throw std::invalid_argument("invalid command-line integer: "+text);
    return value;
}

}
int main(int argc,char** argv) {
    try {
        // max_new is upstream's --max-new default (src/program/generate.cpp:340: 16); the serve path
        // takes the limit per request, so this is the one-shot --tokens default.
        ModelOptions options;GenerationOptions generation_options;std::string path,tokens,keys,layer_split_spec,split_device_spec;uint32_t max_new=16,mtp_max_t=0;bool have_tokens=false,serve=false,session_save_reclaim=false,plan_only=false,plan_json=false,elastic=false,prefill_reference=false,stats_report=false;bool prompt_cache_given=false,spec_given=false,host_core_given=false;
        // Upstream's end-of-turn stop law lives in strata/hpu/stop_tokens.hpp: `--stop-eos` (:2097) and
        // `--eos-ids` (:2187-2191) arm the ordinary token loop, and serving without a token input arms it by
        // itself (:2528-2532). Both flags default off/upstream's pair, so an unarmed run emits past its EOS.
        bool stop_eos=false,eos_ids_given=false,context_positional_given=false;std::vector<int32_t> eos_ids;
        // --aux-cpus / STRATA_AUX_CPUS (src/program/generate.cpp:1998-2008, :2299-2308, :5071-5076). Empty means
        // the flag was absent, which is also what lets the environment variable apply exactly as upstream.
        std::string aux_cpus_spec;
        SpecKnobRequest spec_knobs; // --spec-oracle / --spec-follow / --spec-corrupt
        DumpOptions dump_options;int64_t logits_stride=1;
        // --batch / --slots / --batch-groups (src/program/generate.cpp:2075-2084). The gates, the warnings and
        // the clamp are resolve_batch's (strata/hpu/batch_slots.hpp).
        // Upstream caps the slots at the verify-window rows, strata::kernels::kVerifyMaxT
        // (include/strata/kernels/verify_kernels.hpp:21, src/program/generate.cpp:4523); take the constant
        // rather than a copy of its value so a pinned change cannot drift here unnoticed.
        BatchOptions batch_options;batch_options.cap=strata::kernels::kVerifyMaxT;
        options.mtp_sampling={strata::core::coupled_draft_env(),strata::core::spec_gate_pick(),strata::core::spec_draft_temp_scale(),strata::core::spec_min_temp()};
        generation_options.cache_min_free=2560ull<<20;
        // The two environment knobs of upstream's --resident-experts (src/program/generate.cpp:2214-2217):
        // STRATA_RESIDENT_PIN=0 drops the page-lock the flag otherwise implies, and
        // STRATA_RESIDENT_HEADROOM_GIB=N overrides its 4 GiB RAM headroom. Both are read before the loop, like
        // the other environment knobs below; an unparsable headroom is 0 GiB, which is what upstream's atof
        // would produce, and 0 is a legal headroom rather than a refusal.
        const bool resident_pin_zero=[] {const auto* pin=std::getenv("STRATA_RESIDENT_PIN");return pin && std::string(pin)=="0";}();
        double resident_headroom_gib=0;bool has_resident_headroom=false;
        if(const auto* headroom=std::getenv("STRATA_RESIDENT_HEADROOM_GIB");headroom && *headroom) {
            try {resident_headroom_gib=number<double>(headroom);has_resident_headroom=true;} catch(const std::exception&) {resident_headroom_gib=0;has_resident_headroom=true;}
        }
        if(const auto* boundary=std::getenv("STRATA_CACHE_MESSAGE_BOUNDARY");boundary && *boundary) generation_options.boundaries.message_checkpoint=number<int>(boundary)!=0;
        if(const auto* limit=std::getenv("STRATA_MEMORY_LIMIT_MIB");limit && *limit) strata::core::set_memory_limit_bytes(checked_product(number<uint64_t>(limit),1ull<<20));
        if(strata::core::spec_prob_env()) generation_options.verify_mode=VerifyMode::Probabilistic;
        for(int i=1;i<argc;++i) {
            const std::string key(argv[i]);
            auto value=[&] {if(++i>=argc) throw std::invalid_argument("missing value: "+key);return std::string(argv[i]);};
            // Bare arguments are upstream's positional form `strata <model.gguf> [context]`: the first is the model path,
            // a second all-digit argument is the context length, and the run prints the plan JSON (the plan tests drive
            // exactly this shape).
            if(key.rfind("--",0)!=0) {
                if(path.empty()) {
                    if(!std::filesystem::exists(key)) throw std::invalid_argument("no such model file: "+key);
                    path=key;plan_only=true;plan_json=true;continue;
                }
                if(context_positional_given || key.empty() ||
                   key.find_first_not_of("0123456789")!=std::string::npos)
                    throw std::invalid_argument("unexpected positional argument: "+key);
                options.memory.context=number<uint32_t>(key);context_positional_given=true;continue;
            }
            // Upstream's `strata --families` (src/program/generate.cpp:1811-1817): the architectures THIS build
            // runs, printed before any model path exists so a setup step can ask which one to install, then exit 0.
            // strata/hpu/families.hpp is the same list the GGUF reader checks, so the report cannot drift from it.
            if(key=="--families") { std::cout<<families_line()<<"\n"; return 0; }
            // Upstream's `strata --version` / `-V` (src/program/generate.cpp:1804-1809) prints the engine version and
            // returns before any GPU is touched. version_line() comes from the same law as the INFO line's `engine=`
            // fact, so the answer a person gets from the shell is the answer the server reads. `-V` is upstream's
            // short alias; it is a strcmp spelling in a pre-scan loop, so the CLI-surface extractor does not see it.
            if(key=="--version"||key=="-V") { std::cout<<version_line()<<"\n"; return 0; }
            if(key=="--help") {
                std::cout<<"strata-hpu --native any-shard.gguf [--plan | --serve | --tokens 1,2,3 | --tokens-file FILE --max-new 32]\n"
                            "  --max-context N --module ID --expert-cache-mib N --hbm-limit-mib N\n"
                            "  --pinned-limit-mib N --host-limit-mib N --sampling key=value\n"
                            "  --kv fp16|int8|q4_0|k8v4 (k8v4 = unrotated INT8 K + rotated Q4_0 V, its drafter runs INT8)\n"
                            "  --kv-mib MIB --kv-grow | --no-kv-grow --kv-resident N (env STRATA_KV_GROW wins)\n"
                            "  --spec-oracle FILE --spec-follow FILE --spec-corrupt N\n"
                            "  --suffix-draft N --spec T (1..8) --spec-prob\n"
                            "  --session-save-reclaim (default off, as upstream): free only the caches a\n"
                            "  snapshot needs before saving; --tokens-file reads the same list as --tokens\n"
                            "  --verify-window N (0..7; opt-in native target candidate windows)\n"
                            "  scalar draft windows need a nonzero --snapshot-budget-mib for atomic rollback\n"
                            "  --prompt-cache N (0..64), bounded by --snapshot-budget-mib\n"
                            "  --prompt-cache-every N (fresh prompt token interval; 0 disables)\n"
                            "  --prefill N|auto (0..32768; auto admits <=128 rows within full model host/HBM budgets)\n"
                            "  --prefill-reference (sequential token fallback; same prompt chunk boundaries)\n"
                            "  --short-read N (default 1) --prompt-cache-tail\n"
                            "  --prompt-cache-root N --turn-token ID --tail-role-token ID --cache-message-boundary\n"
                            "  --conversation-cache-min-free-mib N --memory-limit-mib N\n"
                            "  --conversation-cache-min-tokens N (--serve parks only conversations at least this long; 0 = all)\n"
                            "  --stop-eos / --eos-ids ID,ID (stop the run on an end-of-turn token; batch slots stop on them either way)\n"
                            "  --session-min-free-mib N (SAVE/RESTORE host-memory floor, default 4096)\n"
                            "  --mtp RUNTIME_DIR --mtp-hnorm pooled|stream --mtp-max-t T (0..8)\n"
                            "  --spec-min-p P --coupled-draft --mtp-expert-cache-mib N --mtp-window N\n"
                            "  --lookup-chain K --lookup-chain-min M\n"
                            "  --adaptive-drafts (opt-in ordinary GEN lookup/MTP policy; unmeasured HPU performance)\n"
                            "  --mtp-draft-vocab FILE (default: <mtp>/draft_vocab.bin when present)\n"
                            "  --ple-gguf FILE (external PLE table; hash metadata comes from --native)\n"
                            "  --no-ple (diagnostic ablation: run without PLE input; tokens are not the model's)\n"
                            "  --fp8-experts (port-only: experts stored as per-row FP8 and run on the MME)\n"
                            "  --embd-gguf FILE (token_embd.weight rows replacing the embedding table)\n"
                            "  --expert-pack DIR (native_experts.txt, optional experts.bin with provenance)\n"
                            "  --mtp-q4 all|proj|head (bounded load-time Q4_0 copies)\n"
                            "  --dump-logits F --dump-final-r F --dump-mixed F --dump-residual F\n"
                            "  --dump-layers F --dump-halves F (scalar token path, --prompt-cache 0); --dump-routing F\n"
                            "  --window-hashes F (per verify window: index position rows fnv1a64 argmaxes...)\n"
                            "  raw layers/halves/final-r with --prefill record only the token-loop suffix\n"
                            "  --vision (--serve takes GENI requests too; SVE1 embeddings + per-cell M-RoPE)\n"
                            "  --batch N / --slots N --serve (2..8 requests in slots; time-sliced through one model)\n"
                            "  --batch-groups G|auto (one card: collapses to 1 group, as upstream warns)\n"
                            "  --expert-profile F --expert-profile-save F --expert-profile-save-every MIN\n"
                            "  --adapt-every N --adapt-swaps N --adapt-decay F --adapt-min-gain F --adapt-async 1|0\n"
                            "  --resident-experts | --resident-cpu-experts | --resident-budget-gib N | --mmap-experts\n"
                            "  --pool-workers N (0..2147483647, default 0 = every physical core; alias --cpu-workers)\n"
                            "  --pool-tasks N (0..4096, default 0 = automatic) --pool-affinity auto|p-cores|all\n"
                            "  --host-core first|last (the pool host worker's core; env STRATA_HOST_CORE)\n"
                            "  --aux-cpus off|auto|LIST (non-pool, non-host threads; env STRATA_AUX_CPUS)\n"
                            "  --expert-cache-per-layer (env STRATA_RESIDENT_PIN=0, STRATA_RESIDENT_HEADROOM_GIB=N)\n"
                            "  --control-vector FILE --control-vector-scaled FILE:SCALE[,...]\n"
                            "  --control-vector-layer-range A B --cvec-mode add|project --cvec-dir per-layer|single:L\n"
                            "Native verify reuses full-accept state and replays a shorter accepted prefix; speed is unmeasured.\n"
                            "--plan / --plan-json do not initialize Synapse or acquire a device. Generation does.\n";return 0;
            }
            if(key=="--native") path=value();
            else if(key=="--serve") serve=true;
            // Upstream :2104/:3197-3215: --vision allocates the mrope table (max_context + 64 cells, identity
            // until a GENI request) and is what lets --serve accept GENI at all.
            else if(key=="--vision") options.vision=true;
            else if(key=="--plan") plan_only=true;
            else if(key=="--plan-json") {plan_only=true;plan_json=true;}
            else if(key=="--vram-elastic") elastic=true;
            else if(key=="--suffix-draft") generation_options.suffix_min_match=number<uint32_t>(value());
            else if(key=="--spec") {const auto rows=number<uint32_t>(value());if(!rows || rows>8) throw std::invalid_argument("--spec expects 1..8");generation_options.max_draft=rows-1;spec_given=true;}
            else if(key=="--spec-prob") generation_options.verify_mode=VerifyMode::Probabilistic;
            else if(key=="--verify-window") {options.memory.verify_rows=number<uint32_t>(value());if(options.memory.verify_rows>7) throw std::invalid_argument("--verify-window expects0..7");}
            else if(key=="--prompt-cache") {generation_options.cache_slots=number<uint32_t>(value());prompt_cache_given=true;}
            else if(key=="--prompt-cache-every") generation_options.cache_every=number<uint32_t>(value());
            else if(key=="--prefill") {
                const auto rows=value();options.prefill_auto=rows=="auto";
                generation_options.prefill_chunk=options.prefill_auto?0:number<uint32_t>(rows);
            }
            else if(key=="--prefill-reference") prefill_reference=true;
            else if(key=="--short-read") generation_options.short_read=number<uint32_t>(value());
            else if(key=="--prompt-cache-tail") generation_options.cache_tail=true;
            else if(key=="--prompt-cache-root") generation_options.boundaries.root_min_tokens=number<uint32_t>(value());
            else if(key=="--turn-token") generation_options.boundaries.turn_token=number<int32_t>(value());
            else if(key=="--tail-role-token") generation_options.boundaries.tail_role_token=number<int32_t>(value());
            else if(key=="--cache-message-boundary") generation_options.boundaries.message_checkpoint=true;
            // Upstream's --conversation-cache-min-tokens (declared :639, parsed in the same bounded-integer group
            // as the other cache floors at :2107-2124 with INT32_MAX as its limit, law conversation_cache.hpp:206).
            // Default 0 parks every conversation, so this is opt-in; the refusal sentence is upstream's.
            else if(key=="--conversation-cache-min-tokens") {
                const auto limit=number<int64_t>(value());
                if(limit<0 || limit>int64_t(INT32_MAX))
                    throw std::invalid_argument("--conversation-cache-min-tokens needs a nonnegative integer within range");
                generation_options.cache_min_tokens=limit;
            }
            // Upstream's stop flags (:2097, :2187-2191). --eos-ids parses through upstream's own list law and
            // arms the loop the way upstream's parser does.
            else if(key=="--stop-eos") stop_eos=true;
            else if(key=="--eos-ids") {eos_ids=parse_token_ids(value(),"--eos-ids");eos_ids_given=true;}
            else if(key=="--conversation-cache-min-free-mib") generation_options.cache_min_free=checked_product(number<uint64_t>(value()),1ull<<20);
            else if(key=="--memory-limit-mib") strata::core::set_memory_limit_bytes(checked_product(number<uint64_t>(value()),1ull<<20));
            else if(key=="--session-min-free-mib") options.session_min_free=checked_product(number<uint64_t>(value()),1ull<<20);
            else if(key=="--control-vector") options.control_vector.files.emplace_back(value(),1.0f);
            else if(key=="--control-vector-scaled") {
                const auto list=value();size_t start=0;
                do {
                    const auto end=list.find(',',start);const auto entry=list.substr(start,end==std::string::npos?end:end-start);const auto colon=entry.rfind(':');
                    if(colon==std::string::npos || !colon) throw std::invalid_argument("--control-vector-scaled expects FILE:SCALE");
                    options.control_vector.files.emplace_back(entry.substr(0,colon),number<float>(entry.substr(colon+1)));
                    if(end==std::string::npos) break;
                    start=end+1;if(start==list.size()) throw std::invalid_argument("empty scaled control vector entry");
                } while(true);
            }
            else if(key=="--control-vector-layer-range") {options.control_vector.first=number<int>(value());options.control_vector.last=number<int>(value());}
            else if(key=="--cvec-mode") {const auto mode=value();if(mode!="add" && mode!="project") throw std::invalid_argument("--cvec-mode expects add or project");options.control_vector.mode=mode=="add"?ControlVectorMode::Add:ControlVectorMode::Project;}
            else if(key=="--cvec-dir") {const auto dir=value();if(dir=="per-layer") options.control_vector.single=-1;else if(dir.starts_with("single:")) options.control_vector.single=number<int>(dir.substr(7));else throw std::invalid_argument("--cvec-dir expects per-layer or single:L");}
            else if(key=="--mtp") options.mtp_runtime=value();
            else if(key=="--mtp-draft-vocab") options.mtp_draft_vocab=value();
            else if(key=="--session-save-reclaim") session_save_reclaim=true;   // upstream defaults it off: the admission is opt-in
            else if(key=="--ple-gguf") {options.ple_gguf=value();if(options.ple_gguf.empty()) throw std::invalid_argument("--ple-gguf expects a file");}
            else if(key=="--no-ple") options.no_ple=true;
            else if(key=="--fp8-experts") {options.fp8_experts=true;options.memory.bf16_head=true;}
        else if(key=="--ple-io") options.ple_io=resolve_ple_io(value(),options.ple_io);
        else if(key=="--ple-row-cache") options.ple_io=resolve_ple_row_cache(number<int64_t>(value()),options.ple_io);
        else if(key=="--ple-inflight") options.ple_io=resolve_ple_inflight(number<int64_t>(value()),options.ple_io);
        else if(key=="--ple-delay-us") options.ple_delay_us=resolve_ple_delay_us(number<double>(value()));
            // Upstream: src/program/generate.cpp:397 field, :1972 parse, :3032-3037 load+install. The row law
            // is strata/hpu/embedding_route.hpp; a file whose rows equal the vocabulary replaces every
            // lookup (upstream's only shape), a shorter one claims sequence positions.
            else if(key=="--embd-gguf") {options.embedding_gguf=value();if(options.embedding_gguf.empty()) throw std::invalid_argument("--embd-gguf expects a file");}
            else if(key=="--expert-pack") {options.expert_pack=value();if(options.expert_pack.empty()) throw std::invalid_argument("--expert-pack expects a directory");}
            else if(key=="--mtp-q4") {const auto mode=value();if(mode!="all" && mode!="proj" && mode!="head") throw std::invalid_argument("--mtp-q4 takes all, proj or head");options.mtp.q4_projections=mode!="head";options.mtp.q4_head=mode!="proj";}
            else if(key=="--mtp-hnorm") {const auto kind=value();if(kind!="pooled" && kind!="stream") throw std::invalid_argument("--mtp-hnorm expects pooled or stream");options.mtp.stream_norm=kind=="stream";}
            else if(key=="--mtp-max-t") {mtp_max_t=number<uint32_t>(value());if(mtp_max_t>8) throw std::invalid_argument("--mtp-max-t expects 0..8");}
            else if(key=="--spec-min-p") options.mtp_min_p=number<float>(value());
            else if(key=="--coupled-draft") options.mtp_sampling.coupled=true;
            else if(key=="--mtp-expert-cache-mib") options.mtp.expert_cache=checked_product(number<uint64_t>(value()),1ull<<20);
            else if(key=="--mtp-window") options.mtp.window=number<uint32_t>(value());
            else if(key=="--lookup-chain") generation_options.lookup_chain=number<uint32_t>(value());
            else if(key=="--lookup-chain-min") generation_options.lookup_chain_min=number<uint32_t>(value());
            else if(key=="--adaptive-drafts") generation_options.adaptive_drafts=true;
            else if(key=="--prefill-until") generation_options.prefill_until=number<uint32_t>(value());
            else if(key=="--tokens") {if(have_tokens) throw std::invalid_argument("supply one token input only");
                tokens=value();have_tokens=true;}
            // Upstream's --tokens-file (src/program/generate.cpp:1902-1911): the file's text goes through
            // the SAME token parser as --tokens, so the format is identical, and the same three failures
            // are refused up front - a second token input, an unopenable path, and a file that cannot be
            // read or contains a NUL. A malformed LIST still reports as a request parse error rather than
            // upstream's "malformed token file" text, which is this backend's one wording difference.
            else if(key=="--tokens-file") {
                if(have_tokens) throw std::invalid_argument("supply one token input only");
                std::ifstream input(value());
                if(!input) throw std::invalid_argument("cannot open token file");
                const std::string text((std::istreambuf_iterator<char>(input)),std::istreambuf_iterator<char>());
                if(input.bad()) throw std::invalid_argument("cannot read token file");
                if(text.find(char(0))!=std::string::npos) throw std::invalid_argument("malformed token file: embedded NUL");
                tokens=text;have_tokens=true;
            }
            else if(key=="--max-new") max_new=number<uint32_t>(value());
            else if(key=="--max-context") options.memory.context=number<uint32_t>(value());
            else if(key=="--module") options.module_id=number<uint32_t>(value());
            else if(key=="--expert-cache-mib") options.memory.expert_cache=checked_product(number<uint64_t>(value()),1ull<<20);
            else if(key=="--hbm-limit-mib") options.memory.hbm_limit=checked_product(number<uint64_t>(value()),1ull<<20);
            else if(key=="--pinned-limit-mib") options.memory.pinned_limit=checked_product(number<uint64_t>(value()),1ull<<20);
            else if(key=="--host-limit-mib") options.memory.host_limit=checked_product(number<uint64_t>(value()),1ull<<20);
            // Upstream declares these at generate.cpp:405-407, parses them at 1950-1953 (the last of
            // --kv-grow / --no-kv-grow wins) and lets the environment beat the flag at 4394-4397.
            // The format only changes the KV cell's bytes and its reader; the planner prices the selection
            // (model_plan.cpp) and QsaState stores it. k8v4 is the hybrid pair - unrotated INT8 K plus rotated
            // Q4_0 V (src/core/layer.cpp:703) - and its MTP drafter runs plain INT8, which kv_drafter_mode says.
            else if(key=="--kv") options.memory.kv_mode=resolve_kv_mode(value()).mode;
        else if(key=="--kv-mib") generation_options.kv.cap_bytes=checked_product(number<uint64_t>(value()),1ull<<20);
            else if(key=="--kv-grow") generation_options.kv.grow=true;
            else if(key=="--no-kv-grow") generation_options.kv.grow=false;
            else if(key=="--kv-resident") generation_options.kv.resident_cells=number<int64_t>(value());
            else if(key=="--spec-oracle") spec_knobs.oracle_path=value();
            else if(key=="--spec-follow") spec_knobs.follow_path=value();
            else if(key=="--spec-corrupt") spec_knobs.corrupt=number<int64_t>(value());
            // Upstream's per-position diagnostic dumps (src/program/generate.cpp:347/408-425 declarations,
            // :1930-1942 parse). Paths only: the record laws live in strata/hpu/dump_sinks.hpp, which is the
            // byte contract. --logits-stride's own law (">1 requires --max-new 1 and --dump-logits", :2631)
            // is enforced by the same module before a byte is written.
            else if(key=="--dump-logits") dump_options.logits=value();
            else if(key=="--dump-layers") dump_options.layers=value();
            else if(key=="--dump-halves") dump_options.halves=value();
            else if(key=="--dump-mixed") dump_options.mixed=value();
            else if(key=="--dump-residual") dump_options.residual=value();
            else if(key=="--dump-final-r") dump_options.final_r=value();
            else if(key=="--dump-routing") dump_options.routing=value();
            // Upstream's --window-hashes (declared :586, parsed :2088, opened :13723): one line per verify
            // window with its FNV-1a over the window's final residual rows and its argmaxes, so two builds that
            // do the same arithmetic must write the same bytes. strata/hpu/window_hashes.hpp is that record law.
            else if(key=="--window-hashes") generation_options.window_hashes=value();
        else if(key=="--check-logits") generation_options.check_logits=true;
        else if(key=="--stats") stats_report=true;
            else if(key=="--logits-stride") logits_stride=number<int64_t>(value());
            // Upstream: `--batch N` and its alias `--slots N` (:2075-2076), and `--batch-groups G|auto`
            // (:2078-2082). On one card the group count collapses to 1 with upstream's own warning
            // (:4549-4553). --batch-mtp adds one proposal to each serialized slot.
            else if(key=="--layer-split") layer_split_spec=value();
            else if(key=="--split-device") split_device_spec=value();
            else if(key=="--batch") batch_options.batch=number<int>(value());
            else if(key=="--slots") batch_options.batch=number<int>(value());
            else if(key=="--batch-mtp") batch_options.batch_mtp=true;
            else if(key=="--batch-groups") {
                const auto groups=value();batch_options.groups_set=true;
                if(groups=="auto") {batch_options.groups_auto=true;batch_options.groups=1;}
                else batch_options.groups=number<int>(groups);
            }
            // Upstream's expert-cache profiling/adaptation knobs (src/program/generate.cpp:2092-2095, 2192,
            // 2201-2204 into the fields at 501-506, 596-606, 628). Defaults stay off/upstream's numbers;
            // --adapt-every/--adapt-swaps arm the tier, which is what upstream's two gates also do.
            else if(key=="--expert-profile") options.expert_profile.profile=value();
            else if(key=="--expert-profile-save") options.expert_profile.save=value();
            else if(key=="--expert-profile-save-every") options.expert_profile.save_every_minutes=number<double>(value());
            else if(key=="--adapt-every") {options.expert_profile.adapt_every=number<int>(value());options.expert_profile.adapt=true;}
            else if(key=="--adapt-swaps") {options.expert_profile.adapt_swaps=number<int>(value());options.expert_profile.adapt=true;}
            else if(key=="--adapt-decay") options.expert_profile.adapt_decay=number<float>(value());
            else if(key=="--adapt-min-gain") options.expert_profile.adapt_min_gain=number<float>(value());
            else if(key=="--adapt-async") options.expert_profile.adapt_async=number<int>(value())!=0;
            // Upstream's expert-cache residency controls (src/program/generate.cpp:2194 --expert-cache-per-layer,
            // :2206 --mmap-experts, :2208 --resident-cpu-experts, :2209-2218 --resident-experts,
            // :2219-2229 --resident-budget-gib). The flag handlers, their combination refusals (:2409-2424) and
            // the --kv-grow exclusion (:4399) are transcribed in strata/hpu/expert_residency.hpp; the model
            // validates the combination before a device exists, so --plan refuses exactly as a run would.
            else if(key=="--expert-cache-per-layer") options.residency.set_expert_cache_per_layer();
            else if(key=="--mmap-experts") options.residency.mmap_experts=true;
            else if(key=="--resident-cpu-experts") options.residency.set_resident_cpu_experts();
            else if(key=="--resident-experts") options.residency.set_resident_experts(resident_pin_zero,has_resident_headroom,resident_headroom_gib);
            else if(key=="--resident-budget-gib") {
                // generate.cpp:2221: N <= 0 is a parse refusal (return 2), not a post-parse validation message.
                if(!options.residency.set_resident_budget_gib(number<double>(value()),resident_pin_zero,has_resident_headroom,resident_headroom_gib)) {
                    std::cerr<<resident_budget_refusal()<<'\n';return 2;
                }
            }
            else if(key=="--vram-reserve-mib") options.memory.driver_margin=checked_product(number<uint64_t>(value()),1ull<<20);
            else if(key=="--snapshot-budget-mib") options.memory.snapshot_budget=checked_product(number<uint64_t>(value()),1ull<<20);
            else if(key=="--cpu-cache-mib") options.memory.cpu_cache=checked_product(number<uint64_t>(value()),1ull<<20);
            else if(key=="--pool-workers") {
                // Upstream parses --pool-workers with atoi (:1980) and rejects a negative result in the
                // shared post-parse check (:2693-2701: "invalid sampling or resource parameter", exit 2).
                // Parse strictly as a signed integer so -1 reaches that same refusal, but do not copy
                // atoi's reading of junk as 0. Counts above INT_MAX would wrap negative in ExpertPool's
                // int parameter and silently mean automatic, so they are refused here too.
                const auto workers=number<int64_t>(value());
                if(workers<0) {
                    std::cerr<<"strata generate: invalid sampling or resource parameter\n";
                    return 2;
                }
                if(workers>int64_t(INT_MAX)) {
                    std::cerr<<"strata-hpu: --pool-workers expects an integer in 0..2147483647 (0 = every physical core)\n";
                    return 2;
                }
                options.cpu_workers=uint32_t(workers);
            }
            else if(key=="--cpu-workers") {
                // Port alias for the field above; same 0 = every physical core domain.
                const auto workers=number<int64_t>(value());
                if(workers<0 || workers>int64_t(INT_MAX)) {
                    std::cerr<<"strata-hpu: --cpu-workers expects an integer in 0..2147483647 (0 = every physical core)\n";
                    return 2;
                }
                options.cpu_workers=uint32_t(workers);
            }
            else if(key=="--pool-tasks") {
                // generate.cpp:1981-1989: strtol, and the same sentence for malformed text, a negative
                // count, or a count above kMaxTasks, with exit 2.
                const std::string text=value();
                int64_t tasks=0;
                const auto [end,error]=std::from_chars(text.data(),text.data()+text.size(),tasks);
                if(error!=std::errc{} || end!=text.data()+text.size() || tasks<0 ||
                   tasks>int64_t(strata::kernels::cpu::ExpertPool::kMaxTasks)) {
                    std::cerr<<"strata generate: --pool-tasks expects an integer in 0..4096 (0 = automatic)\n";
                    return 2;
                }
                options.cpu_tasks=uint32_t(tasks);
            }
            else if(key=="--pool-affinity") {
                const std::string text=value();
                if(!parse_pool_affinity(text,options.cpu_affinity.affinity)) {
                    std::cerr<<"strata-hpu: unknown --pool-affinity value '"<<text
                             <<"' (expected auto, p-cores, or all)\n";
                    return 2;
                }
            }
            else if(key=="--host-core") {
                // generate.cpp:1991-1997: exactly first|last, refused with upstream's wording.
                const std::string text=value();
                if(!parse_host_core(text,options.cpu_affinity.host_core)) {
                    std::cerr<<"strata-hpu: unknown --host-core value '"<<text<<"' (expected first or last)\n";
                    return 2;
                }
                host_core_given=true;
            }
            else if(key=="--aux-cpus") {
                // generate.cpp:1998-2008: the spec is checked at parse time with the original law's own refusal.
                const std::string text=value();
                std::string error;
                if(!parse_aux_cpus(text,aux_cpus_spec,error)) {
                    std::cerr<<"strata-hpu: "<<error<<'\n';
                    return 2;
                }
            }
            else if(key=="--hpu-load-budget") options.hpu_load_budget=number<uint32_t>(value());
            else if(key=="--position-base") options.position_base=number<uint64_t>(value());
            else if(key=="--rope-scaling") {
                const auto type=value();using Type=strata::kernels::RopeScalingType;
                if(type=="none") options.rope.type=Type::None;
                else if(type=="linear") options.rope.type=Type::Linear;
                else if(type=="yarn") options.rope.type=Type::YaRN;
                else throw std::invalid_argument("--rope-scaling expects none, linear or yarn");
            }
            else if(key=="--rope-scale") options.rope.factor=number<double>(value());
            else if(key=="--rope-freq-base") options.rope.freq_base=number<double>(value());
            else if(key=="--rope-freq-scale") options.rope.freq_scale=number<double>(value());
            else if(key=="--yarn-orig-ctx") options.rope.orig_ctx=number<double>(value());
            else if(key=="--yarn-ext-factor") options.rope.ext_factor=number<double>(value());
            else if(key=="--yarn-attn-factor") options.rope.attn_factor=number<double>(value());
            else if(key=="--yarn-beta-fast") options.rope.beta_fast=number<double>(value());
            else if(key=="--yarn-beta-slow") options.rope.beta_slow=number<double>(value());
            else if(key=="--sampling") keys+=" "+value();
            else throw std::invalid_argument("unknown HPU option: "+key);
        }
        if(path.empty() || (serve && !tokens.empty()) || (!serve && !plan_only && tokens.empty())) throw std::invalid_argument("use --native and --plan, --serve, or --tokens (see --help)");
        // generate.cpp:2635-2637: the ablation has nothing to feed the table, so the combination is refused.
        if(options.no_ple && !options.ple_gguf.empty()) {
            std::cerr<<"strata generate: --no-ple and --ple-gguf are mutually exclusive\n";
            return 2;
        }
        // Upstream consults STRATA_HOST_CORE only when --host-core was not given, and only "last" changes
        // anything (generate.cpp:2293-2299); an unknown value there is silently first, as upstream does.
        if(!host_core_given)
            if(const char* env=std::getenv("STRATA_HOST_CORE"); env && std::string(env)=="last")
                options.cpu_affinity.host_core=HostCore::Last;
        // generate.cpp:2299-2308: --aux-cpus when given, else STRATA_AUX_CPUS. The allowed CPUs are read here,
        // before the pool or the Synapse runtime pins any thread; the plan waits for the workers' CPUs.
        {
            std::string error;
            if(!start_aux_cpus(aux_cpus_spec,error)) {
                std::cerr<<"strata-hpu: STRATA_AUX_CPUS: "<<error<<'\n';
                return 2;
            }
        }
        // Upstream's global expert-pool writer survives every serve/slot call.
        // Other dumps retain their one-run position/head lifetime for now.
        if(serve && !dump_options.none() && !dump_options.routing_only())
            throw std::invalid_argument("only --dump-routing currently has a resident --serve writer; other --dump-* need one run's positions");
        // Preflight must price the same diagnostic buffers as Model's actual
        // constructor, including --plan-json and prefill-auto resolution.
        options.capture_layer_ladder=!dump_options.layers.empty();
        options.capture_routing=!dump_options.routing.empty();
        options.capture_halves=!dump_options.halves.empty();
        // Upstream's stop law separates two decisions: WHICH ids end a turn (:612, or --eos-ids) and whether the
        // ordinary token loop tests them at all (:2097, :2190, and serving-without-tokens at :2528-2532). Batch
        // slots test them either way (:10077, :12424), which is why only Generation consults `stop_eos` here.
        options.eos_ids=eos_ids;
        generation_options.stop_eos=stop_tokens_armed(stop_eos,eos_ids_given,serve,have_tokens);
        // Upstream's one --spec-min-p value drives BOTH the drafter's extension loop (:13755, this port's
        // ModelOptions::mtp_min_p read in model.cpp:695) and the verify-window truncation (:12566-12568, which
        // Generation reads here). A request may override the window gate with `spec_min_p=` (:10697).
        generation_options.spec_min_p=options.mtp_min_p;
        // ---- --batch: resolve upstream's gates FIRST (the warnings are its own), then decide the serve path.
        batch_options.serve=serve;
        if(const auto* enabled=std::getenv("STRATA_BATCH_MTP");enabled && *enabled && *enabled!='0') batch_options.batch_mtp=true;
        // Upstream loads the MTP drafter only when a spec window was asked for: `if (o.spec < 2) { ...; o.mtp.clear(); }`
        // with the operator-facing reason at src/program/generate.cpp:4613, and its default --spec is 0 on every
        // non-Metal build (:564). This backend keeps --spec absent meaning "the verification window stays at its cap",
        // so the same default combination (--mtp without --spec) used to draft here while upstream discarded the
        // drafter; the gate below reproduces upstream's decision and says why, instead of leaving the flag pair to
        // behave differently per backend.
        // Discarding must not swallow input validation: the Linux run caught that clearing the path before the
        // Model was constructed made a nonexistent --mtp runtime accepted (hpu/tests/prepare.py) and dropped the
        // drafter's terms out of the plan (hpu/tests/model_plan.py), so the path is checked first and the plan
        // tests ask for the drafter the way the gate now requires.
        if(!options.mtp_runtime.empty() && !std::filesystem::exists(options.mtp_runtime))
            throw std::invalid_argument("MTP runtime does not exist: "+options.mtp_runtime);
        if(!options.mtp_runtime.empty() && (!spec_given || generation_options.max_draft<1)) {
            std::cerr<<"strata-hpu: --mtp is ignored without --spec T (T >= 2): upstream discards the drafter the same way"
                     <<" (src/program/generate.cpp:4613); pass --spec to run MTP\n";
            options.mtp_runtime.clear();
        }
        batch_options.mtp=!options.mtp_runtime.empty();
        batch_options.spec=int(generation_options.max_draft)+1; // --spec T is kept as max_draft = T-1
        // Upstream validates --layer-split / --split-device before any device work and only then lets the batch
        // gates see the split (src/program/generate.cpp:2355-2406). Here the visible device count is one Gaudi2
        // card, so the law's own answers are the whole story: "auto" clears with its no-split line, a single
        // boundary on device 0 is the same-GPU split (which conversation parking must refuse, because the parked
        // images live on that one card), and any multi-stage split is fatal because upstream demands one distinct
        // device per boundary and there is no device 1..0 to hand out. Before this, both flags were only ever
        // "unknown HPU option", which hid the rule upstream states.
        LayerSplitRequest split_request;
        split_request.spec=layer_split_spec;split_request.devices=split_device_spec;split_request.serve=serve;
        split_request.device_count=1;
        split_request.parking=default_prompt_cache_slots(prompt_cache_given,options.memory.snapshot_budget,
                                                          generation_options.cache_slots)>0;
        const auto split=resolve_layer_split(split_request);
        // Upstream prints once and exits; the thrown message is that print, so a fatal answer is not said twice.
        if(!split.fatal && !split.message.empty()) std::cerr<<"strata-hpu: "<<split.message<<'\n';
        if(split.fatal) throw std::invalid_argument(split.message);
        batch_options.has_layer_split=split.accepted;batch_options.split_same=split.same_gpu;
        // Upstream's per-stage park loop counts stages as `stages.size()` (generate.cpp:8244). The accepted
        // boundary list is what that number comes from, and it is handed to the generation seam so a park is
        // refused by name if a multi-stage session ever reaches this single-card backend.
        const uint32_t parking_stage_count=split.accepted?uint32_t(split.at.size())+1u:0u;
        const auto batch=resolve_batch(batch_options);
        for(const auto& warning:batch.warnings) std::cerr<<"strata-hpu: "<<warning<<'\n';
        if(batch.slots>0) {
            // Slot MTP uses one proposal and a native [pending,proposal] pass.
            // The independent solo draft sources/diagnostic knobs still need
            // their own per-slot histories; do not silently drop those flags.
            const bool drafts=generation_options.suffix_min_match!=0 ||
                generation_options.lookup_chain!=0 || (options.memory.verify_rows!=0 && !batch.batch_mtp) ||
                !spec_knobs.oracle_path.empty() || !spec_knobs.follow_path.empty() || spec_knobs.corrupt!=0;
            if(drafts) throw std::invalid_argument("--batch: suffix/lookup/oracle/follow/corrupt and solo --verify-window are not wired into slots");
            // Reserve shared native scratch BEFORE preflight/device creation.
            if(batch.batch_mtp) options.memory.verify_rows=std::max(2u,options.memory.verify_rows);
            // Every parked slot holds one Model snapshot, and snapshots share the bounded snapshot budget.
            if(!options.memory.snapshot_budget)
                throw std::invalid_argument("--batch parks one snapshot per slot, so it needs a nonzero --snapshot-budget-mib");
        }
        options.memory.batch_slots=uint32_t(batch.slots);
        generation_options.cache_bytes=options.memory.snapshot_budget;
        // Upstream's --prompt-cache default is 6 checkpoints (src/program/generate.cpp:632). This backend can
        // only honour it once --snapshot-budget-mib granted the RAM, so the default is applied here rather than
        // in the options struct, and an explicit --prompt-cache (including 0) is never overridden.
        generation_options.cache_slots=default_prompt_cache_slots(prompt_cache_given,options.memory.snapshot_budget,
            generation_options.cache_slots);
        if(const auto* grow=std::getenv("STRATA_KV_GROW");grow&&*grow) generation_options.kv.grow=*grow!='0';
        // Upstream's opt-in INT8 rotation (generate.cpp:2554-2556): with STRATA_KV_ROT=1 an int8 state stores
        // FWHT-rotated K/V and rotates queries and the attention output around them. This backend's int8 state is
        // unrotated, so an int8 run that asks for the rotation is refused rather than storing K/V a rotated reader
        // would misread. With fp16 upstream's rotation has no effect; --kv q4_0 rotates by design and k8v4 rotates
        // only V while its K stays unrotated INT8 (layer.cpp:703 sets kv_rot false for the hybrid), so the env is
        // inert in all three.
        if(options.memory.kv_mode==KvMode::Int8)
            if(const auto* rot=std::getenv("STRATA_KV_ROT");rot && rot[0]=='1')
                throw std::invalid_argument("STRATA_KV_ROT=1: the INT8 K/V Hadamard rotation is not implemented; unset it or use --kv fp16 / --kv q4_0");
        // generate.cpp:4399: the elastic K/V arena and the resident CPU expert tier compete for the same RAM,
        // so the resident tier turns --kv-grow off - after the environment override, as upstream orders it.
        if(kv_growth_blocked_by_residency(options.residency)) generation_options.kv.grow=false;
        validate_generation_options(generation_options);
        if(options.memory.verify_rows) generation_options.max_draft=std::min(generation_options.max_draft,options.memory.verify_rows);
        const bool scalar_source=!options.mtp_runtime.empty() || generation_options.suffix_min_match ||
            generation_options.lookup_chain || !spec_knobs.oracle_path.empty();
        if(generation_options.max_draft && scalar_source && !options.memory.verify_rows && !options.memory.snapshot_budget)
            throw std::invalid_argument("scalar candidate verification requires nonzero --snapshot-budget-mib (or a native --verify-window)");
        if(options.prefill_auto && prefill_reference) {
            // Resolve the same fully charged native policy first, then retain
            // only its prompt boundaries while executing the row fallback.
            generation_options.prefill_chunk=std::max(1u,Model::preflight(path,options).prefill_rows);
            options.prefill_auto=false;
        }
        // Measured (round 33): widening this clamp to the resident window works end to end -- PP steps become
        // 256/512 -- but at 256 rows it bought only 1.3% (397.0 -> 402.0 tok/s on a 512-token prefill), because the
        // expert path's cost is per row, not per window. Round 45 raises the cap for the resident chain anyway: the
        // grouped-MME expert path is a per-window constant, so it only pays off above ~512 rows
        // (work/DESIGN_PREFILL_STREAMING_20261024.md §18). A run without resident IQ4 keeps the old 128-row clamp,
        // because model.cpp refuses a wider window without it and a silent clamp is not a refusal.
        const uint32_t window_cap=resident_iq4_enabled()?kResidentWindowRows:128u;
        if(!prefill_reference) options.memory.prefill_rows=std::min(window_cap,generation_options.prefill_chunk);
        if(prefill_qsa32_enabled()) generation_options.canonical_prefill_rows=32;
        if(plan_only) {
            const auto plan=Model::preflight(path,options);
            if(plan_json) {write_plan_json(std::cout,plan);return 0;}
            std::cout<<"dense_bytes="<<plan.dense_bytes<<" hbm_required="<<plan.hbm_required<<" host_required="<<plan.host_required
                     <<" residency_per_layer_experts="<<plan.residency_per_layer_experts
                     <<" residency_admitted="<<plan.residency_admitted<<" no_device_acquired\n";return 0;
        }
        std::optional<GenerationRequest> one_shot;
        if(!serve) {
            const WeightTable table(path);
            one_shot=parse_generation_request("GEN "+std::to_string(max_new)+keys+" "+tokens,uint32_t(table.geometry().n_vocab),options.memory.context);
        }
        // Refuse unsupported position-dump coverage before Model acquires a
        // device. Planning alone still reports the requested resource policy.
        if(dump_options.token_positions() && one_shot) {
            if(generation_options.cache_slots) throw std::invalid_argument("per-position dumps cannot resume a prompt cache: use --prompt-cache 0");
            if(!dump_options.logits.empty() && (generation_options.prefill_chunk || options.prefill_auto) && one_shot->prompt.size()>1)
                throw std::invalid_argument("per-position dumps record the token path only: drop --prefill");
            const bool drafts=generation_options.max_draft && (!options.mtp_runtime.empty() || generation_options.suffix_min_match || generation_options.lookup_chain || !spec_knobs.oracle_path.empty() || !spec_knobs.follow_path.empty() || spec_knobs.corrupt);
            if(drafts) throw std::invalid_argument("per-position dumps record the token path only: drop speculative drafts");
        }
        // generate.cpp:3433-3436: the ablation announces itself once, before any token can be mistaken for the
        // model's own output. --plan stays quiet: it acquires no device and produces no tokens.
        if(options.no_ple) std::cerr<<ple_off_notice();
        // The two dumps that need an in-model capture: --dump-layers keeps this position's residual ladder and
        // --dump-routing keeps its routed ids/weights. Both flags are off by default and cost nothing then.
        Model model(path,options);
        // generate.cpp:5071-5076: the plan needs the pool's workers. The CPU-expert tier builds that pool in the
        // Model above; a build without the tier has no workers to reserve, so every CPU but the (unknown) host is
        // spare. The sweep here also catches threads the Synapse runtime created during Model construction.
        std::string aux_cpus_note;
#if defined(STRATA_HPU_CPU_EXPERTS) && STRATA_HPU_CPU_EXPERTS
        {
            const CpuPlacement placement=CpuExperts::placement(options.cpu_workers,options.cpu_affinity);
            aux_cpus_note=plan_aux_cpus(placement.host_core,placement.worker_cores);
        }
#else
        aux_cpus_note=plan_aux_cpus(-1,{});
#endif
        std::cerr<<aux_cpus_note<<'\n';
        if(options.prefill_auto) generation_options.prefill_chunk=std::max(1u,model.plan().prefill_rows);
        // generate.cpp:5433/5444 report what the residency policy actually installed ("each layer owns N
        // slots"). The same numbers the no-card plan printed, now from the live cache.
        if(options.residency.expert_cache_per_layer || options.residency.resident_cpu_experts) {
            const auto installed=model.residency_report();
            std::cerr<<"strata-hpu: resident experts per_layer="<<(installed.per_layer_honored?
                "each layer owns "+std::to_string(installed.per_layer_experts)+" experts":"uniform cache (split could not fit)")
                <<" admitted="<<installed.admitted<<" pinned="<<installed.pinned<<'\n';
        }
        SlotSession* shared_slots=nullptr;
        GenerationOperations operations;
        operations.select_prefix=[&](std::span<const int32_t> prompt,uint64_t maximum,bool control,
                                     std::span<const RequestImage> images)->std::optional<GenerationPrefix> {
            return shared_slots?shared_slots->select_prefix(prompt,maximum,control,images):std::nullopt;
        };
        // Upstream's #2228 serve line (:10979-10982): a deeper state has this prompt's tokens but not its
        // pictures, so it cannot be reused and the request falls back. The body is upstream's sentence with
        // this backend's process prefix; the line goes to stderr exactly once per request.
        operations.picture_mismatch=[](uint64_t tokens,uint64_t picture,int64_t start) {
            std::cerr<<"strata-hpu: conversation cache: the state at "<<tokens<<" tokens has this prompt's tokens "
                       "but picture "<<picture<<" (at token "<<start<<") is not the same as when it was saved "
                       "(its embeddings changed): it cannot be reused (#2228)\n";
        };
        operations.reset=[&]{model.reset();};
        // --vision: the request's M-RoPE table and image rows. Generation and SlotSession call this for every
        // request; a request without images restores the identity table, upstream's text branch at :10741-10743.
        std::function<void(const GenerationRequest&)> install_vision;
        if(options.vision) install_vision=[&](const GenerationRequest& request) {
            Model::VisionSelection selection;
            selection.embeddings=request.embeddings;
            selection.rows=request.image_rows;
            selection.positions=request.positions;
            model.set_vision(std::move(selection));
        };
        operations.set_vision=install_vision;
        if(!options.control_vector.files.empty()) operations.control_vector=[&](bool enabled) {model.set_control_vector(enabled);};
        operations.decode=[&](int32_t token,const std::function<bool()>& cancelled) {
            try {return model.decode(token,cancelled);}
            catch(const std::exception& error) {if(!model.healthy()) throw SessionFatal(std::string("model state could not be rolled back: ")+error.what());throw;}
        };
        operations.prefill=[&](std::span<const int32_t> tokens,const PromptCommit& commit,const std::function<bool()>& cancelled) {
            try {model.prefill_window(tokens,commit,cancelled);}
            catch(const std::exception& error) {if(!model.healthy()) throw SessionFatal(std::string("prefill state could not be rolled back: ")+error.what());throw;}
        };
        operations.prefill_next=[&](std::span<const int32_t> tokens) {model.prefill_next(tokens);};
        if(options.capture_routing) operations.prefill_routing=[&](std::span<const int32_t> tokens,const PromptCommit& commit,const std::function<bool()>& cancelled,const RoutingWindowSink& routing) {
            try {model.prefill_window(tokens,commit,cancelled,routing);}
            catch(const std::exception& error) {if(!model.healthy()) throw SessionFatal(std::string("traced prefill state could not be rolled back: ")+error.what());throw;}
        };
        if(!options.mtp_runtime.empty()) operations.draft=[&](const DraftContext& context,uint32_t maximum,const std::function<bool()>& cancelled) {
            try {return model.draft(context,mtp_max_t?std::min(maximum,mtp_max_t-1):maximum,cancelled);}
            catch(const std::exception& error) {if(!model.healthy()) throw SessionFatal(std::string("MTP state could not be rolled back: ")+error.what());throw;}
        };
        if(model.plan().verify_rows) operations.verify=[&](const DraftContext& context,std::span<const DraftRow> drafts,const std::function<bool()>& cancelled) {
            try {return model.verify_candidates(context,drafts,cancelled);}
            catch(const std::exception& error) {if(!model.healthy()) throw SessionFatal(std::string("native verification state could not be rolled back: ")+error.what());throw;}
        };
        if(model.dflash_ready()) operations.dflash=[&](int32_t pending,uint32_t maximum,const DflashSelection& select,const std::function<bool()>& cancelled) {
            try {return model.dflash_window(pending,maximum,select,cancelled);}
            catch(const std::exception& error) {if(!model.healthy()) throw SessionFatal(std::string("DFlash window state could not be rolled back: ")+error.what());throw;}
        };
        operations.verify_routing=options.capture_routing && bool(operations.verify);
        operations.checkpoint_bytes=model.snapshot_size_limit();
        // Price a park by the image it writes; the limit above stays the fallback for seams without a model.
        operations.checkpoint_park_bytes=[&](uint64_t cells){return model.parked_bytes(cells);};
        // The stage count the accepted split produced, handed to the park site so a multi-stage session is
        // refused by name there rather than parked as one image (upstream parks per stage, :8244-8285).
        operations.parking_stages=[=] {return parking_stage_count;};
        operations.checkpoint=[&] {
            auto point=model.snapshot();return std::function<void()>([&,point=std::move(point)] {model.restore(point);});
        };
        // The parked-page reuse seam: what the last park carried over from the retained image, and the way to
        // drop that image when the conversation it described is gone.
        operations.parked_reused_bytes=[&]{return model.parked_reused_bytes();};
        operations.discard_parked=[&]{model.discard_parked();};
        operations.parked_retained_bytes=[&]{return model.parked_retained_bytes();};
        // --dump-final-r / --dump-residual read the post-last-layer residual, --dump-mixed the head input.
        // Both come from the model's committed state, so a dump never re-runs a graph to obtain them.
        operations.residual=[&]{return model.residual();};
        operations.mixed=[&]{return model.mixed();};
        // The in-model capture the ladder and routing dumps consume. Absent hooks are refused at arm time.
        if(!dump_options.layers.empty()) operations.layer_ladder=[&]{return model.layer_ladder();};
        if(!dump_options.routing.empty()) operations.routing=[&]{return model.routing_records();};
        if(!dump_options.halves.empty()) operations.halves=[&]{return model.half_records();};
        if(!dump_options.none() || !generation_options.window_hashes.empty()) {
            const WeightTable table(path);const auto& shape=table.geometry();
            generation_options.dumps=dump_options;
            generation_options.dump_shape.n_vocab=shape.n_vocab;generation_options.dump_shape.n_layers=shape.n_layers;
            generation_options.dump_shape.hc=shape.hc;generation_options.dump_shape.n_embd=shape.n_embd;
            generation_options.dump_shape.n_head=shape.n_head;generation_options.dump_shape.head_dim=shape.head_dim;
            generation_options.dump_shape.n_head_kv=shape.n_head_kv;generation_options.dump_shape.n_expert=shape.n_expert;
            generation_options.dump_shape.stride=logits_stride;
            // Producers this backend wires: the logits come from the token path, the residual/mixed from the
            // model's committed state, the ladder from the in-model per-layer capture and the routing records
            // from each MoE layer's router. Halves read the Model-owned complete
            // scalar frame assembled from actual Layer/QSA capture callbacks.
            generation_options.dump_producers.logits=true;
            generation_options.dump_producers.final_r=!dump_options.final_r.empty();
            generation_options.dump_producers.head_residual=!dump_options.residual.empty();
            generation_options.dump_producers.head_mixed=!dump_options.mixed.empty();
            generation_options.dump_producers.layer_ladder=!dump_options.layers.empty();
            generation_options.dump_producers.routing=!dump_options.routing.empty();
            generation_options.dump_producers.half_capture=!dump_options.halves.empty();
            if(serve && dump_options.routing_only())
                generation_options.resident_routing=std::make_shared<DumpSinks>(dump_options,generation_options.dump_shape,generation_options.dump_producers);
        }
        RoutingWindowSink slot_routing;
        if(generation_options.resident_routing) slot_routing=[sink=generation_options.resident_routing](const RoutingWindowView& view,bool window){sink->routing_view(view,window);};
        Generation generation(model.vocabulary(),options.memory.context,model.stop_tokens(),std::move(operations),generation_options);
        if(!spec_knobs.oracle_path.empty()||!spec_knobs.follow_path.empty()||spec_knobs.corrupt) generation.set_spec_knobs(spec_knobs);
        if(serve) {
            ProtocolControl control;
            // --vision: GENI's SVE1 records must carry the model's n_embd width (upstream compares hdr[4] to
            // g.n_embd at :10758-10762). The table is allocated only when the flag is on.
            control.vision=options.vision;
            if(options.vision) control.vision_width=WeightTable(path).geometry().n_embd;
            control.info="expert_cache_mib="+std::to_string(options.memory.expert_cache/(1ull<<20))+" hbm_limit_mib="+std::to_string(options.memory.hbm_limit/(1ull<<20))+" verify="+(model.plan().verify_rows?std::string("layer-major"):std::string("sequential"))+" verify_rows="+std::to_string(model.plan().verify_rows)+" suffix_draft="+std::to_string(generation_options.suffix_min_match)+" spec_prob="+(generation_options.verify_mode==VerifyMode::Probabilistic?"1":"0")+" prompt_cache="+std::to_string(generation_options.cache_slots)+" prompt_cache_default="+prompt_cache_note(generation_options.cache_slots);
            {
                // The keys the pinned server actually reads from INFO (serve/server.py:839-842 and its
                // info.get(...) sites): context, kv_resident, arena_mib, conversation_cache_mib/slots, engine, and -
                // only while slots are on, as upstream prints them at generate.cpp:9608-9610 - batch_slots,
                // slot_cache and batch_groups. protocol_info_append refuses a value containing whitespace, because the
                // server splits the line on spaces.
                std::vector<ProtocolInfoFact> facts{
                    {"context",std::to_string(options.memory.context)},
                    {"kv",kv_mode_name(options.memory.kv_mode)},
                    {"kv_resident",std::to_string(model.plan().kv_resident_cells)},
                    {"arena_mib",std::to_string(model.plan().kv_arena_bytes/(1ull<<20))},
                    {"conversation_cache_mib",std::to_string(options.memory.snapshot_budget/(1ull<<20))},
                    // Upstream prints its own cache floor keys in INFO (:9594, :9606); the stop keys are this
                    // backend's, because an unarmed run has no EOS stop and an operator must be able to see that.
                    {"conversation_cache_min_tokens",std::to_string(generation_options.cache_min_tokens)},
                    {"stop_eos",generation_options.stop_eos?"1":"0"},
                    {"stop_ids",std::to_string(eos_ids_given?eos_ids.size():0)},
                    {"conversation_cache_slots",std::to_string(generation_options.cache_slots)},
                    {"no_ple",options.no_ple?"1":"0"},
                    {"ple_io",options.ple_io.lock?"ram":options.ple_io.mode==strata::kernels::PleIo::Mmap?"mmap":"direct"},
                    {"ple_ram_locked",model.ple_ram_locked()?"1":"0"},
                    {"ple_delay_us",std::to_string(options.ple_delay_us)},
                    {"adaptive_drafts",generation_options.adaptive_drafts?"1":"0"},
                    {"engine",engine_value()}};
                for(auto& fact:batch_info_facts(uint32_t(batch.slots),uint32_t(batch.groups),
                                                generation_options.cache_slots!=0)) facts.push_back(fact);
                control.info=protocol_info_append(control.info,facts);
            }
            control.info+=" mtp="+std::string(options.mtp_runtime.empty()?"0":"1")+" mtp_dense_bytes="+std::to_string(model.plan().mtp_dense_bytes)+" lookup_chain="+std::to_string(generation_options.lookup_chain);
            control.info+=" prompt_cache_every="+std::to_string(generation_options.cache_every);
            const auto& plan=model.plan();
            control.info+=" prefill="+std::string(plan.window_rows?"layer-major-grouped-moe":(plan.prefill_rows>=2?"resident-verify-window":"sequential-window"))+" prefill_rows="+std::to_string(plan.prefill_rows)+" prefill_chunk="+std::to_string(generation_options.prefill_chunk)+(generation_options.canonical_prefill_rows?" canonical_prefill_rows="+std::to_string(generation_options.canonical_prefill_rows):std::string())+" short_read="+std::to_string(generation_options.short_read)+" prompt_cache_tail="+(generation_options.cache_tail?"1":"0");
            control.info+=" prompt_cache_root="+std::to_string(generation_options.boundaries.root_min_tokens)+" turn_token="+std::to_string(generation_options.boundaries.turn_token)+" tail_role_token="+std::to_string(generation_options.boundaries.tail_role_token)+" message_boundary="+(generation_options.boundaries.message_checkpoint?"1":"0");
            control.info+=" conversation_cache_min_free_mib="+std::to_string(generation_options.cache_min_free>>20)+" memory_limit_mib="+std::to_string(strata::core::memory_limit_bytes()>>20);
            control.info+=" cvec="+model.control_vector_summary();
            control.info+=(session_save_reclaim?" session_save_reclaim=1":" session_save_reclaim=0");
            control.info+=" pool_affinity="+std::string(pool_affinity_name(options.cpu_affinity.affinity))+" host_core="+(options.cpu_affinity.host_core==HostCore::Last?"last":"first");
            control.info+=" session_min_free_mib="+std::to_string(options.session_min_free>>20);
            if(elastic) control.vram=[&](std::optional<uint64_t> reserve){return model.reserve_vram(reserve);};
            control.session=[&](bool save,const std::string& path) {
                // Parked checkpoints share the native snapshot budget. Upstream's SAVE admission
                // (core::session_save_reclaim) frees only the caches a snapshot of this size actually
                // needs, re-measures after each release, and keeps pins plus the checkpoint the restored
                // session would mount; live state is never touched. The blunt release remains the
                // fallback so a save that would have worked before still works.
                if(session_save_reclaim) {
                    if(!generation.reclaim_for_save(model.snapshot_memory_cost(),model.tokens()).enough)
                        generation.release_checkpoints();
                } else generation.release_checkpoints();   // upstream's default path: the blunt release only
                if(save) model.save(path,generation.live_images());
                else {
                    bool loaded=false;
                    try {
                        auto images=model.load(path);loaded=true;
                        generation.adopt(model.tokens(),model.logits(),model.control_vector_enabled(),images);
                    }
                    catch(const std::exception& error) {
                        if(loaded || !model.healthy()) {generation.invalidate();throw SessionFatal(std::string("session restore changed model state before failure: ")+error.what());}
                        throw;
                    }
                }
                std::error_code error;const auto bytes=std::filesystem::file_size(path,error);
                if(error) {if(save) throw StateIoError("state published but file size is unavailable",true);throw SessionFatal("restored state is active but source file size is unavailable");}
                return ProtocolSessionResult{model.size(),bytes};
            };
            if(batch.slots>0) {
                // The slot protocol: BGEN/BSTOP/BADM/BT/BDONE over the same pipe. The INFO line says the schedule
                // is time-sliced, so a client cannot mistake the slot count for batched throughput.
                SlotSessionOperations slot_op;
                slot_op.before_foreign=[&]{generation.suspend_live();};
                slot_op.select_prefix=[&](std::span<const int32_t> prompt,uint64_t maximum,bool control,std::span<const RequestImage> images){return generation.select_prefix(prompt,maximum,control,images);};
                slot_op.logits=[&]{return model.logits();};
                slot_op.import_prefixes=[&](const GenerationRead& read){generation.import_prefixes(read);};
                slot_op.reset=[&]{model.reset();};
                slot_op.set_vision=install_vision;
                slot_op.prefill=[&](std::span<const int32_t> ids){
                    if(!slot_routing) return model.prefill(ids);
                    std::vector<float> logits;
                    model.prefill_window(ids,[&](uint32_t,std::span<const float> values){logits.assign(values.begin(),values.end());},{},slot_routing,true);
                    return logits;
                };
                slot_op.prefill_window=[&](std::span<const int32_t> ids){
                    std::vector<float> logits;
                    model.prefill_window(ids,[&](uint32_t,std::span<const float> values){logits.assign(values.begin(),values.end());},{},slot_routing);
                    return logits;
                };
                slot_op.prefill_cancellable=[&](std::span<const int32_t> ids,bool window,const PromptCommit& commit,const std::function<bool()>& stop) {
                    if(slot_routing) model.prefill_window(ids,commit,stop,slot_routing,!window);
                    else if(window) model.prefill_window(ids,commit,stop);
                    else for(uint32_t at=0;at<ids.size();++at) {const auto values=model.prefill(ids.subspan(at,1),stop);commit(at+1,values);}
                };
                slot_op.decode=[&](int32_t id){return model.decode(id,{},slot_routing);};
                slot_op.checkpoint=[&]{auto point=model.snapshot();return std::function<void()>([&,point=std::move(point)]{model.restore(point);});};
                if(!options.control_vector.files.empty()) slot_op.control_vector=[&](bool enabled){model.set_control_vector(enabled);};
                slot_op.discard_retained=[&]{model.discard_parked();};
                if(batch.batch_mtp) slot_op.speculate=[&](int32_t pending,std::span<const int32_t> history,const strata::kernels::SamplerParams& sampling) {
                    try {return model.slot_window(pending,history,sampling,slot_routing);}
                    catch(const std::exception& error) {if(!model.healthy()) throw SessionFatal(std::string("slot MTP rollback failed: ")+error.what());throw;}
                };
                control.info+=" batch_mtp="+std::to_string(batch.batch_mtp?1:0)+" batch_target_rows="+(batch.batch_mtp?std::string("2"):std::string("1"))+" disk_batch=idle-only session_target=main";
                SlotCheckpointOptions slot_checks;
                slot_checks.checkpoints=generation_options.cache_slots;
                slot_checks.boundaries=generation_options.boundaries;slot_checks.interval=generation_options.cache_every;
                slot_checks.prefill_chunk=generation_options.prefill_chunk;slot_checks.prefill_until=generation_options.prefill_until;
                slot_checks.canonical_prefill_rows=generation_options.canonical_prefill_rows;slot_checks.short_read=generation_options.short_read;slot_checks.tail=generation_options.cache_tail;
                SlotSession engine(uint32_t(batch.slots),model.vocabulary(),options.memory.context,
                                   model.stop_tokens(),std::move(slot_op),generation_options.cache_slots>0,slot_checks);
                shared_slots=&engine;control.solo=&generation;control.batch_mtp=batch.batch_mtp;
                serve_batch_protocol(STDIN_FILENO,std::cout,engine,batch.slots,model.vocabulary(),
                                     options.memory.context,std::move(control));
            } else
                serve_protocol(STDIN_FILENO,std::cout,generation,model.vocabulary(),options.memory.context,std::move(control));
        }
        else {
            GenerationEvents events;events.token=[](int32_t id){std::cout<<"T "<<id<<'\n'<<std::flush;};
            const auto result=generation.run(*one_shot,events);std::cout<<"DONE "<<result.generated<<" "<<result.prompt_tokens<<" "<<result.prompt_ms<<" "<<result.decode_ms<<" "<<result.finish<<'\n';

            if(stats_report) {
                // Upstream prints --stats at shutdown (:14080-14125). This backend prints only the terms it really
                // collects: the generation result, the expert cache counters and the PLE reader's io report. The
                // token-loop host phases and the CPU pool phases have no timers here and are NOT printed.
                StageStats stats;
                stats.prompt_ms=result.prompt_ms;stats.decode_ms=result.decode_ms;
                stats.prompt_tokens=result.prompt_tokens;stats.generated=result.generated;stats.reused=result.reused;
                stats.verify_windows=result.verify_windows;stats.drafts_proposed=result.drafts_proposed;
                stats.drafts_accepted=result.drafts_accepted;
                if(options.memory.expert_cache) {
                    const auto cache=model.expert_cache_stats();
                    stats.expert_cache_configured=true;
                    stats.expert_hits=cache.hits;stats.expert_loads=cache.loads;stats.expert_evictions=cache.evictions;
                    stats.expert_failed=cache.failed_loads;stats.expert_entries=cache.entries;
                    stats.expert_resident_bytes=cache.resident_bytes;stats.expert_peak_bytes=cache.peak_bytes;
                }
                std::cout<<format_stats(stats)<<std::flush;
            }            if(!dump_options.none()) std::cerr<<"strata-hpu: dumps logits_rows="<<result.dumped_logits_rows<<" final_r_records="<<result.dumped_final_r_records<<" routing_records="<<result.dumped_routing_records<<" half_records="<<result.dumped_half_records<<'\n';
        }
        if(generation_options.resident_routing && !generation_options.resident_routing->close())
            throw std::runtime_error(generation_options.resident_routing->close_error());
        // --expert-profile-save writes what the tier learned (upstream's generate.cpp:9414-9432 shutdown path;
        // --expert-profile-save-every 0 means exactly "on shutdown only").
        if(!options.expert_profile.save.empty()) {
            std::string error;
            if(!model.save_expert_profile(error)) std::cerr<<"strata-hpu: expert profile save failed: "<<error<<'\n';
            else std::cerr<<"strata-hpu: expert profile written to "<<options.expert_profile.save<<'\n';
        }
    } catch(const std::exception& e) {
        // strata/hpu/expert_residency.hpp keeps upstream's refusal text byte for byte, including the
        // "strata generate: " prefix its messages carry. Printing that as it stands instead of stacking this
        // tool's prefix on top is what makes --resident-cpu-experts' refusal read the way upstream's does.
        const std::string what=e.what();
        if(what.rfind("strata generate: ",0)==0) std::cerr<<what<<'\n';else std::cerr<<"strata-hpu: "<<what<<'\n';
        return 1;
    }
}
