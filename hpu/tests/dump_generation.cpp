// hpu/tests/dump_generation.cpp - host evidence that the per-position dumps are armed, REFUSED and written from
// the ACTUAL Generation loop, not just from DumpSinks in isolation (hpu/tests/dump_sinks.cpp covers the bytes).
//
// A fake backend stands in for the model, exactly as hpu/tests/generation.cpp does: `decode` returns a logits
// spike at (token+1)%5 so the emitted chain is deterministic, and the diagnostic producers return hand-built
// values. Nothing here touches a device or the SDK.
//
// What is pinned:
//   * upstream's row law: prompt_tokens-1+max_new records, one per processed position from 0
//     (src/program/generate.cpp:6077, :6088-6095, :13405-13432);
//   * one record per position for --dump-final-r and --dump-layers, one routing record per layer per position;
//   * the two ONE-SHOT files (--dump-mixed, --dump-residual) written once at the end of the run;
//   * the refusals: a dump with no producer (halves everywhere, a ladder with producers.layer_ladder false), a
//     per-position dump together with a prompt cache (a resumed prefix would leave the header promising rows
//     never written), --prefill or speculative drafts. End-of-run one-shots support these combinations.
#include "strata/hpu/generation.hpp"
#include "strata/hpu/protocol.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace strata::hpu;

int failures = 0;
void check(const bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cout << "FAIL " << what << "\n";
    }
}
template <class F>
bool refused(F&& call, const std::string& needle = {}) {
    try {
        call();
    } catch (const std::exception& error) {
        return needle.empty() || std::string(error.what()).find(needle) != std::string::npos;
    }
    return false;
}

std::filesystem::path scratch() {
    static const auto dir = std::filesystem::temp_directory_path() / "strata-hpu-dump-generation";
    std::filesystem::create_directories(dir);
    return dir;
}
std::string path_for(const char* const name) { return (scratch() / name).string(); }
uint64_t size_of(const std::string& path) {
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    return error ? 0 : uint64_t(size);
}
std::vector<unsigned char> bytes_of(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    return std::vector<unsigned char>(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

constexpr uint32_t VOCAB = 5, CONTEXT = 16;
constexpr int64_t LAYERS = 2, STREAMS = 1, HIDDEN = 2, EXPERTS = 3;
constexpr uint64_t POSITIONS = 4; // prompt 2 - 1 + max_new 3, upstream's law

// A fake trunk: the token chain the real generation test uses, plus the four diagnostic producers.
struct Backend {
    uint64_t decodes = 0, ladder_calls = 0, routing_calls = 0;
    GenerationOperations operations() {
        GenerationOperations op;
        op.reset = [] {};
        op.decode = [this](const int32_t token, const std::function<bool()>& cancelled) {
            if (cancelled && cancelled()) throw Cancelled();
            ++decodes;
            std::vector<float> logits(VOCAB, -10.0f);
            logits[size_t((token + 1) % int32_t(VOCAB))] = 10.0f;
            return logits;
        };
        op.residual = [this] {
            const float base = float(decodes);
            return std::vector<float>{base, base + 0.5f};
        };
        op.mixed = [] { return std::vector<float>{1.25f, -2.5f}; };
        op.layer_ladder = [this] {
            std::vector<float> ladder(size_t((LAYERS + 1) * STREAMS * HIDDEN));
            for (size_t slot = 0; slot < ladder.size(); ++slot) ladder[slot] = float(++ladder_calls);
            return ladder;
        };
        op.routing = [this] {
            ++routing_calls;
            std::vector<RoutingRecord> records;
            for (int32_t layer = 0; layer < int32_t(LAYERS); ++layer) {
                RoutingRecord record;
                record.layer = layer;
                record.ids = {layer, layer + 1};
                record.weights = {0.5f + float(layer), 0.25f};
                records.push_back(std::move(record));
            }
            return records;
        };
        return op;
    }
};

DumpGeometry shape(const int64_t stride = 1) {
    DumpGeometry geometry;
    geometry.n_vocab = int64_t(VOCAB);
    geometry.n_layers = LAYERS;
    geometry.hc = STREAMS;
    geometry.n_embd = HIDDEN;
    geometry.n_head = 1;
    geometry.head_dim = 1;
    geometry.n_head_kv = 1;
    geometry.n_expert = EXPERTS;
    geometry.stride = stride;
    return geometry;
}

DumpProducers all_producers() {
    DumpProducers producers;
    producers.logits = true;
    producers.layer_ladder = true;
    producers.routing = true;
    producers.final_r = true;
    producers.head_residual = true;
    producers.head_mixed = true;
    return producers; // half_capture stays false: this backend captures no attention halves
}

GenerationOptions options_with(const DumpOptions& dumps) {
    GenerationOptions options;
    options.cache_bytes = 1u << 20; // the cache itself is off (cache_slots 0); this is just a sane bound
    options.dumps = dumps;
    options.dump_shape = shape();
    options.dump_producers = all_producers();
    return options;
}

GenerationResult run_one(Generation& generation, std::vector<int32_t>* emitted = nullptr) {
    GenerationEvents events;
    events.token = [emitted](const int32_t token) {
        if (emitted) emitted->push_back(token);
    };
    const auto request = parse_generation_request("GEN 3 0,1", VOCAB, CONTEXT);
    return generation.run(request, events);
}

// ---- 1. the whole record set from one real run ---------------------------------------------------------
void writes_every_wired_record() {
    DumpOptions dumps;
    dumps.logits = path_for("logits.bin");
    dumps.final_r = path_for("final-r.bin");
    dumps.layers = path_for("layers.bin");
    dumps.routing = path_for("routing.bin");
    dumps.mixed = path_for("mixed.bin");
    dumps.residual = path_for("residual.bin");
    for (const auto* name : {"logits.bin", "final-r.bin", "layers.bin", "routing.bin", "mixed.bin", "residual.bin"})
        std::filesystem::remove(path_for(name));

    Backend backend;
    Generation generation(VOCAB, CONTEXT, {4}, backend.operations(), options_with(dumps));
    std::vector<int32_t> emitted;
    const auto result = run_one(generation, &emitted);

    check(emitted == std::vector<int32_t>({2, 3, 4}), "the fake chain still emits 2,3,4 (EOS stops the run)");
    check(result.generated == 3 && result.finish == "stop", "the run finished on EOS");
    check(backend.decodes == POSITIONS, "one decode per processed position, exactly prompt-1+max_new");

    // logits: int32 {n_vocab, n_rows} then one float32 row per position.
    const auto logits = bytes_of(dumps.logits);
    check(logits.size() == 8 + POSITIONS * uint64_t(VOCAB) * 4, "--dump-logits holds the header then one row per position");
    if (logits.size() >= 8) {
        int32_t header[2]{};
        std::memcpy(header, logits.data(), sizeof header);
        check(header[0] == int32_t(VOCAB) && header[1] == int32_t(POSITIONS), "the logits header is {n_vocab, n_rows} with upstream's row count");
    }
    // final-r: int64 {pos, tok} + hc*n_embd float32 per position.
    check(size_of(dumps.final_r) == POSITIONS * (2 * 8 + uint64_t(STREAMS * HIDDEN) * 4),
          "--dump-final-r appends one {pos,tok}+residual record per position");
    const auto final_r = bytes_of(dumps.final_r);
    if (final_r.size() >= 16) {
        int64_t first[2]{};
        std::memcpy(first, final_r.data(), sizeof first);
        check(first[0] == 0 && first[1] == 0, "the first final-r record is position 0, token 0");
    }
    // layers: one (n_layers+1)*hc*n_embd ladder per position.
    check(size_of(dumps.layers) == POSITIONS * uint64_t((LAYERS + 1) * STREAMS * HIDDEN) * 4,
          "--dump-layers writes one ladder per position");
    check(backend.ladder_calls == POSITIONS * uint64_t((LAYERS + 1) * STREAMS * HIDDEN),
          "the ladder producer ran once per position (its hook is read, not re-run)");
    // routing: one record per MoE layer per position.
    check(size_of(dumps.routing) == POSITIONS * uint64_t(LAYERS) * (8 + 2 * 4 + 2 * 4),
          "--dump-routing writes one record per layer per position");
    check(result.dumped_logits_rows == POSITIONS && result.dumped_final_r_records == POSITIONS &&
              result.dumped_routing_records == POSITIONS * uint64_t(LAYERS),
          "the per-kind counters the tool reports match the files");
    // the two one-shots: n_embd floats with no header, and {hc,n_embd} + hc*n_embd.
    check(size_of(dumps.mixed) == uint64_t(HIDDEN) * 4, "--dump-mixed is n_embd float32 and no header");
    check(size_of(dumps.residual) == 8 + uint64_t(STREAMS * HIDDEN) * 4, "--dump-residual is {hc,n_embd} then the residual");
}

// ---- 2. the refusals -----------------------------------------------------------------------------------
void refuses_what_it_cannot_record() {
    // --dump-halves has no producer in this backend: refused by the module's own law.
    {
        DumpOptions dumps;
        dumps.halves = path_for("halves.bin");
        Backend backend;
        Generation generation(VOCAB, CONTEXT, {4}, backend.operations(), options_with(dumps));
        check(refused([&] { (void)run_one(generation); }, "CAPTURED"), "--dump-halves is refused (no captured path)");
    }
    // A ladder asked for while the declared producers say there is none: refused before a byte is written.
    {
        DumpOptions dumps;
        dumps.layers = path_for("layers-none.bin");
        auto options = options_with(dumps);
        options.dump_producers.layer_ladder = false;
        Backend backend;
        Generation generation(VOCAB, CONTEXT, {4}, backend.operations(), options);
        check(refused([&] { (void)run_one(generation); }, "staging buffer"), "--dump-layers without its producer is refused");
        check(size_of(dumps.layers) == 0, "the refused ladder dump left no file");
    }
    // A prompt cache may resume a parked prefix, so the record set would start above position 0.
    {
        DumpOptions dumps;
        dumps.logits = path_for("logits-cache.bin");
        auto options = options_with(dumps);
        options.cache_slots = 2;
        Backend backend;
        auto operations = backend.operations();
        operations.checkpoint_bytes = 1u << 20;
        operations.checkpoint = [] { return std::function<void()>([] {}); };
        Generation generation(VOCAB, CONTEXT, {4}, std::move(operations), options);
        check(refused([&] { (void)run_one(generation); }, "prompt cache"),
              "dumps together with a resumable prompt cache are refused");
    }
    // The token path is where the records come from, so the batched prompt path is refused.
    {
        DumpOptions dumps;
        dumps.logits = path_for("logits-prefill.bin");
        auto options = options_with(dumps);
        options.prefill_chunk = 8;
        Backend backend;
        auto operations = backend.operations();
        operations.prefill = [](std::span<const int32_t>, const PromptCommit&, const std::function<bool()>&) {};
        Generation generation(VOCAB, CONTEXT, {4}, std::move(operations), options);
        check(refused([&] { (void)run_one(generation); }, "--prefill"), "dumps with --prefill are refused");
    }
    // Speculative drafts consume rows outside the per-position logits path.
    {
        DumpOptions dumps;
        dumps.logits = path_for("logits-draft.bin");
        auto options = options_with(dumps);
        options.max_draft = 2;
        Backend backend;
        auto operations = backend.operations();
        operations.draft = [](const DraftContext&, uint32_t, const std::function<bool()>&) { return std::vector<DraftRow>{}; };
        Generation generation(VOCAB, CONTEXT, {4}, std::move(operations), options);
        check(refused([&] { (void)run_one(generation); }, "speculative"), "dumps with speculative drafts are refused");
    }
    // An unarmed Generation writes nothing and reports zero counters.
    {
        GenerationOptions plain;
        plain.cache_bytes = 1u << 20;
        Backend backend;
        Generation generation(VOCAB, CONTEXT, {4}, backend.operations(), plain);
        const auto result = run_one(generation);
        check(result.dumped_logits_rows == 0 && result.dumped_final_r_records == 0 && result.dumped_routing_records == 0,
              "dumps off means zero records, not a silent file");
        check(backend.ladder_calls == 0 && backend.routing_calls == 0, "dumps off never calls a diagnostic producer");
    }
}

// One-shot diagnostics read the last COMMITTED head state, not speculative
// candidate tails and not a guessed zero row. Use a stateful owned-image fake
// so restore/reuse mistakes are visible in the bytes, not just file sizes.
void one_shots_with_windows_and_reuse() {
    for(bool native:{false,true}) for(bool prefill:{false,true}) for(bool cache:{false,true}) for(int mismatch:{0,1,2}) {
        std::vector<int32_t> fed,output;uint32_t prefill_calls=0,native_calls=0,mixed_calls=0,residual_calls=0;
        const auto values=[](int32_t token){std::vector<float> logits(VOCAB,-10);logits[size_t((token+1)%int32_t(VOCAB))]=10;return logits;};
        GenerationOperations op;op.reset=[&]{fed.clear();};
        op.decode=[&](int32_t token,const std::function<bool()>& stop){if(stop && stop()) throw Cancelled();fed.push_back(token);return values(token);};
        op.prefill=[&](std::span<const int32_t> ids,const PromptCommit& commit,const std::function<bool()>& stop){++prefill_calls;for(uint32_t at=0;at<ids.size();++at) {if(stop && stop()) throw Cancelled();fed.push_back(ids[at]);const auto logits=values(ids[at]);commit(at+1,logits);}};
        op.checkpoint_bytes=128;op.checkpoint=[&]{return std::function<void()>([&,saved=fed]{fed=saved;});};
        op.draft=[&](const DraftContext& c,uint32_t maximum,const std::function<bool()>&){std::vector<DraftRow> rows;for(uint32_t at=0;at<maximum;++at) rows.push_back({int32_t((c.history.back()+1+int32_t(at)+(int(at)==mismatch?2:0))%int32_t(VOCAB)),{},{}});return rows;};
        if(native) op.verify=[&](const DraftContext& c,std::span<const DraftRow> rows,const std::function<bool()>& stop){
            ++native_calls;check(std::equal(fed.begin(),fed.end(),c.history.begin(),c.history.end()),"native dump window received another physical prefix");
            std::vector<float> after;for(const auto& row:rows) {const auto logits=values(row.token);after.insert(after.end(),logits.begin(),logits.end());}
            const std::array<int32_t,1> stops{4};auto selected=select_verified_candidates(c.logits,after,rows,c.history,c.sampling,c.bias,c.mode,stops,stop,c.follow);
            for(uint32_t at=0;at<selected.retained;++at) fed.push_back(rows[at].token);
            selected.logits=values(fed.back());return selected;
        };
        op.mixed=[&]{++mixed_calls;check(!fed.empty(),"one-shot mixed read without a committed head");return std::vector<float>{float(fed.size()),float(fed.back())};};
        op.residual=[&]{++residual_calls;check(!fed.empty(),"one-shot residual read without a committed head");return std::vector<float>{float(fed.back()),float(fed.size())+.5f};};
        op.layer_ladder=[]()->std::vector<float>{throw std::logic_error("one-shot diagnostics unexpectedly read a ladder");};
        op.routing=[]()->std::vector<RoutingRecord>{throw std::logic_error("one-shot diagnostics unexpectedly read routing");};
        DumpOptions dumps;dumps.mixed=path_for("window-mixed.bin");dumps.residual=path_for("window-residual.bin");
        auto options=options_with(dumps);options.max_draft=2;options.prefill_chunk=prefill?4:0;options.short_read=0;options.cache_slots=cache?3:0;
        Generation generation(VOCAB,CONTEXT,{4},std::move(op),options);
        for(int run=0;run<(cache?2:1);++run) {
            std::filesystem::remove(dumps.mixed);std::filesystem::remove(dumps.residual);output.clear();
            const auto result=run_one(generation,&output);
            check(output==std::vector<int32_t>({2,3,4}) && fed==std::vector<int32_t>({0,1,2,3}) && result.finish=="stop","one-shot diagnostic changed generation or retained a rejected/EOS tail");
            if(cache && run) check(result.reused>0,"one-shot diagnostics disabled real cached prompt reuse");
            check(!result.dumped_logits_rows && !result.dumped_final_r_records && !result.dumped_routing_records,"one-shot diagnostic invented per-position records");
            const auto mixed=bytes_of(dumps.mixed),residual=bytes_of(dumps.residual);
            check(mixed.size()==8 && residual.size()==16,"window one-shot byte layouts changed");
            if(mixed.size()==8 && residual.size()==16) {
                float m[2],r[2];int32_t header[2];std::memcpy(m,mixed.data(),8);std::memcpy(header,residual.data(),8);std::memcpy(r,residual.data()+8,8);
                check(m[0]==4 && m[1]==3 && r[0]==3 && r[1]==4.5f && header[0]==STREAMS && header[1]==HIDDEN,"one-shot dump read rejected, stale, EOS-fed or pre-restore head state");
            }
            check(mixed_calls==uint32_t(run+1) && residual_calls==uint32_t(run+1),"one-shot producer was polled during candidate evaluation/replay");
        }
        if(native) check(native_calls>0,"one-shot diagnostic silently bypassed native verification");
        if(prefill) check(prefill_calls>0,"one-shot diagnostic silently disabled prompt prefill");
    }
    // A combined one-shot/per-position request retains the position-coverage
    // restriction; the two files must not hide an unsupported producer.
    DumpOptions combined;combined.mixed=path_for("combined-mixed.bin");combined.logits=path_for("combined-logits.bin");
    auto options=options_with(combined);options.prefill_chunk=4;Backend backend;
    Generation generation(VOCAB,CONTEXT,{4},backend.operations(),options);
    check(refused([&]{run_one(generation);},"--prefill") && !backend.decodes,"one-shot flag bypassed combined per-position coverage validation");
}
} // namespace

int main() {
    writes_every_wired_record();
    refuses_what_it_cannot_record();
    one_shots_with_windows_and_reuse();
    std::filesystem::remove_all(scratch());
    if (failures != 0) {
        std::cout << "FAILED dump generation checks: " << failures << "\n";
        return 1;
    }
    std::cout << "PASS dump generation: one real Generation run writes upstream's record set (logits row law "
                 "prompt-1+max_new, final-r per position, one ladder and one routing record per layer per position, "
                 "both one-shot files), the counters match the files, and a dump with no producer, with a resumable "
                 "prompt cache, with --prefill or with speculative drafts is refused before any state changes; "
                 "end-of-run one-shots support native/scalar windows, prefill and actual cache reuse; no device\n";
    return 0;
}
