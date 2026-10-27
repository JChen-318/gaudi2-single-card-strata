// hpu/tests/stop_tokens.cpp - host evidence for upstream's end-of-turn STOP LAW and its park-length FLOOR.
//
// Two upstream behaviours are pinned here, both without a model, an SDK or a device.
//
// 1. The stop law is two laws (strata/hpu/stop_tokens.hpp). The id list lives in `eos_ids` (:612, the pair
//    248044/248046 that upstream's tokenizer-less build hardcodes) and batch or verify windows test it ungated
//    (:10077, :12424). The ordinary token loop tests it only while ARMED, and only on generated positions
//    (:13493-13494, :13511, :13881); --stop-eos (:2097), --eos-ids which arms as it parses (:2187-2191), and a
//    serve whose requests bring their own tokens (:2528-2532) are the three things that arm it. So a scripted
//    chain that emits an end-of-turn token must STOP armed and RUN ON unarmed - the unarmed half is the whole
//    point, and a test that only ever checks the armed half would pass against the port's old always-stops code.
// 2. --conversation-cache-min-tokens (declared :639, parsed bounded at :2107-2124, law
//    include/strata/core/conversation_cache.hpp:206, park site :8227-8236): a conversation shorter than the
//    minimum is not worth one of the slots. It is a SKIP, not an error, and 0 - upstream's default - parks
//    everything whatever its length.
#include "strata/hpu/stop_tokens.hpp"
#include "strata/hpu/generation.hpp"
#include "strata/hpu/prefix_cache.hpp"
#include "strata/hpu/protocol.hpp"

#include <algorithm>
#include <cstdint>
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
bool refused(F&& call, const std::string& needle) {
    try {
        call();
    } catch (const std::exception& error) {
        return std::string(error.what()).find(needle) != std::string::npos;
    }
    return false;
}

constexpr uint32_t VOCAB = 5, CONTEXT = 16;
constexpr int32_t STOP = 3; // the scripted end-of-turn token in the chain below

// The list law (upstream's parse_i64_list, :965-981) and the arming law.
void list_and_arming_law() {
    check((parse_token_ids("248044,248046", "--eos-ids") == std::vector<int32_t>({248044, 248046})),
          "--eos-ids reads upstream's default pair");
    check((parse_token_ids(" 7, 8 ,,9 ", "--eos-ids") == std::vector<int32_t>({7, 8, 9})),
          "commas and spaces separate, empty entries are skipped as upstream's stream parser does");
    check((parse_token_ids("1,2,", "--eos-ids") == std::vector<int32_t>({1, 2})), "a trailing separator is not a token");
    check(refused([&] { (void)parse_token_ids("", "--eos-ids"); }, "token list was empty"),
          "an empty list is refused with upstream's sentence");
    check(refused([&] { (void)parse_token_ids("x", "--eos-ids"); }, "invalid token id: expected an integer in [0, 2147483647]"),
          "a non-integer is refused with upstream's sentence");
    check(refused([&] { (void)parse_token_ids("-1", "--eos-ids"); }, "invalid token id"), "a negative id is refused");
    check(refused([&] { (void)parse_token_ids("2147483648", "--eos-ids"); }, "invalid token id"),
          "upstream's list is int32, so 2147483648 is refused rather than truncated");
    check(refused([&] { (void)parse_token_ids("x", "--tokens"); }, "--tokens: invalid token id"),
          "the refusal carries the option that produced it");

    // The arming truth table, upstream's three sources (:2097, :2190, :2528-2532).
    check(!stop_tokens_armed(false, false, false, true), "a plain run with tokens and no flag is unarmed");
    check(!stop_tokens_armed(false, false, false, false), "no flag, no serve: still unarmed");
    check(stop_tokens_armed(true, false, false, true), "--stop-eos arms even with an explicit token input");
    check(stop_tokens_armed(false, true, false, true), "--eos-ids arms as it parses");
    check(stop_tokens_armed(false, false, true, false), "a serve whose requests bring their own tokens is armed");
    check(!stop_tokens_armed(false, false, true, true), "a serve that was handed tokens is not the armed case");

    // The implicit default list, clipped to what this vocabulary could ever emit.
    check((implicit_stop_tokens(300000) == std::vector<int32_t>({default_eos_token, default_eot_token})),
          "upstream's pair is the default where the model can emit it");
    check((implicit_stop_tokens(248045) == std::vector<int32_t>({default_eos_token})),
          "an id above the vocabulary is dropped, because it could never be emitted");
    check(implicit_stop_tokens(100).empty(), "a pair wholly above the vocabulary stops nothing, as upstream's would");
}

// Generation is the ordinary single-request path: armed stops, unarmed runs on to max_new.
struct Backend {
    GenerationOperations operations() {
        GenerationOperations op;
        op.reset = [] {};
        // A deterministic chain: the logits always spike at (token+1)%VOCAB, so from prompt {0,1} the run emits
        // 2, then 3 (the scripted end-of-turn token), then 4.
        op.decode = [](const int32_t token, const std::function<bool()>& cancelled) {
            if (cancelled && cancelled()) throw Cancelled();
            std::vector<float> logits(VOCAB, -10.0f);
            logits[size_t((token + 1) % int32_t(VOCAB))] = 10.0f;
            return logits;
        };
        // Scalar draft verification needs an owned checkpoint to rewind to (hpu/src/generation.cpp:284-288). This
        // fake's only state is the token argument it was handed, so restoring it is genuinely doing nothing.
        op.checkpoint = [] { return std::function<void()> { [] {} }; };
        return op;
    }
};

// One fresh run through Generation, so the armed and unarmed cases cannot share state.
struct Outcome { GenerationResult result; std::vector<int32_t> emitted; };
Outcome outcome(const bool armed, const bool drafts, const std::vector<int32_t>& stops) {
    GenerationOptions options;
    options.max_draft = drafts ? 2u : 0u;
    options.cache_slots = 0;
    options.cache_bytes = 1u << 20;
    options.stop_eos = armed;
    auto op = Backend().operations();
    // An accepted proposal makes the verify window emit two tokens per round, which is the path through the
    // window's own `stopped` flag rather than the single-token break.
    if (drafts) op.draft = [](const DraftContext&, uint32_t, const std::function<bool()>&) {
        return std::vector<DraftRow>{{2, {}, {}}};
    };
    Generation generation(VOCAB, CONTEXT, stops, std::move(op), options);
    Outcome outcome;
    GenerationEvents events;
    events.token = [&outcome](const int32_t token) { outcome.emitted.push_back(token); };
    outcome.result = generation.run(parse_generation_request("GEN 3 0,1", VOCAB, CONTEXT), events);
    return outcome;
}

void loop_respects_the_arming(const bool drafts) {
    const std::string where = drafts ? "verify window" : "token loop";
    const auto armed = outcome(true, drafts, {STOP});
    const auto unarmed = outcome(false, drafts, {STOP});
    check(armed.result.finish == "stop", "an armed " + where + " ends on the end-of-turn token");
    check(unarmed.result.finish != "stop", "an unarmed " + where + " does not end on the end-of-turn token at all");
    check(unarmed.result.generated == 3 && unarmed.result.finish == "length",
          "an unarmed " + where + " runs on to max_new and reports its real reason");
    check(armed.result.generated < unarmed.result.generated, "arming emits fewer tokens than the unarmed run");
    check(std::find(armed.emitted.begin(), armed.emitted.end(), STOP) == armed.emitted.end() ||
          armed.emitted.back() == STOP, "an armed " + where + " ends its emission on the stop token it did emit");
    check(std::find(unarmed.emitted.begin(), unarmed.emitted.end(), STOP) != unarmed.emitted.end() &&
          unarmed.emitted.back() != STOP, "the unarmed " + where + " emits past the stop token and carries on");
    // The id list is still the list: armed but with an id this chain never emits, nothing stops.
    const auto other = outcome(true, drafts, {1});
    check(other.result.finish == "length" && other.result.generated == 3,
          "arming on a token the chain never emits changes nothing, as upstream's find() over eos_ids does");
}

void defaults_are_upstreams() {
    GenerationOptions options;
    check(options.cache_min_tokens == 0, "--conversation-cache-min-tokens defaults to 0, upstream's park-everything default");
    check(options.stop_eos, "Generation's own default stays armed, because an in-process caller has no flag to read");
    check(default_eos_token == 248044 && default_eot_token == 248046, "upstream's hardcoded pair is quoted, not invented");
}

// The park floor: PrefixCache::store is the only door, so the floor cannot be parked past.
bool park(PrefixCache& cache, const std::vector<int32_t>& tokens) {
    const std::vector<float> logits(tokens.empty() ? 0 : tokens.size() * VOCAB, 0.5f);
    return cache.store(tokens, logits.empty() ? std::vector<float>{0.5f} : logits, 1024,
                       [] { return std::function<void()> { [] {} }; }, false, false, false, true, {});
}

void park_length_floor() {
    PrefixCache everything(4, 1u << 20);
    check(everything.wants(1) && everything.wants(0) && everything.wants(1 << 20),
          "min_tokens 0 wants every conversation, upstream's default");
    check(park(everything, {0}), "the default floor still parks a one-token conversation");

    PrefixCache floored(4, 1u << 20, {}, 0, 0, 0, 3);
    check(!floored.wants(2) && floored.wants(3) && floored.wants(4), "wants() is upstream's >= against the minimum");
    check(!park(floored, {0, 1}), "a short conversation is not parked");
    const auto skipped = floored.stats();
    check(skipped.min_tokens_skipped == 1 && skipped.skipped == 1 && skipped.entries == 0,
          "the skip is counted, said, and takes no slot");
    check(park(floored, {0, 1, 2}), "a conversation at exactly the minimum is worth a slot");
    check(floored.stats().entries == 1 && floored.stats().min_tokens_skipped == 1, "the qualifying park still happens");

    // Upstream's floor is a bound, not a switch: a negative value parks everything (:206 tests <= 0).
    PrefixCache negative(4, 1u << 20, {}, 0, 0, 0, -1);
    check(negative.wants(0) && park(negative, {1}), "a nonpositive floor parks everything like upstream");
    check(refused([&] { (void)PrefixCache(4, 1u << 20, {}, 0, 0, 0, int64_t(UINT32_MAX) + 1); }, "invalid prefix cache limits"),
          "the ctor refuses a floor above the bound the CLI itself enforces");
}

}

int main() {
    try {
        list_and_arming_law();
        loop_respects_the_arming(false);
        loop_respects_the_arming(true);
        defaults_are_upstreams();
        park_length_floor();
    } catch (const std::exception& error) {
        std::cout << "FAIL stop-token check threw before finishing: " << error.what() << "\n";
        return 1;
    }
    if (failures) {
        std::cout << "stop tokens: " << failures << " failed\n";
        return 1;
    }
    std::cout << "PASS stop-token list and arming law, unarmed loop runs to length, park-length floor skips and counts\n";
    return 0;
}
