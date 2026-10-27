// hpu/tests/check_logits.cpp - host evidence for `--check-logits`.
//
// Upstream's flag is a FULL-VOCABULARY finite scan of every read-back row (src/program/generate.cpp:13414-13420): it
// forces the whole row to be read and, when a value is not finite, it fails with
//
//     strata generate: %d of %lld logits are not finite at position %lld
//
// and exit 1 - a HARD failure, not the protocol's `DONE nonfinite`. This backend always copies the whole row for host
// sampling, so the flag only decides which report wins. The test pins both behaviours: with the flag on the exact
// upstream text (count, vocabulary and position) is thrown, and with it off the existing NonFiniteOutput path (which
// the protocol turns into a `nonfinite` finish) is untouched.
#include "strata/hpu/generation.hpp"
#include "strata/hpu/protocol.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using strata::hpu::Generation;
using strata::hpu::GenerationEvents;
using strata::hpu::GenerationOperations;
using strata::hpu::GenerationOptions;
using strata::hpu::NonFiniteOutput;
using strata::hpu::parse_generation_request;

int failures = 0;
void check(const bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cout << "FAIL " << what << "\n";
    }
}
std::string refusal(auto&& call) {
    try {
        call();
    } catch (const NonFiniteOutput&) {
        return "NonFiniteOutput";
    } catch (const std::exception& error) {
        return error.what();
    }
    return {};
}
bool has(const std::string& message, const std::string& needle) {
    return message.find(needle) != std::string::npos;
}
// A generation whose logits row is chosen by the caller, so a bad row can be injected at a known position.
struct Harness {
    std::vector<int32_t> state;
    std::vector<float> row;
    GenerationOperations operations;
    GenerationOptions options;
    Harness(const uint32_t vocabulary, std::vector<float> logits, const bool check_logits) {
        row = std::move(logits);
        options.check_logits = check_logits;
        operations.reset = [this] { state.clear(); };
        operations.available_host_bytes = [] { return std::optional<uint64_t>(UINT64_MAX); };
        operations.decode = [this](int32_t id, const auto&) {
            state.push_back(id);
            return row;
        };
        (void)vocabulary;
    }
    std::vector<int32_t> run(const uint32_t vocabulary) {
        Generation engine(vocabulary, 64, {}, operations, options);
        GenerationEvents events;
        events.token = [](int32_t) {};
        const auto request = parse_generation_request("GEN 1 0,1", vocabulary, 64);
        (void)engine.run(request, events);
        return state;
    }
};
} // namespace

int main() {
    const uint32_t vocabulary = 16;
    std::vector<float> clean(vocabulary, -50.0f);
    clean[3] = 10.0f;

    // 1. a clean row generates normally with the flag on
    {
        Harness harness(vocabulary, clean, true);
        const auto tokens = harness.run(vocabulary);
        check(tokens.size() >= 2, "a finite row still generates the prompt plus at least one token");
    }

    // 2. a NaN IN the row with the flag on: upstream's exact report, with the vocabulary size and the position
    {
        auto bad = clean;
        bad[7] = std::numeric_limits<float>::quiet_NaN();
        Harness harness(vocabulary, bad, true);
        const auto message = refusal([&] { (void)harness.run(vocabulary); });
        // The position is the engine's 0-based index of the row the logits came from (the count of tokens already
        // committed), which is what upstream's `pos` is at that point of its loop.
        check(has(message, "strata generate: 1 of 16 logits are not finite at position 0"),
              "the report names the count, the vocabulary and the position: " + message);
    }

    // 3. several bad values are counted, and an infinity counts as non-finite like a NaN
    {
        auto bad = clean;
        bad[0] = std::numeric_limits<float>::quiet_NaN();
        bad[5] = std::numeric_limits<float>::infinity();
        bad[9] = -std::numeric_limits<float>::infinity();
        Harness harness(vocabulary, bad, true);
        const auto message = refusal([&] { (void)harness.run(vocabulary); });
        check(has(message, "3 of 16 logits are not finite"), "every non-finite value is counted: " + message);
    }

    // 4. WITH THE FLAG OFF the protocol path is untouched: the same bad row does NOT hard-fail. The engine converts
    //    its own non-finite detection into a `nonfinite` finish (the protocol's `DONE ... nonfinite`), so `run`
    //    returns rather than throwing - which is exactly the behaviour the flag must not change.
    {
        auto bad = clean;
        bad[7] = std::numeric_limits<float>::quiet_NaN();
        Harness harness(vocabulary, bad, false);
        bool threw = false;
        try {
            (void)harness.run(vocabulary);
        } catch (const std::exception&) {
            threw = true;
        }
        check(!threw, "without the flag a non-finite row ends the request instead of hard-failing");
    }

    // 5. a constant (all-zero) row is finite but can only come from a broken forward pass: refused with the flag on,
    //    and still generates with the flag off (the default path is unchanged)
    {
        Harness harness(vocabulary, std::vector<float>(vocabulary, 0.0f), true);
        const auto message = refusal([&] { (void)harness.run(vocabulary); });
        check(has(message, "all 16 logits equal 0.000000 at position 0"), "an all-zero row is refused: " + message);
        Harness off(vocabulary, std::vector<float>(vocabulary, 0.0f), false);
        check(refusal([&] { (void)off.run(vocabulary); }).empty(), "without the flag an all-zero row still generates");
    }

    if (failures != 0) {
        std::cout << "FAILED check-logits checks: " << failures << "\n";
        return 1;
    }
    std::cout << "PASS check logits: a finite row generates normally, one NaN reports upstream's exact "
                 "\"1 of 16 logits are not finite at position 0\", NaN/inf/-inf are all counted (3 of 16), and with "
                 "the flag off the same row ends the request through the engine's `nonfinite` finish instead of "
                 "hard-failing; no device\n";
    return 0;
}
