// hpu/tests/kv_mode.cpp - host evidence for upstream's `--kv` selector.
//
// Upstream accepts --kv from {fp16, int8, q4/q4_0, k8v4} (src/program/generate.cpp:2545-2558) and refuses anything
// else with "strata generate: --kv must be fp16, int8, q4_0 or k8v4". This backend runs host fp16, int8, q4_0 and
// k8v4 KV, so all four are accepted and keep upstream's spelling; q4 normalizes to q4_0; a bare number is refused
// with the pointer to the port-only --kv-mib; anything else gets upstream's message. Under k8v4 the MTP drafter is
// plain INT8 (upstream builds its ring with the globals toggled to INT8, src/core/mtp.cpp:344-349).
#include "strata/hpu/kv_mode.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {
int failures = 0;
void check(const bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cout << "FAIL " << what << "\n";
    }
}
std::string refusal(const std::string& text) {
    try {
        (void)strata::hpu::resolve_kv_mode(text);
    } catch (const std::invalid_argument& error) {
        return error.what();
    }
    return {};
}
bool contains(const std::string& message, const std::string& needle) {
    return message.find(needle) != std::string::npos;
}
} // namespace

int main() {
    using strata::hpu::KvMode;
    using strata::hpu::kv_drafter_mode;
    using strata::hpu::kv_mode_name;
    using strata::hpu::resolve_kv_mode;

    // The four modes this backend runs are accepted, and they keep upstream's spelling.
    {
        const auto fp16 = resolve_kv_mode("fp16");
        check(fp16.mode == KvMode::Fp16 && fp16.name == "fp16", "fp16 is accepted and reported as fp16");
        const auto int8 = resolve_kv_mode("int8");
        check(int8.mode == KvMode::Int8 && int8.name == "int8", "int8 is accepted and reported as int8");
        const auto q4 = resolve_kv_mode("q4_0");
        check(q4.mode == KvMode::Q4 && q4.name == "q4_0", "q4_0 is accepted and reported as q4_0");
        const auto hybrid = resolve_kv_mode("k8v4");
        check(hybrid.mode == KvMode::K8V4 && hybrid.name == "k8v4", "k8v4 is accepted and reported as k8v4");
        check(std::string(kv_mode_name(KvMode::K8V4)) == "k8v4", "INFO reports k8v4, not a fallback");
    }
    // `q4` is upstream's alias for q4_0 (generate.cpp:2545): it normalizes to the canonical spelling.
    {
        const auto q4 = resolve_kv_mode("q4");
        check(q4.mode == KvMode::Q4 && q4.name == "q4_0", "q4 normalizes to q4_0");
    }

    // The drafter follows the trunk except under k8v4, where upstream's ring state is plain INT8.
    {
        check(kv_drafter_mode(KvMode::Fp16) == KvMode::Fp16, "fp16 drafter stays fp16");
        check(kv_drafter_mode(KvMode::Int8) == KvMode::Int8, "int8 drafter stays int8");
        check(kv_drafter_mode(KvMode::Q4) == KvMode::Q4, "q4_0 drafter stays q4_0");
        check(kv_drafter_mode(KvMode::K8V4) == KvMode::Int8, "k8v4 drafter is plain INT8");
    }

    // The old port meaning of the flag (a MiB number) is refused with the pointer to --kv-mib, not accepted.
    for (const char* number : {"2048", "0", "16"}) {
        const auto message = refusal(number);
        check(contains(message, "--kv-mib") && contains(message, "not a byte count"),
              std::string("a number is redirected to --kv-mib: ") + number);
    }

    // Everything else gets upstream's own message, unchanged.
    for (const char* other : {"", "q8", "fp32", "FP16", "INT8", "int4"}) {
        const auto message = refusal(other);
        check(message == "--kv must be fp16, int8, q4_0 or k8v4",
              std::string("an unknown value gets upstream's message: '") + other + "' -> " + message);
    }

    if (failures != 0) {
        std::cout << "FAILED kv-mode checks: " << failures << "\n";
        return 1;
    }
    std::cout << "PASS kv mode: --kv fp16, int8, q4_0 and k8v4 accepted (upstream spelling); q4 normalizes to q4_0; "
                 "the k8v4 drafter is plain INT8 while every other mode follows the trunk; a bare MiB "
                 "number is refused as 'not a byte count'; any other value gets upstream's exact message; no device\n";
    return 0;
}
