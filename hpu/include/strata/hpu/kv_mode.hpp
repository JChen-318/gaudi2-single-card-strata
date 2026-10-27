#pragma once
// hpu/include/strata/hpu/kv_mode.hpp - upstream's `--kv <format>` selector, resolved honestly.
//
// Upstream (src/program/generate.cpp:1950, 2545-2558) accepts `--kv` from {fp16, int8, q4/q4_0, k8v4} and picks the
// KV state format from it: int8 = INT8 K/V with one FP16 scale per 64 values (kv_q8.hpp:5-12), q4_0 = 4-bit codes
// after a Hadamard rotation (kv_q4.hpp), k8v4 = INT8 K with rotated Q4_0 V, 816 B/cell
// (src/core/layer.cpp:649-661,1080-1097). This backend keeps the KV state in HOST memory: fp16 cells, upstream's
// int8 codes + scales (kv_cell.hpp), upstream's rotated Q4_0 codes (kv_q4_cell.hpp), or the hybrid pair
// (int8 K cells + rotated Q4_0 V blocks, same headers).
//
// The MTP drafter does NOT use the hybrid: upstream builds its state with the globals toggled to INT8
// (src/core/mtp.cpp:344-349), so under --kv k8v4 the drafter runs plain INT8. `kv_drafter_mode` below is that rule.
//
// A name that only matched by string would be a silent lie: the port previously parsed `--kv` as a NUMBER (a MiB
// cap), so `--kv int8` failed as "invalid number" and `--kv 2048` was accepted although upstream rejects it. This
// header makes the selector explicit, and anything unknown gets upstream's own message. The byte cap moved to the
// port-only `--kv-mib`.
//
// Header-only and free of the SDK, so a host test pins the rule.
#include <stdexcept>
#include <string>
#include <string_view>

namespace strata::hpu {
// The KV formats this backend can actually run. Adding one here without implementing it would be a lie.
enum class KvMode { Fp16, Int8, Q4, K8V4 };

struct KvSelection {
    KvMode mode = KvMode::Fp16;
    std::string name;   // upstream's spelling, for INFO/logs
};

// upstream's own spelling of the selected format, for INFO and docs. `q4` is never reported: :2545 normalizes it.
inline const char* kv_mode_name(const KvMode mode) {
    switch (mode) {
        case KvMode::Int8: return "int8";
        case KvMode::Q4: return "q4_0";
        case KvMode::K8V4: return "k8v4";
        case KvMode::Fp16: break;
    }
    return "fp16";
}

// The format the MTP drafter runs: upstream creates its ring with the KV globals toggled to INT8, so --kv k8v4's
// drafter is plain INT8 (src/core/mtp.cpp:344-349); every other mode follows the trunk.
inline KvMode kv_drafter_mode(const KvMode trunk) {
    return trunk == KvMode::K8V4 ? KvMode::Int8 : trunk;
}

inline KvSelection resolve_kv_mode(const std::string_view text) {
    std::string name(text);
    if (name == "q4") name = "q4_0";   // upstream normalizes this spelling (generate.cpp:2545)
    if (name == "fp16") return {KvMode::Fp16, name};
    if (name == "int8") return {KvMode::Int8, name};
    if (name == "q4_0") return {KvMode::Q4, name};
    if (name == "k8v4") return {KvMode::K8V4, name};
    // A number is the OLD (port-only) meaning of this flag; say where the cap went instead of accepting it.
    bool numeric = !name.empty();
    for (const char c : name) if (c < '0' || c > '9') numeric = false;
    if (numeric)
        throw std::invalid_argument("--kv expects a KV format (fp16, int8, q4_0 or k8v4), not a byte count; the byte cap is --kv-mib MIB");
    throw std::invalid_argument("--kv must be fp16, int8, q4_0 or k8v4");   // upstream's message (:2547)
}
} // namespace strata::hpu
