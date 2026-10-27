#pragma once
// hpu/include/strata/hpu/protocol_info.hpp - the INFO line's key=value facts, and why they may not carry whitespace.
//
// The pinned server splits the engine's INFO line on spaces and parses every token as key=value
// (serve/server.py:839-842), then reads specific keys from it: `batch_slots`, `batch_groups`, `slot_cache`,
// `arena_mib`, `conversation_cache_mib`, `conversation_cache_slots`, `context`, `engine`, `kv_resident`, `vram`,
// `vram_free_mib`, `version`. A malformed fact (space inside a value, an empty key) would silently shift or drop
// every following field, so the builder REFUSES it instead of emitting a corrupt line.
//
// Upstream's own line is generated at src/program/generate.cpp:9590-9610; the batch facts are printed only while
// slots are on, as `batch_slots=N slot_cache=<0|1> batch_groups=G`.
//
// This header is deliberately free of the Synapse runtime and of POSIX, so a host test can pin the rule.
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::hpu {
// The engine name the server reports as `engine=` (it only displays it). hpu/CMakeLists.txt declares the same
// project version; this fallback keeps every host syntax check free of quoting pitfalls.
inline constexpr const char* strata_hpu_engine_version = "0.1.0";

// Upstream's `strata --version` / `-V` (src/program/generate.cpp:1804-1809) prints "strata <STRATA_VERSION>" and
// returns before any GPU is touched, and STRATA_VERSION is the SAME string its INFO line writes as `engine=`
// (:9595, `engine=" STRATA_VERSION`). There is no separate `version=` fact on that line: the pinned server reads
// BUILD.json for `version` and falls back to the engine value (serve/server.py:790, :895), so the engine fact IS the
// reported version to anything that talks to the engine. These two functions keep that one-law property here: the
// report line is built FROM the engine fact, so a `--version` answer cannot disagree with what the server reads.
inline std::string engine_value() { return std::string("strata-hpu-") + strata_hpu_engine_version; }
inline std::string version_line() { return "strata " + engine_value(); }

struct ProtocolInfoFact {
    std::string key;
    std::string value;
};

namespace protocol_info_detail {
inline bool valid_key(const std::string& key) {
    if (key.empty()) return false;
    for (const char c : key)
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '.' || c == '-')) return false;
    return true;
}
inline bool valid_value(const std::string& value) {
    for (const char c : value)
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') return false;
    return true;
}
} // namespace protocol_info_detail

// Appends " key=value" per fact. Throws std::invalid_argument for an unsafe key or a value with whitespace.
inline std::string protocol_info_append(const std::string& line, const std::span<const ProtocolInfoFact> facts) {
    std::string result = line;
    for (const auto& fact : facts) {
        if (!protocol_info_detail::valid_key(fact.key))
            throw std::invalid_argument("INFO fact key is not a safe token: " + fact.key);
        if (!protocol_info_detail::valid_value(fact.value))
            throw std::invalid_argument("INFO fact value contains whitespace: " + fact.key);
        result += ' ';
        result += fact.key;
        result += '=';
        result += fact.value;
    }
    return result;
}

// Upstream prints these only while slots are on (generate.cpp:9608-9610); `slot_cache` is 0|1.
inline std::vector<ProtocolInfoFact> batch_info_facts(const uint32_t slots, const uint32_t groups, const bool slot_cache) {
    std::vector<ProtocolInfoFact> facts;
    if (slots == 0) return facts;
    facts.push_back({"batch_slots", std::to_string(slots)});
    facts.push_back({"slot_cache", slot_cache ? "1" : "0"});
    facts.push_back({"batch_groups", std::to_string(groups ? groups : 1)});
    return facts;
}

// The vision gate is a request-admission fact: without it GENI is refused by name and no M-RoPE table exists
// (generate.cpp:10735, :3197-3215). The width is the n_embd an SVE1 record must carry (:10758-10762); it is
// reported as 0 when the gate is off. The pinned server ignores keys it does not read.
inline std::vector<ProtocolInfoFact> vision_info_facts(const bool enabled, const uint32_t width) {
    return {{"vision", enabled ? "1" : "0"}, {"vision_width", std::to_string(enabled ? width : 0)}};
}
} // namespace strata::hpu
