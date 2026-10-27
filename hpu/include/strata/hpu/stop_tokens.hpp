#pragma once
// hpu/include/strata/hpu/stop_tokens.hpp - upstream's end-of-turn stop law, which is TWO laws and not one.
//
// The id list is `eos_ids` (src/program/generate.cpp:612, the hardcoded pair 248044/248046 that upstream's
// tokenizer-less build names in its own help text at :610-611). Every batch or verify window tests it UNGATED:
// a row whose token is in the list ends its slot with `BDONE <slot> <produced> stop` (:10077) and the single
// decode-verify window sets `finish = "stop"` the same way (:12424). The ordinary token loop is the opposite:
// it breaks only while the run is ARMED (:13493-13494, :13511, :13881 all begin with `o.stop_eos &&`) and only
// for generated positions, never while the prompt is still being teacher-forced. Arming comes from --stop-eos
// (:2097), from --eos-ids which arms as it parses (:2187-2191), or from --serve whose requests bring their own
// tokens (:2528-2532) - so the default-off flag is not a default-off behaviour under serving.
//
// The port's adaptation, said rather than assumed: this backend can read a GGUF, so the DEFAULT list comes from
// `tokenizer.ggml.eos_token_id`/`tokenizer.ggml.eot_token_id` (hpu/src/model.cpp) and --eos-ids replaces it.
// Upstream never bounds an id by the vocabulary; here an id no row could ever emit is dropped from the implicit
// default and refused from an explicit list, because Generation's stop set doubles as a verify contract that
// this backend validates (hpu/src/generation.cpp:76).

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::hpu {

// Upstream's default pair, verbatim (src/program/generate.cpp:612).
inline constexpr int32_t default_eos_token = 248044, default_eot_token = 248046;

// Upstream's parse_i64_list (src/program/generate.cpp:965-981): commas or spaces, each entry an integer in
// [0, 2147483647], an empty list an error, and both refusal sentences are upstream's own. `option` is the flag
// name, because the CLI prints "--eos-ids: <sentence>" (:2189).
inline std::vector<int32_t> parse_token_ids(const std::string& text, const std::string& option) {
    std::vector<int32_t> ids;
    std::size_t at = 0;
    while (at <= text.size()) {
        const auto end = text.find_first_of(", ", at);
        const auto entry = text.substr(at, end == std::string::npos ? std::string::npos : end - at);
        if (!entry.empty()) {
            int64_t value = 0;
            const auto parsed = std::from_chars(entry.data(), entry.data() + entry.size(), value);
            if (parsed.ec != std::errc{} || parsed.ptr != entry.data() + entry.size() || value < 0 ||
                value > int64_t(std::numeric_limits<int32_t>::max()))
                throw std::invalid_argument(option + ": invalid token id: expected an integer in [0, 2147483647]");
            ids.push_back(int32_t(value));
        }
        if (end == std::string::npos) break;
        at = end + 1;
    }
    if (ids.empty()) throw std::invalid_argument(option + ": token list was empty");
    return ids;
}

// Upstream's arming law, in one place so a caller cannot arm the loop by accident: --stop-eos, or --eos-ids,
// or a serve whose requests bring their own tokens (src/program/generate.cpp:2528-2532).
inline bool stop_tokens_armed(const bool stop_eos, const bool eos_ids, const bool serve, const bool have_tokens) {
    return stop_eos || eos_ids || (serve && !have_tokens);
}

// The implicit default list: upstream's pair, minus any id this vocabulary cannot emit. A pair entirely above
// the vocabulary stops nothing, exactly as it does upstream, where such an id simply never matches.
inline std::vector<int32_t> implicit_stop_tokens(const uint32_t vocabulary) {
    std::vector<int32_t> ids;
    for (const int32_t id : {default_eos_token, default_eot_token})
        if (id >= 0 && uint64_t(id) < uint64_t(vocabulary) &&
            std::find(ids.begin(), ids.end(), id) == ids.end()) ids.push_back(id);
    return ids;
}

}
