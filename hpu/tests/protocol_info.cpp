// hpu/tests/protocol_info.cpp - host evidence for the INFO line's key=value facts.
//
// The pinned server (serve/server.py:839-842) splits the engine's INFO line on spaces and parses every token as
// key=value, then reads specific keys from it (batch_slots, batch_groups, slot_cache, arena_mib,
// conversation_cache_mib, conversation_cache_slots, context, engine, kv_resident, vram, vram_free_mib, version).
// Upstream's own line is built at src/program/generate.cpp:9590-9610 and prints the batch facts only while slots are
// on. These rules are pinned here without the SDK, POSIX or a device: this header is standalone on purpose.
#include "strata/hpu/protocol_info.hpp"

#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using strata::hpu::batch_info_facts;
using strata::hpu::ProtocolInfoFact;
using strata::hpu::protocol_info_append;
using strata::hpu::vision_info_facts;

int failures = 0;
void check(const bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cout << "FAIL " << what << "\n";
    }
}
template <class F>
bool refused(F&& call) {
    try {
        call();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}
// The server's own parser: split on spaces, then key=value.
std::map<std::string, std::string> parse(const std::string& line) {
    std::map<std::string, std::string> facts;
    std::istringstream stream(line);
    std::string token;
    while (stream >> token) {
        const auto at = token.find('=');
        if (at == std::string::npos) continue;
        facts[token.substr(0, at)] = token.substr(at + 1);
    }
    return facts;
}
std::string render(const std::vector<ProtocolInfoFact>& facts) {
    return protocol_info_append("INFO backend=hpu", std::span<const ProtocolInfoFact>(facts));
}
} // namespace

int main() {
    // 1. a fact list renders as one space-separated key=value token per fact, in order, after the base line
    {
        const std::vector<ProtocolInfoFact> facts{{"logit_bias", "1"}, {"context", "32768"}, {"engine", "strata-hpu-0.1.0"}};
        const auto line = render(facts);
        check(line == "INFO backend=hpu logit_bias=1 context=32768 engine=strata-hpu-0.1.0",
              "facts append in order after the base line: " + line);
        const auto parsed = parse(line);
        check(parsed.at("logit_bias") == "1" && parsed.at("context") == "32768" &&
                  parsed.at("engine") == "strata-hpu-0.1.0" && parsed.at("backend") == "hpu",
              "the server's own parser recovers every fact");
    }

    // 2. an empty fact list leaves the line untouched
    check(protocol_info_append("INFO backend=hpu", {}) == "INFO backend=hpu",
          "no facts means no trailing space and no change");

    // 3. the values the server must read are all legal tokens
    {
        const std::vector<ProtocolInfoFact> facts{{"batch_slots", "4"}, {"batch_groups", "1"}, {"slot_cache", "1"},
                                                  {"arena_mib", "512"}, {"conversation_cache_mib", "1024"},
                                                  {"conversation_cache_slots", "8"}, {"kv_resident", "13"},
                                                  {"vram", "elastic"}, {"vram_free_mib", "0"}};
        const auto parsed = parse(render(facts));
        for (const char* key : {"batch_slots", "batch_groups", "slot_cache", "arena_mib", "conversation_cache_mib",
                                "conversation_cache_slots", "kv_resident", "vram", "vram_free_mib"})
            check(parsed.count(key) == 1, std::string("the server-consumed key is present: ") + key);
    }

    // 4. a value with whitespace is REFUSED: it would shift every following field for the server
    check(refused([] { protocol_info_append("INFO", {{{"engine", "two words"}}}); }), "a space in a value is refused");
    check(refused([] { protocol_info_append("INFO", {{{"engine", "tab\there"}}}); }), "a tab in a value is refused");
    check(refused([] { protocol_info_append("INFO", {{{"engine", "line\nbreak"}}}); }), "a newline in a value is refused");
    check(refused([] { protocol_info_append("INFO", {{{"engine", "carriage\rreturn"}}}); }), "a CR in a value is refused");

    // 5. a key that is not a single shell-safe token is REFUSED, so a fact can never masquerade as two
    check(refused([] { protocol_info_append("INFO", {{{"", "1"}}}); }), "an empty key is refused");
    check(refused([] { protocol_info_append("INFO", {{{"two keys", "1"}}}); }), "a space in a key is refused");
    check(refused([] { protocol_info_append("INFO", {{{"weird=key", "1"}}}); }), "a second '=' in a key is refused");
    check(refused([] { protocol_info_append("INFO", {{{"UPPER", "1"}}}); }), "an upper-case key is refused");

    // 6. the batch facts exist only while slots are on, and carry upstream's three keys (:9608-9610)
    check(batch_info_facts(0, 1, false).empty(), "no slots means no batch facts");
    {
        const auto facts = batch_info_facts(4, 2, true);
        const auto parsed = parse(render(facts));
        check(facts.size() == 3, "a batch line carries exactly three facts");
        check(parsed.at("batch_slots") == "4" && parsed.at("batch_groups") == "2" &&
                  parsed.at("slot_cache") == "1",
              "batch_slots/batch_groups/slot_cache carry the live numbers");
    }
    {
        const auto parsed = parse(render(batch_info_facts(2, 0, false)));
        check(parsed.at("batch_groups") == "1", "a zero group count is reported as one group");
        check(parsed.at("slot_cache") == "0", "a disabled slot cache is reported as 0");
    }

    // 7. the vision gate reports itself and the SVE1 width it enforces; off is 1/0 rather than an absent key
    {
        const auto enabled = parse(render(vision_info_facts(true, 3584)));
        check(enabled.at("vision") == "1" && enabled.at("vision_width") == "3584",
              "the enabled vision gate reports its width");
        const auto disabled = parse(render(vision_info_facts(false, 3584)));
        check(disabled.at("vision") == "0" && disabled.at("vision_width") == "0",
              "a disabled vision gate reports 0 and no width");
        check(vision_info_facts(true, 3584).size() == 2, "the vision gate adds exactly two facts");
    }

    if (failures != 0) {
        std::cout << "FAILED protocol-info checks: " << failures << "\n";
        return 1;
    }
    std::cout << "PASS protocol info: facts append in order after the base line and the server's own key=value "
                 "parser recovers them; every key the pinned server reads (batch_slots, batch_groups, slot_cache, "
                 "arena_mib, conversation_cache_mib, conversation_cache_slots, kv_resident, vram, vram_free_mib, "
                 "context, engine) is a legal token; a value with space/tab/CR/LF and a key that is empty, upper-case "
                 "or carries a space are all REFUSED instead of corrupting the line; the batch facts exist only "
                 "while slots are on; and the vision fact is always present as 1/0 with its enforced width; no device\n";
    return 0;
}
