// hpu/tests/version_law.cpp - the version this build reports is the version it WRITES into its INFO line.
//
// Upstream answers `strata --version` / `-V` in main()'s pre-scan loop (src/program/generate.cpp:1804-1809) by
// printing "strata <STRATA_VERSION>" before any GPU is touched, and STRATA_VERSION is the same literal its INFO line
// emits as `engine=` (:9595). The pinned server has no `version=` fact to read: it takes BUILD.json's version and
// otherwise falls back to the engine value (serve/server.py:790, :895). So a port that printed a version of its own
// invention would tell the shell one thing and the server another - which is exactly what these checks refuse.
#include "strata/hpu/protocol_info.hpp"

#include <iostream>
#include <string>
#include <vector>

namespace {
using namespace strata::hpu;
int failures = 0;
void check(const bool ok, const std::string& what) {
    if (!ok) { ++failures; std::cout << "FAIL " << what << "\n"; }
}
}

int main() {
    check(engine_value() == "strata-hpu-0.1.0",
          "the engine fact names this backend and carries its version (the value a server reads as `engine=`)");
    check(version_line() == "strata " + engine_value(),
          "the report line is BUILT FROM the engine fact, so it cannot drift from what the server reads");
    check(version_line().rfind("strata ", 0) == 0, "upstream's `strata ` prefix is kept in the report line");
    const std::vector<ProtocolInfoFact> facts{{"engine", engine_value()}};
    bool refused = false;
    std::string line;
    try { line = protocol_info_append("INFO", facts); } catch (const std::exception&) { refused = true; }
    check(!refused && line == "INFO engine=strata-hpu-0.1.0",
          "the engine fact still satisfies the INFO-line rule: a safe key and a value with no whitespace");
    const std::string prefix = "strata-hpu-";
    check(engine_value().size() > prefix.size() && engine_value().rfind(prefix, 0) == 0
          && engine_value().substr(prefix.size()) == std::string(strata_hpu_engine_version),
          "the version constant is what the engine fact ends with, not a second copy of it");
    if (failures) { std::cout << "version_law: " << failures << " failed\n"; return 1; }
    std::cout << "PASS one engine-version law behind both `--version` and the INFO line's engine fact\n";
    return 0;
}
