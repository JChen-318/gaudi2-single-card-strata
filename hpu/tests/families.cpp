// hpu/tests/families.cpp - host evidence for the one place that says which GGUF architectures this build runs.
//
// Upstream's `strata --families` (src/program/generate.cpp:1811-1817) prints the compile-time list and exits before
// any model path exists. This port had that knowledge in TWO places - the literal comparison in hpu/src/weights.cpp and
// nothing at all for the report - so the report could not exist without being a second copy that might lie. It is one
// law now (strata/hpu/families.hpp), which these checks pin.
#include "strata/hpu/families.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace strata::hpu;
int failures = 0;
void check(const bool ok, const std::string& what) {
    if (!ok) { ++failures; std::cout << "FAIL " << what << "\n"; }
}
template <class F> bool refused(F&& call, const std::string& needle) {
    try { call(); } catch (const std::exception& error) {
        return std::string(error.what()).find(needle) != std::string::npos;
    }
    return false;
}
}

int main() {
    check(supported_families() == std::vector<std::string>({"qwen4exp"}),
          "this build runs the single architecture upstream's non-Qwen35 branch prints (:1815)");
    check(families_line() == "qwen4exp", "the report line is the names space-separated, as upstream prints them");
    check(join_families({"qwen4exp", "qwen35moe"}) == "qwen4exp qwen35moe",
          "two families are joined by exactly one space, which this one-family build cannot show by itself");
    check(!refused([] { require_supported_family("qwen4exp"); }, "requires"),
          "the supported family passes the reader's own check");
    check(refused([] { require_supported_family("qwen35moe"); }, "this build runs: qwen4exp (this file declares 'qwen35moe')"),
          "an unsupported family is refused WITH the list, so an operator learns what to install");
    check(refused([] { require_supported_family(""); }, "declares ''"),
          "a file whose architecture is absent or not a string is refused the same way, not treated as a match");
    check(refused([] { require_supported_family("qwen4exp qwen35moe"); }, "declares 'qwen4exp qwen35moe'"),
          "the comparison is exact: a name that merely starts with a supported one does not pass");
    if (failures) { std::cout << "families: " << failures << " failed\n"; return 1; }
    std::cout << "PASS one supported-family list, its report line, and the refusals that quote it\n";
    return 0;
}
