// hpu/tests/kernel_path.cpp - host evidence for where an INSTALLED backend finds its TPC kernel library.
//
// The build bakes the absolute BUILD-TREE path of libstrata_hpu_tpc.so into strata_hpu (as one entry of a list that
// also carries the SDK's own kernel paths); cmake --install also puts that file in <prefix>/lib. strata/hpu/
// kernel_path.hpp is the resolution rule, and this test pins it against real directories (nothing here compares
// strings to strings): an existing entry is used as-is, a MISSING entry whose file name is this backend's own
// kernel is retried next to the executable, in ../lib and in ../lib64, and every other entry - an SDK kernel path,
// an unrelated name, a path that exists nowhere - comes back UNCHANGED so the SDK names the real path.
#include "strata/hpu/kernel_path.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {
using strata::hpu::resolve_kernel_entry;

int failures = 0;
void check(const bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cout << "FAIL " << what << "\n";
    }
}

void touch(const std::filesystem::path& path) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    output << "fake kernel";
}

bool same(const std::string& a, const std::filesystem::path& b) {
    return std::filesystem::path(a).lexically_normal() == b.lexically_normal();
}
const std::string kOwn = "libstrata_hpu_tpc.so";
} // namespace

int main() {
    const auto root = std::filesystem::temp_directory_path() / "strata-hpu-kernel-path";
    std::filesystem::remove_all(root);
    const auto prefix = root / "prefix";
    const auto build = root / "build";
    // an installed layout: <prefix>/bin/strata-hpu and <prefix>/lib/libstrata_hpu_tpc.so
    touch(prefix / "bin" / "strata-hpu");
    touch(prefix / "lib" / kOwn);
    // a build tree that still exists
    touch(build / kOwn);

    const auto executable_dir = (prefix / "bin").string();

    // 1. an entry that exists is used AS IS: the build tree keeps working and an SDK path is never rewritten
    const auto build_entry = (build / kOwn).string();
    check(resolve_kernel_entry(build_entry, executable_dir, kOwn) == build_entry,
          "an existing kernel entry is used unchanged");

    // 2. a build-tree entry whose file is gone resolves to the INSTALLED library, via ../lib
    const auto gone_entry = (root / "deleted-build" / kOwn).string();
    check(!std::filesystem::exists(gone_entry), "the fixture's build-tree path really is gone");
    check(same(resolve_kernel_entry(gone_entry, executable_dir, kOwn), prefix / "lib" / kOwn),
          "a missing own-kernel entry resolves to the installed library in ../lib");

    // 3. a copy NEXT TO the executable is preferred over ../lib (it is tried first)
    const auto side = prefix / "bin" / kOwn;
    touch(side);
    check(same(resolve_kernel_entry(gone_entry, executable_dir, kOwn), side),
          "a library next to the executable is preferred over ../lib");
    std::filesystem::remove(side);

    // 4. lib64 is the other install convention
    std::filesystem::rename(prefix / "lib", prefix / "lib64");
    check(same(resolve_kernel_entry(gone_entry, executable_dir, kOwn), prefix / "lib64" / kOwn),
          "a ../lib64 install is found as well");
    std::filesystem::remove_all(prefix / "lib64");

    // 5. an own-kernel entry that exists NOWHERE is returned unchanged: the SDK's error names the real path
    check(resolve_kernel_entry(gone_entry, executable_dir, kOwn) == gone_entry,
          "an unresolvable own-kernel entry is passed through unchanged, never silently dropped");
    check(resolve_kernel_entry(gone_entry, "", kOwn) == gone_entry, "with no executable directory the entry is unchanged");

    // 6. SDK entries are never touched, even when a same-named file sits next to the executable
    touch(prefix / "bin" / "some_sdk_kernel.elf");
    const auto sdk_entry = (root / "opt" / "synapse" / "some_sdk_kernel.elf").string();
    check(resolve_kernel_entry(sdk_entry, executable_dir, kOwn) == sdk_entry,
          "a missing SDK kernel entry is not replaced by an unrelated same-named file");

    // 7. an empty entry or an empty own-name disables the rule rather than guessing
    check(resolve_kernel_entry("", executable_dir, kOwn).empty(), "an empty entry stays empty");
    check(resolve_kernel_entry(gone_entry, executable_dir, "") == gone_entry,
          "with no own-kernel name nothing is rewritten");

    std::filesystem::remove_all(root);
    if (failures != 0) {
        std::cout << "FAILED kernel-path checks: " << failures << "\n";
        return 1;
    }
    std::cout << "PASS kernel path: an existing kernel entry is used unchanged; a build-tree entry whose file is gone "
                 "resolves to the installed library next to the executable, in ../lib and in ../lib64; an entry that "
                 "exists nowhere is passed through unchanged (so the SDK names the real path); a missing SDK entry is "
                 "never replaced by a same-named file; and an empty entry stays empty; no device\n";
    return 0;
}
