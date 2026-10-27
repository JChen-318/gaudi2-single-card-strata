#pragma once
// hpu/include/strata/hpu/kernel_path.hpp - where an INSTALLED backend finds its TPC kernel library.
//
// The build bakes the absolute path of the compiled kernel (libstrata_hpu_tpc.so) into strata_hpu through
// STRATA_HPU_DEFAULT_KERNEL_PATH, because the runtime hands GC_KERNEL_PATH to the Synapse SDK. That path is a
// BUILD-TREE path: an installed prefix cannot reach it. cmake --install therefore also puts the same file in
// <prefix>/lib, and this header is the rule that finds it - a pure filesystem lookup with no SDK call, so a host
// test can pin it (hpu/tests/kernel_path.cpp) without a card.
//
// The law: the entry whose FILE NAME is this backend's own kernel library and which does not exist is retried next
// to the executable and in ../lib / ../lib64 (the places cmake --install puts it); every other entry - an existing
// one, an SDK kernel path, anything else - is returned UNCHANGED, so the SDK's own "kernel not found" message names
// the real path and an unrelated same-named file can never stand in for an SDK library.
#include <filesystem>
#include <string>

namespace strata::hpu {

inline std::string resolve_kernel_entry(const std::string& entry, const std::string& executable_dir,
                                        const std::string& own_kernel_name) {
    if (entry.empty() || own_kernel_name.empty()) return entry;
    std::error_code error;
    if (std::filesystem::exists(entry, error) && !error) return entry;
    const auto name = std::filesystem::path(entry).filename().string();
    if (name != own_kernel_name || executable_dir.empty()) return entry;
    const std::filesystem::path directory(executable_dir);
    for (const auto& candidate : {directory / name, directory / ".." / "lib" / name,
                                  directory / ".." / "lib64" / name}) {
        error.clear();
        if (std::filesystem::exists(candidate, error) && !error)
            return candidate.lexically_normal().string();
    }
    return entry;
}
} // namespace strata::hpu
