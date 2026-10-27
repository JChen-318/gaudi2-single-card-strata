// Host value-domain checks; reader budget and actual IO effects have separate tests.
#include "strata/hpu/ple_io_mode.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using strata::hpu::resolve_ple_delay_us;
using strata::hpu::resolve_ple_inflight;
using strata::hpu::resolve_ple_io;
using strata::hpu::resolve_ple_row_cache;
namespace k = strata::kernels;

int failures = 0;
void check(const bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cout << "FAIL " << what << "\n";
    }
}
std::string refusal(auto&& call) {
    try {
        (void)call();
    } catch (const std::invalid_argument& error) {
        return error.what();
    }
    return {};
}
bool has(const std::string& message, const std::string& needle) {
    return message.find(needle) != std::string::npos;
}
} // namespace

int main() {
    check(!strata::hpu::reported_ple_ram_lock(false),"a failed OS lock is never reported successful");
    check(strata::hpu::reported_ple_ram_lock(true)==!strata::hpu::kPleLockResultSimulated,
          "sanitizer's simulated lock success is not reported as physical residency");
    // 1. the three reader modes, where `ram` is mmap PLUS the whole-table lock (not a third reader)
    {
        const auto direct = resolve_ple_io("direct", {});
        check(direct.mode == k::PleIo::Direct && !direct.lock, "direct is the direct reader, no lock");
        const auto mmap = resolve_ple_io("mmap", {});
        check(mmap.mode == k::PleIo::Mmap && !mmap.lock, "mmap is the mapped reader, no lock");
        const auto ram = resolve_ple_io("ram", {});
        check(ram.mode == k::PleIo::Mmap && ram.lock, "ram is mmap with lock set");
        strata::hpu::validate_ple_io_policy(direct);strata::hpu::validate_ple_io_policy(mmap);
        strata::hpu::validate_ple_io_policy(ram);
        auto wrong=ram;wrong.mode=k::PleIo::Direct;
        check(has(refusal([&]{strata::hpu::validate_ple_io_policy(wrong);}),"requires mmap"),
              "RAM locking cannot silently become ignored in direct mode");
    }
    // 2. anything else gets upstream's message
    for (const char* other : {"", "DIRECT", "io_uring", "mmap ", "ramdisk"})
        check(refusal([&] { (void)resolve_ple_io(other, {}); }) ==
                  "invalid --ple-io/--ple-row-cache/--ple-inflight/--ple-delay-us",
              std::string("unknown mode gets upstream's message: '") + other + "'");

    // 3. the row cache: nonnegative, with large capacities admitted by checked byte pricing
    check(resolve_ple_row_cache(0, {}).cache_rows == 0, "0 disables the row cache");
    check(resolve_ple_row_cache(4096, {}).cache_rows == 4096, "a small cache is taken as given");
    check(resolve_ple_row_cache(int64_t(strata::hpu::kPleRowCacheRows), {}).cache_rows ==
              uint64_t(strata::hpu::kPleRowCacheRows),
          "the priced default is accepted");
    check(refusal([] { (void)resolve_ple_row_cache(-1, {}); }) ==
              "invalid --ple-io/--ple-row-cache/--ple-inflight/--ple-delay-us",
          "a negative cache is refused with upstream's message");
    check(resolve_ple_row_cache(8ll<<20,{}).cache_rows==(8ull<<20),"larger cache is represented for checked byte admission");

    // 4. the full upstream inflight range
    check(resolve_ple_inflight(1, {}).max_inflight == 1, "one outstanding read is accepted");
    check(resolve_ple_inflight(256, {}).max_inflight == 256, "the priced bound is accepted");
    for (const int64_t bad : {int64_t(0), int64_t(-1), int64_t(1025), int64_t(4096)})
        check(refusal([&] { (void)resolve_ple_inflight(bad, {}); }) ==
                  "invalid --ple-io/--ple-row-cache/--ple-inflight/--ple-delay-us",
              "outside upstream's 1..1024 is refused: " + std::to_string(bad));
    for(const int64_t reads:{257,512,1024}) check(resolve_ple_inflight(reads,{}).max_inflight==reads,"full upstream inflight range");

    // 5. the injected delay: finite and >= 0, and it is NOT a PleIoOptions field (the reader owns it)
    check(refusal([] { (void)resolve_ple_delay_us(-1.0); }) ==
              "invalid --ple-io/--ple-row-cache/--ple-inflight/--ple-delay-us",
          "a negative delay is refused");
    check(refusal([] { (void)resolve_ple_delay_us(std::numeric_limits<double>::quiet_NaN()); }).size() != 0,
          "a NaN delay is refused");
    check(refusal([] { (void)resolve_ple_delay_us(std::numeric_limits<double>::infinity()); }).size() != 0,
          "an infinite delay is refused");
    {
        check(resolve_ple_delay_us(3000.0)==3000.0,"positive delay retains its actual requested value");
        check(resolve_ple_delay_us(0.0)==0.0,"zero disables injection");
    }

    if (failures != 0) {
        std::cout << "FAILED ple-io checks: " << failures << "\n";
        return 1;
    }
    std::cout << "PASS ple io: direct/mmap/ram policy, nonnegative row counts for byte admission, full 1..1024 inflight, finite nonnegative delay; no device\n";
    return 0;
}
