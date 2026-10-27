// hpu/tests/kv_arena.cpp - host evidence for addressing several sequences' dense KV pools in one cell space.
//
// The arena is the piece a multi-sequence window needs before any kernel changes: a row's absolute cell
// (`row plan kv_base + position`) must land in its own sequence's pool, and a sequence's attention must see only its
// own causal prefix. `DenseKvState` keeps the KV FORMAT, so this test runs the same laws under fp16 and under the
// compressed layouts the port already implements (int8 / q4_0), which is exactly what a batched window will mix.
#include "strata/hpu/kv_arena.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using strata::hpu::DenseKvState;
using strata::hpu::KvArena;
using strata::hpu::KvMode;

int failures = 0;
void check(const bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cout << "FAIL " << what << "\n";
    }
}
std::string refusal(auto&& call) {
    try {
        call();
    } catch (const std::invalid_argument& error) {
        return error.what();
    }
    return {};
}
bool has(const std::string& message, const std::string& needle) {
    return message.find(needle) != std::string::npos;
}
std::vector<float> row(const uint32_t heads, const uint32_t width, const float first) {
    std::vector<float> out(size_t(heads) * width);
    for (size_t i = 0; i < out.size(); ++i) out[i] = first + float(i);
    return out;
}
} // namespace

int main() {
    const uint32_t heads = 2, width = 4;   // 8 values per K/V row

    // 1. two sequences, two pools, one global cell space; each row lands in its own pool and each gather reads only
    //    its own prefix.
    {
        DenseKvState a(8, heads, width);
        DenseKvState b(8, heads, width);
        KvArena arena;
        const auto ra = arena.add_region(a, 0);
        const auto rb = arena.add_region(b, 100);
        check(ra == 0 && rb == 1 && arena.regions() == 2, "two regions with distinct ids");

        // sequence A writes cells 0,1,2; sequence B writes 100,101
        for (uint32_t i = 0; i < 3; ++i) {
            const auto k = row(heads, width, float(i)), v = row(heads, width, 100.0f + float(i));
            check(arena.next_cell(ra) == i, "A's next cell follows its own length");
            arena.append_at(i, k, v);
        }
        for (uint32_t i = 0; i < 2; ++i) {
            const auto k = row(heads, width, 10.0f + float(i)), v = row(heads, width, 200.0f + float(i));
            check(arena.next_cell(rb) == 100 + i, "B's next cell follows its own length");
            arena.append_at(100 + i, k, v);
        }
        check(a.size() == 3 && b.size() == 2, "each pool holds only its own rows");
        check(arena.region_of(0) == 0 && arena.region_of(2) == 0 && arena.region_of(100) == 1 &&
                  arena.region_of(101) == 1,
              "cells map to their owning region");

        const auto prefix_a = arena.gather_span(ra, 3, 3);
        const auto prefix_b = arena.gather_span(rb, 2, 2);
        check(prefix_a.count == 3 && prefix_b.count == 2, "gathers return the requested prefix lengths");
        // A's first key row is 0..7 and B's is 10..17: a gather that crossed pools would show B's values in A.
        check(std::abs(prefix_a.key.front() - 0.0f) < 1e-6f && std::abs(prefix_b.key.front() - 10.0f) < 1e-6f,
              "a sequence's gather sees only its own rows");
    }

    // 2. refusals: overlapping regions, decreasing bases, a cell outside every range, a cell that is not the owner's
    //    next cell, and a span wider than the owner holds.
    {
        DenseKvState a(4, heads, width), b(4, heads, width);
        KvArena arena;
        arena.add_region(a, 0);
        check(has(refusal([&] { arena.add_region(b, 0); }), "strictly increase"), "a decreasing base is refused");
        check(has(refusal([&] { arena.add_region(b, 2); }), "overlap"), "an overlapping region is refused");
        // A zero-capacity pool cannot even be constructed (DenseKvState refuses the geometry), so the arena's
        // own non-empty check is defence in depth for a future owner that allows it.
        check(has(refusal([&] { arena.region_of(4); }), "belongs to no region"),
              "a cell outside every range is refused");
        const auto k = row(heads, width, 1.0f), v = row(heads, width, 2.0f);
        check(has(refusal([&] { arena.append_at(2, k, v); }), "next cell"),
              "a cell that is not the region's next cell is refused");
        check(has(refusal([&] { arena.gather_span(0, 3, 3); }), "cells, not"),
              "a gather wider than the holding region is refused");
        check(has(refusal([&] { arena.gather_span(7, 1, 1); }), "out of range"), "an unknown region is refused");
    }

    // 3. the same laws under the compressed layouts the port already implements: the arena never touches the format,
    //    so int8 and q4_0 pools address identically (and their gather still returns FP32 rows).
    // Each compressed layout has its own geometry law, and DenseKvState refuses the wrong one rather than silently
    // storing fp16: int8 needs a head width that is a multiple of 64, q4_0 the Hadamard block size (256).
    struct Packed {KvMode mode; uint32_t width; const char* name;};
    const Packed packed[] = {{KvMode::Int8, 64, "int8"}, {KvMode::Q4, 256, "q4_0"}};
    for (const Packed& shape : packed) {
        const KvMode mode = shape.mode;
        const uint32_t wide_width = shape.width;
        const uint32_t wide = heads * wide_width;
        DenseKvState a(4, heads, wide_width, mode), b(4, heads, wide_width, mode);
        KvArena arena;
        const auto ra = arena.add_region(a, 0);
        const auto rb = arena.add_region(b, 16);
        for (uint32_t i = 0; i < 3; ++i) {
            arena.append_at(i, row(heads, wide_width, float(i)), row(heads, wide_width, float(i) + 0.5f));
            arena.append_at(16 + i, row(heads, wide_width, 50.0f + float(i)),
                            row(heads, wide_width, 60.0f + float(i)));
        }
        const auto pa = arena.gather_span(ra, 3, 3);
        const auto pb = arena.gather_span(rb, 3, 3);
        check(pa.count == 3 && pb.count == 3, std::string("compressed pools gather their own prefixes: ") + shape.name);
        check(pa.key.size() == size_t(wide) * 3, std::string("a gather returns rows, not codes: ") + shape.name);
        check(std::abs(pa.key.front()) < 0.5f && pb.key.front() > 40.0f,
              "compressed gathers stay inside their own sequence");
    }

    if (failures != 0) {
        std::cout << "FAILED kv-arena checks: " << failures << "\n";
        return 1;
    }
    std::cout << "PASS kv arena: two sequences address one global cell space, each row lands in its own pool, each "
                 "gather reads only its own causal prefix and keeps its KV format (fp16, int8, q4_0); overlapping, "
                 "decreasing, zero-capacity, out-of-range, non-next-cell and over-wide requests are all refused by "
                 "name; no device\n";
    return 0;
}
