// Host test for the parked-conversation K/V page law (hpu/src/parked_pages.cpp).
//
// The layout arithmetic is pinned two ways: against an independent transcription of the pinned upstream
// function it ports (src/core/conversation_snapshot.cpp:29-73, mirroring the same recomputation
// upstream's own conversation_snapshot_test.cpp:54-59 does), and against properties a wrong byte map
// cannot satisfy - every part is an exact multiple of its per-cell stride, the cell count is a whole number
// of pages, and the pooled row count carries the moving spare row. The reuse half is pinned to upstream's
// save loop (:147-181): keep runs to the last WHOLE page, the partial page and the spare indexer row are
// always re-read, reused+copied equals the layout exactly, and a prefix larger than the retained image is
// refused rather than silently shortened. The format tag is read from the pinned upstream enum, not typed.
// No device, no allocation, no claim about what a real KV pool contains.
#include "strata/hpu/parked_pages.hpp"

#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/qsa.hpp"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace strata::hpu;
namespace {

uint32_t layout_cases = 0, reuse_cases = 0, refusal_cases = 0, capture_cases = 0;

void require(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }

struct Format { bool q4, int8, hybrid, rot, ring; };
const char* format_name(const Format& f) {
    if (f.hybrid) return "k8v4";
    if (f.q4) return "q4_0";
    if (f.int8) return "int8";
    return "f16";
}

// the transcription: the same formulas read out of conversation_snapshot.cpp:42-71
struct Oracle { uint64_t cells, pooled; int format; uint64_t sizes[5]; };
Oracle oracle(const Format& f, uint64_t upto, bool index, uint64_t head_dim, uint64_t heads, uint64_t idx_dim) {
    const auto s = strata::kernels::qsa_real_shapes();
    const bool int8_keys = f.int8 || f.hybrid;
    Oracle o{};
    o.cells = ((upto + s.page_size - 1) / s.page_size) * s.page_size;
    o.pooled = index && upto ? upto / s.idx_block + 1 : 0;
    const int base = f.hybrid ? strata::kernels::kKvHybrid
                              : f.q4 ? strata::kernels::kKvQ4 : f.int8 ? strata::kernels::kKvInt8
                                                                       : strata::kernels::kKvF16;
    o.format = base + (f.rot && !f.q4 && !f.hybrid ? 16 : 0);
    const uint64_t per = f.q4 ? strata::kernels::kv_q4_bytes_per_head(static_cast<int>(head_dim))
                              : head_dim * (int8_keys ? 1 : 2);
    o.sizes[0] = o.cells * heads * per;
    o.sizes[2] = o.cells * heads * (int8_keys && !f.q4 ? (head_dim / 64) * 2 : 0);
    o.sizes[4] = o.pooled * idx_dim * sizeof(float);
    o.sizes[1] = o.sizes[0];
    o.sizes[3] = o.sizes[2];
    if (f.hybrid) {
        o.sizes[1] = o.cells * heads * strata::kernels::kv_q4_bytes_per_head(static_cast<int>(head_dim));
        o.sizes[3] = 0;
    }
    return o;
}

void layouts() {
    const auto s = strata::kernels::qsa_real_shapes();
    const uint64_t head_dim = s.head_dim, heads = s.n_head_kv, idx_dim = s.idx_dim;
    const std::vector<Format> formats{{false,false,false,false,false}, {false,false,false,true,false},
                                      {false,true,false,false,false}, {false,true,false,true,false},
                                      {true,false,false,true,false}, {false,false,true,false,false}};
    const std::vector<uint64_t> extents{0, 1, 3, 4, 5, 7, 8, 31, 32, 33, 127, 128, 2049, 4096, 4097};
    for (const auto& f : formats) {
        for (uint64_t upto : extents) {
            for (bool index : {false, true}) {
                ParkedKvLayout l;
                std::string error;
                require(parked_kv_layout(f.q4, f.int8, f.hybrid, f.rot, f.ring, upto, index, head_dim, heads,
                                         idx_dim, l, error),
                        std::string(format_name(f)) + " extent " + std::to_string(upto) + " rejected: " + error);
                const Oracle o = oracle(f, upto, index, head_dim, heads, idx_dim);
                require(l.cells == o.cells && l.pooled_rows == o.pooled && l.format == o.format,
                        std::string(format_name(f)) + " extent " + std::to_string(upto) +
                        " disagrees with the transcribed layout: cells " + std::to_string(l.cells) + "/" +
                        std::to_string(o.cells) + " rows " + std::to_string(l.pooled_rows) + "/" +
                        std::to_string(o.pooled) + " format " + std::to_string(l.format) + "/" +
                        std::to_string(o.format));
                for (unsigned part = 0; part < parked_parts; ++part)
                    require(l.sizes[part] == o.sizes[part],
                            std::string(format_name(f)) + " part " + std::to_string(part) + " is " +
                            std::to_string(l.sizes[part]) + " bytes, the pinned formulas give " +
                            std::to_string(o.sizes[part]));
                // properties a wrong byte map cannot satisfy
                require(l.cells % s.page_size == 0 && l.cells >= upto, "cells are not page-rounded");
                require(l.cells == 0 || upto == 0 || (l.cells - s.page_size) < upto, "cells overshoot a page");
                for (unsigned part : {0u, 1u, 2u, 3u})
                    require(l.cells == 0 || l.sizes[part] % l.cells == 0,
                            "part " + std::to_string(part) + " is not cell-periodic");
                require(index == (l.sizes[4] != 0) || (upto == 0), "pooled rows without the index flag");
                ++layout_cases;
            }
        }
    }
    // the per-cell byte figures the upstream header documents: 512 B f16, 256 B int8 (+8 scales),
    // 144 B per Q4_0 head, and K8V4's mixed pair
    ParkedKvLayout l; std::string error;
    require(parked_kv_layout(false,false,false,false,false, 1, true, head_dim, heads, idx_dim, l, error), "f16");
    require(l.sizes[0] / l.cells == heads * head_dim * 2, "f16 K is not 2 bytes per value per head");
    require(parked_kv_layout(false,true,false,false,false, 1, true, head_dim, heads, idx_dim, l, error), "int8");
    require(l.sizes[0] / l.cells == heads * head_dim && l.sizes[2] / l.cells == heads * (head_dim / 64) * 2,
            "int8 K/scale stride is wrong");
    require(parked_kv_layout(true,false,false,false,false, 1, true, head_dim, heads, idx_dim, l, error), "q4");
    require(l.sizes[0] / l.cells == heads * 144 && l.sizes[2] == 0, "Q4_0 K is not 144 bytes per head");
    require(parked_kv_layout(false,false,true,false,false, 1, true, head_dim, heads, idx_dim, l, error), "k8v4");
    require(l.sizes[0] / l.cells == heads * head_dim && l.sizes[1] / l.cells == heads * 144 && l.sizes[3] == 0,
            "K8V4 does not mix INT8 K with Q4_0 V");
    ++layout_cases;
}

void refusals() {
    const auto s = strata::kernels::qsa_real_shapes();
    ParkedKvLayout l; std::string error;
    const std::vector<std::pair<Format, const char*>> bad{
        {{false,false,true,false,true}, "k8v4 cannot also be a streaming ring"},
        {{true,false,true,false,false}, "k8v4 carries its own V quantization"},
        {{false,true,true,false,false}, "k8v4 and int8 set the same flag twice"},
    };
    for (const auto& item : bad) {
        require(!parked_kv_layout(item.first.q4, item.first.int8, item.first.hybrid, item.first.rot,
                                  item.first.ring, 8, true, s.head_dim, s.n_head_kv, s.idx_dim, l, error),
                std::string("accepted ") + item.second);
        require(!error.empty(), std::string("no reason given for rejecting ") + item.second);
        ++refusal_cases;
    }
    // geometry that cannot tile: Q4_0 needs whole 32-value blocks, INT8 scales whole 64-value blocks
    for (uint64_t dim : {31ull, 63ull}) {
        const bool q4_ok = parked_kv_layout(true,false,false,false,false, 8, true, dim, s.n_head_kv, s.idx_dim, l, error);
        const bool int8_ok = parked_kv_layout(false,true,false,false,false, 8, true, dim, s.n_head_kv, s.idx_dim, l, error);
        require(!q4_ok || dim % strata::kernels::QK4_0 == 0, "a head dimension that is not whole Q4_0 blocks was accepted");
        require(!int8_ok || dim % 64 == 0, "a head dimension that is not whole INT8 scale blocks was accepted");
        ++refusal_cases;
    }
    require(!parked_kv_layout(false,false,false,false,false, 8, true, 0, s.n_head_kv, s.idx_dim, l, error),
            "a zero head dimension was accepted");
    require(!parked_kv_layout(false,false,false,false,false, 1ull << 60, true, s.head_dim, s.n_head_kv,
                              s.idx_dim, l, error) &&
            error.find("overflow") != std::string::npos, "a byte count that overflows was not refused");
    ++refusal_cases;
}

void reuse() {
    const auto s = strata::kernels::qsa_real_shapes();
    const std::vector<Format> formats{{false,false,false,false,false}, {false,true,false,true,false},
                                      {true,false,false,false,false}, {false,false,true,false,false}};
    for (const auto& f : formats) {
        for (uint64_t upto : {4ull, 8ull, 12ull, 100ull, 4097ull}) {
            ParkedKvLayout l; std::string error;
            require(parked_kv_layout(f.q4, f.int8, f.hybrid, f.rot, f.ring, upto, true, s.head_dim,
                                     s.n_head_kv, s.idx_dim, l, error), "layout for the reuse sweep");
            // Upstream's guard (:153) tests the extent as asked for, not the page-rounded buffer length, so
            // the sweep runs to `upto` and the padding range is asserted to be refused instead.
            for (uint64_t padding = l.upto + 1; padding <= l.cells; ++padding) {
                const std::vector<uint64_t> prior(l.sizes, l.sizes + parked_parts);
                ParkedKvReuse r;
                require(!parked_kv_reuse(l, prior.data(), padding, r, error) &&
                        error.find("invalid unchanged prefix") != std::string::npos,
                        std::string(format_name(f)) + " accepted an unchanged count inside the last page's padding");
                ++refusal_cases;
            }
            for (uint64_t unchanged = 0; unchanged <= l.upto; ++unchanged) {
                // an image retained at this exact extent is what the caller may reuse from
                const std::vector<uint64_t> prior(l.sizes, l.sizes + parked_parts);
                ParkedKvReuse r;
                require(parked_kv_reuse(l, prior.data(), unchanged, r, error),
                        std::string(format_name(f)) + " extent " + std::to_string(upto) + " unchanged " +
                        std::to_string(unchanged) + " refused: " + error);
                uint64_t total_reuse = 0, total_copy = 0;
                const uint64_t whole = (unchanged / l.page_size) * l.page_size;
                for (unsigned part = 0; part < parked_parts; ++part) {
                    require(r.keep[part] + r.copied[part] == l.sizes[part],
                            "keep and copied do not add up to the layout");
                    total_reuse += r.keep[part]; total_copy += r.copied[part];
                    if (part == 4) {
                        // the indexer keeps whole completed blocks and always re-reads the moving spare row
                        const uint64_t expected = (unchanged / l.idx_block) * l.idx_dim * sizeof(float);
                        require(r.keep[part] == expected, "the pooled prefix is not whole indexer blocks");
                    } else if (l.cells) {
                        require(r.keep[part] == (l.sizes[part] / l.cells) * whole,
                                "the K/V prefix is not whole pages of this part's cell stride");
                        require(r.keep[part] % l.page_size == 0, "the prefix is not page-aligned");
                    }
                }
                require(r.reused_bytes == total_reuse, "the reused-bytes counter is not the sum of the parts");
                // the page rule that matters: everything from the first rewritten token's page onward is
                // re-read, so a partial page is never carried over
                if (unchanged < l.cells) {
                    require(r.copied[0] >= (l.cells - whole) * (l.sizes[0] / (l.cells ? l.cells : 1)),
                            "the tail page was not re-copied");
                    require(r.copied[4] >= (l.cells - unchanged) / l.idx_block * l.idx_dim * sizeof(float) ||
                            l.sizes[4] == 0, "the spare indexer row was not re-copied");
                } else {
                    // "Nothing changed" still re-reads the indexer's moving spare row: part 4's prefix is
                    // whole completed blocks while the layout carries one row more than the completed count
                    // (conversation_snapshot.cpp:45-47, :171), so this is the one part that is never fully
                    // carried over.
                    for (unsigned part = 0; part < 4; ++part)
                        require(r.copied[part] == 0, "an unchanged image still copied K/V bytes");
                    require(r.copied[4] == l.sizes[4] - (l.cells / l.idx_block) * l.idx_dim * sizeof(float),
                            "the spare indexer row was not the only thing re-read");
                }
                ++reuse_cases;
            }
        }
    }
    // the draft layer re-refreshes its last page: max(0, unchanged - 1) (conversation_state.cpp:291-294).
    // On a page boundary that one token costs a whole page of reuse, which is the point of the rule.
    ParkedKvLayout l; std::string error;
    require(parked_kv_layout(false,true,false,false,false, 8, true, s.head_dim, s.n_head_kv, s.idx_dim, l, error),
            "int8 layout for the draft page rule");
    ParkedKvReuse at_boundary, one_back;
    require(parked_kv_reuse(l, l.sizes, 4, at_boundary, error), "reuse at the page boundary");
    require(parked_kv_reuse(l, l.sizes, 3, one_back, error), "reuse one token back");
    // at 4 the last page is complete and carried; one token back it is not, so exactly one page of
    // this part stops being reused - the reason the draft layer pays that token
    // (conversation_state.cpp:291-294).
    const uint64_t per_cell = l.sizes[0] / l.cells;
    require(at_boundary.keep[0] - one_back.keep[0] == per_cell * l.page_size,
            "one token back did not drop exactly one page of reuse");
    ++reuse_cases;
    // a caller that believes in more retained prefix than the image holds is refused, not shortened
    std::vector<uint64_t> small(l.sizes, l.sizes + parked_parts); small[0] = 8;
    ParkedKvReuse r;
    require(!parked_kv_reuse(l, small.data(), 8, r, error) &&
            error.find("missing reusable prefix") != std::string::npos,
            "a prefix the image does not hold was not refused");
    ++refusal_cases;
    require(!parked_kv_reuse(l, nullptr, 0, r, error), "a missing retained image was accepted");
    ++refusal_cases;
    // unchanged cannot exceed the extent being written (upstream's third guard, in unsigned terms)
    require(!parked_kv_reuse(l, l.sizes, l.upto + 1, r, error) &&
            error.find("invalid unchanged prefix") != std::string::npos,
            "an unchanged prefix longer than the extent was accepted");
    ++refusal_cases;
}

}

// Upstream's capture-side pricing (src/core/conversation_snapshot.cpp:124-145): a park is charged what the
// image's own parts cost at the extent being parked, not what the whole arena reservation can hold.
void capture() {
    const auto s = strata::kernels::qsa_real_shapes();
    const uint64_t head_dim = s.head_dim, heads = s.n_head_kv, idx_dim = s.idx_dim;
    const std::vector<Format> formats{{false,false,false,false,false}, {false,true,false,true,false},
                                      {true,false,false,false,false}, {false,false,true,false,false}};
    const std::vector<uint64_t> extents{1, 4, 5, 33, 4097};
    for (const auto& f : formats) {
        const bool int8_keys = f.int8 || f.hybrid;
        const uint64_t per = f.q4 ? strata::kernels::kv_q4_bytes_per_head(static_cast<int>(head_dim))
                                  : head_dim * (int8_keys ? 1 : 2);
        const uint64_t scales = int8_keys && !f.q4 ? (head_dim / 64) * 2 : 0;
        const uint64_t v_per = f.hybrid ? strata::kernels::kv_q4_bytes_per_head(static_cast<int>(head_dim)) : per;
        for (uint64_t upto : extents) {
            ParkedKvLayout l; std::string error;
            require(parked_kv_layout(f.q4,f.int8,f.hybrid,f.rot,f.ring,upto,true,head_dim,heads,idx_dim,l,error),
                    "layout for the capture sweep");
            uint64_t total = 0;
            require(parked_kv_image_bytes(l, upto, total, error),
                    std::string("capture pricing refused its own case: ") + error);
            // The independent statement: the exact extent's rows, each head's bytes, and the indexer rows the
            // extent completes plus the one spare row.
            // Both scale parts (K and V) carry their own bytes, except in the hybrid image where V is Q4_0 and
            // has no scales at all - the same asymmetry the layout law reports.
            const uint64_t expected = upto*heads*per + upto*heads*v_per
                                    + upto*heads*scales*(f.hybrid ? 1 : (scales ? 2 : 0))
                                    + (upto/s.idx_block+1)*idx_dim*sizeof(float);
            require(total == expected, std::string(format_name(f)) + " extent " + std::to_string(upto) +
                    " priced at " + std::to_string(total) + ", not the parked image's " + std::to_string(expected));
            // At the page-rounded extent the charge is the layout total itself - the padding is what the port
            // names as its own divergence, not a different formula.
            uint64_t rounded = 0;
            require(parked_kv_image_bytes(l, l.cells, rounded, error) &&
                    rounded == l.sizes[0]+l.sizes[1]+l.sizes[2]+l.sizes[3]+l.sizes[4],
                    "the rounded extent is not the layout's own byte total");
            // One more cell costs exactly one more cell of K, V and scales, and crossing into a new indexer
            // block costs exactly one more pooled row - the row the spare keeps re-reading.
            if (upto > 1) {
                ParkedKvLayout back;
                require(parked_kv_layout(f.q4,f.int8,f.hybrid,f.rot,f.ring,upto-1,true,head_dim,heads,idx_dim,back,error),
                        "layout one cell back");
                uint64_t before = 0;
                require(parked_kv_image_bytes(back, upto-1, before, error), "capture pricing refused one back");
                const uint64_t rows = ((upto/s.idx_block) - ((upto-1)/s.idx_block)) * idx_dim * sizeof(float);
                require(total - before == heads*(per + v_per + (f.hybrid ? scales : 2*scales)) + rows,
                        "a cell did not cost exactly its own K/V/scale bytes and its indexer row");
            }
            ++capture_cases;
        }
    }
    ParkedKvLayout l; std::string error; uint64_t total = 0;
    require(parked_kv_layout(false,false,false,false,false,8,true,head_dim,heads,idx_dim,l,error),
            "layout for the capture refusals");
    require(!parked_kv_image_bytes(l, 9, total, error) && error.find("exceeds its own layout") != std::string::npos,
            "an extent past its own layout was priced anyway");
    ++refusal_cases;
    auto broken = l; broken.sizes[0] += 1;
    require(!parked_kv_image_bytes(broken, 8, total, error) && error.find("not cell-periodic") != std::string::npos,
            "a part that is not cell-periodic was priced anyway");
    ++refusal_cases;
    // Four cells of a rate that fits per part, summed over the two K/V parts, does not fit in a byte count at
    // all: the accumulation, not the multiplication, is what has to refuse.
    auto huge = l; huge.cells = 4; huge.sizes[0] = huge.sizes[1] = UINT64_MAX - 3;
    huge.sizes[2] = huge.sizes[3] = huge.sizes[4] = 0;
    require(!parked_kv_image_bytes(huge, 4, total, error) && error.find("overflow") != std::string::npos,
            "an overflowing byte count was priced anyway");
    ++refusal_cases;
    ParkedKvLayout none;
    require(parked_kv_layout(false,false,false,false,false,0,false,head_dim,heads,idx_dim,none,error) &&
            !parked_kv_image_bytes(none, 0, total, error), "an empty layout was priced");
    ++refusal_cases;
}

int main() {
    try {
        layouts();
        refusals();
        reuse();
        capture();
        std::cout << "PASS parked K/V page law: " << layout_cases << " layouts equal to the transcribed pinned "
                     "formulas and cell-periodic, " << reuse_cases << " reuse decisions where keep is whole "
                     "pages, the partial page and spare indexer row are always re-read and reused+copied is "
                     "exact, " << capture_cases << " parked images priced at their own extent, " << refusal_cases << " refusals (hybrid/ring and flag collisions, untileable "
                     "geometry, overflow, a prefix the image does not hold, a draft-layer page, an extent past its own layout); format tags "
                     "read from the pinned upstream enum; no device\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
