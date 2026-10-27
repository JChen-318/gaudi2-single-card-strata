#include "strata/hpu/parked_pages.hpp"

#include "strata/kernels/kv_q4.hpp"       // block_q4_0 and kv_q4_bytes_per_head, the pinned upstream ones
#include "strata/kernels/kv_stream.hpp"   // the KvFormat enum the snapshot's format tag is built from
#include "strata/kernels/qsa.hpp"         // qsa_real_shapes(): page_size and idx_block, not retyped here

#include <limits>

namespace strata::hpu {
namespace {

// conversation_detail::product() upstream: multiply only if no step overflows.
bool product(uint64_t& out, uint64_t a, uint64_t b, uint64_t c) {
    const uint64_t limit = std::numeric_limits<uint64_t>::max();
    if (a && b > limit / a) return false;
    const uint64_t ab = a * b;
    if (ab && c > limit / ab) return false;
    out = ab * c;
    return true;
}

// layer.hpp:322 `qsa_kv_format`, whose QsaState argument this unit does not have.
int base_format(bool kv_q4, bool kv_int8, bool kv_hybrid) {
    if (kv_hybrid) return strata::kernels::kKvHybrid;
    return kv_q4 ? strata::kernels::kKvQ4 : kv_int8 ? strata::kernels::kKvInt8 : strata::kernels::kKvF16;
}

}

bool parked_kv_layout(bool kv_q4, bool kv_int8, bool kv_hybrid, bool kv_rot, bool ring, uint64_t upto, bool index,
                      uint64_t head_dim, uint64_t n_head_kv, uint64_t idx_key_dim,
                      ParkedKvLayout& out, std::string& error) {
    const auto shapes = strata::kernels::qsa_real_shapes();
    out = {};
    // A streamed ring reads its own host copy, so K8V4 cannot also be a ring or reuse another quantization
    // flag: upstream refuses the combination rather than inventing a fourth tag (:31-34).
    if (kv_hybrid && (ring || kv_q4 || kv_int8)) {
        error = "parked K/V: hybrid K8V4 cannot be a ring and needs distinct format flags";
        return false;
    }
    const bool int8_keys = kv_int8 || kv_hybrid;
    if (!n_head_kv || !head_dim || head_dim > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
        !idx_key_dim || (kv_q4 && head_dim % strata::kernels::QK4_0) ||
        (int8_keys && !kv_q4 && head_dim % 64)) {
        error = "parked K/V: invalid geometry for the requested format";
        return false;
    }
    const uint64_t page = shapes.page_size;
    out.upto = upto;
    out.cells = ((upto + page - 1) / page) * page;
    out.page_size = page;
    out.idx_block = shapes.idx_block;
    out.idx_dim = idx_key_dim;
    // The rotating spare indexer row is carried even when it is not complete, because a rewind to an earlier
    // prefix rebuilds it (:45-47).
    out.pooled_rows = index && upto ? upto / shapes.idx_block + 1 : 0;
    out.format = base_format(kv_q4, kv_int8, kv_hybrid) + (kv_rot && !kv_q4 && !kv_hybrid ? 16 : 0);
    const uint64_t per = kv_q4 ? strata::kernels::kv_q4_bytes_per_head(static_cast<int>(head_dim))
                               : head_dim * (int8_keys ? 1 : 2);
    uint64_t scales_per_head = 0;
    if (int8_keys && !kv_q4) {
        if (head_dim % 64) { error = "parked K/V: invalid geometry for the requested format"; return false; }
        scales_per_head = (head_dim / 64) * 2;
    }
    if (!product(out.sizes[0], out.cells, n_head_kv, per) ||
        !product(out.sizes[2], out.cells, n_head_kv, scales_per_head) ||
        !product(out.sizes[4], out.pooled_rows, idx_key_dim, sizeof(float))) {
        error = "parked K/V: byte count overflow";
        return false;
    }
    out.sizes[1] = out.sizes[0];
    out.sizes[3] = out.sizes[2];
    if (kv_hybrid) {   // V is Q4_0 while K stays INT8 with its scales (:64-71)
        const uint64_t v_per = strata::kernels::kv_q4_bytes_per_head(static_cast<int>(head_dim));
        if (!product(out.sizes[1], out.cells, n_head_kv, v_per)) {
            error = "parked K/V: hybrid V byte count overflow";
            return false;
        }
        out.sizes[3] = 0;
    }
    return true;
}

bool parked_kv_reuse(const ParkedKvLayout& layout, const uint64_t* prior_sizes, uint64_t unchanged_tokens,
                     ParkedKvReuse& out, std::string& error) {
    out = {};
    if (prior_sizes == nullptr) { error = "parked K/V: no retained image to reuse from"; return false; }
    if (!layout.cells) { error = "parked K/V: layout has no cells"; return false; }
    // :153's guard, expressed without signed arithmetic. Upstream tests the extent as it was asked for, not the
    // page-rounded buffer length, so a count that lands in the padding of the last page is refused rather than
    // silently clamped. The second half of upstream's guard - the prefix cannot exceed the image believed to
    // hold it - is the keep/prior comparison below.
    if (unchanged_tokens > layout.upto) {
        error = "parked K/V: invalid unchanged prefix";
        return false;
    }
    const uint64_t whole_cells = (unchanged_tokens / layout.page_size) * layout.page_size;
    for (unsigned part = 0; part < parked_parts; ++part) {
        uint64_t keep = 0;
        if (part == 4) {
            // The indexer keeps whole completed blocks; the moving spare row is always re-copied (:171). A pool
            // with NO indexer part (the draft pool: layout.pooled_rows == 0) keeps nothing there - upstream's
            // keep for part 4 is gated on the same `index` flag that zeroes pooled_rows (:171, :45-47).
            keep = (layout.idx_block && layout.pooled_rows)
                       ? (unchanged_tokens / layout.idx_block) * layout.idx_dim * sizeof(float) : 0;
            if (keep > layout.sizes[4]) { error = "parked K/V: pooled reuse exceeds this extent"; return false; }
        } else {
            if (layout.sizes[part] % layout.cells) { error = "parked K/V: part is not cell-periodic"; return false; }
            keep = (layout.sizes[part] / layout.cells) * whole_cells;
        }
        if (keep > prior_sizes[part]) { error = "parked K/V: missing reusable prefix"; return false; }
        out.keep[part] = keep;
        out.copied[part] = layout.sizes[part] - keep;
        out.reused_bytes += keep;
    }
    return true;
}

bool parked_kv_image_bytes(const ParkedKvLayout& layout, uint64_t exact_cells, uint64_t& out, std::string& error) {
    out = 0;
    if (!layout.cells) { error = "parked K/V: layout has no cells"; return false; }
    if (exact_cells > layout.cells) { error = "parked K/V: extent exceeds its own layout"; return false; }
    for (unsigned part = 0; part < parked_parts; ++part) {
        uint64_t bytes = 0;
        if (part == 4) {
            // The indexer rows are already counted from the unrounded extent: completed blocks plus the one
            // moving spare row, which is why this part is charged exactly as the law reports it.
            bytes = layout.sizes[4];
        } else {
            if (layout.sizes[part] % layout.cells) { error = "parked K/V: part is not cell-periodic"; return false; }
            const uint64_t per_cell = layout.sizes[part] / layout.cells;
            if (per_cell && exact_cells > std::numeric_limits<uint64_t>::max() / per_cell) {
                error = "parked K/V: byte count overflow"; return false;
            }
            bytes = per_cell * exact_cells;
        }
        if (bytes > std::numeric_limits<uint64_t>::max() - out) { error = "parked K/V: byte count overflow"; return false; }
        out += bytes;
    }
    return true;
}

}
