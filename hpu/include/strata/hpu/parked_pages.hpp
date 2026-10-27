#pragma once
// The parked-conversation K/V page law. Upstream does not park a conversation as a rebuildable pile of
// full-precision rows: it writes the K/V in the quantized pool format the device actually uses, and when the
// prefix has not changed since the image was last restored it re-copies only the part past the last WHOLE
// PAGE and counts the rest as reused (src/core/conversation_snapshot.cpp:147-181, layout at :29-73). This
// unit is that arithmetic with no state behind it: it takes the format switches and the extents, and returns
// byte sizes, so a host test can pin it. It allocates nothing and touches no device. Layer::snapshot_reused
// calls it to place the boundaries of a park that grows the retained image (hpu/src/layer.cpp,
// hpu/src/qsa_state.cpp), and hpu/tests/parked_park.cpp runs the park itself.
//
// The page/indexer granules come from the pinned upstream shapes header (include/strata/kernels/qsa.hpp:109
// -117: page_size 4 cells = one indexer block) and the Q4_0 block size from
// include/strata/kernels/kv_q4.hpp:33, so no byte count is retyped here.
#include <cstdint>
#include <string>

namespace strata::hpu {

// The five parts, in upstream's order: K, V, K scale, V scale, pooled indexer rows.
inline constexpr unsigned parked_parts = 5;

struct ParkedKvLayout {
    int format = 0;                       // upstream's KvFormat tag, plus 16 when the K/V is rotated
    uint64_t upto = 0;                    // the extent being parked, BEFORE the page round (:153 tests this one)
    uint64_t cells = 0;                   // page-rounded cell count the buffers are sized to
    uint64_t pooled_rows = 0;             // completed indexer blocks plus the one moving spare row
    uint64_t page_size = 0, idx_block = 0, idx_dim = 0;
    uint64_t sizes[parked_parts] = {};
};

// `conversation_snapshot.cpp:29-73`. Rejected, with a reason, exactly where upstream rejects: a hybrid K8V4
// that is also a ring (`ring`, upstream's kv_mode == 2) or carries its own quantization flags, a geometry
// that cannot tile, and a byte count that would overflow.
bool parked_kv_layout(bool kv_q4, bool kv_int8, bool kv_hybrid, bool kv_rot, bool ring, uint64_t upto, bool index,
                      uint64_t head_dim, uint64_t n_head_kv, uint64_t idx_key_dim,
                      ParkedKvLayout& out, std::string& error);

// The reuse decision `conversation_kv_save` makes. `unchanged_tokens` is how many leading tokens of this
// extent are still byte-identical to what the retained image holds - upstream only trusts it for buffers it
// actually restored (include/strata/core/conversation_cache.hpp:217-231), and the draft layer always
// re-refreshes its last page (`conversation_state.cpp:291-294`), so callers pass the already-decremented
// value rather than expecting this unit to know which layer it is.
struct ParkedKvReuse {
    uint64_t keep[parked_parts] = {};     // bytes carried over from the retained image
    uint64_t copied[parked_parts] = {};   // bytes read out of the live pools
    uint64_t reused_bytes = 0;            // upstream's *reused_bytes counter, logged at generate.cpp:8371
};

// `prior_sizes` are the retained image's own part sizes; a keep larger than them means the image the caller
// believes it holds is not there, which upstream reports as a missing reusable prefix rather than copying
// less. The extents must come from the SAME layout the image was written with.
bool parked_kv_reuse(const ParkedKvLayout& layout, const uint64_t* prior_sizes, uint64_t unchanged_tokens,
                     ParkedKvReuse& out, std::string& error);

// What one parked conversation's K/V costs at its own extent: upstream's `conversation_kv_capture_bytes`
// (src/core/conversation_snapshot.cpp:124-145), which prices the segmented allocation peak of the five parts
// instead of a whole-arena limit. `layout` is the image's layout for this extent; `exact_cells` is the extent the
// port actually parks (its images are not page-rounded, the named divergence), so the K/V parts are charged at
// the law's per-cell rate over `exact_cells` while the indexer part keeps its own row count. Returns false with
// a reason where a multiplication would overflow, exactly like upstream's add()/product() pair.
bool parked_kv_image_bytes(const ParkedKvLayout& layout, uint64_t exact_cells, uint64_t& out, std::string& error);

}
