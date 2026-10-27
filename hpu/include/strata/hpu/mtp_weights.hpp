#pragma once
#include "strata/hpu/weights.hpp"
#include <array>

namespace strata::hpu {
struct MtpTensor {
    std::string name,kind;
    uint64_t rows=0,columns=0,offset=0,bytes=0;
};
// Original draft_vocab.bin: a nonempty LE int32 token-id list, no duplicates.
// Mapping/read size is bounded by the target vocabulary, preserving file order.
class DraftVocabulary {
public:
    DraftVocabulary(const std::string& path,uint32_t vocabulary);
    const std::vector<int32_t>& ids() const;
    std::string file_identity() const;
private:
    struct Impl;std::shared_ptr<Impl> impl_;
};
// Original tools/mtp_rt.py format, without full-file materialization, prefault
// or SDK calls. Dense Q8_0/BF16/F32 rows reuse the bounded WeightSlice decoder.
// Routed Q2_0 experts require one <=1,382,400-byte lossless relayout per call;
// consumers must budget how many owning slices they keep concurrently.
// Runtime norms already contain Gemma's 1+w; no extra fold is performed here.
// Source files must remain immutable while any mapping/slice is alive.
class MtpWeights {
public:
    explicit MtpWeights(const std::string& runtime_directory,uint64_t expert_scratch_budget=2ull<<20);
    const std::vector<MtpTensor>& inventory() const;
    const MtpTensor& info(const std::string&) const;
    WeightSlice matrix(const std::string&) const;
    WeightSlice vector(const std::string&) const; // [1,N] or [N,1] -> one row
    std::vector<float> constant(const std::string&,uint64_t elements) const;
    std::array<WeightSlice,3> expert(uint32_t id) const; // gate, up, down
    uint64_t dense_payload_bytes() const;
    // Same conservative total, but six FC/QKV/O projections use raw Q8_0 or
    // Q4_0 copies under --mtp-q4. Shared experts remain conservatively BF16
    // budgeted here even when their actual separate cache payload is smaller.
    uint64_t packed_projection_payload_bytes(bool q4=false) const;
    std::string file_identity() const; // local path/stat identity, not SHA256
private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};
}
