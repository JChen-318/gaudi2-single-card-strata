#pragma once
#include "strata/hpu/weights.hpp"
#include <array>
#include <functional>

namespace strata::hpu {
struct NativePackRole {
    uint32_t type=0;uint64_t columns=0,rows=0,offset=0,bytes=0;
    std::string path;
};
struct NativePackLayer {
    std::array<NativePackRole,3> roles;
    std::vector<std::array<float,3>> scales;
};
// Catalog bound to the main GGUF by the production constructor. Explicit
// catalogs allow host embedding/tests without creating a device or CPU pool.
struct NativePackCatalog {
    uint32_t experts=0;
    std::vector<NativePackLayer> layers;
    std::vector<std::pair<std::string,uint64_t>> shards;
    std::function<std::array<WeightSlice,3>(uint32_t,uint32_t)> gguf_expert;
};
class NativeExpertPack {
public:
    NativeExpertPack(const std::string& directory,const WeightTable&);
    NativeExpertPack(const std::string& directory,NativePackCatalog);
    std::array<WeightSlice,3> slices(uint32_t layer,uint32_t expert) const;
    std::array<float,3> scales(uint32_t layer,uint32_t expert) const;
    std::vector<uint8_t> native_blob(uint32_t layer,uint32_t expert) const;
    uint64_t blob_bytes(uint32_t layer) const;
    bool has_binary() const;
    std::string file_identity() const;
    // Includes bounded catalog, manifest and all per-expert scale metadata.
    // Mapped payload pages are OS-backed, never copied as an entire model.
    static constexpr uint64_t host_reserve=4ull<<20;
private:
    struct Impl;std::shared_ptr<const Impl> impl_;
};
}
