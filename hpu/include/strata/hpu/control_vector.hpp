#pragma once
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace strata::hpu {
enum class ControlVectorMode:uint32_t {Project=0,Add=1};
struct ControlVectorOptions {
    std::vector<std::pair<std::string,float>> files;
    int first=-1,last=-1,single=-1;
    ControlVectorMode mode=ControlVectorMode::Add;
};
// Bounded host constants from original controlvector GGUFs. Each input is
// mmap-read once, not prefaulted or copied wholesale. Layer 0 is never steered.
class ControlVectors {
public:
    ControlVectors(ControlVectorOptions,uint32_t layers,uint32_t hidden);
    bool covers(uint32_t layer) const;
    std::span<const float> direction(uint32_t layer) const;
    float scale(uint32_t layer) const;
    ControlVectorMode mode() const;
    uint64_t signature() const; // non-cryptographic identity of exact resolved tables/options
    std::string file_identity() const; // canonical path/stat identity, not a content hash
    std::string summary() const;
    uint64_t host_bytes() const;
    // Host arithmetic oracle for tests. Production Model uses the HPU graph.
    void apply_reference(uint32_t layer,std::span<float> residual,bool enabled=true) const;
private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};
}
