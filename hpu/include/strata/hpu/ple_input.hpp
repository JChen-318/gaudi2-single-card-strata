#pragma once
#include "strata/hpu/weights.hpp"
#include "strata/hpu/ple_ram_budget.hpp"
#include "strata/hpu/ple_reader_budget.hpp"
#include "strata/kernels/ngram.hpp"
#include <array>

namespace strata::hpu {
struct PleHashConfig {
    strata::kernels::PleConsts constants{};
    uint32_t vocabulary=248320;
    uint64_t table_rows=0;
    void validate() const;
};
// External table files need no model/split-family metadata. Hash constants and
// vocabulary always come from the main model; only storage comes from here.
struct PleSourceInfo {std::string path,identity;uint64_t rows=0,payload_offset=0,payload_bytes=0;};
PleSourceInfo inspect_ple_source(const std::string&);
std::string ple_source_identity(const std::string&);
PleRamMemory ple_ram_memory(const PleSourceInfo&,const strata::kernels::PleIoOptions&);
PleReaderMemory ple_reader_memory(const PleSourceInfo&,const strata::kernels::PleIoOptions&);
// Shared by the Model's real input and host-only source tests. A RAM request
// must carry the preflight reservation before PleTable may fault/lock pages.
void open_ple_source(strata::kernels::PleTable&,const PleSourceInfo&,
                     const strata::kernels::PleIoOptions&,uint64_t ram_reservation,
                     uint64_t reader_reservation=kPleReaderBaselineHostBytes,double delay_us=0);
PleHashConfig ple_hash_config(const WeightTable&,uint64_t table_rows=0);
// previous is chronological; only its last two IDs are read.
std::array<uint32_t,16> ple_rows(int32_t token,std::span<const int32_t> previous,const PleHashConfig&);
// The rows of a whole batch, token-major: token r's predecessors are the committed prefix's last
// two and then the batch's own earlier tokens, so every token matches ple_rows(token,prefix).
std::vector<uint32_t> ple_batch_rows(std::span<const int32_t> tokens,std::span<const int32_t> previous,const PleHashConfig&);
// Bounded original Strata SSD reader and table decoders. Hash parameters come
// from metadata; geometry/EOS are checked against the fixed upstream target.
class PleInput {
public:
    explicit PleInput(const WeightTable&,strata::kernels::PleIoOptions={},const std::string& external_table={},uint64_t ram_reservation=0,
                      uint64_t reader_reservation=kPleReaderBaselineHostBytes,double delay_us=0);
    std::vector<float> gather(int32_t token,std::span<const int32_t> previous);
    // One reader request for the whole batch (PleTable::gather_batch): a per-row gather pays one
    // SSD request per token, which a prefill window otherwise serializes in front of layer 1.
    // previous is chronological, only its last two IDs seed the batch's first token; every later
    // token takes its predecessors from the batch itself. out holds tokens*2560 floats.
    void gather_rows(std::span<const int32_t> tokens,std::span<const int32_t> previous,float* out,uint64_t width);
    std::string io_report() const;
    bool ram_locked() const; // true only when the OS lock result is trustworthy
private:
    PleHashConfig config_;
    strata::kernels::PleTable table_;
    bool ram_requested_=false;
};
}
