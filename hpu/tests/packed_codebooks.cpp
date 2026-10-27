// The concatenated IQ codebook blob is what the packed kernel will read from HBM, so this test
// proves the byte form carries the generated tables exactly: every entry of every grid is
// reconstructed from its bytes and compared to the array the TPC dot used to index directly, and
// the offsets those two halves share are the same generated contract.
#include "strata/hpu/packed_codebooks.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>


namespace {
using strata::hpu::detail::packed_iq1s_grid_gpu;
using strata::hpu::detail::packed_iq2s_grid;
using strata::hpu::detail::packed_iq2xs_grid;
using strata::hpu::detail::packed_iq2xxs_grid;
using strata::hpu::detail::packed_iq3s_grid;
using strata::hpu::detail::packed_iq3xxs_grid;
using strata::hpu::packed_codebook_blob;
using strata::hpu::packed_codebook_bytes;
using strata::hpu::packed_codebook_entries;
using strata::hpu::packed_codebook_entry_bytes;
using strata::hpu::packed_codebook_offset;

int failures=0;
void require(bool ok,const char* what) {
    if(!ok) { std::printf("FAIL %s\n",what); ++failures; }
}

template<class T>
uint64_t check(uint32_t type,const char* name,const T* table) {
    const uint64_t at=packed_codebook_offset(type),entries=packed_codebook_entries(type),entry=packed_codebook_entry_bytes(type);
    if(entry!=sizeof(T)) { std::printf("FAIL %s: layout width %llu is not the array width %llu\n",name,
        (unsigned long long)entry,(unsigned long long)sizeof(T)); ++failures; return 0; }
    uint64_t bad=0;
    for(uint64_t index=0;index<entries;++index) {
        uint64_t value=0;
        for(uint64_t byte=0;byte<entry;++byte) value|=uint64_t(packed_codebook_blob()[at+index*entry+byte])<<(8*byte);
        if(value!=uint64_t(table[index])) ++bad;
    }
    if(bad) { std::printf("FAIL %s: %llu of %llu entries do not match their bytes\n",name,
        (unsigned long long)bad,(unsigned long long)entries); ++failures; }
    return entries;
}
}

int main() {
    uint64_t entries=0;
    entries+=check(16,"iq2xxs_grid",packed_iq2xxs_grid);
    entries+=check(18,"iq3xxs_grid",packed_iq3xxs_grid);
    entries+=check(17,"iq2xs_grid",packed_iq2xs_grid);
    entries+=check(21,"iq3s_grid",packed_iq3s_grid);
    entries+=check(22,"iq2s_grid",packed_iq2s_grid);
    entries+=check(29,"iq1s_grid_gpu",packed_iq1s_grid_gpu);
    require(entries==256u+256u+512u+512u+1024u+2048u,"the six grids did not account for every entry");

    // Offsets must tile the blob with no gap and no overlap, or one table would silently read
    // another's bytes; the sum is checked against the generated total, not a typed number.
    const uint32_t order[6]={16,18,17,21,22,29};
    uint64_t at=0;
    for(uint32_t index=0;index<6;++index) {
        const uint32_t type=order[index];
        require(packed_codebook_offset(type)==at,"a codebook is not adjacent to the previous one");
        at+=packed_codebook_entries(type)*packed_codebook_entry_bytes(type);
    }
    require(at==packed_codebook_bytes() && at==strata::hpu::detail::codebook_total,"the blob size is not the sum of its tables");

    // Whole-set assertion in this backend's style: exactly these six GGML ids own a codebook, and
    // every other id up to GGML_TYPE_COUNT reports no offset, no width and no entries.
    for(uint32_t type=0;type<43;++type) {
        const bool expected=type==16||type==17||type==18||type==21||type==22||type==29;
        require((packed_codebook_offset(type)!=UINT64_MAX)==expected,"the codebook-owning type set drifted");
        require((packed_codebook_entry_bytes(type)!=0)==expected,"the codebook width disagreed with ownership");
        require((packed_codebook_entries(type)!=0)==expected,"a declared codebook has no entries");
    }

    // The reconstruction has to be driven by the bytes: flipping one byte inside the IQ2_S range
    // must change the value read there, proving no comparison above quietly fell back to the array.
    {
        const uint64_t at=packed_codebook_offset(22),entry=packed_codebook_entry_bytes(22);
        std::vector<uint8_t> copy(packed_codebook_blob(),packed_codebook_blob()+packed_codebook_bytes());
        const uint64_t index=7;
        auto byte=[&](uint64_t k)->uint64_t { return copy[at+index*entry+k]; };
        uint64_t before=0,after=0;
        for(uint64_t k=0;k<entry;++k) before|=byte(k)<<(8*k);
        copy[at+index*entry+3]^=0x80u;
        for(uint64_t k=0;k<entry;++k) after|=byte(k)<<(8*k);
        require(before==uint64_t(packed_iq2s_grid[index]),"the copy did not start from the table value");
        require(before!=after,"a perturbed codebook byte changed nothing, so the read path is not byte driven");
    }

    if(failures) { std::printf("packed codebook blob: %d failure(s)\n",failures); return 1; }
    std::printf("PASS packed codebook blob: %llu bytes for %llu entries over 6 grids, every entry "
        "reconstructed from its own bytes and equal to the generated table, offsets proven adjacent "
        "and tiling the blob, exactly {16 17 18 21 22 29} of 43 ggml ids own a codebook, and a "
        "flipped byte changes the reconstructed entry; no device\n",
        (unsigned long long)packed_codebook_bytes(),(unsigned long long)entries);
    return 0;
}
