#pragma once
// The six generated IQ codebooks as one little-endian byte blob, inline in one header.
//
// A TPC kernel gets 16,384 bytes of static scalar data in total while the six grids alone need
// 25,600 (measured: work/validation/tpc-static-memory-blocker-v1.json), so the packed dot reads
// grid magnitudes from a read-only tensor instead of kernel statics. Byte order is one magnitude
// byte per value, least significant byte of the pinned entry first, which is exactly how the
// kernel extracted `(entry >> 8*k) & 0xff` when the tables were static; reading bytes also means
// the kernel needs no scalar 64-bit type at all, and a future codebook cannot hit that wall again.
//
// The offsets and widths are the generated contract in hpu/tpc/packed_codebook_layout.inc, which
// the kernel includes on its own. This header is inline rather than a translation unit because the
// host TPC-emulation tests each build their own executable and must be able to hand the same bytes
// to the kernel without gaining a link dependency.
#include <array>
#include <cstdint>


namespace strata::hpu {
namespace detail {
// The generated tables live here so a test can read them by name instead of including the same
// declarations twice in one translation unit.
#include "../../../tpc/packed_grid_type.inc"
#include "../../../tpc/packed_codebook_layout.inc"
#include "../../../tpc/iq2xxs_grid.inc"
#include "../../../tpc/iq3xxs_grid.inc"
#include "../../../tpc/iq2xs_grid.inc"
#include "../../../tpc/iq3s_grid.inc"
#include "../../../tpc/iq2s_grid.inc"
#include "../../../tpc/iq1s_grid.inc"

constexpr uint64_t codebook_total=STRATA_PACKED_CODEBOOK_BYTES;
struct PackedCodebookBlob { std::array<uint8_t,codebook_total> bytes{}; };

template<uint64_t at,uint64_t entries,uint64_t entry,class T>
constexpr void put_codebook(PackedCodebookBlob& out,const T* values) {
    static_assert(entry==sizeof(T),"codebook entry width disagrees with the generated layout");
    static_assert(at+entries*entry<=codebook_total,"a codebook runs past the blob");
    for(uint64_t index=0;index<entries;++index)
        for(uint64_t byte=0;byte<entry;++byte)
            out.bytes[at+index*entry+byte]=uint8_t((uint64_t(values[index])>>(8*byte))&0xffu);
}

constexpr PackedCodebookBlob build_codebook() {
    PackedCodebookBlob out{};
    put_codebook<STRATA_PACKED_CODEBOOK_IQ2XXS_AT,STRATA_PACKED_CODEBOOK_IQ2XXS_ENTRIES,STRATA_PACKED_CODEBOOK_IQ2XXS_ENTRY_BYTES>(out,packed_iq2xxs_grid);
    put_codebook<STRATA_PACKED_CODEBOOK_IQ3XXS_AT,STRATA_PACKED_CODEBOOK_IQ3XXS_ENTRIES,STRATA_PACKED_CODEBOOK_IQ3XXS_ENTRY_BYTES>(out,packed_iq3xxs_grid);
    put_codebook<STRATA_PACKED_CODEBOOK_IQ2XS_AT,STRATA_PACKED_CODEBOOK_IQ2XS_ENTRIES,STRATA_PACKED_CODEBOOK_IQ2XS_ENTRY_BYTES>(out,packed_iq2xs_grid);
    put_codebook<STRATA_PACKED_CODEBOOK_IQ3S_AT,STRATA_PACKED_CODEBOOK_IQ3S_ENTRIES,STRATA_PACKED_CODEBOOK_IQ3S_ENTRY_BYTES>(out,packed_iq3s_grid);
    put_codebook<STRATA_PACKED_CODEBOOK_IQ2S_AT,STRATA_PACKED_CODEBOOK_IQ2S_ENTRIES,STRATA_PACKED_CODEBOOK_IQ2S_ENTRY_BYTES>(out,packed_iq2s_grid);
    put_codebook<STRATA_PACKED_CODEBOOK_IQ1S_AT,STRATA_PACKED_CODEBOOK_IQ1S_ENTRIES,STRATA_PACKED_CODEBOOK_IQ1S_ENTRY_BYTES>(out,packed_iq1s_grid_gpu);
    return out;
}
inline constexpr PackedCodebookBlob packed_codebook_blob_data=build_codebook();
}

inline const uint8_t* packed_codebook_blob() { return detail::packed_codebook_blob_data.bytes.data(); }
inline uint64_t packed_codebook_bytes() { return detail::codebook_total; }
// UINT64_MAX for a type that owns no codebook.
inline uint64_t packed_codebook_offset(uint32_t type) {
    switch(type) {
        case 16:return STRATA_PACKED_CODEBOOK_IQ2XXS_AT;
        case 18:return STRATA_PACKED_CODEBOOK_IQ3XXS_AT;
        case 17:return STRATA_PACKED_CODEBOOK_IQ2XS_AT;
        case 21:return STRATA_PACKED_CODEBOOK_IQ3S_AT;
        case 22:return STRATA_PACKED_CODEBOOK_IQ2S_AT;
        case 29:return STRATA_PACKED_CODEBOOK_IQ1S_AT;
        default:return UINT64_MAX;
    }
}
// Zero for a type that owns no codebook.
inline uint64_t packed_codebook_entry_bytes(uint32_t type) {
    switch(type) {
        case 16:return STRATA_PACKED_CODEBOOK_IQ2XXS_ENTRY_BYTES;
        case 18:return STRATA_PACKED_CODEBOOK_IQ3XXS_ENTRY_BYTES;
        case 17:return STRATA_PACKED_CODEBOOK_IQ2XS_ENTRY_BYTES;
        case 21:return STRATA_PACKED_CODEBOOK_IQ3S_ENTRY_BYTES;
        case 22:return STRATA_PACKED_CODEBOOK_IQ2S_ENTRY_BYTES;
        case 29:return STRATA_PACKED_CODEBOOK_IQ1S_ENTRY_BYTES;
        default:return 0;
    }
}
inline uint64_t packed_codebook_entries(uint32_t type) {
    switch(type) {
        case 16:return STRATA_PACKED_CODEBOOK_IQ2XXS_ENTRIES;
        case 18:return STRATA_PACKED_CODEBOOK_IQ3XXS_ENTRIES;
        case 17:return STRATA_PACKED_CODEBOOK_IQ2XS_ENTRIES;
        case 21:return STRATA_PACKED_CODEBOOK_IQ3S_ENTRIES;
        case 22:return STRATA_PACKED_CODEBOOK_IQ2S_ENTRIES;
        case 29:return STRATA_PACKED_CODEBOOK_IQ1S_ENTRIES;
        default:return 0;
    }
}
}
