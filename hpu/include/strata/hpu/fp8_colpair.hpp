#pragma once
#include "strata/hpu/packed_quant.hpp"
#include <cstring>
#include <span>
#include <vector>
namespace strata::hpu {
// Codes only; per-row F32 scales keep their original order and offset. Each 128-row block is
// [columns/2][128][2], with rows 0,64,1,65,... so the native BF16 MAC halves write contiguous rows.
inline void pack_fp8_colpair(std::span<uint8_t> codes,uint64_t rows,uint64_t columns) {
    if(!rows || rows%128 || !columns || columns%128 || codes.size()!=checked_product(rows,columns))
        throw std::invalid_argument("FP8 column-pair layout geometry mismatch");
    std::vector<uint8_t> packed(codes.size());
    for(uint64_t block=0;block<rows/128;++block) for(uint64_t c=0;c<columns;c+=2) for(uint64_t r=0;r<128;++r) {
        const auto dest=block*columns*128+(c/2)*256+((r%64)*2+r/64)*2;
        packed[dest]=codes[(block*128+r)*columns+c];packed[dest+1]=codes[(block*128+r)*columns+c+1];
    }
    std::memcpy(codes.data(),packed.data(),packed.size());
}
}
