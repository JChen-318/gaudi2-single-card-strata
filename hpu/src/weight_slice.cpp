#include "strata/hpu/weights.hpp"
#include "strata/hpu/q4_requantize.hpp"
#include "weight_format.hpp"
#include <array>
#include <cstring>
namespace strata::hpu {
using detail::format;using detail::convertible;
WeightSlice WeightSlice::raw_rows(std::shared_ptr<const void> owner,std::span<const uint8_t> bytes,uint32_t type,uint64_t columns,uint64_t rows) {
    const auto f=format(type);
    if(!owner || !columns || !rows || !convertible(type) || columns%f.elements) throw std::invalid_argument("invalid raw weight rows");
    const auto row_bytes=checked_product(columns/f.elements,f.bytes);
    if(bytes.size()!=checked_product(row_bytes,rows)) throw std::invalid_argument("raw weight payload size mismatch");
    WeightSlice out;out.owner_=std::move(owner);out.source_=bytes;out.columns_=columns;out.rows_=rows;out.type_=type;out.row_bytes_=row_bytes;return out;
}
void WeightSlice::convert_rows(uint64_t first_row, std::span<uint16_t> output) const {
    convert_impl(first_row,output.data(),output.size(),true);
}
void WeightSlice::convert_rows_f32(uint64_t first_row, std::span<float> output) const {
    convert_impl(first_row,output.data(),output.size(),false);
}
void WeightSlice::convert_impl(uint64_t first_row, void* output, uint64_t elements, bool bf16) const {
    if (!owner_ || !columns_) throw std::logic_error("empty weight slice");
    if (elements % columns_) throw std::invalid_argument("weight destination must contain whole rows");
    const auto count = elements / columns_;
    check_range(rows_, first_row, count);
    const auto f = format(type_);
    auto* out16 = static_cast<uint16_t*>(output);
    auto* out32 = static_cast<float*>(output);
    std::array<float, 256> decoded{};
    for (uint64_t row = 0; row < count; ++row) {
        const auto* input = source_.data() + (first_row + row) * row_bytes_;
        for (uint64_t col = 0; col < columns_; col += f.elements) {
            if (type_ == 30 && bf16) { // Preserve every BF16 bit, including signed zero and NaN payload.
                std::memcpy(out16 + row * columns_ + col, input, 2);
            } else {
                if (type_ == 0) std::memcpy(decoded.data(), input, 4);
                else if (type_ == 30) decoded[0] = from_bf16(strata::read_u16(input));
                else if (type_ == 42) strata::dequantize_q2_0(input, decoded.data());
                else f.decode(input, decoded.data(), static_cast<int64_t>(f.elements));
                if (bf16) for (uint64_t i = 0; i < f.elements; ++i) out16[row * columns_ + col + i] = to_bf16(decoded[i]);
                else std::memcpy(out32 + row * columns_ + col,decoded.data(),f.elements*sizeof(float));
            }
            input += f.bytes;
        }
    }
}
void WeightSlice::convert_rows_q4(uint64_t first,std::span<uint8_t> output,std::span<float> scratch) const {
    if(!columns_ || columns_>65536 || columns_%32 || output.empty()) throw std::invalid_argument("invalid Q4 weight conversion geometry");
    const auto row=columns_/32*18;
    if(output.size()%row) throw std::invalid_argument("partial Q4 weight output row");
    const auto count=output.size()/row;check_range(rows_,first,count);
    std::vector<float> owned;
    if(scratch.empty()) {owned.resize(columns_);scratch=owned;}
    if(scratch.size()!=columns_) throw std::invalid_argument("Q4 weight scratch must hold exactly one float row");
    for(uint64_t r=0;r<count;++r) {
        convert_rows_f32(first+r,scratch);quantize_q4_rows(scratch,uint32_t(columns_),output.subspan(r*row,row));
    }
}
}
