#include "strata/hpu/weight_transfer.hpp"
#include "strata/hpu/packed_quant.hpp"
#include <algorithm>
#include <array>
#include <cstring>

namespace strata::hpu {
DeviceBuffer upload_packed_weights(const Device& device,const WeightSlice& source,const Stream& copy,uint64_t staging,std::span<const int32_t> ids,bool q4) {
    const auto row_bytes=packed_row_bytes(q4?2:source.ggml_type(),source.columns());
    const uint64_t rows=ids.empty()?source.rows():ids.size(),bytes=checked_product(row_bytes,rows);
    if(copy.state()!=device.state() || !rows || (!q4 && source.row_bytes()!=row_bytes) || staging/2<row_bytes) throw std::invalid_argument("packed weight shape/staging/device mismatch");
    for(auto id:ids) if(id<0 || uint64_t(id)>=source.rows()) throw std::invalid_argument("packed weight subset ID out of range");
    if(!q4) {
        if(ids.empty()) validate_packed_rows(source.bytes(),source.ggml_type(),source.columns(),source.rows());
        else for(auto id:ids) validate_packed_rows(source.bytes().subspan(uint64_t(id)*row_bytes,row_bytes),source.ggml_type(),source.columns(),1);
    }
    const uint64_t slot_rows=std::min(rows,staging/2/row_bytes),slot_bytes=slot_rows*row_bytes;
    std::vector<float> q4_scratch(q4?source.columns():0);
    DeviceBuffer output(device,bytes);std::array<std::unique_ptr<HostBuffer>,2> slots;std::array<std::unique_ptr<Event>,2> done;
    for(size_t i=0;i<2;++i) {slots[i]=std::make_unique<HostBuffer>(device,slot_bytes);done[i]=std::make_unique<Event>(device);}
    try {
        uint64_t copies=0;
        for(uint64_t row=0;row<rows;) {
            const auto index=copies++%2,count=std::min(slot_rows,rows-row);if(done[index]->recorded()) done[index]->synchronize();
            auto* target=static_cast<uint8_t*>(slots[index]->data());
            for(uint64_t j=0;j<count;++j) {
                const uint64_t from=ids.empty()?row+j:uint32_t(ids[row+j]);
                if(q4) source.convert_rows_q4(from,{target+j*row_bytes,size_t(row_bytes)},q4_scratch);
                else std::memcpy(target+j*row_bytes,source.bytes().data()+from*row_bytes,row_bytes);
            }
            upload(copy,output,*slots[index],count*row_bytes,row*row_bytes);done[index]->record(copy);row+=count;
        }
        for(auto& event:done) if(event->recorded()) event->synchronize();
    } catch(...) {device.synchronize();throw;}
    return output;
}
struct Bf16WeightUpload::Impl {
    struct Slot {
        HostBuffer data;
        Event copied;
        Slot(const Device& device, uint64_t bytes) : data(device, bytes), copied(device) {}
    };
    DeviceBuffer output;
    std::array<std::unique_ptr<Slot>, 2> slots;
    Event ready;
    uint64_t pinned = 0, copies = 0;
    Impl(const Device& device, const WeightSlice& source,std::span<const int32_t> ids,const Stream& stream, uint64_t budget)
        : output(device, allocation_size(source,ids,budget)), ready(device) {
        if (stream.state() != device.state()) throw std::invalid_argument("weight upload stream device mismatch");
        const auto row_bytes = checked_product(source.columns(), 2);
        const auto selected=ids.empty()?source.rows():ids.size();
        const auto rows_per_slot = std::min(uint64_t(selected), budget / 2 / row_bytes);
        const auto slot_bytes = checked_product(rows_per_slot, row_bytes);
        for (auto& slot : slots) slot = std::make_unique<Slot>(device, slot_bytes);
        pinned = checked_product(slot_bytes, slots.size());
        for (uint64_t row = 0; row < selected;) {
            auto& slot = *slots[copies % slots.size()];
            // The CPU must not overwrite a pinned source still used by DMA.
            if (slot.copied.recorded()) slot.copied.synchronize();
            const auto rows = std::min(rows_per_slot, selected - row);
            auto* destination=static_cast<uint16_t*>(slot.data.data());
            if(ids.empty()) source.convert_rows(row,{destination,size_t(checked_product(rows,source.columns()))});
            else for(uint64_t r=0;r<rows;++r) source.convert_rows(uint32_t(ids[row+r]),{destination+r*source.columns(),size_t(source.columns())});
            upload(stream, output, slot.data, checked_product(rows, row_bytes), checked_product(row, row_bytes));
            slot.copied.record(stream);
            ++copies;
            row += rows;
        }
        ready.record(stream);
    }
    static uint64_t allocation_size(const WeightSlice& source,std::span<const int32_t> ids,uint64_t budget) {
        const auto row_bytes = checked_product(source.columns(), 2);
        if (!row_bytes || !source.rows() || budget / 2 < row_bytes)
            throw std::invalid_argument("pinned budget must hold two complete BF16 rows");
        for(auto id:ids) if(id<0 || uint64_t(id)>=source.rows()) throw std::out_of_range("selected weight row out of range");
        return checked_product(row_bytes,ids.empty()?source.rows():ids.size());
    }
};
Bf16WeightUpload::Bf16WeightUpload(const Device& device, const WeightSlice& source,
                                 const Stream& stream, uint64_t budget)
    : impl_(std::make_unique<Impl>(device, source,std::span<const int32_t>{},stream,budget)) {}
Bf16WeightUpload::Bf16WeightUpload(const Device& device,const WeightSlice& source,std::span<const int32_t> rows,const Stream& stream,uint64_t budget)
    :impl_(std::make_unique<Impl>(device,source,rows,stream,budget)) {}
Bf16WeightUpload::~Bf16WeightUpload() = default;
const DeviceBuffer& Bf16WeightUpload::buffer() const { if(!impl_) throw std::logic_error("weight upload already consumed");return impl_->output; }
const Event& Bf16WeightUpload::completion() const { if(!impl_) throw std::logic_error("weight upload already consumed");return impl_->ready; }
uint64_t Bf16WeightUpload::pinned_bytes() const { return impl_?impl_->pinned:0; }
uint64_t Bf16WeightUpload::transfer_count() const { if(!impl_) throw std::logic_error("weight upload already consumed");return impl_->copies; }
DeviceBuffer Bf16WeightUpload::finish() {
    if(!impl_) throw std::logic_error("weight upload already consumed");
    impl_->ready.synchronize();DeviceBuffer result(std::move(impl_->output));impl_.reset();return result;
}
} // namespace strata::hpu
