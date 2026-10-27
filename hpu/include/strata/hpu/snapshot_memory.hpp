#pragma once
#include "strata/hpu/allocation_budget.hpp"
#include "strata/hpu/tensor.hpp"
#include <algorithm>
#include <memory>
#include <span>
#include <type_traits>

namespace strata::hpu {
inline uint64_t snapshot_add_bytes(uint64_t left,uint64_t right) {
    if(right>UINT64_MAX-left) throw std::overflow_error("snapshot capacity sum overflow");
    return left+right;
}
template<class T,class Allocator>
uint64_t snapshot_vector_bytes(const std::vector<T,Allocator>& values) {return checked_product(values.capacity(),sizeof(T));}
// Declare BEFORE the snapshot's buffers so the charge dies after them, also
// on exceptions. All shared aliases keep this one payload reservation alive.
class SnapshotReservation {
public:
    SnapshotReservation(std::shared_ptr<AllocationBudget> budget,uint64_t resident):budget_(std::move(budget)) {
        if(!budget_ || !resident) throw std::invalid_argument("snapshot requires a budget and resident envelope");
        budget_->charge(resident);bytes_=resident;resident_=resident;
    }
    ~SnapshotReservation(){budget_->release(bytes_);}
    SnapshotReservation(const SnapshotReservation&)=delete;
    SnapshotReservation& operator=(const SnapshotReservation&)=delete;
    uint64_t bytes() const {return bytes_;}
    // Reserve final residency plus the largest replaced component BEFORE
    // resizing. A failed caller drops the exclusive image, retaining this
    // charge until even its partially grown buffers have been destroyed.
    void prepare(uint64_t resident,uint64_t overlap) {
        if(!resident || overlap>UINT64_MAX-resident) throw std::overflow_error("snapshot growth envelope overflow");
        if(resident<resident_) throw std::invalid_argument("snapshot reuse cannot shrink its resident envelope");
        const auto peak=std::max(bytes_,resident+overlap);
        budget_->charge(peak-bytes_);bytes_=peak;
    }
    void commit(uint64_t resident) {
        if(resident<resident_ || resident>bytes_) throw std::logic_error("snapshot resident envelope was not reserved");
        const auto freed=bytes_-resident;bytes_=resident;resident_=resident;budget_->release(freed);
    }
private:
    std::shared_ptr<AllocationBudget> budget_;
    uint64_t bytes_=0,resident_=0;
};
// No implicit geometric capacity: cache admission prices an extent. A growth
// replaces one component at a time under the owner's pre-reserved overlap.
template<class T,class Allocator>
void snapshot_resize(std::vector<T,Allocator>& values,size_t size) {
    static_assert(std::is_trivially_copyable_v<T>);
    if(size<=values.capacity()) {values.resize(size);return;}
    std::vector<T,Allocator> next(size,T{},values.get_allocator());
    // The pinned libstdc++ constructs exactly size capacity. Refuse a future
    // STL whose capacity would escape that bound rather than hiding the excess.
    if(next.capacity()!=size) throw std::length_error("snapshot STL capacity exceeds requested extent");
    std::copy_n(values.begin(),std::min(values.size(),size),next.begin());
    values.swap(next);
}
template<class T>
void snapshot_assign(std::vector<T>& values,std::span<const T> input) {
    snapshot_resize(values,input.size());std::copy(input.begin(),input.end(),values.begin());
}
// An aliased cache entry already prices this payload; only an exclusive
// retained slot adds another contribution after take() drops the entry.
inline uint64_t exclusive_snapshot_bytes(const std::shared_ptr<void>& image,uint64_t resident) {
    return image && image.use_count()==1?resident:0;
}
}
