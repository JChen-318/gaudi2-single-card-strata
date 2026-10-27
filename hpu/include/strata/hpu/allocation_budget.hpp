#pragma once
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <exception>
#include <string>

namespace strata::hpu {
struct AllocationStats {uint64_t limit=0,used=0,peak=0;};
class AllocationBudgetExceeded:public std::runtime_error {
public:
    AllocationBudgetExceeded():std::runtime_error("native allocation budget exceeded") {}
    AllocationBudgetExceeded(uint64_t requested,uint64_t used,uint64_t limit):std::runtime_error(
        "native allocation budget exceeded: requested="+std::to_string(requested)+" used="+std::to_string(used)+" limit="+std::to_string(limit)) {}
};
// Tracks requested bytes, including temporary overlaps and recipe workspaces.
// Driver overhead is outside this ledger and needs a separate reserved margin.
class AllocationBudget {
public:
    void configure(uint64_t limit) {
        std::lock_guard lock(mutex_);
        if(stats_.used || !limit) throw std::logic_error("allocation limits require an empty pool and nonzero cap");
        stats_={limit,0,0};
    }
    void charge(uint64_t bytes) {
        std::lock_guard lock(mutex_);
        if(bytes>stats_.limit-stats_.used) throw AllocationBudgetExceeded(bytes,stats_.used,stats_.limit);
        stats_.used+=bytes;if(stats_.used>stats_.peak) stats_.peak=stats_.used;
    }
    void release(uint64_t bytes) noexcept {
        std::lock_guard lock(mutex_);if(bytes>stats_.used) std::terminate();stats_.used-=bytes;
    }
    AllocationStats stats() const {std::lock_guard lock(mutex_);return stats_;}
private:
    mutable std::mutex mutex_;
    AllocationStats stats_{UINT64_MAX,0,0};
};
}
