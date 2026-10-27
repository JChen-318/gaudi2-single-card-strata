#include "strata/hpu/prefill_schedule.hpp"
#include <algorithm>
#include <climits>
#include <stdexcept>

namespace strata::hpu {
uint32_t canonical_prefill_count(uint32_t remaining,uint32_t requested,uint32_t short_read,uint32_t unit) {
    if(!unit) return std::min(remaining,requested);
    if(remaining<=short_read) return 0;
    const uint64_t windows=(uint64_t(remaining)-short_read+unit-1)/unit;
    return uint32_t(std::min<uint64_t>(std::min(remaining,requested),windows*unit));
}
PromptCheckpointSchedule::PromptCheckpointSchedule(uint32_t resume,uint32_t total,uint32_t interval,bool tail,uint32_t short_read)
    :reached_(resume),total_(total),interval_(interval),short_read_(short_read),next_(uint64_t(resume)+interval),tail_(tail) {
    if(resume>total || total>INT32_MAX || interval>INT32_MAX || short_read>INT32_MAX) throw std::invalid_argument("invalid prompt checkpoint schedule");
}
PromptCheckpointFlags PromptCheckpointSchedule::complete(uint32_t done,uint32_t actual) {
    if(done<=reached_ || done>total_ || actual!=done-reached_) throw std::invalid_argument("noncontiguous committed prompt chunk");
    reached_=done;PromptCheckpointFlags result;
    result.periodic=interval_ && done>=next_;
    const auto remaining=total_-done;
    result.tail=tail_ && interval_ && !tail_seen_ && remaining>std::max(1u,short_read_) && remaining<=actual;
    if(result.periodic) next_=uint64_t(done)+interval_;
    if(result.tail) tail_seen_=true; // a periodic overlap consumes the one tail too
    return result;
}
}
