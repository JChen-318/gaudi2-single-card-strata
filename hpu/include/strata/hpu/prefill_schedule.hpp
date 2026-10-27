#pragma once
#include <cstdint>

namespace strata::hpu {
// Keep the same scalar suffix a reference batch size would leave under --short-read.
uint32_t canonical_prefill_count(uint32_t remaining,uint32_t requested,uint32_t short_read,uint32_t unit);
struct PromptCheckpointFlags {bool periodic=false,tail=false;};
// Reports completed chunks, never inserts an interval/tail cut. Periodic
// checkpoints may overshoot their target; next target is done+interval, just
// as the upstream Prefill::on_chunk callback. All positions are committed.
class PromptCheckpointSchedule {
public:
    PromptCheckpointSchedule(uint32_t resume,uint32_t total,uint32_t interval,bool tail,uint32_t short_read);
    PromptCheckpointFlags complete(uint32_t done,uint32_t actual_chunk);
private:
    uint32_t reached_,total_,interval_,short_read_;
    uint64_t next_;
    bool tail_,tail_seen_=false;
};
}
