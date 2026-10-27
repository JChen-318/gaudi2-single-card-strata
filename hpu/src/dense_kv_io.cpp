#include "strata/hpu/dense_kv.hpp"
#include "strata/hpu/kv_cell.hpp"
#include "strata/hpu/state_io.hpp"
#include "strata/kernels/f16_bits.hpp"
#include <cmath>
#include <functional>

namespace strata::hpu {
void DenseKvState::write_snapshot(StateWriter& writer,const Checkpoint& p) const {
    if(p.owner!=owner_ || !p.persistent) throw std::invalid_argument("dense KV serialization needs an owned full snapshot");
    writer.integer(p.cells);
    if(int8()) {
        writer.vector<int8_t>(p.key_codes);writer.vector<int8_t>(p.value_codes);
        writer.vector<uint16_t>(p.key_scales);writer.vector<uint16_t>(p.value_scales);
    } else if(q4()) {
        writer.vector<uint8_t>(p.q4_key);writer.vector<uint8_t>(p.q4_value);
    } else {
        writer.vector<uint16_t>(p.key);writer.vector<uint16_t>(p.value);
    }
}

DenseKvState::Checkpoint DenseKvState::read_snapshot(StateReader& reader,uint32_t cells) const {
    if(cells>capacity_ || reader.integer()!=cells) throw std::invalid_argument("dense KV serialized position mismatch");
    Checkpoint p;p.owner=owner_;p.cells=cells;p.persistent=true;
    const auto count=checked_product(cells,checked_product(heads_,width_));
    if(int8()) {
        const auto scales=checked_product(cells,checked_product(heads_,kv_int8_groups(width_)));
        p.key_codes=reader.vector<int8_t>(count);p.value_codes=reader.vector<int8_t>(count);
        p.key_scales=reader.vector<uint16_t>(scales);p.value_scales=reader.vector<uint16_t>(scales);
        for(const auto* row:{&p.key_scales,&p.value_scales}) for(const auto scale:*row)
            if(!std::isfinite(kernels::f32_from_f16(scale))) throw std::invalid_argument("non-finite serialized dense INT8 KV scale");
    } else if(q4()) {
        const auto bytes=checked_product(cells,checked_product(heads_,q4_bytes()));
        p.q4_key=reader.vector<uint8_t>(bytes);p.q4_value=reader.vector<uint8_t>(bytes);
        for(const auto* row:{&p.q4_key,&p.q4_value})
            for(size_t at=0;at+sizeof(strata::kernels::block_q4_0)<=row->size();at+=sizeof(strata::kernels::block_q4_0)) {
                const uint16_t scale=uint16_t((*row)[at]|(uint16_t((*row)[at+1])<<8));
                if(!std::isfinite(kernels::f32_from_f16(scale))) throw std::invalid_argument("non-finite serialized dense Q4_0 KV scale");
            }
    } else {
        p.key=reader.vector<uint16_t>(count);p.value=reader.vector<uint16_t>(count);
        for(const auto* row:{&p.key,&p.value}) for(auto bits:*row)
            if(!std::isfinite(kernels::f32_from_f16(bits))) throw std::invalid_argument("non-finite serialized dense KV");
    }
    if(generation_==UINT64_MAX) throw std::overflow_error("draft KV snapshot lineage exhausted");
    p.prefix=++generation_;
    return p;
}
}
