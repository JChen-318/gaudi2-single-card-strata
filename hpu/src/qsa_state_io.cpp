#include "strata/hpu/qsa_state.hpp"
#include "strata/hpu/kv_cell.hpp"
#include "strata/hpu/parked_pages.hpp"
#include "strata/hpu/state_io.hpp"
#include "strata/kernels/f16_bits.hpp"
#include <cmath>
#include <functional>

namespace strata::hpu {
namespace {
void finite(std::span<const float> values) {for(float value:values) if(!std::isfinite(value)) throw std::invalid_argument("non-finite serialized QSA value");}
void finite_q4_scales(std::span<const uint8_t> bytes) {
    for(size_t at=0;at+sizeof(strata::kernels::block_q4_0)<=bytes.size();at+=sizeof(strata::kernels::block_q4_0)) {
        const uint16_t scale=uint16_t(bytes[at]|(uint16_t(bytes[at+1])<<8));
        if(!std::isfinite(strata::kernels::f32_from_f16(scale))) throw std::invalid_argument("non-finite serialized Q4_0 KV scale");
    }
}
// The header the parked image is described by upstream: format, cells, heads, head_dim, page_size, pooled rows,
// index dim (src/core/conversation_file.cpp:676-678 put_kv_header), and then the five parts in that order -
// K, V, K scale, V scale, pooled (conversation_snapshot.cpp:128-129, :166-167). The port parks a host pool whose
// keys are already rotated: format kKvF16 + 16 for --kv fp16, kKvInt8 + 16 with its two scale parts for
// --kv int8, kKvQ4 for --kv q4_0 and kKvHybrid for --kv k8v4 (those two tags already mean rotated, :57). The
// extents come from the parked-page law rather than being retyped here; the port's cells field is the parked
// extent itself (its images are not page-rounded), which is the divergence named in hpu/PROGRESS.md.
ParkedKvLayout snapshot_layout(const QsaConfig& config,uint64_t cells) {
    ParkedKvLayout layout;std::string error;
    if(!parked_kv_layout(config.kv==KvMode::Q4,config.kv==KvMode::Int8,config.kv==KvMode::K8V4,true,false,cells,true,config.width,config.kv_heads,config.index_width,layout,error))
        throw std::invalid_argument("QSA snapshot header: "+error);
    return layout;
}
} // namespace
void QsaState::write_snapshot(StateWriter& writer,const Checkpoint& point) const {
    if(point.owner!=owner_ || !point.persistent) throw std::invalid_argument("QSA serialization requires an owned full snapshot");
    const auto& c=config_;const auto layout=snapshot_layout(c,point.cells);
    if(layout.pooled_rows!=uint64_t(point.pooled.size()/c.index_width))
        throw std::invalid_argument("QSA snapshot header: pooled rows disagree with the image");
    // Named one per upstream field, in upstream order, so the header this writer emits is comparable to
    // put_kv_header's list without reading past it - work/check_snapshot_field_order.py does exactly that.
    const uint64_t format_field=static_cast<uint64_t>(layout.format),cells_field=point.cells,
        heads_field=c.kv_heads,head_dim_field=c.width,page_size_field=layout.page_size,
        pooled_rows_field=layout.pooled_rows,idx_dim_field=c.index_width;
    for(const uint64_t value:{format_field,cells_field,heads_field,head_dim_field,page_size_field,pooled_rows_field,
                              idx_dim_field}) writer.integer(value);
    if(int8()) {
        writer.vector<int8_t>(point.key_codes);writer.vector<int8_t>(point.value_codes);
        writer.vector<uint16_t>(point.key_scales);writer.vector<uint16_t>(point.value_scales);
    } else if(hybrid()) {
        // K stays the unrotated INT8 group pair; V is the rotated Q4_0 half, whose blocks carry their own FP16
        // scale, so the V scale part is empty.
        writer.vector<int8_t>(point.key_codes);writer.vector<uint8_t>(point.q4_value);
        writer.vector<uint16_t>(point.key_scales);writer.vector<float>({});
    } else if(q4()) {
        // The q4_0 blocks carry their own FP16 scale, so the two scale parts are empty like the F16 image's.
        writer.vector<uint8_t>(point.q4_key);writer.vector<uint8_t>(point.q4_value);
        writer.vector<float>({});writer.vector<float>({});
    } else {
        writer.vector<uint16_t>(point.keys);writer.vector<uint16_t>(point.values);
        writer.vector<float>({});writer.vector<float>({});              // the F16 image has no scale parts
    }
    writer.vector<float>(point.pooled);
    // Port-side fields with no upstream counterpart in the K/V blob: the raw rows of the indexer block that is
    // not yet complete and its rotating spare row, which upstream keeps inside the live pool instead.
    writer.vector<float>(point.tail);writer.vector<float>(point.spare);
}
QsaState::Checkpoint QsaState::read_snapshot(StateReader& reader,uint32_t cells) const {
    const auto& c=config_;const auto layout=snapshot_layout(c,cells);
    const auto expected_pooled=uint64_t(cells?cells/c.block+1:0);
    const auto expected_elements=checked_product(uint64_t(cells),checked_product(c.kv_heads,c.width));
    // conversation_kv_validate (:183-201) refuses an image whose own geometry is not the live one; the same rule
    // here, so a file written by a different arithmetic or geometry cannot be half-restored.
    auto same=[&](uint64_t value,const char* what) {
        if(reader.integer()!=value) throw std::invalid_argument(std::string("QSA serialized K/V ")+what+" mismatch");
    };
    same(uint64_t(layout.format),"format");
    same(uint64_t(cells),"cells");
    same(uint64_t(c.kv_heads),"head count");
    same(uint64_t(c.width),"head dim");
    same(uint64_t(layout.page_size),"page size");
    same(expected_pooled,"pooled rows");
    same(uint64_t(c.index_width),"index dim");
    Checkpoint point;point.owner=owner_;point.cells=cells;point.persistent=true;
    if(c.kv==KvMode::Int8) {
        point.key_codes=reader.vector<int8_t>(expected_elements);point.value_codes=reader.vector<int8_t>(expected_elements);
        const auto expected_scales=checked_product(uint64_t(cells),checked_product(c.kv_heads,kv_int8_groups(c.width)));
        point.key_scales=reader.vector<uint16_t>(expected_scales);point.value_scales=reader.vector<uint16_t>(expected_scales);
        for(const auto& scales:{std::cref(point.key_scales),std::cref(point.value_scales)})
            for(const auto scale:scales.get())
                if(!std::isfinite(strata::kernels::f32_from_f16(scale))) throw std::invalid_argument("non-finite serialized INT8 KV scale");
    } else if(c.kv==KvMode::K8V4) {
        const auto expected_scales=checked_product(uint64_t(cells),checked_product(c.kv_heads,kv_int8_groups(c.width)));
        const auto expected_bytes=checked_product(uint64_t(cells),checked_product(c.kv_heads,kv_q4_bytes_per_head(c.width)));
        point.key_codes=reader.vector<int8_t>(expected_elements);
        point.q4_value=reader.vector<uint8_t>(expected_bytes);
        point.key_scales=reader.vector<uint16_t>(expected_scales);
        reader.vector<float>(0);                                         // the empty V scale part
        for(const auto scale:point.key_scales)
            if(!std::isfinite(strata::kernels::f32_from_f16(scale))) throw std::invalid_argument("non-finite serialized INT8 KV scale");
        finite_q4_scales(point.q4_value);
    } else if(c.kv==KvMode::Q4) {
        const auto expected_bytes=checked_product(uint64_t(cells),checked_product(c.kv_heads,kv_q4_bytes_per_head(c.width)));
        point.q4_key=reader.vector<uint8_t>(expected_bytes);point.q4_value=reader.vector<uint8_t>(expected_bytes);
        reader.vector<float>(0);reader.vector<float>(0);                 // the empty scale parts
        finite_q4_scales(point.q4_key);finite_q4_scales(point.q4_value);
    } else {
        point.keys=reader.vector<uint16_t>(expected_elements);point.values=reader.vector<uint16_t>(expected_elements);
        reader.vector<float>(0);reader.vector<float>(0);                 // the empty scale parts
        for(const auto& values:{std::cref(point.keys),std::cref(point.values)}) for(auto bits:values.get())
            if(!std::isfinite(strata::kernels::f32_from_f16(bits))) throw std::invalid_argument("non-finite serialized FP16 KV");
    }
    point.pooled=reader.vector<float>(expected_pooled*c.index_width);
    point.tail=reader.vector<float>(tail_.size());point.spare=reader.vector<float>(spare_.size());
    finite(point.tail);finite(point.spare);finite(point.pooled);
    // The disk header carries tensor geometry, not this process's lineage IDs.
    // Mint an identity so two imported images at the same length cannot alias.
    if(generation_==UINT64_MAX) throw std::overflow_error("QSA snapshot lineage exhausted");
    point.prefix=++generation_;
    return point;
}
}
