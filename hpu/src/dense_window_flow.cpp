#include "strata/hpu/dense_window_flow.hpp"
#include "strata/hpu/session_flow.hpp"
#include "strata/hpu/window_limits.hpp"
#include <algorithm>
#include <cmath>
namespace strata::hpu {
namespace {
void valid(std::span<const float> x,uint64_t count) {
    if(x.size()!=count) throw std::invalid_argument("dense window tensor shape mismatch");
    for(float v:x) if(!std::isfinite(v)) throw std::runtime_error("non-finite dense window tensor");
}
void geometry(const DenseKvState& state,const QsaConfig& c,uint32_t window) {
    c.validate();if(c.top_k<c.capacity || c.capacity!=state.capacity() || c.kv_heads!=state.heads() || c.width!=state.width() || window>c.capacity)
        throw std::invalid_argument("dense window geometry/policy mismatch");
}
}
uint32_t dense_attention_bucket(uint32_t capacity,uint32_t cells,uint32_t window) {
    if(!cells || cells>capacity || window>capacity) throw std::invalid_argument("dense attention cells/window mismatch");
    const auto count=window?std::min(window,cells):cells;
    for(auto b:{1u,8u,32u,128u,512u,2048u,8192u,32768u,capacity}) if(b>=count && b<=capacity) return b;
    throw std::logic_error("no dense attention bucket");
}
QsaWindowBatch stage_dense_window(DenseKvState& state,const QsaConfig& c,const DenseProjected& p,uint32_t rows,uint32_t window,DenseNorms norms,const std::function<bool()>& cancelled) {
    geometry(state,c,window);const uint64_t q=uint64_t(c.query_heads)*c.width,k=uint64_t(c.kv_heads)*c.width;
    if(!rows || rows>kQsaWindowRows || rows>c.capacity-state.size()) throw std::invalid_argument("dense stage row/capacity mismatch");
    valid(p.q_gate,uint64_t(rows)*q*2);valid(p.key,uint64_t(rows)*k);valid(p.value,uint64_t(rows)*k);
    auto poll=[&]{if(cancelled && cancelled()) throw Cancelled();};poll();
    const auto start=state.size();const auto point=state.checkpoint();
    try {
        QsaWindowBatch out;out.rows=rows;out.bucket=dense_attention_bucket(c.capacity,start+rows,window);
        out.query.resize(rows*q);out.gate.resize(rows*q);
        for(uint32_t row=0;row<rows;++row) {
            poll();auto query=std::span<float>(out.query).subspan(row*q,q),gate=std::span<float>(out.gate).subspan(row*q,q);
            for(uint32_t head=0;head<c.query_heads;++head) {
                const auto at=p.q_gate.begin()+row*q*2+uint64_t(head)*2*c.width;
                std::copy_n(at,c.width,query.begin()+uint64_t(head)*c.width);std::copy_n(at+c.width,c.width,gate.begin()+uint64_t(head)*c.width);
            }
            std::vector<float> key(p.key.begin()+row*k,p.key.begin()+(row+1)*k);
            rms_norm_rows(query,c.width,norms.query,c.epsilon);rms_norm_rows(key,c.width,norms.key,c.epsilon);
            const auto rope=rope_row_at(c.positions,c.rotary,c.position_base+start+row,c.rope);rotate_neox(query,c.width,rope);rotate_neox(key,c.width,rope);
            state.append(key,std::span<const float>(p.value).subspan(row*k,k));
        }
        const auto mask=checked_product(checked_product(rows,c.kv_heads),out.bucket),kv=checked_product(mask,c.width);
        out.key.resize(kv);out.value.resize(kv);out.mask.resize(mask);
        for(uint32_t row=0;row<rows;++row) {
            poll();const auto selected=state.gather_prefix(start+row+1,out.bucket,window);const auto at=uint64_t(row)*selected.key.size();
            std::copy(selected.key.begin(),selected.key.end(),out.key.begin()+at);std::copy(selected.value.begin(),selected.value.end(),out.value.begin()+at);
            for(uint32_t head=0;head<c.kv_heads;++head) std::copy(selected.mask.begin(),selected.mask.end(),out.mask.begin()+(uint64_t(row)*c.kv_heads+head)*out.bucket);
        }
        valid(out.query,rows*q);valid(out.gate,rows*q);poll();return out;
    } catch(...) {if(state.size()!=start) state.restore(point);throw;}
}
std::vector<float> execute_dense_window(DenseKvState& state,const QsaConfig& c,std::span<const float> input,uint32_t hidden,uint32_t window,DenseNorms norms,const DenseWindowOperations& op,const std::function<bool()>& cancelled) {
    geometry(state,c,window);
    if(!hidden || input.empty() || input.size()%hidden || input.size()/hidden>kResidentWindowRows || input.size()/hidden>c.capacity-state.size() || !op.project || !op.attention || !op.output)
        throw std::invalid_argument("dense execution requires1.."+std::to_string(kResidentWindowRows)+" complete rows and all stages");
    valid(input,input.size());auto poll=[&]{if(cancelled && cancelled()) throw Cancelled();};poll();
    const auto start=state.size();const auto point=state.checkpoint();std::vector<float> result(input.size());
    try {
        for(size_t row=0;row<input.size()/hidden;) {
            poll();const auto count=uint32_t(std::min<size_t>(kQsaWindowRows,input.size()/hidden-row));
            const auto projected=op.project(input.subspan(row*hidden,uint64_t(count)*hidden));poll();
            const auto batch=stage_dense_window(state,c,projected,count,window,norms,cancelled);poll();
            const auto attention=op.attention(batch);valid(attention,uint64_t(count)*c.query_heads*c.width);poll();
            const auto output=op.output(attention);valid(output,uint64_t(count)*hidden);poll();
            std::copy(output.begin(),output.end(),result.begin()+row*hidden);row+=count;
        }
        poll();return result;
    } catch(...) {if(state.size()!=start) state.restore(point);throw;}
}

QsaWindowBatch stage_arena_rows(KvArena& arena,const WindowRowPlan& plan,const QsaConfig& c,const DenseProjected& p,
                               uint32_t window,DenseNorms norms,const std::function<bool()>& cancelled) {
    if(!plan.rows() || plan.rows()>kQsaWindowRows) throw std::invalid_argument("arena window row count outside 1..kQsaWindowRows");
    if(plan.sequences()>arena.regions()) throw std::invalid_argument("arena window plan names a sequence the arena does not hold");
    const uint64_t q=uint64_t(c.query_heads)*c.width,k=uint64_t(c.kv_heads)*c.width;
    valid(p.q_gate,uint64_t(plan.rows())*q*2);valid(p.key,uint64_t(plan.rows())*k);valid(p.value,uint64_t(plan.rows())*k);
    c.validate();if(c.top_k<c.capacity) throw std::invalid_argument("arena window needs top_k >= capacity");
    auto poll=[&]{if(cancelled && cancelled()) throw Cancelled();};poll();
    // The widest sequence decides the padded bucket: a row's own cells are still its own prefix, the padding is masked.
    std::vector<uint32_t> cells_before(plan.sequences()), rows_in_plan(plan.sequences());
    uint32_t widest=0;
    // By BASE, not by the window's sequence index: a window carries only the slots that are ready, so index 1 of this
    // window is not region 1 of the arena.
    std::vector<uint32_t> region_ids(plan.sequences());
    for(uint32_t s=0;s<plan.sequences();++s) {
        region_ids[s]=arena.region_with_base(plan.kv_base[s]);
        const auto& region=arena.region(region_ids[s]);
        if(uint32_t(region.state->heads())!=c.kv_heads || uint32_t(region.state->width())!=c.width ||
           uint32_t(region.state->capacity())!=c.capacity)
            throw std::invalid_argument("arena window geometry does not match the sequence pool");
        if(region.base!=plan.kv_base[s]) throw std::invalid_argument("arena window plan KV base differs from the arena region");
        cells_before[s]=region.state->size();
        for(uint32_t row=0;row<plan.rows();++row) if(plan.row_sequence[row]==s) ++rows_in_plan[s];
        if(cells_before[s]+rows_in_plan[s]>uint32_t(region.state->capacity()))
            throw std::invalid_argument("arena window overflows a sequence pool");
        if(cells_before[s]+rows_in_plan[s]>widest) widest=cells_before[s]+rows_in_plan[s];
    }
    if(widest>c.capacity || window>c.capacity) throw std::invalid_argument("arena window bucket exceeds the capacity");
    std::vector<DenseKvState::Checkpoint> points(plan.sequences());
    for(uint32_t s=0;s<plan.sequences();++s) points[s]=arena.region(region_ids[s]).state->checkpoint();
    try {
        QsaWindowBatch out;out.rows=plan.rows();out.bucket=dense_attention_bucket(c.capacity,widest,window);
        out.query.resize(uint64_t(plan.rows())*q);out.gate.resize(uint64_t(plan.rows())*q);
        for(uint32_t row=0;row<plan.rows();++row) {
            poll();
            const auto s=plan.row_sequence[row];
            auto query=std::span<float>(out.query).subspan(uint64_t(row)*q,q),gate=std::span<float>(out.gate).subspan(uint64_t(row)*q,q);
            for(uint32_t head=0;head<c.query_heads;++head) {
                const auto at=p.q_gate.begin()+uint64_t(row)*q*2+uint64_t(head)*2*c.width;
                std::copy_n(at,c.width,query.begin()+uint64_t(head)*c.width);
                std::copy_n(at+c.width,c.width,gate.begin()+uint64_t(head)*c.width);
            }
            std::vector<float> key(p.key.begin()+uint64_t(row)*k,p.key.begin()+uint64_t(row+1)*k);
            rms_norm_rows(query,c.width,norms.query,c.epsilon);rms_norm_rows(key,c.width,norms.key,c.epsilon);
            // The plan's absolute position, not this call's row index: a sequence may resume past its first cell.
            const auto rope=rope_row_at(c.positions,c.rotary,c.position_base+plan.row_position[row],c.rope);
            rotate_neox(query,c.width,rope);rotate_neox(key,c.width,rope);
            arena.append_at(uint64_t(plan.kv_base[s])+plan.row_position[row],key,std::span<const float>(p.value).subspan(uint64_t(row)*k,k));
        }
        const auto mask=checked_product(checked_product(plan.rows(),c.kv_heads),out.bucket),kv=checked_product(mask,c.width);
        out.key.resize(kv);out.value.resize(kv);out.mask.resize(mask);
        for(uint32_t row=0;row<plan.rows();++row) {
            poll();
            const auto s=plan.row_sequence[row];
            const auto cells=plan.row_position[row]+1;
            // Through the REGION the row's base names, not the window's sequence index (the append and validation loops
            // already resolve it this way).
            const auto selected=arena.gather_span(region_ids[s],cells,out.bucket,window);
            const auto at=uint64_t(row)*selected.key.size();
            std::copy(selected.key.begin(),selected.key.end(),out.key.begin()+at);
            std::copy(selected.value.begin(),selected.value.end(),out.value.begin()+at);
            for(uint32_t head=0;head<c.kv_heads;++head)
                std::copy(selected.mask.begin(),selected.mask.end(),out.mask.begin()+(uint64_t(row)*c.kv_heads+head)*out.bucket);
        }
        valid(out.query,uint64_t(plan.rows())*q);valid(out.gate,uint64_t(plan.rows())*q);poll();
        return out;
    } catch(...) {
        for(uint32_t s=0;s<plan.sequences();++s)
            if(arena.region(region_ids[s]).state->size()!=cells_before[s]) arena.region(region_ids[s]).state->restore(points[s]);
        throw;
    }
}
}
