#include "strata/hpu/qsa_window_flow.hpp"
#include "strata/hpu/qsa_phase_stats.hpp"
#include "strata/hpu/session_flow.hpp"
#include <algorithm>
#include <cmath>
namespace strata::hpu {
namespace {
void valid(std::span<const float> values,uint64_t count) {
    if(values.size()!=count) throw std::invalid_argument("QSA window tensor shape mismatch");
    for(float value:values) if(!std::isfinite(value)) throw std::runtime_error("non-finite QSA window tensor");
}
}
uint32_t qsa_attention_bucket(const QsaConfig& config,uint32_t selected) {
    if(!selected) throw std::invalid_argument("QSA attention requires a selected cell");
    const auto maximum=config.selection_width(config.capacity);
    for(auto b:{1u,8u,32u,128u,512u,2048u,maximum}) if(b>=selected && b<=maximum) return b;
    throw std::invalid_argument("QSA selected count exceeds recipe capacity");
}
uint32_t qsa_device_bucket(const QsaConfig& config,uint32_t selected,uint32_t extra) {
    if(!extra) throw std::invalid_argument("QSA device window requires a current row");
    const auto width=config.selection_width(config.capacity);
    if(extra>=width) throw std::invalid_argument("QSA device window exceeds the selection width");
    const auto bucket=std::min(qsa_attention_bucket(config,std::max(selected,1u)),width-extra);
    if(bucket<selected) throw std::invalid_argument("QSA device bucket does not cover the cache");
    return bucket;
}
bool qsa_window_geometry_equal(const QsaConfig& a,const QsaConfig& b) {
    return a.query_heads==b.query_heads && a.kv_heads==b.kv_heads && a.width==b.width && a.index_heads==b.index_heads &&
        a.index_width==b.index_width && a.capacity==b.capacity && a.top_k==b.top_k && a.block==b.block;
}
uint64_t qsa_window_attention_pinned_bytes(const QsaConfig& c,uint32_t rows,uint32_t bucket) {
    c.validate();if(!rows || rows>kResidentWindowRows || !bucket || bucket>c.selection_width(c.capacity)) throw std::invalid_argument("QSA attention pinned geometry mismatch");
    const auto q=checked_product(checked_product(rows,c.query_heads),c.width),mask=checked_product(checked_product(rows,c.kv_heads),bucket);
    const auto kv=checked_product(mask,c.width);
    auto sum=[](uint64_t a,uint64_t b) {if(b>UINT64_MAX-a) throw std::overflow_error("QSA attention pinned byte overflow");return a+b;};
    return checked_product(sum(sum(checked_product(q,3),checked_product(kv,2)),sum(mask,1)),4);
}
uint64_t qsa_window_gather_pinned_bytes(const QsaConfig& c,uint32_t rows,uint32_t bucket) {
    // The old complete payload also validates geometry and checked arithmetic;
    // this subset cannot overflow if that larger payload fits.
    (void)qsa_window_attention_pinned_bytes(c,rows,bucket);
    const uint64_t q=checked_product(checked_product(rows,c.query_heads),c.width);
    const uint64_t ids=checked_product(rows,bucket);
    return checked_product(checked_product(q,2)+ids+1,4);
}
QsaWindowBatch stage_qsa_window(QsaState& state,const QsaProjected& p,uint32_t rows,QsaNorms norms,const std::function<bool()>& cancelled,
                                QsaWindowBatch* reuse,bool selected_ids_only) {
    const auto& c=state.config();const uint64_t q=uint64_t(c.query_heads)*c.width,k=uint64_t(c.kv_heads)*c.width,iq=uint64_t(c.index_heads)*c.index_width;
    if(!rows || rows>kQsaWindowRows || rows>c.capacity-state.size()) throw std::invalid_argument("QSA stage window row/capacity mismatch");
    valid(p.q_gate,uint64_t(rows)*q*2);valid(p.key,uint64_t(rows)*k);valid(p.value,uint64_t(rows)*k);
    valid(p.index_query,uint64_t(rows)*iq);valid(p.index_key,uint64_t(rows)*c.index_width);
    auto poll=[&] {if(cancelled && cancelled()) throw Cancelled();};poll();
    const auto point=state.checkpoint();const auto start=state.size();
    try {
        QsaWindowBatch batch=reuse?std::move(*reuse):QsaWindowBatch{};batch.rows=rows;
        batch.query.resize(rows*q);batch.gate.resize(rows*q);
        std::vector<std::vector<int32_t>> ids;ids.reserve(rows);uint32_t maximum=0;
        std::vector<float> key(k),index(iq);
        QsaPhaseTimer timer;
        for(uint32_t row=0;row<rows;++row) {
            poll();auto query=std::span<float>(batch.query).subspan(row*q,q),gate=std::span<float>(batch.gate).subspan(row*q,q);
            for(uint32_t head=0;head<c.query_heads;++head) {
                const auto begin=p.q_gate.begin()+row*q*2+uint64_t(head)*2*c.width;
                std::copy_n(begin,c.width,query.begin()+uint64_t(head)*c.width);
                std::copy_n(begin+c.width,c.width,gate.begin()+uint64_t(head)*c.width);
            }
            std::copy_n(p.key.begin()+row*k,k,key.begin());
            std::copy_n(p.index_query.begin()+row*iq,iq,index.begin());
            rms_norm_rows(query,c.width,norms.query,c.epsilon);rms_norm_rows(key,c.width,norms.key,c.epsilon);
            rms_norm_rows(index,c.index_width,norms.index_query,c.epsilon);
            const auto rotation=rope_row_at(c.positions,c.rotary,c.position_base+state.size(),c.rope);
            rotate_neox(query,c.width,rotation);rotate_neox(key,c.width,rotation);rotate_neox(index,c.index_width,rotation);
            timer.mark("window.norm-rope");
            state.append(key,std::span<const float>(p.value).subspan(row*k,k),std::span<const float>(p.index_key).subspan(uint64_t(row)*c.index_width,c.index_width),norms.index_key);
            timer.mark("window.append");
            ids.push_back(state.select(index));maximum=std::max(maximum,uint32_t(ids.back().size()));
            timer.mark("window.select");
            for(auto id:ids.back()) if(id<0 || uint32_t(id)>=start+row+1) throw std::logic_error("QSA selection escaped causal prefix");
        }
        batch.bucket=qsa_attention_bucket(c,maximum);
        if(selected_ids_only) {
            batch.key.clear();batch.value.clear();batch.mask.clear();
            batch.ids.assign(uint64_t(rows)*batch.bucket,-1.0f);
            for(uint32_t row=0;row<rows;++row)
                std::copy(ids[row].begin(),ids[row].end(),batch.ids.begin()+uint64_t(row)*batch.bucket);
            valid(batch.query,uint64_t(rows)*q);valid(batch.gate,uint64_t(rows)*q);poll();return batch;
        }
        batch.ids.clear();
        const auto cells=checked_product(checked_product(rows,c.kv_heads),batch.bucket);
        batch.key.resize(checked_product(cells,c.width));batch.value.resize(batch.key.size());batch.mask.resize(cells);
        const auto kv_count=uint64_t(c.kv_heads)*batch.bucket*c.width;
        for(uint32_t row=0;row<rows;++row) {
            poll();
            auto mask=std::span<float>(batch.mask).subspan(uint64_t(row)*c.kv_heads*batch.bucket,uint64_t(c.kv_heads)*batch.bucket);
            // The gathered rows land in the batch buffers directly: no per-row temporaries and no second copy.
            state.gather_into(ids[row],batch.bucket,std::span<float>(batch.key).subspan(uint64_t(row)*kv_count,kv_count),
                              std::span<float>(batch.value).subspan(uint64_t(row)*kv_count,kv_count),mask.first(batch.bucket));
            for(uint32_t head=1;head<c.kv_heads;++head) std::copy_n(mask.begin(),batch.bucket,mask.begin()+uint64_t(head)*batch.bucket);
        }
        timer.mark("window.gather");
        valid(batch.query,uint64_t(rows)*q);valid(batch.gate,uint64_t(rows)*q);poll();return batch;
    } catch(...) {if(state.size()!=start) state.restore(point);throw;}
}
std::vector<float> execute_qsa_window(QsaState& state,std::span<const float> input,uint32_t hidden,QsaNorms norms,
                                     const QsaWindowOperations& op,const std::function<bool()>& cancelled) {
    const auto& c=state.config();
    if(!hidden || input.empty() || input.size()%hidden || input.size()/hidden>kResidentWindowRows || input.size()/hidden>c.capacity-state.size() || !op.project || !op.attention || !op.output)
        throw std::invalid_argument("QSA window requires1.."+std::to_string(kResidentWindowRows)+" complete rows and all stages");
    valid(input,input.size());auto poll=[&] {if(cancelled && cancelled()) throw Cancelled();};poll();
    const auto point=state.checkpoint();const auto start=state.size();
    try {
        std::vector<float> result(input.size());const auto rows=input.size()/hidden;
        QsaPhaseTimer timer;QsaWindowBatch reuse;
        for(size_t row=0;row<rows;) {
            poll();const auto count=uint32_t(std::min<size_t>(kQsaWindowRows,rows-row));
            auto projected=op.project(input.subspan(row*hidden,uint64_t(count)*hidden));poll();
            timer.mark("window.project");
            auto batch=stage_qsa_window(state,projected,count,norms,cancelled,&reuse,op.selected_ids_only);poll();
            timer.mark("window.stage");
            const auto attention=op.attention(batch);valid(attention,uint64_t(count)*c.query_heads*c.width);poll();
            // The next window overwrites every element, so hand the buffers straight back.
            reuse=std::move(batch);
            timer.mark("window.attention");
            const auto output=op.output(attention);valid(output,uint64_t(count)*hidden);poll();
            timer.mark("window.output");
            std::copy(output.begin(),output.end(),result.begin()+row*hidden);row+=count;
        }
        poll();return result;
    } catch(...) {if(state.size()!=start) state.restore(point);throw;}
}
}
