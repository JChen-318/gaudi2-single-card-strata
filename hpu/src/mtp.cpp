#include "strata/hpu/mtp.hpp"
#include "strata/hpu/session_flow.hpp"
#include "strata/hpu/parked_pages.hpp"
#include "strata/hpu/dense_weights.hpp"
#include "strata/hpu/host_recipe.hpp"
#include "strata/hpu/layer_flow.hpp"
#include "strata/hpu/model_head.hpp"
#include "strata/hpu/moe.hpp"
#include "strata/hpu/qsa.hpp"
#include "strata/hpu/packed_quant.hpp"
#include "strata/hpu/hc_window.hpp"
#include "strata/hpu/qsa_window.hpp"
#include "strata/hpu/dense_window_flow.hpp"
#include "strata/hpu/session_flow.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <mutex>

namespace strata::hpu {
struct MtpBlock::Impl {
    Device device;MtpWeights runtime;MtpOptions options;core::ModelGeometry geometry=core::kCompiledGeometry;QsaConfig config;
    DenseKvState kv;
    std::vector<float> embedding_gamma,hidden_gamma,query_gamma,key_gamma;
    std::unique_ptr<DenseWeights> weights;
    std::unique_ptr<ExpertCache> cache;
    std::unique_ptr<HostRecipe> read_attention,read_ffn,write,attention,final_mix;
    std::unique_ptr<ModelHead> input_embedding,input_hidden,query_gate,key,value,output,head;
    std::unique_ptr<MoeBlock> moe;
    std::unique_ptr<HcWindow> hc_window;
    std::unique_ptr<QsaWindow> dense_window;
    uint32_t bucket=0;bool poisoned=false;mutable std::mutex mutex;
    Impl(const Device& d,MtpWeights w,MtpOptions o):device(d),runtime(std::move(w)),options(std::move(o)),
        config(qsa_config(geometry,options.context)),kv(options.context,uint32_t(geometry.n_head_kv),uint32_t(geometry.head_dim),kv_drafter_mode(options.kv)) {
        config.top_k=options.context;config.position_base=options.position_base;config.rope=options.rope;config.positions=options.positions;config.validate();
    }
    std::vector<Binding> select(std::initializer_list<const char*> names,const std::string& prefix={}) const {
        std::vector<Binding> out;for(const auto* name:names) {auto b=weights->binding(prefix+name);b.name=name;out.push_back(std::move(b));}return out;
    }
    uint32_t attention_bucket(uint32_t cells) const {
        return dense_attention_bucket(options.context,cells,options.window);
    }
    std::vector<float> mixer(std::span<const float> x) {
        const auto qg=query_gate->run(x);
        const auto& c=config;std::vector<float> query(uint64_t(c.query_heads)*c.width),gate(query.size());
        for(uint32_t h=0;h<c.query_heads;++h) {
            std::copy_n(qg.begin()+uint64_t(h)*2*c.width,c.width,query.begin()+uint64_t(h)*c.width);
            std::copy_n(qg.begin()+(uint64_t(h)*2+1)*c.width,c.width,gate.begin()+uint64_t(h)*c.width);
        }
        auto k=key->run(x);const auto v=value->run(x);
        rms_norm_rows(query,c.width,query_gamma,c.epsilon);rms_norm_rows(k,c.width,key_gamma,c.epsilon);
        const auto rope=rope_row_at(c.positions,c.rotary,c.position_base+kv.size(),c.rope);rotate_neox(query,c.width,rope);rotate_neox(k,c.width,rope);
        kv.append(k,v);const auto b=attention_bucket(kv.size());const auto selected=kv.gather(b,options.window);
        if(bucket!=b) {
            attention.reset();bucket=0;
            GraphBuilder graph(device);define_qsa_attention(graph,c,b);
            const uint64_t pinned=checked_product(uint64_t(c.kv_heads)*c.width*b,8)+uint64_t(c.query_heads)*c.width*16+uint64_t(b)*4+(1ull<<20);
            auto next=std::make_unique<HostRecipe>(device,graph.compile("strata_mtp_dense_"+std::to_string(b)),std::span<const Binding>{},std::vector<std::string>{"out"},pinned);
            const std::array<float,1> scale{1/std::sqrt(float(c.width))};next->set("scale",scale);attention=std::move(next);bucket=b;
        }
        attention->set("query",query);attention->set("key",selected.key);attention->set("value",selected.value);
        attention->set("mask",selected.mask);attention->set("q_gate",gate);attention->run();
        return output->run(attention->output("out"));
    }
    std::vector<float> mixer_window(std::span<const float> x,const std::function<bool()>& cancelled) {
        DenseWindowOperations op;
        op.project=[&](auto input) {return DenseProjected{query_gate->run_rows(input),key->run_rows(input),value->run_rows(input)};};
        op.attention=[&](const auto& batch){return dense_window->attention(batch);};
        op.output=[&](auto input){return output->run_rows(input);};
        return execute_dense_window(kv,config,x,uint32_t(geometry.n_embd),options.window,{query_gamma,key_gamma},op,cancelled);
    }
    std::vector<float> finish_head(std::span<const float> residual) {
        final_mix->set("residual",residual);final_mix->run();auto logits=head->run(final_mix->output("mixed"));
        for(float x:logits) if(!std::isfinite(x)) throw NonFiniteOutput("non-finite MTP head logits");
        if(!options.draft_vocab.empty()) {
            std::vector<float> full(size_t(geometry.n_vocab),-INFINITY);
            for(size_t i=0;i<logits.size();++i) full[uint32_t(options.draft_vocab[i])]=logits[i];
            logits=std::move(full);
        }
        return logits;
    }
};
MtpBlock::MtpBlock(const Device& device,MtpWeights runtime,Binding head_weight,MtpOptions options,WeightSlice head_source) {
    const auto& g=core::kCompiledGeometry;const uint64_t h=g.n_embd,r=g.hc,n=h*r,lr=g.hc_lr;
    if(!options.context || options.window>options.context || options.staging/4<h || options.expert_cache<checked_product(h,uint64_t(g.n_ff)*6) || head_weight.name!="weight" || !head_weight.buffer ||
       head_weight.buffer->state()!=device.state() || (head_weight.buffer->address()+head_weight.offset)%128)
        throw std::invalid_argument("invalid MTP options/shared head binding");
    const uint32_t source_type=head_source.rows()?head_source.ggml_type():30,head_type=options.q4_head?2:source_type;
    const uint64_t shared_row=packed_supported(source_type)?packed_row_bytes(source_type,h):h*2;
    const uint64_t head_row=options.q4_head?packed_row_bytes(2,h):shared_row;
    check_range(head_weight.buffer->size(),head_weight.offset,checked_product(shared_row,g.n_vocab));
    if(options.q4_head && (head_source.rows()!=uint64_t(g.n_vocab) || head_source.columns()!=h)) throw std::invalid_argument("MTP Q4 head requires the complete mapped source");
    if(!options.draft_vocab.empty()) {
        if(options.draft_vocab.size()>uint64_t(g.n_vocab) || head_source.rows()!=uint64_t(g.n_vocab) || head_source.columns()!=h) throw std::invalid_argument("invalid MTP subset head source");
        std::vector<bool> seen(size_t(g.n_vocab));
        for(auto id:options.draft_vocab) {if(id<0 || id>=g.n_vocab || seen[uint32_t(id)]) throw std::invalid_argument("invalid MTP draft subset token");seen[uint32_t(id)]=true;}
    }
    auto config=qsa_config(g,options.context);config.top_k=options.context;config.position_base=options.position_base;config.rope=options.rope;config.positions=options.positions;config.validate();
    auto p=std::make_unique<Impl>(device,std::move(runtime),options);
    p->embedding_gamma=p->runtime.constant("pre_fc_norm_embedding.weight",h);p->hidden_gamma=p->runtime.constant("pre_fc_norm_hidden.weight",n);
    p->query_gamma=p->runtime.constant("self_attn.q_norm.weight",g.head_dim);p->key_gamma=p->runtime.constant("self_attn.k_norm.weight",g.head_dim);
    const auto private_head_rows=options.q4_head?(options.draft_vocab.empty()?uint64_t(g.n_vocab):options.draft_vocab.size()):options.draft_vocab.size();
    const auto subset_bytes=checked_product(private_head_rows,head_row);
    p->weights=std::make_unique<DenseWeights>(device,p->runtime.packed_projection_payload_bytes(options.q4_projections)+subset_bytes,options.staging);
    auto matrix=[&](const std::string& binding,const std::string& name,uint64_t columns,uint64_t rows) {p->weights->bf16(binding,p->runtime.matrix(name),columns,rows);};
    auto packed_projection=[&](const std::string& name) {
        const auto source=p->runtime.matrix(name);
        if(options.q4_projections) p->weights->requantized_q4(name,source);else p->weights->packed(name,source);
        auto binding=p->weights->binding(name);binding.name="weight";
        return std::make_unique<ModelHead>(device,binding,uint32_t(source.columns()),uint32_t(source.rows()),options.q4_projections?2:source.ggml_type());
    };
    p->input_embedding=packed_projection("fc_embedding.weight");p->input_hidden=packed_projection("fc_hidden.weight");
    p->query_gate=packed_projection("self_attn.q_proj.weight");p->key=packed_projection("self_attn.k_proj.weight");
    p->value=packed_projection("self_attn.v_proj.weight");p->output=packed_projection("self_attn.o_proj.weight");
    matrix("w_router","mlp.gate.weight",h,g.n_expert);p->weights->bf16("w_shared_gate",p->runtime.vector("mlp.shared_expert_gate.weight"),h,1);
    const HcConfig hc{uint32_t(h),uint32_t(r),uint32_t(lr),float(g.rms_eps)};
    for(const auto& names:std::array<std::pair<std::string,std::string>,3>{{{"attn_","attn_hyper_connection."},{"ffn_","mlp_hyper_connection."},{"final_","hyper_connection_mixer."}}}) {
        const auto& [binding,source]=names;matrix(binding+"w_down",source+"input_mix_weight_down.weight",n,lr);matrix(binding+"w_up",source+"input_mix_weight_up.weight",lr,n);
        p->weights->f32(binding+"norm",p->runtime.constant(source+"hc_norm.weight",n));
        if(binding!="final_") matrix(binding+"w_inject",source+"block_inject_weight.weight",n,r);
    }
    p->read_attention=std::make_unique<HostRecipe>(device,build_hc_read(device,hc),p->select({"w_down","w_up","w_inject","norm"},"attn_"),std::vector<std::string>{"mixed","inject"});
    p->read_ffn=std::make_unique<HostRecipe>(device,build_hc_read(device,hc),p->select({"w_down","w_up","w_inject","norm"},"ffn_"),std::vector<std::string>{"mixed","inject"});
    p->final_mix=std::make_unique<HostRecipe>(device,build_hc_read(device,hc,true),p->select({"w_down","w_up","norm"},"final_"),std::vector<std::string>{"mixed"});
    const std::array<float,1> inverse{1/float(r)};for(auto* read:{p->read_attention.get(),p->read_ffn.get(),p->final_mix.get()}) read->set("inverse_streams",inverse);
    p->write=std::make_unique<HostRecipe>(device,build_hc_write(device,hc),std::span<const Binding>{},std::vector<std::string>{"out"});
    if(options.q4_head || !options.draft_vocab.empty()) {
        if(options.q4_head) p->weights->requantized_q4("subset_head",head_source,options.draft_vocab);
        else if(packed_supported(head_type)) p->weights->packed("subset_head",head_source,options.draft_vocab);
        else p->weights->bf16_subset("subset_head",head_source,options.draft_vocab);
        head_weight=p->weights->binding("subset_head");head_weight.name="weight";
    }
    const auto head_rows=options.draft_vocab.empty()?uint32_t(g.n_vocab):uint32_t(options.draft_vocab.size());
    p->head=std::make_unique<ModelHead>(device,head_weight,uint32_t(h),head_rows,head_type);
    const auto source=p->runtime;
    ExpertResolver resolve=[source](uint32_t layer,int32_t expert) {
        if(layer!=0) throw std::invalid_argument("MTP expert layer must be zero");
        if(expert==-1) return ExpertSlices{source.matrix("mlp.shared_expert.gate_proj.weight"),source.matrix("mlp.shared_expert.up_proj.weight"),source.matrix("mlp.shared_expert.down_proj.weight")};
        const auto slices=source.expert(uint32_t(expert));return ExpertSlices{slices[0],slices[1],slices[2]};
    };
    p->cache=std::make_unique<ExpertCache>(device,g,std::move(resolve),options.expert_cache,options.staging);
    p->moe=std::make_unique<MoeBlock>(device,g,0,*p->cache,p->weights->binding("w_router"),p->weights->binding("w_shared_gate"));impl_=std::move(p);
}
MtpBlock::~MtpBlock()=default;
MtpOutput MtpBlock::step(std::span<const float> residual,std::span<const float> embedding) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);const auto& g=p.geometry;const auto h=uint32_t(g.n_embd);
    if(p.poisoned || p.kv.size()>=p.options.context || residual.size()!=uint64_t(h)*g.hc || embedding.size()!=h) throw std::invalid_argument("MTP state/input/capacity mismatch");
    std::vector<float> hidden(residual.begin(),residual.end()),e(embedding.begin(),embedding.end());
    rms_norm_rows(e,h,p.embedding_gamma,float(g.rms_eps));rms_norm_rows(hidden,p.options.stream_norm?h:uint32_t(hidden.size()),p.hidden_gamma,float(g.rms_eps));
    const auto cells=p.kv.size();const auto point=p.kv.checkpoint();bool rewound=false;
    auto rewind=[&] {
        if(rewound) return;
        rewound=true;
        try {if(p.kv.size()!=cells) p.kv.restore(point);} catch(...) {p.poisoned=true;throw;}
    };
    try {
        const auto projected_embedding=p.input_embedding->run(e);std::vector<float> initial(hidden.size());
        for(uint32_t stream=0;stream<uint64_t(g.hc);++stream) {
            const auto projected_hidden=p.input_hidden->run(std::span<const float>(hidden).subspan(uint64_t(stream)*h,h));
            for(uint32_t i=0;i<h;++i) initial[uint64_t(stream)*h+i]=projected_hidden[i]+projected_embedding[i];
        }
        LayerOperations op;
        op.read=[&](auto r,bool ffn) {auto& read=ffn?*p.read_ffn:*p.read_attention;read.set("residual",r);read.run();return HcReadResult{read.values("mixed"),read.values("inject")};};
        op.mixer=[&](auto x){return p.mixer(x);};op.moe=[&](auto x){return p.moe->run(x);};
        op.write=[&](auto r,auto block,auto inject,bool) {p.write->set("residual",r);p.write->set("block",block);p.write->set("inject",inject);p.write->run();return p.write->values("out");};
        op.rollback_mixer=rewind;auto result=execute_layer(initial,h,uint32_t(g.hc),op);
        auto logits=p.finish_head(result);
        return {std::move(result),std::move(logits)};
    } catch(...) {rewind();throw;}
}
MtpOutput MtpBlock::prefill(std::span<const float> residual,std::span<const float> embedding,const std::function<bool()>& cancelled) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);const auto& g=p.geometry;const auto h=uint32_t(g.n_embd),r=uint32_t(g.hc);
    if(p.poisoned || embedding.empty() || embedding.size()%h || embedding.size()/h>128 || embedding.size()/h>p.options.context-p.kv.size() || residual.size()!=embedding.size()*r)
        throw std::invalid_argument("MTP prefill state/pair/capacity mismatch");
    if(!p.hc_window) p.hc_window=std::make_unique<HcWindow>(p.device,HcConfig{h,r,uint32_t(g.hc_lr),float(g.rms_eps)},kMtpWindowRows);
    if(!p.dense_window) p.dense_window=std::make_unique<QsaWindow>(p.device,p.config,h);
    const auto point=p.kv.checkpoint();const auto cells=p.kv.size();bool rewound=false;
    auto rewind=[&] {
        if(rewound) return;
        rewound=true;
        try {if(p.kv.size()!=cells) p.kv.restore(point);} catch(...) {p.poisoned=true;throw SessionFatal("MTP known-prefix rollback failed");}
    };
    MtpWindowOperations op;
    op.embedding=[&](auto x){return p.input_embedding->run_rows(x);};op.hidden=[&](auto x){return p.input_hidden->run_rows(x);};
    op.head=[&](auto x){return p.finish_head(x);};op.rollback=rewind;
    auto& layer=op.layer;
    layer.read=[&](auto x,bool ffn){auto& read=ffn?*p.read_ffn:*p.read_attention;read.set("residual",x);read.run();return HcReadResult{read.values("mixed"),read.values("inject")};};
    layer.mixer=[&](auto x){return p.mixer(x);};layer.moe=[&](auto x){return p.moe->run(x);};
    layer.write=[&](auto x,auto block,auto inject,bool){p.write->set("residual",x);p.write->set("block",block);p.write->set("inject",inject);p.write->run();return p.write->values("out");};
    layer.rollback_mixer=rewind;
    layer.window_read=[&](auto x,bool ffn){return p.hc_window->read(x,p.select({"w_down","w_up","w_inject","norm"},ffn?"ffn_":"attn_"));};
    layer.window_write=[&](auto x,auto block,auto inject,bool){return p.hc_window->write(x,block,inject);};
    layer.window_mixer=[&](auto x){return p.mixer_window(x,cancelled);};
    return execute_mtp_window(residual,embedding,h,r,uint32_t(g.n_vocab),{p.embedding_gamma,p.hidden_gamma,float(g.rms_eps),p.options.stream_norm},op,cancelled);
}
DenseKvState::Checkpoint MtpBlock::checkpoint() const {auto& p=*impl_;std::lock_guard lock(p.mutex);if(p.poisoned) throw std::logic_error("MTP state failed");return p.kv.checkpoint();}
DenseKvState::Checkpoint MtpBlock::snapshot() const {auto& p=*impl_;std::lock_guard lock(p.mutex);if(p.poisoned) throw std::logic_error("MTP state failed");return p.kv.snapshot();}
uint64_t MtpBlock::snapshot_reused(DenseKvState::Checkpoint&& prior,uint64_t unchanged_tokens,
                                  DenseKvState::Checkpoint& out) const {
    auto& p=*impl_;std::lock_guard lock(p.mutex);if(p.poisoned) throw std::logic_error("MTP state failed");
    const uint64_t cells=p.kv.size(),heads=p.kv.heads(),width=p.kv.width();
    // The cap, applied here rather than by the caller: one token back, so a boundary case loses exactly the last
    // page of reuse (hpu_parked_pages pins that arithmetic).
    const uint64_t unchanged=unchanged_tokens?unchanged_tokens-1:0;
    if(!cells) {
        if(unchanged) throw std::invalid_argument("draft K/V reuse: invalid unchanged prefix");
        // A one-token target conversation has no private draft rows. There
        // is no page layout (and no per-cell divisor) to consult in this case.
        out=p.kv.snapshot_reused(std::move(prior),0);return 0;
    }
    uint64_t sizes[5]={};p.kv.snapshot_sizes(prior,sizes);
    ParkedKvLayout layout;ParkedKvReuse decision;std::string error;
    // The draft pool has no indexer part (index=false) and parks in the drafter's own --kv format: rotated F16,
    // INT8 codes+scales, or rotated Q4_0 blocks. The drafter follows the trunk except under k8v4, where upstream's
    // ring is plain INT8 (src/core/mtp.cpp:344-349) - so the flags come from kv_drafter_mode, never the trunk's
    // hybrid pair, and never a hybrid layout dense_kv.cpp refuses to build.
    const auto drafter=kv_drafter_mode(p.options.kv);
    const bool q4=drafter==KvMode::Q4,int8=drafter==KvMode::Int8;
    if(!parked_kv_layout(q4,int8,false,true,false,cells,false,width,heads,1,layout,error))
        throw std::invalid_argument("draft K/V layout: "+error);
    if(!parked_kv_reuse(layout,sizes,unchanged,decision,error))
        throw std::invalid_argument("draft K/V reuse: "+error);
    out=p.kv.snapshot_reused(std::move(prior),uint32_t(decision.keep[0]/(layout.sizes[0]/layout.cells)));
    return decision.reused_bytes;
}
uint64_t MtpBlock::parked_bytes() const {auto& p=*impl_;std::lock_guard lock(p.mutex);if(p.poisoned) throw std::logic_error("MTP state failed");return p.kv.parked_bytes();}
uint64_t MtpBlock::parked_bytes(uint64_t cells) const {auto& p=*impl_;std::lock_guard lock(p.mutex);if(p.poisoned) throw std::logic_error("MTP state failed");return p.kv.parked_bytes(cells);}
void MtpBlock::restore(const DenseKvState::Checkpoint& point) {auto& p=*impl_;std::lock_guard lock(p.mutex);if(p.poisoned) throw std::logic_error("MTP state failed");p.kv.restore(point);}
void MtpBlock::reset() {auto& p=*impl_;std::lock_guard lock(p.mutex);p.kv.reset();p.poisoned=false;}
uint32_t MtpBlock::size() const {auto& p=*impl_;std::lock_guard lock(p.mutex);return p.kv.size();}
bool MtpBlock::healthy() const {auto& p=*impl_;std::lock_guard lock(p.mutex);return !p.poisoned;}
void MtpBlock::write_snapshot(StateWriter& writer,const DenseKvState::Checkpoint& point) const {auto& p=*impl_;std::lock_guard lock(p.mutex);p.kv.write_snapshot(writer,point);}
DenseKvState::Checkpoint MtpBlock::read_snapshot(StateReader& reader,uint32_t cells) const {auto& p=*impl_;std::lock_guard lock(p.mutex);return p.kv.read_snapshot(reader,cells);}
}
