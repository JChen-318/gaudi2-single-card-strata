#include "strata/hpu/moe.hpp"
#include "strata/hpu/host_recipe.hpp"
#include "strata/hpu/cpu_experts.hpp"
#include "strata/hpu/model_head.hpp"
#include "strata/hpu/packed_expert_flow.hpp"
#include "strata/hpu/model_abi.hpp"
#include "strata/hpu/performance_modes.hpp"
#include <array>
#include <cmath>
#include <cstring>
#include <map>
#include <mutex>
#include <optional>
#include <tuple>
#include <perf_lib_layer_params.h>

namespace strata::hpu {
namespace {
uint32_t bucket_for(uint32_t rows) {for(uint32_t b:{1,8,32,128}) if(rows && rows<=b) return b;throw std::invalid_argument("MoE supports 1..128 rows per call");}
}
struct LegacyExpertExecutor {
    Device device;core::ModelGeometry geometry;uint32_t bucket;
    std::unique_ptr<Recipe> recipe;
    HostBuffer x,y;DeviceBuffer dx,dy;
    Stream input,compute,readback;Event ready,done;
    LegacyExpertExecutor(const Device& d,core::ModelGeometry g,uint32_t b):device(d),geometry(g),bucket(b),recipe(build_bf16_expert(d,g,b)),
        x(d,checked_product(checked_product(b,g.n_embd),2)),y(d,x.size()),dx(d,x.size()),dy(d,y.size()),
        input(d),compute(d),readback(d),ready(d),done(d) {}
};
struct LinearExpertExecutor {
    ExpertFormat format;std::array<std::unique_ptr<ModelHead>,3> projection;
    LinearExpertExecutor(const Device& device,const core::ModelGeometry& g,uint32_t bucket,const ExpertWeights& weights):format(weights.format) {
        const std::array<const DeviceBuffer*,3> buffers{&weights.gate,&weights.up,&weights.down};
        for(size_t role=0;role<3;++role) projection[role]=std::make_unique<ModelHead>(device,Binding{"weight",buffers[role]},uint32_t(role==2?g.n_ff:g.n_embd),
            uint32_t(role==2?g.n_embd:g.n_ff),format.types[role],bucket,format.packed?expert_projection_activation(format,role):PackedActivation::Q8_1);
    }
};
// --fp8-experts: `slots` whole experts as one graph and one launch. Per slot, gate/up/down are fp8_linear
// (MME FP8 x FP8, F32 accumulation, per-row scales bound at their offset behind the codes) and
// silu(gate)*up stays on the device in F32; the slots share nothing, so each result equals its own launch.
// The recipe is built on the first call from that call's buffers; later calls rebind. A slot the call
// does not use is bound to the first expert with zero rows and its output is dropped.
struct Fp8ExpertExecutor {
    Device device;uint32_t slots,bucket;uint64_t hidden,intermediate;std::unique_ptr<HostRecipe> recipe;std::mutex mutex;
    Fp8ExpertExecutor(const Device& d,const core::ModelGeometry& g,uint32_t s,uint32_t b):device(d),slots(s),bucket(b),hidden(uint64_t(g.n_embd)),intermediate(uint64_t(g.n_ff)) {}
    static std::string slot_name(const char* base,uint32_t slot) {return std::string(base)+"_"+std::to_string(slot);}
    std::vector<Binding> bindings(std::span<const ExpertWeights* const> weights) const {
        const auto gu=intermediate*hidden;std::vector<Binding> out;
        for(uint32_t s=0;s<slots;++s) {
            const auto& w=*weights[s<weights.size()?s:0];
            const auto* up=w.format.joined_gate_up?&w.gate:&w.up;
            for(const auto& [base,buffer,offset]:{std::tuple{"w_gate",&w.gate,uint64_t(0)},std::tuple{"s_gate",&w.gate,w.format.joined_gate_up?2*gu:gu},std::tuple{"w_up",up,w.format.joined_gate_up?gu:uint64_t(0)},
                                                  std::tuple{"s_up",up,w.format.joined_gate_up?2*gu+4*intermediate:gu},std::tuple{"w_down",&w.down,uint64_t(0)},std::tuple{"s_down",&w.down,gu}})
                out.push_back({slot_name(base,s),buffer,offset});
        }
        return out;
    }
    void build(const std::vector<Binding>& bound) {
        GraphBuilder graph(device);const uint64_t b=bucket,h=hidden,ff=intermediate;std::vector<std::string> outputs;
        for(uint32_t s=0;s<slots;++s) {
            const auto name=[&](const char* base) {return slot_name(base,s);};
            const auto x=graph.tensor({name("x"),DType::F32,{b,h}});
            const auto wg=graph.tensor({name("w_gate"),DType::F8,{ff,h}}),sg=graph.tensor({name("s_gate"),DType::F32,{1,ff}});
            const auto wu=graph.tensor({name("w_up"),DType::F8,{ff,h}}),su=graph.tensor({name("s_up"),DType::F32,{1,ff}});
            const auto wd=graph.tensor({name("w_down"),DType::F8,{h,ff}}),sd=graph.tensor({name("s_down"),DType::F32,{1,h}});
            const auto gate=graph.tensor({name("gate"),DType::F32,{b,ff},false}),up=graph.tensor({name("up"),DType::F32,{b,ff},false});
            const auto sigmoid=graph.tensor({name("gate_sigmoid"),DType::F32,{b,ff},false}),silu=graph.tensor({name("gate_silu"),DType::F32,{b,ff},false});
            const auto activation=graph.tensor({name("activation"),DType::F32,{b,ff},false}),out=graph.tensor({name("out"),DType::F32,{b,h}});
            graph.fp8_linear(x,wg,sg,gate);graph.fp8_linear(x,wu,su,up);
            graph.sigmoid(gate,sigmoid);graph.multiply(gate,sigmoid,silu);graph.multiply(silu,up,activation);
            graph.fp8_linear(activation,wd,sd,out);outputs.push_back(name("out"));
        }
        recipe=std::make_unique<HostRecipe>(device,graph.compile("strata_fp8_experts_"+std::to_string(slots)+"_"+std::to_string(bucket)),
                                            bound,std::move(outputs),uint64_t(256)<<20);
    }
    // calls[i] runs expert weights[i]; each row block holds at most `bucket` rows.
    std::vector<std::vector<float>> run(std::span<const std::span<const float>> calls,std::span<const ExpertWeights* const> weights) {
        std::lock_guard lock(mutex);
        if(calls.empty() || calls.size()>slots || calls.size()!=weights.size()) throw std::invalid_argument("FP8 expert group size mismatch");
        const auto bound=bindings(weights);
        if(!recipe) build(bound);else for(const auto& binding:bound) recipe->bind_external(binding);
        std::vector<float> padded(uint64_t(bucket)*hidden);
        for(uint32_t s=0;s<slots;++s) {
            std::fill(padded.begin(),padded.end(),0.0f);
            if(s<calls.size()) {
                if(calls[s].empty() || calls[s].size()%hidden || calls[s].size()/hidden>bucket) throw std::invalid_argument("FP8 expert rows exceed bucket");
                std::copy(calls[s].begin(),calls[s].end(),padded.begin());
            }
            recipe->set(slot_name("x",s),padded);
        }
        recipe->run();
        std::vector<std::vector<float>> results;results.reserve(calls.size());
        for(uint32_t s=0;s<calls.size();++s) {
            auto result=recipe->values(slot_name("out",s));result.resize(calls[s].size());
            if(!all_finite(result)) throw std::runtime_error("non-finite FP8 expert output");
            results.push_back(std::move(result));
        }
        return results;
    }
};
// One executor per (device, slots, bucket), shared by every MoE block: the graph does not depend on the
// layer, so it is compiled once instead of once per layer.
static std::shared_ptr<Fp8ExpertExecutor> shared_fp8_executor(const Device& device,const core::ModelGeometry& g,uint32_t slots,uint32_t bucket) {
    static std::mutex registry_mutex;
    static std::map<std::tuple<const void*,uint32_t,uint32_t>,std::weak_ptr<Fp8ExpertExecutor>> registry;
    std::lock_guard lock(registry_mutex);
    auto& entry=registry[{device.state().get(),slots,bucket}];
    if(auto existing=entry.lock()) return existing;
    auto created=std::make_shared<Fp8ExpertExecutor>(device,g,slots,bucket);entry=created;return created;
}
// --fp8-experts single-token decode: every HPU expert of one MoE call reads the same row, so the whole group
// is a handful of large MME ops instead of one small graph per expert (measured 1.36 ms per layer for 11
// experts as separate fp8_linear subgraphs, almost all of it node overhead). The row is FP8-encoded once
// (hi + lo/16, per-row dynamic scale, as fp8_linear); gate and up are one GEMM each against the slots'
// weights concatenated to [E*ff,h]; silu(gate)*up is encoded per expert row; down is one batch GEMM
// [E,1,ff] x [E,h,ff]^T. Output row e is expert e's unweighted result, the same law as fp8_linear.
struct Fp8ExpertGroup {
    // With `write` set (fused decode) the graph also takes the slot weights [E,1] (routed probabilities and
    // the shared gate), sums the weighted rows and applies the FFN HC write: residual [r,h] + inject [r]
    // in, "residual_out" [r,h] out; that sum runs on the device in its own order, not the host's.
    Device device;uint32_t slots;uint64_t hidden,intermediate;std::optional<HcConfig> write;std::unique_ptr<HostRecipe> recipe;std::mutex mutex;
    // STRATA_HPU_CHAIN: the same write graph compiled a second time with x, residual, inject and write.out bound
    // to the token's device buffers instead of host slots (built while building_chain is set).
    std::unique_ptr<HostRecipe> chain_recipe;bool building_chain=false;std::vector<Binding> chain_io;
    Fp8ExpertGroup(const Device& d,const core::ModelGeometry& g,uint32_t s,std::optional<HcConfig> w={})
        :device(d),slots(s),hidden(uint64_t(g.n_embd)),intermediate(uint64_t(g.n_ff)),write(w) {}
    static std::string slot_name(const char* base,uint32_t slot) {return std::string(base)+"_"+std::to_string(slot);}
    void install_write(GraphBuilder& g,const std::string& name,const std::vector<Binding>& bound) {
        if(!building_chain) {recipe=std::make_unique<HostRecipe>(device,g.compile(name),bound,std::vector<std::string>{"write.out"},uint64_t(64)<<20);return;}
        auto all=bound;all.insert(all.end(),chain_io.begin(),chain_io.end());
        chain_recipe=std::make_unique<HostRecipe>(device,g.compile(name+"_chain"),all,std::vector<std::string>{},uint64_t(64)<<20);
    }
    std::vector<Binding> bindings(std::span<const ExpertWeights* const> weights) const {
        const auto gu=intermediate*hidden;std::vector<Binding> out;
        for(uint32_t s=0;s<slots;++s) {
            const auto& w=*weights[s<weights.size()?s:0];
            if(tpc_experts_enabled()) {
                // The TPC GEMV reads each resident buffer whole: codes, then the per-row scales.
                if(!w.format.joined_gate_up) throw std::invalid_argument("TPC expert requires joined gate/up storage");
                if(w.format.fp8_column_pairs!=tpc_colpair_enabled()) throw std::invalid_argument("TPC expert column-pair layout mismatch");
                out.push_back({slot_name("w_gu",s),&w.gate,0});out.push_back({slot_name("w_down",s),&w.down,0});
                continue;
            }
            if(direct_experts_enabled()) {
                if(!w.format.joined_gate_up) throw std::invalid_argument("direct expert requires joined gate/up storage");
                out.push_back({slot_name("w_gu",s),&w.gate,0});out.push_back({slot_name("s_gu",s),&w.gate,2*gu});
                out.push_back({slot_name("w_down",s),&w.down,0});out.push_back({slot_name("s_down",s),&w.down,gu});
                continue;
            }
            for(const auto& [base,buffer,offset]:{std::tuple{"w_gate",&w.gate,uint64_t(0)},std::tuple{"s_gate",&w.gate,gu},std::tuple{"w_up",&w.up,uint64_t(0)},
                                                  std::tuple{"s_up",&w.up,gu},std::tuple{"w_down",&w.down,uint64_t(0)},std::tuple{"s_down",&w.down,gu}})
                out.push_back({slot_name(base,s),buffer,offset});
        }
        return out;
    }
    // FP8 hi/lo codes of every row of x[rows,columns] and its per-row scale [rows,1]. The hi codes are a
    // persistent tensor: as an internal tensor the compiler folds the F32->FP8->F32 round trip away.
    struct Encoded {TensorId high,low,scale;};
    static Encoded encode(GraphBuilder& g,TensorId x,uint64_t rows,uint64_t columns,const std::string& name) {
        const auto t=[&](const char* part,DType dtype,std::vector<uint64_t> shape,bool persistent=false) {
            return g.tensor({name+"_"+part,dtype,std::move(shape),persistent});
        };
        if(fused_fp8_enabled()) {
            const auto hi=t("hi",DType::F8,{rows,columns},true),lo=t("lo",DType::F8,{rows,columns});
            const auto scale=t("scale",DType::F32,{rows,1});
            abi::Fp8Encode args{int32_t(columns),int32_t(rows)};
            const TensorId ins[]={x},outs[]={hi,lo,scale};
            g.op("strata_fp8_encode_f32",ins,outs,&args,sizeof(args));
            return {hi,lo,scale};
        }
        const auto magnitude=t("abs",DType::F32,{rows,columns}),peak=t("peak",DType::F32,{rows,1});
        const auto inverse_range=t("inverse_range",DType::F32,{1}),epsilon=t("epsilon",DType::F32,{1});
        const auto scaled_peak=t("scaled_peak",DType::F32,{rows,1}),scale=t("scale",DType::F32,{rows,1});
        const auto scaled=t("scaled",DType::F32,{rows,columns}),high=t("hi",DType::F8,{rows,columns},true);
        const auto high_f32=t("hi_f32",DType::F32,{rows,columns}),residual=t("residual",DType::F32,{rows,columns});
        const auto sixteen=t("sixteen",DType::F32,{1}),lifted=t("lifted",DType::F32,{rows,columns}),low=t("lo",DType::F8,{rows,columns});
        const auto constant=[&](TensorId out,float value) {
            ns_ConstantKernel::Params params{};params.constant.f=value;const TensorId outs[]={out};g.op("constant_f32",{},outs,&params,sizeof(params));
        };
        const auto binary=[&](const char* guid,TensorId a,TensorId b,TensorId out) {const TensorId ins[]={a,b},outs[]={out};g.op(guid,ins,outs);};
        {const TensorId ins[]={x},outs[]={magnitude};g.op("abs_fwd_f32",ins,outs);}
        {ns_Reduction::Params params{};params.reductionDimension=0;const TensorId ins[]={magnitude},outs[]={peak};g.op("reduce_max_fwd_f32",ins,outs,&params,sizeof(params));}
        constant(inverse_range,1.0f/240.0f);constant(epsilon,1e-30f);constant(sixteen,16.0f);
        binary("mult_fwd_f32",peak,inverse_range,scaled_peak);binary("add_fwd_f32",scaled_peak,epsilon,scale);
        binary("div_fwd_f32",x,scale,scaled);g.cast(scaled,high);g.cast(high,high_f32);
        binary("sub_fwd_f32",scaled,high_f32,residual);binary("mult_fwd_f32",residual,sixteen,lifted);g.cast(lifted,low);
        return {high,low,scale};
    }
    // Direct MME projections bind each expert's resident joined gate/up matrix. Only small
    // activations/results are explicitly concatenated; the compiler may still stage MME inputs.
    void build_direct(const std::vector<Binding>& bound) {
        GraphBuilder g(device);const uint64_t e=slots,h=hidden,ff=intermediate;
        const auto t=[&](const std::string& name,DType dtype,std::vector<uint64_t> shape,bool persistent=false) {
            return g.tensor({name,dtype,std::move(shape),persistent});
        };
        const auto binary=[&](const char* guid,TensorId a,TensorId b,TensorId out) {const TensorId in[]={a,b},outputs[]={out};g.op(guid,in,outputs);};
        std::vector<TensorId> wgu,sgu,wd,sd;
        for(uint32_t s=0;s<slots;++s) {
            wgu.push_back(t(slot_name("w_gu",s),DType::F8,{2*ff,h},true));sgu.push_back(t(slot_name("s_gu",s),DType::F32,{1,2*ff},true));
            wd.push_back(t(slot_name("w_down",s),DType::F8,{h,ff},true));sd.push_back(t(slot_name("s_down",s),DType::F32,{1,h},true));
        }
        const auto x=t("x",DType::F32,{1,h},true);const auto input=encode(g,x,1,h,"x");
        const auto sixteenth=t("sixteenth",DType::F32,{1});
        {ns_ConstantKernel::Params p{};p.constant.f=1.0f/16;const TensorId out[]={sixteenth};g.op("constant_f32",{},out,&p,sizeof(p));}
        const auto project=[&](const std::string& name,TensorId high,TensorId low,TensorId weights,TensorId row_scale,TensorId scales,uint64_t width) {
            const auto main=t(name+"_main",DType::F32,{1,width}),lower=t(name+"_low",DType::F32,{1,width});
            const auto lower_scaled=t(name+"_low_scaled",DType::F32,{1,width}),sum=t(name+"_sum",DType::F32,{1,width});
            const auto rescaled=t(name+"_rescaled",DType::F32,{1,width}),out=t(name,DType::F32,{1,width});
            g.gemm(high,weights,main,true);g.gemm(low,weights,lower,true);
            binary("mult_fwd_f32",lower,sixteenth,lower_scaled);binary("add_fwd_f32",main,lower_scaled,sum);
            binary("mult_fwd_f32",sum,row_scale,rescaled);binary("mult_fwd_f32",rescaled,scales,out);return out;
        };
        std::vector<TensorId> gu;
        for(uint32_t s=0;s<slots;++s) gu.push_back(project(slot_name("gu",s),input.high,input.low,wgu[s],input.scale,sgu[s],2*ff));
        const auto gu_rows=t("gu_rows",DType::F32,{e,2*ff});g.concatenate_rows(gu,gu_rows);
        const auto gate=t("gate",DType::F32,{e,ff}),up=t("up",DType::F32,{e,ff});
        {synSplitParams p{};p.axis=0;const TensorId in[]={gu_rows},out[]={gate,up};g.op("split",in,out,&p,sizeof(p));}
        const auto sigmoid=t("gate_sigmoid",DType::F32,{e,ff}),silu=t("gate_silu",DType::F32,{e,ff}),activation=t("activation",DType::F32,{e,ff});
        g.sigmoid(gate,sigmoid);g.multiply(gate,sigmoid,silu);g.multiply(silu,up,activation);
        const auto codes=encode(g,activation,e,ff,"activation");
        std::vector<TensorId> highs,lows,scales;
        for(uint32_t s=0;s<slots;++s) {
            highs.push_back(t(slot_name("activation_hi",s),DType::F8,{1,ff}));
            lows.push_back(t(slot_name("activation_lo",s),DType::F8,{1,ff}));
            scales.push_back(t(slot_name("activation_scale",s),DType::F32,{1,1}));
        }
        g.split_rows(codes.high,highs);g.split_rows(codes.low,lows);g.split_rows(codes.scale,scales);
        std::vector<TensorId> down;
        for(uint32_t s=0;s<slots;++s) down.push_back(project(slot_name("down",s),highs[s],lows[s],wd[s],scales[s],sd[s],h));
        const auto rows=t("expert_rows",DType::F32,{e,h},!write);g.concatenate_rows(down,rows);
        if(!write) {
            recipe=std::make_unique<HostRecipe>(device,g.compile("strata_fp8_direct_group_"+std::to_string(slots)),bound,std::vector<std::string>{"expert_rows"},uint64_t(64)<<20);
            return;
        }
        const auto route=t("route_weights",DType::F32,{e,1},true),weighted=t("weighted_rows",DType::F32,{e,h});
        binary("mult_fwd_f32",rows,route,weighted);
        const auto summed=t("expert_sum",DType::F32,{1,h}),block=t("expert_block",DType::F32,{h});
        {ns_Reduction::Params p{};p.reductionDimension=1;const TensorId in[]={weighted},out[]={summed};g.op("reduce_sum_fwd_f32",in,out,&p,sizeof(p));}
        g.reshape(summed,block);
        const auto residual=t("residual",DType::F32,{write->streams,h},true),inject=t("inject",DType::F32,{write->streams},true);
        g.alias("residual",residual);g.alias("block",block);g.alias("inject",inject);
        g.set_scope("write.");define_hc_write(g,*write);g.set_scope({});
        install_write(g,"strata_fp8_direct_group_write_"+std::to_string(slots),bound);
    }
    // STRATA_HPU_TPC_EXPERTS: one TPC GEMV node for all slots' gate/up (FP8 codes x BF16 row, F32 accumulation,
    // no activation quantization), silu(gate)*up, one TPC GEMV node for all slots' down. The kernel reads x as
    // BF16 permuted per 256-column chunk (position c+128k+j holds x[c+2j+k], the native f8->bf16 lane order),
    // zero padded to a multiple of 256.
    void build_tpc(const std::vector<Binding>& bound) {
        GraphBuilder g(device);const uint64_t e=slots,h=hidden,ff=intermediate;
        if(e!=11 || h%256 || ff%64 || h>2560 || ff>2560) throw std::invalid_argument("TPC expert group supports 11 slots, hidden%256==0");
        const auto t=[&](const std::string& name,DType dtype,std::vector<uint64_t> shape,bool persistent=false) {
            return g.tensor({name,dtype,std::move(shape),persistent});
        };
        const auto binary=[&](const char* guid,TensorId a,TensorId b,TensorId out) {const TensorId in[]={a,b},outputs[]={out};g.op(guid,in,outputs);};
        std::vector<TensorId> gate_up{0},down{0}; // slot 0 of each list is the x input, filled below
        for(uint32_t s=0;s<slots;++s) {
            gate_up.push_back(t(slot_name("w_gu",s),DType::U8,{2*ff*h+2*ff*4},true));
            down.push_back(t(slot_name("w_down",s),DType::U8,{h*ff+h*4},true));
        }
        const auto permuted=[&](TensorId in,uint64_t rows,uint64_t width,const std::string& name) {
            const auto chunks=t(name+"_chunks",DType::F32,{rows*width/256,128,2}),swapped=t(name+"_swapped",DType::F32,{rows*width/256,2,128});
            const auto wide=t(name+"_permuted_f32",DType::F32,{rows,width}),out=t(name+"_permuted",DType::BF16,{rows,width});
            g.reshape(in,chunks);
            synTransposeParams p{};p.tensorDim=3;p.permutation[0]=TPD_Width;p.permutation[1]=TPD_Channel;p.permutation[2]=TPD_Height;
            const TensorId i[]={chunks},o[]={swapped};g.op("transpose",i,o,&p,sizeof(p));
            g.reshape(swapped,wide);g.cast(wide,out);return out;
        };
        const auto gemv=[&](std::vector<TensorId>& inputs,TensorId x,TensorId out,uint64_t columns,uint64_t rows,bool shared,const std::string& name) {
            abi::Fp8Gemv a{int32_t(columns),int32_t(rows),shared?1:0};const TensorId o[]={out};
            const uint64_t padded=tpc_colpair_enabled()?columns:(columns+255)/256*256,xrows=shared?1:e;
            if(tpc_colpair_enabled()) {
                const auto bf=t(name+"_colpair_bf16",DType::BF16,{xrows,columns});g.cast(x,bf);
                inputs[0]=bf;g.op("strata_fp8_gemv_colpair_f32",inputs,o,&a,sizeof(a));
            } else if(tpc_hilo_enabled()) {
                const auto codes=encode(g,x,xrows,padded,name+"_codes");
                const auto high=t(name+"_high_f32",DType::F32,{xrows,padded}),low=t(name+"_low_f32",DType::F32,{xrows,padded});
                g.cast(codes.high,high);g.cast(codes.low,low);
                std::vector<TensorId> all{permuted(high,xrows,padded,name+"_hi"),permuted(low,xrows,padded,name+"_lo"),codes.scale};
                all.insert(all.end(),inputs.begin()+1,inputs.end());
                g.op("strata_fp8_gemv_hilo_f32",all,o,&a,sizeof(a));
            } else {
                inputs[0]=permuted(x,xrows,padded,name);
                g.op("strata_fp8_gemv_f32",inputs,o,&a,sizeof(a));
            }
        };
        const auto x=t("x",DType::F32,{1,h},true);
        const auto gu=t("gu_rows",DType::F32,{e,2*ff});gemv(gate_up,x,gu,h,2*ff,true,"x");
        const auto gate=t("gate",DType::F32,{e,ff}),up=t("up",DType::F32,{e,ff});
        {synSplitParams p{};p.axis=0;const TensorId in[]={gu},out[]={gate,up};g.op("split",in,out,&p,sizeof(p));}
        const auto sigmoid=t("gate_sigmoid",DType::F32,{e,ff}),silu=t("gate_silu",DType::F32,{e,ff}),activation=t("activation",DType::F32,{e,ff});
        g.sigmoid(gate,sigmoid);g.multiply(gate,sigmoid,silu);g.multiply(silu,up,activation);
        const uint64_t padded=tpc_colpair_enabled()?ff:(ff+255)/256*256;auto wide=activation;
        if(padded!=ff) {
            const auto zeros=t("activation_zeros",DType::F32,{e,padded-ff});wide=t("activation_padded",DType::F32,{e,padded});
            {ns_ConstantKernel::Params p{};p.constant.f=0;const TensorId out[]={zeros};g.op("constant_f32",{},out,&p,sizeof(p));}
            const TensorId parts[]={activation,zeros};g.concat(parts,wide,1);
        }
        const auto rows=t("expert_rows",DType::F32,{e,h},!write);gemv(down,wide,rows,ff,h,false,"activation");
        if(!write) {
            recipe=std::make_unique<HostRecipe>(device,g.compile("strata_fp8_tpc_group_"+std::to_string(slots)),bound,std::vector<std::string>{"expert_rows"},uint64_t(64)<<20);
            return;
        }
        const auto route=t("route_weights",DType::F32,{e,1},true),weighted=t("weighted_rows",DType::F32,{e,h});
        binary("mult_fwd_f32",rows,route,weighted);
        const auto summed=t("expert_sum",DType::F32,{1,h}),block=t("expert_block",DType::F32,{h});
        {ns_Reduction::Params p{};p.reductionDimension=1;const TensorId in[]={weighted},out[]={summed};g.op("reduce_sum_fwd_f32",in,out,&p,sizeof(p));}
        g.reshape(summed,block);
        const auto residual=t("residual",DType::F32,{write->streams,h},true),inject=t("inject",DType::F32,{write->streams},true);
        g.alias("residual",residual);g.alias("block",block);g.alias("inject",inject);
        g.set_scope("write.");define_hc_write(g,*write);g.set_scope({});
        install_write(g,"strata_fp8_tpc_group_write_"+std::to_string(slots),bound);
    }
    void build(const std::vector<Binding>& bound) {
        if(tpc_experts_enabled()) {build_tpc(bound);return;}
        if(direct_experts_enabled()) {build_direct(bound);return;}
        GraphBuilder g(device);const uint64_t e=slots,h=hidden,ff=intermediate;
        const auto t=[&](const std::string& name,DType dtype,std::vector<uint64_t> shape,bool persistent=false) {
            return g.tensor({name,dtype,std::move(shape),persistent});
        };
        const auto binary=[&](const char* guid,TensorId a,TensorId b,TensorId out) {const TensorId ins[]={a,b},outs[]={out};g.op(guid,ins,outs);};
        std::vector<TensorId> wg,sg,wu,su,wd,sd;
        for(uint32_t s=0;s<slots;++s) {
            wg.push_back(t(slot_name("w_gate",s),DType::F8,{ff,h},true));sg.push_back(t(slot_name("s_gate",s),DType::F32,{1,ff},true));
            wu.push_back(t(slot_name("w_up",s),DType::F8,{ff,h},true));su.push_back(t(slot_name("s_up",s),DType::F32,{1,ff},true));
            wd.push_back(t(slot_name("w_down",s),DType::F8,{h,ff},true));sd.push_back(t(slot_name("s_down",s),DType::F32,{1,h},true));
        }
        const auto x=t("x",DType::F32,{1,h},true);
        const auto all_gate=t("gate_weights",DType::F8,{e*ff,h}),all_up=t("up_weights",DType::F8,{e*ff,h}),all_down=t("down_weights",DType::F8,{e*h,ff});
        const auto gate_scales=t("gate_scales",DType::F32,{1,e*ff}),up_scales=t("up_scales",DType::F32,{1,e*ff}),down_scales=t("down_scales",DType::F32,{e,h});
        g.concat(wg,all_gate,0);g.concat(wu,all_up,0);g.concat(wd,all_down,0);
        g.concat(sg,gate_scales,1);g.concat(su,up_scales,1);g.concat(sd,down_scales,0);
        const auto input=encode(g,x,1,h,"x");
        const auto sixteenth=t("sixteenth",DType::F32,{1});
        {ns_ConstantKernel::Params params{};params.constant.f=1.0f/16.0f;const TensorId outs[]={sixteenth};g.op("constant_f32",{},outs,&params,sizeof(params));}
        // y = ((hi.W + (lo.W)/16) * row scale) * weight scales, for one [rows,width] projection.
        const auto project=[&](const std::string& name,TensorId high,TensorId low,TensorId weights,TensorId row_scale,TensorId weight_scales,
                               std::vector<uint64_t> shape,bool batched) {
            const auto main=t(name+"_main",DType::F32,shape),lower=t(name+"_low",DType::F32,shape),lower_scaled=t(name+"_low_scaled",DType::F32,shape);
            const auto sum=t(name+"_sum",DType::F32,shape),rescaled=t(name+"_rescaled",DType::F32,shape),out=t(name,DType::F32,shape);
            if(batched) {g.batch_gemm(high,weights,main,true);g.batch_gemm(low,weights,lower,true);}
            else {g.gemm(high,weights,main,true);g.gemm(low,weights,lower,true);}
            binary("mult_fwd_f32",lower,sixteenth,lower_scaled);binary("add_fwd_f32",main,lower_scaled,sum);
            binary("mult_fwd_f32",sum,row_scale,rescaled);binary("mult_fwd_f32",rescaled,weight_scales,out);
            return out;
        };
        const auto gate=project("gate",input.high,input.low,all_gate,input.scale,gate_scales,{1,e*ff},false);
        const auto up=project("up",input.high,input.low,all_up,input.scale,up_scales,{1,e*ff},false);
        const auto sigmoid=t("gate_sigmoid",DType::F32,{1,e*ff}),silu=t("gate_silu",DType::F32,{1,e*ff}),activation=t("activation",DType::F32,{1,e*ff});
        g.sigmoid(gate,sigmoid);g.multiply(gate,sigmoid,silu);g.multiply(silu,up,activation);
        const auto rows=t("activation_rows",DType::F32,{e,ff});g.reshape(activation,rows);
        const auto hidden_codes=encode(g,rows,e,ff,"activation");
        const auto high3=t("activation_hi3",DType::F8,{e,1,ff}),low3=t("activation_lo3",DType::F8,{e,1,ff});
        g.reshape(hidden_codes.high,high3);g.reshape(hidden_codes.low,low3);
        const auto down3=t("down_weights3",DType::F8,{e,h,ff});g.reshape(all_down,down3);
        const auto scale3=t("activation_scale3",DType::F32,{e,1,1});g.reshape(hidden_codes.scale,scale3);
        const auto down_scales3=t("down_scales3",DType::F32,{e,1,h});g.reshape(down_scales,down_scales3);
        const auto down=project("down",high3,low3,down3,scale3,down_scales3,{e,1,h},true);
        if(!write) {
            const auto out=t("out",DType::F32,{e,h},true);g.reshape(down,out);
            recipe=std::make_unique<HostRecipe>(device,g.compile("strata_fp8_expert_group_"+std::to_string(slots)),bound,std::vector<std::string>{"out"},uint64_t(64)<<20);
            return;
        }
        const auto rows_out=t("expert_rows",DType::F32,{e,h});g.reshape(down,rows_out);
        const auto route=t("route_weights",DType::F32,{e,1},true),weighted=t("weighted_rows",DType::F32,{e,h});
        binary("mult_fwd_f32",rows_out,route,weighted);
        const auto summed=t("expert_sum",DType::F32,{1,h}),block=t("expert_block",DType::F32,{h});
        {ns_Reduction::Params params{};params.reductionDimension=1; // logical axis 0 of [E,h]
         const TensorId ins[]={weighted},outs[]={summed};g.op("reduce_sum_fwd_f32",ins,outs,&params,sizeof(params));}
        g.reshape(summed,block);
        const auto residual=t("residual",DType::F32,{write->streams,h},true),inject=t("inject",DType::F32,{write->streams},true);
        g.alias("residual",residual);g.alias("block",block);g.alias("inject",inject);
        g.set_scope("write.");define_hc_write(g,*write);g.set_scope({});
        install_write(g,"strata_fp8_expert_group_write_"+std::to_string(slots),bound);
    }
    // Fused decode: the FFN half from the router's routes to the next residual (`write` mode only).
    std::vector<float> run_write(std::span<const float> row,std::span<const float> route,std::span<const float> residual,
                                 std::span<const float> inject,std::span<const ExpertWeights* const> weights) {
        std::lock_guard lock(mutex);
        if(!write || weights.empty() || weights.size()>slots || row.size()!=hidden || route.size()!=weights.size())
            throw std::invalid_argument("FP8 fused FFN shape mismatch");
        const auto bound=bindings(weights);
        if(!recipe) build(bound);else for(const auto& binding:bound) recipe->bind_external(binding);
        std::vector<float> padded(slots,0.0f);std::copy(route.begin(),route.end(),padded.begin()); // unused slots weigh zero
        recipe->set("x",row);recipe->set("route_weights",padded);recipe->set("residual",residual);recipe->set("inject",inject);
        recipe->run();
        auto out=recipe->values("write.out");
        if(!all_finite(out)) throw std::runtime_error("non-finite fused FFN output");
        return out;
    }
    // STRATA_HPU_CHAIN: the same FFN half reading x, mid (as the residual) and inject from the chain and writing chain.out(),
    // launched on the chain's stream without waiting.
    void launch_chain(std::span<const float> route,std::span<const ExpertWeights* const> weights,DecodeChain& chain) {
        std::lock_guard lock(mutex);
        if(!write || weights.empty() || weights.size()>slots || route.size()!=weights.size()) throw std::invalid_argument("FP8 chained FFN shape mismatch");
        const auto bound=bindings(weights);
        chain_io={{"x",&chain.x},{"residual",&chain.mid},{"inject",&chain.inject},{"write.out",&chain.out()}};
        if(!chain_recipe) {building_chain=true;try {build(bound);} catch(...) {building_chain=false;throw;} building_chain=false;}
        else {for(const auto& binding:bound) chain_recipe->bind_external(binding);for(const auto& binding:chain_io) chain_recipe->bind_external(binding);}
        std::vector<float> padded(slots,0.0f);std::copy(route.begin(),route.end(),padded.begin()); // unused slots weigh zero
        chain_recipe->set("route_weights",padded);chain_recipe->launch(chain.stream);
    }
    // Every call carries the same single row (one decode token); result i is expert weights[i] on it.
    std::vector<std::vector<float>> run(std::span<const float> row,std::span<const ExpertWeights* const> weights) {
        std::lock_guard lock(mutex);
        if(weights.empty() || weights.size()>slots || row.size()!=hidden) throw std::invalid_argument("FP8 expert group shape mismatch");
        const auto bound=bindings(weights);
        if(!recipe) build(bound);else for(const auto& binding:bound) recipe->bind_external(binding);
        recipe->set("x",row);recipe->run();
        const auto all=recipe->values(direct_experts_enabled() || tpc_experts_enabled()?"expert_rows":"out");
        if(!all_finite(all)) throw std::runtime_error("non-finite FP8 expert group output");
        std::vector<std::vector<float>> results;results.reserve(weights.size());
        for(size_t i=0;i<weights.size();++i) results.emplace_back(all.begin()+int64_t(i*hidden),all.begin()+int64_t((i+1)*hidden));
        return results;
    }
};
static std::shared_ptr<Fp8ExpertGroup> shared_fp8_group(const Device& device,const core::ModelGeometry& g,uint32_t slots,std::optional<HcConfig> write={}) {
    static std::mutex registry_mutex;
    static std::map<std::tuple<const void*,uint32_t,bool>,std::weak_ptr<Fp8ExpertGroup>> registry;
    std::lock_guard lock(registry_mutex);
    auto& entry=registry[{device.state().get(),slots,bool(write)}];
    if(auto existing=entry.lock()) return existing;
    auto created=std::make_shared<Fp8ExpertGroup>(device,g,slots,write);entry=created;return created;
}
struct ExpertExecutor::Impl {
    Device device;core::ModelGeometry geometry;uint32_t bucket;std::mutex mutex;
    std::unique_ptr<LegacyExpertExecutor> legacy;
    std::shared_ptr<Fp8ExpertExecutor> fp8;
    // Keep one context per activation contract: shared and routed calls
    // alternate in every token. They must not recompile each other's graphs.
    std::array<std::unique_ptr<LinearExpertExecutor>,3> packed;
    std::unique_ptr<LinearExpertExecutor> scaled; // BF16 weights, separate F32 projection outputs
    Impl(const Device& d,core::ModelGeometry g,uint32_t b):device(d),geometry(g),bucket(b) {}
};
ExpertExecutor::ExpertExecutor(const Device& d,core::ModelGeometry g,uint32_t b) {
    if(!b || b>128 || g.n_embd<=0 || g.n_ff<=0) throw std::invalid_argument("invalid expert executor geometry");
    impl_=std::make_unique<Impl>(d,g,b);
}
ExpertExecutor::~ExpertExecutor()=default;
std::vector<float> ExpertExecutor::run(std::span<const float> rows,ExpertLease lease) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);const uint64_t h=p.geometry.n_embd,ff=p.geometry.n_ff;
    if(!lease || rows.empty() || rows.size()%h || rows.size()/h>p.bucket) throw std::invalid_argument("expert activation shape/lease mismatch");
    const auto& w=lease.get();
    const std::array<const DeviceBuffer*,3> buffers{&w.gate,&w.up,&w.down};
    if(w.format.fp8) {
        if(w.format.fp8_column_pairs) throw std::invalid_argument("column-pair experts require single-row TPC group execution");
        for(size_t role=0;role<3;++role)
            if(buffers[role]->state()!=p.device.state() || buffers[role]->size()!=(role==2?fp8_role_bytes(h,ff):
                w.format.joined_gate_up?(role==0?2*fp8_role_bytes(ff,h):0):fp8_role_bytes(ff,h)))
                throw std::invalid_argument("FP8 expert resident weight device/shape mismatch");
        if(!p.fp8) p.fp8=shared_fp8_executor(p.device,p.geometry,1,p.bucket);
        const std::array<std::span<const float>,1> calls{rows};const std::array<const ExpertWeights*,1> weights{&w};
        return std::move(p.fp8->run(calls,weights).front());
    }
    for(size_t role=0;role<3;++role) {
        const auto expected=w.format.packed?checked_product(packed_row_bytes(w.format.types[role],role==2?ff:h),role==2?h:ff):checked_product(checked_product(h,ff),2);
        if(buffers[role]->state()!=p.device.state() || buffers[role]->size()!=expected) throw std::invalid_argument("expert resident weight device/shape mismatch");
    }
    if(w.format.packed || w.scaled_outputs) {
        const auto context=uint32_t(w.format.activation);if(context>=p.packed.size()) throw std::invalid_argument("invalid expert activation contract");
        // A packed expert keeps its own context even when it carries global scales; only a
        // decoded reference expert uses the scaled executor.
        auto& execution=w.format.packed?p.packed[context]:p.scaled;
        if(!execution || execution->format!=w.format) {execution.reset();execution=std::make_unique<LinearExpertExecutor>(p.device,p.geometry,p.bucket,w);}
        for(size_t role=0;role<3;++role) execution->projection[role]->bind({"weight",buffers[role]});
        const auto count=rows.size()/h;
        return execute_packed_expert(rows,uint32_t(h),uint32_t(ff),[&](uint32_t role,std::span<const float> activation) {
            const uint64_t columns=role==2?ff:h,width=role==2?h:ff;
            std::vector<float> padded(uint64_t(p.bucket)*columns);std::copy(activation.begin(),activation.end(),padded.begin());
            auto result=execution->projection[role]->run(padded);result.resize(count*width);return result;
        },w.output_scales);
    }
    if(!p.legacy) p.legacy=std::make_unique<LegacyExpertExecutor>(p.device,p.geometry,p.bucket);
    auto& b=*p.legacy;
    for(float value:rows) if(!std::isfinite(value) || !std::isfinite(from_bf16(to_bf16(value)))) throw std::invalid_argument("expert requires finite BF16 activations");
    auto* input=static_cast<uint16_t*>(b.x.data());std::fill_n(input,b.x.size()/2,0);
    for(size_t i=0;i<rows.size();++i) input[i]=to_bf16(rows[i]);
    try {
        upload(b.input,b.dx,b.x,b.x.size());b.ready.record(b.input);b.compute.wait(b.ready);
        const std::array<Binding,5> bindings{{{"x",&b.dx},{"out",&b.dy},{"w_gate",&w.gate},{"w_up",&w.up},{"w_down",&w.down}}};
        b.recipe->enqueue(b.compute,bindings);b.readback.wait(b.recipe->completion());download(b.readback,b.y,b.dy,b.y.size());
        b.done.record(b.readback);b.done.synchronize();
    } catch(...) {p.device.synchronize();throw;}
    std::vector<float> output(rows.size());const auto* bf16=static_cast<const uint16_t*>(b.y.data());
    for(size_t i=0;i<output.size();++i) {output[i]=from_bf16(bf16[i]);if(!std::isfinite(output[i])) throw std::runtime_error("non-finite BF16 expert output");}
    return output;
}
struct MoeBlock::Impl {
    Device device;core::ModelGeometry geometry;uint32_t layer;ExpertCache* cache;
    // Owned router buffers exist only for the WeightTable constructor. Bindings
    // provided through the other constructor are borrowed for this block's life.
    std::unique_ptr<DeviceBuffer> owned_router,owned_shared_gate;
    Binding router_weight,shared_gate_weight;
    std::unique_ptr<HostRecipe> router;uint32_t router_bucket=0;
    std::map<uint32_t,std::unique_ptr<ExpertExecutor>> executors;std::mutex mutex;
    std::shared_ptr<Fp8ExpertGroup> group,group_write; // --fp8-experts one-launch decode / fused FFN half
    MoeOptions options;
    Impl(const Device& d,core::ModelGeometry g,uint32_t l,ExpertCache& c):device(d),geometry(g),layer(l),cache(&c) {}
    ExpertExecutor& executor(uint32_t rows) {
        const auto b=bucket_for(rows);auto it=executors.find(b);
        if(it==executors.end()) it=executors.emplace(b,std::make_unique<ExpertExecutor>(device,geometry,b)).first;
        return *it->second;
    }
};
namespace {
void validate_geometry(const core::ModelGeometry& g,uint32_t layer,const ExpertCache& cache) {
    const auto& c=cache.geometry();
    if(g.n_embd<=0 || g.n_ff<=0 || g.n_expert<=0 || g.n_expert>4096 || g.n_expert_used<=0 || g.n_expert_used>g.n_expert ||
       g.n_expert_used>64 || layer>=uint64_t(g.n_layers) || g.n_embd!=c.n_embd || g.n_ff!=c.n_ff ||
       g.n_layers!=c.n_layers || g.n_expert!=c.n_expert) throw std::invalid_argument("MoE block/cache geometry mismatch");
}
void validate_binding(const Device& d,const Binding& b,const char* name,uint64_t bytes) {
    if(b.name!=name || !b.buffer || b.buffer->state()!=d.state() || (b.buffer->address()+b.offset)%128) throw std::invalid_argument("MoE router binding mismatch");
    check_range(b.buffer->size(),b.offset,bytes);
}
}
MoeBlock::MoeBlock(const Device& d,core::ModelGeometry g,uint32_t layer,ExpertCache& cache,Binding router,Binding shared,MoeOptions options) {
    if(options.hpu_load_budget!=UINT32_MAX && !options.cpu) throw std::invalid_argument("hybrid MoE requires CPU executor");
    validate_geometry(g,layer,cache);validate_binding(d,router,"w_router",checked_product(checked_product(g.n_embd,g.n_expert),2));
    validate_binding(d,shared,"w_shared_gate",checked_product(g.n_embd,2));
    auto p=std::make_unique<Impl>(d,g,layer,cache);p->options=options;p->router_weight=std::move(router);p->shared_gate_weight=std::move(shared);impl_=std::move(p);
}
MoeBlock::MoeBlock(const Device& d,WeightTable table,uint32_t layer,ExpertCache& cache,MoeOptions options) {
    if(options.hpu_load_budget!=UINT32_MAX && !options.cpu) throw std::invalid_argument("hybrid MoE requires CPU executor");
    const auto g=table.geometry();validate_geometry(g,layer,cache);
    const auto prefix="blk."+std::to_string(layer)+".",router=prefix+"ffn_gate_inp.weight",shared=prefix+"ffn_gate_inp_shexp.weight";
    if(table.info(router).gguf_shape!=std::vector<uint64_t>{uint64_t(g.n_embd),uint64_t(g.n_expert)}) throw std::invalid_argument("GGUF router shape mismatch");
    const auto& ss=table.info(shared).gguf_shape;
    if(ss!=std::vector<uint64_t>{uint64_t(g.n_embd)} && ss!=std::vector<uint64_t>{uint64_t(g.n_embd),1}) throw std::invalid_argument("GGUF shared gate shape mismatch");
    auto p=std::make_unique<Impl>(d,g,layer,cache);Stream copy(d);
    Bf16WeightUpload r(d,table.matrix(router),copy);p->owned_router=std::make_unique<DeviceBuffer>(r.finish());
    Bf16WeightUpload s(d,ss.size()==1?table.vector(shared):table.matrix(shared),copy);p->owned_shared_gate=std::make_unique<DeviceBuffer>(s.finish());
    p->options=options;p->router_weight={"w_router",p->owned_router.get()};p->shared_gate_weight={"w_shared_gate",p->owned_shared_gate.get()};
    if(router_merge_enabled()) {
        // STRATA_HPU_MERGED_GEMM: the shared gate row right after the router rows, so the merged layer graphs
        // run both as one GEMM on "w_router" [experts+1, h]; the two bindings address the same bytes as before.
        const uint64_t router_bytes=p->owned_router->size(),shared_bytes=p->owned_shared_gate->size();
        if(router_bytes%128) throw std::invalid_argument("merged router rows must be 128-byte aligned");
        auto merged=std::make_unique<DeviceBuffer>(d,router_bytes+shared_bytes);HostBuffer bounce(d,router_bytes+shared_bytes);
        download(copy,bounce,*p->owned_router,router_bytes);copy.synchronize();upload(copy,*merged,bounce,router_bytes);copy.synchronize();
        download(copy,bounce,*p->owned_shared_gate,shared_bytes);copy.synchronize();upload(copy,*merged,bounce,shared_bytes,router_bytes,0);copy.synchronize();
        p->owned_router=std::move(merged);p->owned_shared_gate.reset();
        p->router_weight={"w_router",p->owned_router.get()};p->shared_gate_weight={"w_shared_gate",p->owned_router.get(),router_bytes};
    }
    impl_=std::move(p);
}
MoeBlock::~MoeBlock()=default;
std::vector<float> MoeBlock::run(std::span<const float> x) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);const auto& g=p.geometry;const uint64_t h=g.n_embd;
    if(x.empty() || x.size()%h || x.size()/h>128) throw std::invalid_argument("MoE input must contain 1..128 complete rows");
    const uint32_t rows=uint32_t(x.size()/h),bucket=bucket_for(rows);
    if(bucket!=p.router_bucket) {
        const std::array<Binding,2> weights{p.router_weight,p.shared_gate_weight};
        p.router=std::make_unique<HostRecipe>(p.device,build_moe_router(p.device,g,bucket),weights,std::vector<std::string>{"logits","shared_gate"});p.router_bucket=bucket;
    }
    std::vector<float> padded(uint64_t(bucket)*h);std::copy(x.begin(),x.end(),padded.begin());p.router->set("x",padded);p.router->run();
    const auto routes=route_reference(p.router->output("logits").first(uint64_t(rows)*g.n_expert),rows,uint32_t(g.n_expert),uint32_t(g.n_expert_used));
    // Upstream counts routing here, per routed entry, before anything decides where the expert runs
    // (src/core/expert_source.cpp:3717-3719). Free when --expert-profile/--adapt-* were not asked for.
    p.cache->note_routing(p.layer,routes.ids.data(),int64_t(routes.ids.size()));
    // --dump-routing's producer: upstream surfaces the fused dispatch's ids and weights at exactly this
    // point (src/program/generate.cpp:1007-1023). Off unless the sink was wired, and it only reads `routes`.
    if(p.options.routing_sink) p.options.routing_sink(p.layer,routes);
    PlacementPolicy placement(p.options.hpu_load_budget);
    std::map<int32_t,ExpertLease> resident;
    auto locate=[&](int32_t expert) {
        auto lease=p.cache->resident(p.layer,expert);const auto choice=placement.choose(expert,bool(lease));
        if(lease) resident.emplace(expert,std::move(lease));
        return choice;
    };
    auto run_hpu=[&](int32_t expert,std::span<const float> gathered) {
        ExpertLease lease;
        if(const auto found=resident.find(expert);found!=resident.end()) lease=std::move(found->second);
        if(!lease) lease=p.cache->acquire(p.layer,expert);
        return p.executor(uint32_t(gathered.size()/h)).run(gathered,std::move(lease));
    };
    auto run_cpu=[&](int32_t expert,std::span<const float> gathered)->std::vector<float> {
#if STRATA_HPU_CPU_EXPERTS
            if(!p.options.cpu) throw std::logic_error("CPU fallback has no executor");
            std::vector<float> result;result.reserve(gathered.size());
            for(size_t begin=0;begin<gathered.size();) {
                const auto count=std::min<size_t>(gathered.size()-begin,8*h);auto out=p.options.cpu->run(p.layer,expert,gathered.subspan(begin,count));
                result.insert(result.end(),out.begin(),out.end());begin+=count;
            }
        return result;
#else
        (void)expert;(void)gathered;
            throw std::logic_error("CPU fallback is disabled in this HPU build");
#endif
    };
    // --fp8-experts decode (one row): the routed HPU experts and the shared expert go out as one launch.
    RoutedGroupExecutor run_group;
    if(p.cache->fp8() && rows==1) run_group=[&](std::span<const std::pair<int32_t,std::vector<float>>> calls) {
        std::vector<ExpertLease> leases;std::vector<const ExpertWeights*> weights;
        for(const auto& [expert,gathered]:calls) {
            // One decode row: every group, and the shared expert, received exactly that row.
            if(gathered.size()!=x.size() || !std::equal(gathered.begin(),gathered.end(),x.begin())) throw std::logic_error("FP8 expert group expects the one decode row");
            ExpertLease lease;
            if(const auto found=resident.find(expert);found!=resident.end()) lease=std::move(found->second);
            if(!lease) lease=p.cache->acquire(p.layer,expert);
            if(!lease.get().format.fp8) throw std::logic_error("FP8 expert group received a non-FP8 expert");
            weights.push_back(&lease.get());leases.push_back(std::move(lease));
        }
        const auto slots=uint32_t(g.n_expert_used)+1;
        if(!p.group) p.group=shared_fp8_group(p.device,g,slots);
        if(calls.size()>slots) throw std::logic_error("FP8 expert group exceeds its slots");
        return p.group->run(x,weights);
    };
    return dispatch_hybrid_moe(routes,x,uint32_t(h),p.router->output("shared_gate").first(rows),locate,run_hpu,run_cpu,run_group);
}
bool MoeBlock::fused_supported() const {
    const auto& p=*impl_;
    return p.cache->fp8() && p.options.hpu_load_budget==UINT32_MAX && !p.options.cpu;
}
std::array<Binding,2> MoeBlock::router_bindings() const {return {impl_->router_weight,impl_->shared_gate_weight};}
std::vector<float> MoeBlock::fused_ffn(std::span<const float> logits,float shared_gate,std::span<const float> x,
                                       std::span<const float> residual,std::span<const float> inject,const HcConfig& hc) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);const auto& g=p.geometry;
    if(!fused_supported() || logits.size()!=uint64_t(g.n_expert) || x.size()!=uint64_t(g.n_embd) ||
       !std::isfinite(shared_gate) || shared_gate<0 || shared_gate>1) throw std::invalid_argument("fused MoE input mismatch");
    const auto routes=route_reference(logits,1,uint32_t(g.n_expert),uint32_t(g.n_expert_used));
    p.cache->note_routing(p.layer,routes.ids.data(),int64_t(routes.ids.size()));
    if(p.options.routing_sink) p.options.routing_sink(p.layer,routes);
    // Slot order: the routed experts in router order, then the shared expert weighted by its gate.
    std::vector<ExpertLease> leases;std::vector<const ExpertWeights*> weights;std::vector<float> route;
    for(size_t i=0;i<routes.ids.size();++i) {
        if(routes.ids[i]<0) continue;
        leases.push_back(p.cache->acquire(p.layer,routes.ids[i]));route.push_back(routes.weights[i]);
    }
    leases.push_back(p.cache->acquire(p.layer,-1));route.push_back(shared_gate);
    for(const auto& lease:leases) {
        if(!lease.get().format.fp8) throw std::logic_error("fused MoE received a non-FP8 expert");
        weights.push_back(&lease.get());
    }
    if(!p.group_write) p.group_write=shared_fp8_group(p.device,g,uint32_t(g.n_expert_used)+1,hc);
    return p.group_write->run_write(x,route,residual,inject,weights);
}
void MoeBlock::fused_ffn_chain(std::span<const float> logits,float shared_gate,DecodeChain& chain,const HcConfig& hc) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);const auto& g=p.geometry;
    if(!fused_supported() || logits.size()!=uint64_t(g.n_expert) || !std::isfinite(shared_gate) || shared_gate<0 || shared_gate>1)
        throw std::invalid_argument("chained MoE input mismatch");
    const auto routes=route_reference(logits,1,uint32_t(g.n_expert),uint32_t(g.n_expert_used));
    p.cache->note_routing(p.layer,routes.ids.data(),int64_t(routes.ids.size()));
    if(p.options.routing_sink) p.options.routing_sink(p.layer,routes);
    // Same slot order as fused_ffn: routed experts in router order, then the shared expert.
    std::vector<ExpertLease> leases;std::vector<const ExpertWeights*> weights;std::vector<float> route;
    for(size_t i=0;i<routes.ids.size();++i) {
        if(routes.ids[i]<0) continue;
        leases.push_back(p.cache->acquire(p.layer,routes.ids[i]));route.push_back(routes.weights[i]);
    }
    leases.push_back(p.cache->acquire(p.layer,-1));route.push_back(shared_gate);
    for(const auto& lease:leases) {
        if(!lease.get().format.fp8) throw std::logic_error("chained MoE received a non-FP8 expert");
        weights.push_back(&lease.get());
    }
    if(!p.group_write) p.group_write=shared_fp8_group(p.device,g,uint32_t(g.n_expert_used)+1,hc);
    p.group_write->launch_chain(route,weights,chain);
    for(auto& lease:leases) chain.leases.push_back(std::move(lease)); // held until the launch has run
}
}
