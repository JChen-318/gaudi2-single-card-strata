// Resident expert route_topk + GEMV instrument at the target model geometry.
// Default: timed device events around queued launches, no input/output DMA in the interval.
// --host-timing retains synchronous host/readback timing. Printed GB/s is an estimated
// logical weight-byte rate, not an HBM hardware counter (reused experts/cache may differ).
// --verify compares both GEMV variants bit for bit. Hardware diagnostic, not CTest.
#include "strata/hpu/host_recipe.hpp"
#include "strata/hpu/model_abi.hpp"
#include "strata/hpu/performance_modes.hpp"
#include "strata/hpu/resident_experts.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
using namespace strata::hpu;
namespace {
// GGUF metadata of the target model (qwen4exp.*): hidden 2560, expert ff 640, 512 experts, 10 routed per token
// plus the shared expert, which is exactly the per_row the resident MoE lays out.
constexpr uint32_t kHidden=2560,kFf=640,kExperts=512,kTopK=10,kPerRow=kTopK+1;
constexpr double kPeakGBs=2450.0; // Gaudi2 HBM peak, printed only as an efficiency column

struct Shape {uint32_t rows;uint64_t slots;};
struct Timing {double ms;double gbs;uint64_t bytes;int iters;double wall_ms;};

// Non-trivial, finite, non-denormal bytes: the codes vary per expert and the scales are small normals, so a
// bit-exactness comparison is meaningful and no MAC sees a denormal or an infinity.
void fill_weights(const Device& device,Stream& copy,DeviceBuffer& buffer,uint64_t per_expert,uint64_t codes_bytes) {
    HostBuffer host(device,per_expert);
    auto* raw=static_cast<uint8_t*>(host.data());
    for(uint64_t e=0;e<buffer.size()/per_expert;++e) {
        for(uint64_t i=0;i<codes_bytes;++i) raw[i]=uint8_t((i*37+e*11)&0xFF);
        for(uint64_t i=codes_bytes;i+4<=per_expert;i+=4) {
            const float s=float(int((i/4+e)%13)+1)/4096.0f;
            std::memcpy(raw+i,&s,4);
        }
        upload(copy,buffer,host,per_expert,e*per_expert);
    }
    copy.synchronize();
}

std::unique_ptr<HostRecipe> build(const Device& device,bool gate_up,const Shape& shape,const char* guid,
                                  std::span<const Binding> weights,bool readback=true,bool group=false) {
    const uint64_t rows=shape.rows,slots=shape.slots;
    const uint64_t columns=gate_up?kHidden:kFf,out_rows=gate_up?2u*kFf:kHidden;
    const uint64_t per_expert=iq4_resident_bytes(out_rows,columns);
    const int parts=gate_up?gemv_parts_env("STRATA_HPU_GEMV_PARTS_GU",2):gemv_parts_env("STRATA_HPU_GEMV_PARTS_DOWN",1);
    GraphBuilder g(device);
    const auto logits=g.tensor({"logits",DType::F32,{rows,uint64_t(kExperts)}});
    const auto shared=g.tensor({"shared_gate",DType::F32,{rows,1}});
    const auto ids=g.tensor({"ids",DType::I32,{slots}}),route=g.tensor({"route_weights",DType::F32,{slots}});
    {abi::RouteTopk a{int32_t(kExperts),int32_t(kTopK),int32_t(kPerRow),int32_t(rows)};
     const TensorId in[]={logits,shared},out[]={ids,route};g.op("strata_route_topk_f32",in,out,&a,sizeof(a));}
    // STRATA_HPU_MOE_GROUP: the same node over the grouped slot order, so the timed interval is comparable.
    TensorId order{};
    if(group) {
        const auto starts=g.tensor({"group_starts",DType::F32,{uint64_t(kExperts)+1}});
        order=g.tensor({"group_order",DType::F32,{slots}});
        abi::RouteGroup rg{int32_t(kExperts),int32_t(slots)};
        {const TensorId in[]={ids},out[]={starts};g.op("strata_route_group_starts_f32",in,out,&rg,sizeof(rg));}
        {const TensorId in[]={ids,starts},out[]={order};g.op("strata_route_group_order_f32",in,out,&rg,sizeof(rg));}
    }
    // gate/up reads one x row per token (x_group = per_row); down reads one activation row per slot.
    const uint64_t count=gate_up?rows:slots;
    const auto codes=g.tensor({"xcodes",DType::U8,{count,columns/16,256}});
    const auto scales=g.tensor({"xdx",DType::F32,{count,columns/32,64}});
    const auto out=g.tensor({"parts",DType::F32,{uint64_t(parts),slots,out_rows}});
    const auto w=g.tensor({gate_up?"w_gu":"w_down",DType::U8,{uint64_t(kExperts),per_expert},true});
    const auto shared_w=g.tensor({gate_up?"s_gu":"s_down",DType::U8,{1,per_expert},true});
    const auto table=g.tensor({"table",DType::U8,{256},true});
    abi::Iq4Gemv a{int32_t(columns),int32_t(out_rows),int32_t(gate_up?kPerRow:1),int32_t(parts),int32_t(slots),int32_t(kPerRow)};
    const TensorId outs[]={out};
    if(group) {const TensorId in[]={codes,scales,table,ids,w,shared_w,order};g.op(guid,in,outs,&a,sizeof(a));}
    else {const TensorId in[]={codes,scales,table,ids,w,shared_w};g.op(guid,in,outs,&a,sizeof(a));}
    return std::make_unique<HostRecipe>(device,g.compile(std::string(guid)+"_"+std::to_string(rows)),
                                       weights,readback?std::vector<std::string>{"parts"}:std::vector<std::string>{});
}

struct Inputs {
    std::vector<float> logits,gate,scales;
    std::vector<uint8_t> codes;
};

Inputs make_inputs(bool gate_up,const Shape& shape) {
    Inputs in;
    in.logits.resize(size_t(shape.rows)*kExperts);
    for(size_t i=0;i<in.logits.size();++i) in.logits[i]=float(int((i*19+7)%257)-128)/1024.0f;
    in.gate.assign(shape.rows,0.25f);
    const uint64_t columns=gate_up?kHidden:kFf,count=gate_up?shape.rows:shape.slots;
    in.codes.resize(size_t(count*columns/16*256));
    for(size_t i=0;i<in.codes.size();++i) in.codes[i]=uint8_t((i*29+3)&0xFF);
    in.scales.resize(size_t(count*columns/32*64));
    for(size_t i=0;i<in.scales.size();++i) in.scales[i]=float(int(i%11)+1)/512.0f;
    return in;
}

void feed(HostRecipe& recipe,const Inputs& in) {
    recipe.set("logits",in.logits);recipe.set("shared_gate",in.gate);
    recipe.set_bytes("xcodes",in.codes);recipe.set("xdx",in.scales);
}

Timing measure_host(HostRecipe& recipe,const Inputs& in,uint64_t bytes) {
    feed(recipe,in);
    recipe.run(); // graph compile + first touch, discarded
    using clock=std::chrono::steady_clock;
    const auto start=clock::now();
    int iters=0;
    do {
        recipe.run();
        ++iters;
    } while(iters<4000 && std::chrono::duration<double>(clock::now()-start).count()<0.25);
    const double seconds=std::chrono::duration<double>(clock::now()-start).count();
    const double ms=seconds*1000.0/iters;
    return Timing{ms,double(bytes)/1e9/(seconds/iters),bytes,iters,ms};
}
struct TimedEvent {
    synEventHandle handle=nullptr;
    explicit TimedEvent(const Device& device) {check(synEventCreate(&handle,device.info().id,EVENT_COLLECT_TIME),"timing event create");}
    ~TimedEvent() {if(handle) (void)synEventDestroy(handle);}
};
Timing measure_queued(const Device& device,HostRecipe& recipe,const Inputs& in,uint64_t bytes) {
    feed(recipe,in);recipe.run(); // initialize once; this recipe has no host outputs
    Stream stream(device);TimedEvent start(device),end(device);
    constexpr int iterations=256;std::vector<std::pair<double,double>> samples;
    for(int pass=0;pass<5;++pass) {
        const auto host_start=std::chrono::steady_clock::now();
        check(synEventRecord(start.handle,stream.handle()),"timing start record");
        for(int i=0;i<iterations;++i) recipe.launch(stream);
        check(synEventRecord(end.handle,stream.handle()),"timing end record");
        check(synEventSynchronize(end.handle),"timing end wait");recipe.wait();
        uint64_t ns=0;check(synEventElapsedTime(&ns,start.handle,end.handle),"timing event elapsed");
        const auto wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-host_start).count()/iterations;
        if(!ns) throw std::runtime_error("zero device interval");
        samples.emplace_back(double(ns)/1e6/iterations,wall);
    }
    std::sort(samples.begin(),samples.end());const auto [ms,wall]=samples[samples.size()/2];
    return Timing{ms,double(bytes)/1e6/ms,bytes,iterations,wall};
}

// Same node, two kernels, same inputs and weights: the outputs must match to the bit.
bool verify(const Device& device,bool gate_up,const Shape& shape,std::span<const Binding> weights,bool group,bool pf) {
    const Inputs in=make_inputs(gate_up,shape);
    auto reference=build(device,gate_up,shape,"strata_iq4q8_gemv_packed_f32",weights);
    auto candidate=group?build(device,gate_up,shape,"strata_iq4q8_gemv_group_f32",weights,true,true)
                 :pf?build(device,gate_up,shape,"strata_iq4q8_gemv_packed_pf_f32",weights)
                 :build(device,gate_up,shape,"strata_iq4q8_gemv_tile_f32",weights);
    feed(*reference,in);feed(*candidate,in);
    reference->run();candidate->run();
    const auto a=reference->values("parts"),b=candidate->values("parts");
    if(a.size()!=b.size()) {std::cout<<"   VERIFY size mismatch\n";return false;}
    if(std::memcmp(a.data(),b.data(),a.size()*sizeof(float))!=0) {
        size_t first=0;
        while(first<a.size() && std::memcmp(&a[first],&b[first],sizeof(float))==0) ++first;
        std::cout<<"   VERIFY MISMATCH at "<<first<<" of "<<a.size()
                 <<" : "<<a[first]<<" vs "<<b[first]<<'\n';
        return false;
    }
    return true;
}

void report(const char* label,const Shape& shape,const Timing& t) {
    std::cout<<"rows="<<shape.rows<<" slots="<<shape.slots<<" "<<label<<" : "
             <<t.ms<<" ms  "<<t.gbs<<" GB/s  "<<100.0*t.gbs/kPeakGBs<<"% of peak  "
             <<(double(t.bytes)/1e6)<<" MB/launch  iters="<<t.iters<<" wall_ms="<<t.wall_ms<<'\n';
}
}

int main(int argc,char** argv) {
    try {
        std::vector<uint32_t> rows{1,2,4,9,16,32};
        bool do_verify=false,host_timing=false,group=false,pf=false;
        for(int i=1;i<argc;++i) {
            const std::string arg(argv[i]);
            if(arg=="--verify") do_verify=true;
            else if(arg=="--group") group=true;
            else if(arg=="--pf") pf=true;
            else if(arg=="--host-timing") host_timing=true;
            else if(arg=="--rows") {
                rows.clear();
                for(int j=i+1;j<argc && argv[j][0]!='-';++j) rows.push_back(uint32_t(std::stoul(argv[j])));
                break;
            } else {std::cerr<<"usage: gemv-throughput-test [--verify] [--group] [--pf] [--rows N...]\n";return 2;}
        }
        Device device;Stream copy(device);
        const uint64_t gu_bytes=iq4_resident_bytes(2u*kFf,kHidden),down_bytes=iq4_resident_bytes(kHidden,kFf);
        DeviceBuffer w_gu(device,uint64_t(kExperts)*gu_bytes),w_down(device,uint64_t(kExperts)*down_bytes);
        DeviceBuffer s_gu(device,gu_bytes),s_down(device,down_bytes),table(device,256);
        std::cout<<"weights: w_gu="<<w_gu.size()<<" w_down="<<w_down.size()<<" s="<<s_gu.size()+s_down.size()<<" bytes\n";
        fill_weights(device,copy,w_gu,gu_bytes,uint64_t(2u*kFf)*kHidden/2);
        fill_weights(device,copy,w_down,down_bytes,uint64_t(kHidden)*kFf/2);
        fill_weights(device,copy,s_gu,gu_bytes,uint64_t(2u*kFf)*kHidden/2);
        fill_weights(device,copy,s_down,down_bytes,uint64_t(kHidden)*kFf/2);
        { // the codebook table is a real 256-byte input in the model, so use the real one.
            const auto values=iq4_codebook_table();HostBuffer host(device,256);
            std::memcpy(host.data(),values.data(),256);upload(copy,table,host,256);copy.synchronize();
        }
        const Binding gu_bindings[]={{"w_gu",&w_gu},{"s_gu",&s_gu},{"table",&table}};
        const Binding down_bindings[]={{"w_down",&w_down},{"s_down",&s_down},{"table",&table}};
        if(do_verify) {
            bool ok=true;
            for(const uint32_t r:rows) {
                const Shape shape{r,uint64_t(r)*kPerRow};
                const bool gu=verify(device,true,shape,gu_bindings,group,pf),down=verify(device,false,shape,down_bindings,group,pf);
                std::cout<<"rows="<<shape.rows<<" gu bit_exact="<<gu<<" down bit_exact="<<down<<'\n';
                ok=ok&&gu&&down;
            }
            std::cout<<(ok?"PASS":"FAIL")<<" gemv_throughput verify\n";
            return ok?0:1;
        }
        const char* guid=group?"strata_iq4q8_gemv_group_f32":pf?"strata_iq4q8_gemv_packed_pf_f32":
                          gemv_tile_enabled()?"strata_iq4q8_gemv_tile_f32":"strata_iq4q8_gemv_packed_f32";
        std::cout<<"kernel "<<guid<<" timing="<<(host_timing?"host_with_output_DMA":"queued_device_interval_no_DMA")
                 <<"; graph includes route_topk and GEMV; estimated weight bytes, not measured HBM counters\n";
        for(const uint32_t r:rows) {
            const Shape shape{r,uint64_t(r)*kPerRow};
            for(const bool gate_up:{true,false}) {
                const uint64_t columns=gate_up?kHidden:kFf,out_rows=gate_up?2u*kFf:kHidden;
                const uint64_t per_expert=iq4_resident_bytes(out_rows,columns);
                const Inputs in=make_inputs(gate_up,shape);
                const auto recipe=gate_up?build(device,true,shape,guid,gu_bindings,host_timing,group)
                                        :build(device,false,shape,guid,down_bindings,host_timing,group);
                report(gate_up?"gu  ":"down",shape,host_timing?measure_host(*recipe,in,shape.slots*per_expert):
                       measure_queued(device,*recipe,in,shape.slots*per_expert));
            }
        }
        std::cout<<"PASS gemv_throughput\n";
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
