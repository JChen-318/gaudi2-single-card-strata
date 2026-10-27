// Hardware-only complete trunk steering/state acceptance, sparse fixture.
#include "strata/hpu/model.hpp"
#include <cmath>
#include <iostream>
#include <filesystem>
using namespace strata::hpu;
namespace {
void require(bool ok,const char* why) {if(!ok) throw std::runtime_error(why);}
void check_model(const Model& model,int32_t token,bool enabled) {
    require(model.control_vector_enabled()==enabled,"restored control mode mismatch");
    const auto values=model.residual();require(values.size()==10240,"full control residual missing");
    for(size_t i=0;i<values.size();++i) {
        const auto column=i%2560;const float expected=float(token*(int(column%7)-3))/8+(enabled && column==0?1:0);
        require(std::abs(values[i]-expected)<1e-5,"control hook not after layer 47 FFN on every HC stream");
    }
    for(auto value:model.logits()) require(value==0,"zero output-head oracle changed");
}
}
int main(int argc,char** argv) {
    if(argc!=4 && argc!=5) return 2;
    try {
        ModelOptions options;options.memory.context=8;options.memory.expert_cache=16ull<<20;options.memory.hbm_limit=24ull<<30;
        struct MemoryReadings {std::vector<uint64_t> values;size_t next=0;};
        const auto memory=std::make_shared<MemoryReadings>();
        options.session_memory_probe=[memory] {return std::optional<uint64_t>(memory->next<memory->values.size()?memory->values[memory->next++]:UINT64_MAX);};
        options.control_vector.files={{argv[3],1}};options.control_vector.first=47;options.control_vector.last=47;
        if(argc==5) {options.mtp_runtime=argv[4];options.mtp_min_p=0;options.mtp.expert_cache=16ull<<20;}
        Model model(argv[1],options);(void)model.decode(1);check_model(model,1,true);const auto on=model.snapshot();
        model.set_control_vector(false);require(!model.size(),"arithmetic switch kept old target/MTP state");(void)model.decode(1);check_model(model,1,false);
        const auto off=model.snapshot();model.save(argv[2]);
        const auto off_path=std::string(argv[2])+".off";model.save(off_path);
        model.set_control_vector(true);(void)model.decode(2);check_model(model,2,true);
        model.restore(off);check_model(model,1,false);(void)model.decode(2);check_model(model,2,false);
        model.restore(on);check_model(model,1,true);(void)model.decode(3);check_model(model,3,true);
        model.load(argv[2]);check_model(model,1,false);require(model.tokens()==std::vector<int32_t>{1},"disk mode/history mismatch");
        model.set_control_vector(true);(void)model.decode(1);(void)model.decode(2);check_model(model,2,true);model.save(argv[2]);
        model.set_control_vector(false);model.load(argv[2]);check_model(model,2,true);
        // Refusal before allocation or after complete capture/parse must never
        // publish a new file or apply any target/MTP/control state.
        const auto refused_path=std::string(argv[2])+".refused";
        for(const auto& readings:{std::vector<uint64_t>{0},std::vector<uint64_t>{UINT64_MAX,0}}) {
            memory->values=readings;memory->next=0;bool refused=false;
            try {model.save(refused_path);} catch(const SessionMemoryRefused&) {refused=true;}
            require(refused && !std::filesystem::exists(refused_path),"RAM-refused save published a file");check_model(model,2,true);
            memory->values=readings;memory->next=0;refused=false;
            try {model.load(off_path);} catch(const SessionMemoryRefused&) {refused=true;}
            require(refused && model.healthy() && model.tokens()==std::vector<int32_t>({1,2}),"RAM-refused load changed native state");check_model(model,2,true);
        }
        memory->values.clear();memory->next=0;
        int polls=0;bool cancelled=false;try {(void)model.decode(3,[&]{return ++polls==(argc==5?52:51);});} catch(const Cancelled&) {cancelled=true;}
        require(cancelled && model.healthy() && model.size()==2,"post-steering cancellation did not rewind complete target/MTP state");check_model(model,2,true);
        GenerationOperations operations;operations.reset=[&]{model.reset();};operations.control_vector=[&](bool enabled){model.set_control_vector(enabled);};
        operations.decode=[&](int32_t token,const auto& stop){return model.decode(token,stop);};
        operations.checkpoint_bytes=model.snapshot_size_limit();operations.checkpoint_park_bytes=[&](uint64_t cells){return model.parked_bytes(cells);};
        operations.parked_retained_bytes=[&]{return model.parked_retained_bytes();};operations.discard_parked=[&]{model.discard_parked();};
        operations.available_host_bytes=[] {return std::optional<uint64_t>(UINT64_MAX);};
        operations.checkpoint=[&] {const auto saved=model.snapshot();return std::function<void()>([&,saved]{model.restore(saved);});};
        GenerationOptions generation_options;generation_options.cache_slots=2;generation_options.cache_bytes=options.memory.snapshot_budget;
        Generation generation(model.vocabulary(),8,model.stop_tokens(),operations,generation_options);generation.adopt(model.tokens(),model.logits(),model.control_vector_enabled());
        GenerationEvents events;events.token=[](int32_t id){require(id==0,"zero-head generation token");};GenerationRequest request;request.prompt={1,2};request.max_new=1;request.sampling.greedy=true;
        require(generation.run(request,events).reused==2,"adopted steered prefix was not retained");check_model(model,2,true);
        request.control_vector=false;require(generation.run(request,events).reused==0,"steered live prefix reused after request switched off");check_model(model,2,false);
        request.control_vector=true;require(generation.run(request,events).reused==2,"steered parked prefix lost on mode switch");check_model(model,2,true);
        request.control_vector=false;require(generation.run(request,events).reused==2,"unsteered parked prefix lost on mode switch");check_model(model,2,false);
        generation.suspend_live();model.set_control_vector(true);(void)model.decode(3);
        generation.attach_live();check_model(model,2,false);
        require(model.tokens()==std::vector<int32_t>({1,2}),"disk main attachment mounted a foreign target/MTP sequence");
        generation.release_checkpoints();require(!generation.cache_stats().entries,"mode bank release retained state");
        std::cout<<"PASS real HPU all 48 layers with post-FFN steering, request mode switch, parked on/off snapshot reuse, disk restore and cancellation"<<(argc==5?" with MTP":"")<<'\n';
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
