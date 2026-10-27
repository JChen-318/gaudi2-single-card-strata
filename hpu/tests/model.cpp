// Full target geometry, every native layer, no real model download required.
// Hardware test only: sparse zero block weights, distinct embedding rows.
#include "strata/hpu/model.hpp"
#include "strata/hpu/slot_session.hpp"
#include "strata/hpu/generation.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <cstring>

using namespace strata::hpu;
namespace {
void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
template<class F> void rejects(F&& f) {bool caught=false;try {f();} catch(const std::exception&) {caught=true;}require(caught,"invalid model operation accepted");}
void zero(std::span<const float> logits) {
    require(logits.size()==248320,"full vocabulary logits missing");for(float value:logits) require(value==0,"zero-weight head differs from independent zero oracle");
}
void residual(const Model& model,int32_t token) {
    const auto values=model.residual();require(values.size()==10240,"complete HC residual missing");
    for(size_t i=0;i<values.size();++i) require(values[i]==float(token*(int(i%2560%7)-3))/8,"published residual does not match committed embedding token");
}
void routing(const Model& model,uint32_t rows) {
    const auto records=model.routing_records();require(records.size()==48*rows,"native routing capture lost physical rows or included replay");
    for(size_t at=0;at<records.size();++at) {
        const auto& record=records[at];require(record.layer==int32_t(at/rows) && !record.ids.empty() && record.ids.size()==record.weights.size(),"native routing capture layer/token order");
        for(size_t k=0;k<record.ids.size();++k) require(record.ids[k]==int32_t(k) && std::abs(record.weights[k]-1.0f/float(record.ids.size()))<1e-6f,"zero-router top-k IDs/true weights changed");
    }
}
// Independent fixed-upstream layout oracle. The zero-weight fixture has zero
// block/inject/QSA arithmetic, but packed IDs/steps must retain raw int bits.
void halves(const Model& model) {
    constexpr size_t stride=13840,a=11272,v=512,ids_at=a+2*v+v/2;
    const auto frame=model.half_records();const auto cells=int32_t(model.size());
    require(cells>0 && frame.size()==48*stride,"native scalar half frame lost layers/extent");
    for(size_t layer=0;layer<48;++layer) {
        const auto* row=frame.data()+layer*stride;const bool qsa=layer%4==3;
        for(size_t at=0;at<stride;++at) if(!qsa || at<ids_at || at>=ids_at+8)
            require(row[at]==0,"zero-weight native half field or declared padding differs");
        if(qsa) {
            int32_t ids[4],step[4];std::memcpy(ids,row+ids_at,sizeof(ids));std::memcpy(step,row+ids_at+4,sizeof(step));
            for(int32_t i=0;i<4;++i) require(ids[i]==(i<cells?i:-1),"native half IDs were sanitized/reordered");
            require(step[0]==cells-1 && step[1]==cells && step[2]==cells/4 && step[3]==cells,"native QSA half steps differ from state");
        }
    }
}
}
int main(int argc,char** argv) {
    if(argc!=3 && argc!=4 && argc!=5) return 2;
    try {
        ModelOptions options;options.memory.context=8;options.memory.expert_cache=16ull<<20;options.memory.hbm_limit=24ull<<30;
        options.memory.prefill_rows=2;
        options.memory.verify_rows=7;
        options.memory.batch_slots=2;
        options.capture_routing=true;
        options.capture_halves=true;
        options.capture_layer_ladder=true;
        if(const auto* external=std::getenv("STRATA_HPU_TEST_PLE_GGUF")) options.ple_gguf=external;
        if(const auto* pack=std::getenv("STRATA_HPU_TEST_EXPERT_PACK")) options.expert_pack=pack;
        if(argc>=4) {options.mtp_runtime=argv[3];options.mtp_min_p=0;options.mtp.expert_cache=16ull<<20;}
        if(argc==5) {const std::string mode=argv[4];require(mode=="all" || mode=="proj" || mode=="head","invalid hardware Q4 fixture mode");options.mtp.q4_projections=mode!="head";options.mtp.q4_head=mode!="proj";}
        std::vector<DraftRow> mtp_reference;
        Model model(argv[1],options);require(model.healthy() && !model.size(),"initial model state");
        // Build-only until the user permits card acceptance: the SAME slot
        // production component that the CLI binds, over a physical Model.
        {
            SlotSession* native_slots=nullptr;
            GenerationOperations shared_op;shared_op.reset=[&]{model.reset();};
            shared_op.decode=[&](int32_t id,const auto& cancel){return model.decode(id,cancel);};
            shared_op.prefill=[&](std::span<const int32_t> ids,const PromptCommit& commit,const auto& cancel){model.prefill_window(ids,commit,cancel);};
            shared_op.prefill_routing=[&](std::span<const int32_t> ids,const PromptCommit& commit,const auto& cancel,const RoutingWindowSink& sink){model.prefill_window(ids,commit,cancel,sink);};
            shared_op.routing=[&]{return model.routing_records();};
            shared_op.checkpoint=[&]{auto point=model.snapshot();return std::function<void()>([&,point=std::move(point)]{model.restore(point);});};
            shared_op.checkpoint_bytes=model.snapshot_size_limit();shared_op.checkpoint_park_bytes=[&](uint64_t cells){return model.parked_bytes(cells);};
            shared_op.parked_retained_bytes=[&]{return model.parked_retained_bytes();};shared_op.discard_parked=[&]{model.discard_parked();};
            shared_op.available_host_bytes=[] {return std::optional<uint64_t>(UINT64_MAX);};
            shared_op.select_prefix=[&](std::span<const int32_t> ids,uint64_t maximum,bool control,std::span<const RequestImage> images)->std::optional<GenerationPrefix>{return native_slots?native_slots->select_prefix(ids,maximum,control,images):std::nullopt;};
            GenerationOptions shared_options;shared_options.max_draft=0;shared_options.cache_slots=1;shared_options.cache_bytes=options.memory.snapshot_budget;
            shared_options.prefill_chunk=2;shared_options.short_read=0;shared_options.cache_every=0;shared_options.boundaries.turn_token=-1;
            shared_options.dumps.routing=std::string(argv[2])+".resident-routing";
            shared_options.dump_shape.n_layers=48;shared_options.dump_shape.n_expert=512;shared_options.dump_producers.routing=true;
            shared_options.resident_routing=std::make_shared<DumpSinks>(shared_options.dumps,shared_options.dump_shape,shared_options.dump_producers);
            uint64_t scalar_routes=0,window_routes=0;
            const RoutingWindowSink trace=[&](const RoutingWindowView& view,bool window){shared_options.resident_routing->routing_view(view,window);(window?window_routes:scalar_routes)+=uint64_t(view.layers)*view.rows;};
            Generation main_session(model.vocabulary(),8,model.stop_tokens(),std::move(shared_op),shared_options);
            SlotSessionOperations slot_op;
            slot_op.before_foreign=[&]{main_session.suspend_live();};slot_op.logits=[&]{return model.logits();};
            slot_op.select_prefix=[&](std::span<const int32_t> ids,uint64_t maximum,bool control,std::span<const RequestImage> images){return main_session.select_prefix(ids,maximum,control,images);};
            slot_op.import_prefixes=[&](const GenerationRead& read){main_session.import_prefixes(read);};
            slot_op.reset=[&]{model.reset();};slot_op.prefill=[&](std::span<const int32_t> ids){return model.prefill(ids);};
            slot_op.prefill_window=[&](std::span<const int32_t> ids){std::vector<float> out;model.prefill_window(ids,[&](uint32_t,std::span<const float> values){out.assign(values.begin(),values.end());});return out;};
            slot_op.prefill_cancellable=[&](std::span<const int32_t> ids,bool window,const PromptCommit& commit,const std::function<bool()>& stop) {model.prefill_window(ids,commit,stop,trace,!window);};
            slot_op.decode=[&](int32_t id){return model.decode(id,{},trace);};
            slot_op.checkpoint=[&]{auto point=model.snapshot();return std::function<void()>([&,point=std::move(point)]{model.restore(point);});};
            slot_op.discard_retained=[&]{model.discard_parked();};
            if(argc>=4) slot_op.speculate=[&](int32_t pending,std::span<const int32_t> history,const strata::kernels::SamplerParams& sampling){return model.slot_window(pending,history,sampling,trace);};
            SlotCheckpointOptions checks;checks.boundaries.turn_token=2;checks.boundaries.root_min_tokens=0;
            checks.interval=0;checks.prefill_chunk=2;checks.short_read=0;
            SlotSession slots(2,model.vocabulary(),8,model.stop_tokens(),std::move(slot_op),true,checks);
            native_slots=&slots;
            GenerationRequest request;request.prompt={1,2};request.sampling.greedy=true;int32_t first=-1;
            require(slots.admit(0,request,4,[&](int32_t id){first=id;}).continues && first==0,"native slot admission oracle");
            require(slots.step(0).token==0,"native slot step oracle");slots.finish(0);
            request.prompt={1,2,0,3};const auto resumed=slots.admit(1,request,4,[](int32_t){});
            require(resumed.reused_tokens==3 && model.tokens()==request.prompt,"native idle slot prefix not restored");residual(model,3);
            slots.finish(1);request.prompt={1,2,5};
            const auto rewound=slots.admit(1,request,4,[](int32_t){});
            require(rewound.reused_tokens==1 && model.tokens()==request.prompt && slots.checkpoint_captures(),"native slot prompt checkpoint did not restore a rewritten suffix");residual(model,5);
            if(argc>=4) {
                const auto before=model.tokens();const auto head=model.mixed();
                rejects([&]{(void)model.slot_window(0,before,request.sampling,[](const RoutingWindowView& view,bool window){require(window && view.layers==48 && view.rows==2,"slot routing callback omitted its physical initial window");throw std::runtime_error("synthetic resident trace IO failure");});});
                require(model.healthy() && model.tokens()==before && model.mixed()==head && model.routing_records().empty(),"slot routing selector failure did not restore numeric root/clear trace");
                // Native [pending,proposal] target pass, including private MTP
                // pairing/rollback and end-image restore. Build-only evidence
                // until whole hardware acceptance is explicitly authorized.
                const auto window=slots.step_window(1,2);
                require(window.count==2 && window.rows[0].token==0 && window.rows[1].token==0,"native per-slot MTP accept/bonus oracle");
                auto fed=request.prompt;fed.push_back(0);fed.push_back(0);
                require(model.tokens()==fed && model.healthy(),"native per-slot MTP committed a different prefix");
                const auto mixed=model.mixed();
                const auto again=slots.step_window(1,1);
                require(again.count==1 && again.rows[0].token==0,"native one-output slot fallback");
                require(!mixed.empty(),"slot window omitted mixed publication");
            }
            GenerationRequest partial;partial.prompt={5,6,7,8,9,10,11};partial.sampling.greedy=true;
            BatchSlotSession::ReadControl read;bool emitted=false;
            read.yield=[&]()->std::optional<int> {if(model.size()==2) return 0;return {};};
            const auto yielded=slots.admit_cooperative(0,partial,2,[&](int32_t){emitted=true;},read);
            require(!emitted && yielded.yielded_slot==0 && yielded.yielded_tokens==2 && model.tokens()==std::vector<int32_t>({5,6}),"native BYIELD did not own exactly its partial prefix");
            const auto continued=slots.admit(0,partial,2,[](int32_t id){require(id==0,"native resumed prompt sample");});
            require(continued.reused_tokens==2 && model.tokens()==partial.prompt && model.healthy(),"native partial target/MTP snapshot resume");
            GenerationRequest solo;solo.prompt={20,21};solo.max_new=2;solo.sampling.greedy=true;GenerationEvents events;events.token=[](int32_t id){require(id==0,"native shared GEN sample");};
            require(main_session.run(solo,events).generated==2 && model.tokens()==std::vector<int32_t>({20,21,0}),"native main GEN after foreign slots");
            require(slots.step_window(1,1).rows[0].token==0,"native slot after main GEN");
            solo.prompt={20,21,0,22};solo.max_new=1;
            require(main_session.run(solo,events).reused==3 && model.tokens()==solo.prompt,"native suspended main restore after a foreign slot");
            slots.release(0);solo.prompt={30,31,32,33,34,35,36};solo.max_new=1;
            events.yield=[&](const GenerationRead& read){return read.tokens.size()==2 && slots.park_partial(0,read)?0:-1;};
            const auto solo_yield=main_session.run(solo,events);
            require(solo_yield.generated==0 && solo_yield.yielded_slot==0 && solo_yield.yielded_tokens==2 && model.tokens()==std::vector<int32_t>({30,31}),"native solo BYIELD target/MTP endpoint");
            require(slots.admit(0,solo,2,[](int32_t id){require(id==0,"native handed-off BGEN sample");}).reused_tokens==2 && model.tokens()==solo.prompt,"native solo to BGEN target/MTP snapshot resume");
            slots.release(0);GenerationRequest stopped;stopped.prompt={12,13,14,15,16,17,18};stopped.sampling.greedy=true;
            BatchSlotSession::ReadControl stopping;bool stop=false;emitted=false;
            stopping.cancelled=[&]{return stop;};stopping.progress=[&](uint32_t n,uint32_t,double){if(n==2) stop=true;};
            const auto cancelled=slots.admit_cooperative(0,stopped,2,[&](int32_t){emitted=true;},stopping);
            require(cancelled.cancelled && !emitted && cancelled.prompt_read==2 && model.tokens()==std::vector<int32_t>({12,13}) && model.healthy(),"native stopped read did not keep the committed target/MTP prefix");
            const auto resumed_stop=slots.admit(0,stopped,2,[](int32_t id){require(id==0,"native cancelled prompt resume sample");});
            require(resumed_stop.reused_tokens==2 && model.tokens()==stopped.prompt && model.healthy(),"native cancelled read target/MTP image did not resume");
            slots.release(0);slots.release(1);model.discard_parked();
            model.reset();const std::array<int32_t,2> scalar_ids{1,2};uint32_t scalar_committed=0,scalar_seen=0;
            model.prefill_window(scalar_ids,[&](uint32_t done,std::span<const float> values){zero(values);scalar_committed=done;},{},[&](const RoutingWindowView& view,bool window){require(!window && view.rows==1 && view.layers==48 && scalar_committed==scalar_seen+1,"forced scalar prefill routing did not follow each committed token");++scalar_seen;},true);
            require(scalar_committed==2 && scalar_seen==2 && model.tokens()==std::vector<int32_t>({1,2}),"forced scalar prefill used configured grouped rows or lost committed prefix");
            halves(model);
            model.reset();bool scalar_failed=false;
            try {(void)model.decode(3,{},[](const RoutingWindowView&,bool){throw std::runtime_error("post-commit scalar observer IO failure");});} catch(const std::runtime_error&) {scalar_failed=true;}
            require(scalar_failed && model.healthy() && model.tokens()==std::vector<int32_t>{3} && model.routing_records().empty(),"scalar observer failure lost committed numeric token or retained failed trace");residual(model,3);
            require(model.half_records().empty(),"scalar post-commit observer retained failed half frame");
            zero(model.logits());require(!model.mixed().empty(),"scalar observer failure hid committed head");
            for(bool scalar:{false,true}) for(bool commit_failure:{false,true}) {
                model.reset();uint32_t notified=0,observed=0;
                try {
                    model.prefill_window(scalar_ids,[&](uint32_t done,std::span<const float> values){zero(values);notified=done;if(commit_failure) throw std::runtime_error("post-commit prefix notifier failure");},{},[&](const RoutingWindowView& view,bool window){require(window!=scalar && view.rows==(scalar?1u:2u),"observer-failure fixture used wrong physical prefill path");++observed;throw std::runtime_error("post-commit prefill trace IO failure");},scalar);
                    require(false,"prefill observer failure was ignored");
                } catch(const std::runtime_error&) {}
                const auto expected=scalar?std::vector<int32_t>{1}:std::vector<int32_t>{1,2};
                require(model.healthy() && model.tokens()==expected && notified==expected.size() && observed==(commit_failure?0u:1u) && model.routing_records().empty(),"prefill observer failure changed committed prefix/notification or retained routing");residual(model,expected.back());zero(model.logits());
                require(model.half_records().empty(),"prefill post-commit observer retained half frame");
            }
            model.reset();zero(model.decode(4));routing(model,1);residual(model,4);
            require(scalar_routes && window_routes && !shared_options.resident_routing->closed(),"native resident slot routing omitted scalar/window producer or closed per request");
            require(shared_options.resident_routing->close(),"native resident routing close failed");
            require(std::filesystem::file_size(shared_options.dumps.routing)==shared_options.resident_routing->routing_records()*88,"native resident routing lost/duplicated file records");
            std::filesystem::remove(shared_options.dumps.routing);
        }
        model.reset(); // Release the slot phase before long-lived singleton snapshots.
        require(model.half_records().empty(),"reset retained scalar half frame");
        {
            GenerationOperations half_op;half_op.reset=[&]{model.reset();};
            half_op.decode=[&](int32_t token,const auto& stop){return model.decode(token,stop);};
            half_op.halves=[&]{return model.half_records();};
            GenerationOptions half_options;half_options.max_draft=0;half_options.dumps.halves=std::string(argv[2])+".halves";
            half_options.dump_producers.half_capture=true;auto& shape=half_options.dump_shape;
            shape.n_vocab=248320;shape.n_layers=48;shape.n_embd=2560;shape.hc=4;shape.n_head=24;shape.head_dim=256;shape.n_head_kv=2;
            Generation half_gen(model.vocabulary(),8,model.stop_tokens(),std::move(half_op),half_options);
            GenerationRequest half_request;half_request.prompt={1,2};half_request.max_new=2;half_request.sampling.greedy=true;
            uint32_t half_emitted=0;GenerationEvents half_events;half_events.token=[&](int32_t token){require(token==0,"native half generation sample differs");++half_emitted;};
            const auto half_result=half_gen.run(half_request,half_events);halves(model);
            require(half_emitted==2,"native half fixture omitted its required token sink");
            require(half_result.dumped_half_records==3 && model.tokens()==std::vector<int32_t>({1,2,0}),"native Generation half count/state differs");
            const auto expected=model.half_records();std::vector<float> last(expected.size());
            std::ifstream file(half_options.dumps.halves,std::ios::binary);file.seekg(std::streamoff(2*expected.size()*4));file.read(reinterpret_cast<char*>(last.data()),std::streamsize(last.size()*4));
            require(bool(file) && std::memcmp(last.data(),expected.data(),expected.size()*4)==0 && std::filesystem::file_size(half_options.dumps.halves)==3*expected.size()*4,"native Generation half raw bytes/order differ");
            file.close();std::filesystem::remove(half_options.dumps.halves);
            uint32_t grouped_calls=0;GenerationOperations grouped_op;grouped_op.reset=[&]{model.reset();};
            grouped_op.decode=[&](int32_t token,const auto& stop){return model.decode(token,stop);};
            grouped_op.prefill=[&](std::span<const int32_t> ids,const PromptCommit& commit,const auto& stop){++grouped_calls;model.prefill_window(ids,commit,stop);};
            grouped_op.halves=[&]{return model.half_records();};grouped_op.layer_ladder=[&]{return model.layer_ladder();};grouped_op.residual=[&]{return model.residual();};
            half_options.prefill_chunk=2;half_options.short_read=0;
            half_options.dumps.layers=std::string(argv[2])+".prefill-layers";half_options.dumps.final_r=std::string(argv[2])+".prefill-final-r";
            half_options.dump_producers.layer_ladder=true;half_options.dump_producers.final_r=true;
            Generation grouped_gen(model.vocabulary(),8,model.stop_tokens(),std::move(grouped_op),half_options);half_request.prompt={1,2,3,4};half_emitted=0;
            const auto grouped_result=grouped_gen.run(half_request,half_events);halves(model);
            require(grouped_calls==2 && half_emitted==2 && grouped_result.dumped_half_records==2 && grouped_result.dumped_final_r_records==2 && model.tokens()==std::vector<int32_t>({1,2,3,4,0}),"native raw dumps bypassed grouped prefill or included grouped records");
            require(std::filesystem::file_size(half_options.dumps.halves)==2*48*13840*4 && std::filesystem::file_size(half_options.dumps.layers)==2*49*10240*4,"native prefill suffix dump extent differs");
            std::ifstream tagged(half_options.dumps.final_r,std::ios::binary);
            for(int64_t row=0;row<2;++row) {int64_t header[2]{};tagged.read(reinterpret_cast<char*>(header),sizeof(header));require(bool(tagged) && header[0]==row+3 && header[1]==(row?0:4),"native final-r suffix position/token differs");tagged.seekg(10240*4,std::ios::cur);}
            tagged.close();for(const auto& path:{half_options.dumps.halves,half_options.dumps.layers,half_options.dumps.final_r}) std::filesystem::remove(path);
        }
        model.reset();
        const auto root=model.snapshot();require(model.snapshot_allocations().used==model.parked_bytes(0),"empty model snapshot charged an arena instead of its envelope");
        zero(model.decode(1));residual(model,1);routing(model,1);require(model.tokens()==std::vector<int32_t>{1},"first full-model token");
        halves(model);
        const auto prior_charge=model.snapshot_allocations().used;
        const auto first=model.snapshot();require(model.snapshot_allocations().used==prior_charge+model.parked_bytes(1),"short model snapshot and admission envelope differ");
        const auto first_mixed=model.mixed();require(first_mixed.size()==2560,"committed head input missing");
        if(argc>=4) {
            auto one=model.snapshot();model.restore(one);one={};
            const auto again=model.snapshot(); // exclusive one-token image, zero private MTP rows
            require(bool(again) && model.tokens()==std::vector<int32_t>{1},"one-token MTP snapshot could not be re-parked");
            model.restore(again);
        }
        zero(model.decode(2));residual(model,2);model.restore(first);residual(model,1);
        require(model.half_records().empty(),"snapshot restore retained another pass's half frame");
        require(model.mixed()==first_mixed,"snapshot restored a different conversation's mixed head input");zero(model.decode(3));
        if(argc>=4) {
            const auto tokens=model.tokens();const auto logits=model.logits();strata::kernels::SamplerParams sampling;sampling.greedy=true;
            for(auto mode:{VerifyMode::Exact,VerifyMode::Probabilistic}) {
                const auto proposal=model.draft({tokens,logits,{},sampling,mode},3);
                require(proposal.size()==3,"native MTP chain/gate did not reach requested rows");
                for(const auto& row:proposal) require(row.token==0,"independent MTP argmax oracle");
                require(model.tokens()==tokens && model.healthy(),"temporary MTP draft changed committed target prefix");
            }
            sampling.greedy=false;sampling.temperature=100;sampling.top_p=1;sampling.seed=42;
            const auto probability=model.draft({tokens,logits,{},sampling,VerifyMode::Probabilistic},3);
            mtp_reference=probability;
            require(probability.size()==3,"native probabilistic MTP chain missing");
            validate_drafts(probability,model.vocabulary(),3);for(const auto& row:probability) require(row.ids.size()>1,"native sampled MTP discarded its q distribution");
            if(std::filesystem::exists(std::filesystem::path(argv[3])/"draft_vocab.bin")) {
                for(const auto& row:probability) {require(row.ids.size()==3,"subset q expanded into missing head rows");for(auto id:row.ids) require(id==10 || id==0 || id==2,"subset q remap mismatch");}
                double sum=0;for(double value:{50.5,25.25,-25.25}) sum+=std::exp(value/100);
                const auto& row=probability.front();
                for(size_t i=0;i<row.ids.size();++i) {
                    const auto value=row.ids[i]==0?50.5:row.ids[i]==10?25.25:-25.25;
                    require(std::abs(row.probabilities[i]-std::exp(value/100)/sum)<1e-5,"nonzero MTP FC/norm/mix/gather/subset head probability oracle");
                }
            }
        }
        require(model.tokens()==std::vector<int32_t>({1,3}),"model-wide snapshot branch restore");
        const auto saved_mixed=model.mixed();model.save(argv[2]);model.reset();require(model.logits().empty() && model.residual().empty() && model.mixed().empty(),"reset retained valid outputs");model.load(argv[2]);
        require(model.mixed()==saved_mixed,"disk restore lost committed mixed head input");
        require(model.tokens()==std::vector<int32_t>({1,3}),"complete native disk restore history");zero(model.logits());residual(model,3);
        {
            // Upstream saves the live pictures with the token prefix and installs them on RESTORE
            // (conversation_file.cpp:683-684,:758-759; generate.cpp:10615).
            const std::vector<RequestImage> pictures{{0,0x0123456789abcdefull},{1,0xfedcba9876543210ull}};
            model.save(argv[2],pictures);model.reset();
            const auto restored_pictures=model.load(argv[2]);
            require(restored_pictures==pictures,"disk restore lost the live picture identity");
            require(model.tokens()==std::vector<int32_t>({1,3}),"picture-carrying disk restore lost the token history");
            const std::string refused=std::string(argv[2])+".bad-image";
            rejects([&]{model.save(refused,std::vector<RequestImage>{{2,7}});});
            require(!std::filesystem::exists(refused),"a refused picture list published a state file");residual(model,3);
        }
        if(!options.ple_gguf.empty()) {
            const auto saved=std::filesystem::last_write_time(options.ple_gguf);
            std::filesystem::last_write_time(options.ple_gguf,saved+std::chrono::seconds(1));
            rejects([&]{model.load(argv[2]);});
            require(model.healthy() && model.tokens()==std::vector<int32_t>({1,3}),"external PLE identity mismatch changed committed state");
            std::filesystem::last_write_time(options.ple_gguf,saved);model.load(argv[2]);residual(model,3);
        }
        if(!options.expert_pack.empty()) {
            const auto index=std::filesystem::path(options.expert_pack)/"native_experts.txt";
            const auto saved=std::filesystem::last_write_time(index);
            std::filesystem::last_write_time(index,saved+std::chrono::seconds(1));
            rejects([&]{model.load(argv[2]);});
            require(model.healthy() && model.tokens()==std::vector<int32_t>({1,3}),"native pack identity mismatch changed committed state");
            std::filesystem::last_write_time(index,saved);model.load(argv[2]);residual(model,3);
        }
        if(argc>=4) {
            const auto tokens=model.tokens();const auto logits=model.logits();strata::kernels::SamplerParams sampling;sampling.temperature=100;sampling.top_p=1;sampling.seed=42;
            const auto restored=model.draft({tokens,logits,{},sampling,VerifyMode::Probabilistic},3);
            require(restored.size()==mtp_reference.size(),"disk restore lost MTP chain state");
            for(size_t row=0;row<restored.size();++row) {
                require(restored[row].token==mtp_reference[row].token && restored[row].ids==mtp_reference[row].ids,"disk restore changed MTP token/q support");
                for(size_t i=0;i<restored[row].probabilities.size();++i) require(std::abs(restored[row].probabilities[i]-mtp_reference[row].probabilities[i])<1e-6,"disk restore changed MTP logits/residual distribution");
            }
        }
        // A checksum failure must not partially apply any of the 48 layer states.
        {
            std::fstream file(argv[2],std::ios::binary|std::ios::in|std::ios::out);file.seekg(-1,std::ios::end);char byte=0;file.get(byte);byte^=1;
            file.seekp(-1,std::ios::end);file.put(byte);require(bool(file),"corruption fixture write");
        }
        rejects([&]{model.load(argv[2]);});require(model.healthy() && model.size()==2,"corrupt file mutated model state");residual(model,3);
        zero(model.decode(4));const auto accepted=model.tokens();int polls=0;
        rejects([&]{(void)model.decode(5,[&]{return ++polls==12;});});require(model.healthy() && model.tokens()==accepted && model.routing_records().empty(),"mid-trunk cancellation did not roll back/clear diagnostic trace");
        residual(model,4);polls=0;
        rejects([&]{(void)model.decode(5,[&]{return ++polls==(argc>=4?52:51);});});
        require(model.healthy() && model.tokens()==accepted,"post-head cancellation did not rewind all layers");residual(model,4);
        zero(model.decode(6));residual(model,6);
        GenerationOperations op;op.reset=[&]{model.reset();};op.decode=[&](int32_t token,const std::function<bool()>& cancelled){return model.decode(token,cancelled);};
        op.mixed=[&]{return model.mixed();};op.residual=[&]{return model.residual();};
        op.verify=[&](const DraftContext& context,std::span<const DraftRow> rows,const auto& cancelled){return model.verify_candidates(context,rows,cancelled);};
        op.verify_routing=true;op.routing=[&]{return model.routing_records();};
        op.prefill=[&](std::span<const int32_t> ids,const PromptCommit& commit,const auto& cancelled){model.prefill_window(ids,commit,cancelled);};
        op.prefill_routing=[&](std::span<const int32_t> ids,const PromptCommit& commit,const auto& cancelled,const RoutingWindowSink& sink){model.prefill_window(ids,commit,cancelled,sink);};
        op.checkpoint_bytes=model.snapshot_size_limit();op.checkpoint_park_bytes=[&](uint64_t cells){return model.parked_bytes(cells);};
        op.parked_retained_bytes=[&]{return model.parked_retained_bytes();};op.discard_parked=[&]{model.discard_parked();};
        op.checkpoint=[&] {auto point=model.snapshot();return std::function<void()>([&,point]{model.restore(point);});};
        if(argc>=4) op.draft=[&](const DraftContext& context,uint32_t maximum,const auto& cancelled){return model.draft(context,maximum,cancelled);};
        else op.draft=[](auto,uint32_t maximum,const auto&) {return std::vector<DraftRow>(maximum,DraftRow{0,{},{}});};
        GenerationOptions generation_options;generation_options.cache_slots=3;generation_options.cache_bytes=options.memory.snapshot_budget;generation_options.prefill_chunk=8;generation_options.short_read=0;
        generation_options.dumps.mixed=std::string(argv[2])+".one-shot-mixed";
        generation_options.dumps.residual=std::string(argv[2])+".one-shot-residual";
        generation_options.dumps.routing=std::string(argv[2])+".routing-trace";
        generation_options.dump_shape.n_vocab=model.vocabulary();generation_options.dump_shape.hc=4;generation_options.dump_shape.n_embd=2560;
        generation_options.dump_shape.n_layers=48;generation_options.dump_shape.n_expert=512;generation_options.dump_producers.routing=true;
        generation_options.dump_producers.head_mixed=true;generation_options.dump_producers.head_residual=true;
        const auto check_one_shots=[&] {
            std::ifstream mixed(generation_options.dumps.mixed,std::ios::binary);std::vector<float> m(2560);
            mixed.read(reinterpret_cast<char*>(m.data()),std::streamsize(m.size()*sizeof(float)));
            require(bool(mixed) && mixed.peek()==std::char_traits<char>::eof() && m==model.mixed(),"native verify/prefill/cache one-shot mixed does not match committed head");
            std::ifstream file(generation_options.dumps.residual,std::ios::binary);int32_t header[2]{};std::vector<float> r(10240);
            file.read(reinterpret_cast<char*>(header),sizeof(header));file.read(reinterpret_cast<char*>(r.data()),std::streamsize(r.size()*sizeof(float)));
            require(bool(file) && file.peek()==std::char_traits<char>::eof() && header[0]==4 && header[1]==2560 && r==model.residual(),"native one-shot residual has stale/rejected state or wrong byte layout");
        };
        const auto check_routing_file=[&](uint64_t expected,bool all_zero) {
            std::ifstream file(generation_options.dumps.routing,std::ios::binary);require(bool(file),"native routing file missing");uint64_t count=0;
            while(file.peek()!=std::char_traits<char>::eof()) {
                int32_t header[2]{};std::array<int32_t,10> ids{};std::array<float,10> weights{};
                file.read(reinterpret_cast<char*>(header),sizeof(header));file.read(reinterpret_cast<char*>(ids.data()),sizeof(ids));file.read(reinterpret_cast<char*>(weights.data()),sizeof(weights));
                require(bool(file) && header[0]>=0 && header[0]<48 && header[1]==10,"native routing byte layout/truncation");
                for(size_t k=0;k<ids.size();++k) require(ids[k]==int32_t(k) && (weights[k]==0 || (!all_zero && std::abs(weights[k]-.1f)<1e-6f)),"native routing expert IDs/zero-or-real weight convention");
                ++count;
            }
            require(count==expected,"native routing window count duplicated replay or lost micro-window");
        };
        Generation generation(model.vocabulary(),8,model.stop_tokens(),std::move(op),generation_options);generation.adopt(model.tokens(),model.logits());
        GenerationRequest request;request.prompt=model.tokens();request.max_new=3;request.sampling.greedy=true;
        int32_t output=-1;GenerationEvents events;events.token=[&](int32_t token){output=token;};const auto result=generation.run(request,events);
        require(output==0 && result.reused==4 && model.size()==6 && result.drafts_accepted==2,"full native Model -> target verify/bonus/generation chain");
        require(result.dumped_routing_records==96,"adopted prompt native window routing count");check_routing_file(result.dumped_routing_records,true);
        check_one_shots();
        request.prompt={1,3,7,8};request.pin=2;request.max_new=2;(void)generation.run(request,events);
        check_one_shots();
        request.prompt={1,3,8,9};request.max_new=1;const auto shared=generation.run(request,events);
        require(shared.reused==2 && output==0 && generation.cache_stats().bytes<=generation_options.cache_bytes,"full native shared prefix restore/HC state/logits");residual(model,9);
        check_routing_file(shared.dumped_routing_records,false);
        check_one_shots();std::filesystem::remove(generation_options.dumps.mixed);std::filesystem::remove(generation_options.dumps.residual);
        std::filesystem::remove(generation_options.dumps.routing);
        generation.release_checkpoints();
        model.restore(root);require(model.healthy() && !model.size() && model.residual().empty(),"restore root after disk and branch changes");
        const std::array<int32_t,3> window_tokens{1,2,3};uint32_t notified=0;
        model.prefill_window(window_tokens,[&](uint32_t done,std::span<const float> values){require(done>notified,"native window notification order");notified=done;zero(values);});
        routing(model,1); // final micro-window of the 2+1 row request only
        require(model.half_records().empty(),"grouped prefill returned stale scalar half frame");
        require(notified==3 && model.tokens()==std::vector<int32_t>({1,2,3}),"native prefill window did not commit complete history");residual(model,3);
        if(argc>=4) {
            const auto batched=model.snapshot(); // also checks MTP cells == target tokens-1
            const auto ids=model.tokens();const auto logits=model.logits();strata::kernels::SamplerParams sampling;sampling.greedy=true;
            const auto proposal=model.draft({ids,logits,{},sampling,VerifyMode::Exact},2);
            model.restore(root);zero(model.prefill(window_tokens));
            const auto scalar=model.draft({ids,model.logits(),{},sampling,VerifyMode::Exact},2);
            require(proposal.size()==scalar.size(),"known-prefix MTP window changed available draft length");
            for(size_t i=0;i<proposal.size();++i) require(proposal[i].token==scalar[i].token,"known-prefix MTP window differs from zero-weight singleton draft oracle");
            model.restore(batched);
        }
        model.restore(root);rejects([&]{model.prefill_window(window_tokens,[&](uint32_t done,std::span<const float>){if(done==2) throw Cancelled();});});
        require(model.healthy() && model.tokens()==std::vector<int32_t>({1,2}),"native prompt callback cancellation lost accepted prefix");residual(model,2);model.restore(root);
        notified=0;polls=0;rejects([&]{model.prefill_window(window_tokens,[&](uint32_t done,std::span<const float>){notified=done;},[&]{return ++polls==12;});});
        require(!notified && model.healthy() && !model.size() && model.residual().empty(),"cancelled atomic layer-major micro-window exposed partial state");
        zero(model.decode(1));zero(model.decode(3));
        const auto verify_root=model.snapshot();const auto original_tokens=model.tokens();const auto original_logits=model.logits();
        const auto original_mixed=model.mixed();
        zero(model.decode(0));const auto zero_mixed=model.mixed();model.restore(verify_root);
        strata::kernels::SamplerParams greedy;greedy.greedy=true;
        for(const int mismatch:{0,1,2,3}) {
            std::array<DraftRow,3> rows{DraftRow{0,{},{}},DraftRow{0,{},{}},DraftRow{0,{},{}}};
            if(mismatch<3) rows[size_t(mismatch)].token=7;
            const auto verified=model.verify_candidates({original_tokens,original_logits,{},greedy,VerifyMode::Exact},rows);
            routing(model,3);
            require(model.half_records().empty(),"candidate window/replay published scalar half frame");
            require(verified.retained==uint32_t(mismatch) && verified.picks.size()==size_t(mismatch+1) && model.size()==size_t(2+mismatch),"native target verify retained rejected candidate tail");
            zero(verified.logits);residual(model,mismatch?0:3);require(model.healthy(),"native candidate partial rewind poisoned model");
            require(model.mixed()==(mismatch?zero_mixed:original_mixed),"candidate head published rejected/stale mixed row");
            if(argc>=4) {
                const auto ids=model.tokens();const auto values=model.logits();
                require(!model.draft({ids,values,{},greedy,VerifyMode::Exact},1).empty(),"native candidate prefix lost MTP alignment");
            }
            model.restore(verify_root);
        }
        {
            const std::array<DraftRow,1> rows{DraftRow{0,{},{}}};DraftContext context{original_tokens,original_logits,{},greedy,VerifyMode::Exact};
            context.routing=[](const RoutingWindowView& view,bool window){require(window && view.rows==1 && view.layers==48,"native routing callback missing initial pass");throw std::runtime_error("synthetic routing file failure");};
            rejects([&]{model.verify_candidates(context,rows);});
            require(model.healthy() && model.tokens()==original_tokens && model.mixed()==original_mixed && model.routing_records().empty(),"native routing callback failure did not restore numeric root/clear trace");
        }
        {
            const std::array<DraftRow,3> rows{DraftRow{7,{},{}},DraftRow{7,{},{}},DraftRow{8,{},{}}};
            const std::array<int32_t,4> follow{7,7,0,0};
            const auto forced=model.verify_candidates({original_tokens,original_logits,{},greedy,VerifyMode::Exact,follow},rows);
            routing(model,3);
            require(forced.retained==2 && forced.picks.size()==3 && forced.picks[0].token==7 && forced.picks[1].token==7 && forced.picks[2].token==0 &&
                    forced.follow_emitted==3 && forced.follow_differ==2 && model.tokens()==std::vector<int32_t>({1,3,7,7}) && model.healthy(),
                    "native follow was applied after commit or retained its rejected tail");
            zero(forced.logits);residual(model,7);
            if(argc>=4) {const auto ids=model.tokens();const auto values=model.logits();require(!model.draft({ids,values,{},greedy,VerifyMode::Exact},1).empty(),"native follow lost MTP alignment");}
            model.restore(verify_root);
        }
        const auto memory=model.hbm_allocations(),pinned=model.pinned_allocations();const auto cache=model.expert_cache_stats();
        require(memory.used<=memory.limit && pinned.used<=pinned.limit && cache.evictions>0 && cache.resident_bytes<=options.memory.expert_cache,"full-path memory/cache budgets not enforced");
        std::cout<<"PASS real HPU complete 48-layer model, zero-logits oracle, target vocabulary, PLE, GDN/QSA, HC, routed/shared MoE\n"
                   "PASS native prefill/decode, mid-trunk cancel, repeated snapshots, atomic disk restore, generation reuse and tiny-cache eviction\n"
                 <<"hbm_peak="<<memory.peak<<" pinned_peak="<<pinned.peak<<" cache_evictions="<<cache.evictions<<'\n';
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
