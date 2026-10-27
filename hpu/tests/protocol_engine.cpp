// Test-only host engine. It shares the production generation/protocol code;
// the production executable has no fake-model flag or bypass.
#include "strata/hpu/protocol.hpp"
#include "strata/hpu/state_io.hpp"
#include "strata/hpu/slot_session.hpp"
#include "strata/hpu/routing_capture.hpp"
#include <chrono>
#include <iostream>
#include <thread>
#include <filesystem>
#include <unistd.h>
using namespace strata::hpu;
int main(int argc,char** argv) {
    try {
        // The original Python adapter always prepends --serve. Keep that
        // launch contract in the host fixture as well as direct pipe tests.
        bool batch=false;bool vision=false;std::string routing_file;
        for(int at=1;at<argc;++at) {
            const std::string arg=argv[at];
            if(arg=="--serve") continue;
            if(arg=="--vision") {vision=true;continue;}
            if(arg=="--batch" && at+1<argc && std::string(argv[at+1])=="3") {batch=true;++at;}
            else if(arg=="--dump-routing" && at+1<argc) routing_file=argv[++at];
            else throw std::invalid_argument("test fixture accepts [--serve] [--batch 3] [--dump-routing FILE] only");
        }
        std::vector<int32_t> state;state.reserve(1024);std::vector<float> last(256);
        RoutingCapture routing;if(!routing_file.empty()) routing.configure(2,7,16,2);
        SlotSession* shared=nullptr;
        GenerationOperations op;op.reset=[&]{state.clear();};
        op.decode=[&](int32_t token,const std::function<bool()>& cancelled) {
            if(cancelled && cancelled()) throw Cancelled();
            std::this_thread::sleep_for(std::chrono::milliseconds(3));
            if(cancelled && cancelled()) throw Cancelled();
            std::vector<float> logits(256,-100);logits[(token+1)%256]=100;state.push_back(token);std::copy(logits.begin(),logits.end(),last.begin());return logits;
        };
        if(!routing_file.empty()) {
            const auto plain_decode=op.decode;
            op.decode=[&,plain_decode](int32_t token,const std::function<bool()>& cancelled) {
                auto round=routing.round(1);auto values=plain_decode(token,cancelled);
                for(uint32_t layer=0;layer<2;++layer) {Routes routes;routes.tokens=1;routes.experts=16;routes.top_k=2;routes.ids={int32_t((token+layer)%16),int32_t((token+layer+1)%16)};routes.weights={.25f,.75f};routing.capture(layer,routes);}
                round.finish();return values;
            };
            op.routing=[&]{return routing.records();};
        }
        op.checkpoint_bytes=65536;
        op.checkpoint=[&] {return std::function<void()>([&,saved=state,values=last]{state=saved;last=values;});};
        op.select_prefix=[&](std::span<const int32_t> ids,uint64_t maximum,bool control,std::span<const RequestImage> images)->std::optional<GenerationPrefix>{return shared?shared->select_prefix(ids,maximum,control,images):std::nullopt;};
        const auto decode=op.decode;const auto checkpoint=op.checkpoint;const auto reset=op.reset;
        GenerationOptions options;options.cache_slots=3;options.cache_bytes=1ull<<20;options.suffix_min_match=batch?0:3;
        if(!routing_file.empty()) {
            options.dumps.routing=routing_file;options.dump_shape.n_layers=2;options.dump_shape.n_expert=16;options.dump_producers.routing=true;
            options.resident_routing=std::make_shared<DumpSinks>(options.dumps,options.dump_shape,options.dump_producers);
        }
        const auto writer=options.resident_routing;
        Generation generator(256,1024,{255},std::move(op),options);ProtocolControl control;
        control.vram=[](std::optional<uint64_t> reserve) {
            const auto bytes=reserve.value_or(512ull<<20);if(bytes>1ull<<30) throw std::invalid_argument("reserve too large");
            return "expert_slots=2 vram_free_mib=1024 reserve_mib="+std::to_string(bytes/(1ull<<20));
        };
        control.session=[&](bool save,const std::string& path) {
            generator.release_checkpoints();
            if(save) {
                StateWriter writer(path,65536);writer.integer(state.size());writer.vector<int32_t>(state);
                writer.vector<float>(state.empty()?std::span<const float>{}:std::span<const float>(last));writer.finish();
            } else {
                StateReader reader(path,65536);const auto count=reader.integer();if(count>1024) throw std::invalid_argument("test state position");
                auto tokens=reader.vector<int32_t>(count);auto logits=reader.vector<float>(count?256:0);reader.finish();
                generator.adopt(tokens,logits);state=std::move(tokens);if(count) last=std::move(logits);
            }
            return ProtocolSessionResult{state.size(),std::filesystem::file_size(path)};
        };
        if(batch) {
            SlotSessionOperations so;so.before_foreign=[&]{generator.suspend_live();};so.reset=reset;so.checkpoint=checkpoint;
            so.decode=[&](int32_t id){auto out=decode(id,{});if(writer) writer->routing_view(routing.physical_view(),false);return out;};
            so.prefill=[&](std::span<const int32_t> ids){std::vector<float> out;for(auto id:ids) {out=decode(id,{});if(writer) writer->routing_view(routing.physical_view(),false);}return out;};so.prefill_window=so.prefill;
            so.prefill_cancellable=[&](std::span<const int32_t> ids,bool,const PromptCommit& commit,const std::function<bool()>& stop){for(uint32_t at=0;at<ids.size();++at) {const auto values=decode(ids[at],stop);commit(at+1,values);if(writer) writer->routing_view(routing.physical_view(),false);}};
            so.logits=[&]{return last;};so.select_prefix=[&](std::span<const int32_t> ids,uint64_t maximum,bool mode,std::span<const RequestImage> images){return generator.select_prefix(ids,maximum,mode,images);};so.import_prefixes=[&](const GenerationRead& read){generator.import_prefixes(read);};
            if(vision) {control.vision=true;control.vision_width=16;so.set_vision=[](const GenerationRequest&){};}
            SlotCheckpointOptions checks;checks.prefill_chunk=8;checks.short_read=0;checks.interval=0;checks.boundaries.turn_token=-1;
            SlotSession slots(3,256,1024,{255},std::move(so),true,checks);shared=&slots;control.solo=&generator;
            serve_batch_protocol(STDIN_FILENO,std::cout,slots,3,256,1024,std::move(control));
        } else serve_protocol(STDIN_FILENO,std::cout,generator,256,1024,std::move(control));
        if(writer && !writer->close()) throw std::runtime_error(writer->close_error());
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
