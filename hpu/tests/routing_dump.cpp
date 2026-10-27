#include "strata/hpu/generation.hpp"
#include "strata/hpu/routing_capture.hpp"
#include "strata/hpu/slot_session.hpp"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>
using namespace strata::hpu;
namespace {
void need(bool ok,const char* text) {if(!ok) throw std::runtime_error(text);}
std::filesystem::path scratch;
std::vector<uint8_t> bytes(const std::filesystem::path& path) {std::ifstream in(path,std::ios::binary);return {std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>()};}
template<class T> void append(std::vector<uint8_t>& out,const T& value) {const auto* p=reinterpret_cast<const uint8_t*>(&value);out.insert(out.end(),p,p+sizeof(value));}
constexpr uint32_t layers=2,experts=16,top_k=2,vocabulary=32;
struct Target {
    std::vector<int32_t> state;RoutingCapture capture;std::vector<uint8_t> expected;
    uint32_t native_calls=0,prefill_calls=0,exports=0,reset_calls=0;int mismatch=2,fault=0;bool native=false;
    Target() {capture.configure(layers,7,experts,top_k);}
    static std::vector<float> logits(int32_t token) {std::vector<float> out(vocabulary,-20);out[size_t((token+1)%int32_t(vocabulary))]=20;return out;}
    static Routes routes(std::span<const int32_t> ids,uint32_t layer) {
        Routes out;out.tokens=uint32_t(ids.size());out.experts=experts;out.top_k=top_k;
        for(auto id:ids) {out.ids.push_back((id+int32_t(layer))%int32_t(experts));out.ids.push_back((id+int32_t(layer)+1)%int32_t(experts));out.weights.push_back(.25f);out.weights.push_back(.75f);}
        return out;
    }
    void route(std::span<const int32_t> ids) {for(uint32_t layer=0;layer<layers;++layer) capture.capture(layer,routes(ids,layer));}
    void expect(const RoutingWindowView& view,bool window) {
        // Independent transcription of {int32 layer,k}, k ids, k floats.
        for(uint32_t layer=0;layer<view.layers;++layer) for(uint32_t row=0;row<view.rows;++row) {
            append(expected,int32_t(layer));append(expected,int32_t(view.top_k));
            const auto at=uint64_t(layer)*view.layer_stride+uint64_t(row)*view.top_k;
            for(uint32_t k=0;k<view.top_k;++k) append(expected,view.ids[at+k]);
            for(uint32_t k=0;k<view.top_k;++k) append(expected,window?0.0f:view.weights[at+k]);
        }
    }
    GenerationOperations operations() {
        GenerationOperations op;op.reset=[&]{++reset_calls;state.clear();capture.clear();};
        op.decode=[&](int32_t id,const std::function<bool()>& stop) {
            if(stop && stop()) throw Cancelled();
            auto round=capture.round(1);state.push_back(id);route(std::span(&id,1));round.finish();return logits(id);
        };
        op.routing=[&]{++exports;if(native) expect(capture.physical_view(),false);return capture.records();};
        op.checkpoint_bytes=128;op.checkpoint=[&]{return std::function<void()>([&,saved=state]{state=saved;capture.clear();});};
        op.draft=[&](const DraftContext& c,uint32_t n,const std::function<bool()>&) {
            std::vector<DraftRow> out;for(uint32_t at=0;at<n;++at) out.push_back({int32_t((c.history.back()+1+int32_t(at)+(int(at)==mismatch?7:0))%int32_t(vocabulary)),{},{}});return out;
        };
        const auto prefill=[&](std::span<const int32_t> ids,const PromptCommit& commit,const std::function<bool()>& stop,const RoutingWindowSink& sink) {
            ++prefill_calls;uint32_t done=0;
            while(done<ids.size()) {
                if(stop && stop()) throw Cancelled();
                const auto part=ids.subspan(done,std::min<size_t>(2,ids.size()-done));
                auto round=capture.round(uint32_t(part.size()));state.insert(state.end(),part.begin(),part.end());route(part);round.finish();done+=uint32_t(part.size());
                const auto values=logits(part.back());const auto view=capture.physical_view();
                if(fault==1 && sink) sink(view,true); // must be after commit
                commit(done,values);
                if(sink && fault!=2) {if(native) expect(view,true);sink(view,true);if(fault==3) sink(view,true);}
            }
        };
        op.prefill=[prefill](std::span<const int32_t> ids,const PromptCommit& commit,const std::function<bool()>& stop){prefill(ids,commit,stop,{});};
        op.prefill_routing=prefill;
        if(native) {
            op.verify_routing=true;
            op.verify=[&](const DraftContext& c,std::span<const DraftRow> rows,const std::function<bool()>& stop) {
                ++native_calls;need(std::equal(state.begin(),state.end(),c.history.begin(),c.history.end()),"native route source saw a different physical prefix");
                const auto root=state;auto round=capture.round(uint32_t(rows.size()));std::vector<int32_t> ids;std::vector<float> after;
                for(const auto& row:rows) {ids.push_back(row.token);const auto values=logits(row.token);after.insert(after.end(),values.begin(),values.end());}
                try {
                    state.insert(state.end(),ids.begin(),ids.end());route(ids);
                    const std::array<int32_t,1> stops{14};auto selected=select_verified_candidates(c.logits,after,rows,c.history,c.sampling,c.bias,c.mode,stops,stop,c.follow);
                    if(c.routing && fault!=4) {
                        auto view=capture.physical_view();expect(view,true);if(fault==5) ++view.rows;
                        c.routing(view,fault!=6);if(fault==7) c.routing(view,true);
                    }
                    state=root;for(uint32_t at=0;at<selected.retained;++at) state.push_back(rows[at].token);
                    // The numeric replay's routing callback must be ignored.
                    if(selected.retained) route(std::span(ids).first(selected.retained));
                    round.finish();selected.logits=logits(state.back());return selected;
                } catch(...) {state=root;throw;}
            };
        }
        return op;
    }
};
GenerationOptions options(const std::filesystem::path& file,bool prefill,bool cache,bool trace=true) {
    GenerationOptions o;o.max_draft=2;o.prefill_chunk=prefill?6:0;o.short_read=0;o.cache_slots=cache?3:0;o.cache_bytes=1u<<20;
    if(trace) o.dumps.routing=file.string();
    o.dump_shape.n_layers=layers;o.dump_shape.n_expert=experts;o.dump_producers.routing=true;return o;
}
GenerationRequest request(uint32_t n=8) {GenerationRequest r;r.prompt={1,2,3,4,5,6,7,8,9};r.max_new=n;r.sampling.greedy=true;return r;}
struct Run {std::vector<uint8_t> bytes;std::vector<int32_t> output,state;uint64_t records=0;};
Run run(bool native,bool prefill,bool cache,int mismatch,bool trace=true,uint32_t window=2) {
    const auto file=scratch/("trace-"+std::to_string(native)+std::to_string(prefill)+std::to_string(cache)+std::to_string(mismatch)+std::to_string(trace));std::filesystem::remove(file);
    Target model;model.native=native;model.mismatch=mismatch;auto config=options(file,prefill,cache,trace);config.max_draft=window;
    Generation gen(vocabulary,128,{14},model.operations(),config);Run out;
    for(int pass=0;pass<(cache?2:1);++pass) {
        model.expected.clear();out.output.clear();GenerationEvents events;events.token=[&](int32_t id){out.output.push_back(id);};
        const auto result=gen.run(request(),events);out.bytes=bytes(file);out.state=model.state;out.records=result.dumped_routing_records;
        need(out.output==std::vector<int32_t>({10,11,12,13,14}) && result.finish=="stop" && out.state==std::vector<int32_t>({1,2,3,4,5,6,7,8,9,10,11,12,13}),"routing dump changed target publication or fed rejected/EOS tail");
        if(cache && pass) need(result.reused==9,"routing dump disabled full prompt cache reuse");
        if(trace) {
            need(out.records && out.bytes.size()==out.records*24,"routing file record count/layout differs");
            if(native) need(out.bytes==model.expected,"native output lost/reordered physical rows, retained replay or wrong weights");
        } else need(out.bytes.empty() && !out.records && !model.exports,"routing off read a producer or wrote bytes");
    }
    if(native) need(model.native_calls>0,"routing silently bypassed native verification");
    if(prefill) need(model.prefill_calls>0,"routing silently bypassed prefill");
    return out;
}
void matrix() {
    for(bool prefill:{false,true}) for(bool cache:{false,true}) for(int mismatch:{0,1,2}) {
        const auto scalar=run(false,prefill,cache,mismatch),native=run(true,prefill,cache,mismatch);
        need(scalar.bytes==native.bytes && scalar.records==native.records,"scalar/native layer-major routing windows differ");
        const auto off=run(true,prefill,cache,mismatch,false);need(off.output==native.output && off.state==native.state,"optional routing changed numerical path");
    }
    for(uint32_t window=1;window<=7;++window) {
        const auto scalar=run(false,true,true,2,true,window),native=run(true,true,true,2,true,window);
        need(scalar.bytes==native.bytes && scalar.records==native.records,"bounded 1..7 row scalar/native routing wire differs");
    }
    // A fully cached one-output call performs no new MoE work: an EMPTY trace
    // is correct; emitting the old Model routing cache would be stale data.
    const auto file=scratch/"cached-only";Target m;m.native=true;Generation g(vocabulary,128,{},m.operations(),options(file,true,true));GenerationEvents e;e.token=[](int32_t){};
    g.run(request(1),e);const auto again=g.run(request(1),e);need(again.reused==9 && bytes(file).empty() && !again.dumped_routing_records,"cached request re-emitted stale routing");
}
void failures() {
    for(int fault=1;fault<=7;++fault) {
        Target m;m.native=true;m.fault=fault;Generation g(vocabulary,128,{14},m.operations(),options(scratch/("bad-"+std::to_string(fault)),fault<=3,false));
        uint32_t emitted=0;GenerationEvents e;e.token=[&](int32_t){++emitted;};bool fatal=false;
        try {g.run(request(),e);} catch(const SessionFatal&) {fatal=true;}
        need(fatal && !emitted,"missing/duplicated/out-of-order routing source was not zero-publication fatal");
    }
    for(bool prefill:{false,true}) {
        Target m;m.native=true;auto op=m.operations();if(prefill) op.prefill_routing={};else op.verify_routing=false;
        const auto file=scratch/(prefill?"missing-prefill":"missing-native");std::filesystem::remove(file);Generation g(vocabulary,128,{14},op,options(file,prefill,false));GenerationEvents e;e.token=[](int32_t){};bool failed=false;
        try {g.run(request(),e);} catch(const std::invalid_argument&) {failed=true;}
        need(failed && m.state.empty() && !m.reset_calls && !std::filesystem::exists(file),"missing routing capability was not refused before mutation/file open");
    }
#ifdef __linux__
    {
        Target m;auto config=options("/dev/full",false,false);DumpSinks writer(config.dumps,config.dump_shape,config.dump_producers);bool failed=false;
        try {auto round=m.capture.round(1);const std::array<int32_t,1> ids{3};m.route(ids);round.finish();writer.routing_view(m.capture.physical_view(),false);}
        catch(const std::exception& error) {failed=std::string(error.what()).find("routing")!=std::string::npos;}
        need(failed && m.capture.records().empty() && !m.capture.physical_view().layers,"real post-finish routing IO failure leaked the diagnostic view");
    }
    for(bool native:{false,true}) {
        // Adopt a complete prompt so /dev/full first fails inside candidate
        // publication, exercising the native selector/scalar root rollback.
        Target m;m.native=native;m.state=request().prompt;const auto root=m.state;Generation g(vocabulary,128,{14},m.operations(),options("/dev/full",false,false));g.adopt(root,Target::logits(root.back()));
        uint32_t emitted=0;GenerationEvents e;e.token=[&](int32_t){++emitted;};bool failed=false;
        try {g.run(request(),e);} catch(const std::exception& error) {failed=std::string(error.what()).find("routing")!=std::string::npos;}
        need(failed && !emitted && m.state==root,"routing IO failure published tokens or lost the native/scalar root");
    }
#endif
}
std::vector<uint8_t> resident(bool native,bool prefill) {
    const auto file=scratch/("resident-"+std::to_string(native)+std::to_string(prefill));
    Target m;m.native=native;auto config=options(file,prefill,true);
    config.resident_routing=std::make_shared<DumpSinks>(config.dumps,config.dump_shape,config.dump_producers);
    const auto sink=config.resident_routing;auto op=m.operations();const auto decode=op.decode;const auto checkpoint=op.checkpoint;
    SlotSession* slots=nullptr;
    op.select_prefix=[&](std::span<const int32_t> ids,uint64_t maximum,bool mode,std::span<const RequestImage> images)->std::optional<GenerationPrefix>{return slots?slots->select_prefix(ids,maximum,mode,images):std::nullopt;};
    Generation g(vocabulary,128,{14},std::move(op),config);std::vector<int32_t> output;
    GenerationEvents e;e.token=[&](int32_t id){output.push_back(id);};
    uint64_t total=0;
    for(int pass=0;pass<2;++pass) {
        const auto before=bytes(file);output.clear();const auto result=g.run(request(),e);total+=result.dumped_routing_records;
        const auto after=bytes(file);
        need(output==std::vector<int32_t>({10,11,12,13,14}) && !sink->closed(),"resident request closed the shared file or changed output");
        need(after.size()==total*24 && after.size()>before.size() && std::equal(before.begin(),before.end(),after.begin()),"resident request truncated/reopened the file or counted historical records");
        if(pass) need(result.reused==9,"resident trace disabled prefix reuse");
        if(native) need(after==m.expected,"resident native physical trace differs from dispatch oracle");
    }
    const auto before_cache=bytes(file);output.clear();const auto cached=g.run(request(1),e);
    need(cached.reused==9 && !cached.dumped_routing_records && bytes(file)==before_cache && !sink->closed(),"fully cached resident request truncated or repeated stale routing");

    // The same production SlotSession uses the process writer for prompt,
    // pending-token decode and its two-row native MTP window. No Model reentry
    // is needed while the borrowed physical view is being published.
    const RoutingWindowSink write=[&](const RoutingWindowView& view,bool window){m.expect(view,window);sink->routing_view(view,window);};
    auto producer=m.operations();SlotSessionOperations so;
    so.before_foreign=[&]{g.suspend_live();};so.reset=producer.reset;so.checkpoint=checkpoint;so.logits=[&]{return Target::logits(m.state.back());};
    so.select_prefix=[&](std::span<const int32_t> ids,uint64_t maximum,bool mode,std::span<const RequestImage> images){return g.select_prefix(ids,maximum,mode,images);};
    so.import_prefixes=[&](const GenerationRead& read){g.import_prefixes(read);};
    const auto slot_decode=[&](int32_t id){auto values=decode(id,{});write(m.capture.physical_view(),false);return values;};so.decode=slot_decode;
    so.prefill=[slot_decode](std::span<const int32_t> ids){std::vector<float> values;for(auto id:ids) values=slot_decode(id);return values;};
    so.prefill_cancellable=[&](std::span<const int32_t> ids,bool window,const PromptCommit& commit,const std::function<bool()>& stop) {
        if(window) producer.prefill_routing(ids,commit,stop,[&](const RoutingWindowView& view,bool phase){if(!m.native) m.expect(view,phase);sink->routing_view(view,phase);});
        else for(uint32_t at=0;at<ids.size();++at) {const auto values=decode(ids[at],stop);commit(at+1,values);write(m.capture.physical_view(),false);}
    };
    uint32_t spec_calls=0;
    so.speculate=[&](int32_t pending,std::span<const int32_t> history,const strata::kernels::SamplerParams& sampling)->std::optional<SlotCandidates> {
        ++spec_calls;const auto root=m.state;
        need(std::equal(root.begin(),root.end(),history.begin(),history.end()),"resident slot window saw the wrong owner");
        SlotCandidates result;result.draft.token=pending+1;const std::array<int32_t,2> ids{pending,pending+1};
        auto round=m.capture.round(2);std::vector<float> after;
        for(auto id:ids) {const auto values=Target::logits(id);after.insert(after.end(),values.begin(),values.end());}
        try {
            m.state.insert(m.state.end(),ids.begin(),ids.end());m.route(ids);
            const std::array<int32_t,1> stops{14};result.selected=select_slot_candidates(after,result.draft,pending,root,sampling,stops);
            write(m.capture.physical_view(),true);
            m.state=root;m.state.insert(m.state.end(),ids.begin(),ids.begin()+1+result.selected.retained);
            m.route(std::span(ids).first(1+result.selected.retained));round.finish();
            result.selected.logits=Target::logits(m.state.back());return result;
        } catch(...) {m.state=root;throw;}
    };
    SlotCheckpointOptions checks;checks.prefill_chunk=prefill?6:0;checks.short_read=0;
    SlotSession engine(2,vocabulary,128,{14},std::move(so),true,checks);slots=&engine;
    auto slot_request=request();slot_request.prompt={2,3,4,5};slot_request.max_new=5;
    output.clear();const auto admitted=engine.admit(0,slot_request,5,[&](int32_t id){output.push_back(id);});
    need(admitted.continues && output==std::vector<int32_t>{6},"resident slot admission changed first token");
    auto window=engine.step_window(0,2);for(uint32_t at=0;at<window.count;++at) output.push_back(window.rows[at].token);
    auto scalar=engine.step_window(0,1);for(uint32_t at=0;at<scalar.count;++at) output.push_back(scalar.rows[at].token);
    need(spec_calls==1 && output==std::vector<int32_t>({6,7,8,9}),"resident slot native/scalar continuation changed output or bypassed window");
    engine.finish(0);need(!sink->closed(),"slot completion closed shared routing writer");
    const auto with_slot=bytes(file);need(with_slot.size()>before_cache.size() && std::equal(before_cache.begin(),before_cache.end(),with_slot.begin()),"slot work truncated resident trace");
    if(native) need(with_slot==m.expected,"slot prompt/window/scalar records or weights differ from dispatch oracle");
    // Interleaved foreign publications must not enter this GEN's own summary.
    bool foreign=false;e.progress=[&](uint32_t,uint32_t,double){if(!foreign) {foreign=true;const std::array<int32_t,1> id{3};auto round=m.capture.round(1);m.route(id);round.finish();write(m.capture.physical_view(),false);}};
    auto different=request(1);different.prompt={3,4,5};const auto count_before=sink->routing_records();
    output.clear();const auto last=g.run(different,e);
    need(foreign && output==std::vector<int32_t>{6} && sink->routing_records()-count_before==last.dumped_routing_records+layers,"resident GEN counted a foreign slot publication");
    need(sink->close(),"resident close failed");const auto final=bytes(file);const auto state=m.state;
    bool refused=false;try {g.run(request(),e);} catch(const std::invalid_argument&) {refused=true;}
    need(refused && state==m.state && bytes(file)==final,"closed resident writer was silently reopened or mutated Model");
    return final;
}
void resident_lifetime() {
    for(bool prefill:{false,true}) need(resident(false,prefill)==resident(true,prefill),"resident scalar/native window byte layouts diverged");
    Target m;auto config=options(scratch/"resident-contract",false,false);
    config.resident_routing=std::make_shared<DumpSinks>(config.dumps,config.dump_shape,config.dump_producers);
    for(int mismatch=0;mismatch<3;++mismatch) {
        auto bad=config;if(mismatch==0) bad.dumps.routing+=".other";else if(mismatch==1) ++bad.dump_shape.n_layers;else bad.dumps.mixed="unused";
        bool rejected=false;try {Generation g(vocabulary,128,{},m.operations(),bad);} catch(const std::invalid_argument&) {rejected=true;}
        need(rejected && m.state.empty(),"mismatched/nonrouting resident sink accepted");
    }
}
}
int main() {
    try {
        const auto temporary=std::filesystem::canonical(std::filesystem::temp_directory_path());
        scratch=temporary/("strata-hpu-routing-dump-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directory(scratch);matrix();failures();resident_lifetime();
        need(std::filesystem::canonical(scratch).parent_path()==temporary,"unsafe temporary cleanup path");std::filesystem::remove_all(scratch);
        std::cout<<"PASS real Generation routing files: native/scalar layer-major windows, micro-window streaming, true/zero weights, replay exclusion, cache/no-work, source contracts and rollback; host only\n";
        std::cout<<"PASS resident routing: one shared file across GEN/slot/prefill/native MTP/scalar continuation, cache/no-work, owner-local counters, close/refusal and scalar/native byte parity; host only\n";
    }
    catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
