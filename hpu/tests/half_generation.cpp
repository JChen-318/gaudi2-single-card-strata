// Real Generation/HalfCapture/DumpSinks over a stateful host target. No HPU.
#include "strata/hpu/generation.hpp"
#include "strata/hpu/half_capture.hpp"
#include "strata/hpu/qsa_state.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
using namespace strata::hpu;
namespace {
void need(bool ok,const char* why) {if(!ok) throw std::runtime_error(why);}
strata::core::ModelGeometry geometry() {auto g=strata::core::kCompiledGeometry;g.n_layers=4;g.n_embd=3;g.hc=2;g.n_head=2;g.n_head_kv=1;g.head_dim=4;g.idx_top_k=2;return g;}
QsaConfig config() {QsaConfig c;c.query_heads=2;c.kv_heads=1;c.width=4;c.rotary=2;c.index_heads=1;c.index_width=4;c.capacity=32;c.top_k=2;return c;}
std::vector<uint8_t> bytes(const std::filesystem::path& path) {std::ifstream f(path,std::ios::binary);return {std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()};}
struct Target {
    HalfCapture frame;QsaState state{config()};std::vector<int32_t> fed;std::vector<uint8_t> expected;uint32_t getters=0,resets=0;bool malformed=false,incomplete=false;
    Target() {frame.configure(geometry(),HalfCapture::host_bytes(geometry()));}
    GenerationOperations operations() {
        GenerationOperations op;op.reset=[&]{++resets;fed.clear();expected.clear();state.reset();frame.clear();};
        op.decode=[&](int32_t token,const std::function<bool()>& stop) {
            if(stop && stop()) throw Cancelled();
            auto round=frame.round();fed.push_back(token);
            for(uint32_t layer=0;layer<4;++layer) for(uint32_t field=0;field<4;++field) {
                if(incomplete && !layer && !field) continue;
                std::vector<float> values(field<2?3:2);for(size_t at=0;at<values.size();++at) values[at]=float(token*100+int(layer)*10+int(field))+float(at)/8;
                frame.half(layer,HalfField(field),values);
            }
            const std::array<float,4> key{.25f,.5f,.75f,1},value{float(token),float(token)+.25f,float(token)+.5f,float(token)+.75f},raw{1,2,3,4},gamma{1,1,1,1},index{};
            state.append(key,value,raw,gamma);const auto ids=state.select(index);const auto kv=state.gather(ids,uint32_t(ids.size()));
            std::array<float,8> query{};const auto attention=qsa_attention_reference(query,kv,2,1,4);std::array<uint16_t,4> selected{};state.selected_value_bits(ids.front(),selected);
            QsaHalfView view{attention,value,value,key,selected,state.value_pool_prefix()};std::copy_n(ids.begin(),std::min<size_t>(4,ids.size()),view.ids.begin());
            const auto cells=state.size();view.step={int32_t(cells-1),int32_t(cells),int32_t(cells/4),int32_t(config().selection_width(cells))};frame.qsa(3,view);round.finish();
            const auto owned=frame.records();const auto* data=reinterpret_cast<const uint8_t*>(owned.data());expected.insert(expected.end(),data,data+owned.size()*4);
            std::vector<float> logits(16,-20);logits[size_t(token+1)%16]=20;return logits;
        };
        op.halves=[&]{++getters;auto out=frame.records();if(malformed) out.pop_back();return out;};return op;
    }
};
GenerationOptions options(const std::string& path) {
    const auto g=geometry();GenerationOptions o;o.max_draft=0;o.dumps.halves=path;o.dump_producers.half_capture=true;
    o.dump_shape.n_vocab=16;o.dump_shape.n_layers=g.n_layers;o.dump_shape.n_embd=g.n_embd;o.dump_shape.hc=g.hc;o.dump_shape.n_head=g.n_head;o.dump_shape.head_dim=g.head_dim;o.dump_shape.n_head_kv=g.n_head_kv;return o;
}
GenerationRequest request() {GenerationRequest r;r.prompt={1,2};r.max_new=4;r.sampling.greedy=true;return r;}
}
int main() {
    try {
        const auto parent=std::filesystem::canonical(std::filesystem::temp_directory_path());const auto dir=parent/("strata-half-generation-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));need(std::filesystem::create_directory(dir),"temporary half directory collision");
        const auto file=dir/"halves.bin";Target m;Generation g(16,32,{15},m.operations(),options(file.string()));std::vector<int32_t> output;GenerationEvents e;e.token=[&](int32_t id){output.push_back(id);};
        const auto result=g.run(request(),e);need(output==std::vector<int32_t>({3,4,5,6}) && m.fed==std::vector<int32_t>({1,2,3,4,5}),"half publishing changed token/fed state");
        need(result.dumped_half_records==5 && m.getters==5 && bytes(file)==m.expected && bytes(file).size()==5*HalfLayout(geometry()).elements()*4,"Generation omitted/repeated/sanitized or miscounted complete half frames");
        for(int fault=0;fault<3;++fault) {
            Target bad;auto op=bad.operations();auto opts=options((dir/("bad-"+std::to_string(fault))).string());
            if(fault==0) op.halves={};
            if(fault==1) bad.malformed=true;
            if(fault==2) bad.incomplete=true;
            Generation gen(16,32,{15},std::move(op),opts);uint32_t emitted=0;GenerationEvents events;events.token=[&](int32_t){++emitted;};bool failed=false;
            try {gen.run(request(),events);} catch(const std::exception&) {failed=true;}
            need(failed && !emitted,"missing/malformed/incomplete half producer published a token");
            if(!fault) need(bad.fed.empty() && !bad.resets && !std::filesystem::exists(opts.dumps.halves),"missing half capability changed state/opened file before refusal");
        }
        for(int mode=0;mode<3;++mode) {
            Target bad;auto op=bad.operations();auto opts=options((dir/("coverage-"+std::to_string(mode))).string());
            if(mode==0) {opts.prefill_chunk=4;opts.dumps.logits=(dir/"coverage-logits").string();}
            if(mode==1) {
                opts.cache_slots=2;op.checkpoint_bytes=1;
                op.checkpoint=[]()->std::function<void()>{throw std::runtime_error("cache producer ran before half coverage refusal");};
            }
            if(mode==2) {opts.max_draft=2;op.draft=[](const DraftContext&,uint32_t,const std::function<bool()>&){return std::vector<DraftRow>{{3,{},{}}};};}
            Generation gen(16,32,{15},std::move(op),opts);bool failed=false;try {gen.run(request(),e);} catch(const std::invalid_argument&) {failed=true;}
            need(failed && bad.fed.empty() && !bad.resets && !std::filesystem::exists(opts.dumps.halves),"half coverage silently disabled cache/window/draft or ran target before refusal");
        }
        Target off;auto op=off.operations();op.halves=[]()->std::vector<float>{throw std::runtime_error("unarmed half getter called");};GenerationOptions plain;plain.max_draft=0;Generation unarmed(16,32,{15},std::move(op),plain);need(unarmed.run(request(),e).dumped_half_records==0,"half off produced records");
#ifdef __linux__
        Target full;Generation failed_io(16,32,{15},full.operations(),options("/dev/full"));uint32_t published=0;GenerationEvents events;events.token=[&](int32_t){++published;};bool failed=false;
        try {failed_io.run(request(),events);} catch(const std::exception& error) {failed=std::string(error.what()).find("half")!=std::string::npos;}
        need(failed && !published && full.fed==std::vector<int32_t>{1},"buffered half IO failure published tokens or pretended numeric rollback");
#endif
        need(std::filesystem::canonical(dir).parent_path()==parent,"unsafe half temporary cleanup");std::filesystem::remove_all(dir);
        std::cout<<"PASS actual Generation/HalfCapture/DumpSinks: dynamic scalar frames/packed bits/counts, missing/malformed/incomplete producers, coverage refusal, disabled getter and buffered IO failure; host target only\n";
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
