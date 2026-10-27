// Actual Generation/DumpSinks over a stateful host target; no HPU execution.
#include "strata/hpu/generation.hpp"
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
using namespace strata::hpu;
namespace {
using Bytes=std::vector<uint8_t>;
void need(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
template<class T> void append(Bytes& out,const T& value) {const auto* data=reinterpret_cast<const uint8_t*>(&value);out.insert(out.end(),data,data+sizeof(T));}
Bytes bytes(const std::filesystem::path& file) {std::ifstream in(file,std::ios::binary);return {std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>()};}
DumpGeometry shape() {DumpGeometry g;g.n_vocab=16;g.n_layers=2;g.n_embd=2;g.hc=1;g.n_head=1;g.n_head_kv=1;g.head_dim=2;g.n_expert=4;return g;}
std::vector<float> sample(int32_t token) {std::vector<float> out(16,-20);out[size_t(token+1)%16]=20;return out;}
struct Target {
    std::vector<int32_t> fed;std::vector<uint32_t> groups;uint32_t scalar=0,halves=0,layers=0,residuals=0;bool raw=false,cancel_group=false,bad_group=false,micro_commits=false;
    void ready() const {need(raw && !fed.empty(),"raw getter used for a grouped or absent frame");}
    GenerationOperations operations(bool grouped,bool routing) {
        GenerationOperations op;op.reset=[&]{fed.clear();groups.clear();scalar=halves=layers=residuals=0;raw=false;};
        op.decode=[&](int32_t token,const std::function<bool()>& stop){if(stop && stop()) throw Cancelled();fed.push_back(token);raw=true;++scalar;return sample(token);};
        op.halves=[&] {
            ready();++halves;std::vector<float> out(52);
            for(size_t i=0;i<out.size();++i) out[i]=float(1000*(fed.size()-1)+100*fed.back()+i);
            const uint32_t nan_bits=0x7fc12345;std::memcpy(out.data()+7,&nan_bits,4);return out;
        };
        op.layer_ladder=[&]{ready();++layers;std::vector<float> out(6);for(size_t i=0;i<out.size();++i) out[i]=float(10*fed.back()+i);return out;};
        op.residual=[&]{ready();++residuals;return std::vector<float>{float(fed.size()-1)+.25f,float(fed.back())+.5f};};
        const auto prefill=[&](std::span<const int32_t> ids,const PromptCommit& commit,const std::function<bool()>& stop) {
            need(!ids.empty(),"empty grouped fixture");groups.push_back(uint32_t(ids.size()));raw=false;
            if(stop && stop()) throw Cancelled();
            if(cancel_group) {fed.insert(fed.end(),ids.begin(),ids.begin()+1);const auto out=sample(fed.back());commit(1,out);throw Cancelled();}
            if(micro_commits) {
                for(uint32_t at=0;at<ids.size();++at) {fed.push_back(ids[at]);const auto out=sample(fed.back());commit(at+1,out);}
            } else {fed.insert(fed.end(),ids.begin(),ids.end());auto out=sample(fed.back());if(bad_group) out[0]=NAN;commit(uint32_t(ids.size()),out);}
        };
        if(grouped) op.prefill=prefill;
        if(routing) {
            op.routing=[&]{std::vector<RoutingRecord> out;for(int32_t layer=0;layer<2;++layer) out.push_back({layer,{int32_t((fed.back()+layer)%4)},{.5f}});return out;};
            if(grouped) op.prefill_routing=[=,this](std::span<const int32_t> ids,const PromptCommit& commit,const auto& stop,const RoutingWindowSink& sink) {
                prefill(ids,commit,stop);std::vector<int32_t> chosen;std::vector<float> weights;
                for(int32_t layer=0;layer<2;++layer) for(auto token:ids) {chosen.push_back((token+layer)%4);weights.push_back(.5f);}
                if(sink) sink({2,uint32_t(ids.size()),1,ids.size(),chosen,weights},true);
            };
        }
        return op;
    }
};
GenerationOptions options(const std::filesystem::path& dir,bool routing) {
    GenerationOptions o;o.max_draft=0;o.dumps.halves=(dir/"halves").string();o.dumps.layers=(dir/"layers").string();o.dumps.final_r=(dir/"final-r").string();
    o.dump_shape=shape();o.dump_producers.half_capture=true;o.dump_producers.layer_ladder=true;o.dump_producers.final_r=true;
    if(routing) {o.dumps.routing=(dir/"routing").string();o.dump_producers.routing=true;}
    return o;
}
struct Case {uint32_t prompt,chunk,until,short_read;bool grouped;std::vector<uint32_t> groups,positions;};
void check(const std::filesystem::path& dir,const Case& c,bool routing,bool micro) {
    Target target;target.micro_commits=micro;auto opts=options(dir,routing);opts.prefill_chunk=c.chunk;opts.prefill_until=c.until;opts.short_read=c.short_read;
    Generation generation(16,64,{15},target.operations(c.grouped,routing),opts);GenerationRequest r;r.max_new=3;r.sampling.greedy=true;for(uint32_t i=0;i<c.prompt;++i) r.prompt.push_back(int32_t(i));
    std::vector<int32_t> output;GenerationEvents events;events.token=[&](int32_t id){output.push_back(id);};const auto result=generation.run(r,events);
    need(output==std::vector<int32_t>({int32_t(c.prompt),int32_t(c.prompt+1),int32_t(c.prompt+2)}),"raw prefill changed generated tokens");
    std::vector<int32_t> fed;for(uint32_t i=0;i<c.prompt+2;++i) fed.push_back(int32_t(i));
    need(target.fed==fed && target.groups==c.groups,"raw dump disabled/repartitioned native prefill or changed fed state");
    const auto count=c.positions.size();need(target.scalar==count && target.halves==count && target.layers==count && target.residuals==count && result.dumped_half_records==count && result.dumped_final_r_records==count && !result.dumped_logits_rows,"raw dump invented/skipped grouped or scalar records");
    Bytes half,layer,tagged;
    for(uint32_t position:c.positions) {
        for(uint32_t i=0;i<52;++i) {if(i==7) append(half,uint32_t(0x7fc12345));else append(half,float(1100*position+i));}
        for(uint32_t i=0;i<6;++i) append(layer,float(10*position+i));
        append(tagged,int64_t(position));append(tagged,int64_t(position));append(tagged,float(position)+.25f);append(tagged,float(position)+.5f);
    }
    need(bytes(opts.dumps.halves)==half && bytes(opts.dumps.layers)==layer && bytes(opts.dumps.final_r)==tagged,"raw prefill suffix bytes/NaN bits/absolute positions differ");
    if(routing) {
        need(result.dumped_routing_records==2*fed.size(),"raw dump lost batched routing or added routes");
        Bytes expected;uint32_t done=0;
        const auto routes=[&](uint32_t first,uint32_t count,bool grouped) {for(int32_t l=0;l<2;++l) for(uint32_t row=first;row<first+count;++row) {append(expected,l);append(expected,int32_t(1));append(expected,int32_t((row+uint32_t(l))%4));append(expected,grouped?0.0f:.5f);}};
        for(auto group:c.groups) {routes(done,group,true);done+=group;}
        for(uint32_t pos:c.positions) routes(pos,1,false);
        need(bytes(opts.dumps.routing)==expected,"raw suffix and real grouped routing publication interfered");
    }
}
}
int main() {
    try {
        const auto parent=std::filesystem::canonical(std::filesystem::temp_directory_path());const auto dir=parent/("strata-position-prefill-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));need(std::filesystem::create_directory(dir),"temporary directory collision");
        const std::vector<Case> cases{
            {7,3,0,0,true,{3,3},{6,7,8}},
            {7,3,4,0,true,{3,1},{4,5,6,7,8}},
            {8,4,0,4,true,{4},{4,5,6,7,8,9}},
            {7,3,0,64,true,{},{0,1,2,3,4,5,6,7,8}},
            {7,3,0,0,false,{},{0,1,2,3,4,5,6,7,8}},
            {1,3,0,0,true,{},{0,1,2}},
        };
        for(bool routing:{false,true}) for(bool micro:{false,true}) for(const auto& c:cases) check(dir,c,routing,micro);
        {
            Target target;GenerationOptions plain;plain.max_draft=0;plain.prefill_chunk=3;plain.short_read=0;
            Generation generation(16,64,{15},target.operations(true,false),plain);GenerationRequest request;request.prompt={0,1,2,3,4,5,6};request.max_new=3;request.sampling.greedy=true;
            GenerationEvents events;events.token=[](int32_t){};const auto result=generation.run(request,events);
            need(target.groups==std::vector<uint32_t>({3,3,1}) && target.scalar==2 && !target.halves && !target.layers && !target.residuals && !result.dumped_half_records,"raw diagnostic boundary changed ordinary batching default or called a disabled producer");
        }
        for(bool invalid:{false,true}) {
            Target target;target.cancel_group=!invalid;target.bad_group=invalid;auto opts=options(dir,false);opts.prefill_chunk=3;opts.short_read=0;
            Generation generation(16,64,{15},target.operations(true,false),opts);GenerationRequest request;request.prompt={0,1,2,3};request.max_new=2;request.sampling.greedy=true;uint32_t emitted=0;
            GenerationEvents events;events.token=[&](int32_t){++emitted;};const auto result=generation.run(request,events);
            need(!emitted && !target.halves && !target.layers && !target.residuals && !result.dumped_half_records && !result.dumped_final_r_records && bytes(opts.dumps.halves).empty() && bytes(opts.dumps.layers).empty() && bytes(opts.dumps.final_r).empty(),"failed grouped prefix published a fake scalar frame");
            need(result.finish==(invalid?"nonfinite":"cancelled") && result.prompt_read==(invalid?0u:1u),"failed prefill diagnostic changed committed prefix notification");
        }
        // Logits still has a full row-count header: combining it with raw
        // files cannot hide unsupported coverage or call/reset the target.
        Target refused;auto opts=options(dir,false);opts.dumps.logits=(dir/"logits").string();opts.prefill_chunk=3;
        std::filesystem::remove(opts.dumps.logits);Generation generation(16,64,{15},refused.operations(true,false),opts);GenerationRequest request;request.prompt={0,1};request.max_new=1;GenerationEvents events;events.token=[](int32_t){};
        bool caught=false;try {generation.run(request,events);} catch(const std::invalid_argument& error) {caught=std::string(error.what()).find("--prefill")!=std::string::npos;}
        need(caught && refused.fed.empty() && !std::filesystem::exists(opts.dumps.logits),"raw flag bypassed logits prefill coverage refusal");
        need(std::filesystem::canonical(dir).parent_path()==parent,"unsafe temporary cleanup");std::filesystem::remove_all(dir);
        std::cout<<"PASS actual Generation raw halves/layers/final-r prefill suffix: grouped/until/short-read/fallback/single prompt, real routing coexistence, absolute positions/packed bytes and prefix failure; host target only\n";
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
