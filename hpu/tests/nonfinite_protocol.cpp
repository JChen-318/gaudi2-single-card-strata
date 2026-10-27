// Host integration of #879 through production Gen/SlotSession/protocol.
// Numeric rows and owned images are fixtures, never an HPU execution claim.
#include "strata/hpu/slot_session.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <sstream>
#include <unistd.h>
using namespace strata::hpu;
namespace {
void need(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
size_t count(const std::string& text,const std::string& prefix) {
    std::istringstream input(text);std::string line;size_t n=0;while(std::getline(input,line)) n+=line.starts_with(prefix);return n;
}
struct Forward:BatchSlotSession {
    SlotSession& native;uint32_t flushes=0;
    explicit Forward(SlotSession& value):native(value) {}
    Admission admit(int s,const GenerationRequest& r,uint32_t n,const std::function<void(int32_t)>& out) override {return native.admit(s,r,n,out);}
    Admission admit_cooperative(int s,const GenerationRequest& r,uint32_t n,const std::function<void(int32_t)>& out,const ReadControl& c) override {return native.admit_cooperative(s,r,n,out,c);}
    bool park_partial(int s,const GenerationRead& read) override {return native.park_partial(s,read);}
    Step step(int s) override {return native.step(s);}
    Window step_window(int s,uint32_t n) override {return native.step_window(s,n);}
    void stop(int s) override {native.stop(s);}
    void release(int s) override {native.release(s);}
    void finish(int s) override {native.finish(s);}
    void invalidate_caches() override {++flushes;native.invalidate_caches();}
};
struct Fixture {
    std::vector<int32_t> fed;bool mode=true,armed=false,fatal=false,poison_prompt=false;
    std::unique_ptr<Generation> main;std::unique_ptr<SlotSession> slots;std::unique_ptr<Forward> engine;
    uint32_t captures=0,session_calls=0;
    std::vector<float> logits() const {std::vector<float> out(32,-10);out[mode?1:2]=10;return out;}
    std::vector<float> decode(int32_t id) {
        if(!mode && id==2 && (armed || fatal)) {
            const bool die=fatal;armed=false;fatal=false;
            if(die) throw SessionFatal("injected poisoned physical window");
            fed.push_back(id);auto out=logits();out[0]=NAN;return out;
        }
        fed.push_back(id);auto out=logits();
        if(poison_prompt && id==31) {poison_prompt=false;out[0]=INFINITY;}
        return out;
    }
    std::function<void()> image() {++captures;auto saved=std::make_shared<std::pair<std::vector<int32_t>,bool>>(fed,mode);return [this,saved]{fed=saved->first;mode=saved->second;};}
    Fixture() {
        GenerationOperations go;go.reset=[&]{fed.clear();};go.decode=[&](int32_t id,const auto&){return decode(id);};
        go.prefill=[&](std::span<const int32_t> ids,const PromptCommit& commit,const auto&) {
            for(uint32_t i=0;i<ids.size();++i) {if(ids[i]==30) armed=true;auto out=decode(ids[i]);commit(i+1,out);}
        };
        go.control_vector=[&](bool value){mode=value;};go.checkpoint=[&]{return image();};go.checkpoint_bytes=100;
        go.available_host_bytes=[] {return std::optional<uint64_t>(UINT64_MAX);};
        go.select_prefix=[&](std::span<const int32_t> p,uint64_t n,bool c,std::span<const RequestImage> images){return slots?slots->select_prefix(p,n,c,images):std::optional<GenerationPrefix>{};};
        GenerationOptions options;options.max_draft=0;options.cache_slots=4;options.cache_bytes=1u<<20;options.prefill_chunk=2;options.short_read=0;options.cache_every=0;options.boundaries.turn_token=-1;
        main=std::make_unique<Generation>(32,128,std::vector<int32_t>{},std::move(go),options);
        SlotSessionOperations so;so.before_foreign=[&]{main->suspend_live();};so.reset=[&]{fed.clear();};so.control_vector=[&](bool value){mode=value;};
        so.decode=[&](int32_t id){return decode(id);};so.prefill=[&](std::span<const int32_t> ids){std::vector<float> out;for(auto id:ids) out=decode(id);return out;};so.prefill_window=so.prefill;
        so.checkpoint=[&]{return image();};so.logits=[&]{return logits();};so.import_prefixes=[&](const GenerationRead& read){main->import_prefixes(read);};
        so.select_prefix=[&](std::span<const int32_t> p,uint64_t n,bool c,std::span<const RequestImage> images){return main->select_prefix(p,n,c,images);};
        SlotCheckpointOptions sc;sc.prefill_chunk=2;sc.short_read=0;sc.interval=0;sc.boundaries.turn_token=-1;
        slots=std::make_unique<SlotSession>(3,32,128,std::vector<int32_t>{},std::move(so),true,sc);engine=std::make_unique<Forward>(*slots);
    }
    std::string serve(const std::string& commands) {
        int fd[2]{};need(::pipe(fd)==0,"nonfinite protocol pipe");
        need(::write(fd[1],commands.data(),commands.size())==ssize_t(commands.size()),"nonfinite command write");::close(fd[1]);
        ProtocolControl control;control.solo=main.get();
        control.session=[&](bool save,const std::string& path) {
            ++session_calls;if(save || path!="good") throw std::invalid_argument("untrusted restore fixture");
            fed={6,7};mode=false;main->adopt(fed,logits(),mode);return ProtocolSessionResult{fed.size(),128};
        };
        std::ostringstream output;serve_batch_protocol(fd[0],output,*engine,3,32,128,control);::close(fd[0]);return output.str();
    }
};
void global_owners() {
    Fixture f;f.serve("GEN 1 temperature=0 cvec=1 10,11\nBGEN 0 2 temperature=0 cvec=0 17,18\n");
    const std::vector<int32_t> idle{17,18,2,5};need(bool(f.slots->select_prefix(idle,3,false)),"missing initial idle owner");
    const auto text=f.serve("BGEN 1 20 temperature=0 cvec=0 23,24\nBGEN 2 20 temperature=0 cvec=1 27,28\n"
        "GEN 1 temperature=0 cvec=1 30,29,28,27\nGEN 1 temperature=0 cvec=0 17,18,2,5\nBGEN 1 1 temperature=0 cvec=1 8,9\n");
    need(count(text,"ERR ")==0 && count(text,"FATAL ")==0 && text.find(" nonfinite ")!=std::string::npos,"numeric slot failure was not an ordinary nonfinite terminal");
    need(count(text,"DONE 1 4 ")==2 && count(text,"RESUME 3")==0,"foreign numerical failure damaged the in-flight main or reused pre-failure idle state");
    need(f.engine->flushes==1 && !f.main->nonfinite_pending(),"global failure not consumed exactly at the next request");
    need(count(text,"BDONE 2 20 length ")==1 && count(text,"BT 2 1")==19,"tainted running slot did not finish its unchanged reply");
    need(!f.slots->select_prefix(idle,3,false),"global flush kept an idle conversation");
    std::vector<int32_t> finished{27,28};finished.insert(finished.end(),19,1);finished.push_back(5);
    need(!f.slots->select_prefix(finished,finished.size()-1,true),"running slot became cache after its tainted reply");
}
void prompt_and_restore() {
    for(bool batch:{false,true}) {
        Fixture f;f.serve("GEN 1 temperature=0 10,11,12\n");f.poison_prompt=true;
        const auto failed=f.serve(batch?"BGEN 0 4 temperature=0 3,31\n":"GEN 4 temperature=0 31,3\n");
        need(count(failed,"T ")==0 && count(failed,"DONE 0 2 ")==1 && failed.find(" nonfinite ")!=std::string::npos && f.main->nonfinite_pending(),"nonfinite prompt published an admission token or missed global marking");
        need(count(failed,"BADM 0 0")==size_t(batch),"nonfinite admission left an active scheduler slot");
        const auto rejected=f.serve("SAVE ignored\nRESTORE bad\n");
        need(count(rejected,"SERR invalid 0 ")==2 && f.main->nonfinite_pending() && f.session_calls==1,"rejected disk operation consumed the pending poison or saved it");
        const auto next=f.serve("GEN 1 temperature=0 10,11,12,13\n");
        need(count(next,"RESUME 0")==1 && f.engine->flushes==1,"next prompt resumed a poisoned main/cache");
    }
    Fixture restored;restored.serve("BGEN 0 2 temperature=0 cvec=1 10,11\n");restored.poison_prompt=true;restored.serve("GEN 1 temperature=0 31,3\n");
    const auto text=restored.serve("RESTORE good\nGEN 1 temperature=0 cvec=0 6,7,8\n");
    need(count(text,"RESTORED 2 ")==1 && count(text,"RESUME 2")==1 && restored.engine->flushes==1 && !restored.main->nonfinite_pending(),"successful disk restore was flushed instead of preserving its new main");
    const std::vector<int32_t> idle{10,11,1,5};need(!restored.slots->select_prefix(idle,3,true),"disk recovery kept a pre-failure idle slot");
}
void fatal_windows() {
    for(bool in_main:{false,true}) {
        Fixture f;f.fatal=true;
        const auto text=f.serve(in_main?"BGEN 0 3 temperature=0 cvec=0 23,24\nGEN 2 temperature=0 cvec=1 30,29,28,27\nGEN 1 temperature=0 6,7\n":"BGEN 0 3 temperature=0 cvec=0 23,24\n");
        need(count(text,"FATAL ")==1 && count(text,"ERR batch slot ")==0 && count(text,"BDONE ")==0,"fatal native window was swallowed as a recoverable slot failure");
        if(in_main) need(count(text,"T ")==1 && count(text,"DONE ")==1,"protocol ran another request after a fatal interleaved window");
    }
}
}
int main() {
    try {global_owners();prompt_and_restore();fatal_windows();std::cout<<"PASS production numerical recovery: next-request global flush, live main during foreign failure, idle drop, running taint/no-recache, nonfinite admission/GEN, refused/successful disk recovery and fatal window termination; host fixtures only\n";}
    catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
