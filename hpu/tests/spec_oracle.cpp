// hpu/tests/spec_oracle.cpp - host test for the upstream speculative-decoding verification/diagnostic
// knobs --spec-oracle / --spec-follow / --spec-corrupt (authorities in src/program/generate.cpp:
// 559-584, 2085-2087, the read_ids lambda at 13514-13519, the refusals at 13521-13528, the follow cap
// at 13530, the corrupt counter at 13730/13803, the spec_q disabling at 13814, the window assembly at
// 13797-13804, the follow replacement at 13836-13842, the acceptance at 13843-13844, the emission and
// stats at 13877-13883 and the report at 13917-13919).
//
// (a) the oracle feeding the true continuation: every draft is accepted and the generated sequence
//     equals the reference sequence (plain greedy decode with no speculation at all);
// (b) --spec-corrupt N: exactly the corrupted positions are rejected - checked against an independent
//     round-by-round emulation of upstream's (++corrupt_counter % N)==0, d=(d+1)%n_vocab - and the
//     final sequence is STILL identical to (a), which is the point of the upstream exactness test;
// (c) the follow rule matches an independently transcribed copy of the upstream expressions over a
//     swept domain, asserted as a whole set, plus an end-to-end follow run;
// (d) each malformed oracle input is refused with its own message.
//
// No device, no Synapse: the mock target model is the deterministic (last token + 1) % vocabulary
// argmax chain used by the existing host tests.  Temporary token files live under the OS temp
// directory and are removed before exiting.
#include "strata/hpu/generation.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using namespace strata::hpu;
namespace {
void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
constexpr int32_t kV=5;          // the mock's vocabulary
constexpr uint32_t kRun=10;      // generated tokens per run
constexpr uint32_t kWindow=7;    // GenerationOptions::max_draft, the per-round draft count
std::filesystem::path g_dir;

void write_bytes(const std::string& name,const std::string& text) {
    std::ofstream out(g_dir/name,std::ios::binary|std::ios::trunc);
    out<<text;
    if(!out) throw std::runtime_error("cannot write test token file");
}
std::string path_of(const std::string& name) {return (g_dir/name).string();}
std::string ids_text(const std::vector<int32_t>& ids) {
    std::string text;
    for(size_t i=0;i<ids.size();++i) {if(i) text+=",";text+=std::to_string(ids[i]);}
    return text;
}
using DraftFn=std::function<std::vector<DraftRow>(const DraftContext&,uint32_t)>;
std::vector<DraftRow> rows_of(std::vector<int32_t> ids) {
    std::vector<DraftRow> rows;
    for(auto token:ids) rows.push_back({token,{},{}});
    return rows;
}
struct Run {std::vector<int32_t> emitted,fed;GenerationResult result;uint32_t native_windows=0;};
// One generation over the mock target: argmax is always (last fed token + 1) % kV, so the reference
// continuation of prompt {0,1} is 2,3,4,0,1,2,...  A zero draft window is plain decode.
Run run_model(const std::vector<int32_t>& prompt,uint32_t window,const DraftFn& draft,const SpecKnobRequest* knobs,
              bool native=false,VerifyMode mode=VerifyMode::Exact) {
    std::vector<int32_t> fed,emitted;
    uint32_t native_windows=0;
    GenerationOperations op;
    op.reset=[&]{fed.clear();};
    op.checkpoint=[&] {return std::function<void()>([&,saved=fed]{fed=saved;});};
    op.decode=[&](int32_t token,const std::function<bool()>&) {
        fed.push_back(token);std::vector<float> values(kV,-10.0f);values[uint32_t(token+1)%uint32_t(kV)]=10.0f;return values;
    };
    if(draft) op.draft=[&](const DraftContext& c,uint32_t maximum,const std::function<bool()>&) {return draft(c,maximum);};
    if(native) op.verify=[&](const DraftContext& c,std::span<const DraftRow> rows,const std::function<bool()>& stop) {
        ++native_windows;
        require(std::equal(fed.begin(),fed.end(),c.history.begin(),c.history.end()),"native follow saw a different physical prefix");
        std::vector<float> after;
        for(const auto& row:rows) {std::vector<float> values(kV,-10);values[uint32_t(row.token+1)%uint32_t(kV)]=10;after.insert(after.end(),values.begin(),values.end());}
        auto selected=select_verified_candidates(c.logits,after,rows,c.history,c.sampling,c.bias,c.mode,{},stop,c.follow);
        for(uint32_t at=0;at<selected.retained;++at) fed.push_back(rows[at].token);
        selected.logits.assign(kV,-10);selected.logits[uint32_t(fed.back()+1)%uint32_t(kV)]=10;
        return selected;
    };
    GenerationOptions options;options.max_draft=window;options.verify_mode=mode;
    Generation generator(kV,64,{},std::move(op),options);
    if(knobs) generator.set_spec_knobs(*knobs);
    GenerationRequest request;request.prompt=prompt;request.max_new=kRun;request.sampling.greedy=mode==VerifyMode::Exact;
    if(mode==VerifyMode::Probabilistic) {request.sampling.temperature=.7f;request.sampling.seed=19;}
    GenerationEvents events;events.token=[&](int32_t token){emitted.push_back(token);};
    Run out;out.result=generator.run(request,events);out.emitted=std::move(emitted);out.fed=std::move(fed);out.native_windows=native_windows;return out;
}
// An independent transcription of the round structure around src/program/generate.cpp:13752-13883:
// window assembly 13797-13804 (draft slot i of the round is `oracle[produced.size()-1+i]`, the file
// padded with token 0 past its end, 13801), the corruptor 13803, the acceptance 13843-13844 and the
// emission 13877-13883, over the port's maximum=min(window, max_new-generated-1).  `reference` is the
// target's own argmax at each emitted position (the chain).
struct Emulation {uint64_t windows=0,checked=0,accepted=0;std::vector<int32_t> emitted;};
Emulation emulate(const std::vector<int32_t>& reference,const std::vector<int32_t>& oracle,size_t window,int64_t corrupt) {
    Emulation e;size_t g=0;int64_t counter=0;
    while(g<reference.size()) {
        ++e.windows;
        const size_t base=g; // upstream's `produced.size()` at assembly time: fixed for the whole window (13799)
        const size_t rows=std::min(window,reference.size()-base-1);
        // Assembly pass (13798-13805): every slot of the window is built - and the corrupt counter fires
        // on every slot - BEFORE verification (13815), so slots past the round's first rejection still
        // advance the counter.
        std::vector<int32_t> assembled;assembled.reserve(rows);
        for(size_t r=0;r<rows;++r) {
            ++counter;
            int32_t draft=base+r<oracle.size()?oracle[base+r]:0; // the `: 0` of 13801
            if(corrupt>0&&counter%corrupt==0) draft=int32_t((int64_t(draft)+1)%int64_t(kV)); // 13803, exactly
            assembled.push_back(draft);
        }
        // Verification pass: the acceptance rule of 13843-13844 over the assembled window.
        for(size_t r=0;r<rows;++r) {
            ++e.checked;
            if(assembled[r]!=reference[base+r]) break; // rejected; the bonus row re-emits the target's own argmax
            ++e.accepted;e.emitted.push_back(reference[base+r]);++g;
        }
        if(g<reference.size()) {e.emitted.push_back(reference[g]);++g;} // the bonus row (upstream outv[a])
    }
    return e;
}
const char* issue_name(SpecTokenFileIssue issue) {
    switch(issue) {
        case SpecTokenFileIssue::Ok: return "ok";
        case SpecTokenFileIssue::Unreadable: return "unreadable";
        case SpecTokenFileIssue::NulByte: return "nul byte";
        case SpecTokenFileIssue::MalformedId: return "malformed id";
        case SpecTokenFileIssue::Empty: return "empty";
        case SpecTokenFileIssue::BelowWindow: return "below window";
        case SpecTokenFileIssue::OutsideVocabulary: return "out of vocabulary";
    }
    return "?";
}
// A second, independent transcription of the follow expressions, with upstream's own types:
// 13530:  max_new = follow.empty() ? o.max_new : std::min<int64_t>(o.max_new,(int64_t) follow.size())
// 13838-13841:  fdiff[i] = outv[i] != follow[k0+i]; outv[i] = follow[k0+i];   // rows past the file keep argmax
std::tuple<uint64_t,int32_t,bool> upstream_follow(uint64_t requested,const std::vector<int64_t>& follow,size_t emitted_index,int32_t argmax) {
    const uint64_t max_new=follow.empty()?requested:std::min(requested,uint64_t(follow.size()));
    int32_t token=argmax;bool differs=false;
    if(emitted_index<follow.size()) {differs=argmax!=int32_t(follow[emitted_index]);token=int32_t(follow[emitted_index]);}
    return {max_new,token,differs};
}
}
int main() {
    try {
        g_dir=std::filesystem::temp_directory_path()/"strata_spec_oracle_test";
        std::filesystem::remove_all(g_dir);std::filesystem::create_directories(g_dir);
        const std::vector<int32_t> prompt{0,1};
        // (a) reference = plain greedy decode, no speculation at all (draft window 0).
        const auto baseline=run_model(prompt,0,nullptr,nullptr);
        require(baseline.result.drafts_checked==0 && baseline.result.verify_windows==kRun,"baseline is one token per window");
        require(baseline.result.follow_emitted==0 && baseline.result.follow_differ==0,"baseline reports no follow statistics");
        const std::vector<int32_t>& reference=baseline.emitted;
        require(reference.size()==kRun,"reference length");
        write_bytes("oracle.ids",ids_text(reference));
        const std::string oracle_path=path_of("oracle.ids");
        SpecKnobRequest oracle{oracle_path,{},0};
        const auto plain=run_model(prompt,kWindow,nullptr,&oracle);
        require(plain.emitted==reference,"(a) the oracle's generated sequence equals the reference sequence");
        require(plain.result.drafts_checked>0 && plain.result.drafts_accepted==plain.result.drafts_checked,"(a) every draft accepted");
        const auto expected_a=emulate(reference,reference,kWindow,0);
        require(plain.result.verify_windows==expected_a.windows && plain.result.drafts_checked==expected_a.checked &&
                plain.result.drafts_accepted==expected_a.accepted,"(a) window/draft counts match the transcription");
        // The oracle replaces the draft source: a hostile draft hook must not be consulted.
        const DraftFn hostile=[](const DraftContext&,uint32_t maximum) {return rows_of(std::vector<int32_t>(maximum,3));};
        const auto overridden=run_model(prompt,kWindow,hostile,&oracle);
        require(overridden.emitted==reference && overridden.result.drafts_accepted==overridden.result.drafts_checked,"(a) the oracle replaces the draft source");
        // A perfect MTP-style drafter (no oracle) reproduces the same run: the default source is unchanged.
        const DraftFn argmax_chain=[](const DraftContext& c,uint32_t maximum) {
            int32_t next=0;for(size_t i=1;i<c.logits.size();++i) if(c.logits[i]>c.logits[next]) next=int32_t(i);
            std::vector<int32_t> ids;for(uint32_t r=0;r<maximum;++r) {if(r) next=(next+1)%kV;ids.push_back(next);}
            return rows_of(ids);
        };
        const auto drafter=run_model(prompt,kWindow,argmax_chain,nullptr);
        require(drafter.emitted==reference && drafter.result.verify_windows==expected_a.windows &&
                drafter.result.drafts_accepted==expected_a.accepted,"(a) the default draft source path is unchanged");
        // (b) --spec-corrupt N over the oracle AND over the drafter's own proposals: the sequence stays
        // identical to the no-speculation run, and the rejection pattern is exactly the corruption hits.
        for(const uint32_t corrupt:{1u,2u,3u}) {
            SpecKnobRequest knobs{oracle_path,{},int64_t(corrupt)};
            const auto run=run_model(prompt,kWindow,nullptr,&knobs);
            const auto e=emulate(reference,reference,kWindow,corrupt);
            require(run.emitted==reference,"(b) corrupt runs still produce the exact no-speculation sequence");
            require(run.result.verify_windows==e.windows && run.result.drafts_checked==e.checked &&
                    run.result.drafts_accepted==e.accepted,"(b) exactly the corrupted positions are rejected");
            require(run.result.drafts_checked>run.result.drafts_accepted,"(b) at least one draft rejected");
            if(corrupt==1) require(run.result.drafts_accepted==0,"(b) corrupt=1 rejects every draft");
            SpecKnobRequest on_drafter{{},{},int64_t(corrupt)};
            const auto drafter_run=run_model(prompt,kWindow,argmax_chain,&on_drafter);
            require(drafter_run.emitted==reference && drafter_run.result.drafts_checked==e.checked &&
                    drafter_run.result.drafts_accepted==e.accepted,"(b) the corruptor also gates the drafter's own proposals");
        }
        // (c) the follow rule: whole-set equality against the transcription over the swept domain, then
        // an end-to-end run whose drafts match the continuation instead of the argmax.
        std::set<std::tuple<uint64_t,int32_t,bool>> swept,transcribed;
        size_t points=0;
        const std::vector<std::vector<int64_t>> follows{{},{4},{0},{4,4,4},{0,1,2,3,4},{4,3,2,1,0},{2,2,2,2}};
        for(const auto& follow:follows) {
            const std::vector<int32_t> port(follow.begin(),follow.end());
            for(uint64_t requested=0;requested<=12;++requested)
                for(size_t index=0;index<9;++index)
                    for(int32_t argmax=0;argmax<kV;++argmax) {
                        ++points;
                        const auto row=spec_follow_row(port,index,argmax);
                        swept.insert({spec_follow_max_new(requested,port),row.token,row.differs});
                        transcribed.insert(upstream_follow(requested,follow,index,argmax));
                    }
        }
        require(points==7*13*9*static_cast<size_t>(kV) && swept==transcribed && swept.size()>=64,"(c) the follow rule matches the upstream transcription over the swept set");
        write_bytes("follow.ids","4,4,4");
        const DraftFn follow_drafts=[](const DraftContext&,uint32_t maximum) {return rows_of(std::vector<int32_t>(maximum,4));};
        SpecKnobRequest follow{{},path_of("follow.ids"),0};
        const auto followed=run_model(prompt,2,follow_drafts,&follow);
        require(followed.emitted==std::vector<int32_t>({4,4,4}),"(c) the run emits the follow continuation, capped at its length (13530)");
        require(followed.result.generated==3 && followed.result.drafts_accepted==2 && followed.result.drafts_checked==2,
                "(c) a draft is accepted when it matches the follow continuation (13844)");
        require(followed.result.follow_emitted==3 && followed.result.follow_differ==3,"(c) follow diff stats accumulate per emitted token (13882)");
        const auto native_followed=run_model(prompt,2,follow_drafts,&follow,true);
        require(native_followed.native_windows==1 && native_followed.emitted==followed.emitted && native_followed.fed==followed.fed &&
                native_followed.result.follow_differ==3 && native_followed.result.follow_emitted==3 && native_followed.result.drafts_accepted==2,
                "(c) native selector did not replace follow before accepted-prefix commit");
        write_bytes("native-follow.ids","4,0,2,2,3,4,1,0,0,3");
        for(auto mode:{VerifyMode::Exact,VerifyMode::Probabilistic}) for(uint32_t window=1;window<=7;++window) for(int64_t corrupt:{0,1,2,3}) {
            SpecKnobRequest native_knobs{oracle_path,path_of("native-follow.ids"),corrupt};
            const auto scalar=run_model(prompt,window,nullptr,&native_knobs,false,mode);
            const auto verified=run_model(prompt,window,nullptr,&native_knobs,true,mode);
            require(verified.native_windows>0 && verified.emitted==scalar.emitted && verified.fed==scalar.fed &&
                    verified.result.follow_differ==scalar.result.follow_differ && verified.result.follow_emitted==scalar.result.follow_emitted &&
                    verified.result.drafts_checked==scalar.result.drafts_checked && verified.result.drafts_accepted==scalar.result.drafts_accepted &&
                    verified.result.drafts_proposed==scalar.result.drafts_proposed && verified.result.verify_windows==scalar.result.verify_windows,
                    "native/scalar follow oracle/corrupt acceptance, publication statistics or final fed state differ");
        }
        // (d) each malformed oracle input is refused, each with its own message.
        write_bytes("empty.ids","");
        write_bytes("short.ids","2,3,4");
        write_bytes("vocab.ids","2,3,4,9");
        write_bytes("nul.ids",std::string("2\0 3",4));
        write_bytes("bad.ids","2,x,3");
        struct Refusal {const char* file;SpecTokenFileIssue issue;const char* needle;};
        const Refusal refusals[]={
            {"empty.ids",SpecTokenFileIssue::Empty,"token list was empty"},
            {"short.ids",SpecTokenFileIssue::BelowWindow,"fewer ids than the verify window"},
            {"vocab.ids",SpecTokenFileIssue::OutsideVocabulary,"outside the vocabulary"},
            {"nul.ids",SpecTokenFileIssue::NulByte,"NUL byte"},
            {"bad.ids",SpecTokenFileIssue::MalformedId,"invalid token id"},
            {"absent.ids",SpecTokenFileIssue::Unreadable,"cannot be read"},
        };
        std::set<std::string> messages;
        for(const auto& refusal:refusals) {
            const auto file=load_spec_tokens(path_of(refusal.file),uint32_t(kV),kWindow,"oracle");
            require(file.issue==refusal.issue,issue_name(refusal.issue));
            require(file.message.find(refusal.needle)!=std::string::npos,"refusal message names its reason");
            messages.insert(file.message);
        }
        require(messages.size()==6,"every malformed oracle input gets a distinct message");
        bool thrown=false;
        try {SpecKnobRequest bad{path_of("short.ids"),{},0};run_model(prompt,kWindow,nullptr,&bad);}
        catch(const std::invalid_argument& error) {thrown=error.what()!=nullptr;}
        require(thrown,"(d) Generation::set_spec_knobs refuses the malformed oracle before any device work");
        const auto good=load_spec_tokens(oracle_path,uint32_t(kV),kWindow,"oracle");
        require(good.issue==SpecTokenFileIssue::Ok && good.ids==reference,"the valid oracle file loads unchanged");
        // A native hook that ignores follow must fail before any publication;
        // allowing configuration cannot mean silently dropping the override.
        bool native_refused=false;std::vector<int32_t> invalid_output;
        try {
            std::vector<int32_t> fed;GenerationOperations op;
            op.reset=[&]{fed.clear();};
            op.decode=[&](int32_t token,const std::function<bool()>&) {fed.push_back(token);std::vector<float> v(kV,-10.0f);v[uint32_t(token+1)%uint32_t(kV)]=10.0f;return v;};
            op.verify=[](const DraftContext&,std::span<const DraftRow>,const std::function<bool()>&) {
                VerifiedCandidates c;c.picks={{0,false}};c.logits=std::vector<float>(kV,0.0f);return c;
            };
            Generation generator(kV,64,{},std::move(op),GenerationOptions{});
            SpecKnobRequest knobs{{},path_of("follow.ids"),0};generator.set_spec_knobs(knobs);
            // Force a real proposal so the deliberately invalid native hook runs.
            knobs.oracle_path=oracle_path;generator.set_spec_knobs(knobs);
            GenerationRequest request;request.prompt=prompt;request.max_new=3;GenerationEvents events;
            events.token=[&](int32_t token){invalid_output.push_back(token);};generator.run(request,events);
        } catch(const SessionFatal&) {native_refused=true;}
        require(native_refused && invalid_output.empty(),"(d) native hook silently ignored follow or published invalid committed picks");
        std::filesystem::remove_all(g_dir);
        std::cout << "spec_oracle: PASS oracle exactness, corrupt positions exactly rejected, follow rule swept, malformed inputs refused; no device" << std::endl;
        return 0;
    } catch(const std::exception& error) {
        std::filesystem::remove_all(g_dir);
        std::cerr << "spec_oracle: FAIL " << error.what() << std::endl;
        return 1;
    }
}
