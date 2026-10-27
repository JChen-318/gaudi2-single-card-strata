#include "strata/hpu/spec_draft_source.hpp"
#include <algorithm>
#include <charconv>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace strata::hpu {
namespace {
std::string refuse(SpecTokenFileIssue issue,const std::string& kind,const std::string& path) {
    switch(issue) {
        case SpecTokenFileIssue::Unreadable: return "spec "+kind+" file cannot be read: "+path;   // upstream: "cannot read --spec-oracle/-follow <path>" (13522,13526)
        case SpecTokenFileIssue::NulByte: return "spec "+kind+" file contains a NUL byte: "+path;
        case SpecTokenFileIssue::MalformedId: return "spec "+kind+" file has an invalid token id: expected an integer in [0, 2147483647]: "+path; // upstream's wording, 975
        case SpecTokenFileIssue::Empty: return "spec "+kind+" file is empty: token list was empty: "+path; // upstream's wording, 981
        case SpecTokenFileIssue::BelowWindow: return "spec "+kind+" file holds fewer ids than the verify window: "+path;
        case SpecTokenFileIssue::OutsideVocabulary: return "spec "+kind+" file holds a token id outside the vocabulary: "+path;
        case SpecTokenFileIssue::Ok: break;
    }
    return {};
}
}
SpecTokenFile load_spec_tokens(const std::string& path,uint32_t vocabulary,size_t minimum_count,const std::string& kind) {
    SpecTokenFile result;
    std::ifstream in(path,std::ios::binary);
    const std::string text((std::istreambuf_iterator<char>(in)),std::istreambuf_iterator<char>());
    if(!in) { result.issue=SpecTokenFileIssue::Unreadable;result.message=refuse(result.issue,kind,path);return result; }
    // Upstream hands the raw text to parse_i64_list (965-983); a NUL byte there fails from_chars and
    // reports the generic invalid-id error. The port names it first, per the diagnostic's demand.
    if(text.find('\0')!=std::string::npos) { result.issue=SpecTokenFileIssue::NulByte;result.message=refuse(result.issue,kind,path);return result; }
    // parse_i64_list verbatim in its accepted grammar: commas become spaces, whitespace-separated
    // tokens, each a non-negative int32 fully consumed.
    std::string scrubbed=text;
    for(char& c:scrubbed) if(c==',') c=' ';
    std::istringstream input(scrubbed);
    std::string token;
    while(input>>token) {
        int32_t id=0;
        const auto parsed=std::from_chars(token.data(),token.data()+token.size(),id);
        if(parsed.ec!=std::errc{}||parsed.ptr!=token.data()+token.size()||id<0) {
            result.issue=SpecTokenFileIssue::MalformedId;result.message=refuse(result.issue,kind,path);result.ids.clear();return result;
        }
        if(vocabulary&&uint32_t(id)>=vocabulary) { // upstream only bounds at 2^31 here; validate_drafts (verify.cpp:11) is its port-side gate
            result.issue=SpecTokenFileIssue::OutsideVocabulary;result.message=refuse(result.issue,kind,path);result.ids.clear();return result;
        }
        result.ids.push_back(id);
    }
    if(result.ids.empty()) { result.issue=SpecTokenFileIssue::Empty;result.message=refuse(result.issue,kind,path);return result; }
    if(minimum_count&&result.ids.size()<minimum_count) { result.issue=SpecTokenFileIssue::BelowWindow;result.message=refuse(result.issue,kind,path);result.ids.clear();return result; }
    result.issue=SpecTokenFileIssue::Ok;return result;
}
SpecCorruptor::SpecCorruptor(int64_t every,int32_t vocabulary):every_(every>0?every:0),vocabulary_(vocabulary) {}
bool SpecCorruptor::next(int32_t& draft) {
    if(!active()) return false;
    // src/program/generate.cpp:13803 exactly: (++corrupt_counter % o.spec_corrupt) == 0, then
    // d = (d + 1) % (int32_t) n_vocab. The counter advances even when the slot is never verified,
    // because assembly (13798-13805) precedes verification (13815).
    ++counter_;
    if(counter_%every_!=0||vocabulary_<=0) return false; // vocabulary_<=0: nothing is in range; refuse to invent a token
    draft=int32_t((int64_t(draft)+1)%int64_t(vocabulary_)); // upstream's int32 arithmetic, widened to avoid d+1 overflow at INT32_MAX
    return true;
}
std::vector<int32_t> spec_oracle_window(const std::vector<int32_t>& oracle,size_t base,size_t rows) {
    std::vector<int32_t> drafts(rows);
    for(size_t r=0;r<rows;++r) { const size_t at=base+r; drafts[r]=at<oracle.size()?oracle[at]:0; } // the `: 0` of 13801
    return drafts;
}
uint64_t spec_follow_max_new(uint64_t requested,const std::vector<int32_t>& follow) {
    return follow.empty()?requested:std::min(requested,uint64_t(follow.size())); // 13530
}
SpecFollowRow spec_follow_row(const std::vector<int32_t>& follow,size_t emitted_index,int32_t argmax) {
    // 13838-13841: rows with no continuation entry keep the window's argmax and record no difference.
    if(emitted_index>=follow.size()) return {argmax,false};
    const int32_t value=follow[emitted_index];
    return {value,argmax!=value};
}
void SpecKnobs::configure(const SpecKnobRequest& request,uint32_t vocabulary,uint32_t window) {
    if(request.corrupt<0) throw std::invalid_argument("spec --spec-corrupt expects N >= 0"); // upstream: int, only N>0 arms the corruptor (13803)
    oracle_.clear();follow_.clear();corruptor_=SpecCorruptor(request.corrupt,int32_t(vocabulary));
    if(!request.oracle_path.empty()) {
        // Upstream never bounds the oracle by the window; it pads with token 0 past the file (13801).
        // The port refuses a file that cannot fill even one draft window up front (declared); at
        // runtime exhaustion still yields token 0, verbatim.
        const auto file=load_spec_tokens(request.oracle_path,vocabulary,window,"oracle");
        if(file.issue!=SpecTokenFileIssue::Ok) throw std::invalid_argument(file.message);
        oracle_=file.ids;
    }
    if(!request.follow_path.empty()) {
        const auto file=load_spec_tokens(request.follow_path,vocabulary,0,"follow");
        if(file.issue!=SpecTokenFileIssue::Ok) throw std::invalid_argument(file.message);
        follow_=file.ids;
    }
}
}
