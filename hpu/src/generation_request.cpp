#include "strata/hpu/protocol.hpp"
#include "strata/hpu/spec_gate.hpp"
#include "strata/core/logit_bias.hpp"
#include <array>
#include <charconv>
#include <cmath>
#include <set>
#include <sstream>

namespace strata::hpu {
namespace {
template<class T> T number(const std::string& text) {
    T value{};const auto [end,error]=std::from_chars(text.data(),text.data()+text.size(),value);
    if(error!=std::errc{} || end!=text.data()+text.size()) throw std::invalid_argument("invalid request number: "+text);
    return value;
}
// Upstream #2228: img_keys=<hex>,<hex>,... A malformed list is IGNORED (the picture then falls back to its
// embedding hash, generate.cpp:10700-10707), so this returns empty instead of throwing.
std::vector<uint64_t> image_keys(const std::string& text) {
    std::vector<uint64_t> keys;
    size_t at=0;
    while(at<text.size()) {
        const auto comma=text.find(',',at);
        const auto part=text.substr(at,comma==std::string::npos?std::string::npos:comma-at);
        uint64_t value{};
        const auto [end,error]=std::from_chars(part.data(),part.data()+part.size(),value,16);
        if(part.empty() || error!=std::errc{} || end!=part.data()+part.size()) return {};
        keys.push_back(value);
        if(comma==std::string::npos) break;
        at=comma+1;
    }
    return keys;
}
}
GenerationRequest parse_generation_request(const std::string& line,uint32_t vocabulary,uint32_t context,VisionLoad vision) {
    if(line.size()>uint64_t(context)*12+65536) throw std::invalid_argument("request line too long");
    std::istringstream input(line);std::string verb,max_new,field;
    if(!(input>>verb>>max_new) || (verb!="GEN" && verb!="GENI"))
        throw std::invalid_argument("expected GEN max_new [sampling keys] comma-separated-ids");
    GenerationRequest request;request.max_new=number<uint32_t>(max_new);request.sampling.temperature=0;request.sampling.top_p=1;
    std::set<std::string> seen;std::string next;std::vector<uint64_t> keys;
    while(input>>field) {
        const auto split=field.find('=');
        if(split==std::string::npos) {next=field;break;}
        const auto key=field.substr(0,split),value=field.substr(split+1);
        if(!seen.insert(key).second) throw std::invalid_argument("duplicate sampling key: "+key);
        auto& p=request.sampling;
        if(key=="temperature") p.temperature=number<float>(value);
        else if(key=="top_p") p.top_p=number<float>(value);
        else if(key=="top_k") p.top_k=number<int>(value);
        else if(key=="min_p") p.min_p=number<float>(value);
        else if(key=="penalty_last_n") p.penalty_last_n=number<int>(value);
        else if(key=="penalty_repeat") p.penalty_repeat=number<float>(value);
        else if(key=="penalty_freq") p.penalty_freq=number<float>(value);
        else if(key=="penalty_present") p.penalty_present=number<float>(value);
        else if(key=="seed") p.seed=number<uint64_t>(value);
        else if(key=="ckpt") {const auto flag=number<uint32_t>(value);if(flag>1) throw std::invalid_argument("ckpt expects 0 or 1");request.checkpoint=flag!=0;}
        else if(key=="pin") request.pin=number<uint32_t>(value);
        else if(key=="cvec") {const auto flag=number<uint32_t>(value);if(flag>1) throw std::invalid_argument("cvec expects 0 or 1");request.control_vector=flag!=0;}
        else if(key=="logit_bias") {std::string error;if(!core::parse_logit_bias(value,int(vocabulary),request.bias,error)) throw std::invalid_argument("logit_bias: "+error);}
        else if(key=="img_keys") keys=image_keys(value);
        // Upstream's two speculative request keys (:10697-10698). `spec_min_p` is CLAMPED into [0,1], not refused, and
        // `aux_cpus` is a 0|1 switch for this request's thread placement; the gate law itself is strata/hpu/spec_gate.hpp.
        else if(key=="spec_min_p") request.spec_min_p=clamp_spec_min_p(number<double>(value));
        else if(key=="aux_cpus") request.aux_cpus=number<int32_t>(value)!=0?1:0;
        // Upstream's `pcie_frac` (:10696) sets the share of the host->device transfer pipeline for this request. This
        // backend loads its weights to HBM once and has no such pipeline, so the key is refused BY NAME - silence here
        // would let a client believe a transfer split was honoured when nothing could have moved.
        else if(key=="pcie_frac") throw std::invalid_argument(
            "pcie_frac is not implemented on a single-card backend: it asks for the host->device transfer split, "
            "which this engine has no pipeline to divide");
        else throw std::invalid_argument("HPU request key is not implemented: "+key);
    }
    if(verb=="GENI") {
        // Upstream's grammar (:10635-10661): GENI <max_new> [keys] <embeddings file> <id,id,...>, and a
        // GENI on an engine started without --vision is refused before the file is touched.
        if(!vision.enabled) throw std::invalid_argument("this engine was started without --vision");
        std::string token_list;
        if(next.empty() || !(input>>token_list) || input>>field)
            throw std::invalid_argument("missing IDs or trailing request fields");
        // The table's cell count is the engine's max_context + 64, exactly upstream's mrope_host size.
        if(uint64_t(context)+64>uint64_t(INT32_MAX)/3) throw std::invalid_argument("image position table exceeds the int32 cell range");
        request.embeddings=std::make_shared<VisionEmbeddings>(VisionEmbeddings::load(next,vision.width));
        next=token_list;
    }
    if(next.empty() || input>>field) throw std::invalid_argument("missing IDs or trailing request fields");
    size_t start=0;
    while(start<next.size()) {
        const auto end=next.find(',',start);const auto text=next.substr(start,end==std::string::npos?end:end-start);
        const auto token=number<int32_t>(text);
        if(token<0 || uint32_t(token)>=vocabulary || request.prompt.size()>=context) throw std::invalid_argument("request token/capacity mismatch");
        request.prompt.push_back(token);
        if(end==std::string::npos) break;
        start=end+1;if(start==next.size()) throw std::invalid_argument("trailing token comma");
    }
    if(verb=="GENI") {
        // The planner needs the ids; rebuild the plan now that they exist (the load above already validated
        // the file and its exact sentences are the ones the caller prints).
        auto plan=plan_vision_prompt(request.prompt,*request.embeddings,uint64_t(context)+64,keys);
        request.image_rows=std::move(plan.rows);
        request.positions=std::move(plan.positions);
        request.images.clear();
        for(const auto& image:plan.images) request.images.push_back({image.start,image.key});
    }
    if(!request.max_new || request.prompt.empty() || request.max_new>context-request.prompt.size()) throw std::invalid_argument("request max_new exceeds context");
    if(request.pin && uint64_t(request.pin)+1>=request.prompt.size()) throw std::invalid_argument("pinned prefix must end before the prompt's last token");
    const std::array<float,1> dummy{0};(void)sampling_distribution(dummy,{},request.sampling);
    return request;
}
}
