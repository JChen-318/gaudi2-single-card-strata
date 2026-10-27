#pragma once
#include <charconv>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace strata::hpu::detail {
struct PackProvenance {uint64_t schema=0,bytes=0;std::string digest;std::vector<std::pair<std::string,uint64_t>> shards;};
// Only the fixed schema written by iq_pack.py; bounded input and no recursive
// generic JSON tree. Reject duplicate/unknown fields and inexact numbers.
class PackJson {
    std::string_view input_;size_t at_=0;
    [[noreturn]] static void bad() {throw std::invalid_argument("malformed native expert provenance JSON");}
    void space() {while(at_<input_.size() && (input_[at_]==' ' || input_[at_]=='\r' || input_[at_]=='\n' || input_[at_]=='\t')) ++at_;}
    bool take(char c) {space();if(at_<input_.size() && input_[at_]==c) {++at_;return true;}return false;}
    void need(char c) {if(!take(c)) bad();}
    uint32_t hex4() {
        uint32_t value=0;for(int i=0;i<4;++i) {
            if(at_==input_.size()) bad();
            const auto c=input_[at_++];unsigned digit=16;
            if(c>='0' && c<='9') digit=unsigned(c-'0');
            else if(c>='a' && c<='f') digit=unsigned(c-'a')+10;
            else if(c>='A' && c<='F') digit=unsigned(c-'A')+10;
            if(digit==16) bad();
            value=(value<<4)|digit;
        }return value;
    }
    static void utf8(std::string& out,uint32_t c) {
        if(c<0x80) out+=char(c);
        else if(c<0x800) {out+=char(0xc0|(c>>6));out+=char(0x80|(c&63));}
        else if(c<0x10000) {out+=char(0xe0|(c>>12));out+=char(0x80|((c>>6)&63));out+=char(0x80|(c&63));}
        else {out+=char(0xf0|(c>>18));out+=char(0x80|((c>>12)&63));out+=char(0x80|((c>>6)&63));out+=char(0x80|(c&63));}
    }
    std::string string() {
        need('"');std::string out;
        while(at_<input_.size()) {
            const auto c=uint8_t(input_[at_++]);
            if(c=='"') {if(out.size()>4096) bad();return out;}
            if(c<32 || out.size()>4096) bad();
            if(c!='\\') {out+=char(c);continue;}
            if(at_==input_.size()) bad();
            switch(input_[at_++]) {
            case '"':out+='"';break;case '\\':out+='\\';break;case '/':out+='/';break;
            case 'b':out+='\b';break;case 'f':out+='\f';break;case 'n':out+='\n';break;case 'r':out+='\r';break;case 't':out+='\t';break;
            case 'u': {
                auto code=hex4();
                if(code>=0xd800 && code<=0xdbff) {
                    if(at_+2>input_.size() || input_[at_]!='\\' || input_[at_+1]!='u') bad();
                    at_+=2;
                    const auto low=hex4();if(low<0xdc00 || low>0xdfff) bad();code=0x10000+((code-0xd800)<<10)+low-0xdc00;
                } else if(code>=0xdc00 && code<=0xdfff) bad();
                utf8(out,code);break;
            }
            default:bad();
            }
        }bad();
    }
    uint64_t number() {
        space();const auto start=at_;while(at_<input_.size() && input_[at_]>='0' && input_[at_]<='9') ++at_;
        if(start==at_ || (at_-start>1 && input_[start]=='0')) bad();
        uint64_t out=0;const auto [end,error]=std::from_chars(input_.data()+start,input_.data()+at_,out);
        if(error!=std::errc{} || end!=input_.data()+at_) bad();
        return out;
    }
    template<class F> void object(F field) {
        need('{');if(take('}')) return;std::set<std::string> seen;
        do {const auto name=string();if(!seen.insert(name).second) bad();need(':');field(name);} while(take(','));need('}');
    }
public:
    explicit PackJson(std::string_view input):input_(input) {if(input.empty() || input.size()>65536) bad();}
    PackProvenance parse() {
        PackProvenance out;bool schema=false,bytes=false,digest=false,shards=false;
        object([&](const std::string& name) {
            if(name=="schema") {out.schema=number();schema=true;}
            else if(name=="bytes") {out.bytes=number();bytes=true;}
            else if(name=="native_experts_sha256") {out.digest=string();digest=true;}
            else if(name=="shards") {
                shards=true;need('[');
                if(!take(']')) {
                    do {
                        if(out.shards.size()>=4096) bad();
                        std::pair<std::string,uint64_t> shard;bool have_name=false,have_size=false;
                        object([&](const std::string& key) {
                            if(key=="name") {shard.first=string();have_name=true;}
                            else if(key=="size") {shard.second=number();have_size=true;}
                            else bad();
                        });
                        if(!have_name || !have_size || shard.first.empty() || !shard.second) bad();
                        out.shards.push_back(std::move(shard));
                    } while(take(','));need(']');
                }
            } else bad();
        });
        space();if(at_!=input_.size() || !schema || !bytes || !digest || !shards || out.schema!=1 || !out.bytes || out.shards.empty() || out.digest.size()!=64) bad();
        for(char c:out.digest) if(!((c>='0' && c<='9') || (c>='a' && c<='f'))) bad();
        return out;
    }
};
}
