#include "strata/hpu/native_pack.hpp"
#include "strata/hpu/expert_scales.hpp"
#include "mapped_file.hpp"
#include "native_pack_json.hpp"
#include "sha256.hpp"
#include "weight_format.hpp"
#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>

namespace strata::hpu {
static_assert(std::endian::native==std::endian::little,"native expert pack requires little-endian host");
namespace {
uint64_t integer(const std::string& text) {
    uint64_t value=0;const auto [end,error]=std::from_chars(text.data(),text.data()+text.size(),value);
    if(text.empty() || error!=std::errc{} || end!=text.data()+text.size()) throw std::invalid_argument("invalid native expert index integer");
    return value;
}
uint64_t add(uint64_t a,uint64_t b) {if(b>UINT64_MAX-a) throw std::overflow_error("native expert layout overflow");return a+b;}
std::string text_file(const std::string& path) {
    const auto bytes=std::filesystem::file_size(path);
    if(!bytes || bytes>65536) throw std::invalid_argument("native expert metadata exceeds 64 KiB bound");
    std::ifstream file(path,std::ios::binary);std::string text(size_t(bytes),'\0');file.read(text.data(),std::streamsize(bytes));
    if(size_t(file.gcount())!=text.size() || file.peek()!=std::char_traits<char>::eof() || text.find('\0')!=std::string::npos)
        throw std::runtime_error("native expert metadata changed or contains NUL");
    return text;
}
std::vector<std::string> shard_names(const std::string& value,uint32_t version) {
    std::vector<std::string> out;size_t begin=0;
    do {const auto end=value.find(',',begin);out.push_back(value.substr(begin,end==std::string::npos?end:end-begin));if(end==std::string::npos) break;begin=end+1;} while(true);
    if(out.size()==1) {const auto name=out.front();out.assign(3,name);}
    else if(version<4 || out.size()!=3) throw std::invalid_argument("native expert shard column requires v4 gate,up,down");
    for(const auto& name:out) if(name.find_first_of("/\\:\0",0,4)!=std::string::npos || name=="." || name=="..") throw std::invalid_argument("native expert shard must be a filename beside the main GGUF");
    return out;
}
}
struct NativeExpertPack::Impl {
    struct Layer {uint64_t offset=0,bytes=0,up=0,down=0,tail=0;};
    NativePackCatalog catalog;std::vector<Layer> layout;std::shared_ptr<detail::ReadOnlyFile> binary;
    std::vector<std::pair<std::string,uint64_t>> metadata;
    std::string identity() const {
        std::string out=binary?"native-expert-pack-v1|binary|":"native-expert-pack-v1|gguf|";
        for(const auto& file:metadata) out+=detail::local_identity(file.first,file.second);
        if(binary) out+=detail::local_identity(binary->path,binary->bytes);
        for(const auto& shard:catalog.shards) out+=detail::local_identity(shard.first,shard.second);
        if(out.size()>65536) throw std::invalid_argument("native expert source identity exceeds snapshot bound");
        return out;
    }
    const Layer& get(uint32_t layer,uint32_t expert) const {
        if(layer>=layout.size() || expert>=catalog.experts) throw std::out_of_range("native expert layer/ID");
        return layout[layer];
    }
};
NativeExpertPack::NativeExpertPack(const std::string& directory,NativePackCatalog catalog) {
    if(!catalog.experts || catalog.experts>4096 || catalog.layers.empty() || catalog.layers.size()>512 ||
       catalog.shards.empty() || catalog.shards.size()>4096 || !catalog.gguf_expert ||
       uint64_t(catalog.layers.size())*catalog.experts>65536) throw std::invalid_argument("invalid bounded native expert catalog");
    auto p=std::make_shared<Impl>();p->catalog=std::move(catalog);auto& c=p->catalog;p->layout.resize(c.layers.size());
    std::set<std::string> names;uint64_t path_bytes=0;
    for(auto& shard:c.shards) {
        shard.first=std::filesystem::canonical(shard.first).string();path_bytes=add(path_bytes,shard.first.size());
        if(path_bytes>65536 || !names.insert(std::filesystem::path(shard.first).filename().string()).second)
            throw std::invalid_argument("duplicate/oversized native expert source filenames");
        (void)detail::local_identity(shard.first,shard.second);
    }
    uint64_t total=0;
    for(size_t l=0;l<c.layers.size();++l) {
        auto& item=c.layers[l];auto& target=p->layout[l];target.offset=total;
        const auto h=item.roles[0].columns,ff=item.roles[0].rows;
        if(!h || !ff || h>65536 || ff>65536 || item.roles[1].type!=item.roles[0].type) throw std::invalid_argument("native expert catalog geometry/type mismatch");
        for(size_t r=0;r<3;++r) {
            auto& role=item.roles[r];role.path=std::filesystem::canonical(role.path).string();
            path_bytes=add(path_bytes,role.path.size());if(path_bytes>(1ull<<20)) throw std::invalid_argument("native expert role paths exceed bounded metadata budget");
            if(role.columns!=(r<2?h:ff) || role.rows!=(r<2?ff:h)) throw std::invalid_argument("native expert role axes mismatch");
            const auto format=detail::format(role.type);
            if(!detail::convertible(role.type) || !format.elements || role.columns%format.elements ||
               role.bytes!=checked_product(checked_product(role.columns/format.elements,format.bytes),role.rows))
                throw std::invalid_argument("native expert catalog row bytes mismatch");
            const auto shard=std::find_if(c.shards.begin(),c.shards.end(),[&](const auto& file){return file.first==role.path;});
            if(shard==c.shards.end()) throw std::invalid_argument("native expert role is outside main GGUF family");
            check_range(shard->second,role.offset,checked_product(role.bytes,c.experts));
        }
        target.up=item.roles[0].bytes;target.down=add(target.up,item.roles[1].bytes);
        target.bytes=add(target.down,item.roles[2].bytes);
        const bool nv=item.roles[0].type==40 || item.roles[2].type==40;
        if(nv) {target.tail=target.bytes;target.bytes=add(target.bytes,16);}
        if(target.bytes>(64ull<<20)) throw std::invalid_argument("native expert blob exceeds bounded 64 MiB cache-load policy");
        if(item.scales.empty() && !nv) item.scales.assign(c.experts,{1,1,1});
        if(item.scales.size()!=c.experts) throw std::invalid_argument("native expert scale catalog length mismatch");
        for(const auto& scales:item.scales) for(size_t r=0;r<3;++r)
            if(!std::isfinite(scales[r]) || scales[r]<=0 || (item.roles[r].type!=40 && scales[r]!=1)) throw std::invalid_argument("native expert invalid scale catalog");
        total=add(total,checked_product(target.bytes,c.experts));
    }
    const auto root=std::filesystem::canonical(directory);
    const auto index=std::filesystem::canonical(root/"native_experts.txt").string();
    p->metadata.emplace_back(index,std::filesystem::file_size(index));const auto index_identity=p->identity();const auto text=text_file(index);
    std::istringstream input(text);std::string line;std::vector<bool> seen(c.layers.size());uint32_t version=0;bool header=false;
    std::string primary=c.shards.front().first;
    const bool have_binary=std::filesystem::exists(root/"experts.bin");
    while(std::getline(input,line)) {
        if(!line.empty() && line.back()=='\r') line.pop_back();
        if(line.empty()) continue;
        if(line[0]=='#') {
            constexpr std::string_view tag="# strata native experts v";
            if(line.starts_with(tag)) {
                if(header) throw std::invalid_argument("duplicate native expert version header");
                header=true;
                const auto end=line.find(':',tag.size());if(end==std::string::npos) throw std::invalid_argument("invalid native expert version header");
                const auto number=integer(line.substr(tag.size(),end-tag.size()));
                if(number<1 || number>4) throw std::invalid_argument("unsupported native expert manifest version");
                version=uint32_t(number);
                constexpr std::string_view source_tag="; absolute offsets in ";
                if(const auto pos=line.find(source_tag);pos!=std::string::npos) {
                    const auto begin=pos+source_tag.size(),stop=line.find(", or in ",begin);
                    if(stop!=std::string::npos) {
                        const auto name=line.substr(begin,stop-begin);
                        const auto source=std::find_if(c.shards.begin(),c.shards.end(),[&](const auto& file){return std::filesystem::path(file.first).filename().string()==name;});
                        if(source==c.shards.end()) throw std::invalid_argument("native expert manifest primary shard is outside main GGUF family");
                        primary=source->first;
                    }
                }
                for(const auto& [marker,expected]:{std::pair<std::string,uint64_t>{"(n_expert ",c.experts},{", total ",total}}) {
                    if(const auto pos=line.find(marker);pos!=std::string::npos) {
                        const auto begin=pos+marker.size(),stop=line.find_first_not_of("0123456789",begin);
                        if(integer(line.substr(begin,stop-begin))!=expected) throw std::invalid_argument("native expert header count/total mismatch");
                    }
                }
            }
            continue;
        }
        std::istringstream fields(line);std::vector<std::string> tokens;std::string token;
        while(fields>>token) {tokens.push_back(token);if(tokens.size()>9) break;}
        if(tokens.size()!=5 && tokens.size()!=8 && tokens.size()!=9) throw std::invalid_argument("malformed native expert manifest row");
        const auto l=integer(tokens[0]);if(l>=c.layers.size() || seen[l]) throw std::invalid_argument("duplicate/out-of-range native expert layer");seen[l]=true;
        const auto& item=c.layers[l];const auto& target=p->layout[l];
        if(integer(tokens[1])!=item.roles[0].type || integer(tokens[2])!=item.roles[2].type ||
           integer(tokens[3])!=target.offset || integer(tokens[4])!=target.bytes) throw std::invalid_argument("native expert manifest differs from main GGUF layout");
        if(tokens.size()==5) {
            if(!have_binary) throw std::invalid_argument("native expert manifest has no source offsets; experts.bin required");
        } else {
            const auto files=shard_names(tokens.size()==9?tokens[8]:"",version);
            const auto parent=std::filesystem::path(primary).parent_path();
            for(size_t r=0;r<3;++r) {
                const auto path=files[r].empty()?primary:std::filesystem::canonical(parent/files[r]).string();
                if(path!=item.roles[r].path || integer(tokens[r+5])!=item.roles[r].offset) throw std::invalid_argument("native expert role shard/absolute offset differs from main GGUF");
            }
        }
    }
    if(std::find(seen.begin(),seen.end(),false)!=seen.end()) throw std::invalid_argument("native expert manifest misses a layer");
    if(p->identity()!=index_identity) throw std::runtime_error("native expert metadata/source changed during parse");
    if(have_binary) {
        const auto provenance=std::filesystem::canonical(root/"experts.bin.src.json").string();
        p->metadata.emplace_back(provenance,std::filesystem::file_size(provenance));
        const auto before=p->identity();const auto source=detail::PackJson(text_file(provenance)).parse();
        if(source.bytes!=total || source.digest!=detail::sha256(text) || source.shards.size()!=c.shards.size()) throw std::invalid_argument("native expert binary provenance does not match layout/source");
        for(size_t i=0;i<c.shards.size();++i) if(source.shards[i].first!=std::filesystem::path(c.shards[i].first).filename().string() || source.shards[i].second!=c.shards[i].second)
            throw std::invalid_argument("native expert binary provenance shard name/size mismatch");
        if(p->identity()!=before) throw std::runtime_error("native expert provenance changed during read");
        if(std::filesystem::file_size(root/"experts.bin")!=total) throw std::invalid_argument("native experts.bin size mismatch");
        p->binary=std::make_shared<detail::ReadOnlyFile>((root/"experts.bin").string());
        if(p->binary->bytes!=total) throw std::invalid_argument("native experts.bin size mismatch");
        const auto loaded=p->identity();
        // Inspect only scale tails, not all weight payloads. Compare against the
        // main GGUF's scales before any device acquisition or cache eviction.
        for(size_t l=0;l<p->layout.size();++l) if(const auto& layout=p->layout[l];layout.tail) for(uint32_t e=0;e<c.experts;++e) {
            const auto tail=p->binary->range(layout.offset+uint64_t(e)*layout.bytes+layout.tail,16);std::array<float,4> values{};std::memcpy(values.data(),tail.data(),16);
            for(size_t r=0;r<3;++r) if(!std::isfinite(values[r]) || values[r]!=c.layers[l].scales[e][r]) throw std::invalid_argument("native expert binary scale tail differs from main GGUF");
            if(std::bit_cast<uint32_t>(values[3])!=0) throw std::invalid_argument("native expert scale tail padding is not zero");
        }
        if(p->identity()!=loaded) throw std::runtime_error("native expert binary changed during validation");
    }
    impl_=std::move(p);
}
std::array<WeightSlice,3> NativeExpertPack::slices(uint32_t layer,uint32_t expert) const {
    const auto& p=*impl_;const auto& layout=p.get(layer,expert);
    if(!p.binary) return p.catalog.gguf_expert(layer,expert);
    std::array<WeightSlice,3> result;const std::array<uint64_t,3> offsets{0,layout.up,layout.down};
    for(size_t r=0;r<3;++r) {const auto& role=p.catalog.layers[layer].roles[r];result[r]=WeightSlice::raw_rows(p.binary,
        p.binary->range(layout.offset+uint64_t(expert)*layout.bytes+offsets[r],role.bytes),role.type,role.columns,role.rows);}
    return result;
}
std::array<float,3> NativeExpertPack::scales(uint32_t layer,uint32_t expert) const {(void)impl_->get(layer,expert);return impl_->catalog.layers[layer].scales[expert];}
std::vector<uint8_t> NativeExpertPack::native_blob(uint32_t layer,uint32_t expert) const {
    const auto& p=*impl_;const auto& layout=p.get(layer,expert);
    if(p.binary) {const auto blob=p.binary->range(layout.offset+uint64_t(expert)*layout.bytes,layout.bytes);return {blob.begin(),blob.end()};}
    return native_expert_blob(slices(layer,expert),layout.up,layout.down,layout.bytes,layout.tail,scales(layer,expert));
}
uint64_t NativeExpertPack::blob_bytes(uint32_t layer) const {return impl_->get(layer,0).bytes;}
bool NativeExpertPack::has_binary() const {return bool(impl_->binary);}
std::string NativeExpertPack::file_identity() const {return impl_->identity();}
}
