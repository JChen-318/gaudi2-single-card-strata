#include "strata/hpu/native_pack.hpp"
#include "../src/native_pack_json.hpp"
#include "../src/sha256.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstring>

using namespace strata::hpu;
namespace {
void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
std::string read(const std::filesystem::path& path) {std::ifstream file(path,std::ios::binary);return {std::istreambuf_iterator<char>(file),{}};}
}
int main(int argc,char** argv) {
    try {
        if(argc==3 && std::string(argv[1])=="--sha256") {std::cout<<detail::sha256(read(argv[2]))<<'\n';return 0;}
        if(argc==3 && std::string(argv[1])=="--json") {
            const auto text=read(argv[2]);const auto provenance=detail::PackJson(text).parse();
            for(const auto& shard:provenance.shards) std::cout<<shard.first<<'\n';
            return 0;
        }
        if(argc!=3) return 2;
        const std::filesystem::path root(argv[1]);const std::string variant(argv[2]);
        const bool mixed=variant.starts_with("v4"),nv=!variant.ends_with("flat");
        NativePackCatalog catalog;catalog.experts=3;catalog.layers.resize(3);
        std::array<std::shared_ptr<std::vector<uint8_t>>,2> owners;
        for(size_t i=0;i<2;++i) {
            const auto path=root/(i?"model-00002-of-00002.gguf":"model-00001-of-00002.gguf");const auto raw=read(path);
            owners[i]=std::make_shared<std::vector<uint8_t>>(raw.begin(),raw.end());catalog.shards.emplace_back(path.string(),raw.size());
        }
        std::array<uint64_t,2> position{128,128};
        auto slices=std::make_shared<std::array<std::array<std::array<WeightSlice,3>,3>,3>>();
        for(uint32_t l=0;l<3;++l) {
            auto& layer=catalog.layers[l];layer.scales.assign(3,l==2 && nv?std::array<float,3>{.25f,.5f,1}:std::array<float,3>{1,1,1});
            for(size_t r=0;r<3;++r) {
                const auto shard=l==1 || (l==0 && mixed && r==1) || (l==2 && mixed && r==2)?1:0;
                const uint32_t type=l==0?42:l==1 || !nv?8:r<2?40:8;
                const uint64_t per=type==42?1152:type==40?2304:4352;
                layer.roles[r]={type,64,64,position[shard],per,catalog.shards[shard].first};
                for(uint32_t e=0;e<3;++e) {
                    const auto& owner=owners[shard];const auto offset=position[shard]+e*per;
                    require(offset<=owner->size() && per<=owner->size()-offset,"fixture source bounds");
                    (*slices)[l][e][r]=WeightSlice::raw_rows(owner,{owner->data()+offset,size_t(per)},type,64,64);
                }
                position[shard]+=per*3;
            }
        }
        catalog.gguf_expert=[slices](uint32_t l,uint32_t e){return (*slices)[l][e];};
        WeightSlice retained;uint8_t marker=0;
        {
            NativeExpertPack pack(root.string(),std::move(catalog));
            require(pack.has_binary()==std::filesystem::exists(root/"experts.bin"),"native binary selection ignored");
            for(uint32_t l=0;l<3;++l) for(uint32_t e=0;e<3;++e) {
                const auto source=pack.slices(l,e);const auto blob=pack.native_blob(l,e);uint64_t offset=0;
                require(blob.size()==pack.blob_bytes(l),"native blob byte count");
                for(size_t r=0;r<3;++r) {
                    const auto expected=uint8_t(pack.has_binary()?31+l*21+r*5+e:7+l*9+r*3+e);
                    require(std::all_of(source[r].bytes().begin(),source[r].bytes().end(),[&](uint8_t x){return x==expected;}),"expert binary/GGUF slice byte oracle");
                    require(std::equal(source[r].bytes().begin(),source[r].bytes().end(),blob.begin()+offset),"CPU blob and HPU slices disagree");offset+=source[r].source_bytes();
                    retained=source[r];marker=expected;
                }
                if(l==2 && nv) {
                    const std::array<float,4> tail{.25f,.5f,1,0};require(std::memcmp(blob.data()+offset,tail.data(),16)==0,"native CPU scale tail oracle");
                    require(pack.scales(l,e)==std::array<float,3>{.25f,.5f,1},"HPU F32 projection scales lost");offset+=16;
                }
                require(offset==blob.size(),"native blob has unaccounted padding");
            }
            bool rejected=false;try {(void)pack.slices(3,0);} catch(const std::exception&) {rejected=true;}require(rejected,"invalid native expert layer accepted");
            rejected=false;try {(void)pack.slices(0,3);} catch(const std::exception&) {rejected=true;}require(rejected,"invalid native expert ID accepted");
            std::cout<<pack.file_identity()<<'\n';
        }
        require(retained.bytes().front()==marker && retained.bytes().back()==marker,"native slice lost mapped owner after pack destruction");
        std::cout<<"PASS native expert source, CPU blob/HPU slice agreement, NVFP4 tails and retained owner; no device\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
