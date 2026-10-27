#include "strata/hpu/weights.hpp"
#include "strata/hpu/rope_config.hpp"
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace strata::hpu;
namespace {
void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
template<class F> void rejects(F&& f) {bool rejected=false;try {f();} catch(const std::exception&) {rejected=true;}require(rejected,"invalid split catalog accepted");}
template<class T> void put(std::vector<uint8_t>& out,T value) {const auto* p=reinterpret_cast<const uint8_t*>(&value);out.insert(out.end(),p,p+sizeof(value));}
void str(std::vector<uint8_t>& out,const std::string& value) {put<uint64_t>(out,value.size());out.insert(out.end(),value.begin(),value.end());}
void integer(std::vector<uint8_t>& out,const std::string& key,uint32_t value) {str(out,key);put<uint32_t>(out,4);put(out,value);}
struct Fixture {
    std::filesystem::path dir=std::filesystem::temp_directory_path()/("strata-hpu-shards-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::vector<std::string> paths;
    Fixture() {
        std::filesystem::create_directory(dir);
        for(auto name:{"model-00001-of-00003.gguf","model-00002-of-00003.gguf","model-00003-of-00003.gguf"}) paths.push_back((dir/name).string());
        write(0);write(1);write(2);
    }
    ~Fixture() {std::error_code error;std::filesystem::remove_all(dir,error);}
    void write(uint32_t shard,bool duplicate=false,uint32_t no=99,uint32_t hidden=2560,const std::string& arch="qwen4exp") {
        std::vector<uint8_t> out;put<uint32_t>(out,0x46554747);put<uint32_t>(out,3);put<uint64_t>(out,shard?1:0);put<uint64_t>(out,shard?3:17);
        integer(out,"split.count",3);integer(out,"split.no",no==99?shard:no);integer(out,"split.tensors.count",2);
        if(!shard) {
            str(out,"general.architecture");put<uint32_t>(out,8);str(out,arch);
            integer(out,"qwen4exp.block_count",48);integer(out,"qwen4exp.embedding_length",hidden);
            integer(out,"qwen4exp.expert_count",512);integer(out,"qwen4exp.expert_used_count",10);
            integer(out,"qwen4exp.attention.head_count",24);integer(out,"qwen4exp.attention.head_count_kv",2);
            str(out,"test.exact");put<uint32_t>(out,9);put<uint32_t>(out,10);put<uint64_t>(out,2);put<uint64_t>(out,9007199254740993ull);put<uint64_t>(out,UINT64_MAX);
            str(out,"test.negative");put<uint32_t>(out,11);put<int64_t>(out,-1);
            str(out,"test.bool");put<uint32_t>(out,7);put<uint8_t>(out,1);
            str(out,"test.large");put<uint32_t>(out,9);put<uint32_t>(out,4);put<uint64_t>(out,65);for(uint32_t i=0;i<65;++i) put(out,i);
            str(out,"qwen4exp.rope.scaling.type");put<uint32_t>(out,8);str(out,"linear");
            str(out,"qwen4exp.rope.scaling.factor");put<uint32_t>(out,12);put<double>(out,4);
            str(out,"qwen4exp.rope.freq_base");put<uint32_t>(out,12);put<double>(out,1e6);
        } else {
            str(out,duplicate?"matrix":shard==1?"matrix":"expert");put<uint32_t>(out,shard==1?2:3);
            put<uint64_t>(out,2);put<uint64_t>(out,3);if(shard==2) put<uint64_t>(out,2);
            put<uint32_t>(out,0);put<uint64_t>(out,0);
        }
        out.resize((out.size()+31)/32*32);
        if(shard) for(uint32_t i=0;i<(shard==1?6:12);++i) put<float>(out,float(shard*10+i));
        std::ofstream file(paths[shard],std::ios::binary);file.write(reinterpret_cast<const char*>(out.data()),out.size());require(bool(file),"fixture write failed");
    }
};
}
int main() {
    try {
        Fixture f;
        WeightSlice retained;
        {
            WeightTable table(f.paths[2]);
            require(table.shard_count()==3 && table.geometry().n_layers==48 && table.inventory().size()==2,"metadata-only first shard/catalog geometry");
            require(table.shard("matrix")==1 && table.shard("expert")==2,"cross-shard resolution");
            require(table.tensor_path("matrix")==f.paths[1],"tensor path resolution");
            const auto location=table.location("matrix");
            require(location.path==std::filesystem::canonical(f.paths[1]).string() && location.offset+24<=location.file_bytes,"absolute tensor location bounds");
            require(table.shard_files().size()==3 && table.shard_files()[1].second==std::filesystem::file_size(f.paths[1]),"native pack source shard inventory");
            const auto identity=table.file_identity();require(identity==table.file_identity() && identity.find("gguf-local-files-v1|")==0,"stable local model-file identity");
            require(table.integers("qwen4exp.block_count")==std::optional<std::vector<uint64_t>>({48}),"scalar integer metadata");
            require(table.integers("test.exact")==std::optional<std::vector<uint64_t>>({9007199254740993ull,UINT64_MAX}),"uint64 metadata rounded");
            require(!table.integers("absent"),"missing metadata not optional");
            for(const auto* key:{"test.negative","test.bool","test.large","general.architecture"}) rejects([&]{(void)table.integers(key);});
            require(table.number("qwen4exp.rope.freq_base")==1e6 && table.text("general.architecture")=="qwen4exp","numeric/string metadata");
            rejects([&]{(void)table.number("general.architecture");});rejects([&]{(void)table.number("test.bool");});rejects([&]{(void)table.text("split.count");});
            auto rope=resolve_rope(table);using Type=strata::kernels::RopeScalingType;
            require(rope.type==Type::Linear && rope.factor==4 && rope.freq_base==1e6,"GGUF RoPE defaults");
            RopeOptions override;override.type=Type::None;override.freq_base=7e6;rope=resolve_rope(table,override);
            require(rope.type==Type::None && rope.factor==1 && rope.freq_base==7e6,"explicit none must clear scaling while preserving base");
            override.type=Type::YaRN;override.factor=8;rope=resolve_rope(table,override);require(rope.ext_factor==1 && rope.factor==8,"YaRN defaults/CLI precedence");
            override.freq_base=0;rejects([&]{(void)resolve_rope(table,override);});
            require(table.constant("matrix",6)==std::vector<float>({10,11,12,13,14,15}),"constant materialization");
            rejects([&]{(void)table.constant("matrix",6,5);});rejects([&]{(void)table.constant("matrix",5);});
            retained=table.expert("expert",1);std::vector<float> values(6);retained.convert_rows_f32(0,values);
            require(values==std::vector<float>({26,27,28,29,30,31}),"expert offset across shards");
            rejects([&]{(void)table.expert("expert",2);});rejects([&]{(void)table.info("missing");});
        }
        std::vector<float> values(6);retained.convert_rows_f32(0,values);require(values[0]==26,"slice must retain model mappings");
        // The lifetime assertion is complete. Windows also disallows writing
        // a shard while its retained read mapping is open; release before the
        // malformed-file fixtures as well as before the final rename.
        retained=WeightSlice{};
        rejects([&]{(void)WeightTable(std::vector<std::string>{f.paths[0],f.paths[1]});});
        f.write(2,true);rejects([&]{(void)WeightTable(f.paths);});f.write(2,false,1);rejects([&]{(void)WeightTable(f.paths);});f.write(2);
        f.write(0,false,99,64);rejects([&]{(void)WeightTable(f.paths);});f.write(0,false,99,2560,"qwen35moe");rejects([&]{(void)WeightTable(f.paths);});f.write(0);
        // The source file is renamed only after every mapping using it closes.
        retained=WeightSlice{};
        std::filesystem::rename(f.paths[2],f.dir/"missing.gguf");rejects([&]{(void)WeightTable(f.paths[0]);});
        std::cout<<"PASS complete GGUF shard family, metadata-only first shard, geometry, cross-shard expert/constant resolution\n"
                   "PASS missing/misordered/duplicate/foreign shards, shape/budget guards and shared mmap lifetime\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
