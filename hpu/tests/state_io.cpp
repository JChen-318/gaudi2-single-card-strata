#include "strata/hpu/state_io.hpp"
#include "strata/hpu/head_mixed.hpp"
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sys/stat.h>

using namespace strata::hpu;
namespace {
void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
template<class F> void rejects(F&& f) {bool caught=false;try {f();} catch(const std::exception&) {caught=true;}require(caught,"invalid state operation accepted");}
std::vector<uint8_t> read(const std::filesystem::path& path) {std::ifstream file(path,std::ios::binary);return {std::istreambuf_iterator<char>(file),std::istreambuf_iterator<char>()};}
void write(const std::filesystem::path& path,std::span<const uint8_t> values) {std::ofstream file(path,std::ios::binary);file.write(reinterpret_cast<const char*>(values.data()),values.size());require(bool(file),"fixture write");}
struct Temporary {
    std::filesystem::path path=std::filesystem::temp_directory_path()/("strata-hpu-state-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    Temporary() {std::filesystem::create_directory(path);}
    ~Temporary() {std::error_code error;std::filesystem::remove_all(path,error);}
};
}
int main() {
    try {
        Temporary temporary;const auto path=temporary.path/"session.bin",bad=temporary.path/"bad.bin";
        const std::array<float,4> floats{0,-0.0f,1.25f,-2.5f};const std::array<uint16_t,3> halves{0,0x8000,0x3c00};
        const std::array<int32_t,3> tokens{0,248044,248319};const std::string identity="model/中文.gguf";
        {
            StateWriter writer(path.string(),4096);writer.integer(123);writer.text(identity);writer.vector<float>(floats);writer.vector<uint16_t>(halves);writer.vector<int32_t>(tokens);writer.finish();
            rejects([&]{writer.finish();});rejects([&]{writer.integer(1);});
        }
        const auto original=read(path);uint64_t checksum=0;
        // Independent bit-at-a-time CRC oracle, no production lookup table.
        for(size_t i=0;i<original.size()-8;++i) {checksum^=uint64_t(original[i])<<56;for(int bit=0;bit<8;++bit) {const bool top=checksum>>63;checksum<<=1;if(top) checksum^=0x42f0e1eba9ea3693ull;}}
        uint64_t stored=0;std::memcpy(&stored,original.data()+original.size()-8,8);require(stored==checksum,"CRC64 independent oracle mismatch");
        struct stat mode{};require(::stat(path.c_str(),&mode)==0 && (mode.st_mode&0077)==0,"state file visible to other users");
        auto consume=[&](const std::filesystem::path& source) {
            StateReader reader(source.string(),4096);require(reader.integer()==123 && reader.text()==identity,"state header/identity roundtrip");
            const auto f=reader.vector<float>(4);require(std::memcmp(f.data(),floats.data(),sizeof(floats))==0,"float bits/sign-zero changed");
            require(reader.vector<uint16_t>(3)==std::vector<uint16_t>(halves.begin(),halves.end()),"FP16 raw state changed");
            require(reader.vector<int32_t>(3)==std::vector<int32_t>(tokens.begin(),tokens.end()),"token state changed");reader.finish();rejects([&]{reader.finish();});
        };
        consume(path);
        for(size_t at:{size_t(0),size_t(18),original.size()-1}) {auto bytes=original;bytes[at]^=1;write(bad,bytes);rejects([&]{consume(bad);});}
        for(size_t size:{size_t(0),size_t(15),original.size()-3}) {write(bad,std::span<const uint8_t>(original).first(size));rejects([&]{consume(bad);});}
        {auto bytes=original;bytes.push_back(0);write(bad,bytes);rejects([&]{consume(bad);});}
        rejects([&]{StateReader reader(path.string(),8);});
        {StateWriter aborted(path.string(),16);rejects([&]{aborted.text(identity);});}
        require(read(path)==original,"failed publication replaced accepted state");
        for(const auto& entry:std::filesystem::directory_iterator(temporary.path)) require(entry.path().filename().string().find(".tmp.")==std::string::npos,"abandoned state temporary leaked");
        {StateWriter writer(bad.string(),128);writer.integer(UINT64_MAX);writer.finish();}
        rejects([&]{StateReader reader(bad.string(),128);(void)reader.vector<float>(UINT64_MAX);});
        rejects([&]{StateReader reader(bad.string(),128);(void)reader.text();});
        rejects([&]{StateReader reader(path.string(),4096);reader.finish();});
        // Exercise the production Model codec through the real bounded POSIX
        // container, including checksum validation BEFORE observable restore.
        HeadMixed mixed(4);mixed.stage(floats,1);mixed.commit();
        for(bool populated:{false,true}) {
            std::vector<float> image;mixed.capture(image,populated);
            {StateWriter writer(path.string(),4096);HeadMixed::write(writer,image,4,populated);writer.finish();}
            StateReader reader(path.string(),4096);const auto restored=HeadMixed::read(reader,4,populated);reader.finish();
            require(restored==image,"production mixed codec changed a conversation image");
            mixed.restore(restored,populated);
            if(!populated) {mixed.stage(floats,1);mixed.commit();}
        }
        const auto mixed_bytes=read(path);
        {auto bytes=mixed_bytes;bytes.back()^=1;write(bad,bytes);}
        const auto before=std::vector<float>(mixed.current().begin(),mixed.current().end());
        rejects([&]{StateReader reader(bad.string(),4096);auto image=HeadMixed::read(reader,4,true);reader.finish();mixed.restore(image,true);});
        require(std::vector<float>(mixed.current().begin(),mixed.current().end())==before,"bad checksum published another head input");
        for(const auto& invalid:{std::vector<float>{1},std::vector<float>{0,1,2,NAN},std::vector<float>{0,INFINITY,2,3}}) {
            {StateWriter writer(bad.string(),4096);writer.vector<float>(invalid);writer.finish();}
            rejects([&]{StateReader reader(bad.string(),4096);(void)HeadMixed::read(reader,4,true);});
            rejects([&]{StateWriter writer(bad.string(),4096);HeadMixed::write(writer,invalid,4,true);});
        }
        require(std::vector<float>(mixed.current().begin(),mixed.current().end())==before,"invalid mixed image mutated live state");
        std::cout<<"PASS bounded streaming state container, exact float/FP16/token bits, independent CRC64, atomic publication and private file mode\n"
                   "PASS corruption/truncation/trailing bytes, preallocation bounds, unfinished-file cleanup and preservation of previous state\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
