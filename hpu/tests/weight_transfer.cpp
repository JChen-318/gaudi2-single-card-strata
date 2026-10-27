#include "strata/hpu/weight_transfer.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <array>

using namespace strata::hpu;
namespace {
template<class T> void write(std::ofstream& out, T value) { out.write(reinterpret_cast<const char*>(&value), sizeof(value)); }
struct Fixture {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("strata-hpu-transfer-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".gguf");
    std::vector<uint16_t> expected;
    Fixture() : expected(2560 * 640) {
        std::ofstream out(path, std::ios::binary);
        write<uint32_t>(out, 0x46554747); write<uint32_t>(out, 3);
        write<uint64_t>(out, 1); write<uint64_t>(out, 0);
        const std::string name = "expert";
        write<uint64_t>(out, name.size()); out.write(name.data(), name.size());
        write<uint32_t>(out, 3);
        write<uint64_t>(out, 2560); write<uint64_t>(out, 640); write<uint64_t>(out, 3);
        write<uint32_t>(out, 30); write<uint64_t>(out, 0);
        while (uint64_t(out.tellp()) % 32) out.put(0);
        for (int expert = 0; expert < 3; ++expert) {
            for (size_t i = 0; i < expected.size(); ++i)
                expected[i] = to_bf16(float(int((i * 17 + expert * 43) % 251) - 125) / 128);
            out.write(reinterpret_cast<const char*>(expected.data()), expected.size() * 2);
        }
        if (!out) throw std::runtime_error("cannot write transfer fixture");
    }
    ~Fixture() { std::error_code error; std::filesystem::remove(path, error); }
};
}
int main() {
    try {
        Fixture fixture;
        WeightSource source(fixture.path.string());
        const auto slice = source.expert("expert", 2);
        Device device;
        Stream copy(device), compute(device), readback(device);
        const uint64_t budget = 3 * 2560 * 2; // Deliberately only one row per slot.
        Bf16WeightUpload uploaded(device, slice, copy, budget);
        if (uploaded.pinned_bytes() > budget || uploaded.transfer_count() != 640)
            throw std::runtime_error("bounded staging plan mismatch");
        // The next stream consumes only after all chunk copies are complete.
        compute.wait(uploaded.completion());
        Event consumed(device), downloaded(device);
        consumed.record(compute);
        readback.wait(consumed);
        HostBuffer result(device, uploaded.buffer().size());
        download(readback, result, uploaded.buffer(), result.size());
        downloaded.record(readback);
        downloaded.synchronize();
        const auto* actual = static_cast<const uint16_t*>(result.data());
        if (!std::equal(fixture.expected.begin(), fixture.expected.end(), actual))
            throw std::runtime_error("mapped expert HBM contents differ");
        bool rejected = false;
        try { Bf16WeightUpload invalid(device, slice, copy, 2560 * 2); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) throw std::runtime_error("undersized staging budget accepted");
        std::cout << "PASS mapped expert [640,2560] H2D chunks=" << uploaded.transfer_count()
                  << " pinned_bytes=" << uploaded.pinned_bytes() << " budget=" << budget << " exact_BF16\n";
        auto resident=uploaded.finish();
        if(uploaded.pinned_bytes()!=0 || resident.size()!=fixture.expected.size()*2) throw std::runtime_error("upload finish retained staging/lost HBM");
        download(readback,result,resident,result.size());downloaded.record(readback);downloaded.synchronize();
        if(!std::equal(fixture.expected.begin(),fixture.expected.end(),static_cast<const uint16_t*>(result.data()))) throw std::runtime_error("resident weight changed after staging release");
        rejected=false;try {(void)uploaded.finish();} catch(const std::logic_error&) {rejected=true;}
        if(!rejected) throw std::runtime_error("upload finish accepted twice");
        const std::array<int32_t,4> ids{639,0,317,639};Bf16WeightUpload gathered(device,slice,ids,copy,budget);
        if(gathered.pinned_bytes()>budget || gathered.transfer_count()!=4) throw std::runtime_error("subset staging budget/chunk count");
        readback.wait(gathered.completion());HostBuffer subset(device,gathered.buffer().size());
        download(readback,subset,gathered.buffer(),subset.size());downloaded.record(readback);downloaded.synchronize();
        const auto* rows=static_cast<const uint16_t*>(subset.data());
        for(size_t row=0;row<ids.size();++row) if(!std::equal(fixture.expected.begin()+uint64_t(ids[row])*2560,fixture.expected.begin()+(uint64_t(ids[row])+1)*2560,rows+row*2560))
            throw std::runtime_error("out-of-order/repeated subset row differs from independent source oracle");
        for(auto id:{-1,640}) {rejected=false;try {const std::array<int32_t,1> bad{id};Bf16WeightUpload invalid(device,slice,bad,copy,budget);} catch(const std::out_of_range&) {rejected=true;}if(!rejected) throw std::runtime_error("invalid subset row accepted");}
        std::cout<<"PASS transfer HBM ownership after releasing both pinned staging slots\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
