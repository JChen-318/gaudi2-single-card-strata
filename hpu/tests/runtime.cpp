#include "strata/hpu/runtime.hpp"
#include <cstring>
#include <iostream>

using namespace strata::hpu;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
int main() {
    try {
        Device device;
        bool second_rejected = false;
        try { Device second; }
        catch (const std::logic_error&) { second_rejected = true; }
        require(second_rejected, "second independent SDK owner was accepted");
        Device shared(device);
        require(shared.state() == device.state(), "Device copy did not share ownership");
        constexpr uint64_t bytes = 4 * 1024 * 1024;
        constexpr uint64_t offset = 256;
        HostBuffer source(device, bytes), output(device, bytes);
        DeviceBuffer memory(device, bytes + offset);
        Stream copy_in(device), copy_out(device);
        Event ready(device), done(device);
        try { ready.synchronize(); throw std::runtime_error("unrecorded synchronize accepted"); }
        catch (const std::logic_error&) {}
        try { (void)ready.ready(); throw std::runtime_error("unrecorded query accepted"); }
        catch (const std::logic_error&) {}
        try { copy_out.wait(ready); throw std::runtime_error("unrecorded wait accepted"); }
        catch (const std::logic_error&) {}
        for (unsigned pass = 0; pass < 16; ++pass) {
            auto* p = static_cast<unsigned char*>(source.data());
            for (uint64_t i = 0; i < bytes; ++i) p[i] = static_cast<unsigned char>((i * 37 + pass * 19) & 255);
            std::memset(output.data(), 0, bytes);
            upload(copy_in, memory, source, bytes, offset);
            ready.record(copy_in);
            copy_out.wait(ready);
            download(copy_out, output, memory, bytes, 0, offset);
            done.record(copy_out);
            done.synchronize();
            require(done.ready(), "completed event is not ready");
            require(std::memcmp(source.data(), output.data(), bytes) == 0, "cross-stream DMA data mismatch");
        }
        try { upload(copy_in, memory, source, bytes + 1); throw std::runtime_error("out-of-range DMA accepted"); }
        catch (const std::out_of_range&) {}
        DeviceBuffer moved(std::move(memory));
        require(memory.address() == 0 && moved.size() == bytes + offset, "move ownership failed");
        DeviceBuffer empty(device, 0);
        upload(copy_in, empty, source, 0);
        std::cout << "PASS DMA bytes=" << bytes << " passes=16 streams=2 offset=" << offset
                  << " event_ordering=ok bounds=ok move=ok\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
