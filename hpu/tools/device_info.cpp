#include "strata/hpu/runtime.hpp"
#include <iostream>

int main() {
    try {
        strata::hpu::Device device;
        const auto info = device.info();
        std::cout << "device=" << info.name << "\ndriver=" << info.driver
                  << "\nfree_bytes=" << info.free_bytes << "\ntotal_bytes=" << info.total_bytes << '\n';
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
