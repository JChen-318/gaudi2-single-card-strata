#include "strata/hpu/weights.hpp"
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: strata-hpu-weights shard.gguf [shard.gguf ...]\n"; return 2; }
    try {
        for (int i = 1; i < argc; ++i) {
            strata::hpu::WeightSource source(argv[i]);
            uint64_t supported = 0, unsupported = 0;
            for (const auto& tensor : source.inventory()) {
                (tensor.bf16_conversion_supported ? supported : unsupported)++;
                std::cout << tensor.name << " type=" << tensor.ggml_type << " gguf_shape=";
                for (auto dim : tensor.gguf_shape) std::cout << dim << ',';
                std::cout << " source_bytes=" << tensor.source_bytes << " bf16_reference_conversion="
                          << (tensor.bf16_conversion_supported ? "supported" : "unsupported") << '\n';
            }
            std::cout << "file=" << argv[i] << " supported=" << supported << " unsupported=" << unsupported
                      << " mmap_only no_device_acquired\n";
        }
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
