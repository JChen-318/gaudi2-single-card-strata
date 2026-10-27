// Host-only: the bit-level FP8 (syn_type_fp8_143) encoder the --fp8-experts loader uses must give the same
// byte as the frexp/nearbyint reference for every input class - zeros, subnormals and their ties, every
// normal tie, the 240 saturation edge, random magnitudes - and every code must decode back to itself.
#include "strata/hpu/fp8.hpp"
#include <cstdio>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
using namespace strata::hpu;
namespace {
void require(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
void same(float value) {
    const auto fast = fp8_143_from_f32(value), slow = fp8_143_from_f32_reference(value);
    if (fast != slow) {
        char buffer[160];
        std::snprintf(buffer, sizeof buffer, "encoders disagree at %.9g: bit form 0x%02x, reference 0x%02x", value, fast, slow);
        throw std::runtime_error(buffer);
    }
}
}
int main() {
    try {
        for (int code = 0; code < 256; ++code) {
            if (((code >> 3) & 15) == 15) continue; // never produced
            const float value = f32_from_fp8_143(uint8_t(code));
            require(fp8_143_from_f32(value) == code, "code does not round-trip: " + std::to_string(code));
            same(value);
            // the midpoint to the next code is a tie and both neighbours of it must agree too
            if ((code & 0x7f) < 0x77) {
                const float next = f32_from_fp8_143(uint8_t(code + 1));
                const float middle = (value + next) / 2;
                same(middle);
                same(std::nextafter(middle, 0.0f));
                same(std::nextafter(middle, 1e9f * (value < 0 ? -1 : 1)));
            }
        }
        for (float value : {0.0f, -0.0f, 239.9f, 240.0f, 247.99f, 248.0f, 250.0f, 1e30f, -1e30f, 1e-30f, -1e-38f})
            same(value);
        std::mt19937 rng(11);
        std::uniform_real_distribution<float> exponent(-14, 9), unit(-1, 1);
        for (int i = 0; i < 2000000; ++i) same(unit(rng) * std::exp2(exponent(rng)));
        bool refused = false;
        try { (void)fp8_143_from_f32(std::numeric_limits<float>::infinity()); } catch (const std::invalid_argument&) { refused = true; }
        require(refused, "infinity was encoded");
        std::cout << "PASS FP8 143 bit encoder equals reference on all codes, ties, saturation and 2M random values\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
