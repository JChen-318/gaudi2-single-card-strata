#include "strata/hpu/tensor.hpp"
#include <cmath>
#include <iostream>

using namespace strata::hpu;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class E, class F> void rejects(F f) {
    try { f(); } catch (const E&) { return; }
    throw std::runtime_error("invalid input was accepted");
}
int main() {
    try {
        require(TensorDesc{"gate_up", DType::BF16, {1280, 2560}}.bytes() == 6553600, "projection byte size");
        rejects<std::invalid_argument>([] { (void)TensorDesc{"empty", DType::F32, {0, 1}}.bytes(); });
        rejects<std::invalid_argument>([] { (void)TensorDesc{"", DType::F32, {1}}.bytes(); });
        rejects<std::overflow_error>([] { (void)TensorDesc{"huge", DType::F32, {UINT64_MAX, 2}}.bytes(); });
        rejects<std::out_of_range>([] { check_range(1024, UINT64_MAX, 128); });
        rejects<std::out_of_range>([] { check_range(1024, 1000, 25); });
        check_range(1024, 1024, 0);
        require(from_bf16(to_bf16(1.0f)) == 1.0f, "BF16 one");
        require(to_bf16(std::bit_cast<float>(0x3f808000U)) == 0x3f80, "BF16 even halfway");
        require(to_bf16(std::bit_cast<float>(0x3f818000U)) == 0x3f82, "BF16 odd halfway");
        require(std::isnan(from_bf16(to_bf16(std::bit_cast<float>(0x7f800001U)))), "BF16 NaN preservation");
        require(std::isinf(from_bf16(to_bf16(INFINITY))), "BF16 infinity");
        require(std::signbit(from_bf16(to_bf16(-0.0f))), "BF16 signed zero");
        std::cout << "PASS tensor sizing, overflow/range rejection, BF16 RNE/NaN/Inf/signed-zero\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
