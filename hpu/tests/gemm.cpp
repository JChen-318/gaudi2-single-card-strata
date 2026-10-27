#include "strata/hpu/recipe.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

using namespace strata::hpu;
void store(HostBuffer& buffer, DType type, uint64_t index, float value) {
    if (type == DType::F32) static_cast<float*>(buffer.data())[index] = value;
    else static_cast<uint16_t*>(buffer.data())[index] = to_bf16(value);
}
float load(const HostBuffer& buffer, DType type, uint64_t index) {
    return type == DType::F32 ? static_cast<const float*>(buffer.data())[index]
                             : from_bf16(static_cast<const uint16_t*>(buffer.data())[index]);
}
void run(const Device& device, uint64_t m, uint64_t k, uint64_t n, DType type,
         bool transpose_b, bool runtime_scale) {
    TensorDesc a_desc{"a", type, {m, k}};
    TensorDesc b_desc{"weight", type, transpose_b ? std::vector<uint64_t>{n, k} : std::vector<uint64_t>{k, n}};
    TensorDesc y_desc{"output", type, {m, n}};
    GraphBuilder graph(device);
    const auto a = graph.tensor(a_desc), b = graph.tensor(b_desc), y = graph.tensor(y_desc);
    auto effective_b = b;
    if (runtime_scale) {
        const auto scale = graph.tensor({"scale", type, {1}});
        auto internal = b_desc;
        internal.name = "scaled_weight";
        internal.persistent = false;
        effective_b = graph.tensor(internal);
        graph.multiply(b, scale, effective_b);
    }
    graph.gemm(a, effective_b, y, transpose_b);
    const auto name = std::string("strata_") + (type == DType::F32 ? "f32_" : "bf16_") +
                      std::to_string(m) + "_" + std::to_string(k) + "_" + std::to_string(n);
    auto recipe = graph.compile(name);
    HostBuffer ha(device, a_desc.bytes()), hb(device, b_desc.bytes()), hy(device, y_desc.bytes());
    HostBuffer hs(device, element_bytes(type));
    DeviceBuffer da(device, a_desc.bytes()), db(device, b_desc.bytes()), alternate_b(device, b_desc.bytes());
    DeviceBuffer dy(device, y_desc.bytes()), ds(device, element_bytes(type));
    Stream copy_in(device), compute(device), alternate_compute(device), copy_out(device);
    Event uploaded(device), returned(device);
    for (uint64_t row = 0; row < m; ++row)
        for (uint64_t inner = 0; inner < k; ++inner)
            store(ha, type, row * k + inner, static_cast<float>(static_cast<int>((row * 7 + inner * 3) % 17) - 8) / 16);
    double max_error = 0;
    for (unsigned phase = 0; phase < 4; ++phase) {
        const float sign = phase == 1 || phase == 2 ? -1.0f : 1.0f;
        const float scale = runtime_scale ? (phase == 2 ? 0.5f : phase == 3 ? 2.0f : 1.0f) : 1.0f;
        for (uint64_t inner = 0; inner < k; ++inner) {
            for (uint64_t col = 0; col < n; ++col) {
                const float value = sign * static_cast<float>(static_cast<int>((inner * 11 + col * 7) % 23) - 11) / 32;
                store(hb, type, transpose_b ? col * k + inner : inner * n + col, value);
            }
        }
        store(hs, type, 0, scale);
        auto& weights = phase == 3 ? alternate_b : db;
        upload(copy_in, da, ha, ha.size());
        upload(copy_in, weights, hb, hb.size());
        if (runtime_scale) upload(copy_in, ds, hs, hs.size());
        uploaded.record(copy_in);
        auto& executor = phase == 3 ? alternate_compute : compute;
        executor.wait(uploaded);
        std::vector<Binding> bindings{{"output", &dy}, {"weight", &weights}, {"a", &da}};
        if (runtime_scale) bindings.push_back({"scale", &ds});
        if (phase == 0) {
            auto invalid = bindings;
            invalid[0].name = "a";
            try { recipe->enqueue(executor, invalid); throw std::runtime_error("invalid binding accepted"); }
            catch (const std::invalid_argument&) {}
        }
        recipe->enqueue(executor, bindings);
        copy_out.wait(recipe->completion());
        download(copy_out, hy, dy, hy.size());
        returned.record(copy_out);
        returned.synchronize();
        for (uint64_t row = 0; row < m; ++row) {
            for (uint64_t col = 0; col < n; ++col) {
                double reference = 0;
                for (uint64_t inner = 0; inner < k; ++inner) {
                    const auto wi = transpose_b ? col * k + inner : inner * n + col;
                    reference += static_cast<double>(load(ha, type, row * k + inner)) * load(hb, type, wi) * scale;
                }
                if (type == DType::BF16) reference = from_bf16(to_bf16(static_cast<float>(reference)));
                const auto actual = load(hy, type, row * n + col);
                const auto error = std::abs(static_cast<double>(actual) - reference);
                max_error = std::max(max_error, error);
                const auto tolerance = type == DType::F32 ? 1e-5 : 0.001 + std::abs(reference) * 0.008;
                if (!std::isfinite(actual) || error > tolerance)
                    throw std::runtime_error(name + " mismatch phase=" + std::to_string(phase) +
                                             " row=" + std::to_string(row) + " col=" + std::to_string(col) +
                                             " expected=" + std::to_string(reference) + " actual=" + std::to_string(actual));
            }
        }
    }
    if (recipe->launches() != 4) throw std::runtime_error("unexpected recipe launch count");
    std::cout << "PASS " << name << " transpose_weight=" << transpose_b << " runtime_multiplier=" << runtime_scale
              << " application_compiles=1 launches=" << recipe->launches() << " max_abs_error=" << max_error
              << " workspace_bytes=" << recipe->workspace_bytes() << " weight_update=ok rebind=ok\n";
}
int main() {
    try {
        Device device;
        run(device, 3, 5, 7, DType::F32, false, false);
        run(device, 1, 2560, 1280, DType::BF16, true, false);
        run(device, 8, 2560, 640, DType::BF16, false, true);
        run(device, 8, 640, 2560, DType::BF16, true, false);
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
