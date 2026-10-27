#pragma once

#include "strata/hpu/runtime.hpp"
#include "strata/hpu/packed_quant.hpp"
#include <memory>
#include <span>
#include <vector>

namespace strata::hpu {
using TensorId = uint32_t;
// An absent optional input of GraphBuilder::op (passed to Synapse as a null tensor).
inline constexpr TensorId kNoTensor = ~TensorId{0};
// Caller owns an initialized standalone SDK session; no device is acquired.
struct OfflineCompilation {};
struct Binding {
    std::string name;
    const DeviceBuffer* buffer = nullptr;
    uint64_t offset = 0;
};
// Every synLaunch of this process so far (the profiler's enqueue index of the next launch).
uint64_t recipe_launch_count();
// Live recipes, and the HBM of their own (unshared) workspaces: a diagnostic of what compiled graphs hold.
uint64_t recipe_live_count();
uint64_t recipe_own_workspace_bytes();
class Recipe {
public:
    ~Recipe();
    Recipe(const Recipe&) = delete;
    Recipe& operator=(const Recipe&) = delete;
    void enqueue(const Stream& stream, std::span<const Binding> bindings);
    // Orchestrator serializes enqueue + consuming this event. It is re-recorded
    // on the next enqueue; do not treat this reference as an immutable ticket.
    const Event& completion() const;
    uint64_t workspace_bytes() const;
    uint64_t launches() const;
    std::span<const TensorDesc> tensors() const;
    // True when a node in this recipe writes the tensor. Such a tensor is allocated and bound like
    // any other, but the host must neither supply it nor expect to read it back.
    bool produced_by_graph(const std::string&) const;
    const std::string& name() const;
private:
    struct Impl;
    explicit Recipe(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
    friend class GraphBuilder;
};

class GraphBuilder {
public:
    explicit GraphBuilder(const Device& device);
    explicit GraphBuilder(OfflineCompilation);
    ~GraphBuilder();
    GraphBuilder(const GraphBuilder&) = delete;
    GraphBuilder& operator=(const GraphBuilder&) = delete;
    TensorId tensor(TensorDesc desc);
    // Composition of define_* graphs into one recipe. While a scope is set, every new tensor name gets the
    // prefix (so two blocks can both own an "out"); an alias makes the next tensor() call asking for that
    // unprefixed name (same dtype and shape) return an existing tensor instead, which is how one block's
    // output becomes the next block's input without a host round trip. Unused aliases are refused at compile.
    void set_scope(std::string prefix);
    void alias(const std::string& name, TensorId existing);
    // The tensor a define_* call created under its full (scoped) name.
    TensorId find(const std::string& name) const;
    const TensorDesc& desc(TensorId) const; // a declared tensor's name, dtype, shape
    // A[M,K] @ B[K,N], or A[M,K] @ W[N,K]^T when transpose_b is true.
    void gemm(TensorId a, TensorId b, TensorId output, bool transpose_b = false);
    // Experimental U4 tensor metadata (one scale/zero point per output row).
    // This does not make mixed BF16/U4 GEMM legal on Gaudi2; use an explicit dequantizer there.
    void quantize_per_channel(TensorId weight, std::span<const float> scales,
                              std::span<const float> zero_points = {});
    // One scale/zero point for a whole U4 tensor, of any rank. Metadata alone is
    // not a decompression node; SDK support must be checked by the capability probe.
    void quantize_uniform(TensorId weight, float scale, float zero_point = 0);
    // One (zero point, scale) pair per flat table entry, for a rank-3 U4 weight: the probe that asks whether the
    // MME's per-channel metadata can carry one table per batch member (numZPScales = batch*N).
    void quantize_table(TensorId weight, std::span<const float> zero_points, std::span<const float> scales);
    void batch_gemm(TensorId a, TensorId b, TensorId output, bool transpose_b = false);
    void add(TensorId a, TensorId b, TensorId output);
    void softmax_last(TensorId input, TensorId output);
    // Runtime data operation; this is not FP8 quantization metadata patching.
    void multiply(TensorId a, TensorId b, TensorId output);
    void sigmoid(TensorId input, TensorId output);
    void cast(TensorId input, TensorId output);
    // out[b,N] = (x[b,K] @ w[N,K]^T) with w FP8 (F8) and its per-row F32 scale[1,N]: the activation is cast to
    // FP8 on the device with a dynamic per-row scale, so the MME runs FP8 x FP8 with F32 accumulation.
    // one_pass: the hi and lo activation codes are stacked into one FP8 GEMM, so the weight is read once, with the
    // encoding and the combination as one TPC kernel each (STRATA_HPU_FP8_DENSE; columns % 64 == 0, <= 64 rows).
    void fp8_linear(TensorId input, TensorId weight, TensorId row_scale, TensorId output, bool one_pass = false);
    // Any registered SDK kernel by guid (elementwise with broadcast, reductions, constants): the caller owns
    // the shapes and the parameter struct. Used to compose fused graphs from plain SDK kernels.
    void op(const char* guid, std::span<const TensorId> inputs, std::span<const TensorId> outputs,
            const void* parameters = nullptr, uint32_t bytes = 0);
    // Concatenation along a logical axis (0 = outermost) of same-dtype tensors that agree on every other axis.
    void concat(std::span<const TensorId> inputs, TensorId output, uint32_t axis);
    void reshape(TensorId input, TensorId output);
    // Rank-2 row operations. Every split piece is [1,N]; only F32/BF16.
    void split_rows(TensorId input, std::span<const TensorId> outputs);
    void concatenate_rows(std::span<const TensorId> inputs, TensorId output);
    void packed_matmul(TensorId weights,TensorId activation,TensorId metadata,TensorId codebook,TensorId output,uint32_t columns,uint32_t type,PackedActivation=PackedActivation::Q8_1);
    void activation_quant(TensorId input,TensorId codes,TensorId metadata,uint32_t batch,uint32_t columns);
    void row_norm(TensorId input,TensorId gamma,TensorId output,float epsilon);
    // STRATA_HPU_HC_FUSED: row_norm (SIMD) / hc_mix_window also writing `rounded`, the output cast to BF16.
    void row_norm_dual(TensorId input,TensorId gamma,TensorId output,TensorId rounded,float epsilon);
    void hc_mix_window_dual(TensorId normalized,TensorId gate,TensorId output,TensorId rounded);
    void hc_mix(TensorId normalized,TensorId gate,TensorId output);
    void hc_write(TensorId residual,TensorId block,TensorId inject,TensorId output);
    void hc_mix_window(TensorId normalized,TensorId gate,TensorId output);
    void hc_write_window(TensorId residual,TensorId block,TensorId inject,TensorId output);
    void ple_gate(TensorId key,TensorId query,TensorId value,TensorId output);
    void ple_conv(TensorId history,TensorId normalized,TensorId weights,TensorId hidden,TensorId gated,
                  TensorId next,TensorId output,uint32_t dilation);
    void gdn_conv(TensorId state, TensorId input, TensorId weights, TensorId next_state, TensorId activated);
    void gdn_prepare(TensorId activated, TensorId alpha, TensorId dt, TensorId ssm_a, TensorId raw_beta,
                     TensorId q, TensorId k, TensorId v, TensorId gate, TensorId beta, float epsilon);
    void gdn_norm(TensorId input, TensorId z, TensorId norm, TensorId output, float epsilon);
    void gdn_step(TensorId state, TensorId q, TensorId k, TensorId v, TensorId gate,
                  TensorId beta, TensorId state_output, TensorId output);
    std::unique_ptr<Recipe> compile(const std::string& name);
    // Recipes compiled with the same non-empty `shared_workspace` key use one device workspace (the largest any of
    // them needs; the device is drained before it grows). Only for recipes never launched concurrently.
    std::unique_ptr<Recipe> compile(const std::string& name,const std::string& shared_workspace);
    // Compile and validate only, destroy the resulting recipe, return workspace
    // bytes. Never acquires a device or allocates HBM. OfflineCompilation only.
    uint64_t compile_offline(const std::string& name);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::hpu
