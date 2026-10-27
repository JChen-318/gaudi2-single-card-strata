#pragma once
#include "strata/hpu/recipe.hpp"

namespace strata::hpu {
// Synchronous correctness executor. Allocates pinned and HBM mutable slots
// once, reuses three streams/events, uploads only dirty inputs, reads named
// outputs. External bindings are borrowed, immutable, ready on entry and must
// outlive this object. Budget covers total pinned slots, not workspace/weights.
// Single caller; returned output spans remain valid until next run().
// While alive, a HostRecipe built on this thread borrows `scratch` (by tensor name, when large enough) for a
// graph-internal persistent tensor -- written and read inside one launch, neither bound nor read by the host --
// instead of allocating its own: recipes that run one after another on one stream (the verify chain's window graphs,
// one set per row count) then share one buffer per name. The buffers must outlive the recipes.
class RecipeScratchScope {
public:
    explicit RecipeScratchScope(std::span<const Binding> scratch);
    ~RecipeScratchScope();
    RecipeScratchScope(const RecipeScratchScope&)=delete;
    RecipeScratchScope& operator=(const RecipeScratchScope&)=delete;
private:
    std::span<const Binding> previous_;
};
class HostRecipe {
public:
    HostRecipe(const Device&,std::unique_ptr<Recipe>,std::span<const Binding> external,
               std::vector<std::string> outputs,uint64_t pinned_budget=64*1024*1024);
    ~HostRecipe();
    HostRecipe(const HostRecipe&)=delete;
    HostRecipe& operator=(const HostRecipe&)=delete;
    void set(const std::string&,std::span<const float>); // F32 or BF16 conversion
    void set_bytes(const std::string&,std::span<const uint8_t>); // exact U8 input, no conversion
    // Direct write into a pinned F32 input instead of copying one in: the slot ends initialized and dirty
    // exactly as set() leaves it. The caller fills every element before the next run()/launch(); the span
    // stays valid until then.
    std::span<float> input(const std::string&);
    // Between synchronous runs, replace a borrowed input weight of unchanged
    // geometry. Caller retains its lease until run/readback finishes.
    void bind_external(Binding);
    // True when `name` was one of the external bindings this recipe was built with (what bind_external accepts).
    bool declares(const std::string& name) const;
    void run();
    // Asynchronous variant for a chain of recipes on one caller-owned stream: dirty inputs, the launch and
    // every host output are enqueued in order on `stream`; wait() blocks until they finished. Inputs must not
    // be set again before wait() returns. Device-only chains may declare no host outputs at all.
    void launch(const Stream& stream);
    void wait();
    std::span<const float> output(const std::string&) const; // F32 outputs only
    std::vector<float> values(const std::string&) const; // F32 or widened BF16 copy
    uint64_t pinned_bytes() const;
    // The HBM this recipe allocated for its own slots (persistent tensors not bound externally); not the workspace.
    uint64_t device_bytes() const;
    // The device_bytes() of every live HostRecipe (a diagnostic).
    static uint64_t total_device_bytes();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
