#pragma once
#include "strata/hpu/snapshot_memory.hpp"
#include "strata/hpu/gdn.hpp"
#include "strata/hpu/recipe.hpp"
#include <array>
#include <memory>
#include <vector>

namespace strata::hpu {
class StateWriter;
class StateReader;
class GdnWindow;
// Stateful GDN correctness runtime. Heavy operations are native HPU. The recurrent and conv state live in
// two device buffers each that alternate (the recipe reads one and writes the other), so a decode neither
// uploads nor reads back state and a failed decode leaves the accepted buffer untouched. Checkpoints are
// device generations (see StateRef). Decode checks its output for non-finite values; a non-finite state
// shows in the next output (and in --check-logits) instead of being scanned on the host every token.
// Borrowed immutable weights must remain ready/alive for this block's lifetime:
// w_qkv,w_gate,w_alpha,w_beta,w_out,conv_weights,norm,dt,ssm_a.
// An optional shared window executor is borrowed and must outlive all calls.
class GdnBlock {
public:
    GdnBlock(const Device&,core::ModelGeometry,std::span<const Binding> weights,GdnWindow* window=nullptr);
    ~GdnBlock();
    GdnBlock(const GdnBlock&)=delete;
    GdnBlock& operator=(const GdnBlock&)=delete;
    std::vector<float> decode(std::span<const float> normalized_hidden);
    // Fused decode (Layer, --fp8-experts): another recipe runs this block's step. external_step() returns
    // the bindings "state","conv_state" (the accepted buffers) and "state_out","conv_state_out" (the other
    // ones) after protecting any checkpoint of the buffers about to be overwritten; commit_external()
    // accepts what that recipe wrote, exactly as decode() does after its own run.
    std::array<Binding,4> external_step();
    void commit_external();
    // STRATA_HPU_DFLASH verify: current_state() is the accepted pair ("state","conv_state") a verify window
    // reads without writing. adopt() makes the buffers that window wrote for its last accepted row the accepted
    // state, `tokens` positions on, by exchanging them with the buffers about to be written (no copy): the caller
    // gets those back as its scratch. Buffers must have state_bytes()/conv_bytes() and nothing may still write them.
    std::array<Binding,2> current_state();
    void adopt(std::unique_ptr<DeviceBuffer>& state,std::unique_ptr<DeviceBuffer>& conv,uint64_t tokens);
    uint64_t state_bytes() const;
    uint64_t conv_bytes() const;
    // Atomic1..128 rows, with native subwindows<=8; optional Model executor.
    std::vector<float> prefill(std::span<const float>,const std::function<bool()>& cancelled={});
    uint64_t size() const;
    // One device generation of the state: which decode produced it, and its host copy once read back. A
    // checkpoint shares the generation instead of copying the state; the host copy is made only when the
    // device buffer holding that generation is about to be overwritten while a checkpoint still refers to
    // it, or when a persistent snapshot needs the bytes. Plain decode therefore reads nothing back.
    struct StateRef {
        uint64_t generation=0;
        std::shared_ptr<const std::vector<float>> state,conv; // empty until materialized
    };
    struct Checkpoint {
    private:
        uint64_t owner=0,epoch=0,position=0;
        std::shared_ptr<StateRef> ref;
        bool persistent=false;
        friend class GdnBlock;
    };
    Checkpoint checkpoint() const;
    Checkpoint snapshot() const;
    static uint64_t snapshot_capacity_bytes(const Checkpoint& p) {
        return p.ref && p.ref->state && p.ref->conv?snapshot_add_bytes(snapshot_vector_bytes(*p.ref->state),snapshot_vector_bytes(*p.ref->conv)):0;
    }
    // What this block contributes to a parked image: the recurrent state and conv window, which a checkpoint
    // carries whole whatever the conversation length is (upstream's reuse does not cover them either).
    uint64_t parked_bytes() const;
    void write_snapshot(StateWriter&,const Checkpoint&) const;
    Checkpoint read_snapshot(StateReader&,uint64_t position) const;
    void restore(const Checkpoint&);
    void reset();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
