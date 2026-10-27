#pragma once
#include "strata/hpu/expert_cache.hpp"
#include "strata/hpu/runtime.hpp"
#include "strata/hpu/recipe.hpp"
#include <algorithm>
#include <array>
#include <functional>
#include <initializer_list>
#include <utility>
#include <memory>
#include <vector>

namespace strata::hpu {
// STRATA_HPU_CHAIN: one decode token through every fused layer with the residual kept on the device. Every
// graph of the token goes to `stream` in order; the host waits only for each layer's router logits (and the
// QSA projections), so a layer's expert group and the next layer's first graph run back to back.
// residual[current] is the residual entering the next graph; mid/x/inject carry the attention half's outputs
// into the expert group. `leases` hold the weights of the expert group still in flight; they are released
// after the next wait, which orders after it on the stream.
struct DecodeChain {
    DecodeChain(const Device& device,uint64_t hidden,uint32_t streams,uint64_t layer_stage_bytes=0)
        :stream(device),residual{DeviceBuffer(device,hidden*streams*4),DeviceBuffer(device,hidden*streams*4)},
         mid(device,hidden*streams*4),x(device,hidden*4),inject(device,uint64_t(streams)*4),staging(device,hidden*streams*4),
         layer_stage(device,std::max<uint64_t>(layer_stage_bytes,128)),layer_stage_host(device,std::max<uint64_t>(layer_stage_bytes,128)) {}
    Stream stream;
    std::array<DeviceBuffer,2> residual;int current=0;
    DeviceBuffer mid,x,inject;
    HostBuffer staging; // the token's input and output residual
    std::vector<ExpertLease> leases;
    // STRATA_HPU_QSA_DEVICE: per-layer rows the device wrote this token (downloaded with the final residual) and
    // the host commits that consume them, in layer order, once the token's work is done.
    DeviceBuffer layer_stage;HostBuffer layer_stage_host;
    std::vector<std::function<void(const uint8_t*)>> commits;
    DeviceBuffer& in() {return residual[current];}
    DeviceBuffer& out() {return residual[1-current];}
    void advance() {current=1-current;}
};
// STRATA_HPU_DFLASH verify / the resident prefill window: `rows` tokens through every layer, one launch per
// layer, on `stream`. residual[current] is the [rows, streams, hidden] window entering the next graph. No layer
// accepts anything before Layer::verify_commit(): GDN layers leave every row's state in their own scratch, QSA
// layers stage every row's K, V and raw index key in `stage` (downloaded into stage_host once the window ran) and
// the layer-1 PLE leaves every row's history in `ple_history`. `mask` is the QSA layers' shared attention mask of
// the window; `mixed` is the head graph's [rows, hidden] head input (the committed row's value).
struct VerifyChain {
    // history_rows: the rows whose PLE history `ple_history` holds (0: max_rows; a final-state prefill window keeps one).
    VerifyChain(const Device& device,uint64_t hidden,uint32_t streams,uint32_t max_rows,uint64_t stage_bytes,uint64_t ple_history_bytes,uint64_t mask_bytes,
                uint64_t vocabulary,uint64_t mask_header_bytes=0,uint32_t logit_rows=0,uint32_t history_rows=0)
        :stream(device),copy(device),max_rows(max_rows),logit_rows(logit_rows?logit_rows:max_rows),
         residual{DeviceBuffer(device,hidden*streams*4*max_rows),DeviceBuffer(device,hidden*streams*4*max_rows)},
         staging(device,hidden*streams*4*max_rows),mixed(device,hidden*4*max_rows),mixed_host(device,hidden*4*max_rows),
         stage(device,std::max<uint64_t>(stage_bytes,128)),stage_host(device,std::max<uint64_t>(stage_bytes,128)),
         ple_history(device,std::max<uint64_t>(ple_history_bytes*(history_rows?history_rows:max_rows),128)),
         ple_history_host(device,std::max<uint64_t>(ple_history_bytes*(history_rows?history_rows:max_rows),128)),
         mask(device,std::max<uint64_t>(mask_bytes+mask_header_bytes,128)),mask_host(device,std::max<uint64_t>(mask_bytes+mask_header_bytes,128)),
         logits(device,vocabulary*4*this->logit_rows),logits_host(device,vocabulary*4*this->logit_rows),copy_ready(device),copy_done(device) {
        // The window graphs' internal persistent tensors, one buffer per name shared by every row count's recipes
        // (RecipeScratchScope in Layer::verify_chain): the post-attention residual and the [rows, hidden] rows.
        const uint64_t row=hidden*4;
        for(const auto& [name,bytes]:std::initializer_list<std::pair<const char*,uint64_t>>{
                {"attn_write.out",row*streams*max_rows},{"attn.mixed",row*max_rows},{"ffn.mixed",row*max_rows},
                {"o.out",row*max_rows},{"gdn.out",row*max_rows},{"router.logits",uint64_t(1024)*4*max_rows}}) {
            scratch_buffers.push_back(std::make_unique<DeviceBuffer>(device,bytes));scratch.push_back({name,scratch_buffers.back().get()});
        }
    }
    // The window graphs' shared internal tensors (see the constructor).
    std::vector<std::unique_ptr<DeviceBuffer>> scratch_buffers;std::vector<Binding> scratch;
    Stream stream;
    // The stage/residual readback: only select() and verify_commit() read it, so it runs alongside the head.
    Stream copy;
    uint32_t max_rows;
    // Rows of `logits`: max_rows, or 1 when only a last-row prefill head writes it (row 0 then holds the last row).
    uint32_t logit_rows;
    uint32_t rows=0;
    bool prefill_final=false; // Full-window prefill only; speculative verification keeps every row state.
    // STRATA_HPU_MOE_MME: the tile rows of this window's grouped MME MoE, 0 for the per-slot GEMV path.
    uint32_t moe_mme_bucket=0;
    std::array<DeviceBuffer,2> residual;int current=0;
    HostBuffer staging; // the window's input and output residual rows
    DeviceBuffer mixed;HostBuffer mixed_host; // the head's [rows, hidden] mixed input, last row downloaded
    DeviceBuffer stage;HostBuffer stage_host;
    DeviceBuffer ple_history;HostBuffer ple_history_host;
    DeviceBuffer mask;HostBuffer mask_host;uint64_t mask_key=UINT64_MAX; // (cell, bucket, rows) of the uploaded mask
    DeviceBuffer logits;HostBuffer logits_host; // the head's [rows, vocabulary]; rows are downloaded on demand
    Event copy_ready,copy_done;
    DeviceBuffer& in() {return residual[current];}
    DeviceBuffer& out() {return residual[1-current];}
    void advance() {current=1-current;}
};
}
