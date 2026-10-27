// Host-only scan of the ONE knob that decides whether the U4 expert route is alive: how many columns share a
// single (scale, offset) affine pair. The measured 17% MoE output error at the production group widths (gate/up
// 512, down 320) was never attributed - it could be uniform 4-bit codes, or it could be the group width, which is
// 16x wider than the IQ4 block the source format uses (one scale per 32 columns). This program answers that on
// real weights, with no device and no Synapse call: it dequantizes GGUF IQ4 rows (the reference the engine
// currently serves), re-expresses them as uniform U4 at each group width, and reports the relative RMS error.
// The same table gives the resident store each width implies, bytes/weight = 0.5 + 8/g, so the quality decision
// and the HBM cost are read off one row: g=32 is the IQ4 store's own footprint, g=512 the one that produced 17%.
//
// Requires a model path, so it is deliberately NOT registered in CTest:
//   strata-hpu-u4-group-error-test <model.gguf> [layer] [experts]
#include "strata/hpu/resident_experts.hpp"
#include "strata/hpu/u4_quant.hpp"
#include "strata/hpu/weights.hpp"
#include "iq4_decode.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <vector>
using namespace strata::hpu;
namespace {
constexpr uint64_t kLayers=48;
// One [rows, columns] GGUF IQ4 matrix as the F32 values the current engine multiplies against.
std::vector<float> reference(const WeightSlice& slice) {
    const uint64_t rows=slice.rows(),columns=slice.columns();
    std::vector<float> out(rows*columns);
    std::vector<uint8_t> codes(columns);std::vector<float> scales(columns/32);
    const auto table=iq4_codebook_table();
    for(uint64_t r=0;r<rows;++r) {
        detail::decode_iq4_row(slice.bytes().data()+r*slice.row_bytes(),slice.ggml_type(),columns,codes.data(),scales.data());
        for(uint64_t c=0;c<columns;++c)
            out[r*columns+c]=float(table[codes[c]])*scales[c/32]; // the same kvalues*scale the GEMV forms
    }
    return out;
}

// Error of the uniform-code re-expression at one group width, per row (one affine pair per group of a row, the
// layout the batch GEMM's epilogue applies), reported as whole-matrix relative RMS and worst row.
void scan(const std::string& role,const WeightSlice& slice,uint64_t blocks,double weights_per_layer) {
    const uint64_t rows=slice.rows(),columns=slice.columns();
    const std::vector<float> source=reference(slice);
    double total_square=0;for(const float v:source) total_square+=double(v)*v;
    std::cout<<role<<" ["<<rows<<"x"<<columns<<"]\n";
    double previous_error=-1;
    // Every width that divides this matrix's columns, including the two the production store picks (gate/up 512
    // of 2560, down 320 of 640) so the measured 17% can be reproduced on the same row before the curve is read.
    std::vector<uint64_t> widths{32,64,128,256,320,512,640,1024,columns};
    const std::vector<uint64_t> divisors{2,4,8,16,32};
    for(const uint64_t test:divisors) if(!(columns%test)) widths.push_back(columns/test);
    std::sort(widths.begin(),widths.end());widths.erase(std::unique(widths.begin(),widths.end()),widths.end());
    for(const uint64_t g:widths) {
        if(g>columns || (columns%g)) continue;
        double sum_square=0,worst_row=0;
        for(uint64_t r=0;r<rows;++r) {
            const std::span<const float> row(source.data()+r*columns,columns);
            const auto matrix=quantize_u4_matrix(row,1,columns,g,true);
            const auto back=dequantize_u4_matrix(matrix);
            double row_square=0;
            for(uint64_t c=0;c<columns;++c) {const double d=double(row[c])-back[c];row_square+=d*d;}
            sum_square+=row_square;worst_row=std::max(worst_row,std::sqrt(row_square));
        }
        const double error=std::sqrt(sum_square/total_square);
        // Widths are visited ascending, so a wider group must never quantize better: this catches a broken fit.
        if(previous_error>=0 && error<previous_error-1e-12)
            std::cout<<"  INVARIANT BROKEN: widening the group lowered the error ("<<error<<" < "<<previous_error<<")\n";
        previous_error=error;
        const double per_weight=0.5+8.0/double(g);
        std::cout<<"  g="<<g<<" groups="<<columns/g<<" rel_rms="<<error*100<<"% worst_abs_row="<<worst_row
                 <<" bytes/weight="<<per_weight
                 <<" resident_store_GiB="<<blocks*kLayers*weights_per_layer*per_weight/(1u<<30)<<'\n';
    }
}
}

int main(int argc,char** argv) {
    try {
        if(argc<2) {std::cerr<<"usage: u4-group-error-test <model.gguf> [layer] [experts]\n";return 2;}
        const uint32_t layer=argc>2?uint32_t(std::stoul(argv[2])):0;
        const uint32_t experts=argc>3?uint32_t(std::stoul(argv[3])):2;
        WeightTable table(argv[1]);const auto& g=table.geometry();
        const std::string prefix="blk."+std::to_string(layer)+".";
        const uint64_t blocks=uint64_t(g.n_expert)+1;
        const double weights=3.0*double(g.n_ff)*double(g.n_embd); // gate + up + down elements per expert block
        std::cout<<"model "<<g.n_embd<<" hidden, "<<g.n_ff<<" expert ff, "<<g.n_expert<<" experts, layer "<<layer
                 <<"; reference = GGUF IQ4 dequant, candidate = uniform U4 + one affine per g columns\n";
        for(uint32_t e=0;e<experts;++e) {
            scan("gate exp "+std::to_string(e),table.expert(prefix+"ffn_gate_exps.weight",e),blocks,weights);
            scan("up   exp "+std::to_string(e),table.expert(prefix+"ffn_up_exps.weight",e),blocks,weights);
            scan("down exp "+std::to_string(e),table.expert(prefix+"ffn_down_exps.weight",e),blocks,weights);
        }
        scan("gate shared",table.matrix(prefix+"ffn_gate_shexp.weight"),blocks,weights);
        scan("up   shared",table.matrix(prefix+"ffn_up_shexp.weight"),blocks,weights);
        scan("down shared",table.matrix(prefix+"ffn_down_shexp.weight"),blocks,weights);
        std::cout<<"resident_store_GiB is the whole model (48 layers, all expert blocks) at that group width; "
                 <<"today's IQ4 store is 75.6 GiB\n";
        std::cout<<"PASS u4_group_error\n";
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
