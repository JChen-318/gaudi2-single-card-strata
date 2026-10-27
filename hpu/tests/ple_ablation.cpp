// --no-ple is upstream's explicit diagnostic ablation (src/program/generate.cpp:1944, :2635-2645,
// :3434-3436). The admission law and the two seams it changes - the model plan's PLE pricing and the
// session's optional layer-1 input - are exercised here without a device or a weight file.
#include "strata/hpu/ple_mode.hpp"
#include "strata/hpu/session_flow.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strata::hpu;
namespace {
void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
template<class F> void rejects(F&& f) {bool caught=false;try {f();} catch(const std::exception&) {caught=true;}require(caught,"expected failure");}
// Independent of the fixture below: one token's two hidden columns through three layers, broadcast to two
// streams, then summed by the head.
float logit_oracle(int32_t token,const std::array<uint32_t,3>& positions,bool ple,float ple_value) {
    std::array<float,2> row{float(token),float(token+1)};
    for(uint32_t layer=0;layer<3;++layer) {
        const float add=float(layer+1+positions[layer])+(ple && layer==1?ple_value:0.0f);
        for(auto& value:row) value+=add;
    }
    return 2*(row[0]+row[1]);
}
struct Fixture {
    static constexpr uint32_t hidden=2,streams=2,layers=3;
    std::array<uint32_t,3> positions{};
    std::array<std::vector<size_t>,3> widths;
    size_t ple_calls=0;
    bool ple_enabled=false;
    SessionOperations operations() {
        SessionOperations op;
        op.embedding=[](int32_t token) {return std::vector<float>{float(token),float(token+1)};};
        if(ple_enabled) op.ple=[this](int32_t token,std::span<const int32_t> previous) {
            ++ple_calls;return std::vector<float>{float(token),float(previous.size())};
        };
        op.checkpoint=[&](uint32_t layer) {
            const auto saved=positions[layer];
            return std::function<void()>([&,layer,saved] {positions[layer]=saved;});
        };
        op.layer=[&](uint32_t layer,std::span<const float> residual,std::span<const float> ple) {
            return layer_step(layer,residual,ple,1);
        };
        op.window_layer=[&](uint32_t layer,std::span<const float> residual,std::span<const float> embeddings,
                            std::span<const float> ple,const std::function<bool()>&) {
            require(!embeddings.empty() && embeddings.size()%hidden==0,"window embedding rows");
            const size_t rows=embeddings.size()/hidden;
            return layer_step(layer,residual,ple,rows);
        };
        op.head=[](std::span<const float> input) {
            const auto sum=std::accumulate(input.begin(),input.end(),0.0f);
            return std::vector<float>(5,sum);
        };
        op.reset_layer=[&](uint32_t layer) {positions[layer]=0;};
        return op;
    }
    std::vector<float> layer_step(uint32_t layer,std::span<const float> residual,std::span<const float> ple,size_t rows) {
        constexpr size_t n=hidden*streams;
        require(residual.size()==rows*n,"layer residual shape");
        widths[layer].push_back(ple.size());
        if(layer==1) require(ple.size()==(ple_enabled?rows*hidden:0),"layer 1 PLE placement");
        else require(ple.empty(),"PLE only reaches layer 1");
        std::vector<float> out(residual.begin(),residual.end());
        for(size_t row=0;row<rows;++row) {
            const float add=float(layer+1+positions[layer]+row);
            const float ple_value=ple.empty()?0.0f:ple[row*hidden];
            for(size_t stream=0;stream<streams;++stream)
                for(size_t column=0;column<hidden;++column) out[row*n+stream*hidden+column]+=add+ple_value;
        }
        positions[layer]+=uint32_t(rows);return out;
    }
    void all(uint32_t value) const {for(auto position:positions) require(position==value,"layer positions diverged");}
    void clear_widths() {for(auto& row:widths) row.clear();}
};
}
int main() {
    try {
        // ---- admission law
        {
            const auto normal=resolve_ple_mode(false,"table.gguf");
            require(normal.enabled && normal.external=="table.gguf","default PLE mode lost the table");
            const auto internal=resolve_ple_mode(false,"");
            require(internal.enabled && internal.external.empty(),"in-model PLE is enabled with no external file");
            const auto off=resolve_ple_mode(true,"");
            require(!off.enabled && off.external.empty(),"--no-ple must not leave a table behind");
            bool conflict=false;
            try {(void)resolve_ple_mode(true,"table.gguf");}
            catch(const std::invalid_argument& error) {
                conflict=std::string(error.what())=="--no-ple and --ple-gguf are mutually exclusive";
            }
            require(conflict,"--no-ple plus --ple-gguf did not refuse with upstream's sentence");
            require(std::string(ple_off_notice())=="strata generate: PLE OFF by explicit --no-ple diagnostic request.\n"
                    "  The tokens below are NOT this model's; this is only useful for A/B measurement.\n",
                    "the ablation notice is not upstream's text");
        }
        // ---- session without PLE: the ablation's empty layer-1 span
        {
            Fixture f;SessionFlow flow(2,2,3,5,8,f.operations());
            const auto logits=flow.decode(2);
            require(logits.size()==5,"decode logits shape");
            for(float value:logits) require(value==logit_oracle(2,{0,0,0},false,0),"ablation decode oracle");
            require(f.ple_calls==0,"ablation still gathered a PLE row");
            require(f.widths[1].size()==1 && f.widths[1][0]==0,"layer 1 accepted a PLE span without a table");
            require(f.widths[0][0]==0 && f.widths[2][0]==0,"PLE reached a non-layer-1 layer");
            f.all(1);require(flow.tokens().size()==1 && flow.tokens()[0]==2,"ablation token did not commit");
            const auto second=flow.decode(3);
            for(float value:second) require(value==logit_oracle(3,{1,1,1},false,0),"ablation history oracle");
            f.all(2);
        }
        // ---- the same flow shape with PLE enabled keeps the original seam
        {
            Fixture f;f.ple_enabled=true;SessionFlow flow(2,2,3,5,8,f.operations());
            const auto logits=flow.decode(3);
            for(float value:logits) require(value==logit_oracle(3,{0,0,0},true,3),"PLE decode oracle");
            require(f.ple_calls==1,"PLE was not gathered once");
            require(f.widths[1].size()==1 && f.widths[1][0]==Fixture::hidden,"layer 1 did not receive [H]");
            require(f.widths[0][0]==0 && f.widths[2][0]==0,"PLE reached a non-layer-1 layer");
            const auto second=flow.decode(4);
            for(float value:second) require(value==logit_oracle(4,{1,1,1},true,4),"PLE history oracle");
        }
        // ---- a wrong-width PLE row is still refused and fully rewound
        {
            Fixture f;auto ops=f.operations();size_t calls=0;
            ops.ple=[&](int32_t,std::span<const int32_t>) {++calls;return std::vector<float>{1};};
            SessionFlow flow(2,2,3,5,8,ops);
            rejects([&] {(void)flow.decode(3);});
            require(calls==1,"short PLE row was not requested");
            f.all(0);require(flow.tokens().empty(),"short PLE row committed layer state");
        }
        // ---- the atomic window path, ablation and normal
        {
            Fixture f;SessionFlow flow(2,2,3,5,8,f.operations());
            const std::array<int32_t,2> tokens{2,3};
            const auto logits=flow.prefill_window(tokens);
            for(float value:logits) require(value==logit_oracle(3,{1,1,1},false,0),"ablation window oracle");
            require(f.ple_calls==0,"ablation window gathered PLE");
            require(f.widths[1].size()==1 && f.widths[1][0]==0,"ablation window layer-1 span");
            require(flow.tokens().size()==2,"ablation window did not commit");
            f.all(2);
        }
        {
            Fixture f;f.ple_enabled=true;SessionFlow flow(2,2,3,5,8,f.operations());
            const std::array<int32_t,2> tokens{2,3};
            const auto logits=flow.prefill_window(tokens);
            for(float value:logits) require(value==logit_oracle(3,{1,1,1},true,3),"PLE window oracle");
            require(f.ple_calls==2,"window PLE gather count");
            require(f.widths[1].size()==1 && f.widths[1][0]==2*Fixture::hidden,
                    "window layer 1 did not receive [T,H]");
            require(f.widths[0][0]==0 && f.widths[2][0]==0,"window PLE reached a non-layer-1 layer");
        }
        // A session is still incomplete without an embedding, even though PLE is now optional.
        SessionOperations empty;rejects([&] {SessionFlow invalid_flow(2,2,3,5,8,empty);});
        std::cout<<"PASS --no-ple admission law, exact upstream refusal/notice, ablation decode/prefill window\n"
                   "PASS PLE-enabled session layer-1 seam, short-row rewind, optional-but-not-incomplete operations\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
