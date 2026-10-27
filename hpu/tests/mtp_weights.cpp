#include "strata/hpu/mtp_weights.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>

using namespace strata::hpu;
namespace {
void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
template<class F> void rejects(F&& f) {bool caught=false;try {f();} catch(const std::exception&) {caught=true;}require(caught,"invalid MTP access accepted");}
}
int main(int argc,char** argv) {
    if(argc!=2) return 2;
    try {
        const std::filesystem::path root(argv[1]);
        DraftVocabulary vocabulary((root/"subset_valid.bin").string(),16);
        require(vocabulary.ids()==std::vector<int32_t>({9,1,4}) && !vocabulary.file_identity().empty(),"draft vocabulary LE order/identity mismatch");
        for(const auto* name:{"duplicate","negative","range","oversize","empty","trailing"}) rejects([&]{DraftVocabulary bad((root/(std::string("subset_")+name+".bin")).string(),16);});
        WeightSlice dense;std::array<WeightSlice,3> expert;
        {
            MtpWeights weights(argv[1]);require(weights.inventory().size()==26,"complete original MTP schema not loaded");
            uint64_t sum=0;for(const auto& t:weights.inventory()) sum+=t.rows*t.columns*(t.kind=="f32"?4:2);
            require(weights.dense_payload_bytes()==sum,"MTP decoded payload planner mismatch");
            uint64_t packed=sum,q4=sum;
            for(const auto* name:{"fc_embedding.weight","fc_hidden.weight","self_attn.q_proj.weight","self_attn.k_proj.weight","self_attn.v_proj.weight","self_attn.o_proj.weight"}) {
                const auto& t=weights.info(name);packed-=t.rows*(t.columns*2-t.columns/32*34);
                q4-=t.rows*(t.columns*2-t.columns/32*18);
            }
            require(weights.packed_projection_payload_bytes()==packed && packed<sum,"MTP six packed projection planner differs from independent row byte sums");
            require(weights.packed_projection_payload_bytes(true)==q4 && q4<packed,"MTP six Q4 projection planner differs from independent byte sums");
            const auto gamma=weights.constant("pre_fc_norm_embedding.weight",2560);
            require(gamma[0]==1.25f && gamma[1]==-0.5f && gamma[2]==0,"runtime norm folded 1+w twice");
            dense=weights.matrix("fc_embedding.weight");expert=weights.expert(511);
            require(weights.vector("mlp.shared_expert_gate.weight").columns()==2560,"runtime shared gate vector axes");
            rejects([&]{(void)weights.vector("fc_hidden.weight");});
            rejects([&]{(void)weights.expert(512);});rejects([&]{(void)weights.constant("pre_fc_norm_embedding.weight",1);});
            rejects([&]{(void)weights.matrix("missing");});rejects([&]{MtpWeights small(argv[1],1024);});
            const auto identity=weights.file_identity();const auto path=std::filesystem::path(argv[1])/"dense.txt";
            std::filesystem::last_write_time(path,std::filesystem::last_write_time(path)+std::chrono::seconds(1));
            require(weights.file_identity()!=identity,"MTP identity missed index change");
        }
        // A slice retains exactly its mapped/relayout owner after the catalog.
        std::vector<float> values(2560);dense.convert_rows_f32(0,values);
        for(size_t i=0;i<values.size();++i) require(values[i]==(i<32?float(int(i)-16):0),"Q8_0 runtime row differs from independent scale/code oracle");
        std::vector<uint16_t> bf16(2560);dense.convert_rows(0,bf16);
        for(size_t i=0;i<values.size();++i) require(from_bf16(bf16[i])==values[i],"MTP Q8_0 BF16 conversion differs");
        std::vector<uint8_t> q4(2560/32*18);dense.convert_rows_q4(0,q4);
        require(q4[0]==0 && q4[1]==0x40,"runtime Q8 -> Q4 signed first-max scale differs");
        for(uint32_t j=0;j<16;++j) {
            const auto low=uint8_t(std::min(15,int((float(j)-16)/2+8.5f))),high=uint8_t(std::min(15,int(float(j)/2+8.5f)));
            require(q4[2+j]==uint8_t(low|(high<<4)),"runtime Q8 -> Q4 nibble or selected row order differs");
        }
        for(uint32_t role=0;role<3;++role) {
            values.assign(role==2?640:2560,0);expert[role].convert_rows_f32(0,values);
            const float scale=role==0?1:role==1?2:0.5f;
            for(size_t i=0;i<values.size();++i) require(values[i]==(i<64?float(int(i%4)-1)*scale:0),"MTP planar expert role/row/code/scale relayout mismatch");
            expert[role].convert_rows_f32(1,values);for(float x:values) require(x==0,"MTP expert relayout read adjacent row");
        }
        const auto raw=std::make_shared<std::vector<uint8_t>>(34);
        rejects([&]{(void)WeightSlice::raw_rows(raw,*raw,8,31,1);});
        rejects([&]{(void)WeightSlice::raw_rows(raw,*raw,8,32,2);});
        rejects([&]{(void)WeightSlice::raw_rows(raw,*raw,999,32,1);});
        rejects([&]{(void)WeightSlice::raw_rows({},*raw,8,32,1);});
        std::cout<<"PASS original MTP runtime complete schema, bounded mmap/decode, retained ownership, no norm refold\n"
                   "PASS last expert's planar Q2_0 gate/up/down against independent values, payload budgets and identity\n";
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
