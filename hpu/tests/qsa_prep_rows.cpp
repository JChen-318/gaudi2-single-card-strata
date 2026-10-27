#include "strata/hpu/host_recipe.hpp"
#include "strata/hpu/model_abi.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
using namespace strata::hpu;
namespace {
void define(GraphBuilder& g,uint32_t rows) {
 const auto q=g.tensor({"q",DType::F32,{rows,24,256}}),k=g.tensor({"k",DType::F32,{rows,2,256}}),v=g.tensor({"v",DType::F32,{rows,2,256}});
 const auto qg=g.tensor({"qg",DType::F32,{256}}),kg=g.tensor({"kg",DType::F32,{256}}),rope=g.tensor({"rope",DType::F32,{rows,64}});
 const auto query=g.tensor({"query",DType::F32,{rows,24,256}}),ks=g.tensor({"ks",DType::F32,{rows,2,256}}),vs=g.tensor({"vs",DType::F32,{rows,2,256}});
 const auto kn=g.tensor({"kn",DType::F32,{rows,2,256}}),vn=g.tensor({"vn",DType::F32,{rows,2,256}});
 abi::QsaPrep a{256,24,2,64,1e-6f,int32_t(rows)};const TensorId in[]={q,k,v,qg,kg,rope},out[]={query,ks,vs,kn,vn};g.op("strata_qsa_prep_f32",in,out,&a,sizeof(a));
}
}
int main(int argc,char** argv) {
 try {
  const bool device=argc==2 && std::string(argv[1])=="--device";
  if(!device) {
   configure_kernel_libraries();check(synInitialize(),"prep offline init");
   for(uint32_t rows:{1u,32u,64u,128u}) {GraphBuilder g(OfflineCompilation{});define(g,rows);std::cout<<"PASS offline QSA prep rows="<<rows<<" workspace="<<g.compile_offline("qsa_prep_rows_"+std::to_string(rows))<<'\n';}
   check(synDestroy(),"prep offline destroy");return 0;
  }
  Device d;GraphBuilder one(d);define(one,1);HostRecipe reference(d,one.compile("qsa_prep_row_reference"),{},{"query","ks","vs","kn","vn"});
  const std::vector<float> gamma(256,1);reference.set("qg",gamma);reference.set("kg",gamma);
  for(uint32_t rows:{32u,64u,128u}) {
   GraphBuilder g(d);define(g,rows);HostRecipe batch(d,g.compile("qsa_prep_window"),{},{"query","ks","vs","kn","vn"});batch.set("qg",gamma);batch.set("kg",gamma);
   for(unsigned phase=0;phase<3;++phase) {
    std::vector<float> q(rows*24*256),k(rows*2*256),v(k.size()),rope(rows*64);
    for(auto* values:{&q,&k,&v}) for(uint64_t i=0;i<values->size();++i) (*values)[i]=phase==1?0:float(int((i*19+i/256*13)%257)-128)/(phase==2?16777216:128);
    for(uint32_t row=0;row<rows;++row) for(uint32_t pair=0;pair<32;++pair) {const float angle=float(row+9)*float(pair+1)/1024;rope[row*64+pair]=std::cos(angle);rope[row*64+32+pair]=std::sin(angle);}
    batch.set("q",q);batch.set("k",k);batch.set("v",v);batch.set("rope",rope);batch.run();
    for(uint32_t row=0;row<rows;++row) {
     reference.set("q",std::span<const float>(q).subspan(uint64_t(row)*24*256,24*256));reference.set("k",std::span<const float>(k).subspan(uint64_t(row)*2*256,2*256));
     reference.set("v",std::span<const float>(v).subspan(uint64_t(row)*2*256,2*256));reference.set("rope",std::span<const float>(rope).subspan(uint64_t(row)*64,64));reference.run();
     for(const std::string name:{"query","ks","vs","kn","vn"}) {
      const auto expected=reference.output(name),actual=batch.output(name).subspan(uint64_t(row)*expected.size(),expected.size());
      if(std::memcmp(actual.data(),expected.data(),actual.size_bytes()))throw std::runtime_error("batched QSA prep differs from singleton at "+name+" row="+std::to_string(row));
     }
    }
    std::cout<<"PASS device QSA prep rows="<<rows<<" phase="<<phase<<" all5_outputs_bit_exact=1\n"<<std::flush;
   }
  }
 } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
