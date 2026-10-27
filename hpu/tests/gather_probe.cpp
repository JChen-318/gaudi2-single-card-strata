// Sparse QSA prefill gather throughput (tpc/qsa_gather_rows_fp8.c): `rows` window rows each gather `top` blocks of 4
// cells (both KV heads, K and V, FP8 codes and scales) from a cache of `cells` cells, at random sorted block ids.
// Prints device ms per launch and the bytes moved per second (codes read + written, scales, mask).
// Hardware only, not CTest.  strata-hpu-gather-probe [rows 1024] [top 513] [cells 131072] [iters 20]
#include "strata/hpu/recipe.hpp"
#include "strata/hpu/model_abi.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <vector>
using namespace strata::hpu;
int main(int argc,char** argv) {
    try {
        const uint32_t rows=argc>1?uint32_t(std::stoul(argv[1])):1024,top=argc>2?uint32_t(std::stoul(argv[2])):513;
        const uint32_t cells=argc>3?uint32_t(std::stoul(argv[3])):131072,iters=argc>4?uint32_t(std::stoul(argv[4])):20;
        const uint64_t hk=2,d=256,n=uint64_t(top)*4;
        Device device;
        DeviceBuffer kc(device,uint64_t(cells)*hk*d),vc(device,uint64_t(cells)*hk*d),ks(device,uint64_t(cells)*128),vs(device,uint64_t(cells)*128);
        DeviceBuffer ids(device,uint64_t(top)*rows*4),info(device,uint64_t(rows)*8+128),start(device,128);
        DeviceBuffer k8(device,n*hk*d*rows*2),v8(device,n*hk*d*rows*2),kss(device,n*hk*rows*4),vss(device,n*hk*rows*4),mask(device,n*rows*4);
        {
            Stream copy(device);
            HostBuffer h(device,uint64_t(top)*rows*4);auto* p=static_cast<int32_t*>(h.data());
            std::mt19937 rng(1);const uint32_t blocks=cells/4;
            for(uint32_t r=0;r<rows;++r) {
                std::vector<int32_t> pick(top);for(auto& b:pick) b=int32_t(rng()%blocks);std::sort(pick.begin(),pick.end());
                for(uint32_t k=0;k<top;++k) p[uint64_t(k)+uint64_t(r)*top]=pick[k];
            }
            upload(copy,ids,h,h.size());
            HostBuffer s(device,128);static_cast<float*>(s.data())[0]=float(cells);static_cast<float*>(s.data())[1]=0;upload(copy,start,s,128);
            HostBuffer in(device,uint64_t(rows)*8);for(uint32_t r=0;r<rows;++r) {static_cast<float*>(in.data())[2*r]=-1;static_cast<float*>(in.data())[2*r+1]=0;}
            upload(copy,info,in,in.size());copy.synchronize();
        }
        GraphBuilder g(device);
        const auto t=[&](const std::string& name,DType dtype,std::vector<uint64_t> shape) {return g.tensor({name,dtype,std::move(shape),true});};
        const auto a=t("kc",DType::F8,{cells,hk,d}),b=t("vc",DType::F8,{cells,hk,d}),c=t("ks",DType::F32,{cells,32}),e=t("vs",DType::F32,{cells,32});
        const auto i=t("ids",DType::I32,{rows,top}),inf=t("info",DType::F32,{rows,2}),st=t("start",DType::F32,{2});
        const auto o1=t("k8",DType::BF16,{rows,hk,n,d}),o2=t("v8",DType::BF16,{rows,hk,n,d}),o3=t("kss",DType::F32,{rows,hk,n}),o4=t("vss",DType::F32,{rows,hk,n});
        const auto o5=t("mask",DType::F32,{rows,n});
        const abi::QsaGatherRows args{int32_t(hk),int32_t(cells),int32_t(top),4,int32_t(rows)};
        {const TensorId in[]={a,b,c,e,i,inf,st},out[]={o1,o2,o3,o4,o5};g.op("strata_qsa_gather_rows_fp8",in,out,&args,sizeof(args));}
        auto recipe=g.compile("strata_gather_probe");
        const std::vector<Binding> bindings{{"kc",&kc},{"vc",&vc},{"ks",&ks},{"vs",&vs},{"ids",&ids},{"info",&info},{"start",&start},
                                            {"k8",&k8},{"v8",&v8},{"kss",&kss},{"vss",&vss},{"mask",&mask}};
        Stream stream(device);recipe->enqueue(stream,bindings);stream.synchronize();
        const auto begin=std::chrono::steady_clock::now();
        for(uint32_t k=0;k<iters;++k) recipe->enqueue(stream,bindings);
        stream.synchronize();
        const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()/iters;
        const double bytes=double(rows)*n*hk*(2*d*3+2*4*2)+double(rows)*n*4; // FP8 read, BF16 written
        std::cout<<"rows "<<rows<<" top "<<top<<" cells "<<cells<<": "<<ms<<" ms/launch, "<<bytes/ms/1e6<<" GB/s moved, "
                 <<ms/rows*1000<<" us/row\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
