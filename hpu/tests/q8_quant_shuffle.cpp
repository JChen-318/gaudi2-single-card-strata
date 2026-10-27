#include "strata/hpu/host_recipe.hpp"
#include "strata/hpu/model_abi.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
using namespace strata::hpu;
int main() {
    try {
        Device device;Stream copy(device);
        for(uint32_t columns:{32u,64u,128u,640u,2560u}) for(uint32_t rows:{1u,11u}) {
            const uint64_t codes_bytes=uint64_t(rows)*columns*256,scale_bytes=uint64_t(rows)*(columns/32)*64*4;
            std::unique_ptr<HostRecipe> graphs[2];std::unique_ptr<DeviceBuffer> codes[2],scales[2];
            for(int mode=0;mode<2;++mode) {
                GraphBuilder g(device);
                const auto x=g.tensor({"x",DType::F32,{rows,columns}}),q=g.tensor({"codes",DType::U8,{rows,columns,256}}),d=g.tensor({"scales",DType::F32,{rows,columns/32,64}});
                codes[mode]=std::make_unique<DeviceBuffer>(device,codes_bytes);scales[mode]=std::make_unique<DeviceBuffer>(device,scale_bytes);
                const Binding bindings[]={{"codes",codes[mode].get()},{"scales",scales[mode].get()}};
                const TensorId in[]={x},out[]={q,d};abi::Q8Quant args{int32_t(columns),int32_t(rows)};
                g.op(mode?"strata_q8_quant_shuffle_f32":"strata_q8_quant_f32",in,out,&args,sizeof(args));
                graphs[mode]=std::make_unique<HostRecipe>(device,g.compile(std::string(mode?"q8_shuffle_":"q8_original_")+std::to_string(columns)+"_"+std::to_string(rows)),bindings,std::vector<std::string>{});
            }
            HostBuffer read_codes(device,codes_bytes),read_scales(device,scale_bytes);
            for(int phase=0;phase<4;++phase) {
                std::vector<float> x(uint64_t(rows)*columns);
                for(size_t i=0;i<x.size();++i) {
                    const int j=int(i%32),block=int(i/32);
                    x[i]=phase==0?0:phase==1?(j==31?127.0f:float(j-15)+0.5f):
                        phase==2?float(int((i*17+3)%97)-48)/13:std::ldexp(float(int((i*19+7)%127)-63),block%41-20);
                }
                std::vector<uint8_t> reference_codes;std::vector<uint8_t> reference_scales;
                for(int mode=0;mode<2;++mode) {
                    graphs[mode]->set("x",x);graphs[mode]->run();
                    download(copy,read_codes,*codes[mode],codes_bytes);download(copy,read_scales,*scales[mode],scale_bytes);copy.synchronize();
                    const auto* q=static_cast<const uint8_t*>(read_codes.data());const auto* d=static_cast<const float*>(read_scales.data());
                    if(mode==0) {
                        reference_codes.assign(q,q+codes_bytes);
                        const auto* b=static_cast<const uint8_t*>(read_scales.data());reference_scales.assign(b,b+scale_bytes);
                    } else if(std::memcmp(q,reference_codes.data(),codes_bytes) || std::memcmp(d,reference_scales.data(),scale_bytes))
                        throw std::runtime_error("Q8 shuffle code/scale bits differ from original");
                    for(size_t i=0;i<x.size();++i) {
                        const auto block=i/32;float peak=0;for(size_t j=0;j<32;++j) peak=std::max(peak,std::abs(x[block*32+j]));
                        const float expected=peak/127;
                        if(std::abs(d[block*64]-expected)>1e-30f+2e-6f*expected) throw std::runtime_error("Q8 scale oracle mismatch");
                        for(unsigned lane=0;lane<64;++lane) if(std::bit_cast<uint32_t>(d[block*64+lane])!=std::bit_cast<uint32_t>(d[block*64]))
                            throw std::runtime_error("Q8 scale lanes not broadcast");
                        const auto code=int8_t(q[i*256]);
                        if(phase<=1 && code!=int8_t(std::round(x[i]))) throw std::runtime_error("zero/half-away rounding oracle mismatch");
                        for(unsigned lane=0;lane<256;++lane) if(q[i*256+lane]!=uint8_t(code)) throw std::runtime_error("Q8 code lanes not broadcast");
                    }
                }
                std::cout<<"PASS columns="<<columns<<" rows="<<rows<<" phase="<<phase<<" bit_exact=1\n";
            }
        }
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
