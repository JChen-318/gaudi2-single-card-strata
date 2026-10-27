#include "strata/hpu/mtp_weights.hpp"
#include <iostream>
int main(int argc,char** argv) {
    if(argc!=2) {std::cerr<<"usage: strata-hpu-mtp-info original-mtp-rt-directory\n";return 2;}
    try {
        const strata::hpu::MtpWeights weights(argv[1]);
        std::cout<<"tensors="<<weights.inventory().size()<<" dense_bf16_f32_payload="<<weights.dense_payload_bytes()<<" no_device_acquired\n";
        for(const auto& t:weights.inventory()) std::cout<<t.name<<' '<<t.kind<<' '<<t.rows<<' '<<t.columns<<' '<<t.bytes<<'\n';
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
