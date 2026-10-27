// Hardware-only packed upload/head acceptance. This program acquires a card.
#include "strata/hpu/dense_weights.hpp"
#include "strata/hpu/model_head.hpp"
#include "strata/hpu/packed_quant.hpp"
#include "strata/kernels/f16_bits.hpp"
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
using namespace strata::hpu;
namespace {
void require(bool ok,const char* why){if(!ok) throw std::runtime_error(why);}
int signed_byte(uint8_t x){return x<128?x:int(x)-256;}
float oracle(std::span<const uint8_t> weights,uint32_t type,const Q8Activation& activation) {
    long double result=0;
    for(uint32_t block=0;block<2;++block) {
        const auto at=type==42?0:block*packed_block_bytes(type);
        const long double d=strata::kernels::f32_from_f16(uint16_t(weights[at])|(uint16_t(weights[at+1])<<8));
        long double dot=0,code_sum=0;
        for(uint32_t j=0;j<32;++j) {
            int code;
            if(type==2) code=(weights[at+2+j%16]>>(j<16?0:4))&15;
            else if(type==3 || type==6 || type==7) {
                const auto high=at+((type==3 || type==7)?4:2),qs=high+(type==3?0:4);
                code=(weights[qs+j%16]>>(j<16?0:4))&15;
                if(type!=3) code|=((weights[high+j/8]>>(j%8))&1)<<4;
            }
            else if(type==8) code=signed_byte(weights[at+2+j]);
            else {const auto n=block*32+j;code=((weights[at+2+n/4]>>((n%4)*2))&3)-1;}
            dot+=code*signed_byte(activation.codes[block*32+j]);
            code_sum+=signed_byte(activation.codes[block*32+j]);
        }
        result+=d*(dot*activation.metadata[block*2]-((type==2?8:type==6?16:0)*activation.metadata[block*2+1]));
        if(type==3 || type==7) {const long double m=strata::kernels::f32_from_f16(uint16_t(weights[at+2])|(uint16_t(weights[at+3])<<8));result+=m*activation.metadata[block*2]*code_sum;}
    }
    return float(result);
}

// K-quant head fixture: 256-value superblocks with a half2 (d,dmin), eight 6-bit
// scale/minimum pairs in the get_scale_min_k4 layout and 4-bit (Q4_K) or 5-bit
// (Q5_K, whose fifth bit sits in qh byte l at bit 2*(g/2)+(g&1)) codes. This
// mirrors the pinned ggml decoder validated on the host by packed_kquant.
std::vector<uint8_t> k_fill(uint32_t type,uint32_t columns,uint32_t rows) {
    const uint32_t size=packed_block_bytes(type),qs=type==12?16u:48u,blocks=columns/256;
    std::vector<uint8_t> bytes(uint64_t(rows)*blocks*size,0);
    auto half=[&](size_t at,float value){const auto bits=strata::kernels::f16_from_f32(value);bytes[at]=uint8_t(bits);bytes[at+1]=uint8_t(bits>>8);};
    for(uint32_t row=0;row<rows;++row) for(uint32_t sb=0;sb<blocks;++sb) {
        const size_t at=(uint64_t(row)*blocks+sb)*size;
        half(at,float(int(row%5)+1)/8.f);half(at+2,float(int(row%3))/32.f);
        for(uint32_t g=0;g<8;++g) {
            const int scale=int((row*7+g*13+sb*5+3)%64),minimum=int((row*5+g*11+sb*3+1)%64);
            if(g<4) {bytes[at+4+g]|=uint8_t(scale);bytes[at+8+g]|=uint8_t(minimum);continue;}
            const uint32_t h=g-4;
            bytes[at+12+h]|=uint8_t(uint32_t(scale&15)|uint32_t(minimum&15)<<4);
            bytes[at+4+h]|=uint8_t(uint32_t(scale>>4)<<6);
            bytes[at+8+h]|=uint8_t(uint32_t(minimum>>4)<<6);
        }
        for(uint32_t g=0;g<8;++g) for(uint32_t l=0;l<32;++l) {
            const int code=int(((sb*256+g*32+l)*19+row*7+sb)%uint32_t(type==13?32:16));
            bytes[at+qs+(g/2)*32+l]|=uint8_t(uint32_t(code&15)<<((g&1)*4));
            if(type==13 && code>=16) bytes[at+16+l]|=uint8_t(uint32_t(1u)<<g);
        }
    }
    return bytes;
}
float k_head_oracle(std::span<const uint8_t> bytes,uint32_t type,uint32_t columns,uint32_t token,uint32_t row,const Q8Activation& activation) {
    const uint32_t size=packed_block_bytes(type),qs=type==12?16u:48u,blocks=columns/256;
    long double total=0;
    for(uint32_t sb=0;sb<blocks;++sb) {
        const size_t at=(uint64_t(row)*blocks+sb)*size;
        const long double d=strata::kernels::f32_from_f16(uint16_t(bytes[at])|(uint16_t(bytes[at+1])<<8));
        const long double dmin=strata::kernels::f32_from_f16(uint16_t(bytes[at+2])|(uint16_t(bytes[at+3])<<8));
        long double sum_d=0,sum_m=0;
        for(uint32_t g=0;g<8;++g) {
            const uint32_t h=g<4?g:g-4,chunk=g/2;
            const int scale=g<4?(bytes[at+4+g]&63):int((bytes[at+12+h]&15)|((bytes[at+4+h]>>6)<<4));
            const int minimum=g<4?(bytes[at+8+g]&63):int((bytes[at+12+h]>>4)|((bytes[at+8+h]>>6)<<4));
            int dot=0,code_sum=0;
            for(uint32_t l=0;l<32;++l) {
                int code=int(bytes[at+qs+chunk*32+l]>>((g&1)*4))&15;
                if(type==13) code|=int((bytes[at+16+l]>>g)&1u)<<4;
                const int a=signed_byte(activation.codes[uint64_t(token)*columns+sb*256+g*32+l]);
                dot+=code*a;code_sum+=a;
            }
            const long double d8=activation.metadata[(uint64_t(token)*columns/32+sb*8+g)*2];
            sum_d+=d8*dot*scale;sum_m+=d8*code_sum*minimum;
        }
        total+=d*sum_d-dmin*sum_m;
    }
    return float(total);
}
}
int main() {
    try {
        Device device;device.allocation_limits(64ull<<20,8ull<<20);
        for(auto type:{2u,3u,6u,7u,8u,42u}) {
            constexpr uint32_t columns=64,rows=5;const auto row_bytes=packed_row_bytes(type,columns);
            auto bytes=std::make_shared<std::vector<uint8_t>>(row_bytes*rows);
            for(size_t i=0;i<bytes->size();++i) (*bytes)[i]=uint8_t(i*17+3);
            for(uint32_t row=0;row<rows;++row) for(uint64_t b=0;b<row_bytes;b+=packed_block_bytes(type)) {
                const auto scale=strata::kernels::f16_from_f32(float(int(row)-2)/4);const auto at=row*row_bytes+b;
                (*bytes)[at]=uint8_t(scale);(*bytes)[at+1]=uint8_t(scale>>8);
                if(type==3 || type==7) {const auto m=strata::kernels::f16_from_f32(float(int(row)-1)/16);(*bytes)[at+2]=uint8_t(m);(*bytes)[at+3]=uint8_t(m>>8);}
            }
            const auto source=WeightSlice::raw_rows(bytes,*bytes,type,columns,rows);
            DenseWeights uploaded(device,8192,512);uploaded.packed("weight",source);
            const std::array<int32_t,3> ids{4,0,2};uploaded.packed("subset",source,ids);
            require(uploaded.payload_bytes()==row_bytes*8,"packed full/subset HBM charge used decoded BF16 size");
            Stream copy(device);HostBuffer back(device,row_bytes*rows);
            download(copy,back,*uploaded.binding("weight").buffer,row_bytes*rows);Event done(device);done.record(copy);done.synchronize();
            require(!std::memcmp(back.data(),bytes->data(),bytes->size()),"packed source bytes changed during bounded H2D staging");
            ModelHead full(device,uploaded.binding("weight"),columns,rows,type);
            auto subset_binding=uploaded.binding("subset");subset_binding.name="weight";ModelHead subset(device,subset_binding,columns,ids.size(),type);
            for(uint32_t pass=0;pass<3;++pass) {
                std::vector<float> input(columns);for(size_t i=0;i<input.size();++i) input[i]=float((int(i%7)-3)*int(pass+1))/8;
                const auto activation=quantize_q8_activation(input,columns);
                const auto all=full.run(input),selected=subset.run(input);
                for(uint32_t row=0;row<rows;++row) {
                    const auto expected=oracle(std::span(*bytes).subspan(row*row_bytes,row_bytes),type,activation);
                    require(std::abs(all[row]-expected)<=std::max(0.002f,std::abs(expected)*0.00002f),"packed HPU head differs from independent integer/raw-sum oracle");
                }
                for(size_t i=0;i<ids.size();++i) require(std::abs(selected[i]-all[uint32_t(ids[i])])<=0.002f,"packed subset row order or mutable activation state changed");
            }
            for(uint32_t batch:{3u,7u,1u,7u,3u}) {
                std::vector<float> input(batch*columns);for(size_t i=0;i<input.size();++i) input[i]=float(int((i*7+batch)%37)-18)/16;
                const auto all=full.run_rows(input),selected=subset.run_rows(input);
                require(all.size()==batch*rows && selected.size()==batch*ids.size(),"dynamic packed head shape mismatch");
                for(uint32_t token=0;token<batch;++token) {
                    const auto row=std::span<const float>(input).subspan(token*columns,columns);const auto activation=quantize_q8_activation(row,columns);
                    for(uint32_t vocabulary=0;vocabulary<rows;++vocabulary) {
                        const auto expected=oracle(std::span(*bytes).subspan(vocabulary*row_bytes,row_bytes),type,activation);
                        require(std::abs(all[token*rows+vocabulary]-expected)<=std::max(0.002f,std::abs(expected)*0.00002f),"dynamic head changed independent per-row packed contract");
                    }
                    for(size_t i=0;i<ids.size();++i) require(std::abs(selected[token*ids.size()+i]-all[token*rows+ids[i]])<=0.002f,"dynamic subset order mismatch");
                }
                const auto scalar=full.run(std::span<const float>(input).first(columns));
                for(uint32_t vocabulary=0;vocabulary<rows;++vocabulary) require(std::abs(scalar[vocabulary]-all[vocabulary])<=0.002f,"run() lost original constructor batch after run_rows()");
            }
            uploaded.requantized_q4("q4_full",source);uploaded.requantized_q4("q4_subset",source,ids);
            const auto q4_row=packed_row_bytes(2,columns);
            std::vector<uint8_t> q4_reference(rows*q4_row);source.convert_rows_q4(0,q4_reference);
            HostBuffer q4_back(device,rows*q4_row);download(copy,q4_back,*uploaded.binding("q4_full").buffer,q4_back.size());done.record(copy);done.synchronize();
            require(!std::memcmp(q4_back.data(),q4_reference.data(),q4_reference.size()),"Q4 full conversion changed during bounded DMA");
            HostBuffer q4_selected(device,ids.size()*q4_row);download(copy,q4_selected,*uploaded.binding("q4_subset").buffer,q4_selected.size());done.record(copy);done.synchronize();
            for(size_t i=0;i<ids.size();++i) require(!std::memcmp(static_cast<const uint8_t*>(q4_selected.data())+i*q4_row,q4_reference.data()+ids[i]*q4_row,q4_row),"requantized Q4 subset order changed during DMA");
            auto q4_binding=uploaded.binding("q4_full");q4_binding.name="weight";ModelHead q4_head(device,q4_binding,columns,rows,2);
            std::vector<float> input(columns);for(size_t i=0;i<input.size();++i) input[i]=float(int(i%17)-8)/16;
            const auto activation=quantize_q8_activation(input,columns);const auto result=q4_head.run(input);
            for(uint32_t row=0;row<rows;++row) {
                const auto expected=oracle(std::span(q4_reference).subspan(row*q4_row,q4_row),2,activation);
                require(std::abs(result[row]-expected)<=std::max(0.002f,std::abs(expected)*0.00002f),"requantized Q4 head differs from integer/raw-sum oracle");
            }
        }
        // K-quant output head on the shared/head Q8_1 contract: raw superblock
        // bytes stay packed and group minima multiply the quantized codes.
        for(auto type:{12u,13u}) {
            constexpr uint32_t columns=512,rows=4;const auto row_bytes=packed_row_bytes(type,columns);
            const auto bytes=k_fill(type,columns,rows);
            validate_packed_rows(bytes,type,columns,rows);
            const std::array<int32_t,3> ids{3,1,0};
            const auto owner=std::make_shared<std::vector<uint8_t>>(bytes);
            const auto source=WeightSlice::raw_rows(owner,*owner,type,columns,rows);
            DenseWeights uploaded(device,row_bytes*(rows+ids.size())+1024,4096);
            uploaded.packed("weight",source);uploaded.packed("subset",source,ids);
            require(uploaded.payload_bytes()==row_bytes*(rows+ids.size()),"K-quant head charge used a decoded size");
            Stream copy(device);HostBuffer back(device,bytes.size());
            download(copy,back,*uploaded.binding("weight").buffer,row_bytes*rows);
            Event done(device);done.record(copy);done.synchronize();
            require(!std::memcmp(back.data(),bytes.data(),bytes.size()),"K-quant head bytes changed during bounded H2D staging");
            ModelHead head(device,uploaded.binding("weight"),columns,rows,type);
            auto subset_binding=uploaded.binding("subset");subset_binding.name="weight";
            ModelHead subset(device,subset_binding,columns,ids.size(),type);
            for(uint32_t batch:{1u,3u}) {
                std::vector<float> input(uint64_t(batch)*columns);
                for(size_t i=0;i<input.size();++i) input[i]=float(int((i*11+batch)%53)-26)/8;
                const auto activation=quantize_q8_activation(input,columns);
                const auto all=batch==1?head.run(input):head.run_rows(input);
                const auto selected=batch==1?subset.run(input):subset.run_rows(input);
                require(all.size()==uint64_t(batch)*rows && selected.size()==batch*ids.size(),"K-quant head output shape");
                for(uint32_t token=0;token<batch;++token) {
                    for(uint32_t row=0;row<rows;++row) {
                        const auto expected=k_head_oracle(std::span(bytes),type,columns,token,row,activation);
                        require(std::abs(all[uint64_t(token)*rows+row]-expected)<=std::max(.002f,std::abs(expected)*.00002f),
                            "HPU K-quant head differs from the independent superblock oracle");
                    }
                    for(size_t i=0;i<ids.size();++i) require(std::abs(selected[uint64_t(token)*ids.size()+i]-all[uint64_t(token)*rows+ids[i]])<=.002f,
                        "K-quant subset row order or mutable activation state changed");
                }
            }
            // Q5_K codes above 15 depend on the separate qh field: flipping that
            // field must change what the oracle decodes.
            if(type==13) {
                auto flipped=bytes;for(uint32_t i=16;i<32;++i) flipped[i]=uint8_t(flipped[i]^0xffu);
                std::vector<float> probe(columns);
                for(size_t i=0;i<probe.size();++i) probe[i]=float(int((i*11+3)%53)-26)/8;
                const auto activation=quantize_q8_activation(probe,columns);
                require(k_head_oracle(std::span(flipped),13u,columns,0,0,activation)!=k_head_oracle(std::span(bytes),13u,columns,0,0,activation),
                    "the Q5_K fifth-bit field never reached the oracle");
            }
        }
        const auto hbm=device.hbm_allocations(),pinned=device.pinned_allocations();require(hbm.used<=hbm.limit && pinned.used<=pinned.limit,"packed head exceeded allocation caps");
        std::cout<<"PASS real HPU packed Q2/Q4/Q5/Q8/Q4_K/Q5_K immutable upload bytes, role-correct raw-sum versus quantized-code-sum arithmetic, ordered subset and repeated mutable activation heads\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
