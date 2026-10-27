#include "strata/hpu/fp8_colpair.hpp"
#include "strata/hpu/expert_format.hpp"
#include <array>
#include <iostream>
using namespace strata::hpu;
int main() {
    try {
        for(uint64_t rows:{128u,256u,1280u,2560u}) for(uint64_t columns:{128u,640u,2560u}) {
            const auto count=rows*columns;std::vector<uint8_t> storage(count+32),source(count);
            for(uint64_t i=0;i<count;++i) source[i]=storage[i+16]=uint8_t((i*73+(i>>7)*31)%251);
            std::fill_n(storage.begin(),16,0xa5);std::fill(storage.end()-16,storage.end(),0x5a);
            pack_fp8_colpair(std::span(storage).subspan(16,count),rows,columns);
            // Inverse walk in physical order: decode row/column coordinates independently of the
            // packer's loops, and check every byte against the untouched row-major reference.
            for(uint64_t i=0;i<count;++i) {
                const auto block=i/(128*columns),within=i%(128*columns),pair=within/256,lane=(within%256)/2;
                const auto row=block*128+lane/2+(lane%2)*64,col=pair*2+i%2;
                if(storage[i+16]!=source[row*columns+col]) throw std::runtime_error("column-pair layout inverse mismatch");
            }
            for(size_t i=0;i<16;++i) if(storage[i]!=0xa5 || storage[count+16+i]!=0x5a) throw std::runtime_error("layout overrun");
        }
        for(const auto& shape:{std::pair{0u,128u},std::pair{64u,128u},std::pair{128u,0u},std::pair{128u,64u}}) {
            bool rejected=false;try {std::vector<uint8_t> x(uint64_t(shape.first)*shape.second);pack_fp8_colpair(x,shape.first,shape.second);}
            catch(const std::invalid_argument&) {rejected=true;}
            if(!rejected) throw std::runtime_error("invalid layout geometry accepted");
        }
        auto plain=fp8_expert_format(2560,640),paired=plain;paired.fp8_column_pairs=true;
        if(plain==paired || plain.bytes!=paired.bytes) throw std::runtime_error("layout policy identity/accounting mismatch");
        std::cout<<"PASS column-pair inverse, guards, geometry and format identity\n";
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
