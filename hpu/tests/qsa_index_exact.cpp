#include "strata/hpu/qsa_index_exact.hpp"
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
using namespace strata::hpu;
namespace {
double reference(std::span<const float> q,const float* pooled,uint32_t width,uint32_t heads) {
    double score=0;
    for(uint32_t head=0;head<heads;++head) {
        double sum=0;for(uint32_t d=0;d<width;++d) sum+=double(q[uint64_t(head)*width+d])*double(pooled[d]);
        score+=std::max(sum,0.0);
    }
    return score;
}
}
int main(int argc,char**) {
 try {
    std::mt19937 rng(31);uint64_t compared=0;
    for(uint32_t width:{1u,7u,64u,128u,256u}) for(uint32_t heads:{1u,3u,4u,5u,15u,16u,17u,31u,32u,33u}) {
        for(uint32_t phase=0;phase<20;++phase) {
            std::vector<float> query(uint64_t(width)*heads),pooled(width);
            for(auto* v:{&query,&pooled}) for(auto& x:*v) {
                const int mantissa=int(rng()%2001)-1000,exponent=int(rng()%81)-40;
                x=phase==0?0:std::ldexp(float(mantissa)/1024,exponent);
            }
            const ExactQsaIndex packed(query,width,heads);
            const auto want=reference(query,pooled.data(),width,heads),scalar=packed.scalar(pooled.data()),actual=packed.score(pooled.data());
            const auto interleaved=packed.score_interleaved(pooled.data());
            if(std::memcmp(&want,&scalar,8) || std::memcmp(&want,&actual,8) || std::memcmp(&want,&interleaved,8)) throw std::runtime_error("exact QSA score mismatch");
            std::vector<float> pools(uint64_t(width)*7);std::vector<double> scores(7);
            for(uint32_t block=0;block<7;++block) for(uint32_t d=0;d<width;++d) pools[uint64_t(block)*width+d]=pooled[d]*float(int(block)-3);
            packed.score_blocks(pools.data(),7,scores);
            for(uint32_t block=0;block<7;++block) {
                const auto expected=reference(query,pools.data()+uint64_t(block)*width,width,heads);
                if(std::memcmp(&expected,&scores[block],8)) throw std::runtime_error("exact QSA interleaved block mismatch");
            }
            ++compared;
        }
    }
    std::cout<<"PASS exact QSA index "<<compared<<" bit-exact scalar/AVX2/interleaved scores, widths1..256 heads1..33, mixed exponents\n";
    if(argc>1) {
        constexpr uint32_t width=128,heads=4,blocks=576;
        std::vector<float> query(width*heads),pool(width*blocks);
        for(auto* v:{&query,&pool}) for(auto& x:*v) x=float(int(rng()%2001)-1000)/1024;
        const ExactQsaIndex packed(query,width,heads);volatile double sink=0;
        std::vector<double> scores(blocks);
        for(unsigned mode:{0u,1u,2u,3u}) {
            const auto start=std::chrono::steady_clock::now();
            for(uint32_t run=0;run<200;++run) {
              if(mode==3) {packed.score_blocks(pool.data(),blocks,scores);sink=scores.back();continue;}
              for(uint32_t b=0;b<blocks;++b) {
                const double v=mode==2?packed.score_interleaved(pool.data()+b*width):mode==1?packed.score(pool.data()+b*width):reference(query,pool.data()+b*width,width,heads);sink=v;
              }
            }
            const double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/200;
            std::cout<<(mode==3?"blocks4":mode==2?"interleaved":mode==1?"exact_simd":"scalar")<<" us/576-block-index-heads4="<<us<<" sink="<<sink<<'\n';
        }
    }
 } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
