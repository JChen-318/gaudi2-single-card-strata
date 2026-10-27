// Four independent head dots in AVX2 lanes. Each lane keeps the scalar dimension
// order; the final sum keeps head order. No horizontal reassociation or FMA.
#pragma once
#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#endif
namespace strata::hpu {
class ExactQsaIndex {
    uint32_t width_,heads_;std::vector<double> packed_;
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    __attribute__((target("avx2,no-fma")))
    double avx2(const float* pooled) const {
        double score=0;
        for(uint32_t head=0;head<heads_;head+=4) {
            const double* query=packed_.data()+uint64_t(head/4)*width_*4;
            __m256d sum=_mm256_setzero_pd();
            for(uint32_t d=0;d<width_;++d)
                sum=_mm256_add_pd(sum,_mm256_mul_pd(_mm256_loadu_pd(query+uint64_t(d)*4),_mm256_set1_pd(double(pooled[d]))));
            alignas(32) double dots[4];_mm256_store_pd(dots,sum);
            for(uint32_t lane=0;lane<4 && head+lane<heads_;++lane) score+=std::max(dots[lane],0.0);
        }
        return score;
    }
    __attribute__((target("avx2,no-fma")))
    double avx2_interleaved(const float* pooled) const {
        double score=0;uint32_t head=0;
        // Four independent head groups hide the add latency on AVX2 hosts.
        // Each dot still visits dimensions in scalar order; heads are summed
        // in their original order after the vector accumulators are stored.
        for(;head+16<=heads_;head+=16) {
            const double* q=packed_.data()+uint64_t(head/4)*width_*4;
            const uint64_t stride=uint64_t(width_)*4;
            __m256d a=_mm256_setzero_pd(),b=a,c=a,e=a;
            for(uint32_t d=0;d<width_;++d) {
                const auto p=_mm256_set1_pd(double(pooled[d]));const uint64_t offset=uint64_t(d)*4;
                a=_mm256_add_pd(a,_mm256_mul_pd(_mm256_loadu_pd(q+offset),p));
                b=_mm256_add_pd(b,_mm256_mul_pd(_mm256_loadu_pd(q+stride+offset),p));
                c=_mm256_add_pd(c,_mm256_mul_pd(_mm256_loadu_pd(q+2*stride+offset),p));
                e=_mm256_add_pd(e,_mm256_mul_pd(_mm256_loadu_pd(q+3*stride+offset),p));
            }
            alignas(32) double dots[16];
            _mm256_store_pd(dots,a);_mm256_store_pd(dots+4,b);_mm256_store_pd(dots+8,c);_mm256_store_pd(dots+12,e);
            for(double dot:dots) score+=std::max(dot,0.0);
        }
        for(;head<heads_;head+=4) {
            const double* q=packed_.data()+uint64_t(head/4)*width_*4;__m256d sum=_mm256_setzero_pd();
            for(uint32_t d=0;d<width_;++d)
                sum=_mm256_add_pd(sum,_mm256_mul_pd(_mm256_loadu_pd(q+uint64_t(d)*4),_mm256_set1_pd(double(pooled[d]))));
            alignas(32) double dots[4];_mm256_store_pd(dots,sum);
            for(uint32_t lane=0;lane<4 && head+lane<heads_;++lane) score+=std::max(dots[lane],0.0);
        }
        return score;
    }
    __attribute__((target("avx2,no-fma")))
    void avx2_blocks(const float* pooled,uint32_t blocks,double* output) const {
        uint32_t block=0;
        for(;block+4<=blocks;block+=4) {
            const float* p=pooled+uint64_t(block)*width_;double scores[4]{};
            for(uint32_t head=0;head<heads_;head+=4) {
                const double* query=packed_.data()+uint64_t(head/4)*width_*4;
                __m256d a=_mm256_setzero_pd(),b=a,c=a,e=a;
                for(uint32_t d=0;d<width_;++d) {
                    const auto q=_mm256_loadu_pd(query+uint64_t(d)*4);
                    a=_mm256_add_pd(a,_mm256_mul_pd(q,_mm256_set1_pd(double(p[d]))));
                    b=_mm256_add_pd(b,_mm256_mul_pd(q,_mm256_set1_pd(double(p[width_+d]))));
                    c=_mm256_add_pd(c,_mm256_mul_pd(q,_mm256_set1_pd(double(p[2*uint64_t(width_)+d]))));
                    e=_mm256_add_pd(e,_mm256_mul_pd(q,_mm256_set1_pd(double(p[3*uint64_t(width_)+d]))));
                }
                alignas(32) double dots[16];
                _mm256_store_pd(dots,a);_mm256_store_pd(dots+4,b);_mm256_store_pd(dots+8,c);_mm256_store_pd(dots+12,e);
                for(uint32_t lane=0;lane<4 && head+lane<heads_;++lane)
                    for(uint32_t row=0;row<4;++row) scores[row]+=std::max(dots[4*row+lane],0.0);
            }
            std::copy_n(scores,4,output+block);
        }
        for(;block<blocks;++block) output[block]=avx2_interleaved(pooled+uint64_t(block)*width_);
    }
#endif
public:
    ExactQsaIndex(std::span<const float> query,uint32_t width,uint32_t heads):width_(width),heads_(heads),packed_(uint64_t((heads+3)/4)*width*4) {
        if(!width || !heads || query.size()!=uint64_t(width)*heads) throw std::invalid_argument("exact QSA query shape");
        for(uint32_t h=0;h<heads;++h) for(uint32_t d=0;d<width;++d)
            packed_[uint64_t(h/4)*width*4+uint64_t(d)*4+h%4]=double(query[uint64_t(h)*width+d]);
    }
    double scalar(const float* pooled) const {
        double score=0;
        for(uint32_t head=0;head<heads_;++head) {
            double sum=0;for(uint32_t d=0;d<width_;++d) sum+=packed_[uint64_t(head/4)*width_*4+uint64_t(d)*4+head%4]*double(pooled[d]);
            score+=std::max(sum,0.0);
        }
        return score;
    }
    double score(const float* pooled) const {
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
        if(__builtin_cpu_supports("avx2")) return avx2(pooled);
#endif
        return scalar(pooled);
    }
    double score_interleaved(const float* pooled) const {
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
        if(__builtin_cpu_supports("avx2")) return avx2_interleaved(pooled);
#endif
        return scalar(pooled);
    }
    void score_blocks(const float* pooled,uint32_t blocks,std::span<double> output) const {
        if(output.size()!=blocks) throw std::invalid_argument("exact QSA score block shape");
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
        if(__builtin_cpu_supports("avx2")) {avx2_blocks(pooled,blocks,output.data());return;}
#endif
        for(uint32_t block=0;block<blocks;++block) output[block]=scalar(pooled+uint64_t(block)*width_);
    }
};
}
