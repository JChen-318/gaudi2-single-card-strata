#pragma once
// Test-only semantic emulator. It runs the original kernel C sources on CPU;
// it does not model TPC scheduling, memory ordering, ISA math accuracy or speed.
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

// Kernel unroll hints are scheduling only.
#define STRATA_UNROLL
namespace emulated {
using int5 = std::array<int,5>;
inline int5 offset{}, extent{};
inline int5 get_index_space_offset() { return offset; }
inline int5 get_index_space_size() { return extent; }
// The host memory model is coherent; device execution must separately validate ASO.
inline void aso() {}
inline int5 operator+(int5 a,const int5& b) { for(unsigned i=0;i<5;++i) a[i]+=b[i]; return a; }
struct float64 {
    std::array<float,64> lanes{};
    float64(float x=0) { lanes.fill(x); }
    float64& operator+=(const float64& b) { for(unsigned i=0;i<64;++i) lanes[i]+=b.lanes[i]; return *this; }
};
#define EMULATED_OP(op) inline float64 operator op(const float64& a,const float64& b) { \
    float64 out; for(unsigned i=0;i<64;++i) out.lanes[i]=a.lanes[i] op b.lanes[i]; return out; }
EMULATED_OP(+)
EMULATED_OP(-)
EMULATED_OP(*)
EMULATED_OP(/)
#undef EMULATED_OP
inline float64 operator-(const float64& a) { return float64(0)-a; }
inline float64 v_exp_f32(const float64& x) { float64 out; for(unsigned i=0;i<64;++i) out.lanes[i]=std::exp(x.lanes[i]); return out; }
inline float64 v_log_f32(const float64& x) { float64 out; for(unsigned i=0;i<64;++i) out.lanes[i]=std::log(x.lanes[i]); return out; }
inline float64 v_rsqrt_f32(const float64& x) { float64 out; for(unsigned i=0;i<64;++i) out.lanes[i]=1/std::sqrt(x.lanes[i]); return out; }
inline float64 v_f32_sel_less_f32_b(const float64& a,const float64& b,const float64& yes,const float64& no) {
    float64 out; for(unsigned i=0;i<64;++i) out.lanes[i]=a.lanes[i]<b.lanes[i]?yes.lanes[i]:no.lanes[i]; return out;
}
struct tensor {
    std::vector<float>* data;
    std::vector<uint64_t> dims;
    bool contains(const int5& position) const {
        for(size_t i=0;i<5;++i) if(position[i]<0 || uint64_t(position[i])>=(i<dims.size()?dims[i]:1)) return false;
        return true;
    }
    uint64_t index(const int5& position) const {
        uint64_t result=0,stride=1;
        for(size_t i=0;i<dims.size();++i) { result+=position[i]*stride; stride*=dims[i]; }
        if(result>=data->size()) throw std::out_of_range("emulated tensor backing storage");
        return result;
    }
};
inline float* gen_addr(const int5& position,tensor t) {
    if(!t.contains(position)) throw std::out_of_range("emulated scalar tensor access");
    return t.data->data()+t.index(position);
}
inline float s_f32_ld_g(float* p) { return *p; }
inline float64 v_f32_ld_tnsr_b(int5 position,tensor t) {
    float64 result;
    for(unsigned lane=0;lane<64;++lane) { auto p=position; p[0]+=lane; if(t.contains(p)) result.lanes[lane]=(*t.data)[t.index(p)]; }
    return result;
}
inline void v_f32_st_tnsr(int5 position,tensor t,const float64& value) {
    for(unsigned lane=0;lane<64;++lane) { auto p=position; p[0]+=lane; if(t.contains(p)) (*t.data)[t.index(p)]=value.lanes[lane]; }
}
}
