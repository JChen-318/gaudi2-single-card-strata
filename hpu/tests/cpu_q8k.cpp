// Reuse upstream's adversarial input corpus and byte comparison without its
// unconditional calls into a reference that asserts on NaN/Inf/subnormals.
#define main strata_upstream_q8k_main
#include "../../src/kernels/cpu/q8k_quant_parity.cpp"
#undef main
#include "strata/hpu/cpu_quant_guard.hpp"

namespace {
struct GuardBlock {float scale;int8_t values[256];int16_t sums[16];};
static_assert(sizeof(GuardBlock)==292);
void guarded(const float* x,void* y,int64_t n) {
    if(!strata::hpu::guard_q8k(x,y,n)) quantize_row_q8_K_ref(x,y,n);
}
bool defined_input(const std::vector<float>& x) {
    for(size_t first=0;first<x.size();first+=256) {
        float maximum=0;
        for(size_t i=0;i<256;++i) {if(!std::isfinite(x[first+i])) return false;maximum=std::max(maximum,std::abs(x[first+i]));}
        if(maximum && !std::isfinite(127.0f/maximum)) return false;
    }
    return true;
}
bool exceptional_values(const std::vector<float>& x,const std::vector<uint8_t>& bytes) {
    for(size_t first=0;first<x.size();first+=256) {
        GuardBlock block{};std::memcpy(&block,bytes.data()+first/256*sizeof(block),sizeof(block));
        bool invalid=false;double maximum=0;
        for(size_t i=0;i<256;++i) {invalid|=!std::isfinite(x[first+i]);maximum=std::max(maximum,std::abs(double(x[first+i])));}
        if(invalid) {if(!std::isnan(block.scale)) return false;continue;}
        for(size_t i=0;i<256;++i) {
            const double decoded=double(block.scale)*block.values[i];
            if(!std::isfinite(decoded) || std::abs(decoded-double(x[first+i]))>maximum/48+double(std::numeric_limits<float>::denorm_min())) return false;
        }
        for(size_t i=0;i<16;++i) {int sum=0;for(size_t j=0;j<16;++j) sum+=block.values[16*i+j];if(sum!=block.sums[i]) return false;}
    }
    return true;
}
}
int main() {
    ggml_cpu_init();int failures=0,normal=0,exceptional=0;
    const auto reference=quantize_row_q8_K_ref;const auto gcpu=ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float;
    for(const int width:{256,512,2560,6144}) for(const auto& x:cases(width)) {
        const auto bytes=ggml_row_size(GGML_TYPE_Q8_K,width);
        if(defined_input(x)) {
            failures+=!same("ordinary ggml-cpu",width,normal,reference,gcpu,x,bytes);
            if(cpu::cpu_avx2_ok()) failures+=!same("ordinary original AVX2",width,normal,reference,cpu::q8k_quant_avx2,x,bytes);
            failures+=!same("ordinary guard passthrough",width,normal,reference,guarded,x,bytes);++normal;
        } else {
            std::vector<uint8_t> output(bytes,0);guarded(x.data(),output.data(),width);
            failures+=!exceptional_values(x,output);++exceptional;
        }
    }
    cpu::NativeFmt format;std::string error;
    if(!cpu::native_fmt(21,21,2560,512,format,error)) {std::fprintf(stderr,"%s\n",error.c_str());return 1;}
    g_fmt=&format;
    for(const auto& x:cases(2560)) failures+=!same("actual guarded native_quant_act",2560,0,guarded,via_act,x,ggml_row_size(GGML_TYPE_Q8_K,2560));
    for(const auto& x:cases(512)) failures+=!same("actual guarded native_quant_h",512,0,guarded,via_h,x,ggml_row_size(GGML_TYPE_Q8_K,512));
    std::printf("%s Q8_K normal=%d exact pinned/original bytes; exceptional=%d double-reciprocal tiny values / NaN marker; actual native entrypoints guarded\n",failures?"FAIL":"PASS",normal,exceptional);
    return failures?1:0;
}
