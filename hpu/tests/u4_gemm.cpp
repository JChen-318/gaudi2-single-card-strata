// Hardware probe for U4 (syn_type_uint4) MME weights. The SDK headers document the API but not the layout or
// the arithmetic, so this test answers, in order:
//  1. whether synGraphCompile accepts a launch-bound [N,K] U4 tensor carrying per-channel SYN_QUANT_METADATA as
//     the B operand of a gemm node, and which activation dtype it accepts (BF16 is tried first, then F32);
//  2. how two codes share a byte (low nibble first or high nibble first);
//  3. the zero-point convention (x = (c-zp)*s, x = c*s+zp, or x = (c+zp)*s);
//  4. that one row of the production expert shapes matches the dequantized reference;
//  5. that uniform (scale=1, zp=0) metadata returns the raw codes, which is what the resident expert graph's
//     per-expert epilogue needs: the MME metadata is fixed at compile time, so one batch GEMM over all experts
//     cannot carry 512 different per-channel tables;
//  6. whether batch_gemm accepts a rank-3 [batch,N,K] U4 weight, numerically on a small shape and at the
//     production grouped-MoE geometry (512 experts, 8 padded rows) for compile acceptance and timing;
//  7. the U4-vs-BF16 device interval at rows 1/2/8/64 for gu [1280,2560] and down [2560,640], from which the
//     per-slot MME cost follows (704 slots x 48 layers).
// Without --device it only compiles the graphs (ctest hpu_u4_gemm_compile); it then needs a Synapse session but
// no card. The numeric half must run on the card: `strata-hpu-u4-gemm-test --device`.
#include "strata/hpu/host_recipe.hpp"
#include "strata/hpu/fp8_batch.hpp"
#include <perf_lib_layer_params.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>
using namespace strata::hpu;
namespace {
// The production expert matrices: gate/up is [2*ff, h] = [1280, 2560], down is [h, ff] = [2560, 640]
// (4.92 MMAC and 3.072 MB of IQ4 per expert slot).
constexpr uint64_t gu_n = 1280, gu_k = 2560;
constexpr uint64_t down_n = 2560, down_k = 640;

std::vector<uint8_t> codes_for(uint64_t n, uint64_t k, uint32_t seed) {
    std::vector<uint8_t> codes(n * k);
    uint32_t state = seed ? seed : 1;
    for (auto& code : codes) {
        state = state * 1664525u + 1013904223u;
        code = static_cast<uint8_t>((state >> 24) & 15);
    }
    return codes;
}
// One byte holds codes 2i and 2i+1 of the row-major [n,k] weight.
std::vector<uint8_t> pack_codes(const std::vector<uint8_t>& codes, bool low_first) {
    if (codes.size() & 1) throw std::invalid_argument("packed codes need an even count");
    std::vector<uint8_t> bytes(codes.size() / 2);
    for (size_t i = 0; i < bytes.size(); ++i) {
        const uint8_t even = codes[2 * i], odd = codes[2 * i + 1];
        bytes[i] = low_first ? static_cast<uint8_t>(even | (odd << 4)) : static_cast<uint8_t>((even << 4) | odd);
    }
    return bytes;
}
std::vector<float> scales_for(uint64_t n) {
    std::vector<float> scales(n);
    for (uint64_t channel = 0; channel < n; ++channel) scales[channel] = 0.25f * float(1 + channel % 4);
    return scales;
}
std::vector<float> zps_for(uint64_t n) {
    std::vector<float> zps(n);
    for (uint64_t channel = 0; channel < n; ++channel) zps[channel] = float(6 + channel % 5);
    return zps;
}
double dequant(uint8_t code, double scale, double zp, int formula) {
    switch (formula) {
        case 0: return (double(code) - zp) * scale;
        case 1: return double(code) * scale + zp;
        default: return (double(code) + zp) * scale;
    }
}
const char* formula_name(int formula) {
    return formula == 0 ? "(c-zp)*s" : formula == 1 ? "c*s+zp" : "(c+zp)*s";
}
void fill_activation(std::vector<float>& a) {
    for (size_t i = 0; i < a.size(); ++i) a[i] = float(int((i * 11) % 29) - 14) / 32;
}
// The device multiplies the BF16-rounded activation; the reference must see the same values.
std::vector<double> rounds(std::span<const float> a, DType type) {
    std::vector<double> out(a.size());
    for (size_t i = 0; i < a.size(); ++i)
        out[i] = type == DType::BF16 ? double(from_bf16(to_bf16(a[i]))) : double(a[i]);
    return out;
}

std::unique_ptr<Recipe> build_u4(const Device& device, uint64_t m, uint64_t k, uint64_t n, DType activation,
                                 const std::vector<float>& scales, const std::vector<float>& zps,
                                 const std::string& role) {
    GraphBuilder graph(device);
    const auto a = graph.tensor({"a", activation, {m, k}});
    const auto w = graph.tensor({"weight", DType::U4, {n, k}});
    const auto y = graph.tensor({"y", DType::F32, {m, n}});
    graph.quantize_per_channel(w, scales, zps);
    graph.gemm(a, w, y, true);
    return graph.compile("u4_" + role + "_" + std::to_string(m) + "x" + std::to_string(k) + "x" + std::to_string(n));
}
std::unique_ptr<Recipe> build_bf16(const Device& device, uint64_t m, uint64_t k, uint64_t n, const std::string& role) {
    GraphBuilder graph(device);
    const auto a = graph.tensor({"a", DType::BF16, {m, k}});
    const auto w = graph.tensor({"weight", DType::BF16, {n, k}});
    const auto y = graph.tensor({"y", DType::F32, {m, n}});
    graph.gemm(a, w, y, true);
    return graph.compile("bf16_" + role + "_" + std::to_string(m) + "x" + std::to_string(k) + "x" + std::to_string(n));
}
// The MME's quantization table is attached to the tensor when the graph is built, so one batch GEMM over all
// resident experts cannot carry 512 different per-channel tables. The resident graph binds one uniform pair
// (scale=1, zp=0) and applies each expert's own scale/offset after the GEMM; this build asks for raw codes.
std::unique_ptr<Recipe> build_u4_uniform(const Device& device, uint64_t m, uint64_t k, uint64_t n, DType activation,
                                         const std::string& role) {
    GraphBuilder graph(device);
    const auto a = graph.tensor({"a", activation, {m, k}});
    const auto w = graph.tensor({"weight", DType::U4, {n, k}});
    const auto y = graph.tensor({"y", DType::F32, {m, n}});
    graph.quantize_uniform(w, 1.0f, 0.0f);
    graph.gemm(a, w, y, true);
    return graph.compile("u4_raw_" + role + "_" + std::to_string(m) + "x" + std::to_string(k) + "x" + std::to_string(n));
}
// The grouped-MoE shape: one batch GEMM over `batch` experts, M padded rows each.
std::unique_ptr<Recipe> build_u4_batch(const Device& device, uint64_t batch, uint64_t m, uint64_t k, uint64_t n,
                                       DType activation, bool /*external_weight*/, const std::string& role,
                                       DType out_dtype = DType::F32) {
    GraphBuilder graph(device);
    const auto a = graph.tensor({"a", activation, {batch, m, k}});
    const auto w = graph.tensor({"weight", DType::U4, {batch, n, k}, true});
    const auto y = graph.tensor({"y", out_dtype, {batch, m, n}});
    graph.quantize_uniform(w, 1.0f, 0.0f);
    graph.batch_gemm(a, w, y, true);
    return graph.compile("u4_batch_" + role + "_" + std::to_string(batch) + "x" + std::to_string(m) + "x" +
                         std::to_string(k) + "x" + std::to_string(n));
}
// Same graph with one (zp, scale) pair per (batch member, output channel): the question is whether
// synTensorSetQuantizationData accepts a table longer than N for a rank-3 weight, which would let every expert
// carry its own affine pair inside the MME instead of in an epilogue.
std::unique_ptr<Recipe> build_u4_batch_table(const Device& device, uint64_t batch, uint64_t m, uint64_t k, uint64_t n,
                                             DType activation, const std::string& role) {
    GraphBuilder graph(device);
    const auto a = graph.tensor({"a", activation, {batch, m, k}});
    const auto w = graph.tensor({"weight", DType::U4, {batch, n, k}});
    const auto y = graph.tensor({"y", DType::F32, {batch, m, n}});
    std::vector<float> scales(batch * n), zps(batch * n);
    for (uint64_t b = 0; b < batch; ++b)
        for (uint64_t c = 0; c < n; ++c) {
            scales[b * n + c] = 0.5f + 0.25f * float((b + c) % 3);
            zps[b * n + c] = float(2 * int(b) + int(c % 4));
        }
    graph.quantize_table(w, zps, scales);
    graph.batch_gemm(a, w, y, true);
    return graph.compile("u4_batch_table_" + role + "_" + std::to_string(batch) + "x" + std::to_string(n));
}

struct TimedEvent {
    synEventHandle handle = nullptr;
    explicit TimedEvent(const Device& device) { check(synEventCreate(&handle, device.info().id, EVENT_COLLECT_TIME), "timing event create"); }
    ~TimedEvent() { if (handle) (void)synEventDestroy(handle); }
};

double queued_ms(const Device& device,HostRecipe& recipe);

// Gaudi2 rejects mixed BF16/U4 GEMM operands. Probe the bridge's explicit
// dequantize_4_bit -> BF16 batch GEMM path, including decompression in device timing.
void expanded_nodes(GraphBuilder& g,uint64_t batch,uint64_t m,uint64_t k,uint64_t n) {
    const auto a=g.tensor({"a",DType::BF16,{batch,m,k}});
    const auto w=g.tensor({"weight",DType::U4,{batch,n,k},true});
    const auto scale=g.tensor({"raw_scale",DType::BF16,{batch,n,1}});
    const auto expanded=g.tensor({"expanded",DType::BF16,{batch,n,k},false});
    const auto y=g.tensor({"y",DType::F32,{batch,m,n}});
    ns_CastKernel::ParamsV3 p{};
    const TensorId in[]={w,scale},out[]={expanded};
    g.op("dequantize_4_bit_bf16",in,out,&p,sizeof(p));
    g.batch_gemm(a,expanded,y,true);
}
std::unique_ptr<Recipe> build_expanded(const Device& device,uint64_t batch,uint64_t m,uint64_t k,uint64_t n,
                                      bool /*external_weight*/) {
    GraphBuilder g(device);expanded_nodes(g,batch,m,k,n);
    return g.compile("u4_expanded_"+std::to_string(batch)+"x"+std::to_string(m)+"x"+std::to_string(k));
}

void verify_expanded(const Device& device) {
    constexpr uint64_t batch=3,m=2,k=32,n=8;
    std::vector<uint8_t> codes(batch*n*k);
    for(uint64_t b=0;b<batch;++b) {
        const auto one=codes_for(n,k,100+uint32_t(b));
        std::copy(one.begin(),one.end(),codes.begin()+b*n*k);
    }
    for(bool low_first:{true,false}) {
        auto r=build_expanded(device,batch,m,k,n,false);
        HostRecipe host(device,std::move(r),std::span<const Binding>{},{"y"});
        host.set_bytes("weight",pack_codes(codes,low_first));
        host.set("raw_scale",std::vector<float>(batch*n,1));
        std::vector<float> a(batch*m*k);fill_activation(a);host.set("a",a);host.run();
        const auto y=host.values("y");double worst=0;
        for(uint64_t b=0;b<batch;++b) for(uint64_t row=0;row<m;++row) for(uint64_t col=0;col<n;++col) {
            double ref=0;
            for(uint64_t j=0;j<k;++j) ref+=double(from_bf16(to_bf16(a[(b*m+row)*k+j])))*codes[(b*n+col)*k+j];
            const double value=y[(b*m+row)*n+col];
            if(!std::isfinite(value)) throw std::runtime_error("expanded U4 produced nonfinite output");
            worst=std::max(worst,std::abs(value-ref));
        }
        std::cout<<"EXPLICIT U4 nibble="<<(low_first?"low-first":"high-first")<<" max_abs="<<worst<<std::endl;
        if(low_first && worst>1e-4) throw std::runtime_error("explicit U4 low-first raw code mismatch");
        if(!low_first && worst<1) throw std::runtime_error("explicit U4 nibble negative control did not differ");
    }
}

void time_expanded(const Device& device,uint64_t rows) {
    for(bool gu:{true,false}) {
        const uint64_t batch=512*(gu?5:2),k=gu?512:320,n=gu?1280:2560;
        auto r=build_expanded(device,batch,rows,k,n,true);
        const auto workspace=r->workspace_bytes();
        DeviceBuffer weight(device,batch*n*k/2);
        {
            HostBuffer chunk(device,16ull<<20);std::memset(chunk.data(),0x5a,chunk.size());Stream copy(device);
            for(uint64_t at=0;at<weight.size();at+=chunk.size()) upload(copy,weight,chunk,std::min(chunk.size(),weight.size()-at),at);
            copy.synchronize();
        }
        const std::vector<Binding> bindings{{"weight",&weight,0}};
        HostRecipe host(device,std::move(r),bindings,{},uint64_t(256)<<20);
        host.set("raw_scale",std::vector<float>(batch*n,1));
        std::vector<float> a(batch*rows*k);fill_activation(a);host.set("a",a);
        const double ms=queued_ms(device,host);
        std::cout<<"EXPLICIT TIMING "<<(gu?"gu":"down")<<" batch="<<batch<<" M="<<rows<<" K="<<k<<" N="<<n
                 <<" ms="<<ms<<" workspace_mib="<<double(workspace)/(1<<20)
                 <<" tmac_s="<<double(batch)*rows*k*n/ms/1e9<<std::endl;
    }
}
// Device-interval time of one launch with no host traffic: the recipe uploads once outside the timed loop and
// every launch after that touches only HBM.
double queued_ms(const Device& device, HostRecipe& recipe) {
    recipe.run();
    Stream stream(device);
    TimedEvent start(device), end(device);
    constexpr int iterations = 16;
    std::vector<double> samples;
    for (int pass = 0; pass < 5; ++pass) {
        check(synEventRecord(start.handle, stream.handle()), "timing start record");
        for (int i = 0; i < iterations; ++i) recipe.launch(stream);
        check(synEventRecord(end.handle, stream.handle()), "timing end record");
        check(synEventSynchronize(end.handle), "timing end wait");
        recipe.wait();
        uint64_t ns = 0;
        check(synEventElapsedTime(&ns, start.handle, end.handle), "timing event elapsed");
        if (!ns) throw std::runtime_error("zero device interval");
        samples.push_back(double(ns) / 1e6 / iterations);
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

std::unique_ptr<Recipe> build_expanded_fp8(const Device& d,uint64_t batch,uint64_t m,uint64_t k,uint64_t n,
                                            uint32_t terms,bool read_weights=false) {
    GraphBuilder g(d);
    const auto a=g.tensor({"a",DType::BF16,{batch,m,k}}),w=g.tensor({"weight",DType::U4,{batch,n,k}});
    const auto y=g.tensor({"y",DType::F32,{batch,m,n}});
    define_u4_fp8_batch(g,a,w,y,batch,m,k,n,terms,"fp8");
    if(read_weights) g.cast(g.find("fp8_weight"),g.tensor({"weight_check",DType::F32,{batch,n,k}}));
    return g.compile("u4_fp8_terms"+std::to_string(terms)+"_"+std::to_string(batch)+"x"+std::to_string(m)+"x"+std::to_string(k));
}
void verify_fp8_batch(const Device& d) {
    constexpr uint64_t batch=3,m=3,n=64;
    for(const uint64_t k:{320u,512u}) for(const uint32_t terms:{1u,2u}) {
        auto codes=codes_for(batch*n,k,914);std::vector<float> a(batch*m*k);
        for(uint64_t b=0;b<batch;++b) for(uint64_t r=0;r<m;++r) for(uint64_t j=0;j<k;++j) {
            const float value=float(int((j*13+b*7+r*11)%97)-48)/173;
            a[(b*m+r)*k+j]=from_bf16(to_bf16(value*(b==0?.001f:b==1?1.f:100.f)));
        }
        std::fill(a.begin(),a.begin()+k,0);
        HostRecipe host(d,build_expanded_fp8(d,batch,m,k,n,terms,true),std::span<const Binding>{},{"y","weight_check"});
        host.set("a",a);host.set_bytes("weight",pack_codes(codes,true));host.run();
        const auto actual=host.values("y"),unpacked=host.values("weight_check");
        for(size_t i=0;i<codes.size();++i) if(unpacked[i]!=float(codes[i])) throw std::runtime_error("FP8 raw U4 weight is not lossless");
        double worst=0,max_row_rms=0;
        for(uint64_t b=0;b<batch;++b) for(uint64_t r=0;r<m;++r) {
            double error=0,signal=0;
            for(uint64_t col=0;col<n;++col) {
                double ref=0;for(uint64_t j=0;j<k;++j) ref+=double(a[(b*m+r)*k+j])*codes[(b*n+col)*k+j];
                const auto value=actual[(b*m+r)*n+col];if(!std::isfinite(value)) throw std::runtime_error("FP8 nonfinite output");
                const double delta=value-ref;worst=std::max(worst,std::abs(delta));error+=delta*delta;signal+=ref*ref;
            }
            const double rms=std::sqrt(error/std::max(signal,1e-30));max_row_rms=std::max(max_row_rms,rms);
        }
        std::cout<<"FP8 QUALITY terms="<<terms<<" K="<<k<<" raw_weights=exact max_abs="<<worst<<" max_row_rel_rms="<<max_row_rms<<std::endl;
        if(max_row_rms>(terms==2?.002:.1)) throw std::runtime_error("FP8 batch activation error too large");
    }
}
void time_expanded_fp8(const Device& d,uint64_t rows,uint32_t terms) {
    for(bool gu:{true,false}) {
        const uint64_t batch=512*(gu?5:2),k=gu?512:320,n=gu?1280:2560;
        auto recipe=build_expanded_fp8(d,batch,rows,k,n,terms);const auto workspace=recipe->workspace_bytes();
        DeviceBuffer weight(d,batch*n*k/2);
        {HostBuffer chunk(d,16ull<<20);std::memset(chunk.data(),0x5a,chunk.size());Stream copy(d);
         for(uint64_t at=0;at<weight.size();at+=chunk.size()) upload(copy,weight,chunk,std::min(chunk.size(),weight.size()-at),at);
         copy.synchronize();}
        const std::vector<Binding> bindings{{"weight",&weight,0}};
        HostRecipe host(d,std::move(recipe),bindings,{},512ull<<20);std::vector<float> a(batch*rows*k);fill_activation(a);host.set("a",a);
        const double ms=queued_ms(d,host),useful=double(batch)*rows*k*n/ms/1e9;
        std::cout<<"FP8 TIMING "<<(gu?"gu":"down")<<" terms="<<terms<<" M="<<rows<<" ms="<<ms<<" workspace_mib="<<double(workspace)/(1<<20)
                 <<" useful_tmac_s="<<useful<<" executed_tmac_s="<<useful*terms<<std::endl;
    }
}

void time_core(const Device& d,uint64_t rows,bool fp8) {
    for(bool gu:{true,false}) {
        const uint64_t batch=512*(gu?5:2),k=gu?512:320,n=gu?1280:2560;
        const auto dtype=fp8?DType::F8:DType::BF16;
        GraphBuilder g(d);
        const auto a=g.tensor({"a",dtype,{batch,rows,k}}),w=g.tensor({"weight",DType::U4,{batch,n,k}});
        const auto expanded=g.tensor({"expanded",dtype,{batch,n,k},false});
        const auto y=g.tensor({"y",DType::F32,{batch,rows,n}});
        ns_CastKernel::ParamsV3 cast{};const TensorId in[]={w},o[]={expanded};
        g.op(fp8?"cast_u4_to_hf8":"cast_u4_to_bf16",in,o,&cast,sizeof(cast));g.batch_gemm(a,expanded,y,true);
        auto recipe=g.compile(std::string("core_")+(fp8?"fp8_":"bf16_")+(gu?"gu":"down"));const auto workspace=recipe->workspace_bytes();
        DeviceBuffer weight(d,batch*n*k/2),activation(d,batch*rows*k*(fp8?1:2));
        HostBuffer chunk(d,16ull<<20);Stream copy(d);std::memset(chunk.data(),0x5a,chunk.size());
        for(uint64_t at=0;at<weight.size();at+=chunk.size()) upload(copy,weight,chunk,std::min(chunk.size(),weight.size()-at),at);
        copy.synchronize();
        if(fp8) std::memset(chunk.data(),0x38,chunk.size());
        else std::fill_n(static_cast<uint16_t*>(chunk.data()),chunk.size()/2,to_bf16(1));
        for(uint64_t at=0;at<activation.size();at+=chunk.size()) upload(copy,activation,chunk,std::min(chunk.size(),activation.size()-at),at);
        copy.synchronize();
        const std::vector<Binding> bindings{{"weight",&weight,0},{"a",&activation,0}};
        HostRecipe host(d,std::move(recipe),bindings,{});const double ms=queued_ms(d,host);
        std::cout<<"CORE TIMING "<<(fp8?"fp8":"bf16")<<' '<<(gu?"gu":"down")<<" M="<<rows<<" ms="<<ms
                 <<" workspace_mib="<<double(workspace)/(1<<20)<<" tmac_s="<<double(batch)*rows*k*n/ms/1e9
                 <<" (unpack+MME only; no activation encode or hi/lo finish)"<<std::endl;
    }
}

struct Match { int packing = 0, formula = 0; double error = std::numeric_limits<double>::infinity(); };

// Two packings x three zero-point conventions against a 4x8x6 GEMM; the wrong ones are off by whole units.
Match measure_convention(const Device& device, DType activation) {
    constexpr uint64_t m = 4, k = 8, n = 6;
    const auto codes = codes_for(n, k, 12345);
    const auto scales = scales_for(n), zps = zps_for(n);
    std::vector<float> a(m * k);
    fill_activation(a);
    const auto av = rounds(a, activation);
    double errors[2][3];
    for (auto& row : errors) for (auto& value : row) value = 1e300;
    for (int packing = 0; packing < 2; ++packing) {
        auto recipe = build_u4(device, m, k, n, activation, scales, zps, "small");
        HostRecipe host(device, std::move(recipe), std::span<const Binding>{}, {"y"});
        host.set_bytes("weight", pack_codes(codes, packing == 0));
        host.set("a", a);
        host.run();
        const auto y = host.values("y");
        for (int formula = 0; formula < 3; ++formula) {
            double worst = 0;
            bool bad = false;
            for (uint64_t row = 0; row < m; ++row)
                for (uint64_t channel = 0; channel < n; ++channel) {
                    double reference = 0;
                    for (uint64_t inner = 0; inner < k; ++inner)
                        reference += av[row * k + inner] * dequant(codes[channel * k + inner], scales[channel], zps[channel], formula);
                    const double error = std::abs(y[row * n + channel] - reference);
                    if (!std::isfinite(error)) bad = true;
                    else worst = std::max(worst, error);
                }
            errors[packing][formula] = bad ? std::numeric_limits<double>::infinity() : worst;
        }
    }
    std::cout << "U4 PROBE m=" << m << " k=" << k << " n=" << n
              << " activation=" << (activation == DType::BF16 ? "bf16" : "f32") << " max_abs_error table:\n";
    Match best;
    for (int packing = 0; packing < 2; ++packing)
        for (int formula = 0; formula < 3; ++formula) {
            std::cout << "  packing=" << (packing == 0 ? "low-first" : "high-first")
                      << " formula=" << formula_name(formula) << " error=" << std::scientific
                      << std::setprecision(3) << errors[packing][formula] << std::defaultfloat << '\n';
            if (errors[packing][formula] < best.error) best = Match{packing, formula, errors[packing][formula]};
        }
    if (!(best.error < 1e-2))
        throw std::runtime_error("no packing/zero-point convention matches (best error " + std::to_string(best.error) + ")");
    std::cout << "U4 PROBE convention nibble=" << (best.packing == 0 ? "low-first" : "high-first")
              << " formula=" << formula_name(best.formula) << " max_abs_error=" << best.error << '\n';
    return best;
}

void verify_shape(const Device& device, const char* label, uint64_t n, uint64_t k,
                  const std::vector<uint8_t>& codes, const std::vector<float>& scales,
                  const std::vector<float>& zps, const Match& match, DType activation) {
    constexpr uint64_t m = 1;
    auto recipe = build_u4(device, m, k, n, activation, scales, zps, label);
    HostRecipe host(device, std::move(recipe), std::span<const Binding>{}, {"y"});
    host.set_bytes("weight", pack_codes(codes, match.packing == 0));
    std::vector<float> a(k);
    fill_activation(a);
    host.set("a", a);
    host.run();
    const auto y = host.values("y");
    const auto av = rounds(a, activation);
    double worst = 0, peak = 0;
    uint64_t worst_channel = 0;
    for (uint64_t channel = 0; channel < n; ++channel) {
        double reference = 0;
        for (uint64_t inner = 0; inner < k; ++inner)
            reference += av[inner] * dequant(codes[channel * k + inner], scales[channel], zps[channel], match.formula);
        const double error = std::abs(y[channel] - reference);
        if (std::isnan(error)) throw std::runtime_error(std::string("U4 ") + label + " produced NaN");
        peak = std::max(peak, std::abs(reference));
        if (error > worst) { worst = error; worst_channel = channel; }
    }
    // The MME may hold the decompressed weight at BF16 precision; the random walk of that rounding over k terms
    // is well inside 0.5% of the row peak, while a wrong scale/zero point is off by whole units.
    const double tolerance = 1e-2 + 5e-3 * peak;
    if (!(worst <= tolerance))
        throw std::runtime_error("U4 " + std::string(label) + " row mismatches the dequantized reference: channel=" +
                                 std::to_string(worst_channel) + " error=" + std::to_string(worst) +
                                 " tolerance=" + std::to_string(tolerance));
    std::cout << "PASS U4 verify shape=" << label << " n=" << n << " k=" << k
              << " max_abs_error=" << worst << " tolerance=" << tolerance << " peak=" << peak << '\n';
}

// Uniform scale=1/zp=0 must reproduce the integer code sum; if it does not, the MME applies an internal factor
// or offset that the resident epilogue must fold in.
void verify_raw(const Device& device, DType activation, const Match& match) {
    constexpr uint64_t m = 4, k = 8, n = 6;
    const auto codes = codes_for(n, k, 777);
    auto recipe = build_u4_uniform(device, m, k, n, activation, "small");
    HostRecipe host(device, std::move(recipe), std::span<const Binding>{}, {"y"});
    host.set_bytes("weight", pack_codes(codes, match.packing == 0));
    std::vector<float> a(m * k);
    fill_activation(a);
    host.set("a", a);
    host.run();
    const auto y = host.values("y");
    const auto av = rounds(a, activation);
    double worst = 0, ratio_sum = 0, peak = 0;
    uint64_t ratio_count = 0;
    for (uint64_t row = 0; row < m; ++row)
        for (uint64_t channel = 0; channel < n; ++channel) {
            double reference = 0;
            for (uint64_t inner = 0; inner < k; ++inner) reference += av[row * k + inner] * codes[channel * k + inner];
            worst = std::max(worst, std::abs(y[row * n + channel] - reference));
            peak = std::max(peak, std::abs(reference));
            if (std::abs(reference) > 1e-6) {
                ratio_sum += y[row * n + channel] / reference;
                ++ratio_count;
            }
        }
    const double ratio = ratio_count ? ratio_sum / double(ratio_count) : 0;
    std::cout << "U4 PROBE raw-metadata scale=1 zp=0 max_abs_error=" << worst << " peak=" << peak
              << " mean_ratio=" << ratio << '\n';
    const double tolerance = 1e-2 + 5e-3 * peak;
    if (!(worst <= tolerance))
        throw std::runtime_error("uniform scale=1,zp=0 metadata did not return the raw codes (mean ratio " +
                                 std::to_string(ratio) + "); the resident epilogue must fold that factor in");
}
void verify_batch(const Device& device, DType activation, const Match& match) {
    constexpr uint64_t batch = 3, m = 2, k = 32, n = 5;
    std::vector<uint8_t> codes(batch * n * k);
    for (uint64_t b = 0; b < batch; ++b) {
        const auto one = codes_for(n, k, 100 + uint32_t(b));
        std::copy(one.begin(), one.end(), codes.begin() + int64_t(b * n * k));
    }
    auto recipe = build_u4_batch(device, batch, m, k, n, activation, false, "small");
    HostRecipe host(device, std::move(recipe), std::span<const Binding>{}, {"y"});
    host.set_bytes("weight", pack_codes(codes, match.packing == 0));
    std::vector<float> a(batch * m * k);
    fill_activation(a);
    host.set("a", a);
    host.run();
    const auto y = host.values("y");
    const auto av = rounds(a, activation);
    double worst = 0, peak = 0;
    for (uint64_t b = 0; b < batch; ++b)
        for (uint64_t row = 0; row < m; ++row)
            for (uint64_t channel = 0; channel < n; ++channel) {
                double reference = 0;
                for (uint64_t inner = 0; inner < k; ++inner)
                    reference += av[(b * m + row) * k + inner] * codes[(b * n + channel) * k + inner];
                worst = std::max(worst, std::abs(y[(b * m + row) * n + channel] - reference));
                peak = std::max(peak, std::abs(reference));
            }
    const double tolerance = 1e-2 + 5e-3 * peak;
    if (!(worst <= tolerance))
        throw std::runtime_error("U4 batch_gemm mismatches the raw-code reference: error=" + std::to_string(worst) +
                                 " tolerance=" + std::to_string(tolerance));
    std::cout << "PASS U4 batch_gemm verify batch=" << batch << " max_abs_error=" << worst
              << " tolerance=" << tolerance << '\n';
}
constexpr uint64_t kBatchExperts = 512, kBatchRows = 8;
// A table of batch*N pairs: which entries does the MME actually read? Each expert gets a different scale and
// offset, so a per-batch table and a broadcast first table give completely different numbers.
void verify_batch_table(const Device& device, DType activation, const Match& match) {
    constexpr uint64_t batch = 3, m = 2, k = 32, n = 5;
    std::vector<uint8_t> codes(batch * n * k);
    for (uint64_t b = 0; b < batch; ++b) {
        const auto one = codes_for(n, k, 100 + uint32_t(b));
        std::copy(one.begin(), one.end(), codes.begin() + int64_t(b * n * k));
    }
    const auto table = [&](uint64_t b, uint64_t c) {
        return std::pair<double, double>{double(2 * int(b) + int(c % 4)), 0.5 + 0.25 * double((b + c) % 3)};
    };
    auto recipe = build_u4_batch_table(device, batch, m, k, n, activation, "small");
    HostRecipe host(device, std::move(recipe), std::span<const Binding>{}, {"y"});
    host.set_bytes("weight", pack_codes(codes, match.packing == 0));
    std::vector<float> a(batch * m * k);
    fill_activation(a);
    host.set("a", a);
    host.run();
    const auto y = host.values("y");
    const auto av = rounds(a, activation);
    double per_batch = 0, broadcast = 0, peak = 0;
    for (uint64_t b = 0; b < batch; ++b)
        for (uint64_t row = 0; row < m; ++row)
            for (uint64_t c = 0; c < n; ++c) {
                double grouped = 0, first = 0;
                const auto [zp_g, s_g] = table(b, c);
                const auto [zp_f, s_f] = table(0, c);
                for (uint64_t inner = 0; inner < k; ++inner) {
                    const double code = codes[(b * n + c) * k + inner];
                    grouped += av[(b * m + row) * k + inner] * (code - zp_g) * s_g;
                    first += av[(b * m + row) * k + inner] * (code - zp_f) * s_f;
                }
                const double value = y[(b * m + row) * n + c];
                per_batch = std::max(per_batch, std::abs(value - grouped));
                broadcast = std::max(broadcast, std::abs(value - first));
                peak = std::max({peak, std::abs(grouped), std::abs(first)});
            }
    const double tolerance = 1e-2 + 5e-3 * peak;
    std::cout << "U4 PROBE batch-table pairs=" << batch * n << " error_per_expert=" << per_batch
              << " error_first_table_only=" << broadcast << " tolerance=" << tolerance << '\n';
    if (per_batch <= tolerance) std::cout << "PASS U4 batch metadata carries one table per expert\n";
    else if (broadcast <= tolerance)
        std::cout << "NOTE U4 batch metadata reads only the first N pairs (per-channel, shared by every expert)\n";
    else throw std::runtime_error("U4 batch metadata matches neither a per-expert nor a broadcast table");
}
// The resident graph binds one external weight tensor; fill it in chunks so the probe never needs a contiguous
// pinned copy of 800+ MB.
double time_batch(const Device& device, const char* label, uint64_t batch, uint64_t m, uint64_t k, uint64_t n,
                  DType activation, DType out_dtype = DType::F32) {
    auto recipe = build_u4_batch(device, batch, m, k, n, activation, true, label, out_dtype);
    const uint64_t weight_bytes = batch * n * k / 2;
    DeviceBuffer weight(device, weight_bytes);
    {
        constexpr uint64_t chunk_bytes = uint64_t(16) << 20;
        HostBuffer chunk(device, chunk_bytes);
        std::memset(chunk.data(), 0x5a, chunk.size());
        Stream copy(device);
        for (uint64_t offset = 0; offset < weight_bytes; offset += chunk_bytes)
            upload(copy, weight, chunk, std::min<uint64_t>(chunk_bytes, weight_bytes - offset), offset);
        copy.synchronize();
    }
    const std::vector<Binding> external{{"weight", &weight, 0}};
    HostRecipe host(device, std::move(recipe), external, {}, uint64_t(256) << 20);
    std::vector<float> a(batch * m * k);
    fill_activation(a);
    host.set("a", a);
    const double ms = queued_ms(device, host);
    const double macs = double(batch) * double(m) * double(n) * double(k);
    std::cout << std::fixed << std::setprecision(3)
              << "BATCH TIMING shape=" << label << " batch=" << batch << " rows=" << m << " u4_ms=" << ms
              << " u4_TMACps=" << macs / ms / 1e9 << std::defaultfloat << '\n';
    return ms;
}

// Production grouped-MoE shapes. G K-slices keep one batch GEMM per projection while giving each slice its own
// per-channel affine pair: batch = experts*G and K = K/G, so the weight bytes are those of the single-slice
// shape. Every expert is padded to `rows` whether the window holds 64 or 256 tokens, so one measurement covers
// every window size.
void time_grouped(const Device& device, DType activation, uint64_t rows = 64) {
    struct Shape { const char* label; uint64_t experts, slices, rows, k, n; };
    const Shape shapes[] = {
        {"gu_g5", kBatchExperts, 5, rows, 512, gu_n},
        {"down_g2", kBatchExperts, 2, rows, 320, down_n},
    };
    double gu = 0, down = 0;
    for (const auto& shape : shapes) {
        const double ms = time_batch(device, shape.label, shape.experts * shape.slices, shape.rows, shape.k,
                                     shape.n, activation, DType::F32);
        if (std::string(shape.label) == "gu_g5") gu = ms;
        if (std::string(shape.label) == "down_g2") down = ms;
    }
    const double layer = gu + down;
    std::cout << std::fixed << std::setprecision(3)
              << "ESTIMATE gemm_only g=5/2 bucket_rows=" << rows << " layer_moe_ms=" << layer << " window64_moe_ms=" << layer * 48
              << " window256_moe_ms=" << layer * 48
              << " window256_moe_only_tok_s=" << 256.0 / (layer * 48 / 1000.0) << std::defaultfloat << '\n';
}
constexpr uint64_t kTimedRows[] = {1, 2, 8, 64}; // 1 = no padding, 2/8 = batch_gemm padding, 64 = reference
std::vector<double> time_shape(const Device& device, const char* label, uint64_t n, uint64_t k,
                const std::vector<uint8_t>& codes, const std::vector<float>& scales,
                const std::vector<float>& zps, const Match& match, DType activation) {
    const auto weight = pack_codes(codes, match.packing == 0);
    std::vector<float> wf(n * k);
    for (size_t i = 0; i < wf.size(); ++i) wf[i] = float(int(i % 37) - 18) / 32;
    std::vector<double> u4_times;
    for (uint64_t m : kTimedRows) {
        std::vector<float> a(m * k);
        fill_activation(a);
        double u4_ms = 0, bf16_ms = 0;
        {
            HostRecipe host(device, build_u4(device, m, k, n, activation, scales, zps, std::string(label) + "_t"),
                            std::span<const Binding>{}, {});
            host.set_bytes("weight", weight);
            host.set("a", a);
            u4_ms = queued_ms(device, host);
        }
        {
            HostRecipe host(device, build_bf16(device, m, k, n, std::string(label) + "_t"),
                            std::span<const Binding>{}, {});
            host.set("weight", wf);
            host.set("a", a);
            bf16_ms = queued_ms(device, host);
        }
        const double macs = double(m) * double(n) * double(k);
        const double u4_gbps = double(n * k / 2) / u4_ms / 1e6;
        const double bf16_gbps = double(n * k * 2) / bf16_ms / 1e6;
        const double u4_tmacs = macs / u4_ms / 1e9;
        const double bf16_tmacs = macs / bf16_ms / 1e9;
        std::cout << std::fixed << std::setprecision(3)
                  << "TIMING shape=" << label << " rows=" << m
                  << " u4_ms=" << u4_ms << " bf16_ms=" << bf16_ms << " speedup=" << bf16_ms / u4_ms << "x"
                  << " u4_GBps=" << std::setprecision(1) << u4_gbps << " bf16_GBps=" << bf16_gbps
                  << " u4_TMACps=" << u4_tmacs << " bf16_TMACps=" << bf16_tmacs
                  << std::defaultfloat << '\n';
        u4_times.push_back(u4_ms);
    }
    return u4_times;
}

void define_offline(const char* label, uint64_t m, uint64_t k, uint64_t n, DType activation) {
    GraphBuilder graph(OfflineCompilation{});
    const auto a = graph.tensor({"a", activation, {m, k}});
    const auto w = graph.tensor({"weight", DType::U4, {n, k}});
    const auto y = graph.tensor({"y", DType::F32, {m, n}});
    graph.quantize_per_channel(w, scales_for(n), zps_for(n));
    graph.gemm(a, w, y, true);
    std::cout << "PASS U4 offline compile shape=" << label << " rows=" << m << " activation="
              << (activation == DType::BF16 ? "bf16" : "f32") << " workspace="
              << graph.compile_offline(std::string("u4_offline_") + label) << '\n';
}
void define_offline_batch(const char* label, uint64_t batch, uint64_t m, uint64_t k, uint64_t n, DType activation,
                          DType out_dtype = DType::F32) {
    GraphBuilder graph(OfflineCompilation{});
    const auto a = graph.tensor({"a", activation, {batch, m, k}});
    const auto w = graph.tensor({"weight", DType::U4, {batch, n, k}, true});
    const auto y = graph.tensor({"y", out_dtype, {batch, m, n}});
    graph.quantize_uniform(w, 1.0f, 0.0f);
    graph.batch_gemm(a, w, y, true);
    std::cout << "PASS U4 batch offline compile shape=" << label << " batch=" << batch << " rows=" << m
              << " workspace=" << graph.compile_offline(std::string("u4_offline_batch_") + label) << '\n';
}
void define_offline_batch_table(uint64_t batch, uint64_t m, uint64_t k, uint64_t n) {
    GraphBuilder graph(OfflineCompilation{});
    const auto a = graph.tensor({"a", DType::BF16, {batch, m, k}});
    const auto w = graph.tensor({"weight", DType::U4, {batch, n, k}, true});
    const auto y = graph.tensor({"y", DType::F32, {batch, m, n}});
    std::vector<float> scales(batch * n), zps(batch * n);
    for (uint64_t i = 0; i < batch * n; ++i) { scales[i] = 1.0f; zps[i] = 0.0f; }
    graph.quantize_table(w, zps, scales);
    graph.batch_gemm(a, w, y, true);
    std::cout << "PASS U4 batch-table offline compile batch=" << batch << " pairs=" << batch * n
              << " workspace=" << graph.compile_offline("u4_offline_batch_table") << '\n';
}
}
int main(int argc, char** argv) {
    try {
        const std::string mode=argc>1?argv[1]:"";
        const bool core_mode=mode=="--device-core-fp8" || mode=="--device-core-bf16";
        const bool fp8_mode=mode=="--device-fp8" || mode=="--device-fp8-single" || mode=="--device-fp8-hilo";
        const bool explicit_mode=mode=="--device-expanded" || mode=="--device-expanded-timing";
        const bool device_mode=mode=="--device" || mode=="--device-convention" || mode=="--device-grouped" || explicit_mode || fp8_mode || core_mode;
        if(argc>1 && !device_mode) throw std::invalid_argument("unknown device probe mode");
        const uint64_t grouped_rows=argc==3?std::stoul(argv[2]):64;
        if(!grouped_rows || grouped_rows>(core_mode?128:64) || argc>3 || (argc==3 && mode!="--device-grouped" && mode!="--device-expanded-timing" && mode!="--device-fp8-single" && mode!="--device-fp8-hilo" && !core_mode))
            throw std::invalid_argument("grouped rows must be in 1..64");
        if (!device_mode) {
            configure_kernel_libraries();
            check(synInitialize(), "init");
            bool direct=false;
            try {define_offline("small",4,8,6,DType::BF16);direct=true;}
            catch(const std::runtime_error& e) {
                if(std::string(e.what()).find("synNodeCreate gemm")!=0) throw;
                std::cout<<"CAPABILITY direct mixed BF16/U4 GEMM unavailable: "<<e.what()<<'\n';
            }
            if(direct) {
                define_offline("small_f32",4,8,6,DType::F32);
                define_offline_batch("gu",kBatchExperts,kBatchRows,gu_k,gu_n,DType::BF16);
                define_offline_batch("down",kBatchExperts,kBatchRows,down_k,down_n,DType::BF16);
                define_offline_batch_table(3,2,32,5);
            }
            for(bool gu:{true,false}) {
                GraphBuilder g(OfflineCompilation{});
                expanded_nodes(g,512*(gu?5:2),64,gu?512:320,gu?1280:2560);
                std::cout<<"PASS explicit U4 dequant + MME compile "<<(gu?"gu":"down")
                         <<" workspace="<<g.compile_offline(gu?"u4_expand_gu":"u4_expand_down")<<'\n';
            }
            check(synDestroy(), "destroy");
            return 0;
        }
        Device device;
        if(core_mode) {time_core(device,grouped_rows,mode=="--device-core-fp8");return 0;}
        if(mode=="--device-fp8") {verify_fp8_batch(device);return 0;}
        if(mode=="--device-fp8-single" || mode=="--device-fp8-hilo") {time_expanded_fp8(device,grouped_rows,mode=="--device-fp8-hilo"?2:1);return 0;}
        if(mode=="--device-expanded") {verify_expanded(device);return 0;}
        if(mode=="--device-expanded-timing") {time_expanded(device,grouped_rows);return 0;}
        DType activation = DType::BF16;
        if(mode=="--device-grouped") {time_grouped(device,activation,grouped_rows);return 0;}
        Match match;
        try {
            match = measure_convention(device, DType::BF16);
        } catch (const std::exception& e) {
            std::cout << "NOTE U4 GEMM with bf16 activation failed: " << e.what() << '\n';
            activation = DType::F32;
            match = measure_convention(device, DType::F32);
        }
        std::cout << "U4 PROBE activation_in_use=" << (activation == DType::BF16 ? "bf16" : "f32") << '\n';
        verify_raw(device, activation, match);
        verify_batch(device, activation, match);
        if(mode=="--device-convention") return 0;
        verify_batch_table(device, activation, match);
        std::vector<double> gu_ms, down_ms;
        {
            const auto codes = codes_for(gu_n, gu_k, 7);
            verify_shape(device, "gu", gu_n, gu_k, codes, scales_for(gu_n), zps_for(gu_n), match, activation);
            gu_ms = time_shape(device, "gu", gu_n, gu_k, codes, scales_for(gu_n), zps_for(gu_n), match, activation);
        }
        {
            const auto codes = codes_for(down_n, down_k, 9);
            verify_shape(device, "down", down_n, down_k, codes, scales_for(down_n), zps_for(down_n), match, activation);
            down_ms = time_shape(device, "down", down_n, down_k, codes, scales_for(down_n), zps_for(down_n), match, activation);
        }
        // One row per slot is what a 64-row window needs (704 slots); padded rows batch over the <=512 routed
        // experts instead. The current TPC IQ4 GEMV path is 87 ms/window at 64 rows.
        for (size_t i = 0; i < std::size(kTimedRows); ++i) {
            const double per_slot = gu_ms[i] + down_ms[i];
            const double experts = kTimedRows[i] == 1 ? 704.0 : 512.0;
            std::cout << std::fixed << std::setprecision(3)
                      << "ESTIMATE rows_per_expert=" << kTimedRows[i] << " per_slot_ms=" << per_slot
                      << " window_moe_ms=" << per_slot * experts * 48 << " (experts_per_window=" << experts
                      << ", current TPC GEMV 87 ms at 64 rows)" << std::defaultfloat << '\n';
        }
        // The grouped path the resident graph would build: one batch GEMM per projection over all experts,
        // M rows padded to the batch bucket. 48 layers at a 64-row window.
        const double batch_gu = time_batch(device, "gu", kBatchExperts, kBatchRows, gu_k, gu_n, activation);
        const double batch_down = time_batch(device, "down", kBatchExperts, kBatchRows, down_k, down_n, activation);
        std::cout << std::fixed << std::setprecision(3)
                  << "ESTIMATE grouped batch=" << kBatchExperts << " rows=" << kBatchRows
                  << " layer_moe_ms=" << batch_gu + batch_down
                  << " window_moe_ms=" << (batch_gu + batch_down) * 48
                  << " (one gu + one down batch GEMM per layer, 48 layers; current TPC GEMV 87 ms)"
                  << std::defaultfloat << '\n';
        time_grouped(device, activation);
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
