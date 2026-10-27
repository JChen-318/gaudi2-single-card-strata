#include "strata/hpu/dense_kv.hpp"
#include "strata/hpu/state_io.hpp"
#include "strata/kernels/f16_bits.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
using namespace strata::hpu;
namespace {
void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
template<class F> void rejects(F&& f) {bool caught=false;try {f();} catch(const std::exception&) {caught=true;}require(caught,"invalid dense KV operation accepted");}
std::filesystem::path temporary(const char* stem) {
    return std::filesystem::temp_directory_path()/("strata-dense-kv-"+std::string(stem)+"-"+
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".bin");
}
// Independent INT8 oracle: fp16(amax/127) as the stored scale, rint(x/stored scale) ties to even, clamp
// [-127,127], decode as fp16(code*stored scale). Written from kv_q8.hpp, not by calling kv_int8_*.
struct Quantized {std::vector<int8_t> codes;std::vector<uint16_t> scales;};
Quantized quantize(std::span<const float> values) {
    Quantized out;out.codes.resize(values.size());out.scales.resize(values.size()/64);
    for(size_t group=0;group<out.scales.size();++group) {
        float amax=0;for(size_t t=0;t<64;++t) amax=std::max(amax,std::fabs(values[group*64+t]));
        out.scales[group]=strata::kernels::f16_from_f32(amax/127.0f);
        const double scale=double(strata::kernels::f32_from_f16(out.scales[group]));
        for(size_t t=0;t<64;++t) {
            int code=0;
            if(scale>0) code=std::clamp(int(std::rint(double(values[group*64+t])/scale)),-127,127);
            out.codes[group*64+t]=int8_t(code);
        }
    }
    return out;
}
float decoded(const Quantized& q,const size_t index) {
    return strata::kernels::f32_from_f16(strata::kernels::f16_from_f32(float(q.codes[index])*strata::kernels::f32_from_f16(q.scales[index/64])));
}
void int8_rows(int n,uint32_t heads,uint32_t width,std::vector<float>& key,std::vector<float>& value) {
    key.resize(uint64_t(heads)*width);value.resize(key.size());
    for(uint32_t head=0;head<heads;++head) for(uint32_t d=0;d<width;++d) {
        const auto i=uint64_t(head)*width+d;
        key[i]=float(((n*37+int(d)*13+int(head)*29)%101)-50)/17.0f;
        value[i]=float(((n*17+int(d)*5+int(head)*11)%71)-35)/9.0f;
        if(d%64==7) key[i]=float(n+1)*0.9f;   // one outlier per group: every group has its own scale
    }
}
void check_gather(const SelectedKv& got,uint32_t heads,uint32_t width,uint32_t first_cell,
                  const std::vector<Quantized>& keys,const std::vector<Quantized>& values,const char* what) {
    for(uint32_t head=0;head<heads;++head) for(uint32_t cell=0;cell<got.count;++cell) for(uint32_t d=0;d<width;++d) {
        const auto at=(uint64_t(head)*got.bucket+cell)*width+d, source=uint64_t(head)*width+d;
        require(got.key[at]==decoded(keys[first_cell+cell],source),what);
        require(got.value[at]==decoded(values[first_cell+cell],source),what);
    }
}
void fp16_fixture(const std::filesystem::path& path) {
    DenseKvState state(4,2,2),other(4,2,2);require(state.allocated_bytes()==104,"dense KV host byte bound");
    const std::array<float,4> k1{1.00048828125f,1.00146484375f,-0.0f,65504},v1{2,-3,4,5};
    const std::array<float,4> k2{8,9,10,11},v2{20,21,22,23};
    state.append(k1,v1);state.append(k2,v2);const auto full=state.snapshot(),rewind=state.checkpoint();
    const auto kv=state.gather(4);
    require(kv.count==2 && kv.key==std::vector<float>({1,1.001953125f,8,9,0,0,0,0,-0.0f,65504,10,11,0,0,0,0}),"independent FP16/head-major/padding oracle");
    require(std::signbit(kv.key[8]) && kv.value[0]==2 && kv.value[2]==20 && kv.value[8]==4 && kv.value[10]==22,"dense KV value/head placement or signed zero");
    state.append(v1,k1);const auto discarded=state.checkpoint();state.restore(rewind);require(state.size()==2,"dense KV fast rewind");
    const auto prefix=state.gather_prefix(1,4);
    require(prefix.count==1 && prefix.key[0]==1 && prefix.value[0]==2 && prefix.mask[1]==-INFINITY,"causal prefix included later cells");
    rejects([&]{(void)state.gather_prefix(0,4);});rejects([&]{(void)state.gather_prefix(3,4);});
    state.append(v2,k2);rejects([&]{state.restore(discarded);});
    const auto child=state.checkpoint();state.append(k1,v1);state.restore(child);state.restore(rewind);
    require(state.size()==2,"nested child rollback invalidated live ancestor prefix");state.restore(full);state.reset();state.restore(full);
    const auto window=state.gather(4,1);require(window.count==1 && window.key[0]==8 && window.key[8]==10 && window.mask[1]==-INFINITY,"dense sliding attention window");
    rejects([&]{other.restore(full);});rejects([&]{(void)state.gather(1);});
    for(float bad:{NAN,INFINITY,65536.0f}) {auto row=k1;row[3]=bad;rejects([&]{state.append(k1,row);});require(state.size()==2,"bad FP16 append published a cell");}
    {StateWriter writer(path.string(),4096);state.write_snapshot(writer,full);writer.finish();}
    {StateReader reader(path.string(),4096);const auto imported=other.read_snapshot(reader,2);reader.finish();other.restore(imported);}
    require(other.gather(4).key==kv.key && other.gather(4).value==kv.value,"dense KV disk restoration across instances");
    {StateReader reader(path.string(),4096);rejects([&]{(void)other.read_snapshot(reader,3);});}
    {StateWriter writer(path.string(),4096);writer.integer(1);const std::array<uint16_t,4> nan{0x7e00,0,0,0},zero{};writer.vector<uint16_t>(nan);writer.vector<uint16_t>(zero);writer.finish();}
    {StateReader reader(path.string(),4096);rejects([&]{(void)other.read_snapshot(reader,1);});}
    require(other.size()==2 && other.gather(4).key==kv.key,"invalid serialized state mutated active KV");
}
void int8_fixture(const std::filesystem::path& path) {
    constexpr uint32_t capacity=4,heads=2,width=64,bucket=4;
    DenseKvState state(capacity,heads,width,KvMode::Int8),reference(capacity,heads,width,KvMode::Int8);
    require(state.allocated_bytes()==1096,"dense INT8 host byte bound");
    const auto per_cell=uint64_t(heads)*width;
    std::vector<Quantized> key_rows,value_rows;
    for(int n=0;n<int(capacity);++n) {
        std::vector<float> key,value;int8_rows(n,heads,width,key,value);
        key_rows.push_back(quantize(key));value_rows.push_back(quantize(value));
        state.append(key,value);reference.append(key,value);
    }
    const auto full_kv=state.gather(bucket);
    require(full_kv.count==capacity && full_kv.mask.size()==bucket && std::all_of(full_kv.mask.begin(),full_kv.mask.begin()+capacity,[](float v){return v==0;}),
            "dense INT8 gather shape");
    check_gather(full_kv,heads,width,0,key_rows,value_rows,"dense INT8 decode/layout differs from the storage law");
    check_gather(state.gather(bucket,2),heads,width,2,key_rows,value_rows,"dense INT8 sliding window differs from the storage law");
    check_gather(state.gather_prefix(2,bucket),heads,width,0,key_rows,value_rows,"dense INT8 causal prefix differs from the storage law");
    uint64_t sizes[5]{};const auto full=reference.snapshot();reference.snapshot_sizes(full,sizes);
    require(sizes[0]==uint64_t(capacity)*per_cell && sizes[1]==sizes[0] && sizes[2]==uint64_t(capacity)*heads*2 &&
            sizes[3]==sizes[2] && sizes[4]==0,"dense INT8 parked part sizes");
    require(reference.parked_bytes()==capacity*264 && reference.parked_bytes(4)==4*264,"dense INT8 parked bytes use the cell law");
    rejects([&]{ (void)reference.parked_bytes(capacity+1); });
    // Grow-in-place: the first cell stays in the retained image, the rest is re-read from the live pool.
    DenseKvState live(capacity,heads,width,KvMode::Int8);
    for(int n=0;n<2;++n) {std::vector<float> key,value;int8_rows(n,heads,width,key,value);live.append(key,value);}
    auto prior=live.snapshot();
    for(int n=2;n<int(capacity);++n) {std::vector<float> key,value;int8_rows(n,heads,width,key,value);live.append(key,value);}
    auto reused=live.snapshot_reused(std::move(prior),1);
    live.reset();live.restore(reused);
    require(live.gather(bucket).key==reference.gather(bucket).key && live.gather(bucket).value==reference.gather(bucket).value,
            "dense INT8 reused snapshot differs from the live pool");
    // Disk restoration is format-tagged by its own vector shapes, in both directions.
    {StateWriter writer(path.string(),65536);reference.write_snapshot(writer,full);writer.finish();}
    DenseKvState imported(capacity,heads,width,KvMode::Int8);
    {StateReader reader(path.string(),65536);const auto loaded=imported.read_snapshot(reader,capacity);reader.finish();imported.restore(loaded);}
    require(imported.gather(bucket).key==reference.gather(bucket).key && imported.gather(bucket).value==reference.gather(bucket).value,
            "dense INT8 disk restoration across instances");
    DenseKvState f16(capacity,heads,width);
    for(int n=0;n<int(capacity);++n) {std::vector<float> key,value;int8_rows(n,heads,width,key,value);f16.append(key,value);}
    const auto f16full=f16.snapshot();const auto f16path=temporary("f16");
    struct Cleanup {std::filesystem::path path;~Cleanup(){std::error_code error;std::filesystem::remove(path,error);}} cleanup{f16path};
    {StateWriter writer(f16path.string(),65536);f16.write_snapshot(writer,f16full);writer.finish();}
    {StateReader reader(f16path.string(),65536);rejects([&]{ (void)imported.read_snapshot(reader,capacity); });}
    {StateReader reader(path.string(),65536);rejects([&]{ (void)f16.read_snapshot(reader,capacity); });}
    // A serialized scale that expands to infinity is refused before it can reach attention.
    {StateWriter writer(path.string(),65536);writer.integer(1);
     writer.vector<int8_t>(std::vector<int8_t>(per_cell,0));writer.vector<int8_t>(std::vector<int8_t>(per_cell,0));
     writer.vector<uint16_t>(std::vector<uint16_t>(heads,0x7c00));writer.vector<uint16_t>(std::vector<uint16_t>(heads,0));
     writer.finish();}
    {StateReader reader(path.string(),65536);rejects([&]{ (void)imported.read_snapshot(reader,1); });}
    std::vector<float> bad_key,bad_value;int8_rows(0,heads,width,bad_key,bad_value);bad_key[3]=NAN;
    const auto count=state.size();rejects([&]{state.append(bad_key,bad_value);});require(state.size()==count,"failed INT8 append published a cell");
    rejects([&]{ (void)DenseKvState(capacity,heads,60,KvMode::Int8); });
    // With two groups per head the second group's scale has to be the one that decodes its values.
    DenseKvState pair(2,1,128,KvMode::Int8);require(pair.allocated_bytes()==552,"dense INT8 multi-group host byte bound");
    std::vector<Quantized> pair_keys,pair_values;
    for(int n=0;n<2;++n) {
        std::vector<float> key(128),value(128);
        for(uint32_t d=0;d<128;++d) {
            key[d]=float(int(d%23)-11)/7.0f;value[d]=float(int(d%19)-9)/5.0f;
            if(d==70) {key[d]=float(n+1)*7.5f;value[d]=float(n+1)*6.25f;}
        }
        pair_keys.push_back(quantize(key));pair_values.push_back(quantize(value));pair.append(key,value);
    }
    const auto pair_kv=pair.gather(2);
    check_gather(pair_kv,1,128,0,pair_keys,pair_values,"dense INT8 second group decoded with the wrong scale");
    require(pair_values[1].scales[1]!=pair_values[1].scales[0],"multi-group fixture did not separate the scales");
}
// Independent Q4_0 oracle: the FP32 Hadamard butterfly and the ggml q4_0 group law from upstream's
// src/kernels/cuda/kv_q4.cu: largest |x| with the larger signed value breaking ties, the unrounded 1/d,
// truncation toward zero, clamp 0..15, elements t / t+16 in the low / high nibble, fp16((q-8)*fp16(d)).
struct Q4Row { std::array<uint16_t,256> rotated{}; };
std::vector<float> hadamard256(std::span<const float> row) {
    require(row.size()==256,"Q4 rotation oracle shape");
    std::vector<float> out(row.begin(),row.end());
    for(auto& x:out) x*=1.0f/16.0f;
    for(uint32_t h=1;h<256;h*=2)
        for(uint32_t i=0;i<256;i+=2*h)
            for(uint32_t j=0;j<h;++j) {
                const float x=out[i+j],y=out[i+j+h];
                out[i+j]=x+y;out[i+j+h]=x-y;
            }
    return out;
}
Q4Row q4_oracle(std::span<const float> row) {
    const auto rotated=hadamard256(row);
    Q4Row out;
    for(uint32_t b=0;b<8;++b) {
        const auto* values=rotated.data()+b*32;
        float amax=0,mval=0;bool first=true;
        for(uint32_t t=0;t<32;++t) {
            const float x=values[t];require(std::isfinite(x),"Q4 oracle needs finite values");
            const float a=std::fabs(x);
            if(first || a>amax || (a==amax && x>mval)) {amax=a;mval=x;first=false;}
        }
        const float d=mval/-8.0f,id=d!=0?1.0f/d:0.0f;
        const auto stored=strata::kernels::f32_from_f16(strata::kernels::f16_from_f32(d));
        std::array<uint8_t,32> q{};
        for(uint32_t t=0;t<32;++t) {
            const float scaled=values[t]*id+8.5f;
            int code=0;
            if(std::isfinite(scaled) && scaled>float(std::numeric_limits<int>::min()) &&
               scaled<float(std::numeric_limits<int>::max())) code=int(std::trunc(scaled));
            q[t]=uint8_t(code<0?0:(code>15?15:code));
            out.rotated[b*32+t]=strata::kernels::f16_from_f32(float(q[t]-8)*stored);
        }
    }
    return out;
}
std::vector<float> q4_decoded(std::span<const float> row) {
    const auto coded=q4_oracle(row);
    std::array<float,256> decoded{};
    for(uint32_t i=0;i<256;++i) decoded[i]=strata::kernels::f32_from_f16(coded.rotated[i]);
    return hadamard256(decoded);
}
void q4_rows(int n,uint32_t heads,uint32_t width,std::vector<float>& key,std::vector<float>& value) {
    key.resize(uint64_t(heads)*width);value.resize(key.size());
    for(uint32_t head=0;head<heads;++head) for(uint32_t d=0;d<width;++d) {
        const auto i=uint64_t(head)*width+d;
        key[i]=float(((n*37+int(d)*13+int(head)*29)%101)-50)/17.0f;
        value[i]=float(((n*17+int(d)*5+int(head)*11)%71)-35)/9.0f;
        if(d==7) key[i]=float(n+1)*0.9f;   // a per-cell outlier inside the first group
    }
}
void check_q4_gather(const SelectedKv& got,uint32_t heads,uint32_t width,uint32_t first_cell,
                     const std::vector<std::vector<float>>& keys,const std::vector<std::vector<float>>& values,
                     const char* what) {
    for(uint32_t head=0;head<heads;++head) for(uint32_t cell=0;cell<got.count;++cell) {
        const auto want_k=q4_decoded(std::span<const float>(keys[first_cell+cell]).subspan(uint64_t(head)*width,width));
        const auto want_v=q4_decoded(std::span<const float>(values[first_cell+cell]).subspan(uint64_t(head)*width,width));
        for(uint32_t d=0;d<width;++d) {
            const auto at=(uint64_t(head)*got.bucket+cell)*width+d;
            require(got.key[at]==want_k[d],what);
            require(got.value[at]==want_v[d],what);
        }
    }
}
void q4_fixture(const std::filesystem::path& path) {
    constexpr uint32_t capacity=2,heads=2,width=256,bucket=2;
    DenseKvState state(capacity,heads,width,KvMode::Q4),reference(capacity,heads,width,KvMode::Q4);
    require(state.allocated_bytes()==1176,"dense Q4 host byte bound");
    std::vector<std::vector<float>> keys,values;
    for(int n=0;n<int(capacity);++n) {
        std::vector<float> key,value;q4_rows(n,heads,width,key,value);
        keys.push_back(key);values.push_back(value);state.append(key,value);reference.append(key,value);
    }
    const auto full_kv=state.gather(bucket);
    require(full_kv.count==capacity && full_kv.mask.size()==bucket &&
            std::all_of(full_kv.mask.begin(),full_kv.mask.begin()+capacity,[](float v){return v==0;}),
            "dense Q4 gather shape");
    check_q4_gather(full_kv,heads,width,0,keys,values,"dense Q4 decode/layout differs from the rotated q4_0 law");
    check_q4_gather(state.gather(bucket,1),heads,width,1,keys,values,"dense Q4 sliding window differs from the rotated q4_0 law");
    check_q4_gather(state.gather_prefix(1,bucket),heads,width,0,keys,values,"dense Q4 causal prefix differs from the rotated q4_0 law");
    uint64_t sizes[5]{};const auto full=reference.snapshot();reference.snapshot_sizes(full,sizes);
    require(sizes[0]==uint64_t(capacity)*heads*144 && sizes[1]==sizes[0] && sizes[2]==0 && sizes[3]==0 && sizes[4]==0,
            "dense Q4 parked part sizes");
    require(reference.parked_bytes()==capacity*576 && reference.parked_bytes(2)==2*576,"dense Q4 parked bytes use the cell law");
    rejects([&]{ (void)reference.parked_bytes(capacity+1); });
    // Grow-in-place: the retained cell stays in the image, the appended one is re-read from the live pool.
    DenseKvState live(capacity,heads,width,KvMode::Q4);
    live.append(keys[0],values[0]);auto prior=live.snapshot();
    live.append(keys[1],values[1]);auto reused=live.snapshot_reused(std::move(prior),1);
    live.reset();live.restore(reused);
    require(live.gather(bucket).key==reference.gather(bucket).key &&
            live.gather(bucket).value==reference.gather(bucket).value,"dense Q4 reused snapshot differs from the live pool");
    // Disk restoration is tagged by the Q4 vector shape, and FP16 images do not cross-read.
    {StateWriter writer(path.string(),65536);reference.write_snapshot(writer,full);writer.finish();}
    DenseKvState imported(capacity,heads,width,KvMode::Q4);
    {StateReader reader(path.string(),65536);const auto loaded=imported.read_snapshot(reader,capacity);reader.finish();imported.restore(loaded);}
    require(imported.gather(bucket).key==reference.gather(bucket).key &&
            imported.gather(bucket).value==reference.gather(bucket).value,"dense Q4 disk restoration across instances");
    DenseKvState f16(capacity,heads,width);
    for(int n=0;n<int(capacity);++n) {std::vector<float> key,value;q4_rows(n,heads,width,key,value);f16.append(key,value);}
    const auto f16full=f16.snapshot();const auto f16path=temporary("q4-f16");
    struct Cleanup {std::filesystem::path path;~Cleanup(){std::error_code error;std::filesystem::remove(path,error);}} cleanup{f16path};
    {StateWriter writer(f16path.string(),65536);f16.write_snapshot(writer,f16full);writer.finish();}
    {StateReader reader(f16path.string(),65536);rejects([&]{ (void)imported.read_snapshot(reader,capacity); });}
    {StateReader reader(path.string(),65536);rejects([&]{ (void)f16.read_snapshot(reader,capacity); });}
    // A serialized block scale that expands to infinity is refused before it can reach attention.
    const auto per_cell_bytes=uint64_t(heads)*144;
    {StateWriter writer(path.string(),65536);writer.integer(1);
     std::vector<uint8_t> block(per_cell_bytes,0);block[0]=0x00;block[1]=0x7c;   // d = +inf
     writer.vector<uint8_t>(block);writer.vector<uint8_t>(std::vector<uint8_t>(per_cell_bytes,0));
     writer.finish();}
    {StateReader reader(path.string(),65536);rejects([&]{ (void)imported.read_snapshot(reader,1); });}
    std::vector<float> bad_key,bad_value;q4_rows(0,heads,width,bad_key,bad_value);bad_key[3]=NAN;
    const auto count=state.size();rejects([&]{state.append(bad_key,bad_value);});require(state.size()==count,"failed Q4 append published a cell");
    rejects([&]{ (void)DenseKvState(capacity,heads,60,KvMode::Q4); });
    // The k8v4 drafter runs plain INT8 (src/core/mtp.cpp:344-349), so the dense pool has no hybrid layout and
    // must refuse it rather than quietly storing another format under the hybrid tag.
    rejects([&]{ (void)DenseKvState(capacity,heads,width,KvMode::K8V4); });
}
}
int main() {
    const auto path=temporary("fp16");
    try {
        fp16_fixture(path);
        int8_fixture(path);
        q4_fixture(path);
        std::filesystem::remove(path);
        std::cout<<"PASS MTP dense FP16 KV independent rounding/layout/window oracle, append guards, rewind/branch/reset and cross-instance disk state\n"
                   "PASS MTP dense INT8 KV stored-scale decode/window/prefix oracle, snapshot parts, parked cell bytes, grow-in-place reuse,\n"
                   "     disk restoration, F16/INT8 shape separation, infinite-scale and non-finite append refusal, multi-group scales\n"
                   "PASS MTP dense Q4_0 KV rotated oracle decode/window/prefix, snapshot parts, parked cell bytes, grow-in-place reuse,\n"
                   "     disk restoration, F16/Q4 shape separation, infinite-scale and non-finite append refusal, geometry refusal\n"
                   "PASS MTP dense K8V4 refusal: the drafter runs plain INT8, so no dense hybrid pool can be constructed\n";
    } catch(const std::exception& error) {std::error_code ignored;std::filesystem::remove(path,ignored);std::cerr<<error.what()<<'\n';return 1;}
}
