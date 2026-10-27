#include "strata/hpu/dense_weights.hpp"
#include "strata/hpu/fp8.hpp"
#include "strata/hpu/tensor.hpp"
#include "strata/hpu/packed_quant.hpp"
#include "strata/hpu/performance_modes.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <map>
#include <thread>

namespace strata::hpu {
struct DenseWeights::Impl {
    Device device;Stream copy;uint64_t budget,staging,used=0;
    std::map<std::string,std::unique_ptr<DeviceBuffer>> weights;
    std::map<std::string,std::pair<std::string,uint64_t>> fused; // part -> (fused buffer, byte offset)
    std::map<std::string,std::pair<uint64_t,uint64_t>> shapes; // BF16 matrices: (rows, columns)
    Impl(const Device& d,uint64_t b,uint64_t s):device(d),copy(d),budget(b),staging(s) {}
    void reserve(const std::string& name,uint64_t bytes) const {
        if(name.empty() || !bytes || weights.contains(name) || fused.contains(name)) throw std::invalid_argument("empty/duplicate dense weight binding");
        if(bytes>budget-used) throw std::invalid_argument("dense weight payload budget exceeded");
    }
};
DenseWeights::DenseWeights(const Device& d,uint64_t budget,uint64_t staging) {
    if(!budget || staging<4) throw std::invalid_argument("invalid dense weight/staging budget");
    impl_=std::make_unique<Impl>(d,budget,staging);
}
DenseWeights::~DenseWeights()=default;
void DenseWeights::bf16(const std::string& name,const WeightSlice& source,uint64_t columns,uint64_t rows) {
    auto& p=*impl_;const auto bytes=checked_product(checked_product(columns,rows),2);
    p.reserve(name,bytes);
    if(!columns || !rows || source.columns()!=columns || source.rows()!=rows) throw std::invalid_argument("dense BF16 source shape mismatch: "+name);
    Bf16WeightUpload upload(p.device,source,p.copy,p.staging);auto buffer=std::make_unique<DeviceBuffer>(upload.finish());
    p.weights.emplace(name,std::move(buffer));p.used+=bytes;p.shapes[name]={rows,columns};
}
void DenseWeights::f32(const std::string& name,std::span<const float> values) {
    auto& p=*impl_;const auto bytes=values.size_bytes();p.reserve(name,bytes);
    for(float x:values) if(!std::isfinite(x)) throw std::invalid_argument("dense constants must be finite: "+name);
    auto buffer=std::make_unique<DeviceBuffer>(p.device,bytes);
    const uint64_t slot_bytes=std::min(uint64_t(bytes),p.staging/4*4);HostBuffer slot(p.device,slot_bytes);Event complete(p.device);
    for(uint64_t offset=0;offset<bytes;) {
        const auto count=std::min(slot_bytes,bytes-offset);
        std::memcpy(slot.data(),reinterpret_cast<const uint8_t*>(values.data())+offset,count);
        upload(p.copy,*buffer,slot,count,offset);complete.record(p.copy);complete.synchronize();offset+=count;
    }
    p.weights.emplace(name,std::move(buffer));p.used+=bytes;
}
void DenseWeights::bf16_subset(const std::string& name,const WeightSlice& source,std::span<const int32_t> rows) {
    auto& p=*impl_;if(rows.empty()) throw std::invalid_argument("empty BF16 weight subset");
    const auto bytes=checked_product(checked_product(source.columns(),rows.size()),2);p.reserve(name,bytes);
    Bf16WeightUpload upload(p.device,source,rows,p.copy,p.staging);auto buffer=std::make_unique<DeviceBuffer>(upload.finish());
    p.weights.emplace(name,std::move(buffer));p.used+=bytes;
}
void DenseWeights::packed(const std::string& name,const WeightSlice& source,std::span<const int32_t> ids) {
    auto& p=*impl_;const auto bytes=checked_product(packed_row_bytes(source.ggml_type(),source.columns()),ids.empty()?source.rows():ids.size());
    p.reserve(name,bytes);auto output=std::make_unique<DeviceBuffer>(upload_packed_weights(p.device,source,p.copy,p.staging,ids));
    p.weights.emplace(name,std::move(output));p.used+=bytes;
}
void DenseWeights::requantized_q4(const std::string& name,const WeightSlice& source,std::span<const int32_t> ids) {
    auto& p=*impl_;const auto bytes=checked_product(packed_row_bytes(2,source.columns()),ids.empty()?source.rows():ids.size());
    p.reserve(name,bytes);auto output=std::make_unique<DeviceBuffer>(upload_packed_weights(p.device,source,p.copy,p.staging,ids,true));
    p.weights.emplace(name,std::move(output));p.used+=bytes;
}
void DenseWeights::matrix(const std::string& binding,const WeightTable& table,const std::string& name,uint64_t columns,uint64_t rows) {
    if(table.info(name).gguf_shape!=std::vector<uint64_t>{columns,rows}) throw std::invalid_argument("GGUF dense matrix shape mismatch: "+name);
    bf16(binding,table.matrix(name),columns,rows);
}
void DenseWeights::constant(const std::string& binding,const WeightTable& table,const std::string& name,uint64_t elements) {
    impl_->reserve(binding,checked_product(elements,4));const auto values=table.constant(name,elements);f32(binding,values);
}
void DenseWeights::transposed(const std::string& binding,const WeightTable& table,const std::string& name,uint64_t columns,uint64_t rows) {
    const auto elements=checked_product(columns,rows);impl_->reserve(binding,checked_product(elements,4));
    if(elements>1048576 || table.info(name).gguf_shape!=std::vector<uint64_t>{columns,rows}) throw std::invalid_argument("GGUF small transpose shape/budget mismatch: "+name);
    std::vector<float> values(elements);transpose_f32(table.matrix(name),values,impl_->staging);f32(binding,values);
}
std::vector<Binding> DenseWeights::bindings() const {
    std::vector<Binding> result;result.reserve(impl_->weights.size());
    for(const auto& [name,buffer]:impl_->weights) result.push_back({name,buffer.get()});
    return result;
}
Binding DenseWeights::binding(const std::string& name) const {
    if(const auto part=impl_->fused.find(name);part!=impl_->fused.end()) return {name,impl_->weights.at(part->second.first).get(),part->second.second};
    const auto found=impl_->weights.find(name);if(found==impl_->weights.end()) throw std::invalid_argument("dense weight binding not found: "+name);
    return {name,found->second.get()};
}
void DenseWeights::to_fp8(const std::string& name) {
    auto& p=*impl_;
    const auto found=p.weights.find(name);
    if(found==p.weights.end() || p.weights.contains(name+"_f8") || found->second->size()%2) throw std::invalid_argument("dense FP8 conversion: "+name);
    const auto& source=*found->second;const uint64_t bytes=source.size();
    // The BF16 loads and fusions record each matrix's (rows, columns).
    const auto shape=p.shapes.find(name);if(shape==p.shapes.end()) throw std::invalid_argument("dense FP8 conversion needs the matrix shape: "+name);
    const uint64_t rows=shape->second.first,columns=shape->second.second;
    if(rows*columns*2!=bytes) throw std::invalid_argument("dense FP8 conversion shape mismatch: "+name);
    HostBuffer host(p.device,bytes);download(p.copy,host,source,bytes);p.copy.synchronize();
    const auto* bf=static_cast<const uint16_t*>(host.data());
    const bool fixed=fp8_static_enabled();
    if(fixed) for(uint64_t k=0;k<rows*columns;++k)
        if(!std::isfinite(from_bf16(bf[k])) || std::fabs(from_bf16(bf[k]))>15.0f)
            throw std::invalid_argument("static FP8 weight outside exact hardware scale range: "+name);
    HostBuffer codes(p.device,rows*columns),scales(p.device,rows*4);
    auto* c=static_cast<uint8_t*>(codes.data());auto* sc=static_cast<float*>(scales.data());
    // Rows split over the host threads (load time only).
    const uint64_t workers=std::max<uint64_t>(1,std::min<uint64_t>(rows,std::thread::hardware_concurrency()));
    std::vector<std::thread> pool;
    for(uint64_t w=0;w<workers;++w) pool.emplace_back([=] {
        for(uint64_t r=rows*w/workers;r<rows*(w+1)/workers;++r) {
            float peak=0;for(uint64_t k=0;k<columns;++k) peak=std::max(peak,std::fabs(from_bf16(bf[r*columns+k])));
            const float scale=fixed?0.0625f:(peak>0?peak/fp8_143_max:1.0f);sc[r]=scale;
            for(uint64_t k=0;k<columns;++k) c[r*columns+k]=fp8_143_from_f32(from_bf16(bf[r*columns+k])/scale);
        }
    });
    for(auto& t:pool) t.join();
    auto code_buffer=std::make_unique<DeviceBuffer>(p.device,rows*columns),scale_buffer=std::make_unique<DeviceBuffer>(p.device,rows*4);
    upload(p.copy,*code_buffer,codes,rows*columns);upload(p.copy,*scale_buffer,scales,rows*4);p.copy.synchronize();
    p.used+=rows*columns+rows*4;
    p.weights.emplace(name+"_f8",std::move(code_buffer));p.weights.emplace(name+"_f8_scale",std::move(scale_buffer));
}
bool DenseWeights::contains(const std::string& name) const {return impl_->weights.contains(name) || impl_->fused.contains(name);}
void DenseWeights::fuse(const std::string& name,std::initializer_list<std::string> parts) {
    auto& p=*impl_;
    if(name.empty() || p.weights.contains(name) || p.fused.contains(name) || parts.size()<2) throw std::invalid_argument("invalid dense weight fusion: "+name);
    uint64_t total=0;
    for(const auto& part:parts) {
        const auto found=p.weights.find(part);
        if(found==p.weights.end() || found->second->size()%128) throw std::invalid_argument("dense weight fusion part missing/unaligned: "+part);
        total+=found->second->size();
    }
    auto merged=std::make_unique<DeviceBuffer>(p.device,total);
    // Device -> pinned -> device in bounded chunks; this runs once at load.
    const uint64_t chunk=std::min<uint64_t>(total,16ull<<20);HostBuffer bounce(p.device,chunk);uint64_t at=0;
    for(const auto& part:parts) {
        const auto& source=*p.weights.at(part);
        for(uint64_t offset=0;offset<source.size();offset+=chunk) {
            const auto count=std::min(chunk,source.size()-offset);
            download(p.copy,bounce,source,count,0,offset);upload(p.copy,*merged,bounce,count,at+offset,0);p.copy.synchronize();
        }
        p.fused[part]={name,at};at+=source.size();
    }
    // The fused matrix keeps the parts' column count; its rows are theirs added up.
    uint64_t fused_rows=0,fused_columns=0;
    for(const auto& part:parts) if(const auto shape=p.shapes.find(part);shape!=p.shapes.end()) {fused_rows+=shape->second.first;fused_columns=shape->second.second;}
    if(fused_columns) p.shapes[name]={fused_rows,fused_columns};
    for(const auto& part:parts) p.weights.erase(part);
    p.weights.emplace(name,std::move(merged));
}
uint64_t DenseWeights::payload_bytes() const {return impl_->used;}
}
