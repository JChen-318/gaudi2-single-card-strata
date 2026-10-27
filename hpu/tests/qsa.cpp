#include "strata/hpu/qsa_block.hpp"
#include "strata/hpu/qsa_window.hpp"
#include "strata/hpu/session_flow.hpp"
#include "strata/kernels/f16_bits.hpp"
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <numeric>

using namespace strata::hpu;
namespace {
void compare(std::span<const float> actual,std::span<const float> expected,double tolerance) {
    if(actual.size()!=expected.size()) throw std::runtime_error("QSA hardware output shape mismatch");
    double peak=0,err=0,signal=0;
    for(size_t i=0;i<actual.size();++i) {
        if(!std::isfinite(actual[i])) throw std::runtime_error("non-finite QSA hardware result");
        const double delta=double(actual[i])-expected[i];peak=std::max(peak,std::abs(delta));err+=delta*delta;signal+=double(expected[i])*expected[i];
    }
    const auto relative=std::sqrt(err/std::max(signal,1e-24));
    if(peak>tolerance && relative>0.01) throw std::runtime_error("QSA hardware parity failed");
    std::cout<<" max_abs_error="<<peak<<" relative_l2="<<relative<<'\n';
}
void attention(const Device& device) {
    QsaConfig c;QsaState cache(c);std::vector<float> k(c.kv_heads*c.width),v(k.size()),raw(c.index_width,1),gamma(raw.size(),1);
    for(uint32_t cell=0;cell<2050;++cell) {
        for(size_t d=0;d<k.size();++d) {k[d]=float(int((cell*13+d*7)%31)-15)/32;v[d]=float(int((cell*17+d*5)%41)-20)/32;}
        cache.append(k,v,raw,gamma);
    }
    std::vector<float> query(c.query_heads*c.width),gate(query.size());
    for(size_t i=0;i<query.size();++i) {query[i]=float(int(i%23)-11)/16;gate[i]=float(int(i%17)-8)/4;}
    for(uint32_t b:{1,8,128,2051}) {
        const auto count=b==2051?2050:b==128?47:b==8?5:1;
        std::vector<int32_t> ids(count);for(int i=0;i<count;++i) ids[i]=i;
        auto kv=cache.gather(ids,b);
        HostRecipe executable(device,build_qsa_attention(device,c,b),{}, {"out"});
        const std::array<float,1> scale{1/std::sqrt(float(c.width))};
        executable.set("query",query);executable.set("q_gate",gate);executable.set("key",kv.key);executable.set("value",kv.value);
        executable.set("mask",kv.mask);executable.set("scale",scale);executable.run();
        auto expected=qsa_attention_reference(query,kv,c.query_heads,c.kv_heads,c.width);
        for(size_t i=0;i<expected.size();++i) expected[i]*=float(1/(1+std::exp(-double(gate[i]))));
        std::cout<<"real HPU QSA attention bucket="<<b<<" selected="<<count;
        compare(executable.output("out"),expected,3e-4);
    }
    QsaWindow shared(device,c,64);
    for(uint32_t rows:{1u,3u}) for(uint32_t bucket:{8u,2051u}) {
        QsaWindowBatch batch;batch.rows=rows;batch.bucket=bucket;std::vector<float> expected;
        for(uint32_t row=0;row<rows;++row) {
            const auto count=row==0?1u:row==1?5u:bucket==8?7u:2050u;
            std::vector<int32_t> ids(count);std::iota(ids.begin(),ids.end(),0);const auto kv=cache.gather(ids,bucket);
            auto q=query,g=gate;for(auto& value:q) value+=float(row)/32;for(auto& value:g) value-=float(row)/8;
            batch.query.insert(batch.query.end(),q.begin(),q.end());batch.gate.insert(batch.gate.end(),g.begin(),g.end());
            batch.key.insert(batch.key.end(),kv.key.begin(),kv.key.end());batch.value.insert(batch.value.end(),kv.value.begin(),kv.value.end());
            for(uint32_t head=0;head<c.kv_heads;++head) batch.mask.insert(batch.mask.end(),kv.mask.begin(),kv.mask.end());
            auto out=qsa_attention_reference(q,kv,c.query_heads,c.kv_heads,c.width);
            for(size_t i=0;i<out.size();++i) out[i]*=float(1/(1+std::exp(-double(g[i]))));
            expected.insert(expected.end(),out.begin(),out.end());
        }
        std::cout<<"real HPU batched QSA attention rows="<<rows<<" bucket="<<bucket;compare(shared.attention(batch),expected,3e-4);
    }
}
struct Weight {
    std::vector<float> values;
    HostBuffer host;DeviceBuffer device;
    Weight(const Device& d,uint64_t n,uint64_t h,int seed):values(n*h),host(d,n*h*2),device(d,n*h*2) {
        for(uint64_t i=0;i<n*h;++i) {values[i]=float(int((i*13+seed)%23)-11)/128;static_cast<uint16_t*>(host.data())[i]=to_bf16(values[i]);}
    }
};
std::vector<float> project(std::span<const float> x,const Weight& w) {
    std::vector<float> out(w.values.size()/x.size());
    for(size_t row=0;row<out.size();++row) for(size_t col=0;col<x.size();++col)
        out[row]+=from_bf16(to_bf16(x[col]))*w.values[row*x.size()+col];
    return out;
}
void block(const Device& device) {
    QsaConfig c;c.query_heads=6;c.kv_heads=2;c.width=64;c.rotary=32;c.index_heads=2;c.index_width=64;c.capacity=32;c.top_k=4;
    const uint64_t h=64;std::map<std::string,Weight> weights;Stream transfer(device);Event ready(device);
    auto add=[&](const char* name,uint64_t rows,int seed) {
        auto [it,ok]=weights.emplace(std::piecewise_construct,std::forward_as_tuple(name),std::forward_as_tuple(device,rows,h,seed));
        if(!ok) throw std::logic_error("duplicate fixture weight");
        upload(transfer,it->second.device,it->second.host,it->second.host.size());
    };
    add("w_q_gate",c.query_heads*2*c.width,3);add("w_k",c.kv_heads*c.width,7);add("w_v",c.kv_heads*c.width,11);
    add("w_index_query",c.index_heads*c.index_width,13);add("w_index_key",c.index_width,17);
    weights.emplace(std::piecewise_construct,std::forward_as_tuple("w_out"),std::forward_as_tuple(device,h,c.query_heads*c.width,19));
    auto& wo=weights.at("w_out");upload(transfer,wo.device,wo.host,wo.host.size());ready.record(transfer);ready.synchronize();
    QsaWeights w;
    for(auto& [name,weight]:weights) if(name!="w_out") w.projections.push_back({name,&weight.device});
    w.output={"w_out",&wo.device};w.query_gamma.assign(c.width,1);w.key_gamma.assign(c.width,1);
    w.index_query_gamma.assign(c.index_width,1);w.index_key_gamma.assign(c.index_width,1);
    QsaState reference(c);std::vector<float> expected_attention,expected_value,expected_key;bool fail_tap=false;uint32_t taps=0;
    const QsaHalfSink tap=[&](const QsaHalfView& view) {
        if(fail_tap) throw std::runtime_error("synthetic half observer failure");
        ++taps;const auto cells=reference.size();
        if(view.attention.size()!=uint64_t(c.query_heads)*c.width || view.step!=std::array<int32_t,4>{int32_t(cells-1),int32_t(cells),int32_t(cells/c.block),int32_t(c.selection_width(cells))}) throw std::runtime_error("native QSA half shape/step mapping");
        compare(view.raw_value,expected_value,.001);compare(view.post_value,expected_value,.001);compare(view.key,expected_key,.002);
        if(cells<=7) compare(view.attention,expected_attention,.004); // identity selection before top-k
        if(view.selected_value.size()!=uint64_t(c.kv_heads)*c.width || view.pool_value.size()!=view.selected_value.size()) throw std::runtime_error("native QSA half FP16 bit widths");
        for(uint32_t at=0;at<4;++at) if(at<std::min(4u,c.selection_width(cells))) {
            if(view.ids[at]<0 || uint32_t(view.ids[at])>=cells || (at && view.ids[at]<=view.ids[at-1])) throw std::runtime_error("native QSA half selected IDs");
        } else if(view.ids[at]!=-1) throw std::runtime_error("native QSA half unused ID sentinel");
    };
    QsaBlock native(device,c,h,w,nullptr,tap);
    std::vector<float> gamma(c.width,1),idxgamma(c.index_width,1),x(h);
    auto step=[&](int token) {
        for(size_t i=0;i<x.size();++i) x[i]=float(int((i*11+token*7)%29)-14)/32;
        auto qg=project(x,weights.at("w_q_gate")),k=project(x,weights.at("w_k")),v=project(x,weights.at("w_v"));
        auto iq=project(x,weights.at("w_index_query")),ik=project(x,weights.at("w_index_key"));
        std::vector<float> q(c.query_heads*c.width),gate(q.size());
        for(uint32_t head=0;head<c.query_heads;++head) for(uint32_t d=0;d<c.width;++d) {
            q[head*c.width+d]=qg[head*2*c.width+d];gate[head*c.width+d]=qg[(head*2+1)*c.width+d];
        }
        rms_norm_rows(q,c.width,gamma,c.epsilon);rms_norm_rows(k,c.width,gamma,c.epsilon);rms_norm_rows(iq,c.index_width,idxgamma,c.epsilon);
        const auto rope=rope_row(c.rotary,reference.size(),c.rope);rotate_neox(q,c.width,rope);rotate_neox(k,c.width,rope);rotate_neox(iq,c.index_width,rope);
        reference.append(k,v,ik,idxgamma);const auto ids=reference.select(iq);auto kv=reference.gather(ids,uint32_t(ids.size()));
        auto attention=qsa_attention_reference(q,kv,c.query_heads,c.kv_heads,c.width);
        expected_attention=attention;expected_value=v;expected_key=k;
        for(size_t i=0;i<attention.size();++i) attention[i]*=float(1/(1+std::exp(-double(gate[i]))));
        auto expected=project(attention,wo),actual=native.decode(x);
        std::cout<<"real HPU complete QSA reference block token="<<token<<" cells="<<native.size();compare(actual,expected,0.004);
    };
    for(int i=0;i<5;++i) step(i);
    const auto before_failure=native.size();fail_tap=true;bool failed=false;
    try {(void)native.decode(x);} catch(const std::runtime_error&) {failed=true;}
    fail_tap=false;if(!failed || native.size()!=before_failure || taps!=5) throw std::runtime_error("QSA half observer failure did not roll back appended state");
    auto native_point=native.checkpoint();auto ref_point=reference.checkpoint();
    for(int i=5;i<10;++i) step(i);
    native.restore(native_point);reference.restore(ref_point);
    for(int i=20;i<25;++i) step(i);
    const auto native_snapshot=native.snapshot(),ref_snapshot=reference.snapshot();
    for(int branch=0;branch<2;++branch) {
        native.reset();reference.reset();if(native.size()) throw std::runtime_error("QSA reset failure");
        native.restore(native_snapshot);reference.restore(ref_snapshot);
        for(int i=30;i<33;++i) step(i);
    }
    QsaWindow shared(device,c,h);QsaBlock scalar(device,c,h,w),window(device,c,h,w,&shared);
    auto mismatch=c;mismatch.query_heads=3;mismatch.width=128;mismatch.rotary=32;mismatch.kv_heads=1;
    if(shared.compatible(mismatch,h) || !shared.compatible(c,h)) throw std::runtime_error("QSA shared geometry equality guard");
    auto rows=[&](QsaBlock& one,QsaBlock& many,uint32_t count) {
        std::vector<float> input(uint64_t(count)*h),expected;
        for(size_t i=0;i<input.size();++i) input[i]=float(int((i*17+one.size()*7)%31)-15)/32;
        for(uint32_t row=0;row<count;++row) {const auto out=one.decode(std::span<const float>(input).subspan(row*h,h));expected.insert(expected.end(),out.begin(),out.end());}
        compare(many.prefill(input),expected,.004);if(one.size()!=many.size()) throw std::runtime_error("QSA batch position mismatch");
    };
    for(uint32_t count:{1u,5u,8u,17u}) {
        scalar.reset();window.reset();rows(scalar,window,3);rows(scalar,window,count);rows(scalar,window,1);
    }
    scalar.reset();window.reset();rows(scalar,window,3);const auto before=window.size();int polls=0;bool cancelled=false;
    try {(void)window.prefill(std::vector<float>(17*h,.125f),[&] {return ++polls==12;});} catch(const Cancelled&) {cancelled=true;}
    if(!cancelled || window.size()!=before) throw std::runtime_error("QSA cancelled native chunk accepted state");
    rows(scalar,window,2);
    {
        Weight alt_out(device,h,c.query_heads*c.width,31),alt_query(device,c.query_heads*2*c.width,h,37);
        upload(transfer,alt_out.device,alt_out.host,alt_out.host.size());upload(transfer,alt_query.device,alt_query.host,alt_query.host.size());ready.record(transfer);ready.synchronize();
        auto other=w;other.output.buffer=&alt_out.device;
        for(auto& binding:other.projections) if(binding.name=="w_q_gate") binding.buffer=&alt_query.device;
        QsaBlock other_one(device,c,h,other),other_many(device,c,h,other,&shared);rows(other_one,other_many,2);rows(scalar,window,2);
    }
    {
        window.reset();constexpr uint32_t count=17;
        const uint64_t q=uint64_t(c.query_heads)*c.width,k=uint64_t(c.kv_heads)*c.width;
        std::vector<float> query(count*q*2,.25f),key(count*k,.125f),value(count*k,.5f);
        std::vector<float> iq(count*uint64_t(c.index_heads)*c.index_width,.0625f),ik(count*uint64_t(c.index_width),.03125f);
        const auto attention=window.stage_rows(query,key,value,iq,ik,count);
        std::vector<float> whole(uint64_t(count)*attention.bucket);window.staged_gather_ids(whole);
        for(uint32_t first=0;first<count;first+=8) {
            const uint32_t n=std::min(8u,count-first);std::vector<float> part(uint64_t(n)*attention.bucket);
            window.staged_gather_ids(part,first,n);
            if(!std::equal(part.begin(),part.end(),whole.begin()+uint64_t(first)*attention.bucket))
                throw std::runtime_error("QSA chunk selection differs from whole-window selection");
            for(uint32_t row=0;row<n;++row) for(uint32_t j=0;j<attention.bucket;++j)
                if(part[uint64_t(row)*attention.bucket+j]>=float(first+row+1))
                    throw std::runtime_error("QSA chunk selection read a future row");
        }
        bool refused=false;try {window.staged_gather_ids(whole,1,count);} catch(const std::invalid_argument&) {refused=true;}
        if(!refused) throw std::runtime_error("QSA chunk selection accepted an overflowing slice");
        window.rewind_staged();window.commit_staged(9);
        if(window.size()!=9) throw std::runtime_error("QSA chunk selection changed partial-commit state");
        window.reset();
    }
    std::cout<<"PASS real HPU bounded QSA projection/attention/output, scalar comparison, causal compression, continuation, cancel, shared weight rebind and staged selection slices\n";
}
}
int main() {
    try {Device device;attention(device);block(device);}
    catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
