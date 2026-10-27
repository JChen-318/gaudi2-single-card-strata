#include "strata/hpu/dflash.hpp"
#include <bit>
#include "strata/hpu/tensor.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <set>
#include <perf_lib_layer_params.h>
#include <stdexcept>

namespace strata::hpu {
namespace {
bool draft_bf16_enabled() {
    static const bool on=[] {const char* e=std::getenv("STRATA_HPU_DFLASH_BF16");return e && *e && *e!='0';}();return on;
}
// The few JSON shapes a safetensors header uses: {"name": {"dtype": "BF16", "shape": [..], "data_offsets": [a, b]}}.
struct Entry {std::string dtype;std::vector<uint64_t> shape;uint64_t begin=0,end=0;};
struct HeaderParser {
    const std::string& s;size_t i=0;
    void space() {while(i<s.size() && (s[i]==' ' || s[i]=='\n' || s[i]=='\r' || s[i]=='\t')) ++i;}
    void expect(char c) {space();if(i>=s.size() || s[i]!=c) throw std::runtime_error("safetensors header: malformed JSON");++i;}
    std::string string() {
        expect('"');std::string out;
        while(i<s.size() && s[i]!='"') {if(s[i]=='\\') ++i;if(i<s.size()) out+=s[i++];}
        expect('"');return out;
    }
    uint64_t number() {space();uint64_t v=0;bool any=false;while(i<s.size() && s[i]>='0' && s[i]<='9') {v=v*10+uint64_t(s[i++]-'0');any=true;}if(!any) throw std::runtime_error("safetensors header: number expected");return v;}
    void skip() { // any value
        space();
        if(s[i]=='"') {string();return;}
        if(s[i]=='{') {++i;space();if(s[i]=='}') {++i;return;}while(true) {string();expect(':');skip();space();if(s[i]==',') {++i;continue;}expect('}');return;}}
        if(s[i]=='[') {++i;space();if(s[i]==']') {++i;return;}while(true) {skip();space();if(s[i]==',') {++i;continue;}expect(']');return;}}
        while(i<s.size() && s[i]!=',' && s[i]!='}' && s[i]!=']') ++i;
    }
    std::vector<uint64_t> numbers() {
        std::vector<uint64_t> out;expect('[');space();if(s[i]==']') {++i;return out;}
        while(true) {out.push_back(number());space();if(s[i]==',') {++i;continue;}expect(']');return out;}
    }
    std::map<std::string,Entry> parse() {
        std::map<std::string,Entry> out;expect('{');
        while(true) {
            const auto name=string();expect(':');
            if(name=="__metadata__") skip();
            else {
                Entry e;expect('{');
                while(true) {
                    const auto key=string();expect(':');
                    if(key=="dtype") e.dtype=string();
                    else if(key=="shape") e.shape=numbers();
                    else if(key=="data_offsets") {auto o=numbers();if(o.size()!=2) throw std::runtime_error("safetensors offsets");e.begin=o[0];e.end=o[1];}
                    else skip();
                    space();if(s[i]==',') {++i;continue;}expect('}');break;
                }
                out[name]=e;
            }
            space();if(s[i]==',') {++i;continue;}expect('}');return out;
        }
    }
};
struct Weights {
    std::map<std::string,std::unique_ptr<DeviceBuffer>> buffers;
    std::map<std::string,std::pair<std::string,uint64_t>> fused; // part -> (fused buffer, byte offset)
    const DeviceBuffer& get(const std::string& name) const {
        if(const auto part=fused.find(name);part!=fused.end()) return get(part->second.first);
        const auto it=buffers.find(name);if(it==buffers.end()) throw std::runtime_error("DFlash weight missing: "+name);return *it->second;
    }
    uint64_t offset(const std::string& name) const {const auto part=fused.find(name);return part==fused.end()?0:part->second.second;}
    // Moves BF16 matrices of one column count back to back into one buffer (rows in `parts` order); runs once at load.
    void fuse(const Device& device,const std::string& name,std::initializer_list<std::string> parts) {
        uint64_t total=0;for(const auto& part:parts) total+=get(part).size();
        auto merged=std::make_unique<DeviceBuffer>(device,total);Stream copy(device);
        const uint64_t chunk=std::min<uint64_t>(total,16ull<<20);HostBuffer bounce(device,chunk);uint64_t at=0;
        for(const auto& part:parts) {
            const auto& source=*buffers.at(part);
            for(uint64_t offset=0;offset<source.size();offset+=chunk) {
                const auto count=std::min(chunk,source.size()-offset);
                download(copy,bounce,source,count,0,offset);upload(copy,*merged,bounce,count,at+offset,0);copy.synchronize();
            }
            fused[part]={name,at};at+=source.size();
        }
        for(const auto& part:parts) buffers.erase(part);
        buffers.emplace(name,std::move(merged));
    }
};
// BF16 tensors are uploaded as stored ([out, in] for linear weights); norms and convolution base kernels as F32.
Weights load(const Device& device,const std::string& path) {
    std::ifstream file(path,std::ios::binary);if(!file) throw std::runtime_error("DFlash: cannot open "+path);
    uint64_t header=0;file.read(reinterpret_cast<char*>(&header),8);
    if(!file || header>(64u<<20)) throw std::runtime_error("DFlash: bad safetensors header");
    std::string json(header,'\0');file.read(json.data(),std::streamsize(header));
    const auto entries=HeaderParser{json}.parse();const uint64_t base=8+header;
    Weights out;Stream copy(device);
    uint64_t largest=0;for(const auto& [name,e]:entries) largest=std::max(largest,e.end-e.begin);
    HostBuffer staging(device,largest*2+128);
    for(const auto& [name,e]:entries) {
        if(e.dtype!="BF16") throw std::runtime_error("DFlash: only BF16 tensors are supported: "+name);
        const uint64_t bytes=e.end-e.begin;
        file.seekg(std::streamoff(base+e.begin));file.read(static_cast<char*>(staging.data()),std::streamsize(bytes));
        if(!file) throw std::runtime_error("DFlash: short read of "+name);
        const bool f32=name.find("norm")!=std::string::npos || name.find("base_kernel")!=std::string::npos;
        if(f32) {
            const auto* bf=static_cast<const uint16_t*>(staging.data());auto* f=reinterpret_cast<float*>(static_cast<uint8_t*>(staging.data())+bytes);
            for(uint64_t k=0;k<bytes/2;++k) f[k]=from_bf16(bf[k]);
            auto buffer=std::make_unique<DeviceBuffer>(device,bytes*2);
            upload(copy,*buffer,staging,bytes*2,0,bytes);copy.synchronize();out.buffers[name]=std::move(buffer);
        } else {
            auto buffer=std::make_unique<DeviceBuffer>(device,bytes);
            upload(copy,*buffer,staging,bytes);copy.synchronize();out.buffers[name]=std::move(buffer);
        }
    }
    return out;
}
std::string layer_name(uint32_t layer,const char* part) {return "layers."+std::to_string(layer)+"."+part;}
// Graph helpers in the GraphBuilder's current scope.
struct Ops {
    GraphBuilder& g;int counter=0;
    TensorId t(const std::string& name,DType dtype,std::vector<uint64_t> shape,bool persistent=false) {return g.tensor({name,dtype,std::move(shape),persistent});}
    std::string fresh(const std::string& stem) {return stem+"_"+std::to_string(counter++);}
    TensorId round_bf16(TensorId x,std::vector<uint64_t> shape,const std::string& stem) {
        if(!draft_bf16_enabled()) return x;
        // Persistent BF16 boundary prevents the compiler folding the rounding into a F32 identity.
        const auto b=t(fresh(stem+"_rounded16"),DType::BF16,shape,true),f=t(fresh(stem+"_rounded"),DType::F32,std::move(shape));
        g.cast(x,b);g.cast(b,f);return f;
    }
    // x F32 [rows, in] -> x BF16 . weight^T: weight is a persistent BF16 [out, in].
    TensorId linear(TensorId x,uint64_t rows,uint64_t in,TensorId weight,uint64_t out_width,const std::string& stem) {
        const auto xb=t(fresh(stem+"_x16"),DType::BF16,{rows,in}),y=t(fresh(stem),DType::F32,{rows,out_width});
        g.cast(x,xb);
        if(draft_bf16_enabled() && stem!="logits") {
            const auto y16=t(fresh(stem+"_product16"),DType::BF16,{rows,out_width},true);
            g.gemm(xb,weight,y16,true);g.cast(y16,y);return y;
        }
        g.gemm(xb,weight,y,true);return y;
    }
    TensorId rms(TensorId x,uint64_t rows,uint64_t width,TensorId gamma,float epsilon,const std::string& stem) {
        const auto y=t(fresh(stem),DType::F32,{rows,width});g.row_norm(x,gamma,y,epsilon);return round_bf16(y,{rows,width},stem);
    }
    TensorId binary(const char* guid,TensorId a,TensorId b,std::vector<uint64_t> shape,const std::string& stem) {
        const auto y=t(fresh(stem),DType::F32,std::move(shape));const TensorId in[]={a,b},out[]={y};g.op(guid,in,out);return y;
    }
    TensorId constant(float value,std::vector<uint64_t> shape,const std::string& stem) {
        const auto y=t(fresh(stem),DType::F32,std::move(shape));
        ns_ConstantKernel::Params p{};p.constant.f=value;const TensorId out[]={y};g.op("constant_f32",{},out,&p,sizeof(p));return y;
    }
    TensorId reshape(TensorId x,DType dtype,std::vector<uint64_t> shape,const std::string& stem) {const auto y=t(fresh(stem),dtype,std::move(shape));g.reshape(x,y);return y;}
    TensorId transpose12(TensorId x,std::vector<uint64_t> out_shape,const std::string& stem) { // [a, b, c] -> [b, a, c]
        const auto y=t(fresh(stem),DType::F32,std::move(out_shape));
        synTransposeParams p{};p.tensorDim=3;p.permutation[0]=TPD_Channel;p.permutation[1]=TPD_Height;p.permutation[2]=TPD_Width;
        const TensorId in[]={x},out[]={y};g.op("transpose",in,out,&p,sizeof(p));return y;
    }
    TensorId transpose4(TensorId x,std::vector<uint64_t> out_shape,const std::string& stem) { // [a, b, c, d] -> [b, a, c, d]
        const auto y=t(fresh(stem),DType::F32,std::move(out_shape));
        synTransposeParams p{};p.tensorDim=4;p.permutation[0]=TPD_Channel;p.permutation[1]=TPD_Width;p.permutation[2]=TPD_Depth;p.permutation[3]=TPD_Height;
        const TensorId in[]={x},out[]={y};g.op("transpose",in,out,&p,sizeof(p));return y;
    }
    // NeoX RoPE over the whole head: x [rows, heads, d], cos/sin [rows, 1, d] (each half repeated).
    TensorId rope(TensorId x,uint64_t rows,uint64_t heads,uint64_t d,TensorId cosine,TensorId sine,const std::string& stem) {
        const auto first=t(fresh(stem+"_x1"),DType::F32,{rows,heads,d/2}),second=t(fresh(stem+"_x2"),DType::F32,{rows,heads,d/2});
        {synSplitParams p{};p.axis=0;const TensorId in[]={x},out[]={first,second};g.op("split",in,out,&p,sizeof(p));}
        const auto negative=binary("mult_fwd_f32",second,constant(-1.0f,{1},stem+"_minus"),{rows,heads,d/2},stem+"_neg");
        const auto rotated=t(fresh(stem+"_rot"),DType::F32,{rows,heads,d});
        {const TensorId parts[]={negative,first};g.concat(parts,rotated,2);}
        const auto a=binary("mult_fwd_f32",x,cosine,{rows,heads,d},stem+"_c"),b=binary("mult_fwd_f32",rotated,sine,{rows,heads,d},stem+"_s");
        return round_bf16(binary("add_fwd_f32",a,b,{rows,heads,d},stem),{rows,heads,d},stem);
    }
    // Two-tap grouped dynamic convolution of one side over a block of `rows` (sglang _grouped_conv): per channel c
    // (group c/group_size), out[n] = (base[side][0][c] + d0[n][g]) * h[n] + (base[side][1][c] + d1[n][g]) * h[n-1].
    // base/projection tensors of this (side, tap) are bound sub-ranges: base {1, hidden} F32, projection {groups, hidden}.
    TensorId conv(TensorId h,TensorId h_bf16_input_unused,uint64_t rows,uint64_t hidden,uint64_t groups,std::array<TensorId,2> bases,
                  std::array<TensorId,2> delta,const std::string& stem) {
        (void)h_bf16_input_unused;const uint64_t group=hidden/groups;
        std::array<TensorId,2> k{};
        for(int tap=0;tap<2;++tap) {
            // base[c] + delta[group of c]: one add broadcasting the base over rows and the delta over the group.
            const auto d3=reshape(delta[tap],DType::F32,{rows,groups,1},stem+"_d3"),b3=reshape(bases[tap],DType::F32,{1,groups,group},stem+"_b3");
            const auto wide=binary("add_fwd_f32",d3,b3,{rows,groups,group},stem+"_dw");
            k[tap]=reshape(wide,DType::F32,{rows,hidden},stem+"_k");
        }
        const auto now=binary("mult_fwd_f32",k[0],h,{rows,hidden},stem+"_now");
        std::vector<TensorId> row_ids;for(uint64_t r=0;r<rows;++r) row_ids.push_back(t(fresh(stem+"_row"),DType::F32,{1,hidden}));
        g.split_rows(h,row_ids);
        std::vector<TensorId> shifted{constant(0.0f,{1,hidden},stem+"_zero")};shifted.insert(shifted.end(),row_ids.begin(),row_ids.end()-1);
        const auto previous=t(fresh(stem+"_prev"),DType::F32,{rows,hidden});g.concatenate_rows(shifted,previous);
        const auto before=binary("mult_fwd_f32",k[1],previous,{rows,hidden},stem+"_before");
        return round_bf16(binary("add_fwd_f32",now,before,{rows,hidden},stem),{rows,hidden},stem);
    }
};
// cos/sin [rows, 1, d] of positions first..first+rows-1, each half repeated (NeoX, rotary = d).
void rope_rows(const DflashConfig& c,uint32_t first,uint32_t rows,std::vector<float>& cosine,std::vector<float>& sine) {
    const uint32_t d=c.head_dim,half=d/2;cosine.assign(uint64_t(rows)*d,0);sine.assign(cosine.size(),0);
    for(uint32_t r=0;r<rows;++r) for(uint32_t i=0;i<half;++i) {
        const double angle=double(first+r)*std::pow(double(c.theta),-2.0*double(i)/double(d));
        const float cs=float(std::cos(angle)),sn=float(std::sin(angle));
        cosine[uint64_t(r)*d+i]=cosine[uint64_t(r)*d+half+i]=cs;sine[uint64_t(r)*d+i]=sine[uint64_t(r)*d+half+i]=sn;
    }
}
DflashConfig configured() {
    DflashConfig c;
    if(const char* block=std::getenv("STRATA_HPU_DFLASH_BLOCK");block && *block) {
        const auto value=std::strtoul(block,nullptr,10);
        if(value<2 || value>8) throw std::invalid_argument("STRATA_HPU_DFLASH_BLOCK must be 2..8");
        c.block=uint32_t(value);
    }
    if(const char* value=std::getenv("STRATA_HPU_DFLASH_TOPK");value && *value) {
        char* end=nullptr;const auto k=std::strtoul(value,&end,10);
        if(!end || *end || (k!=16 && k!=32 && k!=64)) throw std::invalid_argument("STRATA_HPU_DFLASH_TOPK must be16,32,64");
        c.top_k=uint32_t(k);
    }
    return c;
}
}
struct DflashDrafter::Impl {
    Device device;DflashConfig c=configured();Binding lm_head;uint32_t vocabulary,capacity,cells=0;
    // Rows one capture pass carries: a draft block during decode, the prefill window when it is wider. The
    // feature rows are the stride of every slot, so append() picks a block out of a wider capture.
    uint32_t feature_rows;
    Weights w;
    // features [slot][feature rows][hidden]: a verify window captures every row of a slot at once.
    DeviceBuffer features;
    std::vector<DeviceBuffer> k_cache,v_cache;
    // The context is a ring of `capacity` (= the sliding window) cells: position p lives in cell p % capacity, so the
    // last `capacity` positions are always kept. Keys are rotated at their absolute positions (relative RoPE is all
    // attention sees), so these tables hold, per cell, the RoPE of the position stored there ([capacity][head_dim],
    // NeoX halves repeated), filled from the host staging copies on every append.
    DeviceBuffer rope_cos,rope_sin;HostBuffer rope_host_cos,rope_host_sin;
    std::map<uint32_t,std::unique_ptr<HostRecipe>> capture; // per row count
    std::map<uint32_t,std::unique_ptr<HostRecipe>> append; // per row count (powers of two)
    std::map<uint32_t,std::unique_ptr<HostRecipe>> drafts;
    HostRecipe* in_flight=nullptr; // begin_draft() launched it; end_draft() collects it
    std::vector<Binding> weight_bindings; // every persistent weight tensor, by graph name
    Impl(const Device& d,const std::string& dir,Binding head,uint32_t v,uint32_t cap,uint32_t rows)
        :device(d),lm_head(std::move(head)),vocabulary(v),capacity(std::min<uint32_t>(cap,c.window)),
         feature_rows(std::max(rows,c.block)),
         w(load(d,dir+"/model.safetensors")),features(d,uint64_t(c.target_layers.size())*feature_rows*c.hidden*4),
         rope_cos(d,uint64_t(capacity)*c.head_dim*4),rope_sin(d,uint64_t(capacity)*c.head_dim*4),
         rope_host_cos(d,uint64_t(capacity)*c.head_dim*4),rope_host_sin(d,uint64_t(capacity)*c.head_dim*4) {
        const uint64_t row=uint64_t(c.kv_heads)*c.head_dim*4;
        for(uint32_t l=0;l<c.layers;++l) {k_cache.emplace_back(d,capacity*row);v_cache.emplace_back(d,capacity*row);}
        // Zero the caches: masked cells must stay finite.
        HostBuffer zero(d,capacity*row);std::memset(zero.data(),0,capacity*row);Stream s(d);
        for(uint32_t l=0;l<c.layers;++l) {upload(s,k_cache[l],zero,capacity*row);upload(s,v_cache[l],zero,capacity*row);}
        s.synchronize();
        const uint64_t h=c.hidden,g=c.groups;
        // q/k/v and gate/up of a layer are one GEMM each in the draft graph: their matrices share a buffer.
        for(uint32_t l=0;l<c.layers;++l) {
            w.fuse(d,layer_name(l,"self_attn.qkv"),{layer_name(l,"self_attn.q_proj.weight"),layer_name(l,"self_attn.k_proj.weight"),layer_name(l,"self_attn.v_proj.weight")});
            w.fuse(d,layer_name(l,"mlp.gate_up"),{layer_name(l,"mlp.gate_proj.weight"),layer_name(l,"mlp.up_proj.weight")});
        }
        const auto add=[&](const std::string& graph_name,const std::string& weight,uint64_t offset=0) {weight_bindings.push_back({graph_name,&w.get(weight),w.offset(weight)+offset});};
        add("fc","fc.weight");add("hidden_norm","hidden_norm.weight");add("norm","norm.weight");
        add("hp","candidate_selector.hidden_projection.weight");add("pred_cb","candidate_selector.predecessor_codebook");add("succ_cb","candidate_selector.successor_codebook");
        for(uint32_t l=0;l<c.layers;++l) {
            const auto L="l"+std::to_string(l)+".";
            for(const auto& [graph,part]:std::initializer_list<std::pair<const char*,const char*>>{{"ln_in","input_layernorm.weight"},{"ln_post","post_attention_layernorm.weight"},
                    {"q","self_attn.q_proj.weight"},{"k","self_attn.k_proj.weight"},{"v","self_attn.v_proj.weight"},{"o","self_attn.o_proj.weight"},
                    {"q_norm","self_attn.q_norm.weight"},{"k_norm","self_attn.k_norm.weight"},{"gate","mlp.gate_proj.weight"},{"up","mlp.up_proj.weight"},{"down","mlp.down_proj.weight"}})
                add(L+graph,layer_name(l,part));
            add(L+"qkv",layer_name(l,"self_attn.qkv"));add(L+"gu",layer_name(l,"mlp.gate_up"));
            for(const char* module:{"attention_conv","mlp_conv"}) add(L+module+".proj_all",layer_name(l,(std::string(module)+".kernel_projection.weight").c_str()));
            for(const char* module:{"attention_conv","mlp_conv"}) for(int side=0;side<2;++side) for(int tap=0;tap<2;++tap) {
                const auto tag=L+module+".s"+std::to_string(side)+"t"+std::to_string(tap);
                add(tag+".base",layer_name(l,(std::string(module)+".base_kernel").c_str()),uint64_t(side*2+tap)*h*4);
                add(tag+".proj",layer_name(l,(std::string(module)+".kernel_projection.weight").c_str()),uint64_t(side*2+tap)*g*h*2);
            }
        }
    }
    Binding weight(const std::string& name) const {
        for(const auto& b:weight_bindings) if(b.name==name) return b;
        throw std::logic_error("DFlash binding missing: "+name);
    }
};
DflashDrafter::DflashDrafter(const Device& d,const std::string& dir,Binding lm_head,uint32_t vocabulary,uint32_t capacity,
                             uint32_t feature_rows)
    :impl_(std::make_unique<Impl>(d,dir,std::move(lm_head),vocabulary,capacity,feature_rows)) {}
DflashDrafter::~DflashDrafter()=default;
const DflashConfig& DflashDrafter::config() const {return impl_->c;}
uint32_t DflashDrafter::cells() const {return impl_->cells;}
void DflashDrafter::reset() {impl_->cells=0;}
void DflashDrafter::truncate(uint32_t cells) {
    auto& p=*impl_;if(p.in_flight || cells>p.cells) throw std::logic_error("DFlash truncate");
    p.cells=cells;
}
int DflashDrafter::feature_slot(uint32_t target_layer) const {
    const auto& t=impl_->c.target_layers;for(size_t i=0;i<t.size();++i) if(t[i]==target_layer) return int(i);return -1;
}
void DflashDrafter::capture(const Stream& stream,const DeviceBuffer& residual,uint32_t streams,int slot,uint32_t rows) {
    auto& p=*impl_;const uint64_t h=p.c.hidden;
    if(slot<0 || uint32_t(slot)>=p.c.target_layers.size() || !rows) throw std::invalid_argument("DFlash capture slot/rows");
    // A capture pass carries at most feature_rows rows; a prefill window wider than a draft block is captured in
    // one pass and append() consumes it block by block, so no window may exceed what the slots hold.
    if(rows>p.feature_rows) throw std::invalid_argument("DFlash capture needs rows <= the drafter's capture rows");
    auto& recipe=p.capture[rows];
    const std::array<Binding,2> io{{{"residual",&residual},{"feature",&p.features,uint64_t(slot)*p.feature_rows*h*4}}};
    if(!recipe) {
        // [rows, streams, hidden] -> stream mean [rows, 1, hidden] (one row: the decode residual).
        GraphBuilder g(p.device);Ops o{g};
        const auto r=o.t("residual",DType::F32,{rows,streams,h},true),sum=o.t("sum",DType::F32,{rows,1,h});
        {ns_Reduction::Params q{};q.reductionDimension=1;const TensorId in[]={r},out[]={sum};g.op("reduce_sum_fwd_f32",in,out,&q,sizeof(q));}
        const auto out=o.t("feature",DType::F32,{rows,1,h},true);
        {const TensorId in[]={sum,o.constant(1.0f/float(streams),{1},"inverse")},outs[]={out};g.op("mult_fwd_f32",in,outs);}
        recipe=std::make_unique<HostRecipe>(p.device,g.compile("strata_dflash_capture_"+std::to_string(rows)),io,std::vector<std::string>{});
    }
    for(const auto& binding:io) recipe->bind_external(binding);
    recipe->launch(stream);
}
void DflashDrafter::append(const Stream& stream,uint32_t position,uint32_t first_row,uint32_t count,bool synchronize) {
    auto& p=*impl_;const auto& c=p.c;const uint64_t h=c.hidden,hk=c.kv_heads,d=c.head_dim,row=hk*d*4,slots=c.target_layers.size();
    if(position!=p.cells || !count || uint64_t(first_row)+count>p.feature_rows) throw std::invalid_argument("DFlash append position/rows");
    if(p.in_flight) throw std::logic_error("DFlash append while a draft is in flight");
    const uint32_t appended=count;
    // Rows that a later row of the same call overwrites in the ring are not written.
    if(count>p.capacity) {const uint32_t skip=count-p.capacity;first_row+=skip;position+=skip;count=p.capacity;}
    // `n` captured rows from `r` become cells `at`.. (contiguous, at + n <= capacity), rotated at positions `pos`..
    const auto io=[&](uint32_t at,uint32_t r) {
        std::vector<Binding> b;
        for(uint64_t s=0;s<slots;++s) b.push_back({"feat"+std::to_string(s),&p.features,(s*p.feature_rows+r)*h*4});
        b.push_back({"cos",&p.rope_cos,uint64_t(at)*d*4});b.push_back({"sin",&p.rope_sin,uint64_t(at)*d*4});
        for(uint32_t l=0;l<c.layers;++l) {
            const auto L="l"+std::to_string(l)+".";
            b.push_back({L+"k_slot",&p.k_cache[l],uint64_t(at)*row});b.push_back({L+"v_slot",&p.v_cache[l],uint64_t(at)*row});
        }
        return b;
    };
    const auto recipe_for=[&](uint32_t n) -> HostRecipe& {
        auto& recipe=p.append[n];
        if(recipe) return *recipe;
        GraphBuilder g(p.device);Ops o{g};
        std::vector<TensorId> parts;for(uint64_t s=0;s<slots;++s) parts.push_back(o.t("feat"+std::to_string(s),DType::F32,{n,h},true));
        const auto feat=o.t("features",DType::F32,{n,slots*h});g.concat(parts,feat,1);
        const auto fc=o.t("fc",DType::BF16,{h,slots*h},true),hn=o.t("hidden_norm",DType::F32,{h},true);
        const auto ctx=o.rms(o.linear(feat,n,slots*h,fc,h,"fc"),n,h,hn,c.epsilon,"ctx");
        const auto cosine=o.t("cos",DType::F32,{n,1,d},true),sine=o.t("sin",DType::F32,{n,1,d},true);
        for(uint32_t l=0;l<c.layers;++l) {
            const auto L="l"+std::to_string(l)+".";
            const auto kw=o.t(L+"k",DType::BF16,{hk*d,h},true),vw=o.t(L+"v",DType::BF16,{hk*d,h},true),kn=o.t(L+"k_norm",DType::F32,{d},true);
            const auto k=o.reshape(o.linear(ctx,n,h,kw,hk*d,L+"kp"),DType::F32,{n*hk,d},L+"k2");
            const auto kr=o.rope(o.reshape(o.rms(k,n*hk,d,kn,c.epsilon,L+"kn"),DType::F32,{n,hk,d},L+"k3"),n,hk,d,cosine,sine,L+"krope");
            const auto ks=o.t(L+"k_slot",DType::F32,{n,hk,d},true);g.reshape(kr,ks);
            const auto vs=o.t(L+"v_slot",DType::F32,{n,hk*d},true);
            {const auto vx=o.t(o.fresh(L+"v16"),DType::BF16,{n,h});g.cast(ctx,vx);
             if(draft_bf16_enabled()) {const auto v16=o.t(o.fresh(L+"v_product16"),DType::BF16,{n,hk*d},true);g.gemm(vx,vw,v16,true);g.cast(v16,vs);}
             else g.gemm(vx,vw,vs,true);}
        }
        auto used=io(0,0);used.push_back(p.weight("fc"));used.push_back(p.weight("hidden_norm"));
        for(uint32_t l=0;l<c.layers;++l) {
            const auto L="l"+std::to_string(l)+".";
            used.push_back(p.weight(L+"k"));used.push_back(p.weight(L+"v"));used.push_back(p.weight(L+"k_norm"));
        }
        recipe=std::make_unique<HostRecipe>(p.device,g.compile("strata_dflash_append_"+std::to_string(n)),used,std::vector<std::string>{});
        return *recipe;
    };
    // No host inputs, so the pieces queue back to back. A draft on another stream needs them finished (synchronize);
    // begin_draft() on this same stream is ordered behind them anyway.
    const uint32_t d32=uint32_t(d);std::vector<float> cosine,sine;
    while(count) {
        const uint32_t at=position%p.capacity,run=std::min(count,p.capacity-at),n=std::bit_floor(run);
        // This piece's RoPE rows into the cells' table entries (the staging rows of a cell are rewritten only
        // `capacity` positions later, after the synchronized append that wrote them).
        rope_rows(c,position,n,cosine,sine);
        std::memcpy(static_cast<float*>(p.rope_host_cos.data())+uint64_t(at)*d32,cosine.data(),cosine.size()*4);
        std::memcpy(static_cast<float*>(p.rope_host_sin.data())+uint64_t(at)*d32,sine.data(),sine.size()*4);
        upload(stream,p.rope_cos,p.rope_host_cos,uint64_t(n)*d*4,uint64_t(at)*d*4,uint64_t(at)*d*4);
        upload(stream,p.rope_sin,p.rope_host_sin,uint64_t(n)*d*4,uint64_t(at)*d*4,uint64_t(at)*d*4);
        auto& recipe=recipe_for(n);
        for(const auto& binding:io(at,first_row)) recipe.bind_external(binding);
        recipe.launch(stream);
        position+=n;first_row+=n;count-=n;
    }
    if(synchronize) stream.synchronize();
    p.cells+=appended;
}
namespace {
std::unique_ptr<Recipe> build_draft(const Device& device,const DflashConfig& c,uint32_t vocabulary,uint32_t bucket) {
    GraphBuilder g(device);Ops o{g};
    const uint64_t n=c.block,h=c.hidden,hq=c.heads,hk=c.kv_heads,d=c.head_dim,f=c.intermediate,gr=c.groups,k=c.top_k,rk=c.rank,b=bucket,group=hq/hk,e=n-1;
    // Host inputs (set per draft) are persistent like the weights.
    const auto x=o.t("x",DType::F32,{n,h},true),cosine=o.t("cos",DType::F32,{n,1,d},true),sine=o.t("sin",DType::F32,{n,1,d},true);
    const auto mask=o.t("mask",DType::F32,{b+n},true),anchor=o.t("anchor",DType::F32,{1,1},true);
    const auto scale=o.constant(1.0f/std::sqrt(float(d)),{1},"scale");
    TensorId hidden=x,residual=x;
    // A conv module's coefficients for both sides and taps ([side][tap][group]) from one GEMM of its input.
    const auto coefficients=[&](TensorId source,const std::string& L,const char* module,const std::string& stem) {
        const auto all=o.linear(source,n,h,o.t(L+module+".proj_all",DType::BF16,{4*gr,h},true),4*gr,stem+"_coef");
        std::array<TensorId,4> parts{};for(auto& part:parts) part=o.t(o.fresh(stem+"_delta"),DType::F32,{n,gr});
        synSplitParams sp{};sp.axis=0;const TensorId in[]={all};g.op("split",in,parts,&sp,sizeof(sp));
        return parts;
    };
    const auto conv=[&](TensorId input,const std::array<TensorId,4>& deltas,const std::string& L,const char* module,int side,const std::string& stem) {
        std::array<TensorId,2> bases{};
        for(int tap=0;tap<2;++tap) bases[tap]=o.t(L+module+".s"+std::to_string(side)+"t"+std::to_string(tap)+".base",DType::F32,{1,h},true);
        return o.conv(input,input,n,h,gr,bases,{deltas[side*2],deltas[side*2+1]},stem);
    };
    for(uint32_t l=0;l<c.layers;++l) {
        const auto L="l"+std::to_string(l)+".";
        const auto t=[&](const std::string& name,DType dtype,std::vector<uint64_t> shape) {return o.t(L+name,dtype,std::move(shape),true);};
        if(l) residual=o.round_bf16(o.binary("add_fwd_f32",hidden,residual,{n,h},L+"res_in"),{n,h},L+"res_in");
        const auto normed=o.rms(residual,n,h,t("ln_in",DType::F32,{h}),c.epsilon,L+"ln_in");
        const auto attention_deltas=coefficients(normed,L,"attention_conv",L+"aconv");
        const auto a_in=conv(normed,attention_deltas,L,"attention_conv",0,L+"aconv0");
        // q, k, v: one GEMM over the layer's fused [q; k; v] rows, split by columns.
        const auto qkv=o.linear(a_in,n,h,t("qkv",DType::BF16,{(hq+2*hk)*d,h}),(hq+2*hk)*d,L+"qkvp");
        const auto qp=o.t(o.fresh(L+"qp"),DType::F32,{n,hq*d}),kp=o.t(o.fresh(L+"kp"),DType::F32,{n,hk*d}),vp=o.t(o.fresh(L+"vp"),DType::F32,{n,hk*d});
        {synSplitParams sp{};sp.axis=0;const TensorId in[]={qkv},out[]={qp,kp,vp};g.op("split",in,out,&sp,sizeof(sp));}
        const auto q=o.rms(o.reshape(qp,DType::F32,{n*hq,d},L+"q2"),n*hq,d,t("q_norm",DType::F32,{d}),c.epsilon,L+"qn");
        const auto kk=o.rms(o.reshape(kp,DType::F32,{n*hk,d},L+"k2"),n*hk,d,t("k_norm",DType::F32,{d}),c.epsilon,L+"kn");
        const auto v=o.reshape(vp,DType::F32,{n,hk,d},L+"v3");
        const auto qr=o.rope(o.reshape(q,DType::F32,{n,hq,d},L+"q3"),n,hq,d,cosine,sine,L+"qrope");
        const auto kr=o.rope(o.reshape(kk,DType::F32,{n,hk,d},L+"k3"),n,hk,d,cosine,sine,L+"krope");
        // [n, hk, group, d] -> [hk, n, group, d] -> [hk, n*group, d]
        const auto qg=o.reshape(o.transpose4(o.reshape(qr,DType::F32,{n,hk,group,d},L+"q4"),{hk,n,group,d},L+"qt"),DType::F32,{hk,n*group,d},L+"qg");
        const auto key=o.t(o.fresh(L+"key"),DType::F32,{hk,b+n,d}),value=o.t(o.fresh(L+"value"),DType::F32,{hk,b+n,d});
        {const TensorId parts[]={o.transpose12(t("k_cache",DType::F32,{b,hk,d}),{hk,b,d},L+"kct"),o.transpose12(kr,{hk,n,d},L+"kbt")};g.concat(parts,key,1);}
        {const TensorId parts[]={o.transpose12(t("v_cache",DType::F32,{b,hk,d}),{hk,b,d},L+"vct"),o.transpose12(v,{hk,n,d},L+"vbt")};g.concat(parts,value,1);}
        const auto scores=o.t(o.fresh(L+"scores"),DType::F32,{hk,n*group,b+n});g.batch_gemm(qg,key,scores,true);
        const auto scaled=o.binary("mult_fwd_f32",scores,scale,{hk,n*group,b+n},L+"scaled");
        const auto masked=o.binary("add_fwd_f32",scaled,mask,{hk,n*group,b+n},L+"masked");
        const auto prob=o.t(o.fresh(L+"prob"),DType::F32,{hk,n*group,b+n});g.softmax_last(masked,prob);
        const auto context=o.t(o.fresh(L+"context"),DType::F32,{hk,n*group,d});g.batch_gemm(prob,value,context);
        const auto ctx=o.reshape(o.transpose4(o.reshape(context,DType::F32,{hk,n,group,d},L+"c4"),{n,hk,group,d},L+"ct"),DType::F32,{n,hq*d},L+"cf");
        const auto attn=o.linear(ctx,n,hq*d,t("o",DType::BF16,{h,hq*d}),h,L+"op");
        const auto a_out=conv(attn,attention_deltas,L,"attention_conv",1,L+"aconv1");
        residual=o.round_bf16(o.binary("add_fwd_f32",a_out,residual,{n,h},L+"res_mid"),{n,h},L+"res_mid");
        const auto post=o.rms(residual,n,h,t("ln_post",DType::F32,{h}),c.epsilon,L+"ln_post");
        const auto mlp_deltas=coefficients(post,L,"mlp_conv",L+"mconv");
        const auto m_in=conv(post,mlp_deltas,L,"mlp_conv",0,L+"mconv0");
        const auto gu=o.linear(m_in,n,h,t("gu",DType::BF16,{2*f,h}),2*f,L+"gup");
        const auto gate=o.t(o.fresh(L+"gp"),DType::F32,{n,f}),up=o.t(o.fresh(L+"upp"),DType::F32,{n,f});
        {synSplitParams sp{};sp.axis=0;const TensorId in[]={gu},out[]={gate,up};g.op("split",in,out,&sp,sizeof(sp));}
        const auto sig=o.t(o.fresh(L+"sig"),DType::F32,{n,f});g.sigmoid(gate,sig);
        const auto act=o.binary("mult_fwd_f32",o.binary("mult_fwd_f32",gate,sig,{n,f},L+"silu"),up,{n,f},L+"act");
        const auto mlp=o.linear(o.round_bf16(act,{n,f},L+"act"),n,f,t("down",DType::BF16,{h,f}),h,L+"dp");
        hidden=conv(mlp,mlp_deltas,L,"mlp_conv",1,L+"mconv1");
    }
    const auto final_in=o.round_bf16(o.binary("add_fwd_f32",hidden,residual,{n,h},"final_in"),{n,h},"final_in");
    const auto out=o.rms(final_in,n,h,o.t("norm",DType::F32,{h},true),c.epsilon,"final");
    std::vector<TensorId> rows;for(uint64_t r=0;r<n;++r) rows.push_back(o.t(o.fresh("out_row"),DType::F32,{1,h}));
    g.split_rows(out,rows);
    const auto drafted=o.t("drafted",DType::F32,{e,h});g.concatenate_rows(std::span<const TensorId>(rows).subspan(1),drafted);
    const auto logits=o.linear(drafted,e,h,o.t("lm_head",DType::BF16,{vocabulary,h},true),vocabulary,"logits");
    const auto values=o.t("cand_values",DType::F32,{e,k}),ids=o.t("cand_ids",DType::I32,{e,k});
    {ns_TopkNode::Params p{};p.bsw=uint32_t(k);p.axis=0;p.bottomK=0;const TensorId in[]={logits},outs[]={values,ids};g.op("topk",in,outs,&p,sizeof(p));}
    const auto hp=o.linear(drafted,e,h,o.t("hp",DType::BF16,{rk,h},true),rk,"hp");
    const auto pred_cb=o.t("pred_cb",DType::BF16,{vocabulary,rk},true),succ_cb=o.t("succ_cb",DType::BF16,{vocabulary,rk},true);
    const auto gather=[&](TensorId table,TensorId index,const std::string& stem) {
        const auto rows16=o.t(o.fresh(stem),DType::BF16,{e,k,rk}),rows32=o.t(o.fresh(stem+"_f"),DType::F32,{e,k,rk});
        ns_GatherKernel::Params p{};p.axis=1;const TensorId in[]={table,index},out[]={rows16};g.op("gather_fwd_bf16",in,out,&p,sizeof(p));
        g.cast(rows16,rows32);return rows32;
    };
    const auto succ=gather(succ_cb,ids,"succ");
    // Predecessor ids: the anchor for row 1's candidates, then row e-1's candidates.
    const auto anchor_ids=o.t("anchor_ids",DType::I32,{1,k});
    {const auto anchor_row=o.t("anchor_row",DType::F32,{1,k});const TensorId in[]={anchor},out[]={anchor_row};g.op("broadcast",in,out);
     const auto ai=o.t("anchor_i32",DType::I32,{1,k});const TensorId cin[]={anchor_row},cout[]={ai};g.op("cast_f32_to_i32",cin,cout);g.reshape(ai,anchor_ids);}
    std::vector<TensorId> id_rows;for(uint64_t r=0;r<e;++r) id_rows.push_back(o.t(o.fresh("id_row"),DType::I32,{1,k}));
    g.split_rows(ids,id_rows);
    std::vector<TensorId> pred_rows{anchor_ids};pred_rows.insert(pred_rows.end(),id_rows.begin(),id_rows.end()-1);
    const auto pred_ids=o.t("pred_ids",DType::I32,{e,k});g.concatenate_rows(pred_rows,pred_ids);
    const auto pred=gather(pred_cb,pred_ids,"pred");
    const auto hp3=o.reshape(hp,DType::F32,{e,1,rk},"hp3");
    const auto weighted=o.binary("mult_fwd_f32",pred,hp3,{e,k,rk},"pred_h");
    const auto pair=o.t("pair",DType::F32,{e,k,k});
    if(draft_bf16_enabled()) {
        const auto a=o.t("selector_a16",DType::BF16,{e,k,rk},true),b=o.t("selector_b16",DType::BF16,{e,k,rk});
        const auto p=o.t("selector_pair16",DType::BF16,{e,k,k},true);
        g.cast(weighted,a);g.cast(succ,b);g.batch_gemm(a,b,p,true);g.cast(p,pair);
    } else g.batch_gemm(weighted,succ,pair,true);
    const auto unary=o.reshape(values,DType::F32,{e,1,k},"unary3");
    const auto scores=o.t("scores",DType::F32,{e,k,k},true);{const TensorId in[]={pair,unary},out[]={scores};g.op("add_fwd_f32",in,out);}
    const auto ids_f=o.t("ids_f",DType::F32,{e,k},true);{const TensorId in[]={ids},out[]={ids_f};g.op("cast_i32_to_f32",in,out);}
    return g.compile("strata_dflash_draft_"+std::to_string(bucket));
}
}
std::vector<int32_t> DflashDrafter::draft(std::span<const float> anchor_embedding,std::span<const float> mask_embedding,int32_t anchor,uint32_t position) {
    Stream stream(impl_->device);begin_draft(stream,anchor_embedding,mask_embedding,anchor,position);return end_draft();
}
bool DflashDrafter::drafting() const {return impl_->in_flight!=nullptr;}
void DflashDrafter::begin_draft(const Stream& stream,std::span<const float> anchor_embedding,std::span<const float> mask_embedding,int32_t anchor,uint32_t position) {
    auto& p=*impl_;const auto& c=p.c;const uint64_t n=c.block,h=c.hidden;
    if(p.in_flight || position!=p.cells || anchor_embedding.size()!=h || mask_embedding.size()!=h) throw std::invalid_argument("DFlash draft input");
    // The ring's used cells (all of them once the context passed the window); attention ignores their order.
    const uint32_t used=std::min(p.cells,p.capacity);
    uint32_t bucket=64;while(bucket<std::max<uint32_t>(used,1)) bucket*=2;
    bucket=std::min(bucket,p.capacity);
    auto& recipe=p.drafts[bucket];
    std::vector<Binding> io;
    for(uint32_t l=0;l<c.layers;++l) {const auto L="l"+std::to_string(l)+".";io.push_back({L+"k_cache",&p.k_cache[l],0});io.push_back({L+"v_cache",&p.v_cache[l],0});}
    if(!recipe) {
        auto first=p.weight_bindings;first.push_back({"lm_head",p.lm_head.buffer,p.lm_head.offset});first.insert(first.end(),io.begin(),io.end());
        // Bind what the draft graph declares (append() owns fc/hidden_norm; the fused matrices replace their parts).
        auto graph=build_draft(p.device,c,p.vocabulary,bucket);std::set<std::string> declared;for(const auto& t:graph->tensors()) declared.insert(t.name);
        std::vector<Binding> used;for(const auto& b:first) if(declared.contains(b.name)) used.push_back(b);
        recipe=std::make_unique<HostRecipe>(p.device,std::move(graph),used,std::vector<std::string>{"scores","ids_f"});
    }
    std::vector<float> x(n*h);std::copy(anchor_embedding.begin(),anchor_embedding.end(),x.begin());
    for(uint64_t r=1;r<n;++r) std::copy(mask_embedding.begin(),mask_embedding.end(),x.begin()+r*h);
    std::vector<float> cosine,sine;rope_rows(c,position,uint32_t(n),cosine,sine);
    std::vector<float> mask(bucket+n,0.0f);for(uint32_t i=used;i<bucket;++i) mask[i]=-INFINITY;
    const std::array<float,1> anchor_value{float(anchor)};
    recipe->set("x",x);recipe->set("cos",cosine);recipe->set("sin",sine);recipe->set("mask",mask);recipe->set("anchor",anchor_value);
    // Queued behind whatever `stream` already holds (the appends of this context's last cells).
    recipe->launch(stream);p.in_flight=recipe.get();
}
std::vector<int32_t> DflashDrafter::end_draft() {
    auto& p=*impl_;const auto& c=p.c;const uint64_t n=c.block,k=c.top_k;
    if(!p.in_flight) throw std::logic_error("DFlash end_draft without begin_draft");
    auto* recipe=p.in_flight;p.in_flight=nullptr;recipe->wait();
    const auto scores=recipe->values("scores"),ids=recipe->values("ids_f");
    // Greedy walk: row 1 over its own candidates (every predecessor row is the anchor), then edge by edge.
    std::vector<int32_t> tokens;uint64_t previous=0;
    for(uint64_t e=0;e+1<n;++e) {
        const uint64_t from=e?previous:0;uint64_t best=0;
        for(uint64_t cand=1;cand<k;++cand) if(scores[(e*k+from)*k+cand]>scores[(e*k+from)*k+best]) best=cand;
        tokens.push_back(int32_t(ids[e*k+best]));previous=best;
    }
    return tokens;
}
}
