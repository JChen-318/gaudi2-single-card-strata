#include "strata/hpu/mtp_weights.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <charconv>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace strata::hpu {
namespace {
struct Mapping {
    int fd=-1;const uint8_t* data=nullptr;uint64_t bytes=0;std::string path;
    explicit Mapping(const std::filesystem::path& input):path(std::filesystem::canonical(input).string()) {
        fd=::open(path.c_str(),O_RDONLY|O_CLOEXEC);if(fd<0) throw std::runtime_error("cannot open MTP file: "+path);
        struct stat st{};
        if(::fstat(fd,&st) || !S_ISREG(st.st_mode) || st.st_size<=0 || uint64_t(st.st_size)>SIZE_MAX) {
            ::close(fd);fd=-1;throw std::invalid_argument("invalid MTP mapped file: "+path);
        }
        bytes=uint64_t(st.st_size);auto* address=::mmap(nullptr,size_t(bytes),PROT_READ,MAP_PRIVATE,fd,0);
        if(address==MAP_FAILED) {::close(fd);fd=-1;throw std::runtime_error("MTP mmap failed: "+path);}
        data=static_cast<const uint8_t*>(address);
    }
    ~Mapping() {if(data) ::munmap(const_cast<uint8_t*>(data),size_t(bytes));if(fd>=0) ::close(fd);}
    Mapping(const Mapping&)=delete;Mapping& operator=(const Mapping&)=delete;
};
uint64_t integer(const std::string& text) {
    uint64_t value=0;const auto [end,error]=std::from_chars(text.data(),text.data()+text.size(),value);
    if(error!=std::errc{} || end!=text.data()+text.size()) throw std::invalid_argument("invalid MTP index integer");
    return value;
}
uint32_t type(const std::string& kind) {
    if(kind=="q8_0") return 8;
    if(kind=="bf16") return 30;
    if(kind=="f32") return 0;
    throw std::invalid_argument("unsupported MTP dense kind: "+kind);
}
std::string identity(const std::string& path,uint64_t bytes) {
    struct stat st{};if(::stat(path.c_str(),&st) || !S_ISREG(st.st_mode) || st.st_size<0 || uint64_t(st.st_size)!=bytes)
        throw std::runtime_error("MTP file changed or unavailable");
    std::ostringstream out;out<<path.size()<<':'<<path<<':'<<st.st_dev<<':'<<st.st_ino<<':'<<st.st_size<<':'<<st.st_mtim.tv_sec<<':'<<st.st_mtim.tv_nsec<<'\n';return out.str();
}
}
struct MtpWeights::Impl {
    std::shared_ptr<Mapping> dense,experts;
    std::string index_path;uint64_t index_bytes=0,payload=0;
    std::vector<MtpTensor> tensors;std::map<std::string,size_t> names;
    const MtpTensor& get(const std::string& name) const {
        const auto found=names.find(name);if(found==names.end()) throw std::invalid_argument("MTP tensor missing: "+name);return tensors[found->second];
    }
};
struct DraftVocabulary::Impl {std::shared_ptr<Mapping> mapping;std::vector<int32_t> ids;};
DraftVocabulary::DraftVocabulary(const std::string& path,uint32_t vocabulary) {
    if(!vocabulary || vocabulary>INT32_MAX) throw std::invalid_argument("invalid draft vocabulary size");
    const auto file_bytes=std::filesystem::file_size(path);
    if(!file_bytes || file_bytes%4 || file_bytes>uint64_t(vocabulary)*4) throw std::invalid_argument("draft vocabulary file size mismatch");
    auto p=std::make_shared<Impl>();p->mapping=std::make_shared<Mapping>(path);const auto bytes=p->mapping->bytes;
    if(bytes%4 || bytes>uint64_t(vocabulary)*4) throw std::invalid_argument("draft vocabulary file size mismatch");
    p->ids.reserve(bytes/4);std::vector<bool> seen(vocabulary);
    for(uint64_t offset=0;offset<bytes;offset+=4) {
        const auto* raw=p->mapping->data+offset;
        const auto bits=uint32_t(raw[0])|(uint32_t(raw[1])<<8)|(uint32_t(raw[2])<<16)|(uint32_t(raw[3])<<24);
        const auto id=std::bit_cast<int32_t>(bits);
        if(id<0 || uint32_t(id)>=vocabulary || seen[uint32_t(id)]) throw std::invalid_argument("invalid/duplicate draft vocabulary token");
        seen[uint32_t(id)]=true;p->ids.push_back(id);
    }
    impl_=std::move(p);
}
const std::vector<int32_t>& DraftVocabulary::ids() const {return impl_->ids;}
std::string DraftVocabulary::file_identity() const {return identity(impl_->mapping->path,impl_->mapping->bytes);}
MtpWeights::MtpWeights(const std::string& directory,uint64_t scratch) {
    using namespace strata::kernels::cpu;
    if(scratch<BLOB) throw std::invalid_argument("MTP expert relayout budget too small");
    auto p=std::make_shared<Impl>();const std::filesystem::path root(directory);
    p->index_path=std::filesystem::canonical(root/"dense.txt").string();
    p->index_bytes=std::filesystem::file_size(p->index_path);
    if(!p->index_bytes || p->index_bytes>65536) throw std::invalid_argument("MTP index exceeds bounded size");
    p->dense=std::make_shared<Mapping>(root/"dense.bin");p->experts=std::make_shared<Mapping>(root/"experts.bin");
    if(p->experts->bytes!=uint64_t(NE)*BLOB) throw std::invalid_argument("MTP experts.bin size does not match 512 Q2_0 blobs");
    std::ifstream input(p->index_path);if(!input) throw std::runtime_error("cannot read MTP index");
    std::string index(size_t(p->index_bytes),'\0');input.read(index.data(),std::streamsize(index.size()));
    if(size_t(input.gcount())!=index.size() || input.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("MTP index changed during read");
    std::istringstream lines(index);
    std::string line;
    while(std::getline(lines,line)) {
        if(line.empty()) continue;
        std::istringstream fields(line);MtpTensor t;std::string rows,columns,offset,bytes,extra;
        if(!(fields>>t.name>>t.kind>>rows>>columns>>offset>>bytes) || fields>>extra || t.name.size()>256 || t.name.find('\0')!=std::string::npos || p->tensors.size()>=128)
            throw std::invalid_argument("malformed MTP dense index");
        t.rows=integer(rows);t.columns=integer(columns);t.offset=integer(offset);t.bytes=integer(bytes);(void)type(t.kind);
        if(!t.rows || !t.columns || t.rows>1048576 || t.columns>1048576 || t.offset%256 || (t.kind=="q8_0" && t.columns%32))
            throw std::invalid_argument("invalid MTP dense geometry/alignment");
        const auto expected=t.kind=="q8_0"?checked_product(checked_product(t.rows,t.columns/32),34):checked_product(checked_product(t.rows,t.columns),t.kind=="f32"?4:2);
        if(t.bytes!=expected) throw std::invalid_argument("MTP dense byte count mismatch");
        check_range(p->dense->bytes,t.offset,t.bytes);
        if(p->names.contains(t.name)) throw std::invalid_argument("duplicate MTP tensor name");
        for(const auto& other:p->tensors) if(t.offset<other.offset+other.bytes && other.offset<t.offset+t.bytes) throw std::invalid_argument("overlapping MTP dense tensors");
        p->names.emplace(t.name,p->tensors.size());
        const auto decoded=checked_product(checked_product(t.rows,t.columns),t.kind=="f32"?4:2);
        if(decoded>UINT64_MAX-p->payload) throw std::overflow_error("MTP dense payload overflow");
        p->payload+=decoded;p->tensors.push_back(std::move(t));
    }
    if(!lines.eof()) throw std::runtime_error("MTP index parse failed");
    auto need=[&](const std::string& name,const std::string& kind,uint64_t rows,uint64_t cols) {
        const auto& t=p->get(name);if(t.kind!=kind || t.rows!=rows || t.columns!=cols) throw std::invalid_argument("MTP schema mismatch: "+name);
    };
    const auto& g=core::kCompiledGeometry;const uint64_t h=H,ff=FF,n=h*g.hc,lr=g.hc_lr;
    for(const auto* name:{"fc_embedding.weight","fc_hidden.weight"}) need(name,"q8_0",h,h);
    need("self_attn.q_proj.weight","q8_0",uint64_t(g.n_head)*g.head_dim*2,h);
    for(const auto* name:{"self_attn.k_proj.weight","self_attn.v_proj.weight"}) need(name,"q8_0",uint64_t(g.n_head_kv)*g.head_dim,h);
    need("self_attn.o_proj.weight","q8_0",h,uint64_t(g.n_head)*g.head_dim);
    need("pre_fc_norm_embedding.weight","f32",1,h);need("pre_fc_norm_hidden.weight","f32",1,n);
    for(const auto* name:{"self_attn.q_norm.weight","self_attn.k_norm.weight"}) need(name,"f32",1,g.head_dim);
    for(const auto* stem:{"attn_hyper_connection.","mlp_hyper_connection.","hyper_connection_mixer."}) {
        const std::string prefix(stem);need(prefix+"hc_norm.weight","f32",1,n);
        need(prefix+"input_mix_weight_down.weight","bf16",lr,n);need(prefix+"input_mix_weight_up.weight","bf16",n,lr);
        if(prefix!="hyper_connection_mixer.") need(prefix+"block_inject_weight.weight","bf16",g.hc,n);
    }
    need("mlp.gate.weight","bf16",NE,h);
    const auto& shared_gate=p->get("mlp.shared_expert_gate.weight");
    if(shared_gate.kind!="bf16" || !((shared_gate.rows==1 && shared_gate.columns==h) || (shared_gate.rows==h && shared_gate.columns==1)))
        throw std::invalid_argument("MTP schema mismatch: mlp.shared_expert_gate.weight");
    for(const auto* name:{"gate_proj.weight","up_proj.weight"}) need(std::string("mlp.shared_expert.")+name,"q8_0",ff,h);
    need("mlp.shared_expert.down_proj.weight","q8_0",h,ff);
    impl_=std::move(p);
    for(const auto& t:impl_->tensors) if(t.kind=="f32") (void)constant(t.name,checked_product(t.rows,t.columns));
}
const std::vector<MtpTensor>& MtpWeights::inventory() const {return impl_->tensors;}
const MtpTensor& MtpWeights::info(const std::string& name) const {return impl_->get(name);}
WeightSlice MtpWeights::matrix(const std::string& name) const {
    const auto& t=info(name);return WeightSlice::raw_rows(impl_->dense,{impl_->dense->data+t.offset,size_t(t.bytes)},type(t.kind),t.columns,t.rows);
}
WeightSlice MtpWeights::vector(const std::string& name) const {
    const auto& t=info(name);if(t.rows!=1 && t.columns!=1) throw std::invalid_argument("MTP tensor is not a vector");
    return WeightSlice::raw_rows(impl_->dense,{impl_->dense->data+t.offset,size_t(t.bytes)},type(t.kind),checked_product(t.rows,t.columns),1);
}
std::vector<float> MtpWeights::constant(const std::string& name,uint64_t elements) const {
    const auto& t=info(name);if(t.kind!="f32" || checked_product(t.rows,t.columns)!=elements || elements>1048576) throw std::invalid_argument("invalid MTP constant shape/budget");
    std::vector<float> out(elements);matrix(name).convert_rows_f32(0,out);for(float x:out) if(!std::isfinite(x)) throw std::invalid_argument("non-finite MTP norm");return out;
}
std::array<WeightSlice,3> MtpWeights::expert(uint32_t id) const {
    using namespace strata::kernels::cpu;
    if(id>=NE) throw std::out_of_range("MTP expert ID");
    auto conventional=std::make_shared<std::vector<uint8_t>>(BLOB);const auto* blob=impl_->experts->data+uint64_t(id)*BLOB;
    const uint64_t role_bytes=BLOB/3;
    for(uint32_t role=0;role<3;++role) {
        const uint32_t rows=role==2?H:FF,blocks=role==2?SC_D:SC_GU;
        for(uint32_t row=0;row<rows;++row) {
            const uint64_t physical=role==2?row:2*row+role;
            for(uint32_t block=0;block<blocks;++block) {
                auto* out=conventional->data()+role*role_bytes+(uint64_t(row)*blocks+block)*BB;
                const auto scale=(role==2?O_D_SCALES:O_GU_SCALES)+(physical*blocks+block)*2;
                const auto codes=(role==2?O_D_CODES:O_GU_CODES)+physical*blocks*16+block*16;
                std::memcpy(out,blob+scale,2);std::memcpy(out+2,blob+codes,16);
            }
        }
    }
    std::array<WeightSlice,3> slices;
    for(uint32_t role=0;role<3;++role) slices[role]=WeightSlice::raw_rows(conventional,{conventional->data()+role*role_bytes,size_t(role_bytes)},42,role==2?FF:H,role==2?H:FF);
    return slices;
}
uint64_t MtpWeights::dense_payload_bytes() const {return impl_->payload;}
uint64_t MtpWeights::packed_projection_payload_bytes(bool q4) const {
    auto total=impl_->payload;
    for(const auto* name:{"fc_embedding.weight","fc_hidden.weight","self_attn.q_proj.weight","self_attn.k_proj.weight","self_attn.v_proj.weight","self_attn.o_proj.weight"}) {
        const auto& t=info(name);const auto decoded=checked_product(checked_product(t.rows,t.columns),2);
        const auto packed=q4?checked_product(checked_product(t.rows,t.columns/32),18):t.bytes;
        if(t.kind!="q8_0" || packed>decoded || decoded-packed>total) throw std::logic_error("MTP packed projection budget inconsistency");
        total-=decoded-packed;
    }
    return total;
}
std::string MtpWeights::file_identity() const {return identity(impl_->index_path,impl_->index_bytes)+identity(impl_->dense->path,impl_->dense->bytes)+identity(impl_->experts->path,impl_->experts->bytes);}
}
