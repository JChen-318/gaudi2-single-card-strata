#include "strata/hpu/control_vector.hpp"
#include "strata/hpu/tensor.hpp"
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
#endif
#include "strata/artifact/gguf_reader.hpp"
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
#include <bit>
#include <charconv>
#include <cstring>
#include <filesystem>
#include <set>
#include <sstream>
#include <sys/stat.h>

namespace strata::hpu {
struct ControlVectors::Impl {
    ControlVectorOptions options;
    uint32_t hidden=0,layers=0,steered=0;
    uint64_t signature=14695981039346656037ull;
    std::vector<float> directions,scales;
    std::vector<std::pair<std::string,uint64_t>> paths;
    void hash(uint64_t value) {for(int i=0;i<8;++i) {signature^=uint8_t(value>>(i*8));signature*=1099511628211ull;}}
};
ControlVectors::ControlVectors(ControlVectorOptions options,uint32_t layers,uint32_t hidden) {
    if(!layers || layers>1024 || !hidden || hidden>65536 || uint64_t(layers)*hidden>1048576 || options.files.empty() || options.files.size()>64 ||
       (options.mode!=ControlVectorMode::Add && options.mode!=ControlVectorMode::Project)) throw std::invalid_argument("invalid control vector geometry/options");
    auto p=std::make_shared<Impl>();p->options=std::move(options);p->hidden=hidden;p->layers=layers;
    std::vector<float> data(uint64_t(layers)*hidden);std::vector<bool> have(layers);
    for(const auto& [input,scale]:p->options.files) {
        if(!std::isfinite(scale)) throw std::invalid_argument("non-finite control vector scale");
        const auto path=std::filesystem::canonical(input).string();const auto bytes=std::filesystem::file_size(path);
        if(!bytes || bytes>64ull*1024*1024) throw std::invalid_argument("control vector file exceeds bounded 64 MiB catalog limit");
        strata::GgufFile file(path);const auto* arch=file.get("general.architecture");
        if(!arch || arch->s!="controlvector") throw std::invalid_argument("not a controlvector GGUF: "+path);
        uint32_t found=0;std::set<std::string> names;
        for(const auto& tensor:file.tensors()) {
            if(!names.insert(tensor.name).second) throw std::invalid_argument("duplicate control vector tensor");
            if(!tensor.name.starts_with("direction.")) continue;
            const auto suffix=tensor.name.substr(10);uint32_t layer=0;
            const auto [end,error]=std::from_chars(suffix.data(),suffix.data()+suffix.size(),layer);
            if(error!=std::errc{} || end!=suffix.data()+suffix.size()) throw std::invalid_argument("malformed control vector layer name");
            if(!layer || layer>=layers) continue; // original layer-0/outside-model rule
            if(tensor.type!=0 || tensor.elements()!=hidden) throw std::invalid_argument("control vector direction must be hidden-width F32");
            check_range(file.file_size(),file.data_start(),0);check_range(file.file_size()-file.data_start(),tensor.offset,uint64_t(hidden)*4);
            const auto* source=file.tensor_data(tensor);
            for(uint32_t column=0;column<hidden;++column) {
                float value;std::memcpy(&value,source+uint64_t(column)*4,4);
                const auto index=uint64_t(layer)*hidden+column;
                if(!std::isfinite(value)) throw std::invalid_argument("non-finite control vector direction");
                const float scaled=scale*value;data[index]+=scaled;
                if(!std::isfinite(data[index])) throw std::invalid_argument("control vector sum overflow");
            }
            have[layer]=true;++found;
        }
        if(!found) throw std::invalid_argument("control vector has no direction in this model");
        p->paths.emplace_back(path,bytes);
    }
    auto& o=p->options;o.first=o.first<=0?1:o.first;o.last=o.last<=0 || o.last>=int(layers)?int(layers)-1:o.last;
    if(o.mode==ControlVectorMode::Add) o.single=-1; // original single-direction applies to project only
    if(o.single>=0 && (o.single>=int(layers) || !have[size_t(o.single)])) throw std::invalid_argument("control vector single layer has no direction");
    p->directions.resize(data.size());p->scales.resize(layers);
    for(int layer=o.first;layer<=o.last && layer<int(layers);++layer) {
        const auto source=o.single>=0?o.single:layer;if(!have[size_t(source)]) continue;
        const auto* d=data.data()+uint64_t(source)*hidden;auto* out=p->directions.data()+uint64_t(layer)*hidden;
        if(o.mode==ControlVectorMode::Project) {
            double norm=0;for(uint32_t column=0;column<hidden;++column) norm+=double(d[column])*d[column];norm=std::sqrt(norm);
            if(norm<=0) continue;
            if(norm>double(std::numeric_limits<float>::max())) throw std::invalid_argument("control vector norm overflows F32");
            p->scales[size_t(layer)]=float(norm);for(uint32_t column=0;column<hidden;++column) out[column]=float(d[column]/norm);
        } else {p->scales[size_t(layer)]=1;std::copy_n(d,hidden,out);}
        ++p->steered;
    }
    if(!p->steered) throw std::invalid_argument("control vector has no direction in the selected layer range");
    p->hash(uint32_t(o.mode));p->hash(uint32_t(o.first));p->hash(uint32_t(o.last));p->hash(uint32_t(o.single));
    for(float value:p->directions) p->hash(std::bit_cast<uint32_t>(value));
    for(float value:p->scales) p->hash(std::bit_cast<uint32_t>(value));
    if(!p->signature) p->signature=1;
    impl_=std::move(p);
}
bool ControlVectors::covers(uint32_t layer) const {return layer<impl_->layers && impl_->scales[layer]!=0;}
std::span<const float> ControlVectors::direction(uint32_t layer) const {if(layer>=impl_->layers) throw std::out_of_range("control vector layer");return std::span(impl_->directions).subspan(uint64_t(layer)*impl_->hidden,impl_->hidden);}
float ControlVectors::scale(uint32_t layer) const {return impl_->scales.at(layer);}
ControlVectorMode ControlVectors::mode() const {return impl_->options.mode;}
uint64_t ControlVectors::signature() const {return impl_->signature;}
uint64_t ControlVectors::host_bytes() const {return uint64_t(impl_->directions.size()+impl_->scales.size())*4;}
std::string ControlVectors::summary() const {const auto& o=impl_->options;return std::string(o.mode==ControlVectorMode::Add?"add:":"project:")+std::to_string(o.first)+"-"+std::to_string(o.last)+(o.single>=0?":single"+std::to_string(o.single):"");}
std::string ControlVectors::file_identity() const {
    std::ostringstream out;
    for(const auto& [path,bytes]:impl_->paths) {
#if defined(_WIN32)
        // Portable host-test path only; the HPU runtime build is Linux. Linux
        // below retains inode/device and nanosecond stat identity.
        if(std::filesystem::file_size(path)!=bytes) throw std::runtime_error("control vector file changed/unavailable");
        out<<path.size()<<':'<<path<<':'<<bytes<<':'<<std::filesystem::last_write_time(path).time_since_epoch().count()<<'\n';
#else
        struct stat st{};if(::stat(path.c_str(),&st) || !S_ISREG(st.st_mode) || st.st_size<0 || uint64_t(st.st_size)!=bytes) throw std::runtime_error("control vector file changed/unavailable");
        out<<path.size()<<':'<<path<<':'<<st.st_dev<<':'<<st.st_ino<<':'<<st.st_size<<':'<<st.st_mtim.tv_sec<<':'<<st.st_mtim.tv_nsec<<'\n';
#endif
    }
    return out.str();
}
void ControlVectors::apply_reference(uint32_t layer,std::span<float> residual,bool enabled) const {
    if(residual.empty() || residual.size()%impl_->hidden) throw std::invalid_argument("control vector residual shape");
    if(!enabled || !covers(layer)) return;
    const auto d=direction(layer);const auto s=scale(layer);
    for(size_t row=0;row<residual.size();row+=impl_->hidden) {
        double dot=0;if(mode()==ControlVectorMode::Project) for(size_t column=0;column<d.size();++column) dot+=double(residual[row+column])*d[column];
        for(size_t column=0;column<d.size();++column) residual[row+column]=mode()==ControlVectorMode::Add?residual[row+column]+d[column]:float(double(residual[row+column])-double(s)*dot*d[column]);
    }
}
}
