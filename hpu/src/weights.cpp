#include "strata/hpu/weights.hpp"
#include "strata/hpu/families.hpp"
// The pinned upstream shard resolver parses exactly five decimal digits, but
// GCC's unoptimized format checker loses that bound at snprintf("%05d"). Keep
// upstream bytes unchanged and scope this diagnostic exception to its header.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
#endif
#include "strata/artifact/gguf_reader.hpp"
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
#include "strata/artifact/dequant.hpp"
#include "weight_format.hpp"
#include "mapped_file.hpp"
#include <ggml.h>
#include <array>
#include <bit>
#include <cstring>
#include <filesystem>
#include <sys/stat.h>

namespace strata::hpu {
using detail::format;using detail::convertible;
static_assert(std::endian::native == std::endian::little, "GGUF mapping requires little-endian host");
namespace detail {
struct MappedWeights {
    std::unique_ptr<strata::GgufFile> single;
    std::shared_ptr<const strata::GgufModel> model;
    const strata::GgufFile* view;
    explicit MappedWeights(const std::string& path):single(std::make_unique<strata::GgufFile>(path)),view(single.get()) {}
    MappedWeights(std::shared_ptr<const strata::GgufModel> m,uint32_t shard):model(std::move(m)),view(&model->shard(shard)) {}
    const strata::GgufFile& file() const {return *view;}
};
}
namespace {
// Q2_0's header-only decoder has a different signature than ggml's row API.
uint64_t payload(const strata::GgufFile& file, const strata::TensorInfo& tensor) {
    if (tensor.shape.empty() || tensor.shape.size() > 4) throw std::invalid_argument("unsupported GGUF rank");
    const auto f = format(tensor.type);
    if (!f.elements)
        throw std::invalid_argument("unknown GGUF type: " + tensor.name);
    uint64_t size = f.bytes;
    for (size_t axis = 0; axis < tensor.shape.size(); ++axis) {
        const auto dim = tensor.shape[axis];
        if (!dim || (axis == 0 && dim % f.elements))
            throw std::invalid_argument("invalid GGUF quant row: " + tensor.name);
        size = checked_product(size, axis ? dim : dim / f.elements);
    }
    check_range(file.file_size(), file.data_start(), 0);
    check_range(file.file_size() - file.data_start(), tensor.offset, size);
    return size;
}
}
WeightSource::WeightSource(const std::string& path) : owner_(std::make_shared<detail::MappedWeights>(path)) {}
std::vector<WeightInfo> WeightSource::inventory() const {
    std::vector<WeightInfo> result;
    for (const auto& tensor : owner_->file().tensors())
        result.push_back({tensor.name, tensor.type, tensor.shape, payload(owner_->file(), tensor),
                          convertible(tensor.type)});
    return result;
}
WeightSlice WeightSource::slice(const std::string& name, uint64_t expert_id, unsigned rank) const {
    const auto* tensor = owner_->file().find(name);
    if (!tensor) throw std::invalid_argument("GGUF tensor not found: " + name);
    const auto size = payload(owner_->file(), *tensor);
    const auto f = format(tensor->type);
    if (!convertible(tensor->type))
        throw std::invalid_argument("GGUF conversion not supported: " + name + " (ggml type " + std::to_string(tensor->type) + ")");
    if (tensor->shape.size() != rank) throw std::invalid_argument("unexpected GGUF tensor rank");
    if (rank == 3 && expert_id >= tensor->shape[2]) throw std::out_of_range("GGUF expert ID");
    const auto row_bytes = checked_product(tensor->shape[0] / f.elements, f.bytes);
    const auto rows = rank == 1 ? 1 : tensor->shape[1];
    const auto matrix_bytes = checked_product(row_bytes, rows);
    const auto offset = checked_product(expert_id, matrix_bytes);
    check_range(size, offset, matrix_bytes);
    WeightSlice result;
    result.owner_ = owner_;
    result.source_ = {owner_->file().tensor_data(*tensor) + offset, size_t(matrix_bytes)};
    result.columns_ = tensor->shape[0];
    result.rows_ = rows;
    result.row_bytes_ = row_bytes;
    result.type_ = tensor->type;
    return result;
}
WeightSlice WeightSource::matrix(const std::string& name) const { return slice(name, 0, 2); }
WeightSlice WeightSource::vector(const std::string& name) const { return slice(name, 0, 1); }
WeightSlice WeightSource::expert(const std::string& name, uint64_t id) const { return slice(name, id, 3); }
struct WeightTable::Impl {
    core::ModelGeometry geometry;
    std::vector<WeightSource> sources;
    struct Record {WeightInfo info;uint32_t shard;};
    std::map<std::string,Record> records;
    const Record& get(const std::string& name) const {
        const auto found=records.find(name);
        if(found==records.end()) throw std::invalid_argument("model tensor not found: "+name);
        return found->second;
    }
};
WeightTable::WeightTable(const std::string& any):WeightTable(strata::gguf_split_paths(any)) {}
WeightTable::WeightTable(const std::vector<std::string>& paths) {
    if(paths.empty() || paths.size()>100000) throw std::invalid_argument("invalid GGUF shard list");
    const auto model=std::make_shared<strata::GgufModel>(paths);
    auto p=std::make_shared<Impl>();
    for(uint32_t shard=0;shard<paths.size();++shard) {
        const auto& f=model->shard(shard);
        if(paths.size()>1 && !f.get("split.tensors.count")) throw std::invalid_argument("split GGUF requires split.tensors.count in every shard");
        for(auto name:{"split.count","split.no","split.tensors.count"}) if(const auto* v=f.get(name))
            if(!v->is_num() || v->type==strata::MetaType::F32 || v->type==strata::MetaType::F64 || v->type==strata::MetaType::BOOL || v->u>uint64_t(INT64_MAX))
                throw std::invalid_argument(std::string("GGUF split key must be a nonnegative integer: ")+name);
        if(paths.size()==1) {
            if(const auto* no=f.get("split.no");no && no->u!=0) throw std::invalid_argument("single GGUF declares nonzero split.no");
            if(const auto* count=f.get("split.count");count && count->u!=1) throw std::invalid_argument("single GGUF split.count mismatch");
            if(const auto* total=f.get("split.tensors.count");total && total->u!=f.tensors().size()) throw std::invalid_argument("single GGUF tensor total mismatch");
        }
        p->sources.push_back(WeightSource(std::make_shared<detail::MappedWeights>(model,shard)));
        for(auto info:p->sources.back().inventory()) {
            const auto name=info.name;
            if(!p->records.emplace(name,Impl::Record{std::move(info),shard}).second) throw std::invalid_argument("duplicate model tensor: "+name);
        }
    }
    const auto& f=model->meta();
    const auto* arch=f.get("general.architecture");
    // The accepted set and `--families` come from one place (strata/hpu/families.hpp), so the list this tool prints
    // can never claim a family the loader would reject. An absent or non-string key keeps upstream's own wording
    // distance: it is reported as the empty architecture, which no family matches.
    require_supported_family(arch && arch->type==strata::MetaType::STRING ? arch->s : std::string());
    // Bound integer dimensions before the upstream geometry parser forms derived
    // products. The parser then validates required keys and cross-key contracts.
    for(auto suffix:{"block_count","embedding_length","expert_count","expert_used_count","attention.head_count","attention.head_count_kv",
        "full_attention_interval","ssm.state_size","ssm.group_count","ssm.time_step_rank","ssm.conv_kernel","ssm.inner_size",
        "attention.key_length","attention.value_length","rope.dimension_count","attention.indexer.head_count","attention.indexer.key_length",
        "attention.indexer.top_k","hyper_connection.count","hyper_connection.low_rank","expert_feed_forward_length","expert_shared_feed_forward_length"}) {
        const std::string key=std::string("qwen4exp.")+suffix;const auto* value=f.get(key);
        if(value && value->is_num() && value->type!=strata::MetaType::F32 && value->type!=strata::MetaType::F64 &&
           value->type!=strata::MetaType::BOOL && value->u>1048576) throw std::invalid_argument("model dimension outside HPU target bounds: "+key);
    }
    auto error=strata::read_geometry(f,p->geometry);
    if(error.empty()) error=strata::geometry_unsupported(p->geometry);
    if(!error.empty() || !std::isfinite(p->geometry.rms_eps)) throw std::invalid_argument("HPU model geometry: "+error);
    impl_=std::move(p);
}
const core::ModelGeometry& WeightTable::geometry() const {return impl_->geometry;}
uint32_t WeightTable::shard_count() const {return uint32_t(impl_->sources.size());}
std::vector<WeightInfo> WeightTable::inventory() const {
    std::vector<WeightInfo> result;result.reserve(impl_->records.size());
    for(const auto& [name,record]:impl_->records) {(void)name;result.push_back(record.info);}
    return result;
}
const WeightInfo& WeightTable::info(const std::string& name) const {return impl_->get(name).info;}
uint32_t WeightTable::shard(const std::string& name) const {return impl_->get(name).shard;}
std::string WeightTable::tensor_path(const std::string& name) const {return impl_->sources[shard(name)].owner_->file().path();}
std::optional<std::vector<uint64_t>> WeightTable::integers(const std::string& key) const {
    const auto* value=impl_->sources.front().owner_->file().get(key);
    if(!value) return std::nullopt;
    auto integer=[&](const strata::MetaValue& v) {
        if(!v.is_num() || v.type==strata::MetaType::BOOL || v.type==strata::MetaType::F32 || v.type==strata::MetaType::F64 ||
           ((v.type==strata::MetaType::I8 || v.type==strata::MetaType::I16 || v.type==strata::MetaType::I32 || v.type==strata::MetaType::I64) && v.u>uint64_t(INT64_MAX)))
            throw std::invalid_argument("expected nonnegative integer metadata: "+key);
        return v.u;
    };
    std::vector<uint64_t> result;
    if(value->type==strata::MetaType::ARRAY) {
        if(value->count>64 || value->count!=value->items.size()) throw std::invalid_argument("metadata array exceeds retained bound: "+key);
        for(const auto& item:value->items) result.push_back(integer(item));
    } else result.push_back(integer(*value));
    return result;
}
WeightSlice WeightTable::matrix(const std::string& name) const {return impl_->sources[shard(name)].matrix(name);}
std::optional<double> WeightTable::number(const std::string& key) const {
    const auto* value=impl_->sources.front().owner_->file().get(key);if(!value) return std::nullopt;
    if(!value->is_num() || value->type==strata::MetaType::BOOL || !std::isfinite(value->num())) throw std::invalid_argument("expected finite numeric metadata: "+key);
    if(value->type!=strata::MetaType::F32 && value->type!=strata::MetaType::F64 && value->u>9007199254740992ull)
        throw std::invalid_argument("numeric metadata exceeds exact double integer range: "+key);
    return value->num();
}
std::optional<std::string> WeightTable::text(const std::string& key) const {
    const auto* value=impl_->sources.front().owner_->file().get(key);if(!value) return std::nullopt;
    if(value->type!=strata::MetaType::STRING) throw std::invalid_argument("expected string metadata: "+key);
    return value->s;
}
std::string WeightTable::file_identity() const {
    std::string identity="gguf-local-files-v1|";
    for(const auto& source:impl_->sources) identity+=detail::local_identity(source.owner_->file().path(),source.owner_->file().file_size());
    if(identity.size()>65536) throw std::invalid_argument("GGUF file identity exceeds state header bound");
    return identity;
}
WeightLocation WeightTable::location(const std::string& name) const {
    const auto& file=impl_->sources[shard(name)].owner_->file();const auto* tensor=file.find(name);
    if(!tensor) throw std::invalid_argument("model tensor not found: "+name);
    check_range(file.file_size(),file.data_start(),0);check_range(file.file_size()-file.data_start(),tensor->offset,info(name).source_bytes);
    return {std::filesystem::canonical(file.path()).string(),file.data_start()+tensor->offset,file.file_size()};
}
std::vector<std::pair<std::string,uint64_t>> WeightTable::shard_files() const {
    std::vector<std::pair<std::string,uint64_t>> files;files.reserve(impl_->sources.size());
    for(const auto& source:impl_->sources) files.emplace_back(std::filesystem::canonical(source.owner_->file().path()).string(),source.owner_->file().file_size());
    return files;
}
WeightSlice WeightTable::vector(const std::string& name) const {return impl_->sources[shard(name)].vector(name);}
WeightSlice WeightTable::expert(const std::string& name,uint64_t id) const {return impl_->sources[shard(name)].expert(name,id);}
std::vector<float> WeightTable::constant(const std::string& name,uint64_t elements,uint64_t maximum) const {
    const auto& dims=info(name).gguf_shape;
    if(!elements || elements>maximum || dims.empty() || dims.size()>2 || checked_product(dims[0],dims.size()==1?1:dims[1])!=elements)
        throw std::invalid_argument("constant shape/budget mismatch: "+name);
    const auto source=dims.size()==1?vector(name):matrix(name);std::vector<float> out(elements);source.convert_rows_f32(0,out);
    for(float x:out) if(!std::isfinite(x)) throw std::invalid_argument("non-finite model constant: "+name);
    return out;
}
void transpose_f32(const WeightSlice& source, std::span<float> output, uint64_t budget) {
    const auto row_bytes = checked_product(source.columns(),sizeof(float));
    if (!row_bytes || budget < row_bytes) throw std::invalid_argument("transpose scratch must hold one FP32 row");
    if (output.size() != checked_product(source.columns(),source.rows())) throw std::invalid_argument("transpose destination size");
    const auto chunk_rows = std::min(source.rows(),budget/row_bytes);
    std::vector<float> scratch(checked_product(chunk_rows,source.columns()));
    for (uint64_t row=0; row<source.rows();) {
        const auto count=std::min(chunk_rows,source.rows()-row);
        source.convert_rows_f32(row,std::span<float>(scratch).first(checked_product(count,source.columns())));
        for (uint64_t local=0;local<count;++local) for (uint64_t col=0;col<source.columns();++col)
            output[col*source.rows()+row+local]=scratch[local*source.columns()+col];
        row+=count;
    }
}
} // namespace strata::hpu
