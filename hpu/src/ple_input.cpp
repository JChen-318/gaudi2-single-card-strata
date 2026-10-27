#include "strata/hpu/ple_input.hpp"
#include "strata/hpu/ple_io_mode.hpp"
#include <algorithm>
#include <cmath>

namespace strata::hpu {
void PleHashConfig::validate() const {
    if(!vocabulary || vocabulary>INT32_MAX || !table_rows || table_rows>uint64_t(UINT32_MAX)+1)
        throw std::invalid_argument("PLE token/table geometry out of bounds");
    for(auto m:constants.mult) if(!m) throw std::invalid_argument("zero PLE hash multiplier");
    uint64_t end=0;
    for(size_t i=0;i<16;++i) {
        if(!constants.vocab[i] || constants.offset[i]!=end || constants.vocab[i]>table_rows-end)
            throw std::invalid_argument("invalid PLE head vocab/offset partition");
        end+=constants.vocab[i];
    }
    // The published table has 90 trailing alignment rows beyond the sum of
    // head vocabularies. Hashes need in-bounds disjoint ranges, not equality
    // between that sum and the storage's padded row count.
    if(end>table_rows) throw std::invalid_argument("PLE head partition exceeds table");
}
PleHashConfig ple_hash_config(const WeightTable& table,uint64_t table_rows) {
    auto required=[&](const std::string& suffix,size_t size) {
        auto values=table.integers("qwen4exp."+suffix);
        if(!values || values->size()!=size) throw std::invalid_argument("missing/wrong PLE metadata: "+suffix);
        return std::move(*values);
    };
    auto scalar=[&](const std::string& suffix,uint64_t expected) {
        if(required(suffix,1)[0]!=expected) throw std::invalid_argument("unsupported PLE metadata: "+suffix);
    };
    scalar("ple.layers",1);scalar("ple.ngram_size",3);scalar("ple.heads_per_ngram",8);
    scalar("ple.conv_kernel",4);scalar("ple.eos_token_id",strata::kernels::PLE_EOS_TOKEN_ID);
    scalar("embedding_length_per_layer_input",160);
    if(!table_rows) {
        const auto& dims=table.info("per_layer_token_embd.weight").gguf_shape;
        if(dims.size()!=2 || dims[0]!=160) throw std::invalid_argument("PLE table shape must be [160,rows]");
        table_rows=dims[1];
    }
    PleHashConfig result;result.table_rows=table_rows;
    const auto& embedding=table.info("token_embd.weight").gguf_shape;
    if(embedding.size()!=2 || embedding[0]!=2560 || embedding[1]>INT32_MAX) throw std::invalid_argument("PLE token embedding geometry");
    result.vocabulary=uint32_t(embedding[1]);
    const auto mult=required("ple.layer_multipliers",3),vocab=required("ple.head_vocab_sizes",16),offset=required("ple.head_offsets",16);
    std::copy(mult.begin(),mult.end(),result.constants.mult);std::copy(vocab.begin(),vocab.end(),result.constants.vocab);std::copy(offset.begin(),offset.end(),result.constants.offset);
    result.validate();return result;
}
std::array<uint32_t,16> ple_rows(int32_t token,std::span<const int32_t> previous,const PleHashConfig& config) {
    config.validate();
    if(token<0 || uint32_t(token)>=config.vocabulary) throw std::invalid_argument("PLE token ID out of vocabulary");
    std::array<int32_t,2> prev{-1,-1};
    const auto count=std::min<size_t>(2,previous.size());
    for(size_t i=0;i<count;++i) {
        const auto id=previous[previous.size()-count+i];
        if(id<0 || uint32_t(id)>=config.vocabulary) throw std::invalid_argument("invalid committed PLE predecessor");
        prev[2-count+i]=id;
    }
    std::array<uint32_t,16> result{};strata::kernels::ngram_rows(&token,prev.data(),1,config.constants,result.data());
    for(auto row:result) if(row>=config.table_rows) throw std::runtime_error("PLE hash row exceeds table");
    return result;
}
std::vector<uint32_t> ple_batch_rows(std::span<const int32_t> tokens,std::span<const int32_t> previous,const PleHashConfig& config) {
    config.validate();
    std::array<int32_t,2> prev{-1,-1};
    const auto count=std::min<size_t>(2,previous.size());
    for(size_t i=0;i<count;++i) {
        const auto id=previous[previous.size()-count+i];
        if(id<0 || uint32_t(id)>=config.vocabulary) throw std::invalid_argument("invalid committed PLE predecessor");
        prev[2-count+i]=id;
    }
    // Token r names the n-gram with its two predecessors: the committed prefix's last two, then
    // the batch's own earlier tokens - the same IDs ple_rows reads from the full history.
    constexpr size_t kHeads=strata::kernels::PLE_N_HEADS;
    std::vector<uint32_t> rows(tokens.size()*kHeads);
    for(size_t r=0;r<tokens.size();++r) {
        auto token=tokens[r];
        if(token<0 || uint32_t(token)>=config.vocabulary) throw std::invalid_argument("PLE token ID out of vocabulary");
        strata::kernels::ngram_rows(&token,prev.data(),1,config.constants,rows.data()+r*kHeads);
        for(size_t h=0;h<kHeads;++h) if(rows[r*kHeads+h]>=config.table_rows) throw std::runtime_error("PLE hash row exceeds table");
        prev[0]=prev[1];prev[1]=token;
    }
    return rows;
}
void open_ple_source(strata::kernels::PleTable& reader,const PleSourceInfo& source,
                     const strata::kernels::PleIoOptions& io,uint64_t ram_reservation,uint64_t reader_reservation,double delay_us) {
    validate_ple_io_policy(io);
    (void)resolve_ple_delay_us(delay_us);
    if(reader.is_open()) throw std::invalid_argument("PLE reader must be closed before source open");
    const auto current=inspect_ple_source(source.path);
    if(current.identity!=source.identity || current.rows!=source.rows ||
       current.payload_offset!=source.payload_offset || current.payload_bytes!=source.payload_bytes)
        throw std::runtime_error("PLE source metadata changed before reader open");
    const auto memory=ple_ram_memory(source,io);
    const auto direct=ple_reader_memory(source,io);
    if(direct.host_bytes>reader_reservation) throw std::invalid_argument("PLE reader exceeds its host memory reservation");
    if(memory.host_bytes>ram_reservation)
        throw std::invalid_argument("PLE RAM reader exceeds its full-table RAM reservation");
    if(ple_source_identity(source.path)!=source.identity) throw std::runtime_error("PLE table changed before reader open");
    std::string error;
    if(!reader.open(source.path,error,io)) throw std::runtime_error("PLE reader: "+error);
    reader.set_injected_delay_us(delay_us);
    try {
        if(reader.rows()!=source.rows) throw std::runtime_error("PLE reader/catalog row mismatch");
        if(ple_source_identity(source.path)!=source.identity) throw std::runtime_error("PLE table changed during reader open");
    } catch(...) {reader.close();throw;}
}
PleInput::PleInput(const WeightTable& table,strata::kernels::PleIoOptions io,const std::string& external_table,uint64_t ram_reservation,uint64_t reader_reservation,double delay_us) {
    const auto source=inspect_ple_source(external_table.empty()?table.tensor_path("per_layer_token_embd.weight"):external_table);
    config_=ple_hash_config(table,source.rows);
    open_ple_source(table_,source,io,ram_reservation,reader_reservation,delay_us);
    ram_requested_=io.lock;
}
std::vector<float> PleInput::gather(int32_t token,std::span<const int32_t> previous) {
    const auto rows=ple_rows(token,previous,config_);std::vector<float> out(2560);std::string error;
    if(!table_.issue(rows.data()) || !table_.collect(out.data(),error)) throw std::runtime_error("PLE gather: "+error);
    for(float value:out) if(!std::isfinite(value)) throw std::runtime_error("non-finite PLE table value/scale");
    return out;
}
void PleInput::gather_rows(std::span<const int32_t> tokens,std::span<const int32_t> previous,float* out,uint64_t width) {
    constexpr uint64_t kRow=strata::kernels::PLE_N_HEADS*strata::kernels::PLE_HEAD_DIM;
    if(width!=kRow) throw std::invalid_argument("PLE batch row width");
    if(tokens.empty()) return;
    const auto rows=ple_batch_rows(tokens,previous,config_);
    std::string error;
    // A window's rows decode on sixteen threads (the same values as one thread).
    if(!table_.gather_batch_threads(rows.data(),tokens.size(),out,error,16)) throw std::runtime_error("PLE gather: "+error);
    for(size_t i=0;i<tokens.size()*kRow;++i) if(!std::isfinite(out[i])) throw std::runtime_error("non-finite PLE table value/scale");
}
std::string PleInput::io_report() const {
    if(table_.mode()==strata::kernels::PleIo::Mmap) return std::string(ram_requested_?"ple_io=ram ple_ram_locked=":"ple_io=mmap ple_ram_locked=")+(ram_locked()?"1":"0");
    return table_.io_report();
}
bool PleInput::ram_locked() const {return reported_ple_ram_lock(table_.locked());}
}
