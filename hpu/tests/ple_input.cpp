#include "strata/hpu/ple_input.hpp"
#include "../../src/kernels/ple_oracle_vectors.inc"
#include <algorithm>
#include <iostream>
#include <chrono>

using namespace strata::hpu;
namespace {
void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
template<class F> void rejects(F&& f) {bool caught=false;try {f();} catch(const std::exception&) {caught=true;}require(caught,"invalid PLE input accepted");}
}
int main(int argc,char** argv) {
    try {
        if(argc!=1 && argc!=3) return 2;
        if(argc==3) {
            const WeightTable main(argv[1]);
            for(auto mode:{strata::kernels::PleIo::Mmap,strata::kernels::PleIo::Direct}) for(double delay:{0.0,5000.0}) {
                strata::kernels::PleIoOptions io;io.mode=mode;io.cache_rows=16;io.max_inflight=4;
                const auto reserved=std::max(kPleReaderBaselineHostBytes,ple_reader_memory(inspect_ple_source(argv[2]),io).host_bytes);
                PleInput input(main,io,argv[2],0,reserved,delay);
                const std::array<int32_t,4> history{0,22,248044,111};
                bool first=true;
                for(auto token:{0,33,248044,248319}) {
                    const auto start=std::chrono::steady_clock::now();
                    const auto values=input.gather(token,history);
                    const auto elapsed=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
                    if(mode==strata::kernels::PleIo::Direct && delay>0 && first) require(elapsed>=delay,"PLE delay did not reach actual reader");
                    first=false;
                    require(values.size()==2560,"external PLE embedding width");
                    for(float value:values) require(value==0,"external PLE gather differs from sparse zero oracle");
                }
                // The prefill window's single-request batch reads the same rows through the reader.
                const std::array<int32_t,2> batch{0,248044};
                std::vector<float> batched(batch.size()*2560);
                input.gather_rows(batch,history,batched.data(),2560);
                for(float value:batched) require(value==0,"external PLE batch gather differs from sparse zero oracle");
                rejects([&]{std::vector<float> wrong(5120);input.gather_rows(batch,history,wrong.data(),1280);});
            }
        }
        const PleHashConfig config{strata::kernels::ple_artifact_consts(),248320,strata::kernels::PLE_TABLE_ROWS};config.validate();
        size_t cases=0;
        for(const auto& item:strata::kernels::ple_oracle::kHashCases) {
            std::vector<uint32_t> out(size_t(item.n_tokens)*16);
            strata::kernels::ngram_rows(item.tokens,item.prev,item.n_tokens,config.constants,out.data());
            require(std::equal(out.begin(),out.end(),item.rows),"frozen external hash oracle mismatch");++cases;
        }
        // Session predecessor ordering, no truncation of uint64 multipliers,
        // token zero, EOS context cuts, and arbitrary older history.
        for(const std::vector<int32_t>& history:{std::vector<int32_t>{}, {0}, {0,22}, {111,0,22}, {0,248044}, {248044,0}}) {
            const auto rows=ple_rows(33,history,config);
            std::array<int32_t,2> prev{-1,-1};
            if(!history.empty()) prev[1]=history.back();
            if(history.size()>1) prev[0]=history[history.size()-2];
            std::array<uint32_t,16> expected{};const int32_t token=33;
            strata::kernels::ngram_rows(&token,prev.data(),1,config.constants,expected.data());
            require(rows==expected,"committed predecessor order mismatch");
        }
        const std::vector<int32_t> short_history{0,22},long_history{111,0,22};
        require(ple_rows(33,short_history,config)==ple_rows(33,long_history,config),"older history affected ngram");
        // The prefill window's batch: every token's 16 rows must equal the per-token law with the
        // history up to that token, including the two committed predecessors that seed the batch.
        const std::vector<int32_t> batch_tokens{33,0,7,248044,22};
        for(const std::vector<int32_t>& history:{std::vector<int32_t>{}, {0}, {0,22}, {111,0,22}, {0,248044}}) {
            const auto rows=ple_batch_rows(batch_tokens,history,config);
            require(rows.size()==batch_tokens.size()*16,"batch row count");
            std::vector<int32_t> prefix=history;size_t at=0;
            for(const auto token:batch_tokens) {
                const auto one=ple_rows(token,prefix,config);
                require(std::equal(one.begin(),one.end(),rows.begin()+at*16),"batch ngram row differs from the per-token law");
                prefix.push_back(token);++at;
            }
        }
        rejects([&]{(void)ple_rows(-1,{},config);});rejects([&]{(void)ple_rows(248320,{},config);});
        const std::vector<int32_t> bad_batch{-1};
        rejects([&]{(void)ple_batch_rows(bad_batch,{},config);});
        const std::array<int32_t,1> bad{-1};rejects([&]{(void)ple_rows(0,bad,config);});
        auto invalid=config;invalid.constants.offset[3]++;rejects([&]{invalid.validate();});
        invalid=config;invalid.constants.vocab[0]=0;rejects([&]{invalid.validate();});
        invalid=config;invalid.table_rows=config.constants.offset[15]+config.constants.vocab[15]-1;rejects([&]{invalid.validate();});
        invalid=config;invalid.constants.mult[0]=0;rejects([&]{invalid.validate();});
        std::cout<<"PASS "<<cases<<" frozen independent hash vectors, chronological session history and PLE guards\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
