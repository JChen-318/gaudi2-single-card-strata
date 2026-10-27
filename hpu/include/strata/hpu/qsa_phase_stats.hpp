#pragma once
#include "strata/hpu/performance_modes.hpp"
#include <chrono>
#include <cstdio>
#include <map>
#include <mutex>
#include <string>

namespace strata::hpu {
// Diagnostic only; process-wide totals. Enable for attribution, never for reported throughput.
class QsaPhaseStats {
    struct Row {uint64_t seen=0,count=0;double us=0;};
    std::mutex mutex_;std::map<std::string,Row> phases_;
public:
    static bool enabled() {static const bool on=performance_env("STRATA_HPU_QSA_PHASE_STATS");return on;}
    static QsaPhaseStats& instance() {static QsaPhaseStats stats;return stats;}
    static uint64_t skip() {
        static const uint64_t count=[] {const char* value=std::getenv("STRATA_HPU_QSA_PHASE_SKIP");
            if(!value) return uint64_t(0);
            char* end=nullptr;const auto n=std::strtoull(value,&end,10);return end!=value && !*end && n<=1000000000?uint64_t(n):uint64_t(0);
        }();return count;
    }
    void add(const char* name,double us) {std::lock_guard lock(mutex_);auto& row=phases_[name];if(++row.seen>skip()) {++row.count;row.us+=us;}}
    // Reported between requests as well as from the destructor: the serve loop is normally stopped by a signal, so a
    // destructor-only report never reaches the log. Diagnostic only.
    void flush() {
        std::lock_guard lock(mutex_);
        if(phases_.empty()) return;
        std::fprintf(stderr,"QSA phase stats (host wall time; nested totals overlap):\n");
        for(const auto& [name,row]:phases_) if(row.count) std::fprintf(stderr,"  %s count=%llu total_us=%.3f mean_us=%.3f skipped=%llu\n",name.c_str(),
            static_cast<unsigned long long>(row.count),row.us,row.us/double(row.count),static_cast<unsigned long long>(row.seen-row.count));
        phases_.clear();
    }
    ~QsaPhaseStats() {flush();}
};
class QsaPhaseTimer {
    using Clock=std::chrono::steady_clock;
    bool on_=QsaPhaseStats::enabled();Clock::time_point started_{};
public:
    QsaPhaseTimer() {if(on_) started_=Clock::now();}
    void mark(const char* name) {
        if(!on_) return;
        const auto ended=Clock::now();QsaPhaseStats::instance().add(name,std::chrono::duration<double,std::micro>(ended-started_).count());started_=Clock::now();
    }
};
}
