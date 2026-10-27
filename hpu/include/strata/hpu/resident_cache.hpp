#pragma once
#include "strata/hpu/tensor.hpp"
#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <utility>

namespace strata::hpu {
struct CacheStats {
    uint64_t hits=0,loads=0,evictions=0,failed_loads=0,busy=0,resident_bytes=0,peak_bytes=0,entries=0;
};
// Payload-byte-limited LRU with immutable generations. A lease owns its entry,
// preventing eviction until the last consumer releases it. The loader returns
// unique ownership; resources cannot escape through an untracked shared_ptr.
// Loader calls are serialized and must not reenter this cache. Device consumers
// retain a lease through their completion event; host policy has no SDK calls.
template<class Resource> class ResidentCache {
    struct Entry {
        uint64_t bytes,generation,last_use;
        std::string partition;
        std::unique_ptr<const Resource> resource;
    };
public:
    class Lease {
    public:
        Lease()=default;
        const Resource& get() const {if(!entry_) throw std::logic_error("empty resident lease");return *entry_->resource;}
        const Resource* operator->() const {return &get();}
        explicit operator bool() const {return bool(entry_);}
        uint64_t generation() const {if(!entry_) throw std::logic_error("empty resident lease");return entry_->generation;}
        uint64_t bytes() const {if(!entry_) throw std::logic_error("empty resident lease");return entry_->bytes;}
    private:
        std::shared_ptr<const Entry> entry_;
        explicit Lease(std::shared_ptr<const Entry> e):entry_(std::move(e)) {}
        friend class ResidentCache;
    };
    explicit ResidentCache(uint64_t capacity):capacity_(capacity) {if(!capacity) throw std::invalid_argument("zero resident cache capacity");}
    Lease acquire(const std::string& key,uint64_t bytes,const std::function<std::unique_ptr<const Resource>()>& loader) {
        return acquire_in(std::string(),key,bytes,loader);
    }
    // ---- partitioned admission (--expert-cache-per-layer) ---------------------------------------------
    // A partition is a byte sub-budget of the whole cache: acquire_in() may only evict entries of its OWN
    // partition, and what it may use is the partition's capacity minus what that partition already holds. An
    // empty partition name means "the whole cache", so the unpartitioned call above is exactly the semantics
    // every caller had before this existed: with no partition installed the capacity is capacity_, the
    // resident figure is stats_.resident_bytes and the victim predicate is the same one. The declared
    // divergence from paged admission lives in strata/hpu/expert_residency.hpp, not here.
    void set_partition_capacity(const std::string& partition,uint64_t bytes) {
        if(partition.empty() || !bytes) throw std::invalid_argument("invalid resident partition capacity");
        std::lock_guard lock(mutex_);
        uint64_t total=bytes;
        for(const auto& [name,value]:partitions_) if(name!=partition) total+=value;
        if(total>capacity_) throw std::invalid_argument("resident partitions exceed the cache capacity");
        partitions_[partition]=bytes;
    }
    void clear_partitions() {std::lock_guard lock(mutex_);partitions_.clear();partition_bytes_.clear();}
    std::map<std::string,uint64_t> partitions() const {std::lock_guard lock(mutex_);return partitions_;}
    uint64_t partition_capacity(const std::string& partition) const {
        std::lock_guard lock(mutex_);
        if(partition.empty()) return capacity_;
        const auto it=partitions_.find(partition);return it==partitions_.end()?0:it->second;
    }
    uint64_t partition_resident(const std::string& partition) const {
        std::lock_guard lock(mutex_);return partition_resident_locked(partition);
    }
    Lease acquire_in(const std::string& partition,const std::string& key,uint64_t bytes,const std::function<std::unique_ptr<const Resource>()>& loader) {
        std::lock_guard lock(mutex_);
        const uint64_t capacity=partition_capacity_locked(partition);
        if(key.empty() || !bytes || bytes>capacity || !loader) throw std::invalid_argument("invalid resident load/key/size");
        if(clock_==UINT64_MAX || generation_==UINT64_MAX) throw std::overflow_error("resident generation clock exhausted");
        if(auto it=entries_.find(key);it!=entries_.end()) {
            if(it->second->bytes!=bytes) throw std::invalid_argument("resident key reused for a different size");
            if(it->second->partition!=partition) throw std::invalid_argument("resident key moved between partitions");
            it->second->last_use=++clock_;++stats_.hits;return Lease(it->second);
        }
        // Select all victims before changing state. Capacity failure does not
        // evict innocent entries or invoke the loader while consumers hold them.
        std::vector<std::pair<std::string,uint64_t>> candidates;
        for(const auto& [name,entry]:entries_) if(entry.use_count()==1 && !pinned_.count(name) &&
            (partition.empty() || entry->partition==partition)) candidates.emplace_back(name,entry->last_use);
        std::sort(candidates.begin(),candidates.end(),[](const auto& a,const auto& b){return a.second<b.second;});
        uint64_t available=capacity-partition_resident_locked(partition);size_t count=0;
        while(available<bytes && count<candidates.size()) available+=entries_.at(candidates[count++].first)->bytes;
        if(available<bytes) {++stats_.busy;throw std::runtime_error("resident cache capacity is leased by active consumers");}
        for(size_t i=0;i<count;++i) {
            const auto it=entries_.find(candidates[i].first);release_locked(it->second->partition,it->second->bytes);
            stats_.resident_bytes-=it->second->bytes;entries_.erase(it);++stats_.evictions;
        }
        stats_.entries=entries_.size();
        try {
            auto resource=loader();if(!resource) throw std::runtime_error("resident loader returned null");
            auto entry=std::make_shared<Entry>(Entry{bytes,++generation_,++clock_,partition,std::move(resource)});
            entries_.emplace(key,entry);stats_.resident_bytes+=bytes;stats_.peak_bytes=std::max(stats_.peak_bytes,stats_.resident_bytes);
            hold_locked(partition,bytes);
            ++stats_.loads;stats_.entries=entries_.size();return Lease(std::move(entry));
        } catch(...) {++stats_.failed_loads;throw;}
    }
    CacheStats stats() const {std::lock_guard lock(mutex_);return stats_;}
    // Read-only residency inspection: no LRU touch, no loader, no state change at all. The expert
    // profiler (strata/hpu/expert_profile.hpp) snapshots which (layer,expert) pairs are resident from
    // keys(), and a host test can therefore observe eviction without perturbing the order it checks.
    bool contains(const std::string& key) const {std::lock_guard lock(mutex_);return entries_.count(key)!=0;}
    std::vector<std::string> keys() const {
        std::lock_guard lock(mutex_);std::vector<std::string> out;out.reserve(entries_.size());
        for(const auto& [name,entry]:entries_) {(void)entry;out.push_back(name);}
        return out;
    }
    // ---- residency pins (the port's --resident-experts / --resident-cpu-experts page-lock) -----------------
    // A pinned key is skipped when a victim is chosen, so neither an acquire() that must evict nor a shrink
    // (resize) that must evict can drop it; the pin survives every eviction the LRU runs.  It is invisible to
    // the eviction ORDER of an all-off cache: with no pins the candidate predicate `!pinned_.count(name)` is
    // always true, so the LRU selects exactly the entries and the order it always did.  Pinning a key that is
    // not resident is allowed (it is admitted-before-routing and simply has no entry yet).
    void pin(const std::string& key) {std::lock_guard lock(mutex_);pinned_.insert(key);}
    void unpin(const std::string& key) {std::lock_guard lock(mutex_);pinned_.erase(key);}
    void clear_pins() {std::lock_guard lock(mutex_);pinned_.clear();}
    bool is_pinned(const std::string& key) const {std::lock_guard lock(mutex_);return pinned_.count(key)!=0;}
    uint64_t pinned_count() const {std::lock_guard lock(mutex_);return pinned_.size();}
    std::vector<std::string> pinned_keys() const {
        std::lock_guard lock(mutex_);return std::vector<std::string>(pinned_.begin(),pinned_.end());
    }
    // Read-only resident hit (no loader/I/O). Returning a lease pins the same
    // immutable generation through the consumer's device completion.
    Lease find(const std::string& key) {
        std::lock_guard lock(mutex_);const auto found=entries_.find(key);
        if(found==entries_.end()) return {};
        if(clock_==UINT64_MAX) throw std::overflow_error("resident LRU clock exhausted");
        found->second->last_use=++clock_;++stats_.hits;return Lease(found->second);
    }
    uint64_t capacity() const {std::lock_guard lock(mutex_);return capacity_;}
    // A resize invalidates the installed partition map: the caller re-installs it from the new capacity
    // (ExpertCache::resize does), because a stale sub-budget is what would let a layer overrun the cache.
    void resize(uint64_t capacity) {
        if(!capacity) throw std::invalid_argument("zero resident cache capacity");
        std::lock_guard lock(mutex_);
        std::vector<std::pair<std::string,uint64_t>> candidates;
        for(const auto& [name,entry]:entries_) if(entry.use_count()==1 && !pinned_.count(name)) candidates.emplace_back(name,entry->last_use);
        std::sort(candidates.begin(),candidates.end(),[](const auto& a,const auto& b){return a.second<b.second;});
        uint64_t remaining=stats_.resident_bytes;size_t count=0;
        while(remaining>capacity && count<candidates.size()) remaining-=entries_.at(candidates[count++].first)->bytes;
        if(remaining>capacity) {++stats_.busy;throw std::runtime_error("cannot shrink resident cache while required entries are leased");}
        for(size_t i=0;i<count;++i) {entries_.erase(candidates[i].first);++stats_.evictions;}
        stats_.resident_bytes=remaining;stats_.entries=entries_.size();capacity_=capacity;
        partitions_.clear();partition_bytes_.clear();
    }
    void clear() {
        std::lock_guard lock(mutex_);
        for(const auto& [name,entry]:entries_) {(void)name;if(entry.use_count()!=1) throw std::logic_error("cannot clear leased cache");}
        entries_.clear();stats_.resident_bytes=0;stats_.entries=0;partition_bytes_.clear();
    }
private:
    uint64_t capacity_;
    std::map<std::string,uint64_t> partitions_;      ///< partition -> its byte sub-budget (empty map: one pool)
    std::map<std::string,uint64_t> partition_bytes_; ///< partition -> what it currently holds
    uint64_t partition_capacity_locked(const std::string& partition) const {
        if(partition.empty()) return capacity_;
        const auto it=partitions_.find(partition);
        if(it==partitions_.end()) throw std::invalid_argument("unknown resident partition: "+partition);
        return it->second;
    }
    uint64_t partition_resident_locked(const std::string& partition) const {
        if(partition.empty()) return stats_.resident_bytes;
        const auto it=partition_bytes_.find(partition);return it==partition_bytes_.end()?0:it->second;
    }
    void hold_locked(const std::string& partition,uint64_t bytes) {
        if(!partition.empty()) partition_bytes_[partition]+=bytes;
    }
    void release_locked(const std::string& partition,uint64_t bytes) {
        if(partition.empty()) return;
        const auto it=partition_bytes_.find(partition);
        if(it==partition_bytes_.end() || it->second<bytes) throw std::logic_error("resident partition accounting underflow");
        it->second-=bytes;
    }
    mutable std::mutex mutex_;
    uint64_t clock_=0,generation_=0;
    std::map<std::string,std::shared_ptr<Entry>> entries_;
    std::set<std::string> pinned_;
    CacheStats stats_;
};
}
