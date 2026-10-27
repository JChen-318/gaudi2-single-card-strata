#pragma once
#include "strata/hpu/routing.hpp"
#include "strata/hpu/dump_sinks.hpp"
#include "strata/hpu/session_flow.hpp"
#include <algorithm>
#include <cmath>
#include <exception>

namespace strata::hpu {
// One bounded physical target micro-window, layer-major then token-major.
// Frozen after its first complete layer pass so accepted-prefix replay cannot
// duplicate/replace the candidate trace. No device access and no per-call growth.
class RoutingCapture {
public:
    static uint64_t host_bytes(uint32_t layers,uint32_t rows,uint32_t experts,uint32_t top_k) {
        if(!layers || layers>4096 || !rows || rows>128 || !experts || experts>4096 ||
           !top_k || top_k>64 || top_k>experts) throw std::invalid_argument("routing capture geometry exceeds bounded envelope");
        const auto records=checked_product(layers,rows),cells=checked_product(records,top_k);
        // Persistent flat slabs + ONE caller export + record descriptors and
        // conservative allocator overhead. This is an envelope, not RSS.
        return checked_product(cells,16)+checked_product(records,sizeof(RoutingRecord)+64)+4096;
    }
    void configure(uint32_t layers,uint32_t rows,uint32_t experts,uint32_t top_k) {
        (void)host_bytes(layers,rows,experts,top_k);
        const auto cells=checked_product(checked_product(layers,rows),top_k);
        std::vector<int32_t> ids(size_t(cells),0);std::vector<float> weights(size_t(cells),0);
        if(ids.capacity()!=cells || weights.capacity()!=cells) throw std::length_error("routing capture capacity exceeds admitted envelope");
        ids_=std::move(ids);weights_=std::move(weights);layers_=layers;maximum_=rows;experts_=experts;top_k_=top_k;clear();
    }
    bool enabled() const {return layers_!=0;}
    void clear() {rows_=seen_=0;active_=ready_=false;}
    uint64_t storage_bytes() const {return ids_.capacity()*sizeof(int32_t)+weights_.capacity()*sizeof(float);}
    uint32_t rows() const {return ready_?rows_:0;}
    // Candidate selection may inspect the completed initial pass BEFORE
    // numeric commit. A thrown sink then participates in SessionFlow rollback.
    RoutingWindowView physical_view() const {
        if(!layers_ || seen_!=layers_) return {};
        return {layers_,rows_,top_k_,uint64_t(maximum_)*top_k_,ids_,weights_};
    }
    class Round {
    public:
        Round(RoutingCapture& owner,uint32_t rows):owner_(owner) {
            if(!owner.enabled()) return;
            if(!rows || rows>owner.maximum_) throw std::invalid_argument("routing capture window exceeds admitted rows");
            exceptions_=std::uncaught_exceptions();
            owner.clear();owner.rows_=rows;owner.active_=true;
        }
        Round(const Round&)=delete;
        // Model observers run after numeric commit/finish. A publication
        // failure still invalidates this ephemeral view, never the committed
        // numeric prefix. Compare with entry so an existing caller unwind or
        // an internally caught exception does not erase a successful round.
        ~Round() {if(!finished_ || (owner_.enabled() && std::uncaught_exceptions()>exceptions_)) owner_.clear();}
        void finish() {
            if(owner_.enabled() && owner_.seen_!=owner_.layers_) throw SessionFatal("routing capture did not observe every target layer");
            owner_.ready_=owner_.enabled();owner_.active_=false;finished_=true;
        }
    private:
        RoutingCapture& owner_;bool finished_=false;int exceptions_=0;
    };
    Round round(uint32_t rows) {return Round(*this,rows);}
    void capture(uint32_t layer,const Routes& routes) {
        if(!active_) return; // disabled or replay after the initial full pass
        const auto cells=uint64_t(rows_)*top_k_;
        if(layer!=seen_ || routes.tokens!=rows_ || routes.experts!=experts_ || routes.top_k!=top_k_ ||
           routes.ids.size()!=cells || routes.weights.size()!=cells) throw std::logic_error("routing capture producer shape/layer contract mismatch");
        for(size_t at=0;at<cells;++at) {
            if(routes.ids[at]<0 || uint32_t(routes.ids[at])>=experts_ || !std::isfinite(routes.weights[at]) || routes.weights[at]<0)
                throw std::invalid_argument("routing capture contains invalid expert/weight");
        }
        const auto offset=uint64_t(layer)*maximum_*top_k_;
        std::copy(routes.ids.begin(),routes.ids.end(),ids_.begin()+offset);
        std::copy(routes.weights.begin(),routes.weights.end(),weights_.begin()+offset);
        if(++seen_==layers_) active_=false;
    }
    // Synchronous borrowed records; callback may not retain spans or re-enter
    // the Model. Keeping this path allocation-free permits a future file sink.
    template<class Visitor> void visit(Visitor&& visitor) const {
        if(!ready_) return;
        for(uint32_t layer=0;layer<layers_;++layer) for(uint32_t row=0;row<rows_;++row) {
            const auto at=(uint64_t(layer)*maximum_+row)*top_k_;
            visitor(int32_t(layer),std::span<const int32_t>(ids_).subspan(at,top_k_),std::span<const float>(weights_).subspan(at,top_k_));
        }
    }
    std::vector<RoutingRecord> records() const {
        std::vector<RoutingRecord> result;if(!ready_) return result;
        const auto count=uint64_t(layers_)*rows_;result.reserve(size_t(count));
        if(result.capacity()!=count) throw std::length_error("routing record export exceeds admitted envelope");
        visit([&](int32_t layer,std::span<const int32_t> ids,std::span<const float> weights) {
            RoutingRecord record;record.layer=layer;record.ids.assign(ids.begin(),ids.end());record.weights.assign(weights.begin(),weights.end());
            if(record.ids.capacity()!=top_k_ || record.weights.capacity()!=top_k_) throw std::length_error("routing payload export exceeds admitted envelope");
            result.push_back(std::move(record));
        });
        return result;
    }
private:
    uint32_t layers_=0,maximum_=0,experts_=0,top_k_=0,rows_=0,seen_=0;
    bool active_=false,ready_=false;
    std::vector<int32_t> ids_;std::vector<float> weights_;
};
}
