#pragma once
#include "strata/hpu/snapshot_memory.hpp"
#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

namespace strata::hpu {
// The diagnostic head input follows the same transaction as logits/residual.
// Heads only stage; the Model publishes after SessionFlow commits. Rejected
// windows may overwrite staging without changing the observable conversation.
class HeadMixed {
public:
    HeadMixed()=default;
    explicit HeadMixed(size_t width):current_(width),pending_(width) {
        if(!width) throw std::invalid_argument("empty mixed head width");
    }
    size_t width() const {return current_.size();}
    std::span<const float> current() const {return current_;}
    void stage(std::span<const float> rows,size_t count) {
        if(!width() || !count || count>rows.size()/width() || rows.size()!=count*width())
            throw std::invalid_argument("mixed head row shape mismatch");
        for(float x:rows) if(!std::isfinite(x)) throw std::invalid_argument("non-finite mixed head input");
        std::copy(rows.end()-width(),rows.end(),pending_.begin());
    }
    // Fixed-size buffers: no allocation or failure after token/layer commit.
    void commit() noexcept {std::copy(pending_.begin(),pending_.end(),current_.begin());}
    static void validate(std::span<const float> values,size_t width,bool populated) {
        if(!width || values.size()!=(populated?width:0)) throw std::invalid_argument("snapshot mixed head shape mismatch");
        for(float x:values) if(!std::isfinite(x)) throw std::invalid_argument("non-finite snapshot mixed head input");
    }
    void capture(std::vector<float>& image,bool populated) const {
        snapshot_assign(image,populated?current():std::span<const float>{});
    }
    void restore(std::span<const float> image,bool populated) {
        validate(image,width(),populated);
        if(populated) std::copy(image.begin(),image.end(),current_.begin());
        else std::fill(current_.begin(),current_.end(),0);
    }
    template<class Writer> static void write(Writer& writer,std::span<const float> image,size_t width,bool populated) {
        validate(image,width,populated);writer.template vector<float>(image);
    }
    template<class Reader> static std::vector<float> read(Reader& reader,size_t width,bool populated) {
        if(!width) throw std::invalid_argument("empty mixed head width");
        auto result=reader.template vector<float>(populated?width:0);
        validate(result,width,populated);return result;
    }
private:
    std::vector<float> current_,pending_;
};
}
