#pragma once
// Upstream keeps the live conversation's pictures beside its tokens and serializes both in one session
// image (src/core/conversation_file.cpp:683-684, :758-759); RESTORE then installs that list on the live
// conversation (src/program/generate.cpp:10615) so the next request still has to carry the same pictures
// below its length. The port's state file carries the list the same way: a count, then {start,key} pairs.
#include "strata/hpu/image_key.hpp"
#include "strata/hpu/state_io.hpp"
#include <stdexcept>
#include <vector>

namespace strata::hpu {
// The most payload a picture list over `position` saved cells can add: one picture per cell, two 64-bit
// fields each, plus the count.
inline uint64_t state_image_bytes(uint64_t position) {return position*2*sizeof(uint64_t)+sizeof(uint64_t);}

// Refusals happen before a byte is written, so a save with a malformed list publishes nothing.
inline void write_image_identity(StateWriter& writer,std::span<const RequestImage> images,uint64_t position) {
    if(images.size()>position) throw std::invalid_argument("state image identity exceeds the saved position");
    uint32_t previous=0;
    for(size_t i=0;i<images.size();++i) {
        if(uint64_t(images[i].start)>=position) throw std::invalid_argument("state image start exceeds the saved position");
        if(i && images[i].start<=previous) throw std::invalid_argument("state image starts are not strictly increasing");
        previous=images[i].start;
    }
    writer.integer(images.size());
    for(const auto& image:images) {writer.integer(image.start);writer.integer(image.key);}
}

// The count is bounded by the saved position before anything is allocated; each pair is checked against
// the token prefix it describes, exactly the law Generation::adopt applies in memory.
inline std::vector<RequestImage> read_image_identity(StateReader& reader,uint64_t position) {
    const auto count=reader.integer();
    if(count>position) throw std::invalid_argument("serialized image identity exceeds the saved position");
    std::vector<RequestImage> images;
    images.reserve(count);
    uint32_t previous=0;
    for(uint64_t i=0;i<count;++i) {
        const auto start=reader.integer(),key=reader.integer();
        if(start>=position) throw std::invalid_argument("serialized image start exceeds the saved position");
        if(i && start<=previous) throw std::invalid_argument("serialized image starts are not strictly increasing");
        previous=uint32_t(start);
        images.push_back(RequestImage{uint32_t(start),key});
    }
    return images;
}
}
