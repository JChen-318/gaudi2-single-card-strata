#pragma once
#include <algorithm>
#include <cstdint>
#include <span>

namespace strata::hpu {
// One picture's conversation-cache identity (upstream's ConversationImageKey, generate.cpp:1499): the prompt
// cell it starts at and the FNV-1a key of its grid + embedding bytes (or the server's img_keys entry).
struct RequestImage {
    uint32_t start=0;
    uint64_t key=0;
    bool operator==(const RequestImage&) const=default;
};
// Upstream's imgs_below (generate.cpp:8414-8416): the pictures that start before `length`, in order. The
// request plan emits pictures in increasing `start`, so the filtered list is always a prefix of the list.
inline std::span<const RequestImage> images_below(std::span<const RequestImage> images,uint64_t length) {
    size_t count=0;
    while(count<images.size() && uint64_t(images[count].start)<length) ++count;
    return images.first(count);
}
// Upstream's starts_with picture half (:10856-10861): a stored prefix at `length` matches only when the
// request's pictures below that length are exactly the stored ones. An empty request list therefore never
// matches a state that contains pictures, and pictures after the prefix do not matter.
inline bool images_match(std::span<const RequestImage> stored,std::span<const RequestImage> request,uint64_t length) {
    const auto below=images_below(request,length);
    return stored.size()==below.size() && std::equal(stored.begin(),stored.end(),below.begin());
}
}
