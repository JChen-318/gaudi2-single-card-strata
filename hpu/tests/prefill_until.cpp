// --prefill-until decides where prompt batching stops. The pinned upstream expression is
// transcribed here independently, and the port must agree with it wherever the flag is set. The one
// divergence is asserted rather than hidden: an unset flag keeps batching the whole prompt in this
// backend, while upstream walks the final position through the token path.
#include "strata/hpu/generation.hpp"

#include <cstdint>
#include <cstdio>

namespace {
int failures=0;
void require(bool ok,const char* what,uint64_t prompt,uint64_t until) {
    if(!ok) {
        std::printf("FAIL %s at prompt=%llu prefill_until=%llu\n",what,
            (unsigned long long)prompt,(unsigned long long)until);
        ++failures;
    }
}
// Pinned upstream: src/program/generate.cpp:13165, with n_prompt taken as the prompt length.
uint64_t upstream(int64_t n_prompt,int64_t prefill_until) {
    return (prefill_until>0 && prefill_until<n_prompt-1)?uint64_t(prefill_until):uint64_t(n_prompt-1);
}
const uint64_t NO_LIMIT=999999999999ull;
}

int main() {
    for(uint64_t prompt=1;prompt<=80;++prompt) {
        for(uint64_t until=0;until<=prompt+2;++until) {
            const uint64_t actual=strata::hpu::prefill_batched_end(size_t(prompt),until);
            require(actual<=prompt,"batching ran past the end of the prompt",prompt,until);
            if(until) require(actual==upstream(int64_t(prompt),int64_t(until)),
                "a set flag did not follow the upstream law",prompt,until);
            require(strata::hpu::prefill_batched_end(size_t(prompt),until,true)==upstream(int64_t(prompt),int64_t(until)),
                "raw diagnostic token-loop suffix did not follow upstream for every limit",prompt,until);
        }
        // The recorded default divergence, checked on every length rather than in one example.
        require(strata::hpu::prefill_batched_end(size_t(prompt),0)==prompt,
            "an unset flag must batch the whole prompt here",prompt,0);
        require(upstream(int64_t(prompt),0)==prompt-1,
            "upstream trims the last position when the flag is unset",prompt,0);
        // A limit at or past the last position behaves like the default of the upstream law, and the
        // last position is never batched when the flag IS set.
        // prompt-1 is zero exactly when the prompt has one position, and zero is the "unset"
        // sentinel rather than a limit, so this assertion only applies above that length.
        if(prompt>1) require(strata::hpu::prefill_batched_end(size_t(prompt),prompt-1)==prompt-1,
            "a limit at the last position must stop just before it",prompt,prompt-1);
        require(strata::hpu::prefill_batched_end(size_t(prompt),prompt)==prompt-1,
            "a limit at the length must not extend batching",prompt,prompt);
        require(strata::hpu::prefill_batched_end(size_t(prompt),NO_LIMIT)==prompt-1,
            "an absurd limit must not extend batching",prompt,NO_LIMIT);
    }
    require(strata::hpu::prefill_batched_end(10,4)==4,"a real limit must batch exactly [0,4)",10,4);
    require(strata::hpu::prefill_batched_end(1,0)==1,"a one-token prompt batches its only position",1,0);
    require(strata::hpu::prefill_batched_end(1,1)==0,"a one-token prompt cannot batch under a limit",1,1);
    if(failures) {
        std::printf("prefill-until law: %d failure(s)\n",failures);
        return 1;
    }
    std::printf("PASS prefill-until law: 80 prompt lengths swept against the transcription of the "
        "pinned upstream expression at generate.cpp:13165 for every set value, the flag never "
        "raises the batched span past the prompt, the last position is never batched under a limit, "
        "and the single divergence (an unset flag batches the whole prompt here, upstream trims the "
        "last position) is asserted on every length rather than hidden; no device\n");
    return 0;
}
