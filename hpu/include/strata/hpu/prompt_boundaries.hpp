#pragma once
#include <cstdint>
#include <span>
namespace strata::hpu {
struct PromptBoundaryOptions {
    int32_t turn_token=248045,tail_role_token=-1;
    uint32_t root_min_tokens=2048;
    bool message_checkpoint=false;
};
struct PromptBoundaries {uint32_t root=0,turn=0,message=0;};
// Lengths exclude the opening marker of the next turn, matching upstream's
// read-to-index checkpoint contract. ckpt=0 still permits a shared root.
// `resumed_from0` is upstream's marker (src/program/generate.cpp:11044, gate :11727): a prompt whose read was
// interrupted and re-issued continues the SAME read that started at token 0, so the shared root is still looked
// for even though tokens are already fed. A resume from a parked or adopted conversation is not such a read and
// must pass false, or every continuation would re-elect a root the chain already carries.
PromptBoundaries prompt_boundaries(std::span<const int32_t>,uint32_t resume,const PromptBoundaryOptions&,bool checkpoint,
                                  bool resumed_from0=false);
}
