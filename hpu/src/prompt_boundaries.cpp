#include "strata/hpu/prompt_boundaries.hpp"
#include "strata/program/message_boundary.hpp"
#include <stdexcept>
namespace strata::hpu {
PromptBoundaries prompt_boundaries(std::span<const int32_t> ids,uint32_t resume,const PromptBoundaryOptions& o,
                                   bool checkpoint,bool resumed_from0) {
    if(resume>ids.size() || ids.size()>INT32_MAX || o.turn_token< -1 || o.tail_role_token< -1 || o.root_min_tokens>INT32_MAX) throw std::invalid_argument("invalid prompt boundary options/resume");
    PromptBoundaries result;if(o.turn_token<0 || ids.empty()) return result;
    int64_t turn=-1;
    for(int64_t i=int64_t(ids.size())-1;i>resume;--i) if(ids[size_t(i)]==o.turn_token) {turn=i;break;}
    if(turn>0 && o.tail_role_token>=0) for(int64_t i=turn-1;i>resume;--i) if(ids[size_t(i)]==o.turn_token) {
        if(ids[size_t(i)+1]==o.tail_role_token) turn=i;
        break;
    }
    // Upstream's gate (:11727): `read_from == 0 || resumed_from0`.
    if((!resume || resumed_from0) && o.root_min_tokens) for(int64_t i=1;i<turn;++i) if(ids[size_t(i)]==o.turn_token) {
        if(i>=o.root_min_tokens) result.root=uint32_t(i);
        break; // only the FIRST non-initial turn boundary can be the root
    }
    if(checkpoint && turn>0) {
        result.turn=uint32_t(turn);
        if(o.message_checkpoint) {
            const std::vector<int64_t> wide(ids.begin(),ids.end());
            const auto message=program::message_checkpoint_boundary(wide,resume,turn,o.turn_token);
            if(message>0) result.message=uint32_t(message);
        }
    }
    return result;
}
}
