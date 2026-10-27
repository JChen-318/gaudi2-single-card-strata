// Host semantic test for the STRATA_HPU_MOE_GROUP scalar pre-pass (tpc/route_group_starts.c and
// tpc/route_group_order.c), executed from the original TPC sources under scalar/vector shims. It models the
// index spaces glue.cpp declares - starts has experts+1 members (the last one closes the final run), order
// has experts members - and checks the contract the grouped GEMV relies on:
//   starts[e] = #{slot : ids[slot] < e}, so the run of e is [starts[e], starts[e+1]);
//   order is a permutation of the slots whose runs are one routed id each, in slot order, with every
//   shared slot (id 0, written by route_topk) inside run 0.
// A naive independent writer and structural invariants (permutation, monotonic prefix, run membership,
// ascending slots) are both checked, so a shared bug has to survive both. Host only: no ISA execution, no
// device, no performance claim.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "strata/hpu/model_abi.hpp"

namespace group_emulator {
using int5 = std::array<int, 5>;
int5 offset{}, extent{};
int5 get_index_space_offset() { return offset; }
int5 get_index_space_size() { return extent; }
int5 operator+(int5 a, const int5& b) { for (size_t i = 0; i < 5; ++i) a[i] += b[i]; return a; }
struct tensor { void* data; std::vector<uint64_t> dims; uint32_t element; };
bool inside(const int5& p, const tensor& t) {
    for (size_t i = 0; i < 5; ++i) {
        const uint64_t size = i < t.dims.size() ? t.dims[i] : 1;
        if (p[i] < 0 || uint64_t(p[i]) >= size) return false;
    }
    return true;
}
uint64_t index_of(const int5& p, const tensor& t) {
    uint64_t index = 0, stride = 1;
    for (size_t i = 0; i < 5; ++i) {
        const uint64_t size = i < t.dims.size() ? t.dims[i] : 1;
        index += uint64_t(p[i]) * stride;
        stride *= size;
    }
    return index;
}
void* gen_addr(const int5& p, const tensor& t) {
    if (!inside(p, t)) throw std::out_of_range("group TPC scalar access outside tensor");
    return static_cast<uint8_t*>(t.data) + index_of(p, t) * t.element;
}
struct float64 {
    std::array<float, 64> lanes{};
    float64(float x = 0) { lanes.fill(x); }
};
float64 operator+(float64 a, const float64& b) { for (unsigned i = 0; i < 64; ++i) a.lanes[i] += b.lanes[i]; return a; }
struct int64 { std::array<int, 64> lanes{}; };
float64 v_f32_sel_less_f32_b(const float64& a, const float64& b, const float64& yes, const float64& no) {
    float64 out;
    for (unsigned i = 0; i < 64; ++i) out.lanes[i] = a.lanes[i] < b.lanes[i] ? yes.lanes[i] : no.lanes[i];
    return out;
}
float64 v_f32_sel_eq_f32_b(const float64& a,const float64& b,const float64& yes,const float64& no) {
    float64 out;
    for(unsigned i=0;i<64;++i) out.lanes[i]=a.lanes[i]==b.lanes[i]?yes.lanes[i]:no.lanes[i];
    return out;
}
// The TPC reduction broadcast its result; the fixture's counts are integers far inside F32's exact range.
float64 v_f32_reduce_add(const float64& v) { float sum = 0; for (float x : v.lanes) sum += x; return float64(sum); }
int64 v_i32_ld_tnsr_b(int5 p, const tensor& t) {
    int64 out;
    for (unsigned lane = 0; lane < 64; ++lane) {
        auto q = p; q[0] += int(lane);
        if (inside(q, t)) out.lanes[lane] = *static_cast<int*>(gen_addr(q, t));
    }
    return out;
}
float64 v_convert_i32_to_f32_b(const int64& v, int) {
    float64 out;
    for (unsigned i = 0; i < 64; ++i) out.lanes[i] = float(v.lanes[i]);
    return out;
}
int s_i32_ld_g(int* p) { return *p; }
void s_i32_st_g(int* p,int value) {*p=value;}
float s_f32_ld_g(float* p) { return *p; }
void s_f32_st_g(float* p, float value) { *p = value; }
void v_f32_st_tnsr_partial(int5 p, const tensor& t, const float64& v, int last, int first) {
    for (int lane = first; lane <= last; ++lane) {
        auto q = p; q[0] += lane;
        if (inside(q, t)) *static_cast<float*>(gen_addr(q, t)) = v.lanes[lane];
    }
}
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#define __global__
#define main starts_kernel
#include "../tpc/route_group_starts.c"
#undef main
#define main order_kernel
#include "../tpc/route_group_order.c"
#undef main
#define main blocks_kernel
#include "../tpc/u4_route_blocks.c"
#undef main
#define main bucket_blocks_kernel
#include "../tpc/u4_bucket_blocks.c"
#undef main
#undef __global__
#pragma GCC diagnostic pop
}

namespace {
using namespace group_emulator;
// glue.cpp validates this exact width for both ops; the kernels read the two ints in this order.
static_assert(sizeof(strata::hpu::abi::RouteGroup) == 8);
uint32_t cases = 0, tails = 0, scalar_only = 0;
uint64_t member_checks = 0, run_checks = 0;
void require(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
uint32_t next(uint32_t& state) { state = state * 1664525u + 1013904223u; return state; }

struct Routing { std::vector<int32_t> ids; int experts, rows, top_k; };

// mode 0 random routed ids, 1 every routed id 0, 2 every routed id experts-1, 3 alternating the two ends.
Routing make_routing(int experts, int rows, int top_k, uint32_t seed, int mode) {
    const int per_row = top_k + 1;
    Routing r{std::vector<int32_t>(size_t(rows) * per_row), experts, rows, top_k};
    uint32_t state = seed;
    for (int row = 0; row < rows; ++row) {
        for (int k = 0; k < top_k; ++k) {
            int32_t id;
            if (mode == 1) id = 0;
            else if (mode == 2) id = experts - 1;
            else if (mode == 3) id = ((row * top_k + k) & 1) ? experts - 1 : 0;
            else id = int32_t(next(state) % uint32_t(experts));
            r.ids[size_t(row) * per_row + k] = id;
        }
        r.ids[size_t(row) * per_row + top_k] = 0;   // route_topk leaves the shared lane at zero
    }
    return r;
}

void run_case(const std::string& name, const Routing& routing) {
    const int experts = routing.experts, slots = int(routing.ids.size());
    const auto nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> starts(size_t(experts) + 1, nan), order(size_t(slots), nan);
    tensor ids_t{const_cast<int32_t*>(routing.ids.data()), {uint64_t(slots)}, 4};
    tensor starts_t{starts.data(), {uint64_t(experts) + 1}, 4};
    tensor order_t{order.data(), {uint64_t(slots)}, 4};

    // The glue passes one abi::RouteGroup to both kernels; feed the test through it so a scalar-parameter
    // mismatch (the struct's first int is `experts`, not `slots`) reaches the kernels the way the card does.
    const strata::hpu::abi::RouteGroup params{int32_t(experts), int32_t(slots)};
    extent = {1, 1, 1, 1, 1};
    for (int e = 0; e <= experts; ++e) {
        offset = {e, 0, 0, 0, 0};
        starts_kernel(ids_t, starts_t, params.experts, params.slots);
        ++member_checks;
    }
    for (int e = 0; e < experts; ++e) {
        offset = {e, 0, 0, 0, 0};
        order_kernel(ids_t, starts_t, order_t, params.experts, params.slots);
        ++member_checks;
    }

    for (int e = 0; e <= experts; ++e)
        require(!std::isnan(starts[size_t(e)]), name + ": starts member did not write its entry");
    for (int s = 0; s < slots; ++s)
        require(!std::isnan(order[size_t(s)]), name + ": slot never appeared in the grouped order");

    // Independent writer: exclusive prefix counts and one slot-order pass per expert.
    std::vector<float> ref_starts(size_t(experts) + 1, 0.0f);
    for (int e = 0; e <= experts; ++e) {
        int count = 0;
        for (int id : routing.ids) count += id < e ? 1 : 0;
        ref_starts[size_t(e)] = float(count);
    }
    std::vector<float> ref_order(size_t(slots), nan);
    int at = 0;
    for (int e = 0; e < experts; ++e)
        for (int slot = 0; slot < slots; ++slot)
            if (routing.ids[size_t(slot)] == e) ref_order[size_t(at++)] = float(slot);
    require(starts == ref_starts, name + ": starts differs from the exclusive prefix-count writer");
    require(order == ref_order, name + ": order differs from the run-by-run stable writer");

    // Structural invariants, independent of the reference writer.
    require(starts.front() == 0.0f, name + ": starts[0] is not zero");
    require(starts.back() == float(slots), name + ": starts[experts] does not close the slot array");
    for (int e = 0; e <= experts; ++e) {
        require(starts[size_t(e)] == float(int(starts[size_t(e)])), name + ": a start is not an exact slot count");
        if (e) require(starts[size_t(e)] >= starts[size_t(e - 1)], name + ": starts is not monotonic");
    }
    std::vector<uint8_t> seen(size_t(slots), 0);
    for (int position = 0; position < slots; ++position) {
        const float value = order[size_t(position)];
        require(value == float(int(value)), name + ": an order entry is not an exact slot number");
        const int slot = int(value);
        require(slot >= 0 && slot < slots, name + ": an order entry is outside the slot array");
        require(!seen[size_t(slot)], name + ": a slot appears twice in the grouped order");
        seen[size_t(slot)] = 1;
    }
    for (int e = 0; e < experts; ++e) {
        const int first = int(starts[size_t(e)]), last = int(starts[size_t(e + 1)]);
        require(first <= last, name + ": a start run is inverted");
        ++run_checks;
        for (int position = first; position < last; ++position) {
            const int slot = int(order[size_t(position)]);
            require(routing.ids[size_t(slot)] == e, name + ": an order run contains a slot routed to another expert");
            if (position > first) require(slot > int(order[size_t(position - 1)]), name + ": a run is not in ascending slot order");
        }
    }
    int shared = 0;
    for (int slot = 0; slot < slots; ++slot) shared += slot % (routing.top_k + 1) == routing.top_k ? 1 : 0;
    int run0 = int(starts[1]) - int(starts[0]);
    require(run0 >= shared, name + ": run 0 does not hold every shared slot");
    for (int row = 0; row < routing.rows; ++row) {
        const int slot = row * (routing.top_k + 1) + routing.top_k;
        require(routing.ids[size_t(slot)] == 0, name + ": fixture shared slot does not carry id 0");
    }
    // Direct U4 buckets must match an independent stable per-expert scan,
    // including the initialized inverse for shared and overflowed slots.
    const int bucket_rows=64,blocks=(slots+63)/64;
    std::vector<float> block_counts(size_t(experts)*blocks,nan);
    std::vector<int32_t> bucket(size_t(experts)*bucket_rows,-999),spill(experts,-999),place(slots,-999);
    tensor counts_t{block_counts.data(),{uint64_t(blocks),uint64_t(experts)},4};
    tensor bucket_t{bucket.data(),{uint64_t(bucket.size())},4};
    tensor spill_t{spill.data(),{uint64_t(experts)},4},place_t{place.data(),{uint64_t(slots)},4};
    for(int i=0;i<experts*blocks;++i) {offset={i,0,0,0,0};blocks_kernel(ids_t,counts_t,experts,slots);}
    for(int e=0;e<experts;++e) {
        offset={e,0,0,0,0};bucket_blocks_kernel(ids_t,counts_t,bucket_t,spill_t,place_t,
                                               experts,bucket_rows,slots,routing.top_k+1,routing.top_k);
        int count=0;
        for(int slot=0;slot<slots;++slot) {
            if(routing.ids[slot]!=e || slot%(routing.top_k+1)==routing.top_k) continue;
            const int pos=count<bucket_rows?e*bucket_rows+count:-1;
            require(place[slot]==pos,name+": direct bucket inverse mismatch");
            if(pos>=0) require(bucket[pos]==slot,name+": direct bucket changed stable slot order");
            ++count;
        }
        require(spill[e]==std::max(0,count-bucket_rows),name+": direct bucket overflow mismatch");
        for(int row=std::min(count,bucket_rows);row<bucket_rows;++row)
            require(bucket[e*bucket_rows+row]==-1,name+": direct bucket padding not initialized");
    }
    for(int slot=routing.top_k;slot<slots;slot+=routing.top_k+1)
        require(place[slot]==-1,name+": direct bucket shared inverse not initialized");
    ++cases;
    if (slots % 64) ++tails;
    if (slots < 64) ++scalar_only;
}

void production_shapes() {
    const int rows[] = {8, 9, 10, 16, 31, 63, 64, 65, 127, 128, 129, 255, 256};
    uint32_t seed = 0x51ed270bu;
    for (int r : rows) run_case("production rows=" + std::to_string(r), make_routing(512, r, 10, seed++, 0));
    run_case("scalar-only rows=9 top_k=1", make_routing(512, 9, 1, seed++, 0));
    run_case("slots=4096 rows=256 top_k=15", make_routing(512, 256, 15, seed++, 0));
    run_case("experts=1", make_routing(1, 9, 1, seed++, 0));
    run_case("experts=2", make_routing(2, 33, 1, seed++, 0));
    run_case("experts=64", make_routing(64, 33, 10, seed++, 0));
    run_case("experts=4096 (glue maximum)", make_routing(4096, 9, 10, seed++, 0));
    run_case("every routed id 0", make_routing(512, 65, 10, seed++, 1));
    run_case("every routed id experts-1", make_routing(512, 65, 10, seed++, 2));
    run_case("alternating id ends", make_routing(512, 65, 10, seed++, 3));
}
}

int main() {
    try {
        production_shapes();
        std::cout << "PASS grouped MoE scalar pre-pass " << cases << " routing cases, " << tails
                  << " with a scalar tail (slots%64), " << scalar_only << " scalar-only (slots<64), "
                  << member_checks << " emulated members and " << run_checks
                  << " expert runs checked against an independent prefix-count writer and permutation/monotonicity/"
                     "run-membership/ascending-order invariants; no device\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
