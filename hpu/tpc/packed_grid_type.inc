// The 64-bit codebook entry type for the generated IQ tables, in both worlds.
//
// TPC rejects `unsigned long long` outright and its `uint64` is a 64-lane SIMD vector, so the
// scalar 64-bit type on the target is `unsigned long` (measured with tpc-clang -march=gaudi2 in
// work/remote_logs/tpc-literals-20261017T035827Z.json). The host keeps long long because Windows
// `unsigned long` is 32 bits there.
#ifdef __TPC__
typedef unsigned long packed_grid_u64;
#else
typedef unsigned long long packed_grid_u64;
#endif
