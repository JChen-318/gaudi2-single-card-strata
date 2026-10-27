#pragma once
#include <cstdint>

namespace strata::hpu::abi {
struct GdnStep { int32_t width, key_heads; };
struct GdnConv { int32_t channels, taps; };
struct GdnPrepare { int32_t width, key_heads, value_heads; float epsilon; };
struct GdnNorm { int32_t width, value_heads; float epsilon; };
// STRATA_HPU_DFLASH verify windows: conv, prepare and norm over `rows` rows in one kernel each.
// alpha_col/beta_col/z_col: the input's column offset when it is a block of the merged F32 projection
// (STRATA_HPU_GDN_FUSED_PARTS); zero when the input is the part's own tensor.
struct GdnConvRows { int32_t channels, taps, rows; };
struct GdnPrepareRows { int32_t width, key_heads, value_heads; float epsilon; int32_t rows, alpha_col, beta_col; };
struct GdnNormRows { int32_t width, value_heads; float epsilon; int32_t rows, z_col; };
struct GdnStepRows { int32_t width, key_heads, rows; }; // strata_gdn_step_rows4_f32 (STRATA_HPU_GDN_STEP_ROWS)
static_assert(sizeof(GdnStepRows) == 12);
static_assert(sizeof(GdnConvRows) == 12 && sizeof(GdnPrepareRows) == 28 && sizeof(GdnNormRows) == 20);
static_assert(sizeof(GdnStep) == 8 && sizeof(GdnConv) == 8 &&
              sizeof(GdnPrepare) == 16 && sizeof(GdnNorm) == 12);
}
