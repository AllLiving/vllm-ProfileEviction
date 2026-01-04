// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#ifndef CPU_ATTN_NEON_BFMMLA_HPP
#define CPU_ATTN_NEON_BFMMLA_HPP

#include "cpu_attn_impl.hpp"

#include <arm_neon.h>

#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

namespace cpu_attention {
namespace {

// Attention implementation constraints
constexpr int64_t NEON_BFMMLA_BLOCK_SIZE_ALIGNMENT = 128;
constexpr int64_t NEON_BFMMLA_HEAD_SIZE_ALIGNMENT = 32;

// BFMMLA tile dimensions
constexpr int32_t TILE_ROWS = 2;  // M dimension
constexpr int32_t TILE_K = 4;     // K reduction
constexpr int32_t TILE_COLS = 2;  // N dimension (column-pair)

// Derived constants
constexpr int32_t OUTPUT_COLS_PER_BLOCK = 8;   // 4 column-pairs
constexpr int32_t K_TOKENS_PER_GROUP = 8;      // Tokens grouped in K cache
constexpr int32_t V_TOKENS_PER_ROW_BLOCK = 4;  // Tokens per V cache row block
constexpr int32_t K_INNER_STRIDE = K_TOKENS_PER_GROUP * TILE_K;
constexpr int32_t V_INNER_STRIDE = V_TOKENS_PER_ROW_BLOCK * TILE_COLS;
constexpr int32_t PACK_ELEMENTS_PER_K_CHUNK = TILE_ROWS * TILE_K;  // A packing

// Stack allocation threshold for A matrix buffer
constexpr int32_t NEON_STACK_PACK_K = 1024;
constexpr int32_t NEON_STACK_PACK_STRIDE =
    (NEON_STACK_PACK_K / TILE_K) * PACK_ELEMENTS_PER_K_CHUNK;

// Cache layout for B matrix
enum class BFMMLABLayout : uint8_t {
  TokenColumn,  // Q @ K^T: tokens are GEMM output columns
  TokenRow      // P @ V: tokens are GEMM reduction rows
};

// BF16/FP32
FORCE_INLINE float32x4_t bf16_u16x4_to_f32(uint16x4_t u16) {
  return vreinterpretq_f32_u32(vshll_n_u16(u16, 16));
}

FORCE_INLINE uint16x4_t f32_to_bf16_u16x4_trunc(float32x4_t f) {
  return vshrn_n_u32(vreinterpretq_u32_f32(f), 16);
}

// Matrix Packing and Accumulator
// Reshape two rows of Q into BFMMLA-friendly interleaved
// Input:  row0 = [a0,a1,a2,a3], row1 = [b0,b1,b2,b3]
// Output: [a0,a1,a2,a3,b0,b1,b2,b3, a4,a5,a6,a7,b4,b5,b6,b7]
FORCE_INLINE void reshape_Q_2xK_for_bfmmla(const c10::BFloat16* __restrict r0,
                                           const c10::BFloat16* __restrict r1,
                                           c10::BFloat16* __restrict dst,
                                           int32_t K) {
  const uint16_t* s0 = reinterpret_cast<const uint16_t*>(r0);
  const uint16_t* s1 = reinterpret_cast<const uint16_t*>(r1);
  uint16_t* d = reinterpret_cast<uint16_t*>(dst);

  // Process TILE_K elements at a time (PACK_ELEMENTS_PER_K_CHUNK
  // output)
  int32_t k = 0;
  for (; k + TILE_K <= K; k += TILE_K, d += PACK_ELEMENTS_PER_K_CHUNK) {
    vst1q_u16(d, vcombine_u16(vld1_u16(s0 + k), vld1_u16(s1 + k)));
  }
}

// 2x2 accumulator load/store
FORCE_INLINE float32x4_t load_acc_2x2(float* base, int64_t ldc, int32_t m_rows,
                                      int col_off) {
  float32x2_t row0 = vld1_f32(base + col_off);
  float32x2_t row1 =
      (m_rows == 2) ? vld1_f32(base + ldc + col_off) : vdup_n_f32(0.f);
  return vcombine_f32(row0, row1);
}

FORCE_INLINE void store_acc_2x2(float32x4_t acc, float* base, int64_t ldc,
                                int32_t m_rows, int col_off) {
  vst1_f32(base + col_off, vget_low_f32(acc));
  if (m_rows == 2) vst1_f32(base + ldc + col_off, vget_high_f32(acc));
}

// Accumulator Macros
// Initialize 4 column-pair accumulators for 2 rows (8 columns total)
#define INIT_ACC_ROWPAIR_4(a0, a1, a2, a3, Crow, ldc, m_rows, accum) \
  do {                                                               \
    if (accum) {                                                     \
      a0 = load_acc_2x2(Crow, ldc, m_rows, 0);                       \
      a1 = load_acc_2x2(Crow, ldc, m_rows, 2);                       \
      a2 = load_acc_2x2(Crow, ldc, m_rows, 4);                       \
      a3 = load_acc_2x2(Crow, ldc, m_rows, 6);                       \
    } else {                                                         \
      a0 = a1 = a2 = a3 = vdupq_n_f32(0.f);                          \
    }                                                                \
  } while (0)

// Store 4 column-pair accumulators back to C matrix
#define STORE_ACC_ROWPAIR_4(a0, a1, a2, a3, Crow, ldc, m_rows) \
  do {                                                         \
    store_acc_2x2(a0, Crow, ldc, m_rows, 0);                   \
    store_acc_2x2(a1, Crow, ldc, m_rows, 2);                   \
    store_acc_2x2(a2, Crow, ldc, m_rows, 4);                   \
    store_acc_2x2(a3, Crow, ldc, m_rows, 6);                   \
  } while (0)

// Perform 4 BFMMLA operations: acc += A @ B for 4 column-pairs
#define BFMMLA_COMPUTE_4(r0, r1, r2, r3, a, b0, b1, b2, b3) \
  do {                                                      \
    r0 = vbfmmlaq_f32(r0, a, b0);                           \
    r1 = vbfmmlaq_f32(r1, a, b1);                           \
    r2 = vbfmmlaq_f32(r2, a, b2);                           \
    r3 = vbfmmlaq_f32(r3, a, b3);                           \
  } while (0)

// Micro-kernel: updates a small fixed tile using BFMMLA.
// Computes C[TILE_ROWS*RP, OUTPUT_COLS_PER_BLOCK] += A_packed @ B.
// A_packed interleaves RP row-pairs; B uses either:
// - TokenColumn layout (Q @ K^T)
// - TokenRow layout (P @ V)
template <int32_t RP, int32_t K, BFMMLABLayout Layout>
FORCE_INLINE void gemm_rowpairs_x8_bfmmla_neon(
    const c10::BFloat16* const* __restrict A_packed_rp,
    const int32_t* __restrict m_rows_rp, const c10::BFloat16* __restrict B_blk,
    float* __restrict C, int64_t ldc, bool accumulate, int64_t b_stride) {
  static_assert(RP == 1 || RP == 2 || RP == 4, "RP must be 1,2,4");
  static_assert(K % TILE_K == 0, "K must be divisible by TILE_K");

  constexpr int32_t K_iters = K / TILE_K;

  auto* C_al = C;
  const auto* B_al = reinterpret_cast<const bfloat16_t*>(B_blk);

  // Setup A pointers
  const bfloat16_t* a_ptr[4] = {
      reinterpret_cast<const bfloat16_t*>(A_packed_rp[0]),
      (RP >= 2) ? reinterpret_cast<const bfloat16_t*>(A_packed_rp[1]) : nullptr,
      (RP >= 4) ? reinterpret_cast<const bfloat16_t*>(A_packed_rp[2]) : nullptr,
      (RP >= 4) ? reinterpret_cast<const bfloat16_t*>(A_packed_rp[3]) : nullptr,
  };

  // Setup B pointers based on layout
  const bfloat16_t* b_ptr[4];
  if constexpr (Layout == BFMMLABLayout::TokenRow) {
    b_ptr[0] = reinterpret_cast<const bfloat16_t*>(B_blk + 0 * b_stride);
    b_ptr[1] = reinterpret_cast<const bfloat16_t*>(B_blk + 1 * b_stride);
    b_ptr[2] = reinterpret_cast<const bfloat16_t*>(B_blk + 2 * b_stride);
    b_ptr[3] = reinterpret_cast<const bfloat16_t*>(B_blk + 3 * b_stride);
  }

  // Declare accumulators for all row-pairs
  float32x4_t acc[4][4];

// Initialize accumulators
#define INIT_RP(rp)                                                            \
  if constexpr (RP > rp) {                                                     \
    INIT_ACC_ROWPAIR_4(acc[rp][0], acc[rp][1], acc[rp][2], acc[rp][3],         \
                       C_al + (rp * 2) * ldc, ldc, m_rows_rp[rp], accumulate); \
  }
  INIT_RP(0);
  INIT_RP(1);
  INIT_RP(2);
  INIT_RP(3);
#undef INIT_RP

  // Main compute loop
  for (int32_t ki = 0; ki < K_iters; ++ki) {
    bfloat16x8_t b0, b1, b2, b3;
    if constexpr (Layout == BFMMLABLayout::TokenRow) {
      b0 = vld1q_bf16(b_ptr[0] + ki * V_INNER_STRIDE);
      b1 = vld1q_bf16(b_ptr[1] + ki * V_INNER_STRIDE);
      b2 = vld1q_bf16(b_ptr[2] + ki * V_INNER_STRIDE);
      b3 = vld1q_bf16(b_ptr[3] + ki * V_INNER_STRIDE);
    } else {
      const bfloat16_t* b_base = B_al + ki * b_stride;
      b0 = vld1q_bf16(b_base + 0 * V_INNER_STRIDE);
      b1 = vld1q_bf16(b_base + 1 * V_INNER_STRIDE);
      b2 = vld1q_bf16(b_base + 2 * V_INNER_STRIDE);
      b3 = vld1q_bf16(b_base + 3 * V_INNER_STRIDE);
    }

#define COMPUTE_RP(rp)                                                       \
  if constexpr (RP > rp) {                                                   \
    bfloat16x8_t a = vld1q_bf16(a_ptr[rp] + ki * PACK_ELEMENTS_PER_K_CHUNK); \
    BFMMLA_COMPUTE_4(acc[rp][0], acc[rp][1], acc[rp][2], acc[rp][3], a, b0,  \
                     b1, b2, b3);                                            \
  }
    COMPUTE_RP(0);
    COMPUTE_RP(1);
    COMPUTE_RP(2);
    COMPUTE_RP(3);
#undef COMPUTE_RP
  }

// Store results
#define STORE_RP(rp)                                                    \
  if constexpr (RP > rp) {                                              \
    STORE_ACC_ROWPAIR_4(acc[rp][0], acc[rp][1], acc[rp][2], acc[rp][3], \
                        C_al + (rp * 2) * ldc, ldc, m_rows_rp[rp]);     \
  }
  STORE_RP(0);
  STORE_RP(1);
  STORE_RP(2);
  STORE_RP(3);
#undef STORE_RP
}

// Micro-kernel: Runtime K (token count) version (TokenRow layout only, used for
// variable block sizes)
template <int32_t RP>
FORCE_INLINE void gemm_rowpairs_x8_bfmmla_tokenrow_neon(
    const c10::BFloat16* const* __restrict A_packed_rp,
    const int32_t* __restrict m_rows_rp, const c10::BFloat16* __restrict B_blk,
    float* __restrict C, int32_t K, int64_t ldc, bool accumulate,
    int64_t b_pair_stride) {
  static_assert(RP == 1 || RP == 2 || RP == 4, "RP must be 1,2,4");
  const int32_t K_iters = K / TILE_K;

  const bfloat16_t* a_ptr[4] = {
      reinterpret_cast<const bfloat16_t*>(A_packed_rp[0]),
      (RP >= 2) ? reinterpret_cast<const bfloat16_t*>(A_packed_rp[1]) : nullptr,
      (RP >= 4) ? reinterpret_cast<const bfloat16_t*>(A_packed_rp[2]) : nullptr,
      (RP >= 4) ? reinterpret_cast<const bfloat16_t*>(A_packed_rp[3]) : nullptr,
  };

  const bfloat16_t* b_ptr[4] = {
      reinterpret_cast<const bfloat16_t*>(B_blk + 0 * b_pair_stride),
      reinterpret_cast<const bfloat16_t*>(B_blk + 1 * b_pair_stride),
      reinterpret_cast<const bfloat16_t*>(B_blk + 2 * b_pair_stride),
      reinterpret_cast<const bfloat16_t*>(B_blk + 3 * b_pair_stride),
  };

  float32x4_t acc[4][4];

#define INIT_RP(rp)                                                    \
  if constexpr (RP > rp) {                                             \
    INIT_ACC_ROWPAIR_4(acc[rp][0], acc[rp][1], acc[rp][2], acc[rp][3], \
                       C + (rp * TILE_ROWS) * ldc, ldc, m_rows_rp[rp], \
                       accumulate);                                    \
  }
  INIT_RP(0);
  INIT_RP(1);
  INIT_RP(2);
  INIT_RP(3);
#undef INIT_RP

  for (int32_t ki = 0; ki < K_iters; ++ki) {
    bfloat16x8_t b0 = vld1q_bf16(b_ptr[0] + ki * V_INNER_STRIDE);
    bfloat16x8_t b1 = vld1q_bf16(b_ptr[1] + ki * V_INNER_STRIDE);
    bfloat16x8_t b2 = vld1q_bf16(b_ptr[2] + ki * V_INNER_STRIDE);
    bfloat16x8_t b3 = vld1q_bf16(b_ptr[3] + ki * V_INNER_STRIDE);

#define COMPUTE_RP(rp)                                                       \
  if constexpr (RP > rp) {                                                   \
    bfloat16x8_t a = vld1q_bf16(a_ptr[rp] + ki * PACK_ELEMENTS_PER_K_CHUNK); \
    BFMMLA_COMPUTE_4(acc[rp][0], acc[rp][1], acc[rp][2], acc[rp][3], a, b0,  \
                     b1, b2, b3);                                            \
  }
    COMPUTE_RP(0);
    COMPUTE_RP(1);
    COMPUTE_RP(2);
    COMPUTE_RP(3);
#undef COMPUTE_RP
  }

// Store results
#define STORE_RP(rp)                                                     \
  if constexpr (RP > rp) {                                               \
    STORE_ACC_ROWPAIR_4(acc[rp][0], acc[rp][1], acc[rp][2], acc[rp][3],  \
                        C + (rp * TILE_ROWS) * ldc, ldc, m_rows_rp[rp]); \
  }
  STORE_RP(0);
  STORE_RP(1);
  STORE_RP(2);
  STORE_RP(3);
#undef STORE_RP
}

// Meso-kernel: packs a small MBxK slice of A, then tiles over N and calls the
// micro-kernel for each OUTPUT_COLS_PER_BLOCK chunk.
template <int32_t MB, int32_t N, int32_t K, BFMMLABLayout Layout>
FORCE_INLINE void gemm_packA_compute_MB_xN(
    const c10::BFloat16* __restrict A, const c10::BFloat16* __restrict B,
    float* __restrict C, int64_t lda, int64_t ldc, int64_t b_layout_stride,
    int64_t b_reduction_stride,  // Only used for TokenColumn layout
    bool accumulate) {
  static_assert(MB >= 1 && MB <= 8, "MB must be in [1,8]");
  static_assert(N % OUTPUT_COLS_PER_BLOCK == 0,
                "N must be a multiple of OUTPUT_COLS_PER_BLOCK");
  static_assert(K % TILE_K == 0, "K must be divisible by TILE_K");

  constexpr int32_t ROW_PAIRS = (MB + 1) / TILE_ROWS;
  constexpr int32_t pack_stride = (K / TILE_K) * PACK_ELEMENTS_PER_K_CHUNK;

  alignas(64) c10::BFloat16 A_packed_stack[ROW_PAIRS * NEON_STACK_PACK_STRIDE];
  std::vector<c10::BFloat16> A_packed_heap;
  c10::BFloat16* A_packed =
      (pack_stride <= NEON_STACK_PACK_STRIDE)
          ? A_packed_stack
          : (A_packed_heap.resize(ROW_PAIRS * pack_stride),
             A_packed_heap.data());

  for (int32_t rp = 0; rp < ROW_PAIRS; ++rp) {
    const int32_t m = rp * TILE_ROWS;
    const int32_t m_rows = (m + 1 < MB) ? TILE_ROWS : 1;
    const c10::BFloat16* A0 = A + m * lda;
    const c10::BFloat16* A1 = (m_rows == TILE_ROWS) ? (A + (m + 1) * lda) : A0;
    reshape_Q_2xK_for_bfmmla(A0, A1, A_packed + rp * pack_stride, K);
  }

  for (int32_t n = 0; n < N; n += OUTPUT_COLS_PER_BLOCK) {
    const c10::BFloat16* B_blk =
        (Layout == BFMMLABLayout::TokenRow)
            ? (B + (n / TILE_COLS) * b_layout_stride)
            : (B + (n / OUTPUT_COLS_PER_BLOCK) * b_layout_stride);

    // Process row-pairs in groups of 4, 2, then 1
    int32_t rp = 0;

#define PROCESS_RP_GROUP(group_size)                               \
  for (; rp + (group_size - 1) < ROW_PAIRS; rp += group_size) {    \
    const c10::BFloat16* Ap[group_size];                           \
    int32_t mr[group_size];                                        \
    for (int32_t i = 0; i < group_size; ++i) {                     \
      Ap[i] = A_packed + (rp + i) * pack_stride;                   \
      mr[i] = (((rp + i) * TILE_ROWS + 1) < MB) ? TILE_ROWS : 1;   \
    }                                                              \
    float* C_blk = C + (rp * TILE_ROWS) * ldc + n;                 \
    gemm_rowpairs_x8_bfmmla_neon<group_size, K, Layout>(           \
        Ap, mr, B_blk, C_blk, ldc, accumulate,                     \
        (Layout == BFMMLABLayout::TokenRow) ? b_layout_stride      \
                                            : b_reduction_stride); \
  }

    PROCESS_RP_GROUP(4);
    PROCESS_RP_GROUP(2);
    PROCESS_RP_GROUP(1);
#undef PROCESS_RP_GROUP
  }
}

// Meso-kernel (runtime-K)
template <int32_t MB, int32_t N>
FORCE_INLINE void gemm_packA_compute_MB_xN(const c10::BFloat16* __restrict A,
                                           const c10::BFloat16* __restrict B,
                                           float* __restrict C, int32_t K,
                                           int64_t lda, int64_t ldc,
                                           int64_t b_pair_stride,
                                           bool accumulate) {
  static_assert(MB >= 1 && MB <= 8, "MB must be in [1,8]");
  static_assert(N % OUTPUT_COLS_PER_BLOCK == 0,
                "N must be a multiple of OUTPUT_COLS_PER_BLOCK");

  constexpr int32_t ROW_PAIRS = (MB + 1) / TILE_ROWS;
  const int32_t pack_stride = (K / TILE_K) * PACK_ELEMENTS_PER_K_CHUNK;

  alignas(64) c10::BFloat16 A_packed_stack[ROW_PAIRS * NEON_STACK_PACK_STRIDE];
  std::vector<c10::BFloat16> A_packed_heap;
  c10::BFloat16* A_packed =
      (pack_stride <= NEON_STACK_PACK_STRIDE)
          ? A_packed_stack
          : (A_packed_heap.resize(ROW_PAIRS * pack_stride),
             A_packed_heap.data());

  for (int32_t rp = 0; rp < ROW_PAIRS; ++rp) {
    const int32_t m = rp * TILE_ROWS;
    const int32_t m_rows = (m + 1 < MB) ? TILE_ROWS : 1;
    const c10::BFloat16* A0 = A + m * lda;
    const c10::BFloat16* A1 = (m_rows == TILE_ROWS) ? (A + (m + 1) * lda) : A0;
    reshape_Q_2xK_for_bfmmla(A0, A1, A_packed + rp * pack_stride, K);
  }

  for (int32_t n = 0; n < N; n += OUTPUT_COLS_PER_BLOCK) {
    const c10::BFloat16* B_blk = B + (n / TILE_COLS) * b_pair_stride;
    int32_t rp = 0;

#define PROCESS_RP_GROUP(group_size)                              \
  for (; rp + (group_size - 1) < ROW_PAIRS; rp += group_size) {   \
    const c10::BFloat16* Ap[group_size];                          \
    int32_t mr[group_size];                                       \
    for (int32_t i = 0; i < group_size; ++i) {                    \
      Ap[i] = A_packed + (rp + i) * pack_stride;                  \
      mr[i] = (((rp + i) * TILE_ROWS + 1) < MB) ? TILE_ROWS : 1;  \
    }                                                             \
    float* C_blk = C + (rp * TILE_ROWS) * ldc + n;                \
    gemm_rowpairs_x8_bfmmla_tokenrow_neon<group_size>(            \
        Ap, mr, B_blk, C_blk, K, ldc, accumulate, b_pair_stride); \
  }

    PROCESS_RP_GROUP(4);
    PROCESS_RP_GROUP(2);
    PROCESS_RP_GROUP(1);
#undef PROCESS_RP_GROUP
  }
}

// Macro-kernel: iterates over M in MB={8,4,2,1} chunks
template <int32_t N>
FORCE_INLINE void gemm_macro_neon_bfmmla(const c10::BFloat16* __restrict A,
                                         const c10::BFloat16* __restrict B,
                                         float* __restrict C, int32_t M,
                                         int32_t K, int64_t lda, int64_t ldc,
                                         int64_t b_pair_stride,
                                         bool accumulate) {
  static_assert(N % OUTPUT_COLS_PER_BLOCK == 0,
                "N must be a multiple of OUTPUT_COLS_PER_BLOCK");

  for (int32_t m = 0; m < M;) {
    const int32_t rem = M - m;
    const c10::BFloat16* A_blk = A + m * lda;
    float* C_blk = C + m * ldc;

#define DISPATCH_MB(mb)                                                        \
  gemm_packA_compute_MB_xN<mb, N>(A_blk, B, C_blk, K, lda, ldc, b_pair_stride, \
                                  accumulate)

    if (rem >= 8) {
      DISPATCH_MB(8);
      m += 8;
    } else if (rem >= 4) {
      DISPATCH_MB(4);
      m += 4;
    } else if (rem >= 2) {
      DISPATCH_MB(2);
      m += 2;
    } else {
      DISPATCH_MB(1);
      m += 1;
    }
#undef DISPATCH_MB
  }
}

// Macro-kernel (PV, TokenRow): compile-time K specializations for common token
// counts (K=128,256). Other K values fall back to the runtime-K path above
template <int32_t N, int32_t K>
FORCE_INLINE void gemm_macro_neon_bfmmla_tokenrow_k(
    const c10::BFloat16* __restrict A, const c10::BFloat16* __restrict B,
    float* __restrict C, int32_t M, int64_t lda, int64_t ldc,
    int64_t b_pair_stride, bool accumulate) {
  static_assert(N % OUTPUT_COLS_PER_BLOCK == 0 && K % TILE_K == 0);

  for (int32_t m = 0; m < M;) {
    const int32_t rem = M - m;
    const c10::BFloat16* A_blk = A + m * lda;
    float* C_blk = C + m * ldc;

#define DISPATCH_MB(mb)                                        \
  gemm_packA_compute_MB_xN<mb, N, K, BFMMLABLayout::TokenRow>( \
      A_blk, B, C_blk, lda, ldc, b_pair_stride, 0, accumulate)

    if (rem >= 8) {
      DISPATCH_MB(8);
      m += 8;
    } else if (rem >= 4) {
      DISPATCH_MB(4);
      m += 4;
    } else if (rem >= 2) {
      DISPATCH_MB(2);
      m += 2;
    } else {
      DISPATCH_MB(1);
      m += 1;
    }
#undef DISPATCH_MB
  }
}

// Macro-kernel (QK, TokenColumn): compile-time K specialization where
// K=head_dim
template <int32_t N, int32_t K>
FORCE_INLINE void gemm_macro_neon_bfmmla_tokencol_k(
    const c10::BFloat16* __restrict A, const c10::BFloat16* __restrict B,
    float* __restrict C, int32_t M, int64_t lda, int64_t ldc,
    int64_t b_token_block_stride, int64_t b_reduction_stride, bool accumulate) {
  static_assert(N % OUTPUT_COLS_PER_BLOCK == 0);

  for (int32_t m = 0; m < M;) {
    const int32_t rem = M - m;
    const c10::BFloat16* A_blk = A + m * lda;
    float* C_blk = C + m * ldc;

#define DISPATCH_MB(mb)                                                    \
  gemm_packA_compute_MB_xN<mb, N, K, BFMMLABLayout::TokenColumn>(          \
      A_blk, B, C_blk, lda, ldc, b_token_block_stride, b_reduction_stride, \
      accumulate)

    if (rem >= 8) {
      DISPATCH_MB(8);
      m += 8;
    } else if (rem >= 4) {
      DISPATCH_MB(4);
      m += 4;
    } else if (rem >= 2) {
      DISPATCH_MB(2);
      m += 2;
    } else {
      DISPATCH_MB(1);
      m += 1;
    }
#undef DISPATCH_MB
  }
}

#undef INIT_ACC_ROWPAIR_4
#undef STORE_ACC_ROWPAIR_4
#undef BFMMLA_COMPUTE_4

}  // namespace

// TileGemm Adapter for Attention

template <typename kv_cache_t>
class TileGemmNEONBFMMLA {
 public:
  template <AttentionGemmPhase phase, int32_t head_dim_ct>
  FORCE_INLINE static void gemm(const int32_t m_size, void* __restrict__ a_tile,
                                kv_cache_t* __restrict__ b_tile,
                                float* __restrict__ c_tile, const int64_t lda,
                                [[maybe_unused]] const int64_t ldb,
                                const int64_t ldc,
                                [[maybe_unused]] const int32_t block_size,
                                [[maybe_unused]] const int32_t dynamic_k_size,
                                const bool accum_c) {
    if constexpr (phase == AttentionGemmPhase::QK) {
      const int64_t b_reduction_stride = K_INNER_STRIDE;
      const int64_t b_token_block_stride =
          (head_dim_ct / TILE_K) * K_INNER_STRIDE;

      gemm_macro_neon_bfmmla_tokencol_k<NEON_BFMMLA_BLOCK_SIZE_ALIGNMENT,
                                        head_dim_ct>(
          reinterpret_cast<c10::BFloat16*>(a_tile), b_tile, c_tile, m_size, lda,
          ldc, b_token_block_stride, b_reduction_stride, accum_c);
    } else {
      const int64_t b_pair_stride =
          (block_size / V_TOKENS_PER_ROW_BLOCK) * V_INNER_STRIDE;

      // PV gemm with runtime K specialization
      if (dynamic_k_size == 128) {
        gemm_macro_neon_bfmmla_tokenrow_k<head_dim_ct, 128>(
            reinterpret_cast<const c10::BFloat16*>(a_tile), b_tile, c_tile,
            m_size, lda, ldc, b_pair_stride, accum_c);
      } else if (dynamic_k_size == 256) {
        gemm_macro_neon_bfmmla_tokenrow_k<head_dim_ct, 256>(
            reinterpret_cast<const c10::BFloat16*>(a_tile), b_tile, c_tile,
            m_size, lda, ldc, b_pair_stride, accum_c);
      } else {
        gemm_macro_neon_bfmmla<head_dim_ct>(
            reinterpret_cast<const c10::BFloat16*>(a_tile), b_tile, c_tile,
            m_size, dynamic_k_size, lda, ldc, b_pair_stride, accum_c);
      }
    }
  }
};

// AttentionImpl Specialization for NEON BFMMLA
template <typename scalar_t, int64_t head_dim>
class AttentionImpl<ISA::NEON_BFMMLA, scalar_t, head_dim> {
 public:
  using query_t = scalar_t;
  using q_buffer_t = c10::BFloat16;
  using kv_cache_t = c10::BFloat16;
  using logits_buffer_t = float;
  using partial_output_buffer_t = float;
  using prob_buffer_t = c10::BFloat16;

  static constexpr int64_t BlockSizeAlignment =
      NEON_BFMMLA_BLOCK_SIZE_ALIGNMENT;
  static constexpr int64_t HeadDimAlignment = NEON_BFMMLA_HEAD_SIZE_ALIGNMENT;
  static constexpr int64_t MaxQHeadNumPerIteration = 512;
  static constexpr int64_t HeadDim = head_dim;
  static constexpr ISA ISAType = ISA::NEON_BFMMLA;
  static constexpr bool scale_on_logits = false;

  static_assert(HeadDim % HeadDimAlignment == 0);
  static_assert(HeadDimAlignment % OUTPUT_COLS_PER_BLOCK == 0);
  static_assert(BlockSizeAlignment % OUTPUT_COLS_PER_BLOCK == 0);
  static_assert(HeadDim % TILE_K == 0, "HeadDim must be a multiple of TILE_K");

 public:
  template <template <typename tile_gemm_t> typename attention>
  FORCE_INLINE void execute_attention(DEFINE_CPU_ATTENTION_PARAMS) {
    attention<TileGemmNEONBFMMLA<kv_cache_t>> attention_iteration;
    attention_iteration(CPU_ATTENTION_PARAMS);
  }

  // Key cache stride per token group (TokenColumn layout; QK)
  static constexpr int64_t k_cache_token_group_stride(
      [[maybe_unused]] const int32_t block_size) {
    static_assert(BlockSizeAlignment % K_TOKENS_PER_GROUP == 0);
    return (BlockSizeAlignment / K_TOKENS_PER_GROUP) *
           ((head_dim / TILE_K) * K_INNER_STRIDE);
  }

  // Value cache stride per token group (TokenRow layout; PV)
  static constexpr int64_t v_cache_token_group_stride(
      [[maybe_unused]] const int32_t block_size) {
    static_assert(BlockSizeAlignment % V_TOKENS_PER_ROW_BLOCK == 0);
    return (BlockSizeAlignment / V_TOKENS_PER_ROW_BLOCK) * V_INNER_STRIDE;
  }

  static constexpr int64_t v_cache_head_group_stride(const int32_t block_size) {
    return HeadDimAlignment * block_size;
  }

  // Convert Q heads to BF16 and apply scale factor
  static void copy_q_heads_tile(scalar_t* __restrict__ src,
                                c10::BFloat16* __restrict__ q_buffer,
                                const int32_t q_num,
                                const int32_t q_heads_per_kv,
                                const int64_t q_num_stride,
                                const int64_t q_head_stride, float scale) {
    constexpr int32_t dim = static_cast<int32_t>(head_dim);
    const float32x4_t scale_vec = vdupq_n_f32(scale);

    for (int32_t qi = 0; qi < q_num; ++qi) {
      for (int32_t hi = 0; hi < q_heads_per_kv; ++hi) {
        scalar_t* __restrict__ curr_q =
            src + qi * q_num_stride + hi * q_head_stride;
        c10::BFloat16* __restrict__ dst =
            q_buffer + qi * q_heads_per_kv * head_dim + hi * head_dim;

        if constexpr (std::is_same_v<scalar_t, c10::BFloat16>) {
          const uint16_t* src_u16 = reinterpret_cast<const uint16_t*>(curr_q);
          uint16_t* dst_u16 = reinterpret_cast<uint16_t*>(dst);

          for (int32_t i = 0; i < dim; i += OUTPUT_COLS_PER_BLOCK) {
            uint16x8_t in8 = vld1q_u16(src_u16 + i);

            float32x4_t lo = bf16_u16x4_to_f32(vget_low_u16(in8));
            float32x4_t hi_vec = bf16_u16x4_to_f32(vget_high_u16(in8));

            lo = vmulq_f32(lo, scale_vec);
            hi_vec = vmulq_f32(hi_vec, scale_vec);

            vst1q_u16(dst_u16 + i,
                      vcombine_u16(f32_to_bf16_u16x4_trunc(lo),
                                   f32_to_bf16_u16x4_trunc(hi_vec)));
          }
        } else {  // Fallback for float/Half types
          for (int32_t i = 0; i < head_dim; ++i) {
            dst[i] = c10::BFloat16(static_cast<float>(curr_q[i]) * scale);
          }
        }
      }
    }
  }

 private:
  FORCE_INLINE static void store4_as_bf16_u16(uint16_t* __restrict dst_u16,
                                              const scalar_t* __restrict src) {
    if constexpr (std::is_same_v<scalar_t, c10::BFloat16>) {
      vst1_u16(dst_u16, vld1_u16(reinterpret_cast<const uint16_t*>(src)));
    } else if constexpr (std::is_same_v<scalar_t,
                                        float>) {  // FP32 to BF16 conversion
      float32x4_t f = vld1q_f32(src);
      vst1_u16(dst_u16, vshrn_n_u32(vreinterpretq_u32_f32(f), 16));
    } else {  // Generic conversion for Half and other types
      for (int i = 0; i < 4; ++i) {
        c10::BFloat16 bf16_val = c10::BFloat16(static_cast<float>(src[i]));
        std::memcpy(&dst_u16[i], &bf16_val, sizeof(uint16_t));
      }
    }
  }

 public:
  // Reshape and cache K/V into BFMMLA-optimized layouts
  // K cache:
  // [block_size/K_TOKENS_PER_GROUP][head_dim/TILE_K][K_INNER_STRIDE]
  // - TokenColumn
  // V cache:
  // [head_dim/TILE_COLS][block_size/V_TOKENS_PER_ROW_BLOCK][V_INNER_STRIDE]
  // - TokenRows
  static void reshape_and_cache(
      const scalar_t* __restrict__ key, const scalar_t* __restrict__ value,
      kv_cache_t* __restrict__ key_cache, kv_cache_t* __restrict__ value_cache,
      const int64_t* __restrict__ slot_mapping, const int64_t token_num,
      const int64_t key_token_num_stride, const int64_t value_token_num_stride,
      const int64_t head_num, const int64_t key_head_num_stride,
      const int64_t value_head_num_stride,
      [[maybe_unused]] const int64_t num_blocks,
      const int64_t num_blocks_stride, const int64_t cache_head_num_stride,
      const int64_t block_size,
      [[maybe_unused]] const int64_t block_size_stride) {
    const int64_t k_block_stride = (head_dim / TILE_K) * K_INNER_STRIDE;
    const int64_t v_pair_stride =
        (block_size / V_TOKENS_PER_ROW_BLOCK) * V_INNER_STRIDE;

#pragma omp parallel for
    for (int64_t head_idx = 0; head_idx < head_num; ++head_idx) {
      for (int64_t token_idx = 0; token_idx < token_num; ++token_idx) {
        const int64_t pos = slot_mapping[token_idx];
        if (pos < 0) continue;

        const int64_t block_idx = pos / block_size;
        const int64_t block_offset = pos % block_size;

        // Key cache: TokenColumn QK
        {
          const scalar_t* __restrict key_src =
              key + token_idx * key_token_num_stride +
              head_idx * key_head_num_stride;

          kv_cache_t* __restrict key_base = key_cache +
                                            block_idx * num_blocks_stride +
                                            head_idx * cache_head_num_stride;

          const int64_t block_in_block = block_offset / K_TOKENS_PER_GROUP;
          const int64_t pair_in_block =
              (block_offset % K_TOKENS_PER_GROUP) / TILE_COLS;
          const int64_t lane_base = (block_offset & 1) ? TILE_K : 0;

          kv_cache_t* __restrict block_base =
              key_base + block_in_block * k_block_stride;

          for (int64_t hd4 = 0; hd4 < head_dim / TILE_K; ++hd4) {
            store4_as_bf16_u16(reinterpret_cast<uint16_t*>(
                                   block_base + hd4 * K_INNER_STRIDE +
                                   pair_in_block * V_INNER_STRIDE + lane_base),
                               key_src + hd4 * TILE_K);
          }
        }

        // Value cache: TokenRow PV
        {
          const scalar_t* __restrict value_src =
              value + token_idx * value_token_num_stride +
              head_idx * value_head_num_stride;

          kv_cache_t* __restrict value_base = value_cache +
                                              block_idx * num_blocks_stride +
                                              head_idx * cache_head_num_stride;

          const int64_t row_block = block_offset / V_TOKENS_PER_ROW_BLOCK;
          const int64_t lane = block_offset & (V_TOKENS_PER_ROW_BLOCK - 1);

          kv_cache_t* __restrict row_block_base =
              value_base + row_block * V_INNER_STRIDE;

          for (int64_t hd2 = 0; hd2 < head_dim / TILE_COLS; ++hd2) {
            kv_cache_t* __restrict dst_val =
                row_block_base + hd2 * v_pair_stride;

            if constexpr (std::is_same_v<scalar_t, c10::BFloat16>) {
              const uint16_t* src_u16 =
                  reinterpret_cast<const uint16_t*>(value_src);
              uint16_t* dst_u16 = reinterpret_cast<uint16_t*>(dst_val);
              dst_u16[lane] = src_u16[hd2 * TILE_COLS + 0];
              dst_u16[lane + V_TOKENS_PER_ROW_BLOCK] =
                  src_u16[hd2 * TILE_COLS + 1];
            } else {  // Convert float/Half to BF16
              dst_val[lane] = c10::BFloat16(
                  static_cast<float>(value_src[hd2 * TILE_COLS + 0]));
              dst_val[lane + V_TOKENS_PER_ROW_BLOCK] = c10::BFloat16(
                  static_cast<float>(value_src[hd2 * TILE_COLS + 1]));
            }
          }
        }
      }
    }
  }
};

}  // namespace cpu_attention

#endif  // CPU_ATTN_NEON_BFMMLA_HPP
