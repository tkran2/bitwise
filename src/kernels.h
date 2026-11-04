#pragma once
// Numerical kernels: matmul, RMSNorm, RoPE, SwiGLU and paged attention.
//
// Invariance contract (KernelMode::Invariant): every output element is reduced in one
// canonical order that depends only on the element's own inputs, never on the number
// of rows in the batch, on which other rows are present, on the thread count, or on
// how a prompt was chunked. Concretely:
//   * matmul: y[m][n] accumulates x[m][k] * w[n][k] for k = 0..K-1 in a single fused
//     multiply-add chain. Wide batches and skinny batches only change which outputs are
//     computed side by side in registers, not how any one output is summed.
//   * attention: the scores of a query at position p are taken over keys 0..p in key
//     order, and softmax sums and value accumulation also run in key order, whether the
//     query arrives in a prefill chunk, a decode step or a speculative verify batch.
//   * reductions inside one vector (dot products, sums of squares) use a fixed 16-lane
//     split and a fixed reduction tree.
//
// KernelMode::ShapeSpecialized picks reduction strategies from the problem shape, the
// way tuned BLAS and attention libraries do: skinny matmuls split K into two
// accumulators, and attention splits the keys across threads when there are too few
// queries to occupy them. These are reasonable optimizations, and they make a
// request's numerics depend on what else is in the batch.

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "thread_pool.h"

namespace bw {

enum class DType : uint8_t { F32 = 0, BF16 = 1 };
enum class KernelMode : uint8_t { Invariant = 0, ShapeSpecialized = 1 };

inline float bf16_to_f32(uint16_t v) {
  uint32_t bits = static_cast<uint32_t>(v) << 16;
  float f;
  __builtin_memcpy(&f, &bits, 4);
  return f;
}

// A weight matrix W[N][K] repacked into column panels of 16 outputs: panel p holds
// W[16p + j][k] at offset (p * K + k) * 16 + j, zero-padded past N. Loading one
// 16-wide vector per k is then a contiguous stream.
class PackedMatrix {
 public:
  PackedMatrix() = default;
  // `src` is row-major [n][k] in `dtype` (BF16 stored as raw uint16).
  PackedMatrix(const void* src, DType dtype, int n, int k);
  // Several row-major matrices with the same K stacked by rows (e.g. Q, K and V fused
  // into one projection). Each part is (pointer, rows).
  PackedMatrix(const std::vector<std::pair<const void*, int>>& parts, DType dtype, int k);

  int rows() const { return n_; }
  int cols() const { return k_; }
  int panels() const { return (n_ + 15) / 16; }
  DType dtype() const { return dtype_; }
  const float* f32() const { return f32_.data(); }
  const uint16_t* bf16() const { return bf16_.data(); }
  float at(int n, int k) const;  // for tests
  std::size_t bytes() const { return f32_.size() * 4 + bf16_.size() * 2; }

 private:
  int n_ = 0, k_ = 0;
  DType dtype_ = DType::F32;
  std::vector<float> f32_;
  std::vector<uint16_t> bf16_;
};

// y[m][n] = sum_k x[m][k] * W[n][k] (+ bias[n]). x is [m][K], y is [m][N], both contiguous.
void matmul(const float* x, int m, const PackedMatrix& w, const float* bias, float* y, ThreadPool& pool,
            KernelMode mode);

// Portable scalar version of the invariant matmul. Uses std::fma in the same order as
// the vector kernel, so it must agree with it bit for bit (checked by the tests).
void matmul_reference(const float* x, int m, const PackedMatrix& w, const float* bias, float* y);

// Canonical dot product and sum of squares: 16 lane-wise FMA chains, then a fixed tree.
float dot_canonical(const float* a, const float* b, int n);
float sumsq_canonical(const float* a, int n);

void rmsnorm(const float* x, const float* weight, float* y, int m, int dim, float eps);

// Rotary embedding in the rotate-half layout used by Llama and Qwen2 checkpoints.
// `vec` holds `heads` vectors of `head_dim` for one token at position `pos`.
void rope(float* vec, int heads, int head_dim, int pos, const float* inv_freq);

// h[i] = silu(gate[i]) * up[i] for rows laid out as [gate (inter) | up (inter)].
void swiglu(const float* gate_up, float* h, int m, int inter);

struct KVLayout {
  int num_blocks = 0;
  int block_size = 16;
  int kv_heads = 0;
  int head_dim = 0;
  // offset of key (or value) vector for token slot `slot` of `block`, head `h`, in one layer
  std::size_t offset(int block, int h, int slot) const {
    return ((static_cast<std::size_t>(block) * kv_heads + h) * block_size + slot) * head_dim;
  }
  std::size_t layer_floats() const {
    return static_cast<std::size_t>(num_blocks) * kv_heads * block_size * head_dim;
  }
};

struct AttentionRow {
  int pos;                     // absolute position of this query token
  const int32_t* block_table;  // the sequence's blocks
};

// out[r][h*hd..] = softmax(q k^T / sqrt(hd)) v over keys 0..pos of row r's sequence.
void paged_attention(const float* q, int q_stride, const std::vector<AttentionRow>& rows, const float* k_cache,
                     const float* v_cache, const KVLayout& layout, int heads, float* out, int out_stride,
                     ThreadPool& pool, KernelMode mode);

}  // namespace bw
