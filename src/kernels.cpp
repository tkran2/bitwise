#include "kernels.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#if defined(__AVX512F__)
#include <immintrin.h>
#define BW_AVX512 1
#else
#define BW_AVX512 0
#endif

namespace bw {

// ---------------------------------------------------------------------------------
// Packing

PackedMatrix::PackedMatrix(const void* src, DType dtype, int n, int k)
    : PackedMatrix(std::vector<std::pair<const void*, int>>{{src, n}}, dtype, k) {}

PackedMatrix::PackedMatrix(const std::vector<std::pair<const void*, int>>& parts, DType dtype, int k)
    : k_(k), dtype_(dtype) {
  for (const auto& part : parts) n_ += part.second;
  const std::size_t total = static_cast<std::size_t>(panels()) * k_ * 16;
  if (dtype_ == DType::BF16)
    bf16_.assign(total, 0);
  else
    f32_.assign(total, 0.0f);
  int row = 0;
  for (const auto& [ptr, rows] : parts) {
    for (int r = 0; r < rows; ++r, ++row) {
      const std::size_t base = static_cast<std::size_t>(row / 16) * k_ * 16 + row % 16;
      for (int c = 0; c < k_; ++c) {
        const std::size_t src_idx = static_cast<std::size_t>(r) * k_ + c;
        if (dtype_ == DType::BF16)
          bf16_[base + static_cast<std::size_t>(c) * 16] = static_cast<const uint16_t*>(ptr)[src_idx];
        else
          f32_[base + static_cast<std::size_t>(c) * 16] = static_cast<const float*>(ptr)[src_idx];
      }
    }
  }
}

float PackedMatrix::at(int n, int k) const {
  const std::size_t idx = (static_cast<std::size_t>(n / 16) * k_ + k) * 16 + n % 16;
  return dtype_ == DType::BF16 ? bf16_to_f32(bf16_[idx]) : f32_[idx];
}

// ---------------------------------------------------------------------------------
// Canonical in-vector reductions

namespace {

// The reduction tree of hsum16 below, written out in scalar form.
inline float tree16(const float* v) {
  float s8[8], s4[4];
  for (int i = 0; i < 8; ++i) s8[i] = v[i] + v[i + 8];
  for (int i = 0; i < 4; ++i) s4[i] = s8[i] + s8[i + 4];
  const float a = s4[0] + s4[2];
  const float b = s4[1] + s4[3];
  return a + b;
}

#if BW_AVX512
inline float hsum16(__m512 v) {
  const __m256 lo = _mm512_castps512_ps256(v);
  const __m256 hi = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(v), 1));
  const __m256 s8 = _mm256_add_ps(lo, hi);
  const __m128 s4 = _mm_add_ps(_mm256_castps256_ps128(s8), _mm256_extractf128_ps(s8, 1));
  const __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));  // [s4[0]+s4[2], s4[1]+s4[3], ...]
  const __m128 s1 = _mm_add_ss(s2, _mm_movehdup_ps(s2));
  return _mm_cvtss_f32(s1);
}

inline __m512 load_w(const float* p) { return _mm512_loadu_ps(p); }
inline __m512 load_w(const uint16_t* p) {
  const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
  return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(v), 16));
}
#endif

}  // namespace

float dot_canonical(const float* a, const float* b, int n) {
#if BW_AVX512
  __m512 acc = _mm512_setzero_ps();
  int i = 0;
  for (; i + 16 <= n; i += 16) acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc);
  if (i < n) {
    const __mmask16 m = static_cast<__mmask16>((1u << (n - i)) - 1);
    acc = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(m, a + i), _mm512_maskz_loadu_ps(m, b + i), acc, m);
  }
  return hsum16(acc);
#else
  float lane[16] = {0};
  for (int i = 0; i < n; ++i) lane[i % 16] = std::fma(a[i], b[i], lane[i % 16]);
  return tree16(lane);
#endif
}

float sumsq_canonical(const float* a, int n) { return dot_canonical(a, a, n); }

// ---------------------------------------------------------------------------------
// Matmul

namespace {

#if BW_AVX512
// MR rows x NV panels (16 outputs each). Each output is one FMA chain over k.
template <int MR, int NV, typename WT>
inline void micro(const float* x, int K, const WT* w, std::size_t panel_stride, const float* bias, float* y,
                  int ldy, int ncols) {
  __m512 acc[MR][NV];
  for (int i = 0; i < MR; ++i)
    for (int v = 0; v < NV; ++v) acc[i][v] = _mm512_setzero_ps();
  for (int k = 0; k < K; ++k) {
    __m512 wv[NV];
    for (int v = 0; v < NV; ++v) wv[v] = load_w(w + v * panel_stride + static_cast<std::size_t>(k) * 16);
    for (int i = 0; i < MR; ++i) {
      const __m512 xb = _mm512_set1_ps(x[static_cast<std::size_t>(i) * K + k]);
      for (int v = 0; v < NV; ++v) acc[i][v] = _mm512_fmadd_ps(xb, wv[v], acc[i][v]);
    }
  }
  for (int v = 0; v < NV; ++v) {
    const int left = ncols - v * 16;
    const __mmask16 m = left >= 16 ? static_cast<__mmask16>(0xFFFF) : static_cast<__mmask16>((1u << left) - 1);
    const __m512 b = bias ? _mm512_maskz_loadu_ps(m, bias + v * 16) : _mm512_setzero_ps();
    for (int i = 0; i < MR; ++i) {
      const __m512 out = bias ? _mm512_add_ps(acc[i][v], b) : acc[i][v];
      _mm512_mask_storeu_ps(y + static_cast<std::size_t>(i) * ldy + v * 16, m, out);
    }
  }
}

// Shape-specialized variant for skinny batches: two accumulators per output over the
// two halves of K, combined at the end. More instruction-level parallelism, different
// rounding.
template <int MR, int NV, typename WT>
inline void micro_split2(const float* x, int K, const WT* w, std::size_t panel_stride, const float* bias, float* y,
                         int ldy, int ncols) {
  __m512 a0[MR][NV], a1[MR][NV];
  for (int i = 0; i < MR; ++i)
    for (int v = 0; v < NV; ++v) a0[i][v] = a1[i][v] = _mm512_setzero_ps();
  const int half = K / 2;
  for (int k = 0; k < half; ++k) {
    for (int v = 0; v < NV; ++v) {
      const __m512 w0 = load_w(w + v * panel_stride + static_cast<std::size_t>(k) * 16);
      const __m512 w1 = load_w(w + v * panel_stride + static_cast<std::size_t>(k + half) * 16);
      for (int i = 0; i < MR; ++i) {
        a0[i][v] = _mm512_fmadd_ps(_mm512_set1_ps(x[static_cast<std::size_t>(i) * K + k]), w0, a0[i][v]);
        a1[i][v] = _mm512_fmadd_ps(_mm512_set1_ps(x[static_cast<std::size_t>(i) * K + k + half]), w1, a1[i][v]);
      }
    }
  }
  for (int k = 2 * half; k < K; ++k)
    for (int v = 0; v < NV; ++v) {
      const __m512 wv = load_w(w + v * panel_stride + static_cast<std::size_t>(k) * 16);
      for (int i = 0; i < MR; ++i)
        a1[i][v] = _mm512_fmadd_ps(_mm512_set1_ps(x[static_cast<std::size_t>(i) * K + k]), wv, a1[i][v]);
    }
  for (int v = 0; v < NV; ++v) {
    const int left = ncols - v * 16;
    const __mmask16 m = left >= 16 ? static_cast<__mmask16>(0xFFFF) : static_cast<__mmask16>((1u << left) - 1);
    const __m512 b = bias ? _mm512_maskz_loadu_ps(m, bias + v * 16) : _mm512_setzero_ps();
    for (int i = 0; i < MR; ++i) {
      __m512 out = _mm512_add_ps(a0[i][v], a1[i][v]);
      if (bias) out = _mm512_add_ps(out, b);
      _mm512_mask_storeu_ps(y + static_cast<std::size_t>(i) * ldy + v * 16, m, out);
    }
  }
}

template <typename WT, bool Split>
void tile(int mr, int nv, const float* x, int K, const WT* w, std::size_t ps, const float* bias, float* y, int ldy,
          int ncols) {
#define BW_CASE(MR, NV)                                                         \
  case MR * 16 + NV:                                                            \
    if constexpr (Split)                                                        \
      micro_split2<MR, NV>(x, K, w, ps, bias, y, ldy, ncols);                   \
    else                                                                        \
      micro<MR, NV>(x, K, w, ps, bias, y, ldy, ncols);                          \
    return;
  switch (mr * 16 + nv) {
    BW_CASE(1, 1) BW_CASE(1, 2) BW_CASE(1, 3) BW_CASE(1, 4) BW_CASE(1, 5) BW_CASE(1, 6) BW_CASE(1, 7) BW_CASE(1, 8)
    BW_CASE(2, 1) BW_CASE(2, 2) BW_CASE(2, 3) BW_CASE(2, 4)
    BW_CASE(3, 1) BW_CASE(3, 2) BW_CASE(3, 3) BW_CASE(3, 4)
    BW_CASE(4, 1) BW_CASE(4, 2) BW_CASE(5, 1) BW_CASE(5, 2) BW_CASE(6, 1) BW_CASE(6, 2)
    default:
      throw std::logic_error("unsupported matmul tile");
  }
#undef BW_CASE
}

template <typename WT>
void matmul_avx512(const float* x, int m, const PackedMatrix& W, const WT* wdata, const float* bias, float* y,
                   ThreadPool& pool, KernelMode mode) {
  const int N = W.rows(), K = W.cols(), P = W.panels();
  const std::size_t ps = static_cast<std::size_t>(K) * 16;
  // How many outputs to keep in registers side by side. This only changes which
  // outputs share a pass over k, never how any single output is accumulated.
  const int nv_main = m == 1 ? 8 : (m <= 3 ? 4 : 2);
  const int mr_main = m <= 3 ? m : 6;
  const bool split = mode == KernelMode::ShapeSpecialized && m < 4 && K >= 512;
  const int groups = (P + nv_main - 1) / nv_main;
  constexpr int kRowBlock = 48;
  pool.parallel_for(groups, [&](std::size_t g0, std::size_t g1) {
    for (int mb = 0; mb < m; mb += kRowBlock) {
      const int mend = std::min(m, mb + kRowBlock);
      for (std::size_t g = g0; g < g1; ++g) {
        const int p0 = static_cast<int>(g) * nv_main;
        const int nv = std::min(nv_main, P - p0);
        const int n0 = p0 * 16;
        const int ncols = std::min(N - n0, nv * 16);
        const WT* wp = wdata + static_cast<std::size_t>(p0) * ps;
        const float* bp = bias ? bias + n0 : nullptr;
        for (int r = mb; r < mend;) {
          const int mr = std::min(mr_main, mend - r);
          const float* xr = x + static_cast<std::size_t>(r) * K;
          float* yr = y + static_cast<std::size_t>(r) * N + n0;
          if (split)
            tile<WT, true>(mr, nv, xr, K, wp, ps, bp, yr, N, ncols);
          else
            tile<WT, false>(mr, nv, xr, K, wp, ps, bp, yr, N, ncols);
          r += mr;
        }
      }
    }
  });
}
#endif

}  // namespace

void matmul_reference(const float* x, int m, const PackedMatrix& w, const float* bias, float* y) {
  const int N = w.rows(), K = w.cols();
  for (int i = 0; i < m; ++i)
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) acc = std::fma(x[static_cast<std::size_t>(i) * K + k], w.at(n, k), acc);
      y[static_cast<std::size_t>(i) * N + n] = bias ? acc + bias[n] : acc;
    }
}

void matmul(const float* x, int m, const PackedMatrix& w, const float* bias, float* y, ThreadPool& pool,
            KernelMode mode) {
  if (m == 0) return;
#if BW_AVX512
  if (w.dtype() == DType::BF16)
    matmul_avx512<uint16_t>(x, m, w, w.bf16(), bias, y, pool, mode);
  else
    matmul_avx512<float>(x, m, w, w.f32(), bias, y, pool, mode);
#else
  (void)pool;
  (void)mode;
  matmul_reference(x, m, w, bias, y);
#endif
}

// ---------------------------------------------------------------------------------
// Pointwise kernels

void rmsnorm(const float* x, const float* weight, float* y, int m, int dim, float eps) {
  for (int r = 0; r < m; ++r) {
    const float* xr = x + static_cast<std::size_t>(r) * dim;
    float* yr = y + static_cast<std::size_t>(r) * dim;
    const float inv = 1.0f / std::sqrt(sumsq_canonical(xr, dim) / static_cast<float>(dim) + eps);
    for (int i = 0; i < dim; ++i) yr[i] = (xr[i] * inv) * weight[i];
  }
}

void rope(float* vec, int heads, int head_dim, int pos, const float* inv_freq) {
  const int half = head_dim / 2;
  float c[256], s[256];
  for (int i = 0; i < half; ++i) {
    const float angle = static_cast<float>(pos) * inv_freq[i];
    c[i] = std::cos(angle);
    s[i] = std::sin(angle);
  }
  for (int h = 0; h < heads; ++h) {
    float* v = vec + static_cast<std::size_t>(h) * head_dim;
    for (int i = 0; i < half; ++i) {
      const float x1 = v[i], x2 = v[i + half];
      v[i] = x1 * c[i] - x2 * s[i];
      v[i + half] = x2 * c[i] + x1 * s[i];
    }
  }
}

void swiglu(const float* gate_up, float* h, int m, int inter) {
  for (int r = 0; r < m; ++r) {
    const float* g = gate_up + static_cast<std::size_t>(r) * 2 * inter;
    const float* u = g + inter;
    float* o = h + static_cast<std::size_t>(r) * inter;
    for (int i = 0; i < inter; ++i) o[i] = g[i] * (1.0f / (1.0f + std::exp(-g[i]))) * u[i];
  }
}

// ---------------------------------------------------------------------------------
// Paged attention

namespace {

struct Partial {
  float max;
  float sum;
};

// Attend one query head to keys [begin, end) of its sequence, accumulating in key order.
// Writes unnormalized output to acc[head_dim] and returns (max, sum of exp).
Partial attend_range(const float* qh, const AttentionRow& row, int kvh, int begin, int end, const float* k_cache,
                     const float* v_cache, const KVLayout& L, float scale, float* acc, std::vector<float>& scores) {
  const int hd = L.head_dim;
  scores.resize(static_cast<std::size_t>(end - begin));
  float mx = -INFINITY;
  for (int j = begin; j < end; ++j) {
    const int blk = row.block_table[j / L.block_size];
    const float s = dot_canonical(qh, k_cache + L.offset(blk, kvh, j % L.block_size), hd) * scale;
    scores[j - begin] = s;
    mx = std::max(mx, s);
  }
  float sum = 0.0f;
  for (int d = 0; d < hd; ++d) acc[d] = 0.0f;
  for (int j = begin; j < end; ++j) {
    const float e = std::exp(scores[j - begin] - mx);
    sum += e;
    const float* v = v_cache + L.offset(row.block_table[j / L.block_size], kvh, j % L.block_size);
#if BW_AVX512
    const __m512 eb = _mm512_set1_ps(e);
    for (int d = 0; d < hd; d += 16)
      _mm512_storeu_ps(acc + d, _mm512_fmadd_ps(eb, _mm512_loadu_ps(v + d), _mm512_loadu_ps(acc + d)));
#else
    for (int d = 0; d < hd; ++d) acc[d] = std::fma(e, v[d], acc[d]);
#endif
  }
  return {mx, sum};
}

}  // namespace

void paged_attention(const float* q, int q_stride, const std::vector<AttentionRow>& rows, const float* k_cache,
                     const float* v_cache, const KVLayout& L, int heads, float* out, int out_stride,
                     ThreadPool& pool, KernelMode mode, std::vector<std::vector<float>>& /*scratch*/) {
  const int hd = L.head_dim;
  const int group = heads / L.kv_heads;
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  const std::size_t items = rows.size() * heads;
  int max_len = 0;
  for (const auto& r : rows) max_len = std::max(max_len, r.pos + 1);

  const bool split = mode == KernelMode::ShapeSpecialized && items < static_cast<std::size_t>(2 * pool.size()) &&
                     max_len >= 256;
  if (!split) {
    pool.parallel_for(items, [&](std::size_t b, std::size_t e) {
      thread_local std::vector<float> scores;
      for (std::size_t it = b; it < e; ++it) {
        const int r = static_cast<int>(it / heads), h = static_cast<int>(it % heads);
        const float* qh = q + static_cast<std::size_t>(r) * q_stride + static_cast<std::size_t>(h) * hd;
        float* o = out + static_cast<std::size_t>(r) * out_stride + static_cast<std::size_t>(h) * hd;
        const Partial p = attend_range(qh, rows[r], h / group, 0, rows[r].pos + 1, k_cache, v_cache, L, scale, o,
                                       scores);
        const float inv = 1.0f / p.sum;
        for (int d = 0; d < hd; ++d) o[d] *= inv;
      }
    });
    return;
  }

  // Split-KV: each (row, head) is cut into `chunks` key ranges processed in parallel,
  // then merged with log-sum-exp rescaling.
  const int chunks = pool.size();
  std::vector<float> part_acc(items * chunks * hd);
  std::vector<Partial> parts(items * chunks);
  pool.parallel_for(items * chunks, [&](std::size_t b, std::size_t e) {
    thread_local std::vector<float> scores;
    for (std::size_t u = b; u < e; ++u) {
      const std::size_t it = u / chunks;
      const int c = static_cast<int>(u % chunks);
      const int r = static_cast<int>(it / heads), h = static_cast<int>(it % heads);
      const int n = rows[r].pos + 1;
      const int lo = n * c / chunks, hi = n * (c + 1) / chunks;
      float* acc = part_acc.data() + u * hd;
      if (lo == hi) {
        parts[u] = {-INFINITY, 0.0f};
        for (int d = 0; d < hd; ++d) acc[d] = 0.0f;
        continue;
      }
      const float* qh = q + static_cast<std::size_t>(r) * q_stride + static_cast<std::size_t>(h) * hd;
      parts[u] = attend_range(qh, rows[r], h / group, lo, hi, k_cache, v_cache, L, scale, acc, scores);
    }
  });
  for (std::size_t it = 0; it < items; ++it) {
    const int r = static_cast<int>(it / heads), h = static_cast<int>(it % heads);
    float mx = -INFINITY;
    for (int c = 0; c < chunks; ++c) mx = std::max(mx, parts[it * chunks + c].max);
    float* o = out + static_cast<std::size_t>(r) * out_stride + static_cast<std::size_t>(h) * hd;
    for (int d = 0; d < hd; ++d) o[d] = 0.0f;
    float sum = 0.0f;
    for (int c = 0; c < chunks; ++c) {
      const Partial& p = parts[it * chunks + c];
      if (p.sum == 0.0f) continue;
      const float w = std::exp(p.max - mx);
      sum += p.sum * w;
      const float* acc = part_acc.data() + (it * chunks + c) * hd;
      for (int d = 0; d < hd; ++d) o[d] += acc[d] * w;
    }
    const float inv = 1.0f / sum;
    for (int d = 0; d < hd; ++d) o[d] *= inv;
  }
}

}  // namespace bw
