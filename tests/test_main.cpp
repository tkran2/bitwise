// Unit tests for kernels, the block manager, the model and the engine.
// Every equality check on floats here is bitwise (memcmp), not approximate.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "block_manager.h"
#include "engine.h"
#include "kernels.h"
#include "model.h"

using namespace bw;

namespace {

int g_failures = 0;
int g_checks = 0;

#define CHECK(cond)                                                           \
  do {                                                                        \
    ++g_checks;                                                               \
    if (!(cond)) {                                                            \
      ++g_failures;                                                           \
      std::fprintf(stderr, "  FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    }                                                                         \
  } while (0)

bool same_bits(const float* a, const float* b, std::size_t n) { return std::memcmp(a, b, n * sizeof(float)) == 0; }

std::vector<float> randvec(std::size_t n, std::mt19937& rng, float scale = 1.0f) {
  std::uniform_real_distribution<float> d(-scale, scale);
  std::vector<float> v(n);
  for (auto& x : v) x = d(rng);
  return v;
}

std::vector<uint16_t> to_bf16(const std::vector<float>& v) {
  std::vector<uint16_t> out(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) {
    uint32_t b;
    std::memcpy(&b, &v[i], 4);
    out[i] = static_cast<uint16_t>(b >> 16);
  }
  return out;
}

ModelConfig tiny_config(int vocab = 97) {
  ModelConfig c;
  c.arch = "qwen2";
  c.vocab_size = vocab;
  c.hidden = 64;
  c.layers = 2;
  c.heads = 4;
  c.kv_heads = 2;
  c.head_dim = 16;
  c.intermediate = 512;
  c.rope_theta = 10000.0f;
  c.rms_eps = 1e-6f;
  c.qkv_bias = true;
  c.eos_id = -1;
  return c;
}

// ---------------------------------------------------------------------------------

void test_matmul_matches_reference() {
  std::mt19937 rng(1);
  ThreadPool pool(4);
  for (DType dt : {DType::F32, DType::BF16}) {
    for (int m : {1, 2, 3, 4, 5, 7, 13, 50}) {
      for (int n : {1, 15, 16, 17, 100, 300}) {
        for (int k : {1, 16, 33, 64, 600}) {
          auto w = randvec(static_cast<std::size_t>(n) * k, rng);
          auto x = randvec(static_cast<std::size_t>(m) * k, rng);
          auto bias = randvec(n, rng);
          auto wb = to_bf16(w);
          PackedMatrix W = dt == DType::F32 ? PackedMatrix(w.data(), dt, n, k) : PackedMatrix(wb.data(), dt, n, k);
          std::vector<float> y(static_cast<std::size_t>(m) * n), ref(y.size());
          for (bool use_bias : {false, true}) {
            matmul(x.data(), m, W, use_bias ? bias.data() : nullptr, y.data(), pool, KernelMode::Invariant);
            matmul_reference(x.data(), m, W, use_bias ? bias.data() : nullptr, ref.data());
            CHECK(same_bits(y.data(), ref.data(), y.size()));
          }
        }
      }
    }
  }
}

void test_matmul_row_invariance() {
  std::mt19937 rng(2);
  const int n = 200, k = 1024, m = 9;
  auto w = to_bf16(randvec(static_cast<std::size_t>(n) * k, rng));
  PackedMatrix W(w.data(), DType::BF16, n, k);
  auto x = randvec(static_cast<std::size_t>(m) * k, rng);
  for (KernelMode mode : {KernelMode::Invariant, KernelMode::ShapeSpecialized}) {
    int mismatches = 0;
    for (int threads : {1, 3, 4}) {
      ThreadPool pool(threads);
      std::vector<float> batch(static_cast<std::size_t>(m) * n), single(n);
      matmul(x.data(), m, W, nullptr, batch.data(), pool, mode);
      for (int r = 0; r < m; ++r) {
        matmul(x.data() + static_cast<std::size_t>(r) * k, 1, W, nullptr, single.data(), pool, mode);
        mismatches += !same_bits(single.data(), batch.data() + static_cast<std::size_t>(r) * n, n);
      }
    }
    if (mode == KernelMode::Invariant)
      CHECK(mismatches == 0);
#if defined(__AVX512F__)
    else
      CHECK(mismatches > 0);  // the shape-specialized path really does change numerics
#endif
    // The portable scalar build has a single matmul path, so both modes agree there.
  }
}

void test_dot_canonical() {
  std::mt19937 rng(3);
  for (int n : {1, 5, 16, 31, 64, 100, 128}) {
    auto a = randvec(n, rng), b = randvec(n, rng);
    float lane[16] = {0};
    for (int i = 0; i < n; ++i) lane[i % 16] = std::fma(a[i], b[i], lane[i % 16]);
    float s8[8], s4[4];
    for (int i = 0; i < 8; ++i) s8[i] = lane[i] + lane[i + 8];
    for (int i = 0; i < 4; ++i) s4[i] = s8[i] + s8[i + 4];
    const float expect = (s4[0] + s4[2]) + (s4[1] + s4[3]);
    const float got = dot_canonical(a.data(), b.data(), n);
    CHECK(std::memcmp(&got, &expect, 4) == 0);
  }
}

void test_attention_invariance() {
  std::mt19937 rng(4);
  KVLayout L;
  L.num_blocks = 64;
  L.block_size = 16;
  L.kv_heads = 2;
  L.head_dim = 16;
  const int heads = 4;
  std::vector<float> kc = randvec(L.layer_floats(), rng), vc = randvec(L.layer_floats(), rng);
  std::vector<int32_t> table(40);
  for (int i = 0; i < 40; ++i) table[i] = (i * 7 + 3) % 64;
  const int q_stride = heads * L.head_dim;
  std::vector<int> positions = {0, 5, 17, 300, 301, 302, 511};
  auto q = randvec(positions.size() * q_stride, rng);
  for (KernelMode mode : {KernelMode::Invariant, KernelMode::ShapeSpecialized}) {
    ThreadPool pool(4);
    std::vector<AttentionRow> rows;
    for (int p : positions) rows.push_back({p, table.data()});
    std::vector<float> batch(positions.size() * q_stride), one(q_stride);
    paged_attention(q.data(), q_stride, rows, kc.data(), vc.data(), L, heads, batch.data(), q_stride, pool, mode);
    int mismatches = 0;
    for (std::size_t r = 0; r < positions.size(); ++r) {
      std::vector<AttentionRow> solo = {rows[r]};
      paged_attention(q.data() + r * q_stride, q_stride, solo, kc.data(), vc.data(), L, heads, one.data(), q_stride,
                      pool, mode);
      mismatches += !same_bits(one.data(), batch.data() + r * q_stride, q_stride);
    }
    if (mode == KernelMode::Invariant)
      CHECK(mismatches == 0);
    else
      CHECK(mismatches > 0);
  }
}

void test_block_manager() {
  BlockManager m(4, true);
  CHECK(m.available() == 4);
  auto a = m.allocate(), b = m.allocate();
  CHECK(a && b && *a != *b);
  int32_t toks[kBlockSize], other[kBlockSize];
  for (int i = 0; i < kBlockSize; ++i) {
    toks[i] = i;
    other[i] = i + 1;
  }
  const uint64_t h = block_hash(0, toks);
  m.register_full(*a, h, toks);
  CHECK(!m.lookup(h, other));  // same hash, different tokens: never a hit
  auto hit = m.lookup(h, toks);
  CHECK(hit && *hit == *a && m.refcount(*a) == 2);
  m.release(*a);
  m.release(*a);  // now cached but unreferenced
  CHECK(m.available() == 3);
  auto again = m.lookup(h, toks);  // revive from the evictable list
  CHECK(again && *again == *a);
  m.release(*a);
  // Exhaust: allocations take free blocks first, then evict the cached one.
  auto c = m.allocate(), d = m.allocate(), e = m.allocate();
  CHECK(c && d && e && m.evictions() == 1);
  CHECK(!m.lookup(h, toks));
  CHECK(!m.allocate());
}

// Logits for the last token of `tokens`, computed with the given chunking.
std::vector<float> logits_chunked(const Model& model, const std::vector<int32_t>& tokens, const std::vector<int>& chunks,
                                  KernelMode mode) {
  ThreadPool pool(4);
  KVPool kv(model.config(), 16);
  Runner runner(model, kv, pool, mode);
  std::vector<int32_t> table(16);
  for (int i = 0; i < 16; ++i) table[i] = 15 - i;
  int pos = 0;
  std::vector<float> out;
  for (int n : chunks) {
    std::vector<BatchRow> rows;
    for (int i = 0; i < n; ++i, ++pos) rows.push_back({tokens[pos], pos, table.data(), pos == (int)tokens.size() - 1});
    int nl = 0;
    const float* l = runner.forward(rows, &nl);
    if (nl) out.assign(l, l + model.config().vocab_size);
  }
  return out;
}

void test_model_chunking_invariance() {
  auto model = Model::random(tiny_config(), 7, DType::BF16);
  std::vector<int32_t> toks;
  for (int i = 0; i < 40; ++i) toks.push_back((i * 31 + 5) % 97);
  auto whole = logits_chunked(*model, toks, {40}, KernelMode::Invariant);
  auto steps = logits_chunked(*model, toks, std::vector<int>(40, 1), KernelMode::Invariant);
  auto mixed = logits_chunked(*model, toks, {3, 17, 1, 19}, KernelMode::Invariant);
  CHECK(same_bits(whole.data(), steps.data(), whole.size()));
  CHECK(same_bits(whole.data(), mixed.data(), whole.size()));
}

std::vector<int32_t> run_alone(const Model& m, const std::vector<int32_t>& prompt, const SamplingParams& sp,
                               uint64_t* fingerprint) {
  EngineConfig cfg;
  cfg.prefix_caching = false;
  Engine e(m, nullptr, cfg);
  auto id = e.add_request(prompt, sp);
  while (!e.idle()) e.step();
  *fingerprint = e.result(id).logits_fingerprint;
  return e.result(id).tokens;
}

void test_engine_batching_preemption_prefix_spec() {
  auto target = Model::random(tiny_config(), 11, DType::BF16);
  auto draft = Model::random(tiny_config(), 12, DType::BF16);
  std::vector<std::vector<int32_t>> prompts;
  std::vector<int32_t> shared;
  for (int i = 0; i < 37; ++i) shared.push_back((i * 13 + 1) % 97);
  for (int r = 0; r < 6; ++r) {
    auto p = shared;
    for (int i = 0; i < 3 + 5 * r; ++i) p.push_back((r * 17 + i * 3) % 97);
    prompts.push_back(p);
  }
  std::vector<SamplingParams> params;
  for (int r = 0; r < 6; ++r) {
    SamplingParams sp;
    sp.max_new_tokens = 20 + r;
    sp.temperature = r % 2 ? 0.8f : 0.0f;
    sp.seed = 100 + r;
    params.push_back(sp);
  }
  std::vector<std::vector<int32_t>> ref(6);
  std::vector<uint64_t> ref_fp(6);
  for (int r = 0; r < 6; ++r) ref[r] = run_alone(*target, prompts[r], params[r], &ref_fp[r]);

  struct Variant {
    const char* name;
    EngineConfig cfg;
  };
  std::vector<Variant> variants;
  {
    EngineConfig c;
    c.max_batched_tokens = 24;
    variants.push_back({"continuous+chunked+prefix", c});
  }
  {
    EngineConfig c;
    c.num_blocks = 9;  // forces preemption and recompute
    c.max_batched_tokens = 32;
    variants.push_back({"tight memory", c});
  }
  {
    EngineConfig c;
    c.spec_tokens = 3;
    variants.push_back({"speculative", c});
  }
  for (auto& v : variants) {
    Engine e(*target, v.cfg.spec_tokens ? draft.get() : nullptr, v.cfg);
    std::vector<int64_t> ids;
    for (int r = 0; r < 6; ++r) ids.push_back(e.add_request(prompts[r], params[r]));
    int guard = 0;
    while (!e.idle() && guard++ < 10000) e.step();
    for (int r = 0; r < 6; ++r) {
      const bool comparable = v.cfg.spec_tokens == 0 || params[r].temperature == 0.0f;
      if (!comparable) continue;  // speculative sampling matches in distribution, not token by token
      CHECK(e.result(ids[r]).tokens == ref[r]);
      CHECK(e.result(ids[r]).logits_fingerprint == ref_fp[r]);
    }
    CHECK(e.free_blocks() == v.cfg.num_blocks);
    if (std::string(v.name) == "tight memory") CHECK(e.stats().preemptions > 0);
    if (std::string(v.name) == "continuous+chunked+prefix") CHECK(e.stats().prefix_hit_tokens > 0);
    if (std::string(v.name) == "speculative") CHECK(e.stats().spec_proposed > 0);
  }
}

void test_spec_with_identical_draft_accepts_everything() {
  auto target = Model::random(tiny_config(), 21, DType::BF16);
  EngineConfig c;
  c.spec_tokens = 4;
  Engine e(*target, target.get(), c);
  SamplingParams sp;
  sp.max_new_tokens = 30;
  std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
  e.add_request(prompt, sp);
  while (!e.idle()) e.step();
  CHECK(e.stats().spec_accepted == e.stats().spec_proposed);
}

}  // namespace

int main() {
  struct T {
    const char* name;
    std::function<void()> fn;
  };
  std::vector<T> tests = {
      {"matmul matches scalar reference bitwise", test_matmul_matches_reference},
      {"matmul rows are batch invariant", test_matmul_row_invariance},
      {"canonical dot product", test_dot_canonical},
      {"paged attention is batch invariant", test_attention_invariance},
      {"block manager and prefix cache", test_block_manager},
      {"model logits independent of chunking", test_model_chunking_invariance},
      {"engine: batching, preemption, prefix cache, speculation", test_engine_batching_preemption_prefix_spec},
      {"speculation with identical draft accepts all", test_spec_with_identical_draft_accepts_everything},
  };
  for (auto& t : tests) {
    const int before = g_failures;
    t.fn();
    std::printf("%s %s\n", g_failures == before ? "[ ok ]" : "[FAIL]", t.name);
  }
  std::printf("%d checks, %d failures\n", g_checks, g_failures);
  return g_failures ? 1 : 0;
}
