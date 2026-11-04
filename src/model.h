#pragma once
// Decoder-only transformer (Llama / Qwen2 family): weights and the batched forward pass.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "kernels.h"
#include "thread_pool.h"

namespace bw {

struct ModelConfig {
  std::string arch = "llama";  // "llama" or "qwen2" (adds Q/K/V biases)
  int vocab_size = 0;
  int hidden = 0;
  int layers = 0;
  int heads = 0;
  int kv_heads = 0;
  int head_dim = 0;
  int intermediate = 0;
  float rope_theta = 10000.0f;
  float rms_eps = 1e-5f;
  bool qkv_bias = false;
  int bos_id = -1;
  int eos_id = -1;

  int q_dim() const { return heads * head_dim; }
  int kv_dim() const { return kv_heads * head_dim; }
};

struct LayerWeights {
  std::vector<float> attn_norm, mlp_norm, qkv_bias;
  PackedMatrix qkv, o, gate_up, down;
};

class Model {
 public:
  // Load a converted checkpoint (see tools/convert.py).
  static std::unique_ptr<Model> load(const std::string& path);
  // Random weights for tests: small, deterministic in `seed`.
  static std::unique_ptr<Model> random(const ModelConfig& cfg, uint64_t seed, DType dtype = DType::F32);

  const ModelConfig& config() const { return cfg_; }
  std::size_t weight_bytes() const;

  // Embedding row for a token, as float.
  void embed(int token, float* out) const;

  ModelConfig cfg_;
  DType embed_dtype_ = DType::F32;
  std::vector<float> embed_f32_;
  std::vector<uint16_t> embed_bf16_;
  std::vector<LayerWeights> layers_;
  std::vector<float> final_norm_;
  PackedMatrix lm_head_;
  std::vector<float> inv_freq_;

 private:
  void finish_init();
};

// Paged KV storage for one model: per layer, [num_blocks][kv_heads][block_size][head_dim].
class KVPool {
 public:
  KVPool(const ModelConfig& cfg, int num_blocks, int block_size = 16);
  const KVLayout& layout() const { return layout_; }
  float* k(int layer) { return k_.data() + layer * layout_.layer_floats(); }
  float* v(int layer) { return v_.data() + layer * layout_.layer_floats(); }
  std::size_t bytes() const { return (k_.size() + v_.size()) * sizeof(float); }

 private:
  KVLayout layout_;
  std::vector<float> k_, v_;
};

// One token to run through the model. Rows of the same sequence must be contiguous and
// in increasing position order, and the block table must already cover `pos`.
struct BatchRow {
  int token;
  int pos;
  const int32_t* block_table;
  bool want_logits;
};

class Runner {
 public:
  Runner(const Model& model, KVPool& kv, ThreadPool& pool, KernelMode mode);

  // Runs the batch, writing K/V for every row into the cache. Returns logits for the rows
  // with want_logits, in row order, as a [count][vocab] array owned by the runner.
  const float* forward(const std::vector<BatchRow>& rows, int* num_logit_rows);

  KernelMode mode() const { return mode_; }

 private:
  void ensure_capacity(int rows);

  const Model& model_;
  KVPool& kv_;
  ThreadPool& pool_;
  KernelMode mode_;
  int capacity_ = 0;
  std::vector<float> x_, xn_, qkv_, attn_, proj_, gate_up_, mlp_h_, logits_, last_;
};

}  // namespace bw
