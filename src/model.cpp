#include "model.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <map>
#include <sstream>
#include <stdexcept>

namespace bw {

namespace {

struct TensorInfo {
  DType dtype;
  std::size_t offset;
  std::vector<int64_t> shape;
  std::size_t numel() const {
    std::size_t n = 1;
    for (auto d : shape) n *= static_cast<std::size_t>(d);
    return n;
  }
};

class MappedFile {
 public:
  explicit MappedFile(const std::string& path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("cannot open " + path);
    struct stat st {};
    ::fstat(fd_, &st);
    size_ = static_cast<std::size_t>(st.st_size);
    data_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (data_ == MAP_FAILED) throw std::runtime_error("mmap failed for " + path);
  }
  ~MappedFile() {
    if (data_ && data_ != MAP_FAILED) ::munmap(data_, size_);
    if (fd_ >= 0) ::close(fd_);
  }
  const char* data() const { return static_cast<const char*>(data_); }
  std::size_t size() const { return size_; }

 private:
  int fd_ = -1;
  void* data_ = nullptr;
  std::size_t size_ = 0;
};

std::vector<float> to_f32(const void* p, DType dt, std::size_t n) {
  std::vector<float> out(n);
  if (dt == DType::F32)
    std::memcpy(out.data(), p, n * 4);
  else
    for (std::size_t i = 0; i < n; ++i) out[i] = bf16_to_f32(static_cast<const uint16_t*>(p)[i]);
  return out;
}

uint64_t splitmix64(uint64_t& s) {
  uint64_t z = (s += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

uint16_t f32_to_bf16(float f) {  // round to nearest even
  uint32_t b;
  std::memcpy(&b, &f, 4);
  b += 0x7FFF + ((b >> 16) & 1);
  return static_cast<uint16_t>(b >> 16);
}

}  // namespace

std::unique_ptr<Model> Model::load(const std::string& path) {
  MappedFile file(path);
  const char* base = file.data();
  const char* end_marker = static_cast<const char*>(memmem(base, std::min<std::size_t>(file.size(), 1 << 22), "\nend\n", 5));
  if (std::strncmp(base, "BITWISE 1\n", 10) != 0 || !end_marker) throw std::runtime_error(path + " is not a bitwise model file");
  const std::size_t header_len = static_cast<std::size_t>(end_marker - base) + 5;
  const std::size_t data_start = (header_len + 4095) / 4096 * 4096;

  auto model = std::make_unique<Model>();
  ModelConfig& c = model->cfg_;
  std::map<std::string, TensorInfo> tensors;
  std::istringstream in(std::string(base + 10, header_len - 10));
  std::string key;
  bool tie = true;
  while (in >> key) {
    if (key == "end") break;
    if (key == "tensor") {
      std::string name, dt;
      TensorInfo t;
      int ndim;
      in >> name >> dt >> t.offset >> ndim;
      t.dtype = dt == "bf16" ? DType::BF16 : DType::F32;
      t.shape.resize(ndim);
      for (auto& d : t.shape) in >> d;
      tensors[name] = t;
    } else if (key == "arch") in >> c.arch;
    else if (key == "vocab_size") in >> c.vocab_size;
    else if (key == "hidden") in >> c.hidden;
    else if (key == "layers") in >> c.layers;
    else if (key == "heads") in >> c.heads;
    else if (key == "kv_heads") in >> c.kv_heads;
    else if (key == "head_dim") in >> c.head_dim;
    else if (key == "intermediate") in >> c.intermediate;
    else if (key == "rope_theta") in >> c.rope_theta;
    else if (key == "rms_eps") in >> c.rms_eps;
    else if (key == "qkv_bias") in >> c.qkv_bias;
    else if (key == "tie_embeddings") in >> tie;
    else if (key == "bos_id") in >> c.bos_id;
    else if (key == "eos_id") in >> c.eos_id;
    else {
      std::string skip;
      std::getline(in, skip);
    }
  }

  auto get = [&](const std::string& name) -> std::pair<const void*, const TensorInfo&> {
    auto it = tensors.find(name);
    if (it == tensors.end()) throw std::runtime_error("missing tensor " + name);
    const std::size_t bytes = it->second.numel() * (it->second.dtype == DType::BF16 ? 2 : 4);
    if (data_start + it->second.offset + bytes > file.size()) throw std::runtime_error("truncated tensor " + name);
    return {base + data_start + it->second.offset, it->second};
  };
  auto vec = [&](const std::string& name) {
    auto [p, t] = get(name);
    return to_f32(p, t.dtype, t.numel());
  };

  {
    auto [p, t] = get("model.embed_tokens.weight");
    model->embed_dtype_ = t.dtype;
    const std::size_t n = t.numel();
    if (t.dtype == DType::BF16)
      model->embed_bf16_.assign(static_cast<const uint16_t*>(p), static_cast<const uint16_t*>(p) + n);
    else
      model->embed_f32_.assign(static_cast<const float*>(p), static_cast<const float*>(p) + n);
  }
  const int H = c.hidden;
  for (int l = 0; l < c.layers; ++l) {
    const std::string pre = "model.layers." + std::to_string(l) + ".";
    LayerWeights lw;
    lw.attn_norm = vec(pre + "input_layernorm.weight");
    lw.mlp_norm = vec(pre + "post_attention_layernorm.weight");
    auto q = get(pre + "self_attn.q_proj.weight");
    auto k = get(pre + "self_attn.k_proj.weight");
    auto v = get(pre + "self_attn.v_proj.weight");
    lw.qkv = PackedMatrix({{q.first, c.q_dim()}, {k.first, c.kv_dim()}, {v.first, c.kv_dim()}}, q.second.dtype, H);
    if (c.qkv_bias) {
      for (const char* nm : {"q_proj", "k_proj", "v_proj"}) {
        auto b = vec(pre + "self_attn." + nm + ".bias");
        lw.qkv_bias.insert(lw.qkv_bias.end(), b.begin(), b.end());
      }
    }
    auto o = get(pre + "self_attn.o_proj.weight");
    lw.o = PackedMatrix(o.first, o.second.dtype, H, c.q_dim());
    auto g = get(pre + "mlp.gate_proj.weight");
    auto u = get(pre + "mlp.up_proj.weight");
    lw.gate_up = PackedMatrix({{g.first, c.intermediate}, {u.first, c.intermediate}}, g.second.dtype, H);
    auto d = get(pre + "mlp.down_proj.weight");
    lw.down = PackedMatrix(d.first, d.second.dtype, H, c.intermediate);
    model->layers_.push_back(std::move(lw));
  }
  model->final_norm_ = vec("model.norm.weight");
  auto head = get(tie ? "model.embed_tokens.weight" : "lm_head.weight");
  model->lm_head_ = PackedMatrix(head.first, head.second.dtype, c.vocab_size, H);
  model->finish_init();
  return model;
}

std::unique_ptr<Model> Model::random(const ModelConfig& cfg, uint64_t seed, DType dtype) {
  auto model = std::make_unique<Model>();
  model->cfg_ = cfg;
  const ModelConfig& c = model->cfg_;
  uint64_t s = seed;
  auto fill = [&](std::size_t n, float scale) {
    std::vector<float> v(n);
    for (auto& x : v) x = scale * (static_cast<float>(splitmix64(s) >> 40) / static_cast<float>(1 << 24) * 2.0f - 1.0f);
    return v;
  };
  auto mat = [&](int n, int k, float scale) {
    auto v = fill(static_cast<std::size_t>(n) * k, scale);
    if (dtype == DType::F32) return PackedMatrix(v.data(), DType::F32, n, k);
    std::vector<uint16_t> b(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) b[i] = f32_to_bf16(v[i]);
    return PackedMatrix(b.data(), DType::BF16, n, k);
  };
  auto norm = [&](int n) {
    auto v = fill(n, 0.1f);
    for (auto& x : v) x += 1.0f;
    return v;
  };
  const int H = c.hidden;
  auto emb = fill(static_cast<std::size_t>(c.vocab_size) * H, 1.0f);
  model->embed_dtype_ = DType::F32;
  model->embed_f32_ = emb;
  const float sh = 1.0f / std::sqrt(static_cast<float>(H));
  for (int l = 0; l < c.layers; ++l) {
    LayerWeights lw;
    lw.attn_norm = norm(H);
    lw.mlp_norm = norm(H);
    lw.qkv = mat(c.q_dim() + 2 * c.kv_dim(), H, 2.0f * sh);
    if (c.qkv_bias) lw.qkv_bias = fill(c.q_dim() + 2 * c.kv_dim(), 0.1f);
    lw.o = mat(H, c.q_dim(), sh);
    lw.gate_up = mat(2 * c.intermediate, H, 2.0f * sh);
    lw.down = mat(H, c.intermediate, 1.0f / std::sqrt(static_cast<float>(c.intermediate)));
    model->layers_.push_back(std::move(lw));
  }
  model->final_norm_ = norm(H);
  model->lm_head_ = mat(c.vocab_size, H, 4.0f * sh);
  model->finish_init();
  return model;
}

void Model::finish_init() {
  const int half = cfg_.head_dim / 2;
  inv_freq_.resize(half);
  for (int i = 0; i < half; ++i)
    inv_freq_[i] = 1.0f / std::pow(cfg_.rope_theta, static_cast<float>(2 * i) / static_cast<float>(cfg_.head_dim));
}

std::size_t Model::weight_bytes() const {
  std::size_t b = embed_f32_.size() * 4 + embed_bf16_.size() * 2 + lm_head_.bytes();
  for (const auto& l : layers_) b += l.qkv.bytes() + l.o.bytes() + l.gate_up.bytes() + l.down.bytes();
  return b;
}

void Model::embed(int token, float* out) const {
  const std::size_t off = static_cast<std::size_t>(token) * cfg_.hidden;
  if (embed_dtype_ == DType::F32)
    std::memcpy(out, embed_f32_.data() + off, cfg_.hidden * sizeof(float));
  else
    for (int i = 0; i < cfg_.hidden; ++i) out[i] = bf16_to_f32(embed_bf16_[off + i]);
}

KVPool::KVPool(const ModelConfig& cfg, int num_blocks, int block_size) {
  layout_.num_blocks = num_blocks;
  layout_.block_size = block_size;
  layout_.kv_heads = cfg.kv_heads;
  layout_.head_dim = cfg.head_dim;
  k_.assign(layout_.layer_floats() * cfg.layers, 0.0f);
  v_.assign(layout_.layer_floats() * cfg.layers, 0.0f);
}

Runner::Runner(const Model& model, KVPool& kv, ThreadPool& pool, KernelMode mode)
    : model_(model), kv_(kv), pool_(pool), mode_(mode) {}

void Runner::ensure_capacity(int rows) {
  if (rows <= capacity_) return;
  const ModelConfig& c = model_.config();
  capacity_ = rows;
  const std::size_t n = static_cast<std::size_t>(rows);
  x_.resize(n * c.hidden);
  xn_.resize(n * c.hidden);
  qkv_.resize(n * (c.q_dim() + 2 * c.kv_dim()));
  attn_.resize(n * c.q_dim());
  proj_.resize(n * c.hidden);
  gate_up_.resize(n * 2 * c.intermediate);
  mlp_h_.resize(n * c.intermediate);
}

const float* Runner::forward(const std::vector<BatchRow>& rows, int* num_logit_rows) {
  const ModelConfig& c = model_.config();
  const int n = static_cast<int>(rows.size());
  ensure_capacity(n);
  const int H = c.hidden, QD = c.q_dim(), KVD = c.kv_dim(), QKV = QD + 2 * KVD;
  const KVLayout& L = kv_.layout();

  for (int r = 0; r < n; ++r) model_.embed(rows[r].token, x_.data() + static_cast<std::size_t>(r) * H);

  std::vector<AttentionRow> att(n);
  for (int r = 0; r < n; ++r) att[r] = {rows[r].pos, rows[r].block_table};

  for (int l = 0; l < c.layers; ++l) {
    const LayerWeights& w = model_.layers_[l];
    rmsnorm(x_.data(), w.attn_norm.data(), xn_.data(), n, H, c.rms_eps);
    matmul(xn_.data(), n, w.qkv, w.qkv_bias.empty() ? nullptr : w.qkv_bias.data(), qkv_.data(), pool_, mode_);
    float* kc = kv_.k(l);
    float* vc = kv_.v(l);
    for (int r = 0; r < n; ++r) {
      float* q = qkv_.data() + static_cast<std::size_t>(r) * QKV;
      float* k = q + QD;
      const float* v = k + KVD;
      rope(q, c.heads, c.head_dim, rows[r].pos, model_.inv_freq_.data());
      rope(k, c.kv_heads, c.head_dim, rows[r].pos, model_.inv_freq_.data());
      const int blk = rows[r].block_table[rows[r].pos / L.block_size];
      const int slot = rows[r].pos % L.block_size;
      for (int h = 0; h < c.kv_heads; ++h) {
        std::memcpy(kc + L.offset(blk, h, slot), k + h * c.head_dim, c.head_dim * sizeof(float));
        std::memcpy(vc + L.offset(blk, h, slot), v + h * c.head_dim, c.head_dim * sizeof(float));
      }
    }
    paged_attention(qkv_.data(), QKV, att, kc, vc, L, c.heads, attn_.data(), QD, pool_, mode_);
    matmul(attn_.data(), n, w.o, nullptr, proj_.data(), pool_, mode_);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n) * H; ++i) x_[i] += proj_[i];
    rmsnorm(x_.data(), w.mlp_norm.data(), xn_.data(), n, H, c.rms_eps);
    matmul(xn_.data(), n, w.gate_up, nullptr, gate_up_.data(), pool_, mode_);
    swiglu(gate_up_.data(), mlp_h_.data(), n, c.intermediate);
    matmul(mlp_h_.data(), n, w.down, nullptr, proj_.data(), pool_, mode_);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n) * H; ++i) x_[i] += proj_[i];
  }

  int count = 0;
  for (const auto& r : rows) count += r.want_logits;
  *num_logit_rows = count;
  if (count == 0) return nullptr;
  last_.resize(static_cast<std::size_t>(count) * H);
  int j = 0;
  for (int r = 0; r < n; ++r)
    if (rows[r].want_logits)
      std::memcpy(last_.data() + static_cast<std::size_t>(j++) * H, x_.data() + static_cast<std::size_t>(r) * H,
                  H * sizeof(float));
  rmsnorm(last_.data(), model_.final_norm_.data(), last_.data(), count, H, c.rms_eps);
  logits_.resize(static_cast<std::size_t>(count) * c.vocab_size);
  matmul(last_.data(), count, model_.lm_head_, nullptr, logits_.data(), pool_, mode_);
  return logits_.data();
}

}  // namespace bw
