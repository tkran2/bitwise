#include "engine.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace bw {

namespace {

uint64_t mix64(uint64_t z) {
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

uint64_t hash_floats(uint64_t h, const float* data, int n) {
  const auto* bytes = reinterpret_cast<const unsigned char*>(data);
  const std::size_t len = static_cast<std::size_t>(n) * sizeof(float);
  std::size_t i = 0;
  for (; i + 8 <= len; i += 8) {
    uint64_t w;
    std::memcpy(&w, bytes + i, 8);
    h = (h ^ w) * 1099511628211ull;
  }
  for (; i < len; ++i) h = (h ^ bytes[i]) * 1099511628211ull;
  return mix64(h);
}

constexpr uint64_t kStreamSample = 0, kStreamAccept = 1, kStreamResidual = 2, kStreamDraft = 3;

}  // namespace

double counter_uniform(uint64_t seed, int64_t position, uint64_t stream) {
  const uint64_t z = mix64(mix64(seed ^ 0x6A09E667F3BCC909ull) ^ mix64(static_cast<uint64_t>(position) * 4 + stream));
  return static_cast<double>(z >> 11) * 0x1.0p-53;
}

int32_t argmax_row(const float* logits, int vocab) {
  int32_t best = 0;
  for (int i = 1; i < vocab; ++i)
    if (logits[i] > logits[best]) best = i;
  return best;
}

namespace {

// Unnormalized softmax weights in vocab order; returns their (sequential) total.
double softmax_weights(const float* logits, int vocab, float temperature, std::vector<float>& w) {
  w.resize(vocab);
  float mx = logits[0];
  for (int i = 1; i < vocab; ++i) mx = std::max(mx, logits[i]);
  double total = 0.0;
  for (int i = 0; i < vocab; ++i) {
    w[i] = std::exp((logits[i] - mx) / temperature);
    total += w[i];
  }
  return total;
}

int32_t inverse_cdf(const std::vector<float>& w, double total, double u) {
  const double target = u * total;
  double cum = 0.0;
  int32_t last = 0;
  for (std::size_t i = 0; i < w.size(); ++i) {
    if (w[i] <= 0.0f) continue;
    cum += w[i];
    last = static_cast<int32_t>(i);
    if (cum > target) return last;
  }
  return last;
}

}  // namespace

int32_t sample_row(const float* logits, int vocab, float temperature, double u, std::vector<float>& scratch) {
  const double total = softmax_weights(logits, vocab, temperature, scratch);
  return inverse_cdf(scratch, total, u);
}

struct Engine::Sequence {
  int64_t id = 0;
  std::vector<int32_t> tokens;
  int prompt_len = 0;
  int computed = 0;  // tokens whose K/V are in the target cache
  std::vector<int32_t> blocks;
  std::vector<uint64_t> hashes;  // hash chain of this sequence's full, registered blocks
  int draft_computed = 0;
  std::vector<int32_t> draft_blocks;
  SamplingParams params;
  bool running = false;
  bool done = false;
  std::vector<int32_t> proposals;
  std::vector<float> draft_logits;  // [k][vocab], kept for stochastic acceptance

  int generated() const { return static_cast<int>(tokens.size()) - prompt_len; }
};

Engine::Engine(const Model& target, const Model* draft, const EngineConfig& cfg)
    : target_(target),
      draft_(cfg.spec_tokens > 0 ? draft : nullptr),
      cfg_(cfg),
      pool_(cfg.threads),
      kv_(target.config(), cfg.num_blocks, kBlockSize),
      blocks_(cfg.num_blocks, cfg.prefix_caching),
      runner_(target, kv_, pool_, cfg.kernels),
      start_(std::chrono::steady_clock::now()) {
  if (cfg_.spec_tokens > 0) {
    if (!draft) throw std::invalid_argument("speculative decoding needs a draft model");
    if (draft->config().vocab_size != target.config().vocab_size)
      throw std::invalid_argument("draft and target vocabularies differ");
    draft_kv_ = std::make_unique<KVPool>(draft->config(), cfg.draft_num_blocks, kBlockSize);
    draft_blocks_ = std::make_unique<BlockManager>(cfg.draft_num_blocks, false);
    draft_runner_ = std::make_unique<Runner>(*draft, *draft_kv_, pool_, cfg.kernels);
  }
}

Engine::~Engine() = default;

double Engine::now() const {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
}

int64_t Engine::add_request(const std::vector<int32_t>& prompt, const SamplingParams& params) {
  const ModelConfig& c = target_.config();
  if (prompt.empty()) throw std::invalid_argument("empty prompt");
  for (int32_t t : prompt)
    if (t < 0 || t >= c.vocab_size) throw std::invalid_argument("token id out of range");
  const int64_t worst = static_cast<int64_t>(prompt.size()) + params.max_new_tokens + cfg_.spec_tokens + 1;
  if (worst > static_cast<int64_t>(cfg_.num_blocks) * kBlockSize ||
      (draft_ && worst > static_cast<int64_t>(cfg_.draft_num_blocks) * kBlockSize))
    throw std::invalid_argument("request can never fit in the KV cache");
  auto s = std::make_unique<Sequence>();
  s->id = next_id_++;
  s->tokens = prompt;
  s->prompt_len = static_cast<int>(prompt.size());
  s->params = params;
  results_[s->id].arrival = now();
  waiting_.push_back(s.get());
  const int64_t id = s->id;
  seqs_[id] = std::move(s);
  return id;
}

bool Engine::ensure_blocks(std::vector<int32_t>& blocks, int tokens, BlockManager& mgr) {
  while (static_cast<int>(blocks.size()) * kBlockSize < tokens) {
    auto b = mgr.allocate();
    if (!b) return false;
    blocks.push_back(*b);
  }
  return true;
}

void Engine::release_all(Sequence& s) {
  for (int32_t b : s.blocks) blocks_.release(b);
  s.blocks.clear();
  s.hashes.clear();
  s.computed = 0;
  if (draft_blocks_)
    for (int32_t b : s.draft_blocks) draft_blocks_->release(b);
  s.draft_blocks.clear();
  s.draft_computed = 0;
}

void Engine::preempt(Sequence& s) {
  release_all(s);
  s.running = false;
  running_.erase(std::find(running_.begin(), running_.end(), &s));
  auto pos = std::find_if(waiting_.begin(), waiting_.end(), [&](Sequence* w) { return w->id > s.id; });
  waiting_.insert(pos, &s);
  ++results_[s.id].preemptions;
  ++stats_.preemptions;
}

void Engine::admit_prefix(Sequence& s) {
  if (!blocks_.prefix_caching()) return;
  // Never reuse the block holding the last token: its logits must be computed.
  int full = (static_cast<int>(s.tokens.size()) - 1) / kBlockSize;
  if (cfg_.fault == kPrefixIncludesLastBlock) full = static_cast<int>(s.tokens.size()) / kBlockSize;
  uint64_t parent = 0;
  for (int i = 0; i < full; ++i) {
    const int32_t* toks = s.tokens.data() + static_cast<std::size_t>(i) * kBlockSize;
    const uint64_t h = block_hash(parent, toks);
    auto hit = blocks_.lookup(h, toks);
    if (!hit) break;
    s.blocks.push_back(*hit);
    s.hashes.push_back(h);
    parent = h;
  }
  s.computed = static_cast<int>(s.blocks.size()) * kBlockSize;
  results_[s.id].prefix_cached_tokens += s.computed;
  stats_.prefix_hit_tokens += s.computed;
}

void Engine::register_full_blocks(Sequence& s) {
  int full = s.computed / kBlockSize;
  if (cfg_.fault == kRegisterPartialBlock) full = std::min<int>((s.computed + kBlockSize - 1) / kBlockSize, s.blocks.size());
  for (int i = static_cast<int>(s.hashes.size()); i < full; ++i) {
    const int32_t* toks = s.tokens.data() + static_cast<std::size_t>(i) * kBlockSize;
    const uint64_t h = block_hash(i ? s.hashes[i - 1] : 0, toks);
    blocks_.register_full(s.blocks[i], h, toks);
    s.hashes.push_back(h);
  }
}

void Engine::finish(Sequence& s, bool cancelled) {
  release_all(s);
  if (s.running) running_.erase(std::find(running_.begin(), running_.end(), &s));
  else {
    auto it = std::find(waiting_.begin(), waiting_.end(), &s);
    if (it != waiting_.end()) waiting_.erase(it);
  }
  s.running = false;
  s.done = true;
  RequestResult& r = results_[s.id];
  r.finished = true;
  r.cancelled = cancelled;
  r.finish = now();
}

void Engine::cancel(int64_t id) {
  auto it = seqs_.find(id);
  if (it == seqs_.end() || it->second->done) return;
  if (cfg_.fault == kCancelLeaksBlocks) {
    it->second->blocks.clear();
    it->second->hashes.clear();
  }
  finish(*it->second, true);
  seqs_.erase(it);
}

Engine::Sequence* Engine::pick_victim(const std::vector<Sequence*>& protect) {
  for (auto it = running_.rbegin(); it != running_.rend(); ++it)
    if (std::find(protect.begin(), protect.end(), *it) == protect.end()) return *it;
  return nullptr;
}

int Engine::append_token(Sequence& s, int32_t token, const float* row, std::vector<StepEvent>& events) {
  RequestResult& r = results_[s.id];
  s.tokens.push_back(token);
  r.tokens.push_back(token);
  const double t = now();
  if (r.first_token < 0) r.first_token = t;
  r.token_times.push_back(t);
  if (cfg_.fingerprint_logits) r.logits_fingerprint = hash_floats(r.logits_fingerprint, row, target_.config().vocab_size);
  ++stats_.generated_tokens;
  if (events.empty() || events.back().id != s.id) events.push_back({s.id, {}, false});
  events.back().tokens.push_back(token);
  const bool eos = !s.params.ignore_eos && token == target_.config().eos_id;
  if (eos || s.generated() >= s.params.max_new_tokens) {
    events.back().finished = true;
    return 1;
  }
  return 0;
}

void Engine::draft_propose(std::vector<Work>& spec) {
  const int V = draft_->config().vocab_size;
  int max_k = 0;
  for (auto& w : spec) {
    max_k = std::max(max_k, w.spec_k);
    w.seq->proposals.clear();
    w.seq->draft_logits.resize(static_cast<std::size_t>(w.spec_k) * V);
  }
  std::vector<BatchRow> rows;
  for (int j = 0; j < max_k; ++j) {
    rows.clear();
    std::vector<Work*> active;
    for (auto& w : spec) {
      if (w.spec_k <= j) continue;
      Sequence& s = *w.seq;
      const int len = static_cast<int>(s.tokens.size());
      if (j == 0) {
        for (int p = s.draft_computed; p < len; ++p)
          rows.push_back({s.tokens[p], p, s.draft_blocks.data(), p == len - 1});
      } else {
        rows.push_back({s.proposals[j - 1], len + j - 1, s.draft_blocks.data(), true});
      }
      active.push_back(&w);
    }
    const double t0 = now();
    int nlog = 0;
    const float* logits = draft_runner_->forward(rows, &nlog);
    stats_.draft_seconds += now() - t0;
    stats_.draft_rows += static_cast<int64_t>(rows.size());
    for (int a = 0; a < static_cast<int>(active.size()); ++a) {
      Sequence& s = *active[a]->seq;
      const float* row = logits + static_cast<std::size_t>(a) * V;
      const int len = static_cast<int>(s.tokens.size());
      int32_t d;
      if (s.params.temperature <= 0.0f) {
        d = argmax_row(row, V);
      } else {
        std::memcpy(s.draft_logits.data() + static_cast<std::size_t>(j) * V, row, V * sizeof(float));
        d = sample_row(row, V, s.params.temperature, counter_uniform(s.params.seed, len + j, kStreamDraft), probs_a_);
      }
      s.proposals.push_back(d);
      s.draft_computed = len + j;
    }
  }
}

std::vector<StepEvent> Engine::step() {
  std::vector<StepEvent> events;
  ++stats_.steps;
  const bool spec_on = draft_ != nullptr;
  int budget = cfg_.max_batched_tokens;
  std::vector<Work> plan;
  std::vector<Sequence*> scheduled;

  // 1. Running sequences, oldest first: decode steps and unfinished chunked prefills.
  const std::vector<Sequence*> snapshot = running_;
  for (Sequence* s : snapshot) {
    if (!s->running) continue;  // preempted earlier in this loop
    const int need = static_cast<int>(s->tokens.size()) - s->computed;
    int k = 0;
    if (spec_on && need == 1) k = std::max(0, std::min(cfg_.spec_tokens, s->params.max_new_tokens - s->generated() - 1));
    int n = need + k;
    if (n == 0) continue;
    if (n > budget) {
      if (budget <= 0) continue;
      if (need > 1) {  // chunk the rest of the prefill
        n = budget;
        k = 0;
      } else {  // draft fewer tokens so the verify batch fits the budget
        k = budget - 1;
        n = budget;
      }
    }
    auto fits = [&] {
      return ensure_blocks(s->blocks, s->computed + n, blocks_) &&
             (k == 0 || ensure_blocks(s->draft_blocks, static_cast<int>(s->tokens.size()) + k - 1, *draft_blocks_));
    };
    bool ok = fits();
    while (!ok) {
      std::vector<Sequence*> protect = scheduled;
      protect.push_back(s);
      Sequence* victim = pick_victim(protect);
      if (!victim) break;
      preempt(*victim);
      ok = fits();
    }
    if (!ok) {
      preempt(*s);
      continue;
    }
    plan.push_back({s, n, k});
    scheduled.push_back(s);
    budget -= n;
  }

  // 2. Admit waiting requests in arrival order.
  const bool may_admit = cfg_.policy == BatchPolicy::Continuous || running_.empty();
  while (may_admit && !waiting_.empty() && static_cast<int>(running_.size()) < cfg_.max_num_seqs) {
    if (budget <= 0 && cfg_.policy == BatchPolicy::Continuous) break;
    Sequence* s = waiting_.front();
    if (s->blocks.empty()) admit_prefix(*s);
    const int need = static_cast<int>(s->tokens.size()) - s->computed;
    const int n = std::min(need, std::max(budget, 0));
    if (!ensure_blocks(s->blocks, s->computed + n, blocks_)) {
      results_[s->id].prefix_cached_tokens -= s->computed;  // the hit is retried on the next attempt
      stats_.prefix_hit_tokens -= s->computed;
      release_all(*s);
      break;
    }
    waiting_.pop_front();
    s->running = true;
    running_.insert(std::upper_bound(running_.begin(), running_.end(), s,
                                     [](Sequence* a, Sequence* b) { return a->id < b->id; }),
                    s);
    if (n > 0) {
      plan.push_back({s, n, 0});
      budget -= n;
    }
  }
  if (plan.empty()) return events;

  // 3. Draft proposals for speculative decode steps.
  std::vector<Work> spec;
  for (const auto& w : plan)
    if (w.spec_k > 0) spec.push_back(w);
  if (!spec.empty()) draft_propose(spec);

  // 4. One target forward pass over every scheduled token.
  std::vector<BatchRow> rows;
  for (const auto& w : plan) {
    Sequence& s = *w.seq;
    const int len = static_cast<int>(s.tokens.size());
    if (w.spec_k > 0) {
      rows.push_back({s.tokens[len - 1], len - 1, s.blocks.data(), true});
      for (int j = 0; j < w.spec_k; ++j) rows.push_back({s.proposals[j], len + j, s.blocks.data(), true});
      stats_.decode_rows += w.spec_k + 1;
    } else {
      for (int p = s.computed; p < s.computed + w.n; ++p)
        rows.push_back({s.tokens[p], p, s.blocks.data(), p == len - 1});
      if (w.n == 1 && s.computed == len - 1 && s.generated() > 0)
        ++stats_.decode_rows;
      else
        stats_.prefill_tokens += w.n;
    }
  }
  const double t0 = now();
  int nlog = 0;
  const float* logits = runner_.forward(rows, &nlog);
  stats_.forward_seconds += now() - t0;

  // 5. Sample, verify and bookkeep.
  const int V = target_.config().vocab_size;
  int cursor = 0;
  std::vector<Sequence*> finished;
  for (const auto& w : plan) {
    Sequence& s = *w.seq;
    const int len0 = static_cast<int>(s.tokens.size());
    bool done = false;
    if (w.spec_k > 0) {
      const float* L = logits + static_cast<std::size_t>(cursor) * V;
      cursor += w.spec_k + 1;
      const bool greedy = s.params.temperature <= 0.0f;
      int accepted = 0;
      bool rejected = false;
      for (int i = 0; i < w.spec_k && !done; ++i) {
        const float* row = L + static_cast<std::size_t>(i) * V;
        const int32_t d = s.proposals[i];
        if (greedy) {
          const int32_t t = argmax_row(row, V);
          done = append_token(s, cfg_.fault == kSpecKeepsRejectedDraft ? d : t, row, events);
          if (t != d) {
            rejected = true;
            break;
          }
        } else {
          const double tp = softmax_weights(row, V, s.params.temperature, probs_a_);
          const float* qrow = s.draft_logits.data() + static_cast<std::size_t>(i) * V;
          const double tq = softmax_weights(qrow, V, s.params.temperature, probs_b_);
          const double p = probs_a_[d] / tp, q = probs_b_[d] / tq;
          if (counter_uniform(s.params.seed, len0 + i, kStreamAccept) * q < p) {
            done = append_token(s, d, row, events);
          } else {
            double rt = 0.0;
            for (int v = 0; v < V; ++v) {
              const double r = std::max(0.0, probs_a_[v] / tp - probs_b_[v] / tq);
              probs_a_[v] = static_cast<float>(r);
              rt += probs_a_[v];
            }
            const int32_t t = inverse_cdf(probs_a_, rt, counter_uniform(s.params.seed, len0 + i, kStreamResidual));
            done = append_token(s, t, row, events);
            rejected = true;
            break;
          }
        }
        ++accepted;
      }
      if (!rejected && !done) {
        const float* row = L + static_cast<std::size_t>(w.spec_k) * V;
        const int32_t t = greedy ? argmax_row(row, V)
                                 : sample_row(row, V, s.params.temperature,
                                              counter_uniform(s.params.seed, len0 + w.spec_k, kStreamSample), probs_a_);
        done = append_token(s, t, row, events);
      }
      const int matched = accepted;  // drafted tokens the target agreed with
      stats_.spec_proposed += w.spec_k;
      stats_.spec_accepted += matched;
      s.computed = len0 + matched;
      s.draft_computed = std::min(s.draft_computed, len0 + std::min(matched, w.spec_k - 1));
    } else {
      s.computed += w.n;
      if (s.computed == len0) {
        const float* row = logits + static_cast<std::size_t>(cursor++) * V;
        const int32_t t = s.params.temperature <= 0.0f
                              ? argmax_row(row, V)
                              : sample_row(row, V, s.params.temperature,
                                           counter_uniform(s.params.seed, cfg_.fault == kRngKeyedByStep ? stats_.steps : len0,
                                                           kStreamSample),
                                           probs_a_);
        done = append_token(s, t, row, events);
      }
    }
    register_full_blocks(s);
    if (done) finished.push_back(&s);
  }
  for (Sequence* s : finished) {
    finish(*s, false);
    seqs_.erase(s->id);
  }
  stats_.evictions = blocks_.evictions();
  return events;
}

std::string Engine::debug_state() const {
  std::string out = "free_blocks=" + std::to_string(blocks_.available()) +
                    (draft_blocks_ ? " draft_free=" + std::to_string(draft_blocks_->available()) : "") + "\n";
  auto line = [&](const char* tag, const Sequence* s) {
    out += std::string(tag) + " id=" + std::to_string(s->id) + " len=" + std::to_string(s->tokens.size()) +
           " prompt=" + std::to_string(s->prompt_len) + " computed=" + std::to_string(s->computed) +
           " blocks=" + std::to_string(s->blocks.size()) + " draft_computed=" + std::to_string(s->draft_computed) +
           " draft_blocks=" + std::to_string(s->draft_blocks.size()) + " gen=" + std::to_string(s->generated()) + "/" +
           std::to_string(s->params.max_new_tokens) + "\n";
  };
  for (const Sequence* s : running_) line("running", s);
  for (const Sequence* s : waiting_) line("waiting", s);
  return out;
}

}  // namespace bw
