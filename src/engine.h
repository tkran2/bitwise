#pragma once
// Serving engine: continuous batching over a paged KV cache, chunked prefill, prefix
// caching, recompute-based preemption and speculative decoding.
//
// The engine never lets scheduling decisions leak into a request's numerics. With
// invariant kernels, the logits used to choose each token depend only on that
// request's tokens, and sampling draws from a counter-based generator keyed by
// (seed, position, stream). So a request produces the same tokens, bit for bit, whether
// it ran alone or in a crowded batch, was chunked, hit the prefix cache, was preempted
// and recomputed, or was decoded speculatively (greedy). `logits_fingerprint` hashes
// the exact logits behind every generated token so tests can check this directly.

#include <chrono>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "block_manager.h"
#include "model.h"

namespace bw {

struct SamplingParams {
  float temperature = 0.0f;  // 0 means greedy
  uint64_t seed = 0;
  int max_new_tokens = 32;
  bool ignore_eos = false;
};

enum class BatchPolicy : uint8_t {
  Continuous = 0,  // admit new requests every step, as budget and memory allow
  Static = 1,      // admit a new batch only after the previous batch has fully finished
};

struct EngineConfig {
  int num_blocks = 512;
  int draft_num_blocks = 512;
  int max_batched_tokens = 256;
  int max_num_seqs = 32;
  bool prefix_caching = true;
  KernelMode kernels = KernelMode::Invariant;
  BatchPolicy policy = BatchPolicy::Continuous;
  int threads = 4;
  int spec_tokens = 0;  // draft tokens per speculative step (requires a draft model)
  bool fingerprint_logits = true;
  int fault = 0;  // test only: deliberately injected bug (see Fault), for checking the simulator
};

// Bugs that the deterministic simulator should catch. Each is a one-line slip that a
// real implementation could plausibly make.
enum Fault : int {
  kNoFault = 0,
  kPrefixIncludesLastBlock = 1,  // prefix cache may cover the block holding the last prompt token
  kRngKeyedByStep = 2,           // sampling randomness keyed by engine step instead of position
  kSpecKeepsRejectedDraft = 3,   // on a greedy mismatch, append the draft's token instead of the target's
  kRegisterPartialBlock = 4,     // register a block in the prefix cache before it is full
  kCancelLeaksBlocks = 5,        // cancellation forgets to release the request's blocks
  kNumFaults = 6,
};

struct RequestResult {
  std::vector<int32_t> tokens;  // generated tokens
  bool finished = false;
  bool cancelled = false;
  uint64_t logits_fingerprint = 1469598103934665603ull;
  int preemptions = 0;
  int prefix_cached_tokens = 0;
  double arrival = 0, first_token = -1, finish = -1;  // seconds since engine start
  std::vector<double> token_times;
};

struct StepEvent {
  int64_t id;
  std::vector<int32_t> tokens;
  bool finished;
};

struct EngineStats {
  int64_t steps = 0;
  int64_t prefill_tokens = 0;
  int64_t decode_rows = 0;
  int64_t generated_tokens = 0;
  int64_t preemptions = 0;
  int64_t prefix_hit_tokens = 0;
  int64_t spec_proposed = 0;
  int64_t spec_accepted = 0;
  int64_t draft_rows = 0;
  int64_t evictions = 0;
  double forward_seconds = 0;
  double draft_seconds = 0;
};

class Engine {
 public:
  Engine(const Model& target, const Model* draft, const EngineConfig& cfg);
  ~Engine();

  int64_t add_request(const std::vector<int32_t>& prompt, const SamplingParams& params);
  void cancel(int64_t id);
  std::vector<StepEvent> step();
  bool idle() const { return running_.empty() && waiting_.empty(); }
  int num_running() const { return static_cast<int>(running_.size()); }
  int num_waiting() const { return static_cast<int>(waiting_.size()); }

  const RequestResult& result(int64_t id) const { return results_.at(id); }
  const EngineStats& stats() const { return stats_; }
  const EngineConfig& config() const { return cfg_; }
  int free_blocks() const { return blocks_.available(); }
  std::string debug_state() const;
  double now() const;

 private:
  struct Sequence;
  struct Work {
    Sequence* seq;
    int n;          // tokens to run through the target model this step
    int spec_k;     // >0: speculative verify of spec_k drafted tokens
  };

  bool ensure_blocks(std::vector<int32_t>& blocks, int tokens, BlockManager& mgr);
  void release_all(Sequence& s);
  void preempt(Sequence& s);
  void admit_prefix(Sequence& s);
  void register_full_blocks(Sequence& s);
  void finish(Sequence& s, bool cancelled);
  Sequence* pick_victim(const std::vector<Sequence*>& protect);
  void draft_propose(std::vector<Work>& spec);
  int append_token(Sequence& s, int32_t token, const float* logits_row, std::vector<StepEvent>& events);

  const Model& target_;
  const Model* draft_;
  EngineConfig cfg_;
  ThreadPool pool_;
  KVPool kv_;
  std::unique_ptr<KVPool> draft_kv_;
  BlockManager blocks_;
  std::unique_ptr<BlockManager> draft_blocks_;
  Runner runner_;
  std::unique_ptr<Runner> draft_runner_;

  std::map<int64_t, std::unique_ptr<Sequence>> seqs_;
  std::map<int64_t, RequestResult> results_;
  std::vector<Sequence*> running_;
  std::deque<Sequence*> waiting_;
  int64_t next_id_ = 0;
  EngineStats stats_;
  std::chrono::steady_clock::time_point start_;
  std::vector<float> probs_a_, probs_b_;
};

// Counter-based uniform in [0, 1): a pure function of (seed, position, stream).
double counter_uniform(uint64_t seed, int64_t position, uint64_t stream);

// Deterministic sampling helpers over a full logits row.
int32_t argmax_row(const float* logits, int vocab);
int32_t sample_row(const float* logits, int vocab, float temperature, double u, std::vector<float>& scratch);

}  // namespace bw
