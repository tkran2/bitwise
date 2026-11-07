// Deterministic trace simulator.
//
// Each trace is a seeded random workload: requests with shared and unique prompt
// prefixes, greedy and sampled decoding, staggered arrivals, cancellations, and an
// engine configuration drawn at random (cache size small enough to force preemption,
// token budgets small enough to force chunked prefill, prefix caching on or off,
// speculative decoding on or off, continuous or static batching, 1 to 4 threads).
//
// The oracle: every request that was not cancelled must produce exactly the tokens it
// produces when run alone on an idle engine, and the logits behind every one of those
// tokens must match bit for bit (compared through a hash of the raw float bits). The
// engine must also finish every trace and return every KV block.
//
//   bitwise_sim --traces 2000                 check the engine
//   bitwise_sim --sweep-faults --traces 300   inject each known bug and report which are caught
//   bitwise_sim --kernels shape               run with shape-specialized kernels

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "engine.h"

using namespace bw;

namespace {

struct Options {
  int traces = 500;
  uint64_t seed = 1;
  KernelMode kernels = KernelMode::Invariant;
  int fault = 0;
  bool sweep = false;
  bool verbose = false;
  int only_trace = -1;
  std::string model_path, draft_path;
};

struct Request {
  std::vector<int32_t> prompt;
  SamplingParams params;
  int arrival_step;
  int cancel_step;  // -1: never
};

struct Outcome {
  int64_t requests_checked = 0;
  int64_t token_mismatch = 0;
  int64_t fingerprint_mismatch = 0;
  int64_t hangs = 0;
  int64_t leaks = 0;
  int64_t preemptions = 0;
  int64_t prefix_hits = 0;
  int64_t spec_accepted = 0;
  int first_bad_trace = -1;
  int64_t violations() const { return token_mismatch + fingerprint_mismatch + hangs + leaks; }
};

ModelConfig sim_config() {
  ModelConfig c;
  c.arch = "qwen2";
  c.vocab_size = 97;
  c.hidden = 64;
  c.layers = 2;
  c.heads = 4;
  c.kv_heads = 2;
  c.head_dim = 16;
  c.intermediate = 512;
  c.rope_theta = 10000.0f;
  c.rms_eps = 1e-6f;
  c.qkv_bias = true;
  c.eos_id = 3;
  return c;
}

class Simulator {
 public:
  Simulator(const Model& target, const Model& draft, const Options& opt) : target_(target), draft_(draft), opt_(opt) {}

  Outcome run(int fault, KernelMode kernels, int traces, int only = -1) {
    Outcome out;
    for (int t = 0; t < traces; ++t) {
      if (only >= 0 && t != only) continue;
      const int64_t before = out.violations();
      run_trace(opt_.seed * 1000003ull + t, fault, kernels, out);
      if (out.violations() > before && out.first_bad_trace < 0) out.first_bad_trace = t;
    }
    return out;
  }

 private:
  struct Ref {
    std::vector<int32_t> tokens;
    uint64_t fingerprint;
  };

  const Ref& reference(const Request& r, KernelMode kernels) {
    std::string key(reinterpret_cast<const char*>(r.prompt.data()), r.prompt.size() * 4);
    key += std::to_string(r.params.temperature) + "/" + std::to_string(r.params.seed) + "/" +
           std::to_string(r.params.max_new_tokens) + "/" + std::to_string(static_cast<int>(kernels));
    auto it = refs_.find(key);
    if (it != refs_.end()) return it->second;
    EngineConfig cfg;
    cfg.prefix_caching = false;
    cfg.kernels = kernels;
    cfg.threads = 1;
    cfg.num_blocks = static_cast<int>(r.prompt.size() + r.params.max_new_tokens) / kBlockSize + 2;
    cfg.max_batched_tokens = 4096;
    Engine e(target_, nullptr, cfg);
    auto id = e.add_request(r.prompt, r.params);
    while (!e.idle()) e.step();
    return refs_[key] = {e.result(id).tokens, e.result(id).logits_fingerprint};
  }

  void run_trace(uint64_t seed, int fault, KernelMode kernels, Outcome& out) {
    std::mt19937_64 rng(seed);
    auto uni = [&](int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); };
    auto coin = [&](double p) { return std::uniform_real_distribution<double>(0, 1)(rng) < p; };
    const int V = target_.config().vocab_size;

    std::vector<std::vector<int32_t>> prefixes(3);
    for (auto& p : prefixes) {
      p.resize(coin(0.5) ? kBlockSize * uni(1, 3) : uni(8, 48));  // block-aligned prefixes are the edge case
      for (auto& x : p) x = uni(4, V - 1);
    }
    std::vector<Request> reqs(uni(1, 10));
    int max_need = 0;
    for (std::size_t i = 0; i < reqs.size(); ++i) {
      Request& r = reqs[i];
      if (i > 0 && coin(0.15)) {
        r.prompt = reqs[uni(0, static_cast<int>(i) - 1)].prompt;  // exact repeat of an earlier prompt
      } else {
        if (coin(0.5)) r.prompt = prefixes[uni(0, 2)];
        const int extra = uni(r.prompt.empty() ? 1 : 0, 40);
        for (int j = 0; j < extra; ++j) r.prompt.push_back(uni(0, V - 1));
      }
      r.params.max_new_tokens = uni(1, 24);
      r.params.temperature = coin(0.5) ? 0.0f : std::uniform_real_distribution<float>(0.3f, 1.5f)(rng);
      r.params.seed = rng();
      r.params.ignore_eos = coin(0.5);
      r.arrival_step = coin(0.3) ? 0 : uni(0, 30);
      r.cancel_step = coin(0.15) ? r.arrival_step + uni(0, 20) : -1;
      max_need = std::max(max_need, static_cast<int>(r.prompt.size()) + r.params.max_new_tokens);
    }

    EngineConfig cfg;
    cfg.kernels = kernels;
    cfg.fault = fault;
    cfg.threads = uni(1, 4);
    cfg.max_batched_tokens = uni(4, 64);
    cfg.max_num_seqs = uni(1, 8);
    cfg.prefix_caching = coin(0.7);
    cfg.policy = coin(0.8) ? BatchPolicy::Continuous : BatchPolicy::Static;
    cfg.spec_tokens = coin(0.4) ? uni(1, 4) : 0;
    const int min_blocks = (max_need + cfg.spec_tokens + 1 + kBlockSize - 1) / kBlockSize;
    cfg.num_blocks = min_blocks + uni(0, 12);
    cfg.draft_num_blocks = cfg.num_blocks;

    Engine e(target_, cfg.spec_tokens ? &draft_ : nullptr, cfg);
    std::vector<int64_t> ids(reqs.size(), -1);
    int added = 0;
    bool hung = true;
    for (int step = 0; step < 20000; ++step) {
      for (std::size_t i = 0; i < reqs.size(); ++i) {
        if (reqs[i].arrival_step == step) {
          ids[i] = e.add_request(reqs[i].prompt, reqs[i].params);
          ++added;
        }
        if (reqs[i].cancel_step == step && ids[i] >= 0) e.cancel(ids[i]);
      }
      if (added == static_cast<int>(reqs.size()) && e.idle()) {
        hung = false;
        break;
      }
      e.step();
    }
    if (hung) {
      ++out.hangs;
      if (opt_.verbose)
        std::printf("  trace %llu: engine did not finish (threads=%d budget=%d max_seqs=%d blocks=%d prefix=%d "
                    "policy=%d spec=%d)\n%s",
                    (unsigned long long)seed, cfg.threads, cfg.max_batched_tokens, cfg.max_num_seqs, cfg.num_blocks,
                    cfg.prefix_caching, (int)cfg.policy, cfg.spec_tokens, e.debug_state().c_str());
      return;
    }
    if (e.free_blocks() != cfg.num_blocks) ++out.leaks;
    out.preemptions += e.stats().preemptions;
    out.prefix_hits += e.stats().prefix_hit_tokens;
    out.spec_accepted += e.stats().spec_accepted;

    for (std::size_t i = 0; i < reqs.size(); ++i) {
      const RequestResult& got = e.result(ids[i]);
      if (got.cancelled) continue;
      // Speculative sampling is exact in distribution, not token for token.
      if (cfg.spec_tokens && reqs[i].params.temperature > 0.0f) continue;
      const Ref& ref = reference(reqs[i], kernels);
      ++out.requests_checked;
      if (got.tokens != ref.tokens) {
        ++out.token_mismatch;
        if (opt_.verbose) std::printf("  trace %llu req %zu: tokens differ\n", (unsigned long long)seed, i);
      } else if (got.logits_fingerprint != ref.fingerprint) {
        ++out.fingerprint_mismatch;
        if (opt_.verbose) std::printf("  trace %llu req %zu: same tokens, different logits\n", (unsigned long long)seed, i);
      }
    }
  }

  const Model& target_;
  const Model& draft_;
  Options opt_;
  std::map<std::string, Ref> refs_;
};

void print(const char* label, const Outcome& o, int traces) {
  std::printf("%-34s traces=%d requests=%lld token_mismatch=%lld logit_mismatch=%lld hangs=%lld leaks=%lld"
              " | preemptions=%lld prefix_hit_tokens=%lld spec_accepted=%lld",
              label, traces, (long long)o.requests_checked, (long long)o.token_mismatch,
              (long long)o.fingerprint_mismatch, (long long)o.hangs, (long long)o.leaks, (long long)o.preemptions,
              (long long)o.prefix_hits, (long long)o.spec_accepted);
  if (o.first_bad_trace >= 0) std::printf(" | first failing trace=%d", o.first_bad_trace);
  std::printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&] { return std::string(i + 1 < argc ? argv[++i] : ""); };
    if (a == "--traces") opt.traces = std::stoi(next());
    else if (a == "--seed") opt.seed = std::stoull(next());
    else if (a == "--kernels") opt.kernels = next() == "shape" ? KernelMode::ShapeSpecialized : KernelMode::Invariant;
    else if (a == "--fault") opt.fault = std::stoi(next());
    else if (a == "--sweep-faults") opt.sweep = true;
    else if (a == "--verbose") opt.verbose = true;
    else if (a == "--trace") opt.only_trace = std::stoi(next());
    else if (a == "--model") opt.model_path = next();
    else if (a == "--draft") opt.draft_path = next();
    else if (a == "--help" || a == "-h") {
      std::printf(
          "usage: bitwise_sim [options]\n"
          "  --traces N        randomized workloads to run (default 500)\n"
          "  --seed S          base seed (default 1)\n"
          "  --kernels MODE    invariant (default) or shape\n"
          "  --fault F         inject test-only fault F (1-5)\n"
          "  --sweep-faults    run clean, every fault, and shape kernels; report which are caught\n"
          "  --trace N         run only trace N, with --verbose for engine state on failure\n"
          "  --verbose         print each violation\n"
          "  --model PATH      use a converted checkpoint instead of the random test model\n"
          "  --draft PATH      draft checkpoint for speculative decoding (default: --model)\n");
      return 0;
    }
    else {
      std::fprintf(stderr, "unknown option %s (see --help)\n", a.c_str());
      return 2;
    }
  }

  std::unique_ptr<Model> target, draft;
  if (!opt.model_path.empty()) {
    target = Model::load(opt.model_path);
    draft = Model::load(opt.draft_path.empty() ? opt.model_path : opt.draft_path);
  } else {
    // The draft is the target's weights at different precision: close enough that most
    // drafted tokens are accepted, different enough that some are rejected.
    target = Model::random(sim_config(), 42, DType::BF16);
    draft = Model::random(sim_config(), 42, DType::F32);
  }
  Simulator sim(*target, *draft, opt);

  if (!opt.sweep) {
    Outcome o = sim.run(opt.fault, opt.kernels, opt.traces, opt.only_trace);
    print(opt.kernels == KernelMode::Invariant ? "invariant kernels" : "shape-specialized kernels", o, opt.traces);
    return o.violations() ? 1 : 0;
  }

  const char* names[] = {"", "prefix cache covers last block", "rng keyed by step", "spec keeps rejected draft",
                         "registers partial block", "cancel leaks blocks"};
  Outcome clean = sim.run(0, KernelMode::Invariant, opt.traces);
  print("no fault (invariant kernels)", clean, opt.traces);
  int caught = 0, total = 0;
  for (int f = 1; f < kNumFaults; ++f) {
    Outcome o = sim.run(f, KernelMode::Invariant, opt.traces);
    print(names[f], o, opt.traces);
    caught += o.violations() > 0;
    ++total;
  }
  Outcome shape = sim.run(0, KernelMode::ShapeSpecialized, opt.traces);
  print("shape-specialized kernels", shape, opt.traces);
  caught += shape.violations() > 0;
  ++total;
  std::printf("caught %d of %d injected faults; clean run violations: %lld\n", caught, total,
              (long long)clean.violations());
  return clean.violations() ? 1 : 0;
}
