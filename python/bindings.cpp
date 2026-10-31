// pybind11 bindings: just enough surface for the tests, the load generator and the
// reference comparisons in Python.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "engine.h"
#include "model.h"

namespace py = pybind11;
using namespace bw;

PYBIND11_MODULE(_bitwise, m) {
  py::enum_<KernelMode>(m, "KernelMode")
      .value("Invariant", KernelMode::Invariant)
      .value("ShapeSpecialized", KernelMode::ShapeSpecialized);
  py::enum_<BatchPolicy>(m, "BatchPolicy")
      .value("Continuous", BatchPolicy::Continuous)
      .value("Static", BatchPolicy::Static);

  py::class_<ModelConfig>(m, "ModelConfig")
      .def_readonly("vocab_size", &ModelConfig::vocab_size)
      .def_readonly("hidden", &ModelConfig::hidden)
      .def_readonly("layers", &ModelConfig::layers)
      .def_readonly("heads", &ModelConfig::heads)
      .def_readonly("kv_heads", &ModelConfig::kv_heads)
      .def_readonly("head_dim", &ModelConfig::head_dim)
      .def_readonly("eos_id", &ModelConfig::eos_id);

  py::class_<Model, std::shared_ptr<Model>>(m, "Model")
      .def_static("load", [](const std::string& p) { return std::shared_ptr<Model>(Model::load(p)); })
      .def_property_readonly("config", &Model::config)
      .def_property_readonly("weight_bytes", &Model::weight_bytes)
      .def("logits", [](const Model& model, const std::vector<int32_t>& tokens, KernelMode mode, int threads) {
            // Logits at every position of one sequence, computed as a single prefill.
            const int n = static_cast<int>(tokens.size());
            ThreadPool pool(threads);
            KVPool kv(model.config(), (n + 15) / 16);
            Runner runner(model, kv, pool, mode);
            std::vector<int32_t> table((n + 15) / 16);
            for (int i = 0; i < static_cast<int>(table.size()); ++i) table[i] = i;
            std::vector<BatchRow> rows;
            for (int i = 0; i < n; ++i) rows.push_back({tokens[i], i, table.data(), true});
            int nl = 0;
            const float* l = runner.forward(rows, &nl);
            const int V = model.config().vocab_size;
            py::array_t<float> out({n, V});
            std::memcpy(out.mutable_data(), l, sizeof(float) * n * V);
            return out;
          },
          py::arg("tokens"), py::arg("mode") = KernelMode::Invariant, py::arg("threads") = 4);

  py::class_<SamplingParams>(m, "SamplingParams")
      .def(py::init<>())
      .def(py::init([](float temperature, uint64_t seed, int max_new_tokens, bool ignore_eos) {
             SamplingParams p;
             p.temperature = temperature;
             p.seed = seed;
             p.max_new_tokens = max_new_tokens;
             p.ignore_eos = ignore_eos;
             return p;
           }),
           py::arg("temperature") = 0.0f, py::arg("seed") = 0, py::arg("max_new_tokens") = 32,
           py::arg("ignore_eos") = false);

  py::class_<EngineConfig>(m, "EngineConfig")
      .def(py::init<>())
      .def_readwrite("num_blocks", &EngineConfig::num_blocks)
      .def_readwrite("draft_num_blocks", &EngineConfig::draft_num_blocks)
      .def_readwrite("max_batched_tokens", &EngineConfig::max_batched_tokens)
      .def_readwrite("max_num_seqs", &EngineConfig::max_num_seqs)
      .def_readwrite("prefix_caching", &EngineConfig::prefix_caching)
      .def_readwrite("kernels", &EngineConfig::kernels)
      .def_readwrite("policy", &EngineConfig::policy)
      .def_readwrite("threads", &EngineConfig::threads)
      .def_readwrite("spec_tokens", &EngineConfig::spec_tokens)
      .def_readwrite("fingerprint_logits", &EngineConfig::fingerprint_logits);

  py::class_<RequestResult>(m, "RequestResult")
      .def_readonly("tokens", &RequestResult::tokens)
      .def_readonly("finished", &RequestResult::finished)
      .def_readonly("cancelled", &RequestResult::cancelled)
      .def_readonly("logits_fingerprint", &RequestResult::logits_fingerprint)
      .def_readonly("preemptions", &RequestResult::preemptions)
      .def_readonly("prefix_cached_tokens", &RequestResult::prefix_cached_tokens)
      .def_readonly("arrival", &RequestResult::arrival)
      .def_readonly("first_token", &RequestResult::first_token)
      .def_readonly("finish", &RequestResult::finish)
      .def_readonly("token_times", &RequestResult::token_times);

  py::class_<EngineStats>(m, "EngineStats")
      .def_readonly("steps", &EngineStats::steps)
      .def_readonly("prefill_tokens", &EngineStats::prefill_tokens)
      .def_readonly("decode_rows", &EngineStats::decode_rows)
      .def_readonly("generated_tokens", &EngineStats::generated_tokens)
      .def_readonly("preemptions", &EngineStats::preemptions)
      .def_readonly("prefix_hit_tokens", &EngineStats::prefix_hit_tokens)
      .def_readonly("spec_proposed", &EngineStats::spec_proposed)
      .def_readonly("spec_accepted", &EngineStats::spec_accepted)
      .def_readonly("draft_rows", &EngineStats::draft_rows)
      .def_readonly("forward_seconds", &EngineStats::forward_seconds)
      .def_readonly("draft_seconds", &EngineStats::draft_seconds);

  py::class_<StepEvent>(m, "StepEvent")
      .def_readonly("id", &StepEvent::id)
      .def_readonly("tokens", &StepEvent::tokens)
      .def_readonly("finished", &StepEvent::finished);

  py::class_<Engine>(m, "Engine")
      .def(py::init([](std::shared_ptr<Model> target, std::shared_ptr<Model> draft, const EngineConfig& cfg) {
             return std::make_unique<Engine>(*target, draft ? draft.get() : nullptr, cfg);
           }),
           py::arg("target"), py::arg("draft") = nullptr, py::arg("config") = EngineConfig(),
           py::keep_alive<1, 2>(), py::keep_alive<1, 3>())
      .def("add_request", &Engine::add_request)
      .def("cancel", &Engine::cancel)
      .def("step", &Engine::step, py::call_guard<py::gil_scoped_release>())
      .def("idle", &Engine::idle)
      .def("now", &Engine::now)
      .def_property_readonly("num_running", &Engine::num_running)
      .def_property_readonly("num_waiting", &Engine::num_waiting)
      .def_property_readonly("free_blocks", &Engine::free_blocks)
      .def("result", &Engine::result, py::return_value_policy::reference_internal)
      .def_property_readonly("stats", &Engine::stats);
}
