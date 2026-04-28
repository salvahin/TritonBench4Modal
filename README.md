# TritonBench-T on Modal

Run the **TritonBench-T** benchmark (translate PyTorch ops → Triton kernels)
end-to-end on a single, **cheap** GPU rented from [Modal](https://modal.com).
Designed to be cloned, configured in ~5 minutes, and shared with students.

- **GPU:** NVIDIA **T4** by default — Modal's cheapest tier at **$0.59 / hour**
  (per Modal's [pricing page](https://modal.com/pricing)). T4 has compute
  capability 7.5, which Triton supports.
- **Benchmark:** [TritonBench](https://github.com/thunlp/TritonBench), track **T**
  (166 PyTorch operators with Alpaca-formatted instructions). Cloned and patched
  inside the container — you don't have to install or fix anything locally.
- **What you get back:** call-accuracy %, execution-accuracy %, and a
  geometric speedup vs. the golden PyTorch baseline.

## Cost expectation

Order-of-magnitude budget for a *full* 166-op run on a single T4: a few dollars.
Quick smoke tests (`--limit 5`) cost cents. Modal bills per second; the GPU is
released as soon as `evaluate` returns.

LLM costs are separate and depend on your provider. With `--limit 5` on
Anthropic Claude Sonnet, generation is well under $0.05.

---

## 1. One-time setup

```bash
# 1. Modal client + auth
pip install -r requirements-local.txt
modal setup                       # opens a browser to link your account

# 2. Pick ONE provider and add its key as a Modal secret named "tritonbench-llm".
#    The function reads whichever key is present.
modal secret create tritonbench-llm ANTHROPIC_API_KEY=sk-ant-...
# or
modal secret create tritonbench-llm OPENAI_API_KEY=sk-...
```

**Already have a secret with a different name?** Point the app at it once with
an environment variable instead of editing code:

```bash
export TRITONBENCH_LLM_SECRET=openai-secret    # or whatever yours is called
```

You only need the secret if you want this project to **generate** the Triton
predictions for you. If you already have a `predictions.jsonl` from somewhere
else, skip it and use `evaluate_only` (see below).

---

## 2. Run it

### End-to-end (generate + evaluate)

```bash
# Smoke test — first 5 ops, costs pennies, finishes in a few minutes
modal run modal_app.py::main --limit 5

# Full run — all 166 ops, defaults to Anthropic Claude Sonnet 4.6
modal run modal_app.py::main

# Use OpenAI instead
modal run modal_app.py::main --provider openai --model gpt-4o-mini

# Use the "complex" instruction variant
modal run modal_app.py::main --dataset comp
```

### Bring your own predictions

```bash
modal run modal_app.py::evaluate_only --predictions ./my_predictions.jsonl
```

`my_predictions.jsonl` must have one JSON object per line. Each object needs:

- the **instruction** text from the Alpaca dataset, *exactly as given* — the
  evaluator parses it to find the matching reference operator (it greps for
  the substring between `"Functional Description: "` and
  `"Wrapper Entry Information:"`)
- a `"predict"` field with the model's full reply, ideally wrapped in a
  ```` ```python ... ``` ```` fence

Example line (truncated):

```json
{"instruction": "You are an expert in Trion programming...\nFunctional Description: Computes the absolute value...\nWrapper Entry Information: abs(input_tensor, out=None) -> Tensor...", "predict": "```python\nimport torch\nimport triton\nimport triton.language as tl\n\n@triton.jit\ndef _abs_kernel(...):\n    ...\n\ndef abs(input_tensor, out=None):\n    ...\n```"}
```

The Alpaca source files live inside the container at
`/opt/TritonBench/data/TritonBench_T_simp_alpac_v1.json` and
`/opt/TritonBench/data/TritonBench_T_comp_alpac_v1.json`. The simplest way to
match instructions exactly is to call `generate_only` once with `--limit 0`
and reuse those instruction strings.

### Generate only

```bash
modal run modal_app.py::generate_only --provider anthropic --model claude-sonnet-4-6
```

Predictions land in the persistent volume `tritonbench-t-data`.

---

## 3. Inspect / download artifacts

Each run writes to the `tritonbench-t-data` volume under `results/`:

```
results/
├── call_acc/         # one .py per operator that passed phase 1 (then pruned by phase 2)
└── perf_results/     # per-op timing JSON consumed by phase 3
```

Browse / download from your laptop:

```bash
modal volume ls   tritonbench-t-data results/
modal volume get  tritonbench-t-data results/ ./local-results/
```

---

## 4. Switch GPU tier (optional)

T4 is the cheapest GPU but has no bf16 tensor cores, so a handful of operators
that rely on bf16 will fail at phase 1. To rerun on something beefier, edit
`DEFAULT_GPU` near the top of `modal_app.py` to e.g. `"L4"` ($0.80/hr) or
`"A10"` ($1.10/hr) and rerun.

---

## What the pipeline does (under the hood)

1. **Image build** — clones the upstream
   [TritonBench](https://github.com/thunlp/TritonBench) repo and patches three
   small things so the eval scripts run unattended:
   - `EVAL/eval_T/0_call_acc.py` — points `statis_path` at `*.jsonl` (upstream
     has `.json`), `py_folder` at `data/TritonBench_T_v1/` (upstream points to
     the G dataset), and `py_interpreter` at `sys.executable`.
   - `EVAL/eval_T/1_exe_acc.py` — same `py_interpreter` fix.
   - `performance_metrics/perf_T/run_bench/multiprocess_gpu_run.py` — sets
     `gpu_count = 1` (upstream assumes 8).
2. **Generate** — runs in a CPU container, calls the configured LLM in parallel
   threads, writes one JSON line per operator.
3. **Evaluate** — single GPU container, runs the three TritonBench-T phases
   sequentially:
   - Phase 1 (`0_call_acc.py::call_4file`) — concatenates the predicted module
     with the golden test driver, executes it; the `.py` files of the operators
     that **ran** end up in `results/call_acc/`.
   - Phase 2 (`1_exe_acc.py::execute_4folder`) — re-runs each survivor and the
     reference side-by-side, deletes any whose `stdout` differs.
   - Phase 3 (`perf_T/run_bench/*` + `2_efficiency.py`) — benchmarks each
     remaining op against the golden PyTorch timings and prints a summary
     speedup.

Final summary is printed as JSON, e.g.

```json
{
  "total_predictions": 166,
  "phase1_call_acc": { "passed": 88, "rate": 53.01 },
  "phase2_exec_acc": { "passed": 71, "rate": 42.77 },
  "phase3_efficiency": { "speedup_vs_pytorch": 0.83, "raw_output_tail": "..." }
}
```

---

## References

- TritonBench paper: <https://arxiv.org/pdf/2502.14752>
- TritonBench repo: <https://github.com/thunlp/TritonBench>
- Modal docs: <https://modal.com/docs/guide>
- Modal GPU pricing: <https://modal.com/pricing>
