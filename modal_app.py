"""
TritonBench-T on Modal — translate PyTorch ops to Triton kernels with an LLM,
then evaluate them on the cheapest available Modal GPU (NVIDIA T4).

Pipeline
--------
1. ``generate_predictions``  — calls a configured LLM provider on each Alpaca
   instruction in ``data/TritonBench_T_<simp|comp>_alpac_v1.json`` and writes a
   ``predictions.jsonl`` into a persistent Modal Volume.
2. ``evaluate``              — runs the three TritonBench-T phases on a GPU:
       phase 1: call accuracy   (does the generated module run at all?)
       phase 2: execution acc.  (does it produce the same outputs as PyTorch?)
       phase 3: efficiency      (speedup vs. the golden PyTorch baseline)

A single ``main`` local entrypoint chains them end-to-end.

Quick start (see README.md for full instructions):

    pip install modal
    modal setup
    modal secret create tritonbench-llm ANTHROPIC_API_KEY=sk-ant-...
    modal run modal_app.py                        # generate + evaluate
    modal run modal_app.py -- --limit 5           # smoke test on 5 ops
    modal run modal_app.py -- --predictions ./preds.jsonl   # bring your own
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import modal

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

APP_NAME = "tritonbench-t"
TRITONBENCH_REPO = "https://github.com/thunlp/TritonBench.git"

# Cheapest Modal GPU (compute capability 7.5 — Triton requires >= 7.0).
# Override at runtime via `--gpu A10` etc. on the local entrypoint.
DEFAULT_GPU = "T4"

VOLUME_NAME = "tritonbench-t-data"
DATA_DIR = "/data"           # mount point of the Modal Volume in the container
REPO_DIR = "/opt/TritonBench"

# Default model targets — students can override from the CLI.
DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL = "claude-sonnet-4-6"

# Name of the Modal Secret that holds your LLM API key(s) (e.g. ANTHROPIC_API_KEY,
# OPENAI_API_KEY). Override with an env var if your existing secret is named
# differently — no code edit required:
#     export TRITONBENCH_LLM_SECRET=openai-secret
LLM_SECRET_NAME = os.environ.get("TRITONBENCH_LLM_SECRET", "tritonbench-llm")

# --------------------------------------------------------------------------- #
# Image — patches TritonBench's hardcoded paths so the eval scripts run inside
# a clean container without any local-machine assumptions.
#
# Each `.run_commands(...)` argument becomes one Dockerfile RUN. Modal's legacy
# image builder treats every newline inside a single argument as a new
# Dockerfile instruction, so we keep each patch on a single line via `sed -i`.
# --------------------------------------------------------------------------- #

# 0_call_acc.py — wrong dataset filename (.json vs .jsonl), wrong test folder
# (G instead of T), and a hardcoded conda interpreter path.
PATCH_CALL_ACC = (
    f"""sed -i """
    f"""-e 's|^statis_path = .*|statis_path = "{REPO_DIR}/data/TritonBench_T_v1.jsonl"|' """
    f"""-e 's|^py_folder = .*|py_folder = "{REPO_DIR}/data/TritonBench_T_v1/"|' """
    f"""-e 's|^py_interpreter = .*|import sys; py_interpreter = sys.executable|' """
    f"""{REPO_DIR}/EVAL/eval_T/0_call_acc.py"""
)

# 1_exe_acc.py — same hardcoded conda interpreter path; gold_folder anchored
# to absolute path.
PATCH_EXE_ACC = (
    f"""sed -i """
    f"""-e 's|^gold_folder = .*|gold_folder = "{REPO_DIR}/data/TritonBench_T_v1/"|' """
    f"""-e 's|^py_interpreter = .*|import sys; py_interpreter = sys.executable|' """
    f"""{REPO_DIR}/EVAL/eval_T/1_exe_acc.py"""
)

# multiprocess_gpu_run.py — assumes 8 GPUs; we have one.
PATCH_PERF = (
    f"""sed -i 's|^gpu_count = .*|gpu_count = 1|' """
    f"""{REPO_DIR}/performance_metrics/perf_T/run_bench/multiprocess_gpu_run.py"""
)


image = (
    modal.Image.from_registry(
        # Python 3.12: TritonBench's eval scripts use PEP-701 nested-quote
        # f-strings, which require >= 3.12 to parse.
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git", "build-essential")
    .pip_install(
        "torch==2.5.1",
        "triton==3.1.0",
        "tqdm==4.66.5",
        "numpy<2",
        "anthropic>=0.40",
        "openai>=1.50",
    )
    .run_commands(f"git clone --depth 1 {TRITONBENCH_REPO} {REPO_DIR}")
    .run_commands(PATCH_CALL_ACC, PATCH_EXE_ACC, PATCH_PERF)
    # ProcessPoolExecutor pickles workers by qualified module name, so the
    # eval scripts must be importable as plain `call_acc` / `exe_acc` from any
    # subprocess. Module names can't start with a digit, so symlink them.
    .run_commands(
        f"ln -s {REPO_DIR}/EVAL/eval_T/0_call_acc.py {REPO_DIR}/EVAL/eval_T/call_acc.py",
        f"ln -s {REPO_DIR}/EVAL/eval_T/1_exe_acc.py {REPO_DIR}/EVAL/eval_T/exe_acc.py",
    )
)

app = modal.App(APP_NAME, image=image)
data_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


# --------------------------------------------------------------------------- #
# Generation — LLM-based PyTorch → Triton translation
# --------------------------------------------------------------------------- #

PROMPT_HEADER = (
    "You are an expert in Triton programming, capable of writing Triton kernels "
    "and wrapper functions based on functional descriptions and function "
    "parameters. The wrapper function must fully match the provided function "
    "signature.\n\n"
    "Output a single, self-contained Python module containing: (a) the necessary "
    "imports (torch, triton, triton.language as tl), (b) the Triton kernel(s), "
    "and (c) the wrapper function that the description specifies. Wrap the "
    "entire module in one ```python ... ``` fenced code block. Do NOT include "
    "any test code or example calls — tests will be appended separately."
)


def _load_alpaca(dataset: str) -> list[dict]:
    assert dataset in ("simp", "comp"), "dataset must be 'simp' or 'comp'"
    path = Path(REPO_DIR) / f"data/TritonBench_T_{dataset}_alpac_v1.json"
    return json.loads(path.read_text())


def _build_messages(item: dict) -> list[dict]:
    instr = item["instruction"]
    inp = item.get("input", "") or ""
    user = instr if not inp else f"{instr}\n\n{inp}"
    return [
        {"role": "system", "content": PROMPT_HEADER},
        {"role": "user", "content": user},
    ]


def _gen_anthropic(messages: list[dict], model: str) -> str:
    import anthropic

    client = anthropic.Anthropic()
    sys_prompt = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_msgs = [m for m in messages if m["role"] != "system"]
    resp = client.messages.create(
        model=model,
        max_tokens=8192,
        system=sys_prompt,
        messages=user_msgs,
    )
    return resp.content[0].text


def _gen_openai(messages: list[dict], model: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=8192,
    )
    return resp.choices[0].message.content


_GENERATORS = {"anthropic": _gen_anthropic, "openai": _gen_openai}


def _extract_code(text: str) -> str:
    """Strip Markdown code fences from an LLM reply; return raw Python source.

    Upstream's ``clear_code()`` only strips the opening ```` ```python ```` fence
    and leaves the closing ```` ``` ```` in place, which trips a ``SyntaxError``
    when the file is executed. So we hand the eval pipeline already-clean code.
    """
    import re

    s = text.strip()
    m = re.search(r"```(?:python|py)?\s*\n(.*?)\n```", s, re.DOTALL)
    if m:
        return m.group(1).strip() + "\n"
    # No closing fence (truncated reply, etc.) — drop only the opening one.
    s = re.sub(r"^```(?:python|py)?\s*\n?", "", s)
    s = re.sub(r"\n?```\s*$", "", s)
    return s.strip() + "\n"


@app.function(
    timeout=60 * 60 * 4,
    cpu=4,
    volumes={DATA_DIR: data_volume},
    secrets=[modal.Secret.from_name(LLM_SECRET_NAME)],
)
def generate_predictions(
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    dataset: str = "simp",
    output_path: str = "predictions.jsonl",
    limit: int | None = None,
    concurrency: int = 8,
) -> str:
    """Generate Triton translations for every entry in the Alpaca dataset.

    Returns the volume-relative path of the produced jsonl.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if provider not in _GENERATORS:
        raise ValueError(
            f"unknown provider {provider!r} — choose one of {list(_GENERATORS)}"
        )

    items = _load_alpaca(dataset)
    if limit:
        items = items[:limit]
    print(f"generating {len(items)} predictions with {provider}/{model}", flush=True)

    gen_fn = _GENERATORS[provider]

    def _do(idx_item):
        i, item = idx_item
        try:
            raw = gen_fn(_build_messages(item), model)
            code = _extract_code(raw)
        except Exception as e:  # noqa: BLE001
            code = f"# generation failed: {e}\n"
        return i, {"instruction": item["instruction"], "predict": code}

    results: list[dict | None] = [None] * len(items)
    done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(_do, (i, it)) for i, it in enumerate(items)]
        for fut in as_completed(futs):
            i, rec = fut.result()
            results[i] = rec
            done += 1
            if done % 5 == 0 or done == len(items):
                print(f"  {done}/{len(items)}", flush=True)

    out = Path(DATA_DIR) / output_path
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    data_volume.commit()
    print(f"wrote {out}", flush=True)
    return output_path


# --------------------------------------------------------------------------- #
# Evaluation — runs all three TritonBench-T phases on one GPU
# --------------------------------------------------------------------------- #


@app.function(
    gpu=DEFAULT_GPU,
    timeout=60 * 60 * 6,
    volumes={DATA_DIR: data_volume},
)
def evaluate(
    predictions_path: str = "predictions.jsonl",
    output_subdir: str = "results",
) -> dict:
    """Run TritonBench-T eval phases against an existing predictions.jsonl."""
    pred_full = Path(DATA_DIR) / predictions_path
    if not pred_full.exists():
        raise FileNotFoundError(f"predictions file not found in volume: {pred_full}")

    out_dir = Path(DATA_DIR) / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    call_acc_dir = out_dir / "call_acc"
    perf_results_dir = out_dir / "perf_results"

    if call_acc_dir.exists():
        shutil.rmtree(call_acc_dir)
    if perf_results_dir.exists():
        shutil.rmtree(perf_results_dir)

    # Make the eval modules importable as `call_acc` / `exe_acc` from any
    # subprocess (ProcessPoolExecutor in the upstream scripts pickles workers
    # by qualified name). Image build adds symlinks so the digit-prefixed
    # filenames resolve as valid module names.
    eval_dir = f"{REPO_DIR}/EVAL/eval_T"
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    os.environ["PYTHONPATH"] = eval_dir + os.pathsep + os.environ.get("PYTHONPATH", "")

    import call_acc  # noqa: E402  — depends on the sys.path tweak above
    import exe_acc   # noqa: E402

    total = sum(1 for _ in pred_full.open())

    # ---- Phase 1: call accuracy -------------------------------------------------
    print("\n" + "=" * 70 + "\n=== Phase 1: call accuracy ===\n" + "=" * 70, flush=True)
    call_acc.call_4file(str(pred_full), str(call_acc_dir), gpus=[0])
    call_survivors = sorted(p.name for p in call_acc_dir.glob("*.py"))
    print(f"\ncall_acc survivors: {len(call_survivors)} / {total}", flush=True)

    # ---- Phase 2: execution accuracy --------------------------------------------
    print("\n" + "=" * 70 + "\n=== Phase 2: execution accuracy ===\n" + "=" * 70, flush=True)
    if call_survivors:
        exe_acc.execute_4folder(str(call_acc_dir), gpus=[0])
    exec_survivors = sorted(p.name for p in call_acc_dir.glob("*.py"))
    print(f"\nexe_acc survivors: {len(exec_survivors)} / {total}", flush=True)

    # ---- Phase 3: efficiency ----------------------------------------------------
    print("\n" + "=" * 70 + "\n=== Phase 3: efficiency ===\n" + "=" * 70, flush=True)
    eff_summary = "skipped (no surviving operators)"
    speedup = None
    if exec_survivors:
        perf_root = f"{REPO_DIR}/performance_metrics/perf_T"

        # 3a — generate per-op perf scripts under ./tmp
        subprocess.run(
            [
                sys.executable,
                "run_bench/write_file.py",
                "--input_folder_path",
                str(call_acc_dir),
                "--results_path",
                str(perf_results_dir),
            ],
            cwd=perf_root,
            check=True,
        )

        # 3b — actually run them on the GPU
        subprocess.run(
            [sys.executable, "run_bench/multiprocess_gpu_run.py"],
            cwd=perf_root,
            check=True,
        )

        # 3c — compute speedup vs. the golden PyTorch numbers
        eff = subprocess.run(
            [
                sys.executable,
                "2_efficiency.py",
                "--gen_folder",
                str(perf_results_dir),
            ],
            cwd=f"{REPO_DIR}/EVAL/eval_T",
            capture_output=True,
            text=True,
        )
        eff_summary = eff.stdout
        if eff.stderr:
            eff_summary += "\n[stderr]\n" + eff.stderr
        for line in eff.stdout.splitlines():
            if line.startswith("speed up:"):
                try:
                    speedup = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass

    data_volume.commit()

    summary = {
        "total_predictions": total,
        "phase1_call_acc": {
            "passed": len(call_survivors),
            "rate": round(100 * len(call_survivors) / total, 2) if total else 0,
        },
        "phase2_exec_acc": {
            "passed": len(exec_survivors),
            "rate": round(100 * len(exec_survivors) / total, 2) if total else 0,
        },
        "phase3_efficiency": {
            "speedup_vs_pytorch": speedup,
            "raw_output_tail": eff_summary[-2000:],
        },
        "artifacts_volume": VOLUME_NAME,
        "artifacts_subdir": output_subdir,
    }
    return summary


# --------------------------------------------------------------------------- #
# Volume helpers + local entrypoint
# --------------------------------------------------------------------------- #


def _upload_local_predictions(local_path: Path) -> str:
    """Upload a local predictions.jsonl to the volume; return its remote path."""
    if not local_path.exists():
        raise FileNotFoundError(local_path)
    remote = f"uploads/{local_path.name}"
    print(f"uploading {local_path} -> volume://{remote}", flush=True)
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(str(local_path), remote)
    return remote


@app.local_entrypoint()
def main(
    predictions: str = "",
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    dataset: str = "simp",
    limit: int = 0,
    output_subdir: str = "results",
    concurrency: int = 8,
):
    """End-to-end: (optionally) generate predictions, then evaluate.

    Args:
        predictions: path to a local predictions.jsonl. If set, generation is
            skipped and this file is uploaded to the volume.
        provider: ``anthropic`` or ``openai``.
        model: model id for the chosen provider.
        dataset: ``simp`` (simple) or ``comp`` (complex) Alpaca instructions.
        limit: only generate the first N items (useful for smoke tests).
        output_subdir: where to write per-run artifacts inside the volume.
        concurrency: parallel LLM requests.
    """
    if predictions:
        remote = _upload_local_predictions(Path(predictions))
    else:
        tag = f"{provider}_{model.replace('/', '_').replace(':', '_')}_{dataset}"
        remote = generate_predictions.remote(
            provider=provider,
            model=model,
            dataset=dataset,
            output_path=f"predictions/{tag}.jsonl",
            limit=limit or None,
            concurrency=concurrency,
        )

    print(f"\nevaluating: volume://{remote}\n", flush=True)
    summary = evaluate.remote(
        predictions_path=remote,
        output_subdir=output_subdir,
    )
    print("\n=== Final summary ===")
    print(json.dumps(summary, indent=2))


@app.local_entrypoint()
def evaluate_only(
    predictions: str,
    output_subdir: str = "results",
):
    """Evaluate an existing local predictions.jsonl without (re)generating."""
    remote = _upload_local_predictions(Path(predictions))
    summary = evaluate.remote(predictions_path=remote, output_subdir=output_subdir)
    print(json.dumps(summary, indent=2))


@app.local_entrypoint()
def generate_only(
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    dataset: str = "simp",
    limit: int = 0,
    output_path: str = "predictions/predictions.jsonl",
    concurrency: int = 8,
):
    """Generate predictions only; do not evaluate."""
    remote = generate_predictions.remote(
        provider=provider,
        model=model,
        dataset=dataset,
        output_path=output_path,
        limit=limit or None,
        concurrency=concurrency,
    )
    print(f"wrote volume://{remote}")
