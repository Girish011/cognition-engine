"""
Evaluate Qwen3.5 models on MLX (Apple Silicon).

Tests:
1) Can we load and run Qwen3.5-9B and Qwen3.5-4B on MLX?
2) What's the tok/s for each?
3) Does hidden state extraction still work?
4) Does CHI monitoring still work?
5) Does the model follow the tool-call format better than Qwen2.5-Coder-7B?

Usage:
  .venv/bin/python scripts/eval_qwen35.py
  .venv/bin/python scripts/eval_qwen35.py --models mlx-community/Qwen3.5-4B-Instruct-4bit
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

from mlx_lm import generate, load


# Ensure project imports work when running as a script.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_MODELS_TO_TEST = [
    # These may or may not exist yet on mlx-community; script handles failures.
    "mlx-community/Qwen3.5-4B-Instruct-4bit",
    "mlx-community/Qwen3.5-9B-Instruct-4bit",
]


@dataclass
class BenchmarkResult:
    model: str
    load_ok: bool
    tps: float | None
    tokens: int | None
    seconds: float | None
    hidden_states_ok: bool | None
    hidden_states_shape: str | None
    chi_ok: bool | None
    tool_format_ok: bool | None
    notes: str | None = None


def _safe_bool(x: Any) -> bool | None:
    if x is None:
        return None
    return bool(x)


def benchmark_model(model_path: str) -> BenchmarkResult:
    print(f"\n{'=' * 70}")
    print(f"Testing: {model_path}")
    print(f"{'=' * 70}")

    try:
        model, tokenizer = load(model_path)
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        print(f"FAILED to load: {msg}")
        print("Model may not be available on mlx-community yet.")
        print("Search: https://huggingface.co/mlx-community?search=qwen3.5")
        return BenchmarkResult(
            model=model_path,
            load_ok=False,
            tps=None,
            tokens=None,
            seconds=None,
            hidden_states_ok=None,
            hidden_states_shape=None,
            chi_ok=None,
            tool_format_ok=None,
            notes=msg[:400],
        )

    # 1) Basic generation + speed
    prompt = "Write a Python function that computes fibonacci numbers."
    max_tokens = 300
    start = time.time()
    response = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
    elapsed = time.time() - start
    tokens = len(tokenizer.encode(response))
    tps = tokens / elapsed if elapsed > 0 else 0.0

    print(f"Speed: {tokens} tokens in {elapsed:.2f}s = {tps:.1f} tok/s")
    print(f"Response preview: {response[:200].replace('\\n', ' ')}...")

    # 2) Hidden state extraction
    hidden_ok: bool | None = None
    hidden_shape: str | None = None
    try:
        from cognition_engine.hidden_states import extract_hidden_states

        _, states = extract_hidden_states(
            model,
            tokenizer,
            "Write hello world in Python.",
            layer_idx=16,
            max_tokens=50,
        )
        hidden_ok = True
        hidden_shape = str(getattr(states, "shape", None))
        print(f"Hidden states: {hidden_shape}")
    except Exception as e:  # noqa: BLE001
        hidden_ok = False
        print(f"Hidden state extraction FAILED: {e}")
        print("May need architecture-specific adaptation (layer indexing / blocks).")

    # 3) CHI monitoring smoke test
    chi_ok: bool | None = None
    try:
        from cognition_engine.chi import CHIMonitor

        mon = CHIMonitor(model, tokenizer, original_task="Create hello.py")
        r1 = generate(
            model,
            tokenizer,
            prompt="User: Create hello.py\nAssistant: ",
            max_tokens=120,
        )
        upd = mon.update(r1, full_context="User: Create hello.py\nAssistant: " + r1)
        chi = float(upd.get("chi_score"))
        level = int(mon.get_intervention_level())
        print(f"CHI smoke: chi={chi:.3f} level={level}")
        chi_ok = True
    except Exception as e:  # noqa: BLE001
        chi_ok = False
        print(f"CHI monitoring FAILED: {e}")

    # 4) Tool format compliance
    tool_prompt = """You are a coding assistant with these tools:
[READ] path — Read a file
[WRITE] path
```
content
```
[EXEC] command — Run a command
[THINK] thought — Record reasoning

Task: Create a hello.py file that prints "Hello World".
Respond with exactly one tool call and NOTHING ELSE.

Valid example:
[WRITE] hello.py
```
print("Hello World")
```
"""
    tool_response = generate(model, tokenizer, prompt=tool_prompt, max_tokens=200)
    tool_response_preview = tool_response[:300].replace("\n", " ")
    print("\nTool format test:")
    print(f"Response: {tool_response_preview}")
    tool_format_ok = any(
        marker in tool_response for marker in ("[READ]", "[WRITE]", "[EXEC]", "[THINK]")
    )
    print(f"Contains tool marker: {tool_format_ok}")

    return BenchmarkResult(
        model=model_path,
        load_ok=True,
        tps=tps,
        tokens=tokens,
        seconds=elapsed,
        hidden_states_ok=_safe_bool(hidden_ok),
        hidden_states_shape=hidden_shape,
        chi_ok=_safe_bool(chi_ok),
        tool_format_ok=_safe_bool(tool_format_ok),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3.5 models on MLX.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS_TO_TEST,
        help="One or more HF model IDs (mlx-community/*).",
    )
    args = parser.parse_args()

    results: list[BenchmarkResult] = []
    for m in args.models:
        results.append(benchmark_model(m))

    print(f"\n{'=' * 70}")
    print("COMPARISON")
    print(f"{'=' * 70}")
    print("Baseline (from SESSION_LOG): Qwen2.5-Coder-7B ~53.0 tok/s")
    for r in results:
        if not r.load_ok:
            print(f"{r.model}: LOAD FAILED ({r.notes})")
            continue
        hs = r.hidden_states_shape or "n/a"
        print(
            f"{r.model}: {r.tps:.1f} tok/s | hidden_states_ok={r.hidden_states_ok} "
            f"({hs}) | chi_ok={r.chi_ok} | tool_format_ok={r.tool_format_ok}"
        )


if __name__ == "__main__":
    main()

