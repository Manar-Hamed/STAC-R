"""Swiglu precision."""

from __future__ import annotations
import torch
import sys
from pathlib import Path
from .common import RESULTS, emit, run_config

MODELS = [("qwen25", "Qwen/Qwen2.5-1.5B"), ("llama32", "meta-llama/Llama-3.2-1B")]
(BS, SEQ, STEPS) = (2, 1024, 30)
AXES = [
    ("fp32_amp_8bit_mlp", torch.float32, True, True, "mlp"),
    ("bf16_amp_8bit_mlp", torch.bfloat16, True, True, "mlp"),
    ("fp32_noamp_adamw_mlp", torch.float32, False, False, "mlp"),
    ("bf16_noamp_8bit_mlp", torch.bfloat16, False, True, "mlp"),
    ("fp32_amp_8bit_mlpattn", torch.float32, True, True, "mlp+attention"),
    ("bf16_amp_8bit_mlpattn", torch.bfloat16, True, True, "mlp+attention"),
]


def main():
    out = RESULTS / "e2b_swiglu_diag.jsonl"
    for tagm, model_id in MODELS:
        for label, dtype, amp, eight, surface in AXES:
            cfg = {
                "rank": 8,
                "residual_budget": 0.005,
                "residual_bits": 8,
                "surface": surface,
                "min_features": 256,
                "chunk_rows": 512,
            }
            base = None
            for method, sc in [("baseline", None), ("stacr_r8", cfg)]:
                try:
                    r = run_config(
                        model_id=model_id,
                        bs=BS,
                        seq=SEQ,
                        steps=STEPS,
                        seed=42,
                        method=method,
                        stacr_cfg=sc,
                        amp=amp,
                        eight_bit=eight,
                        dtype=dtype,
                        eval_batches=2,
                        measure_from=6,
                        baseline_peak_bytes=base,
                        tag=f"diag/{tagm}/{label}",
                    )
                except Exception as exc:
                    emit(
                        {
                            "tag": f"diag/{tagm}/{label}",
                            "method": method,
                            "status": "OOM_or_error",
                            "error": f"{type(exc).__name__}: {exc}"[:200],
                        },
                        out,
                    )
                    continue
                if method == "baseline":
                    base = r["peak_bytes"]
                else:
                    r["measured_saving_pct"] = 100.0 * (base - r["peak_bytes"]) / base
                    r["baseline_peak_GiB"] = base / 1024**3
                r["axis"] = label
                emit(r, out)


if __name__ == "__main__":
    main()
