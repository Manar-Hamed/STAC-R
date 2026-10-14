"""Modern architectures."""

from __future__ import annotations
import torch
import shutil, sys
from pathlib import Path
from .common import RESULTS, emit, run_config

CFG = {
    "rank": 8,
    "residual_budget": 0.005,
    "residual_bits": 8,
    "surface": "mlp",
    "min_features": 256,
    "chunk_rows": 512,
}
MODELS = [
    ("qwen3_1_7b", "Qwen/Qwen3-1.7B"),
    ("qwen25_3b", "Qwen/Qwen2.5-3B"),
    ("llama32_3b", "meta-llama/Llama-3.2-3B"),
]
SHAPES = [(1, 1024), (2, 1024), (4, 1024), (2, 2048)]


def main():
    out = RESULTS / "q2_modern.jsonl"
    for tagm, model_id in MODELS:
        for bs, seq in SHAPES:
            tag = f"{tagm}/b{bs}_s{seq}"
            try:
                base = run_config(
                    model_id=model_id,
                    bs=bs,
                    seq=seq,
                    steps=24,
                    seed=42,
                    method="baseline",
                    amp=True,
                    eight_bit=True,
                    dtype=torch.bfloat16,
                    eval_batches=2,
                    measure_from=6,
                    tag=tag,
                )
            except Exception as exc:
                emit(
                    {
                        "tag": tag,
                        "method": "baseline",
                        "status": "OOM_or_error",
                        "error": f"{type(exc).__name__}: {exc}"[:200],
                    },
                    out,
                )
                continue
            emit(base, out)
            bp = base["peak_bytes"]
            for surf, cfg in [
                ("mlp_all", dict(CFG)),
                ("mlp_attn", dict(CFG, surface="mlp+attention")),
            ]:
                try:
                    r = run_config(
                        model_id=model_id,
                        bs=bs,
                        seq=seq,
                        steps=24,
                        seed=42,
                        method=f"stacr_{surf}_r8",
                        stacr_cfg=cfg,
                        amp=True,
                        eight_bit=True,
                        dtype=torch.bfloat16,
                        eval_batches=2,
                        measure_from=6,
                        baseline_peak_bytes=bp,
                        tag=tag,
                    )
                except Exception as exc:
                    emit(
                        {
                            "tag": tag,
                            "method": f"stacr_{surf}_r8",
                            "status": "OOM_or_error",
                            "error": f"{type(exc).__name__}: {exc}"[:200],
                        },
                        out,
                    )
                    continue
                r["measured_saving_pct"] = 100.0 * (bp - r["peak_bytes"]) / bp
                r["baseline_peak_GiB"] = base["peak_GiB"]
                emit(r, out)


if __name__ == "__main__":
    main()
