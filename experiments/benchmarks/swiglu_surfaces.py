"""Swiglu surfaces."""

from __future__ import annotations
import torch
import sys
from pathlib import Path
from .common import RESULTS, emit, run_config

SEED = 42
STEPS = 40
EVAL = 4
MODELS = [
    ("qwen25", "Qwen/Qwen2.5-1.5B"),
    ("llama32", "meta-llama/Llama-3.2-1B"),
    ("qwen3", "Qwen/Qwen3-1.7B"),
]
SHAPES = [
    ("paper_b2_s1024", 2, 1024),
    ("b4_s1024", 4, 1024),
    ("b8_s1024", 8, 1024),
    ("b2_s2048", 2, 2048),
    ("b2_s4096", 2, 4096),
    ("b4_s2048", 4, 2048),
]
SURFACES = [
    ("mlp_all", {"surface": "mlp", "min_features": 256}),
    ("down_proj_only", {"surface": "none", "min_features": 256, "_down_only": True}),
]
RANKS = [8, 32]


def down_proj_names(model):
    return [
        n
        for (n, m) in model.named_modules()
        if n.endswith("down_proj") and hasattr(m, "weight") and (m.weight.dim() == 2)
    ]


def cfg_for(model_id, surf_name, surf_cfg, rank):
    cfg = {
        "rank": rank,
        "residual_budget": 0.005,
        "residual_bits": 8,
        "chunk_rows": 512,
        **{k: v for (k, v) in surf_cfg.items() if not k.startswith("_")},
    }
    if surf_cfg.get("_down_only"):
        from .common import load_model

        probe = load_model(model_id, "cpu")
        cfg["extra_names"] = down_proj_names(probe)
        del probe
        torch.cuda.empty_cache()
    return cfg


def main():
    out = RESULTS / "e2_swiglu.jsonl"
    for tagm, model_id in MODELS:
        for shape_name, bs, seq in SHAPES:
            base = None
            try:
                base = run_config(
                    model_id=model_id,
                    bs=bs,
                    seq=seq,
                    steps=STEPS,
                    seed=SEED,
                    method="baseline",
                    amp=True,
                    eight_bit=True,
                    eval_batches=EVAL,
                    measure_from=6,
                    tag=f"{tagm}/{shape_name}",
                )
            except Exception as exc:
                print(f"  baseline OOM/fail {tagm} {shape_name}: {type(exc).__name__}", flush=True)
                emit(
                    {
                        "tag": f"{tagm}/{shape_name}",
                        "method": "baseline",
                        "status": "OOM_or_error",
                        "error": str(exc)[:200],
                    },
                    out,
                )
                continue
            emit(base, out)
            bp = base["peak_bytes"]
            for surf_name, surf_cfg in SURFACES:
                for rank in RANKS:
                    try:
                        cfg = cfg_for(model_id, surf_name, surf_cfg, rank)
                        rec = run_config(
                            model_id=model_id,
                            bs=bs,
                            seq=seq,
                            steps=STEPS,
                            seed=SEED,
                            method=f"stacr_{surf_name}_r{rank}",
                            stacr_cfg=cfg,
                            amp=True,
                            eight_bit=True,
                            eval_batches=EVAL,
                            measure_from=6,
                            baseline_peak_bytes=bp,
                            tag=f"{tagm}/{shape_name}",
                        )
                    except Exception as exc:
                        print(
                            f"  fail {tagm} {shape_name} {surf_name} r{rank}: {type(exc).__name__}",
                            flush=True,
                        )
                        emit(
                            {
                                "tag": f"{tagm}/{shape_name}",
                                "method": f"stacr_{surf_name}_r{rank}",
                                "status": "OOM_or_error",
                                "error": str(exc)[:200],
                            },
                            out,
                        )
                        continue
                    rec["measured_saving_pct"] = 100.0 * (bp - rec["peak_bytes"]) / bp
                    rec["baseline_peak_GiB"] = base["peak_GiB"]
                    emit(rec, out)


if __name__ == "__main__":
    main()
