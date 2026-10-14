"""Allocator stability."""

from __future__ import annotations
import torch
import json, sys, time
from pathlib import Path
from .common import (
    RESULTS,
    emit,
    load_model,
    get_batches,
    build_optimizer,
    amp_ctx,
    free_all,
    set_seed,
    gib,
    sync,
    nvml_used,
)
from stacr.surgery import apply_stacr, set_step

(MODEL, BS, SEQ, STEPS) = ("Qwen/Qwen2.5-1.5B", 2, 1024, 600)
CFG = {
    "turbo_after_step": None,
    "rank": 8,
    "residual_budget": 0.005,
    "residual_bits": 8,
    "surface": "mlp",
    "min_features": 256,
    "chunk_rows": 512,
}


def trace(method, cfg):
    free_all()
    set_seed(42)
    m = load_model(MODEL, "cuda")
    layers = apply_stacr(m, {"turbo_after_step": None, **cfg}) if cfg else []
    opt = build_optimizer(m, 5e-05, True)
    data = get_batches(MODEL, "train", BS, SEQ, STEPS, "wikitext", "cuda")
    marks = []
    sync()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for i, b in enumerate(data):
        if layers:
            set_step(layers, i)
        opt.zero_grad(set_to_none=True)
        with amp_ctx(True):
            loss = m(**b).loss
        loss.backward()
        opt.step()
        if i in (10, 50, 100, 200, 300, 400, 500, STEPS - 1):
            sync()
            marks.append(
                {
                    "step": i,
                    "allocated_GiB": gib(torch.cuda.max_memory_allocated()),
                    "reserved_GiB": gib(torch.cuda.max_memory_reserved()),
                    "nvml_GiB": nvml_used(0),
                    "elapsed_s": time.perf_counter() - t0,
                }
            )
            print(
                f"  [{method}] step {i:4d} alloc {marks[-1]['allocated_GiB']:.3f} reserved {marks[-1]['reserved_GiB']:.3f} nvml {marks[-1]['nvml_GiB']:.3f}",
                flush=True,
            )
    rec = {
        "tag": "drift",
        "method": method,
        "model": MODEL,
        "bs": BS,
        "seq": SEQ,
        "steps": STEPS,
        "marks": marks,
        "final_allocated_GiB": marks[-1]["allocated_GiB"],
        "final_reserved_GiB": marks[-1]["reserved_GiB"],
        "final_nvml_GiB": marks[-1]["nvml_GiB"],
    }
    del m, opt, layers, data
    free_all()
    return rec


def main():
    out = RESULTS / "q6_drift.jsonl"
    res = {}
    for method, cfg in [("baseline", None), ("stacr_r8", CFG)]:
        try:
            r = trace(method, cfg)
            res[method] = r
            emit(r, out)
        except Exception as exc:
            emit(
                {
                    "tag": "drift",
                    "method": method,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}"[:300],
                },
                out,
            )
    if len(res) == 2:
        (b, s) = (res["baseline"], res["stacr_r8"])
        summary = {"tag": "drift", "summary": True}
        for col in ("allocated", "reserved", "nvml"):
            (bv, sv) = (b[f"final_{col}_GiB"], s[f"final_{col}_GiB"])
            summary[f"{col}_delta_pct"] = 100.0 * (bv - sv) / bv if bv else None
        print("SUMMARY", json.dumps(summary), flush=True)
        emit(summary, out)


if __name__ == "__main__":
    main()
