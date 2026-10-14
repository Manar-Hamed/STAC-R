"""Composed frontier."""

from __future__ import annotations
import torch
import sys
from pathlib import Path
from .common import RESULTS, emit, run_config

SEEDS = [42, 123, 456]
CFG8 = {
    "rank": 8,
    "residual_budget": 0.005,
    "residual_bits": 8,
    "surface": "mlp",
    "min_features": 256,
    "chunk_rows": 512,
}
TURBO = dict(CFG8, turbo_after_step=2)
DENSITIES = [8, 6, 4, 2, 1]
COHORTS = [
    dict(name="cf_gpt2m", model_id="gpt2-medium", bs=8, seq=1024, steps=400),
    dict(name="cf_gpt2l", model_id="gpt2-large", bs=4, seq=1024, steps=300),
    dict(name="cf_lh2000", model_id="gpt2-medium", bs=8, seq=1024, steps=2000, arms="lh"),
    dict(
        name="cf_gpt2xl",
        model_id="gpt2-xl",
        bs=2,
        seq=1024,
        steps=200,
        dtype=torch.bfloat16,
        arms="xl",
    ),
]
LH_ARMS = {
    "baseline",
    "turbo_fast",
    "ckpt_k4",
    "ckpt_k4_plus_turbo",
    "ckpt_k2",
    "ckpt_k2_plus_turbo",
}
XL_ARMS = {
    "baseline",
    "stacr_r8",
    "turbo_fast",
    "ckpt_k4",
    "ckpt_k4_plus_turbo",
    "ckpt_k2",
    "ckpt_k2_plus_turbo",
    "ckpt_k1",
    "ckpt_k1_plus_turbo",
}
ARM_SETS = {"lh": LH_ARMS, "xl": XL_ARMS}


def complement(k: int):
    return lambda i, _k=k: i % _k != 0


def rows():
    out = [
        ("baseline", {}),
        ("stacr_r8", dict(stacr_cfg=CFG8)),
        ("turbo_fast", dict(stacr_cfg=TURBO)),
    ]
    for k in DENSITIES:
        out.append((f"ckpt_k{k}", dict(ckpt_every=k, ckpt_offset=0)))
        out.append(
            (
                f"ckpt_k{k}_plus_turbo",
                dict(ckpt_every=k, ckpt_offset=0, stacr_cfg=TURBO, stacr_block_pred=complement(k)),
            )
        )
        out.append(
            (
                f"ckpt_k{k}_plus_stacr",
                dict(ckpt_every=k, ckpt_offset=0, stacr_cfg=CFG8, stacr_block_pred=complement(k)),
            )
        )
    for budget in (0.0025, 0.01):
        lbl = str(budget).replace("0.", "p")
        out.append(
            (
                f"ckpt_k4_plus_res{lbl}",
                dict(
                    ckpt_every=4,
                    ckpt_offset=0,
                    stacr_cfg=dict(CFG8, residual_budget=budget),
                    stacr_block_pred=complement(4),
                ),
            )
        )
    return out


def main():
    out = RESULTS / "x1_composed_frontier.jsonl"
    want = set(sys.argv[1:])
    cohorts = [c for c in COHORTS if not want or c["name"] in want]
    if not cohorts:
        raise SystemExit(
            f"no cohort matched {sorted(want)}; available: {[c['name'] for c in COHORTS]}"
        )
    for coh in cohorts:
        dtype = coh.get("dtype", torch.float32)
        print(
            f"\n=== cohort {coh['name']}  {coh['model_id']} b={coh['bs']} s={coh['seq']} steps={coh['steps']} dtype={dtype} ===",
            flush=True,
        )
        base_peak = None
        keep = ARM_SETS.get(coh.get("arms"))
        arms = [(m, e) for (m, e) in rows() if keep is None or m in keep]
        if keep is not None and len(arms) != len(keep):
            raise SystemExit(
                f"{coh['name']}: arm set names {sorted(keep - {m for (m, _) in arms})} that rows() does not produce"
            )
        print(f"  {len(arms)} arms x {len(SEEDS)} seeds", flush=True)
        for method, extra in arms:
            for seed in SEEDS:
                try:
                    rec = run_config(
                        model_id=coh["model_id"],
                        bs=coh["bs"],
                        seq=coh["seq"],
                        steps=coh["steps"],
                        seed=seed,
                        method=method,
                        dtype=dtype,
                        tag=coh["name"],
                        baseline_peak_bytes=base_peak,
                        **extra,
                    )
                except torch.cuda.OutOfMemoryError as exc:
                    print(f"  OOM {method} seed={seed}: {exc}", flush=True)
                    emit({"tag": coh["name"], "method": method, "seed": seed, "status": "OOM"}, out)
                    continue
                except Exception as exc:
                    print(f"  FAIL {method} seed={seed}: {type(exc).__name__}: {exc}", flush=True)
                    emit(
                        {
                            "tag": coh["name"],
                            "method": method,
                            "seed": seed,
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                        out,
                    )
                    continue
                if method == "baseline" and base_peak is None:
                    base_peak = rec["peak_bytes"]
                    print(f"  baseline peak = {rec['peak_GiB']:.3f} GiB", flush=True)
                emit(rec, out)
                n_ck = rec.get("n_checkpointed_blocks")
                n_wr = rec.get("n_wrapped_layers")
                print(
                    f"  {method:28s} seed={seed} peak={rec['peak_GiB']:.3f} GiB t={rec['time_s']:.1f}s ppl={rec['val_ppl']:.4f} ckpt_blocks={n_ck} wrapped={n_wr}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
