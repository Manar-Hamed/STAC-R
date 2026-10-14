"""Residual selection at matched rank, budget, and saved surface."""

from .common import RESULTS, emit, run_config

SEEDS = [42, 123]
(MODEL, BS, SEQ, STEPS) = ("gpt2-medium", 8, 1024, 300)
CFG = {
    "rank": 8,
    "residual_budget": 0.005,
    "residual_bits": 8,
    "surface": "mlp",
    "min_features": 256,
    "chunk_rows": 512,
}
VARIANTS = [
    ("exact", {"residual_mode": "chunk_topk"}),
    ("bigchunk", {"residual_mode": "global_topk"}),
    ("block16", {"residual_mode": "block_sparse", "residual_block_size": 16}),
    ("block64", {"residual_mode": "block_sparse", "residual_block_size": 64}),
    ("sampled", {"residual_mode": "sampled", "residual_sample_ratio": 0.1}),
    (
        "block16_ui100",
        {
            "residual_mode": "block_sparse",
            "residual_block_size": 16,
            "update_interval": 100,
            "basis_dtype": "float16",
            "freeze_after_warmup": True,
            "warmup_steps": 20,
        },
    ),
    ("turbo_ref", {"turbo_after_step": 2}),
]


def main():
    out = RESULTS / "r5_residual_fast.jsonl"
    baseline = {}
    for seed in SEEDS:
        record = run_config(
            model_id=MODEL, bs=BS, seq=SEQ, steps=STEPS, seed=seed, method="baseline", tag="resfast"
        )
        baseline[seed] = record
        emit(record, out)
    for name, options in VARIANTS:
        for seed in SEEDS:
            reference = baseline[seed]
            record = run_config(
                model_id=MODEL,
                bs=BS,
                seq=SEQ,
                steps=STEPS,
                seed=seed,
                method=name,
                tag="resfast",
                stacr_cfg=dict(CFG, **options),
                baseline_peak_bytes=reference["peak_bytes"],
            )
            record["time_x"] = record["time_s"] / reference["time_s"]
            record["measured_saving_pct"] = (
                100 * (reference["peak_bytes"] - record["peak_bytes"]) / reference["peak_bytes"]
            )
            record["selector"] = name
            emit(record, out)


if __name__ == "__main__":
    main()
