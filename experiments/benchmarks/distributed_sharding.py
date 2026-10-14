"""Distributed sharding."""

from __future__ import annotations
import torch
import os
import sys
import time
from pathlib import Path
import torch.distributed as dist
from .common import (
    RESULTS,
    apply_selective_checkpointing,
    amp_ctx,
    emit,
    free_all,
    get_batches,
    gib,
    load_model,
    set_seed,
    set_step,
    stacr_on_blocks,
    blocks_of,
    sync,
)

CFG8 = {
    "rank": 8,
    "residual_budget": 0.005,
    "residual_bits": 8,
    "surface": "mlp",
    "min_features": 256,
    "chunk_rows": 512,
}
TURBO = dict(CFG8, turbo_after_step=2)


def complement(k: int):
    return lambda i, _k=k: i % _k != 0


ARMS = [
    ("baseline", dict()),
    ("turbo_fast", dict(stacr_cfg=TURBO)),
    ("stacr_r8", dict(stacr_cfg=CFG8)),
    ("ckpt_k4", dict(ckpt_every=4)),
    ("ckpt_k4_plus_turbo", dict(ckpt_every=4, stacr_cfg=TURBO, stacr_block_pred=complement(4))),
    ("ckpt_all", dict(ckpt_every=1)),
    ("ckpt_all_plus_turbo", dict(ckpt_every=1, stacr_cfg=TURBO)),
]
ARMS_BIG = [a for a in ARMS if a[0] != "stacr_r8"]
TURBO_WIDE = dict(TURBO, surface="mlp+attention")
ARMS_WIDE = [
    ("baseline", dict()),
    ("turbo_wide", dict(stacr_cfg=TURBO_WIDE)),
    (
        "ckpt_k4_plus_turbo_wide",
        dict(ckpt_every=4, stacr_cfg=TURBO_WIDE, stacr_block_pred=complement(4)),
    ),
    ("ckpt_all", dict(ckpt_every=1)),
]


def arm_filter(arms):
    only = os.environ.get("STACR_ARM", "").strip()
    skip = {s for s in os.environ.get("STACR_SKIP_ARMS", "").replace(",", " ").split() if s}
    out = list(arms)
    if only:
        out = [a for a in out if a[0] == only]
        if not out:
            raise SystemExit(f"STACR_ARM={only!r} matches no arm in this cohort")
    if skip:
        out = [a for a in out if a[0] not in skip]
        if not out:
            raise SystemExit(f"STACR_SKIP_ARMS={sorted(skip)} removed every arm in this cohort")
    return out


COHORTS = {
    "qwen15": dict(
        model_id="Qwen/Qwen2.5-1.5B",
        bs=2,
        seq=1024,
        steps=60,
        measure_from=10,
        seeds=[42, 123, 456],
        arms=ARMS,
    ),
    "qwen3b": dict(
        model_id="Qwen/Qwen2.5-3B",
        bs=1,
        seq=1024,
        steps=40,
        measure_from=8,
        seeds=[42],
        arms=ARMS_BIG,
    ),
    "starcoder3b": dict(
        model_id="bigcode/starcoder2-3b",
        bs=1,
        seq=1024,
        steps=40,
        measure_from=8,
        seeds=[42],
        arms=ARMS_BIG,
    ),
    "qwen7b": dict(
        model_id="Qwen/Qwen2.5-7B",
        bs=1,
        seq=1024,
        steps=30,
        measure_from=8,
        seeds=[42],
        arms=ARMS_BIG,
    ),
    "qwen14b": dict(
        model_id="Qwen/Qwen2.5-14B",
        bs=1,
        seq=1024,
        steps=20,
        measure_from=5,
        seeds=[42],
        arms=ARMS_BIG,
    ),
    "qwen7b_s2048": dict(
        model_id="Qwen/Qwen2.5-7B",
        bs=1,
        seq=2048,
        steps=25,
        measure_from=8,
        seeds=[42],
        arms=ARMS_BIG,
    ),
    "qwen7b_s3072": dict(
        model_id="Qwen/Qwen2.5-7B",
        bs=1,
        seq=3072,
        steps=20,
        measure_from=6,
        seeds=[42],
        arms=ARMS_BIG,
    ),
    "qwen7b_s4096": dict(
        model_id="Qwen/Qwen2.5-7B",
        bs=1,
        seq=4096,
        steps=20,
        measure_from=6,
        seeds=[42],
        arms=ARMS_BIG,
    ),
    "qwen7b_s3072_wide": dict(
        model_id="Qwen/Qwen2.5-7B",
        bs=1,
        seq=3072,
        steps=20,
        measure_from=6,
        seeds=[42],
        arms=ARMS_WIDE,
    ),
    "qwen14b_s2048": dict(
        model_id="Qwen/Qwen2.5-14B",
        bs=1,
        seq=2048,
        steps=15,
        measure_from=5,
        seeds=[42],
        arms=ARMS_BIG,
    ),
}


def build_opt_honest(model, lr=5e-05):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    pdt = next(model.parameters()).dtype
    return (opt, f"torch_adamw_{str(pdt).replace('torch.', '')}_states", str(pdt))


def shard(model):
    from torch.distributed.fsdp import fully_shard

    for blk in blocks_of(model):
        fully_shard(blk)
    fully_shard(model)
    return model


def one_run(
    method,
    seed,
    spec,
    *,
    ckpt_every=0,
    stacr_cfg=None,
    stacr_block_pred=None,
    device="cuda",
    smoke=False,
):
    if device == "cuda":
        free_all()
    set_seed(seed)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    (model_id, bs, seq) = (spec["model_id"], spec["bs"], spec["seq"])
    steps = 4 if smoke else spec["steps"]
    measure_from = 1 if smoke else spec["measure_from"]
    model = load_model(model_id, device, dtype=torch.bfloat16)
    n_ckpt = apply_selective_checkpointing(model, ckpt_every, 0) if ckpt_every else 0
    layers = stacr_on_blocks(model, stacr_cfg, stacr_block_pred) if stacr_cfg else []
    n_wrapped = len(layers)
    if dist.is_initialized():
        model = shard(model)
    (opt, opt_name, state_dt) = build_opt_honest(model)
    train = get_batches(model_id, "train", bs, seq, steps * world, "wikitext", device)
    train = train[rank::world][:steps]
    if device == "cuda":
        sync()
        torch.cuda.reset_peak_memory_stats()
    (t0, n_measured, loss) = (None, 0, None)
    for i, batch in enumerate(train):
        if layers:
            set_step(layers, i)
        if i == measure_from:
            if device == "cuda":
                sync()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with amp_ctx(device == "cuda"):
            loss = model(**batch).loss
        loss.backward()
        opt.step()
        if t0 is not None:
            n_measured += 1
    if device == "cuda":
        sync()
    dt = time.perf_counter() - t0 if t0 is not None else float("nan")
    peak = torch.cuda.max_memory_allocated() if device == "cuda" else 0
    peaks = [peak]
    if world > 1:
        buf = [torch.zeros(1, dtype=torch.float64, device=device) for _ in range(world)]
        dist.all_gather(buf, torch.tensor([float(peak)], dtype=torch.float64, device=device))
        peaks = [float(b.item()) for b in buf]
    rec = {
        "tag": f"fsdp_w{world}",
        "method": method,
        "seed": seed,
        "world_size": world,
        "model": model_id,
        "bs_per_rank": bs,
        "seq": seq,
        "steps": steps,
        "steps_measured": n_measured,
        "n_checkpointed_blocks": n_ckpt,
        "n_wrapped_layers": n_wrapped,
        "time_s": dt,
        "sharding": "fsdp2_full",
        "optimizer_actual": opt_name,
        "optimizer_state_dtype": state_dt,
        "dtype": "bfloat16",
        "peak_bytes": int(peak),
        "peak_GiB": gib(peak),
        "peak_GiB_all_ranks": [gib(p) for p in peaks],
        "tokens_per_s_per_rank": bs * seq * n_measured / dt if dt == dt else None,
        "final_train_loss": float(loss) if loss is not None else None,
    }
    del model, opt, train, layers
    if device == "cuda":
        free_all()
    return rec


def run_cohort(name, spec, out):
    rank = dist.get_rank()
    world = dist.get_world_size()
    arms = arm_filter(spec["arms"])
    if rank == 0:
        print(
            f"\n########## FSDP2 {name}  world={world}  {spec['model_id']} b={spec['bs']}/rank s={spec['seq']} steps={spec['steps']}  {len(arms)} arms x {len(spec['seeds'])} seeds ##########",
            flush=True,
        )
    base = None
    for method, kw in arms:
        for seed in spec["seeds"]:
            try:
                rec = one_run(method, seed, spec, **kw)
            except torch.cuda.OutOfMemoryError:
                if rank == 0:
                    emit(
                        {
                            "tag": f"fsdp_w{world}",
                            "cohort": name,
                            "method": method,
                            "seed": seed,
                            "status": "OOM",
                        },
                        out,
                    )
                    print(f"  OOM  {method} seed={seed}", flush=True)
                torch.cuda.empty_cache()
                continue
            except Exception as exc:
                if rank == 0:
                    emit(
                        {
                            "tag": f"fsdp_w{world}",
                            "cohort": name,
                            "method": method,
                            "seed": seed,
                            "status": "error",
                            "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                        },
                        out,
                    )
                    print(f"  FAIL {method} seed={seed}: {type(exc).__name__}: {exc}", flush=True)
                torch.cuda.empty_cache()
                continue
            rec["cohort"] = name
            if method == "baseline" and base is None:
                base = rec["peak_bytes"]
            if rank == 0:
                emit(rec, out)
                sav = 100.0 * (base - rec["peak_bytes"]) / base if base else 0.0
                print(
                    f"  {method:20s} seed={seed} peak/rank={rec['peak_GiB']:7.3f} GiB ({sav:+6.2f}%) all={['%.3f' % p for p in rec['peak_GiB_all_ranks']]} t={rec['time_s']:6.1f}s opt={rec['optimizer_actual']} wrapped={rec['n_wrapped_layers']} ckpt={rec['n_checkpointed_blocks']}",
                    flush=True,
                )


def main():
    if "--smoke" in sys.argv:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29519")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group("nccl")
        torch.cuda.set_device(0)
        spec = dict(model_id="gpt2", bs=1, seq=256, steps=4, measure_from=1, seeds=[42], arms=ARMS)
        ok = True
        for method, kw in spec["arms"]:
            try:
                rec = one_run(method, 42, spec, smoke=True, **kw)
                print(
                    f"  smoke {method:20s} ok  wrapped={rec['n_wrapped_layers']:3d} ckpt={rec['n_checkpointed_blocks']:2d} opt={rec['optimizer_actual']} loss={rec['final_train_loss']:.4f} finite={rec['final_train_loss'] == rec['final_train_loss']}",
                    flush=True,
                )
            except Exception as exc:
                ok = False
                print(f"  smoke {method:20s} FAIL {type(exc).__name__}: {exc}", flush=True)
        print("FSDP SMOKE " + ("OK" if ok else "FAILED"), flush=True)
        dist.destroy_process_group()
        sys.exit(0 if ok else 1)
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", dist.get_rank())))
    out = RESULTS / "x8_fsdp.jsonl"
    want = [a for a in sys.argv[1:] if a in COHORTS] or list(COHORTS)
    for name in want:
        run_cohort(name, COHORTS[name], out)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
