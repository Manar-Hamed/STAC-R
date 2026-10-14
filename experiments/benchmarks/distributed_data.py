"""Distributed data."""

from __future__ import annotations
import torch
import os
import sys
import time
from pathlib import Path
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .common import (
    RESULTS,
    apply_selective_checkpointing,
    amp_ctx,
    build_optimizer,
    emit,
    free_all,
    get_batches,
    gib,
    load_model,
    set_seed,
    set_step,
    stacr_on_blocks,
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
(MODEL, BS, SEQ, STEPS, MEASURE_FROM) = ("gpt2-medium", 4, 1024, 120, 20)
SEEDS = [42, 123, 456]


def complement(k: int):
    return lambda i, _k=k: i % _k != 0


ARMS = [
    ("baseline", dict()),
    ("stacr_r8", dict(stacr_cfg=CFG8)),
    ("turbo_fast", dict(stacr_cfg=TURBO)),
    ("ckpt_k4", dict(ckpt_every=4)),
    ("ckpt_k4_plus_turbo", dict(ckpt_every=4, stacr_cfg=TURBO, stacr_block_pred=complement(4))),
    ("ckpt_k2", dict(ckpt_every=2)),
    ("ckpt_k2_plus_turbo", dict(ckpt_every=2, stacr_cfg=TURBO, stacr_block_pred=complement(2))),
]
_only = os.environ.get("STACR_ARM", "").strip()
_skip = {s for s in os.environ.get("STACR_SKIP_ARMS", "").replace(",", " ").split() if s}
if _only:
    ARMS = [a for a in ARMS if a[0] == _only]
    if not ARMS:
        raise SystemExit(f"STACR_ARM={_only!r} matches no arm in this file")
if _skip:
    ARMS = [a for a in ARMS if a[0] not in _skip]
    if not ARMS:
        raise SystemExit(f"STACR_SKIP_ARMS={sorted(_skip)} removed every arm in this file")


def one_run(
    method,
    seed,
    *,
    ckpt_every=0,
    stacr_cfg=None,
    stacr_block_pred=None,
    model_id=MODEL,
    bs=BS,
    seq=SEQ,
    steps=STEPS,
    device="cuda",
    smoke=False,
):
    if device == "cuda":
        free_all()
    set_seed(seed)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    model = load_model(model_id, device, dtype=torch.float32)
    n_ckpt = apply_selective_checkpointing(model, ckpt_every, 0) if ckpt_every else 0
    layers = stacr_on_blocks(model, stacr_cfg, stacr_block_pred) if stacr_cfg else []
    if dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        model = DDP(
            model,
            device_ids=[local_rank] if device == "cuda" else None,
            static_graph=True,
            broadcast_buffers=True,
        )
    opt = build_optimizer(model, 5e-05, device == "cuda")
    train = get_batches(model_id, "train", bs, seq, steps * world, "wikitext", device)
    train = train[rank::world][:steps]
    if device == "cuda":
        sync()
        torch.cuda.reset_peak_memory_stats()
    t0 = None
    n_measured = 0
    for i, batch in enumerate(train):
        if layers:
            set_step(layers, i)
        if i == MEASURE_FROM:
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
        if smoke and i >= 2:
            break
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
        "tag": f"ddp_w{world}",
        "method": method,
        "seed": seed,
        "world_size": world,
        "model": model_id,
        "bs_per_rank": bs,
        "seq": seq,
        "steps": steps,
        "steps_measured": n_measured,
        "n_checkpointed_blocks": n_ckpt,
        "n_wrapped_layers": len(layers),
        "time_s": dt,
        "peak_bytes": peak,
        "peak_GiB": gib(peak),
        "peak_GiB_all_ranks": [gib(p) for p in peaks],
        "tokens_per_s_per_rank": bs * seq * n_measured / dt if dt == dt else None,
        "final_train_loss": float(loss),
    }
    del model, opt, train, layers
    if device == "cuda":
        free_all()
    return rec


def main():
    smoke = "--smoke" in sys.argv
    if smoke:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29517")
        dist.init_process_group("gloo", rank=0, world_size=1)
        for method, kw in ARMS:
            rec = one_run(
                method, 42, device="cpu", model_id="gpt2", bs=1, seq=128, steps=4, smoke=True, **kw
            )
            print(
                f"  smoke {method:22s} ok  wrapped={rec['n_wrapped_layers']} ckpt={rec['n_checkpointed_blocks']} loss={rec['final_train_loss']:.4f}",
                flush=True,
            )
        dist.destroy_process_group()
        return
    dist.init_process_group("nccl")
    (rank, world) = (dist.get_rank(), dist.get_world_size())
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    out = RESULTS / "x7_ddp.jsonl"
    if rank == 0:
        print(
            f"=== DDP world={world}  {MODEL} b={BS}/rank s={SEQ} steps={STEPS} {len(ARMS)} arms x {len(SEEDS)} seeds ===",
            flush=True,
        )
    for method, kw in ARMS:
        for seed in SEEDS:
            try:
                rec = one_run(method, seed, **kw)
            except torch.cuda.OutOfMemoryError:
                if rank == 0:
                    emit(
                        {"tag": f"ddp_w{world}", "method": method, "seed": seed, "status": "OOM"},
                        out,
                    )
                    print(f"  OOM {method} seed={seed}", flush=True)
                torch.cuda.empty_cache()
                continue
            except Exception as exc:
                if rank == 0:
                    emit(
                        {
                            "tag": f"ddp_w{world}",
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
            if rank == 0:
                emit(rec, out)
                print(
                    f"  {method:22s} seed={seed} peak/rank={rec['peak_GiB']:7.3f} GiB all={['%.3f' % p for p in rec['peak_GiB_all_ranks']]} t={rec['time_s']:6.1f}s wrapped={rec['n_wrapped_layers']}",
                    flush=True,
                )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
