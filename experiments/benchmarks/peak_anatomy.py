"""Peak anatomy."""

from __future__ import annotations
import torch
import os
import sys
from pathlib import Path
from .common import (
    RESULTS,
    apply_selective_checkpointing,
    amp_ctx,
    blocks_of,
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
(MODEL, BS, SEQ) = ("gpt2-medium", 8, 1024)
ALLOC = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "<unset>")
PHASE_MODELS = [("gpt2-medium", 8, 1024), ("gpt2-large", 4, 1024)]


def complement(k: int):
    return lambda i, _k=k: i % _k != 0


ARMS = [
    ("baseline", dict()),
    ("turbo_fast", dict(stacr_cfg=TURBO)),
    ("stacr_r8", dict(stacr_cfg=CFG8)),
    ("ckpt_k8", dict(ckpt_every=8)),
    ("ckpt_k4", dict(ckpt_every=4)),
    ("ckpt_k2", dict(ckpt_every=2)),
    ("ckpt_full", dict(ckpt_every=1)),
    ("ckpt_k8_plus_turbo", dict(ckpt_every=8, stacr_cfg=TURBO, stacr_block_pred=complement(8))),
    ("ckpt_k4_plus_turbo", dict(ckpt_every=4, stacr_cfg=TURBO, stacr_block_pred=complement(4))),
    ("ckpt_k2_plus_turbo", dict(ckpt_every=2, stacr_cfg=TURBO, stacr_block_pred=complement(2))),
]


def phase_run(
    method,
    ckpt_every=0,
    stacr_cfg=None,
    stacr_block_pred=None,
    steps=14,
    model_id=MODEL,
    bs=BS,
    seq=SEQ,
):
    free_all()
    set_seed(42)
    model = load_model(model_id, "cuda", dtype=torch.float32)
    attn = getattr(model.config, "_attn_implementation", "<unknown>")
    n_ckpt = apply_selective_checkpointing(model, ckpt_every, 0) if ckpt_every else 0
    layers = stacr_on_blocks(model, stacr_cfg, stacr_block_pred) if stacr_cfg else []
    opt = build_optimizer(model, 5e-05, True)
    train = get_batches(model_id, "train", bs, seq, steps, "wikitext", "cuda")
    marks: list[tuple[str, int, int]] = []

    def mark(name):
        marks.append((name, torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated()))

    blks = blocks_of(model)
    h1 = blks[-1].register_forward_hook(lambda *a: mark("end_last_block_fwd"))
    head = getattr(model, "lm_head", None)
    h2 = head.register_forward_hook(lambda *a: mark("after_lm_head")) if head is not None else None
    for i, batch in enumerate(train):
        if layers:
            set_step(layers, i)
        if i == 10:
            sync()
            torch.cuda.reset_peak_memory_stats()
            marks.clear()
            mark("step_start")
        opt.zero_grad(set_to_none=True)
        with amp_ctx(True):
            loss = model(**batch).loss
        if i == 10:
            mark("after_loss")
        loss.backward()
        if i == 10:
            mark("after_backward")
        opt.step()
        if i == 10:
            mark("after_opt_step")
            break
    h1.remove()
    if h2 is not None:
        h2.remove()
    peak = torch.cuda.max_memory_allocated()
    where = next((n for (n, _cur, mx) in marks if mx >= peak), "<not captured>")
    rec = {
        "probe": "phase",
        "alloc_conf": ALLOC,
        "attn_impl": str(attn),
        "model": model_id,
        "bs": bs,
        "seq": seq,
        "method": method,
        "n_checkpointed_blocks": n_ckpt,
        "n_wrapped_layers": len(layers),
        "peak_GiB": gib(peak),
        "peak_phase": where,
        "marks": [{"at": n, "cur_GiB": gib(c), "max_GiB": gib(m)} for (n, c, m) in marks],
    }
    del model, opt, train, layers
    free_all()
    return rec


def ce_run(chunked: bool, steps=14, chunk=128):
    free_all()
    set_seed(42)
    model = load_model(MODEL, "cuda", dtype=torch.float32)
    opt = build_optimizer(model, 5e-05, True)
    train = get_batches(MODEL, "train", BS, SEQ, steps, "wikitext", "cuda")
    losses = []
    for i, batch in enumerate(train):
        if i == 10:
            sync()
            torch.cuda.reset_peak_memory_stats()
        opt.zero_grad(set_to_none=True)
        with amp_ctx(True):
            if not chunked:
                loss = model(**batch).loss
            else:
                ids = batch["input_ids"]
                hid = model.transformer(input_ids=ids).last_hidden_state
                sh = hid[:, :-1, :].reshape(-1, hid.shape[-1])
                tg = ids[:, 1:].reshape(-1)
                tot = sh.new_zeros(())
                n = sh.shape[0]
                for a in range(0, n, chunk * BS):
                    z = model.lm_head(sh[a : a + chunk * BS])
                    tot = tot + torch.nn.functional.cross_entropy(
                        z.float(), tg[a : a + chunk * BS], reduction="sum"
                    )
                loss = tot / n
        loss.backward()
        opt.step()
        losses.append(float(loss))
        if i == 10:
            break
    peak = torch.cuda.max_memory_allocated()
    rec = {
        "probe": "ce",
        "alloc_conf": ALLOC,
        "model": MODEL,
        "bs": BS,
        "seq": SEQ,
        "method": "chunked_ce" if chunked else "fused_ce",
        "chunk_rows": chunk if chunked else None,
        "peak_GiB": gib(peak),
        "loss_at_probe": losses[-1],
    }
    del model, opt, train
    free_all()
    return rec


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "phase"
    out = RESULTS / "x0_peak_anatomy.jsonl"
    print(f"alloc_conf = {ALLOC}", flush=True)
    if mode == "phase":
        for model_id, bs, seq in PHASE_MODELS:
            print(f"--- {model_id} b={bs} s={seq} ---", flush=True)
            for method, kw in ARMS:
                try:
                    rec = phase_run(method, model_id=model_id, bs=bs, seq=seq, **kw)
                except Exception as exc:
                    print(f"  FAIL {method}: {type(exc).__name__}: {exc}", flush=True)
                    emit(
                        {
                            "probe": "phase",
                            "model": model_id,
                            "method": method,
                            "alloc_conf": ALLOC,
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                        out,
                    )
                    free_all()
                    continue
                emit(rec, out)
                print(
                    f"  {method:22s} peak={rec['peak_GiB']:7.3f} GiB  peak_phase={rec['peak_phase']:22s} attn={rec['attn_impl']}",
                    flush=True,
                )
    elif mode == "ce":
        for chunked in (False, True):
            try:
                rec = ce_run(chunked)
            except Exception as exc:
                print(f"  FAIL ce chunked={chunked}: {type(exc).__name__}: {exc}", flush=True)
                emit(
                    {
                        "probe": "ce",
                        "method": "chunked_ce" if chunked else "fused_ce",
                        "alloc_conf": ALLOC,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    out,
                )
                free_all()
                continue
            emit(rec, out)
            print(
                f"  {rec['method']:12s} peak={rec['peak_GiB']:7.3f} GiB  loss={rec['loss_at_probe']:.5f}",
                flush=True,
            )
    else:
        raise SystemExit("Use --cohort phase or --cohort ce")


if __name__ == "__main__":
    main()
