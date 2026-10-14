from __future__ import annotations
import contextlib
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
from stacr.surgery import apply_stacr, select_target_names, set_step
from stacr.layers import STACRLinear

RESULTS = Path(os.environ.get("STACR_RESULTS", "outputs/benchmarks"))
OPTIONS: dict = {}
RUN_ID = os.environ.get("STACR_RUN_ID", "")


def configure(**options):
    """Set explicitly requested command-line overrides for one experiment process."""
    global RESULTS, RUN_ID
    import uuid

    OPTIONS.clear()
    OPTIONS.update({k: v for (k, v) in options.items() if v is not None})
    RESULTS = Path(OPTIONS.get("output", RESULTS))
    RUN_ID = OPTIONS.get("run_id") or uuid.uuid4().hex


def resolve_parameters(params):
    """Resolve overrides before either planning or running a measurement."""
    out = dict(params)
    if isinstance(out.get("stacr_cfg"), dict) and out.get("comparator") is None:
        out["stacr_cfg"] = {"turbo_after_step": None, **out["stacr_cfg"]}
    for key in (
        "model_id",
        "bs",
        "seq",
        "steps",
        "eval_batches",
        "device",
        "dataset",
        "eight_bit",
        "attn_impl",
        "validation_protocol",
    ):
        if key in OPTIONS:
            out[key] = OPTIONS[key]
    if "measure_from" in OPTIONS:
        out["measure_from"] = OPTIONS["measure_from"]
    elif "steps" in OPTIONS and "measure_from" in out:
        out["measure_from"] = min(out["measure_from"], max(0, out["steps"] - 1))
    return out


def runtime_metadata():
    """Actual library versions and device; never records environment secrets."""
    import importlib.metadata

    versions = {}
    for name in ("torch", "transformers", "datasets", "bitsandbytes", "peft"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "versions": versions,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
    }


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gib(x: int) -> float:
    return float(x) / 1024**3


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def free_all() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def nvml_used(dev: int = 0) -> float:
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(dev)
        return gib(pynvml.nvmlDeviceGetMemoryInfo(h).used)
    except Exception:
        return float("nan")


_TOK_CACHE: dict = {}


def _tokenizer(model_id: str):
    from transformers import AutoTokenizer

    if model_id not in _TOK_CACHE:
        tok = AutoTokenizer.from_pretrained(model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        _TOK_CACHE[model_id] = tok
    return _TOK_CACHE[model_id]


def _stream_blocks(tok, split: str, seq: int, n_blocks: int, dataset: str, skip_blocks: int = 0):
    from datasets import load_dataset

    if dataset == "wikitext":
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split, streaming=True)
        key = "text"
    elif dataset == "openwebtext":
        ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        key = "text"
    else:
        raise ValueError(dataset)
    buf: list[int] = []
    out: list[list[int]] = []
    for row in ds:
        t = row[key]
        if not t or not t.strip():
            continue
        buf.extend(tok(t).input_ids)
        while len(buf) >= seq:
            if skip_blocks:
                skip_blocks -= 1
            else:
                out.append(buf[:seq])
            buf = buf[seq:]
            if len(out) >= n_blocks:
                return out
    return out


_DATA_CACHE: dict = {}


def get_batches(
    model_id: str,
    split: str,
    bs: int,
    seq: int,
    n_batches: int,
    dataset: str = "wikitext",
    device: str = "cuda",
    skip_blocks: int = 0,
):
    if min(bs, seq, n_batches) <= 0:
        raise ValueError("Batch size, sequence length and batch count must be positive")
    tok = _tokenizer(model_id)
    key = (tok.name_or_path, split, seq, bs * n_batches, dataset, skip_blocks)
    if key not in _DATA_CACHE:
        _DATA_CACHE[key] = _stream_blocks(tok, split, seq, bs * n_batches, dataset, skip_blocks)
    blocks = _DATA_CACHE[key]
    n = min(n_batches, len(blocks) // bs)
    if n != n_batches:
        raise ValueError(
            f"Dataset supplied {n} complete batches; requested {n_batches}. Reduce the step count or select a larger dataset."
        )
    batches = []
    for i in range(n):
        ids = torch.tensor(blocks[i * bs : (i + 1) * bs], dtype=torch.long, device=device)
        batches.append({"input_ids": ids, "labels": ids.clone()})
    return batches


def load_model(
    model_id: str, device: str = "cuda", dtype=torch.float32, attn_impl: str | None = None
):
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification

    cls = AutoModelForSequenceClassification if "bert" in model_id.lower() else AutoModelForCausalLM
    extra = {"num_labels": 2} if cls is AutoModelForSequenceClassification else {}
    if attn_impl:
        extra["attn_implementation"] = attn_impl
    m = cls.from_pretrained(model_id, torch_dtype=dtype, **extra)
    m.config.use_cache = False
    return m.to(device).train()


def blocks_of(model) -> list[nn.Module]:
    for path in (
        "transformer.h",
        "model.layers",
        "gpt_neox.layers",
        "bert.encoder.layer",
        "model.decoder.layers",
    ):
        obj = model
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
            if isinstance(obj, (nn.ModuleList, list)):
                return list(obj)
        except AttributeError:
            continue
    raise RuntimeError(f"cannot locate blocks for {type(model).__name__}")


def apply_cpu_offload(model, every: int = 1, offset: int = 0) -> int:
    from stacr.composition import saved_tensors_on_cpu

    n = 0
    for i, blk in enumerate(blocks_of(model)):
        if every <= 0 or i % every != offset:
            continue
        orig = blk.forward

        def fwd(*a, __orig=orig, **kw):
            if torch.is_grad_enabled():
                with saved_tensors_on_cpu(pin_memory=True):
                    return __orig(*a, **kw)
            return __orig(*a, **kw)

        blk.forward = fwd
        n += 1
    return n


def freeze_surface_params(model, surface: str = "mlp", min_features: int = 256) -> int:
    from stacr.surgery import select_target_names

    names = set(select_target_names(model, surface=surface, min_features=min_features))
    n = 0
    for name, module in model.named_modules():
        if name in names:
            for p in module.parameters(recurse=False):
                p.requires_grad_(False)
                n += 1
    return n


def apply_selective_checkpointing(model, every: int, offset: int = 0) -> int:
    import torch.utils.checkpoint as cp

    n = 0
    for i, blk in enumerate(blocks_of(model)):
        if every <= 0 or i % every != offset:
            continue
        orig = blk.forward

        def fwd(*a, __orig=orig, **kw):
            if torch.is_grad_enabled():
                return cp.checkpoint(__orig, *a, use_reentrant=False, **kw)
            return __orig(*a, **kw)

        blk.forward = fwd
        n += 1
    return n


def restrict_cfg_to_blocks(model, cfg: dict, block_pred) -> dict:
    names = select_target_names(
        model,
        surface=str(cfg.get("surface", "mlp")),
        min_features=int(cfg.get("min_features", 256)),
    )
    blks = blocks_of(model)
    id2idx = {}
    for i, b in enumerate(blks):
        for _, mod in b.named_modules():
            id2idx[id(mod)] = i
    keep = []
    for nm in names:
        mod = model.get_submodule(nm)
        idx = id2idx.get(id(mod))
        if idx is not None and block_pred(idx):
            keep.append(nm)
    sub = dict(cfg)
    sub["surface"] = "none"
    sub["extra_names"] = keep
    return sub


def stacr_on_blocks(model, cfg: dict, block_pred=None) -> list[STACRLinear]:
    cfg = {"turbo_after_step": None, **cfg}
    if block_pred is None:
        return apply_stacr(model, cfg)
    return apply_stacr(model, restrict_cfg_to_blocks(model, cfg, block_pred))


class SSPLedger:
    def __init__(self):
        self.wrapped_bytes = 0
        self.payload_bytes = 0
        self.basis_bytes = 0
        self.calls = 0

    def probe(self, layers: list[STACRLinear], rows_per_call: dict[int, int], value_bytes: int):
        for lay in layers:
            rows = rows_per_call.get(id(lay), 0)
            if rows == 0:
                continue
            self.calls += 1
            self.wrapped_bytes += rows * lay.in_features * value_bytes
            sb = lay.stored_bytes(rows, value_bytes=value_bytes)
            residual = sb["residual"]
            if getattr(lay, "_turbo", None) is not None and lay._turbo():
                residual = 0
            self.payload_bytes += sb["coeff"] + residual
            self.basis_bytes += sb["basis"]

    def ssp(self, baseline_peak_bytes: float) -> dict:
        a_c = self.wrapped_bytes - (self.payload_bytes + self.basis_bytes)
        return {
            "wrapped_GiB": gib(self.wrapped_bytes),
            "payload_GiB": gib(self.payload_bytes + self.basis_bytes),
            "A_C_GiB": gib(a_c),
            "A_C_pos_GiB": gib(max(a_c, 0)),
            "SSP_pct": 100.0 * max(a_c, 0) / baseline_peak_bytes
            if baseline_peak_bytes
            else float("nan"),
            "layer_calls": self.calls,
        }


def instrument_rows(layers: list[STACRLinear]) -> tuple[dict, list]:
    rows: dict[int, int] = {}
    handles = []
    for lay in layers:

        def hook(mod, inp, _rows=rows):
            x = inp[0]
            _rows[id(mod)] = _rows.get(id(mod), 0) + x.numel() // x.shape[-1]

        handles.append(lay.register_forward_pre_hook(hook))
    return (rows, handles)


def build_optimizer(model, lr: float, eight_bit: bool):
    if eight_bit:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise RuntimeError(
                "This protocol requires bitsandbytes AdamW8bit; install stacr[comparators] or explicitly use --optimizer adamw."
            ) from exc
        return bnb.optim.AdamW8bit(model.parameters(), lr=lr)
    return torch.optim.AdamW(model.parameters(), lr=lr)


@contextlib.contextmanager
def amp_ctx(enabled: bool, device: str = "cuda"):
    if not enabled:
        yield
        return
    device_type = str(device).split(":")[0]
    dtype = (
        torch.bfloat16 if device_type == "cpu" or torch.cuda.is_bf16_supported() else torch.float16
    )
    with torch.autocast(device_type=device_type, dtype=dtype):
        yield


@torch.no_grad()
def validation_ppl(model, batches, amp: bool) -> float:
    was_training = model.training
    model.eval()
    (tot_nll, tot_tok) = (0.0, 0)
    for batch in batches:
        with amp_ctx(amp, batch["input_ids"].device.type):
            output = model(**batch)
        ntok = batch["labels"].numel() - batch["labels"].shape[0]
        tot_nll += float(output.loss) * ntok
        tot_tok += ntok
    model.train(was_training)
    return math.exp(tot_nll / max(tot_tok, 1))


def run_config(
    *,
    model_id: str,
    bs: int,
    seq: int,
    steps: int,
    seed: int,
    method: str,
    stacr_cfg: dict | None = None,
    ckpt_every: int = 0,
    ckpt_offset: int = 0,
    stacr_block_pred=None,
    comparator: str | None = None,
    amp: bool = True,
    eight_bit: bool = True,
    lr: float = 5e-05,
    dataset: str = "wikitext",
    eval_batches: int = 16,
    measure_from: int = 8,
    baseline_peak_bytes: float | None = None,
    dtype=torch.float32,
    attn_impl: str | None = None,
    tag: str = "",
    offload_every: int = 0,
    offload_offset: int = 0,
    freeze_surface: bool = False,
    device: str = "cuda",
    validation_protocol: str = "disjoint",
) -> dict:
    """Train one matched cell, retaining the complete resolved protocol in its output."""
    resolved = resolve_parameters(locals())
    stacr_cfg = resolved["stacr_cfg"]
    (model_id, bs, seq, steps) = (resolved[k] for k in ("model_id", "bs", "seq", "steps"))
    (eval_batches, measure_from) = (resolved[k] for k in ("eval_batches", "measure_from"))
    (device, dataset, eight_bit, attn_impl, validation_protocol) = (
        resolved[k] for k in ("device", "dataset", "eight_bit", "attn_impl", "validation_protocol")
    )
    if min(bs, seq, steps, eval_batches) <= 0 or not 0 <= measure_from < steps:
        raise ValueError("Positive sizes and 0 <= measure_from < steps are required")
    if str(device).startswith("cuda") and (not torch.cuda.is_available()):
        raise RuntimeError(
            "CUDA is required for this run; use --device cpu --optimizer adamw for a wiring check"
        )
    if validation_protocol not in ("disjoint", "shared-prefix"):
        raise ValueError("validation_protocol must be disjoint or shared-prefix")
    free_all()
    set_seed(seed)
    model = load_model(model_id, device, dtype=dtype, attn_impl=attn_impl)
    n_ckpt = apply_selective_checkpointing(model, ckpt_every, ckpt_offset) if ckpt_every else 0
    n_offload = apply_cpu_offload(model, offload_every, offload_offset) if offload_every else 0
    n_frozen = (
        freeze_surface_params(
            model,
            str((stacr_cfg or {}).get("surface", "mlp")),
            int((stacr_cfg or {}).get("min_features", 256)),
        )
        if freeze_surface
        else 0
    )
    layers = []
    if stacr_cfg is not None:
        if comparator is not None:
            from comparators.activation import apply_comparator

            cfg = (
                restrict_cfg_to_blocks(model, stacr_cfg, stacr_block_pred)
                if stacr_block_pred
                else stacr_cfg
            )
            layers = apply_comparator(model, comparator, cfg)
        else:
            layers = stacr_on_blocks(model, stacr_cfg, stacr_block_pred)
    optimizer = build_optimizer(model, lr, eight_bit)
    train = get_batches(model_id, "train", bs, seq, steps, dataset, device)
    skip = bs * steps if dataset == "openwebtext" and validation_protocol == "disjoint" else 0
    validation = get_batches(
        model_id, "validation", bs, seq, eval_batches, dataset, device, skip_blocks=skip
    )
    ledger = SSPLedger()
    (rows_map, handles) = instrument_rows(layers) if layers else ({}, [])
    losses = []
    cuda = str(device).startswith("cuda")
    t0 = None
    try:
        for i, batch in enumerate(train):
            set_step(layers, i)
            if i == measure_from:
                sync()
                if cuda:
                    torch.cuda.reset_peak_memory_stats()
                rows_map.clear()
                t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with amp_ctx(amp, device):
                loss = model(**batch).loss
            loss_value = float(loss.detach())
            if not math.isfinite(loss_value):
                raise FloatingPointError(f"Nonfinite training loss at batch {i}: {loss_value}")
            loss.backward()
            optimizer.step()
            losses.append(loss_value)
            if layers and i == measure_from:
                ledger.probe(
                    layers, rows_map, 2 if amp else torch.empty((), dtype=dtype).element_size()
                )
        sync()
        elapsed = time.perf_counter() - t0
        peak = int(torch.cuda.max_memory_allocated()) if cuda else 0
        ppl = validation_ppl(model, validation, amp)
        measured_steps = len(losses) - measure_from
        out = {
            "tag": tag,
            "method": method,
            "model": model_id,
            "bs": bs,
            "seq": seq,
            "steps": steps,
            "steps_ran": len(losses),
            "steps_measured": measured_steps,
            "seed": seed,
            "amp": amp,
            "eight_bit": eight_bit,
            "dataset": dataset,
            "comparator": comparator,
            "dtype": str(dtype),
            "device_type": str(device),
            "attn_impl": getattr(model.config, "_attn_implementation", None),
            "ckpt_every": ckpt_every,
            "ckpt_offset": ckpt_offset,
            "n_checkpointed_blocks": n_ckpt,
            "n_wrapped_layers": len(layers),
            "wrapped_layers": [layer.source_name for layer in layers],
            "n_offloaded_blocks": n_offload,
            "offload_every": offload_every,
            "offload_offset": offload_offset,
            "offload_saved_tensor_layout": "preserve_format" if n_offload else None,
            "n_frozen_params": n_frozen,
            "peak_bytes": peak,
            "peak_GiB": gib(peak),
            "peak_source": "torch.cuda.max_memory_allocated" if cuda else "unavailable",
            "nvml_GiB": nvml_used(torch.cuda.current_device()) if cuda else None,
            "time_s": elapsed,
            "tokens_per_s": bs * seq * measured_steps / elapsed,
            "val_ppl": ppl,
            "final_train_loss": sum(losses[-20:]) / len(losses[-20:]),
            "stacr_cfg": stacr_cfg,
            "lr": lr,
            "eval_batches": eval_batches,
            "measure_from": measure_from,
            "validation_protocol": validation_protocol,
            "validation_overlaps_training": dataset == "openwebtext" and skip == 0,
            "validation_skip_blocks": skip,
            "data_storage": str(device),
            "optimizer_actual": type(optimizer).__module__ + "." + type(optimizer).__name__,
            "runtime": runtime_metadata(),
        }
        if layers:
            out["ssp"] = ledger.ssp(baseline_peak_bytes or peak)
        return out
    finally:
        for handle in handles:
            handle.remove()
        del model, optimizer, layers, train, validation
        free_all()


def emit(record: dict, path: Path) -> None:
    """Append one result with an explicit run identifier."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched = dict(record)
    enriched.setdefault("run_id", RUN_ID)
    enriched.setdefault("runtime", runtime_metadata())
    enriched.setdefault("protocol_overrides", OPTIONS)

    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for (k, v) in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, float) and (not math.isfinite(value)):
            return None
        if isinstance(value, Path) or isinstance(value, torch.dtype):
            return str(value)
        return value

    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(clean(enriched), allow_nan=False) + "\n")
    keys = ("tag", "method", "seed", "peak_GiB", "time_s", "val_ppl")
    print("  " + "  ".join((f"{key}={record.get(key)}" for key in keys)), flush=True)


def _cos(a, b) -> float:
    a = a.reshape(-1).double()
    b = b.reshape(-1).double()
    d = a.norm() * b.norm()
    return float(a @ b / d) if float(d) > 0 else float("nan")


def gradient_fidelity(
    *,
    model_id: str,
    bs: int,
    seq: int,
    seed: int,
    steps: int,
    stacr_cfg: dict,
    comparator: str | None = None,
    amp: bool = False,
    dataset: str = "wikitext",
    tag: str = "",
    device: str = "cuda",
) -> dict:
    resolved = resolve_parameters(locals())
    stacr_cfg = resolved["stacr_cfg"]
    model_id, bs, seq, steps, device, dataset = (
        resolved[k] for k in ("model_id", "bs", "seq", "steps", "device", "dataset")
    )
    free_all()
    dev = device
    set_seed(seed)
    ref = load_model(model_id, dev)
    set_seed(seed)
    test = load_model(model_id, dev)
    test.load_state_dict(ref.state_dict())
    ref.train()
    test.train()
    if comparator is not None:
        from comparators.activation import apply_comparator

        layers = apply_comparator(test, comparator, stacr_cfg)
    else:
        layers = apply_stacr(test, stacr_cfg)
    wrapped = {lay.source_name for lay in layers}
    if not layers:
        raise ValueError("Gradient fidelity requires at least one wrapped layer")
    batches = get_batches(model_id, "train", bs, seq, steps, dataset, dev)
    (full, local, worst_un, loss_gap) = ([], [], [], [])
    for i, batch in enumerate(batches):
        set_step(layers, i)
        ref.zero_grad(set_to_none=True)
        set_seed(seed + 1000 + i)
        with amp_ctx(amp, dev):
            lr_ = ref(**batch).loss
        lr_.backward()
        g_ref = {
            n: p.grad.detach().clone() for (n, p) in ref.named_parameters() if p.grad is not None
        }
        test.zero_grad(set_to_none=True)
        set_seed(seed + 1000 + i)
        with amp_ctx(amp, dev):
            lt_ = test(**batch).loss
        lt_.backward()
        loss_gap.append(abs(float(lr_) - float(lt_)))
        num = den_a = den_b = 0.0
        (loc, un) = ([], [])
        for n, p in test.named_parameters():
            if p.grad is None or n not in g_ref:
                continue
            a = g_ref[n].reshape(-1).double()
            b = p.grad.detach().reshape(-1).double()
            num += float(a @ b)
            den_a += float(a @ a)
            den_b += float(b @ b)
            base = n.rsplit(".", 1)[0]
            (loc if base in wrapped else un).append(_cos(a, b))
        full.append(num / max(den_a**0.5 * den_b**0.5, 1e-30))
        if loc:
            local.append(sum(loc) / len(loc))
        if un:
            worst_un.append(min(un))
    out = {
        "tag": tag,
        "model": model_id,
        "bs": bs,
        "seq": seq,
        "seed": seed,
        "comparator": comparator or "stacr",
        "rank": stacr_cfg.get("rank"),
        "residual_budget": stacr_cfg.get("residual_budget"),
        "n_wrapped_layers": len(layers),
        "steps": len(full),
        "mean_cosine_full_model": sum(full) / max(len(full), 1),
        "min_cosine_full_model": min(full) if full else float("nan"),
        "mean_cosine_local_wrapped": sum(local) / max(len(local), 1) if local else float("nan"),
        "worst_unwrapped_cosine": min(worst_un) if worst_un else float("nan"),
        "mean_loss_gap": sum(loss_gap) / max(len(loss_gap), 1),
    }
    out["forward_mode"] = "train_rng_pinned"
    out["stacr_cfg"] = dict(stacr_cfg)
    out["amp"] = amp
    out["dataset"] = dataset
    out["device_type"] = str(device)
    out["compression_training_layers"] = sum(layer.training for layer in layers)
    out["initialized_bases"] = sum(bool(layer.basis_ready) for layer in layers)
    del ref, test, layers, batches
    free_all()
    return out
