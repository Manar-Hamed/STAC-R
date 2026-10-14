"""Transformer training with adaptive activation compression and fixed experimental protocols."""

from __future__ import annotations
import sys
from pathlib import Path

import argparse
from collections import deque
import fnmatch
import hashlib
import json
import logging
import math
import os

_project_root = Path(os.environ.get("STACR_CACHE", ".cache/stacr"))
import re
import time
import warnings
from contextlib import nullcontext
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from experiments.protocols.logging import RunLogger, NullLogger, TrainConfig
from stacr.protocols.config.training import BaselineConfig
from stacr.protocols.profiling import reset_peaks, read_peaks, PhaseProfiler
from stacr.protocols.utils import set_all_seeds
from stacr.protocols.baselines.checkpointing import apply_checkpointing
from stacr.protocols.baselines.amp import AMPManager, resolve_amp_dtype
from stacr.protocols.baselines.offload import SavedTensorOffloader
from stacr.protocols.baselines.saved_tensor_compressor import SavedTensorCompressor
from stacr.protocols.baselines.actnn_adapter import try_enable_actnn
from stacr.protocols.baselines.gact_adapter import try_enable_gact
from stacr.protocols.baselines.torchao_adapter import try_enable_torchao
from stacr.protocols.baselines.galore_adapter import (
    try_enable_galore,
    parse_galore_config_string,
    GaLoreConfig,
)
from stacr.adaptive import (
    STACRConfig,
    STACRLinear,
    _resolve_basis_dtype,
    get_workspace_pool,
    set_stacr_runtime_state,
    quantize_int8,
    dequantize_int8,
    quantize_int4,
    dequantize_int4,
)
from stacr.protocols.attention import apply_stacr_to_attention
from stacr.protocols.checkpointing import (
    collect_c1_site_candidates,
    clear_active_c1_runtime,
    enable_c1_checkpointing_for_model,
    get_active_c1_runtime,
)

logger = logging.getLogger(__name__)


def _build_hf_map_cache_paths(
    config: "ConvergenceConfig",
    dataset_tag: str,
    stage: str,
    splits: List[str],
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, str], Path]:
    """Build deterministic cache paths for HuggingFace Dataset.map stages.

    Parallel WT-103/SST-2 runs can race on the default shared Arrow cache and
    produce truncated files. We use deterministic cache filenames plus a small
    stage-level file lock so concurrent runs either build once or wait and load
    the completed cache.
    """
    payload = {
        "dataset_tag": dataset_tag,
        "stage": stage,
        "task": getattr(config, "task", None),
        "model": getattr(config, "model", None),
        "max_length": getattr(config, "max_length", None),
        "padding_mode": getattr(config, "padding_mode", None),
        "seed": getattr(config, "seed", None),
        "extra": extra or {},
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    cache_dir = _project_root / ".hf_map_cache" / dataset_tag / digest
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_files = {split: str(cache_dir / f"{stage}_{split}.arrow") for split in splits}
    lock_path = cache_dir / f"{stage}.lock"
    return (cache_files, lock_path)


class _MapStageLock:
    """Tiny cross-platform file lock for dataset preprocessing stages."""

    def __init__(self, lock_path: Path, timeout_s: float = 900.0, poll_s: float = 0.2):
        self.lock_path = lock_path
        self.timeout_s = timeout_s
        self.poll_s = poll_s
        self._fd: Optional[int] = None

    def __enter__(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.time() + self.timeout_s
        while True:
            try:
                self._fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.write(self._fd, str(os.getpid()).encode("utf-8"))
                return self
            except FileExistsError:
                if time.time() >= deadline:
                    raise TimeoutError(f"Timed out waiting for dataset map lock: {self.lock_path}")
                time.sleep(self.poll_s)

    def __exit__(self, exc_type, exc, tb):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            pass
        return False


def _parse_galore_target_modules(raw: Optional[str]) -> Optional[List[str]]:
    """Parse comma-separated GaLore target modules string."""
    if raw is None:
        return None
    normalized = raw.strip()
    if not normalized or normalized.lower() == "none":
        return None
    return [part.strip() for part in normalized.split(",") if part.strip()]


@dataclass
class ConvergenceConfig:
    """Configuration for convergence training experiments."""

    experiment: str = "convergence_training"
    task: str = "wikitext103"
    model: Optional[str] = None
    method: str = "baseline"
    method_label: Optional[str] = None
    run_tag: Optional[str] = None
    rank: int = 8
    reshape_shape: Optional[Tuple[int, ...]] = None
    factor_quantization_bits: Optional[int] = None
    factor_native_backward: bool = True
    use_lowrank_svd: bool = False
    tact_v3_config_path: Optional[str] = None
    tact_v3_target_ratio: int = 30
    baseline: Optional[str] = None
    baseline_cfg: BaselineConfig = field(default_factory=BaselineConfig)
    epochs: int = 125
    batch_size: int = 128
    lr: float = 0.1
    weight_decay: float = 0.0005
    momentum: float = 0.9
    max_length: int = 128
    padding_mode: str = "dynamic"
    seed: int = 42
    deterministic: bool = True
    measure_memory: bool = True
    log_memory_both: bool = True
    monitor_grad_fidelity: bool = False
    grad_fidelity_interval: int = 100
    monitor_per_layer_grad_fidelity: bool = False
    per_layer_grad_fidelity_interval: int = 120
    monitor_svd_per_layer_grad_fidelity: bool = False
    svd_per_layer_grad_fidelity_interval: int = 240
    svd_per_layer_rank: Optional[int] = None
    svd_per_layer_oversampling: int = 16
    svd_per_layer_niter: int = 2
    monitor_activation_spectrum: bool = False
    activation_spectrum_interval: int = 120
    activation_spectrum_rank: int = 8
    activation_spectrum_oversampling: int = 16
    activation_spectrum_niter: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    use_data_parallel: bool = False
    use_ddp: bool = False
    use_tensor_parallel: bool = False
    use_8bit_optimizer: bool = False
    tp_scope: str = "full"
    tp_stacr_scope: str = "row_only"
    tp_profile_activations: bool = False
    max_steps: Optional[int] = None
    grad_clip: float = 0.0
    save_final_model: bool = False
    save_every_n_steps: Optional[int] = None
    debug_step_metrics_every: int = 0
    phase_profile_full_run: bool = False
    saved_tensors_compress_min_bytes: int = 100000
    saved_tensors_compress_dims: int = 4
    quiet: bool = False
    is_canonical_run: bool = False
    stacr_residual_mode: str = "block_sparse"
    stacr_residual_budget: float = 0.005
    stacr_residual_threshold: float = 0.1
    stacr_update_trigger: str = "hybrid"
    stacr_update_interval: int = 100
    stacr_energy_threshold: float = 0.05
    stacr_streaming_algorithm: str = "oja"
    stacr_oja_learning_rate: float = 0.01
    stacr_warmup_steps: int = 0
    stacr_freeze_basis_after_warmup: bool = True
    stacr_quantize_coefficients: bool = True
    stacr_quantize_bits: int = 8
    stacr_quantize_residuals: bool = True
    stacr_residual_quantize_bits: int = 4
    stacr_randomized_svd_threshold: int = 1024
    stacr_randomized_svd_oversampling: int = 4
    stacr_randomized_svd_niter: int = 2
    stacr_adaptive_oja_lr: bool = False
    stacr_adaptive_oja_scale: float = 5.0
    stacr_adaptive_oja_max_lr: float = 0.1
    stacr_basis_dtype: str = "auto"
    stacr_backward_mode: str = "sparse"
    stacr_residual_block_size: int = 16
    stacr_cpu_offload: bool = False
    stacr_use_mgs: bool = False
    stacr_lazy_update: bool = True
    stacr_lazy_update_threshold: float = 0.01
    stacr_use_fused_kernel: bool = False
    stacr_rank_mode: str = "fixed"
    stacr_rank_variance_threshold: float = 0.95
    stacr_rank_min: int = 4
    stacr_rank_max: int = 64
    stacr_rank_adjust_interval: int = 100
    stacr_rank_sensitivity_low: float = 0.03
    stacr_rank_sensitivity_high: float = 0.1
    stacr_rank_step_fraction: float = 0.125
    stacr_streaming_topk: bool = False
    stacr_streaming_topk_sample_ratio: float = 0.1
    stacr_chunk_size: int = 512
    stacr_index_compression: str = "none"
    stacr_skip_residual: bool = False
    stacr_skip_residual_threshold: float = 0.05
    stacr_buffer_reuse: bool = False
    stacr_min_features: int = 0
    stacr_small_layer_rank_scale: float = 1.0
    stacr_small_layer_residual_scale: float = 1.0
    stacr_decoder_target_mode: str = "all"
    stacr_use_planner: bool = False
    stacr_planner_enforce: bool = False
    stacr_force_compress_layers: Optional[str] = None
    stacr_adaptive_layer_rank: str = "none"
    stacr_per_layer_rank_schedule: Optional[str] = None
    stacr_residual_schedule: str = "none"
    stacr_residual_start: float = 0.05
    stacr_residual_end: float = 0.001
    stacr_post_warmup_mode: str = "full"
    stacr_residual_warmup_steps: int = 0
    stacr_post_warmup_threshold: float = 0.1
    stacr_refresh_trigger: str = "off"
    stacr_refresh_period: int = 1000
    stacr_refresh_energy_threshold: float = 0.1
    stacr_cache_basis_per_optimizer_step: bool = False
    stacr_oja_update_per_optimizer_step: bool = False
    stacr_enable_low_rank_only_fast_path: bool = True
    stacr_use_random_qr_init: bool = False
    stacr_min_adaptation_steps: int = 50
    stacr_allow_unadapted_random_qr: bool = False
    stacr_attention: bool = False
    stacr_attention_rank: Optional[int] = None
    stacr_attention_residual_mode: Optional[str] = None
    stacr_compress_checkpoint: bool = False
    stacr_ckpt_target: str = "decoder_block_input"
    stacr_ckpt_rank: Optional[int] = None
    stacr_ckpt_warmup_steps: int = 500
    stacr_ckpt_residual_mode: str = "none"
    stacr_ckpt_residual_budget: float = 0.005
    stacr_ckpt_refresh_trigger: str = "residual_energy"
    stacr_ckpt_refresh_period: int = 200
    stacr_ckpt_refresh_energy_threshold_mode: str = "auto"
    stacr_ckpt_refresh_energy_threshold: float = 0.15
    stacr_ckpt_min_numel: int = 1 << 20
    stacr_ckpt_basis_dtype: str = "auto"
    stacr_ckpt_quantize_coefficients: bool = False
    stacr_ckpt_quantize_bits: int = 8
    stacr_ckpt_quantize_residuals: bool = True
    stacr_ckpt_residual_quantize_bits: int = 4
    stacr_ckpt_allow_inner_stacr: bool = False
    stacr_ckpt_planner: str = "heuristic"
    prac_update_proj_gap: int = 50
    prac_seed_base: int = 42
    prac_proj_dtype: str = "auto"
    stacr_basis_sharing: str = "none"
    stacr_joint_qk: bool = False
    stacr_gpt2_split_c_attn: bool = False
    use_neighbor_sampling: bool = False
    num_neighbors: str = "30,30"
    use_random_node_loader: bool = False
    num_parts: int = 40
    use_stacr_graph: bool = False
    stacr_graph_tt_rank: int = 8
    stacr_graph_max_rank: int = 16
    stacr_graph_sparse_residual: bool = False
    stacr_graph_residual_budget: float = 0.1
    stacr_graph_residual_per_node: bool = False
    stacr_graph_chunk_size: int = 64
    stacr_graph_edge_eval_chunk_size: int = 100000
    stacr_graph_use_phantom: bool = False
    stacr_graph_phantom_samples: int = 5000
    stacr_graph_phantom_budget_frac: float = 0.2
    stacr_graph_phantom_seed: Optional[int] = None
    stacr_graph_use_hybrid: bool = False
    stacr_graph_hybrid_global_frac: float = 0.8
    stacr_graph_share_tt: bool = True
    stacr_graph_random_residual: bool = False
    stacr_graph_random_seed: int = 42
    stacr_graph_node_reorder: str = "none"
    conv_compression_mode: str = "none"
    conv_compression_layers: str = "all"
    conv_skip_3x3: bool = True
    conv_min_channels: int = 64
    tucker_ranks: Optional[Tuple[int, int, int]] = None
    conv_v2: bool = False
    conv_checkpoint: bool = False
    conv_block_checkpoint: bool = False
    conv_stage_checkpoint: bool = False
    conv_saved_hooks: bool = False
    conv_hooks_mode: str = "channel"

    def __post_init__(self):
        if os.environ.get("STACR_QUIET") == "1":
            self.quiet = True
        if self.method_label is None:
            self.method_label = self.method
        if self.baseline_cfg.is_default() and self.baseline not in (None, "none", "baseline"):
            self.baseline_cfg = BaselineConfig.from_legacy(self.baseline)
        if self.task == "sst2":
            if self.epochs == 125:
                self.epochs = 3
            if self.lr == 0.1:
                self.lr = 2e-05
            if self.batch_size == 128:
                self.batch_size = 32
        elif self.task == "wikitext103":
            if self.epochs == 125:
                self.epochs = 3
            if self.lr == 0.1:
                self.lr = 5e-05
            if self.batch_size == 128:
                self.batch_size = 4
            if self.max_length == 128:
                self.max_length = 512
        elif self.task == "ogbn-arxiv":
            if self.epochs == 125:
                self.epochs = 500
            if self.lr == 0.1:
                self.lr = 0.01
        if self.padding_mode not in ("dynamic", "max_length"):
            raise ValueError(
                f"padding_mode must be 'dynamic' or 'max_length', got {self.padding_mode}"
            )


def measure_memory_for_cap() -> Optional[int]:
    """For iso-peak calibration: use RESERVED (conservative)."""
    peaks = read_peaks()
    if peaks.get("device_mode") == "cuda":
        return peaks.get("max_reserved_bytes")
    return None


def measure_memory_for_report() -> Optional[int]:
    """For paper figures/tables: use ALLOCATED (standard)."""
    peaks = read_peaks()
    if peaks.get("device_mode") == "cuda":
        return peaks.get("max_allocated_bytes")
    return None


def _current_allocated_bytes(device: torch.device) -> Optional[int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        return int(torch.cuda.memory_allocated(device))
    except Exception:
        return None


def log_memory_complete(logger: Any, step: int) -> Dict[str, Any]:
    """Log BOTH metrics for complete picture."""
    peaks = read_peaks()
    allocated = peaks.get("max_allocated_bytes")
    reserved = peaks.get("max_reserved_bytes")
    memory_info: Dict[str, Any] = {
        "step": step,
        "device_mode": peaks.get("device_mode"),
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
    }
    if (
        isinstance(allocated, (int, float))
        and isinstance(reserved, (int, float))
        and (allocated > 0)
    ):
        overhead = max(0, reserved - allocated)
        memory_info["allocator_overhead_pct"] = 100 * overhead / allocated
    logger.log_profile(memory_info)
    return memory_info


def _get_phase_profiler(config: ConvergenceConfig) -> Optional[PhaseProfiler]:
    profiler = getattr(config, "_phase_profiler", None)
    if profiler is not None and getattr(profiler, "enabled", False):
        return profiler
    return None


def _phase_profile_context(config: ConvergenceConfig, name: str):
    profiler = _get_phase_profiler(config)
    if profiler is not None:
        return profiler.phase(name)
    return nullcontext()


def _attach_phase_profiler_to_stacr_layers(model: nn.Module, profiler: PhaseProfiler) -> int:
    """Attach a PhaseProfiler to STAC-R modules for layer-local subphases."""
    attached = 0
    for module in model.modules():
        if isinstance(module, STACRLinear):
            module.layer_state["phase_profiler"] = profiler
            attached += 1
    return attached


def _ddp_reduce_train_metrics(metrics: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """All-reduce raw training accumulators across DDP ranks, then recompute metrics.

    Uses raw counts (_raw_total_loss, _raw_correct, _raw_total, _raw_total_tokens,
    _raw_steps) from train_epoch to produce correct global train_loss, train_accuracy,
    and throughput. This avoids the bias of averaging pre-divided per-rank values.
    """
    if not dist.is_initialized():
        return metrics
    raw = torch.tensor(
        [
            metrics.get("_raw_total_loss", 0.0),
            metrics.get("_raw_correct", 0.0),
            metrics.get("_raw_total", 0.0),
            metrics.get("_raw_total_tokens", 0.0),
            metrics.get("_raw_steps", 0.0),
        ],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(raw, op=dist.ReduceOp.SUM)
    (total_loss, correct, total, total_tokens, steps) = raw.tolist()
    metrics["train_loss"] = total_loss / max(steps, 1)
    metrics["train_accuracy"] = correct / max(total, 1) * 100
    epoch_time_t = torch.tensor([metrics.get("epoch_time_s", 1.0)], device=device)
    dist.all_reduce(epoch_time_t, op=dist.ReduceOp.MAX)
    epoch_time = epoch_time_t.item()
    metrics["epoch_time_s"] = epoch_time
    for key in ("tokens_per_s", "samples_per_s"):
        if key in metrics:
            if key == "tokens_per_s":
                metrics[key] = total_tokens / max(epoch_time, 1e-06)
            else:
                metrics[key] = total / max(epoch_time, 1e-06)
    return metrics


def init_ddp():
    """Initialize DDP from torchrun environment variables.

    Returns (local_rank, world_size, is_main_process).
    Must be called before any model/data creation.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return (local_rank, world_size, rank == 0)


def cleanup_ddp():
    """Clean up DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model_ddp(model, local_rank):
    """Wrap model with DistributedDataParallel.

    STAC-R saved-tensor hooks are per-rank (operate on local activations)
    and do not cross device boundaries. DDP only all-reduces gradients
    after backward, so hooks are fully compatible.
    """
    return DDP(
        model, device_ids=[local_rank], find_unused_parameters=False, broadcast_buffers=False
    )


def make_distributed_loader(
    dataset, batch_size, seed, num_workers=4, collate_fn=None, drop_last=False, shuffle=True
):
    """Create a DataLoader with DistributedSampler for DDP."""
    sampler = DistributedSampler(dataset, shuffle=shuffle, seed=seed)
    return (
        DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=drop_last,
        ),
        sampler,
    )


def setup_sst2_data(config: ConvergenceConfig) -> Tuple[DataLoader, DataLoader, Any]:
    """Load and preprocess SST-2 dataset.

    Returns:
        (train_loader, val_loader, tokenizer)
    """
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer, DataCollatorWithPadding
    except ImportError:
        raise RuntimeError(
            "HuggingFace libraries required for SST-2.\nInstall with: pip install transformers datasets"
        )
    if config.quiet:
        try:
            from datasets.utils.logging import disable_progress_bar as _disable_hf_bar

            _disable_hf_bar()
        except Exception:
            pass
        try:
            from transformers.utils.logging import disable_progress_bar as _disable_tx_bar

            _disable_tx_bar()
        except Exception:
            pass
    raw_datasets = load_dataset("nyu-mll/glue", "sst2")
    model_name = getattr(config, "model", None) or "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_function(examples):
        result = tokenizer(
            examples["sentence"],
            truncation=True,
            max_length=config.max_length,
            padding="max_length" if config.padding_mode == "max_length" else False,
        )
        result["labels"] = examples["label"]
        return result

    columns_to_remove = [c for c in raw_datasets["train"].column_names if c not in ["label"]]
    (tokenized_cache_files, tokenized_lock) = _build_hf_map_cache_paths(
        config,
        dataset_tag="glue_sst2",
        stage="tokenize",
        splits=list(raw_datasets.keys()),
        extra={"tokenizer_name": model_name, "columns_to_remove": columns_to_remove},
    )
    with _MapStageLock(tokenized_lock):
        tokenized_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=columns_to_remove,
            cache_file_names=tokenized_cache_files,
            load_from_cache_file=True,
        )
    train_lengths = [len(ex["input_ids"]) for ex in tokenized_datasets["train"]]
    if not config.quiet:
        print(
            f"SST-2 train lengths: min={min(train_lengths)}, max={max(train_lengths)}, mean={sum(train_lengths) / len(train_lengths):.1f}"
        )
    if config.padding_mode == "max_length":
        data_collator = DataCollatorWithPadding(
            tokenizer, padding="max_length", max_length=config.max_length
        )
    else:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    g = torch.Generator()
    g.manual_seed(config.seed)
    train_loader = DataLoader(
        tokenized_datasets["train"],
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=data_collator,
        generator=g,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        tokenized_datasets["validation"],
        batch_size=config.batch_size,
        collate_fn=data_collator,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    return (train_loader, val_loader, tokenizer)


def create_bert_sst2(model_name: Optional[str] = None) -> nn.Module:
    """Create BERT model for SST-2 classification.

    Args:
        model_name: HuggingFace model ID (default: bert-base-uncased).
                    Supports bert-large-uncased, etc.
    """
    try:
        from transformers import AutoModelForSequenceClassification
    except ImportError:
        raise RuntimeError("transformers required: pip install transformers")
    model_name = model_name or "bert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    return model


def setup_wikitext_data(config: ConvergenceConfig) -> Tuple[DataLoader, DataLoader, Any]:
    """Load and preprocess WikiText-103 dataset for causal language modeling.

    WikiText-103 contains 103M training tokens from Wikipedia articles.
    We use GPT-2 tokenizer and chunk text into max_length sequences.

    Returns:
        (train_loader, val_loader, tokenizer)
    """
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError("WikiText-103 requires: pip install datasets transformers") from e
    if config.quiet:
        try:
            from datasets.utils.logging import disable_progress_bar as _disable_hf_bar

            _disable_hf_bar()
        except Exception:
            pass
        try:
            from transformers.utils.logging import disable_progress_bar as _disable_tx_bar

            _disable_tx_bar()
        except Exception:
            pass
    tokenizer_name = getattr(config, "model", None) or "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    raw_datasets = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    max_length = config.max_length
    try:
        from transformers import AutoConfig

        _model_name_check = getattr(config, "model", None) or "gpt2"
        _model_cfg = AutoConfig.from_pretrained(_model_name_check)
        _model_max_len = int(getattr(_model_cfg, "max_position_embeddings", 1024))
    except Exception:
        _model_name_check = "gpt2"
        _model_max_len = 1024
    if max_length > _model_max_len:
        raise ValueError(
            f"max_length={max_length} exceeds {_model_name_check} max_position_embeddings={_model_max_len}. Use --max-length {_model_max_len} or less."
        )

    def tokenize_function(examples):
        """Tokenize all texts."""
        return tokenizer(examples["text"], truncation=True, max_length=max_length)

    def group_texts(examples):
        """Concatenate all texts and split into chunks of max_length."""
        concatenated_ids = sum(examples["input_ids"], [])
        total_length = len(concatenated_ids)
        total_length = total_length // max_length * max_length
        all_input_ids = [
            concatenated_ids[i : i + max_length] for i in range(0, total_length, max_length)
        ]
        all_attention_mask = [[1] * max_length for _ in range(len(all_input_ids))]
        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": [seq[:] for seq in all_input_ids],
        }

    (tokenized_cache_files, tokenized_lock) = _build_hf_map_cache_paths(
        config,
        dataset_tag="wikitext103_raw_v1",
        stage="tokenize",
        splits=list(raw_datasets.keys()),
        extra={
            "tokenizer_name": tokenizer_name,
            "remove_columns": raw_datasets["train"].column_names,
        },
    )
    with _MapStageLock(tokenized_lock):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Tokenizing" if not config.quiet else None,
            cache_file_names=tokenized_cache_files,
            load_from_cache_file=True,
        )
    (grouped_cache_files, grouped_lock) = _build_hf_map_cache_paths(
        config,
        dataset_tag="wikitext103_raw_v1",
        stage="group_texts",
        splits=list(tokenized_datasets.keys()),
        extra={"tokenizer_name": tokenizer_name, "max_length": max_length},
    )
    with _MapStageLock(grouped_lock):
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            desc="Grouping texts" if not config.quiet else None,
            cache_file_names=grouped_cache_files,
            load_from_cache_file=True,
        )
    lm_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    if not config.quiet:
        print(
            f"WikiText-103: {len(lm_datasets['train'])} train chunks, {len(lm_datasets['validation'])} val chunks (max_length={max_length})"
        )
    g = torch.Generator()
    g.manual_seed(config.seed)
    tp_num_workers = 0 if config.use_tensor_parallel else config.num_workers
    train_loader = DataLoader(
        lm_datasets["train"],
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=tp_num_workers,
        pin_memory=True,
        generator=g,
        drop_last=config.use_tensor_parallel,
    )
    val_loader = DataLoader(
        lm_datasets["validation"],
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=tp_num_workers,
        pin_memory=True,
    )
    return (train_loader, val_loader, tokenizer)


def create_gpt2_lm(model_name: str = "gpt2") -> nn.Module:
    """Create a causal-LM decoder (GPT-2, Pythia, Llama, ...).

    Supported families (tested):
    - gpt2, gpt2-medium, gpt2-large, gpt2-xl
    - EleutherAI/pythia-*
    - meta-llama/Llama-3.2-* (1B/3B)
    - meta-llama/Llama-2-* (7B via LoRA/ckpt)

    Args:
        model_name: HuggingFace model name/path

    Returns:
        AutoModelForCausalLM (any causal LM architecture)
    """
    try:
        from transformers import AutoModelForCausalLM
        from transformers.utils import is_torch_available as _hf_torch_available
    except ImportError as e:
        raise ImportError("Decoder LM requires: pip install transformers") from e
    if not _hf_torch_available():
        raise ImportError(
            "Decoder LM requires a transformers installation with an enabled PyTorch backend. Current environment disables the transformers torch backend."
        )
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    except ImportError as e:
        raise ImportError(
            "Decoder LM requires a transformers installation with an enabled PyTorch backend."
        ) from e
    param_count = sum((p.numel() for p in model.parameters()))
    return model


def get_compression_targets_gpt2(model: nn.Module, target_mode: str = "all") -> List[str]:
    """Get decoder-LM compression targets (GPT-2, Pythia, Llama, ...).

    Target patterns (activation-heavy MLP layers only -- skip attention Q/K/V/O):
    - GPT-2: transformer.h.{i}.mlp.c_fc | mlp.c_proj (Conv1D, up/down projection)
    - Pythia: gpt_neox.layers.{i}.mlp.dense_h_to_4h | mlp.dense_4h_to_h
    - Llama/Mistral: model.layers.{i}.mlp.gate_proj | mlp.up_proj | mlp.down_proj

    target_mode:
    - "all": current behavior, compress all decoder MLP projections
    - "largest_mlp_only": compress only the projection whose input lives in
      the expanded 4H space (GPT-2 c_proj, Pythia dense_4h_to_h,
      Llama/Mistral/Qwen down_proj)
    """
    if target_mode == "all":
        mlp_patterns = (
            "mlp.c_fc",
            "mlp.c_proj",
            "mlp.dense_h_to_4h",
            "mlp.dense_4h_to_h",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        )
    elif target_mode == "largest_mlp_only":
        mlp_patterns = ("mlp.c_proj", "mlp.dense_4h_to_h", "mlp.down_proj")
    else:
        raise ValueError(
            f"Unsupported decoder target mode: {target_mode}. Use 'all' or 'largest_mlp_only'."
        )
    targets = []
    for name, module in model.named_modules():
        if any((p in name for p in mlp_patterns)):
            targets.append(name)
    return targets


def _exclude_checkpointed_decoder_targets_for_c1(
    model: nn.Module, config: "ConvergenceConfig", targets: List[str]
) -> List[str]:
    if not getattr(config, "stacr_compress_checkpoint", False):
        return targets
    if getattr(config, "stacr_ckpt_allow_inner_stacr", False):
        return targets
    try:
        candidates = collect_c1_site_candidates(model, config)
    except Exception:
        return targets
    if not candidates:
        return targets
    blocked_prefixes = tuple((f"{candidate.spec.site_id}." for candidate in candidates))
    return [name for name in targets if not name.startswith(blocked_prefixes)]


def auto_reshape_shape(in_features: int) -> Optional[Tuple[int, ...]]:
    """Automatically determine reshape_shape for tensorization.

    Args:
        in_features: Number of input features to reshape

    Returns:
        Tuple of dimensions or None if no good factorization found
    """
    known_shapes = {
        512: (8, 8, 8),
        768: (8, 8, 12),
        1024: (8, 8, 16),
        2048: (8, 16, 16),
        3072: (12, 16, 16),
        4096: (16, 16, 16),
    }
    if in_features in known_shapes:
        return known_shapes[in_features]
    cube_root = int(round(in_features ** (1 / 3)))
    for d1 in range(cube_root, 0, -1):
        if in_features % d1 != 0:
            continue
        remainder = in_features // d1
        sqrt_rem = int(round(remainder**0.5))
        for d2 in range(sqrt_rem, 0, -1):
            if remainder % d2 != 0:
                continue
            d3 = remainder // d2
            dims = sorted([d1, d2, d3])
            if dims[2] <= 4 * dims[0]:
                return tuple(sorted([d1, d2, d3], reverse=True))
    return None


def replace_linear_with_compressed(
    model: nn.Module, config: ConvergenceConfig, target_layers: Optional[List[str]] = None
) -> nn.Module:
    """Replace Linear layers with CompressedLinear.

    Args:
        model: Model to modify
        config: Compression config
        target_layers: List of layer names to replace (None = auto-detect)

    Returns:
        Modified model
    """
    from stacr.protocols.layers import CompressedLinear

    if config.method == "baseline":
        return model
    replaced = 0
    module_dict = dict(model.named_modules())
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            has_bias = module.bias is not None
            is_conv1d = False
        elif _is_hf_conv1d(module):
            (in_features, out_features) = module.weight.shape
            has_bias = hasattr(module, "bias") and module.bias is not None
            is_conv1d = True
        elif _is_pyg_linear(module):
            in_features = getattr(module, "in_channels", None) or getattr(
                module, "in_features", None
            )
            out_features = getattr(module, "out_channels", None) or getattr(
                module, "out_features", None
            )
            has_bias = hasattr(module, "bias") and module.bias is not None
            is_conv1d = False
        else:
            continue
        if target_layers is not None and name not in target_layers:
            continue
        parts = name.rsplit(".", 1)
        if len(parts) == 1:
            parent = model
            attr = parts[0]
        else:
            parent = module_dict[parts[0]]
            attr = parts[1]
        old_linear = getattr(parent, attr)
        reshape_shape = config.reshape_shape
        if reshape_shape is None and config.method in ("tt", "tucker"):
            reshape_shape = auto_reshape_shape(in_features)
            if reshape_shape is None:
                if not config.quiet:
                    print(
                        f"  Warning: No good reshape_shape for {name} ({in_features} features), skipping"
                    )
                continue
        if config.method == "tucker":
            tucker_ranks = [config.rank] * (len(reshape_shape) + 1)
            compression_config = {
                "method": config.method,
                "ranks": tucker_ranks,
                "reshape_shape": reshape_shape,
                "factor_quantization_bits": config.factor_quantization_bits,
            }
        else:
            compression_config = {
                "method": config.method,
                "rank": config.rank,
                "reshape_shape": reshape_shape,
                "factor_quantization_bits": config.factor_quantization_bits,
                "factor_native_backward": config.factor_native_backward,
                "use_lowrank_svd": config.use_lowrank_svd,
            }
        new_linear = CompressedLinear(in_features, out_features, compression_config, bias=has_bias)
        target_device = new_linear.linear.weight.device
        target_dtype = new_linear.linear.weight.dtype
        if is_conv1d:
            new_linear.linear.weight.data = (
                old_linear.weight.data.t().clone().to(device=target_device, dtype=target_dtype)
            )
        else:
            new_linear.linear.weight.data = old_linear.weight.data.clone().to(
                device=target_device, dtype=target_dtype
            )
        if has_bias and new_linear.linear.bias is not None:
            new_linear.linear.bias.data = old_linear.bias.data.clone().to(
                device=new_linear.linear.bias.device, dtype=new_linear.linear.bias.dtype
            )
        setattr(parent, attr, new_linear)
        replaced += 1
        if not config.quiet:
            kind = "Conv1D" if is_conv1d else "Linear"
            print(f"  Replaced {name} ({kind}): {in_features}x{out_features}")
    if not config.quiet:
        print(
            f"Replaced {replaced} layers with CompressedLinear (method={config.method}, rank={config.rank})"
        )
    return model


def _is_hf_conv1d(module: nn.Module) -> bool:
    """Detect HuggingFace Conv1D (GPT-2 style) without importing transformers.

    Conv1D is used in GPT-2 for MLP layers. It's essentially a Linear layer
    with transposed weights: weight shape is (in_features, out_features) instead
    of (out_features, in_features).
    """
    return module.__class__.__name__ == "Conv1D"


def _is_pyg_linear(module: nn.Module) -> bool:
    """Detect PyTorch Geometric Linear without importing torch_geometric.

    PyG uses its own Linear class (torch_geometric.nn.dense.linear.Linear)
    inside GCNConv and other message passing layers. It has the same weight
    layout as nn.Linear: (out_features, in_features).
    """
    cls_name = module.__class__.__name__
    mod_name = module.__class__.__module__
    if cls_name != "Linear":
        return False
    if "torch_geometric" in mod_name:
        return True
    return hasattr(module, "in_channels") and hasattr(module, "out_channels")


def apply_min_adaptation_rule(
    stacr_cfg: STACRConfig, max_steps: Optional[int], log_fn=print
) -> STACRConfig:
    """Guard random-QR init against freezing before any meaningful adaptation."""
    if max_steps is None:
        return stacr_cfg
    adaptation_steps = max_steps
    if stacr_cfg.freeze_basis_after_warmup:
        adaptation_steps = min(adaptation_steps, max(int(stacr_cfg.warmup_steps), 0))
    triggers = (
        stacr_cfg.use_random_qr_init
        and stacr_cfg.freeze_basis_after_warmup
        and (adaptation_steps < stacr_cfg.min_adaptation_steps)
        and (not stacr_cfg.allow_unadapted_random_qr)
    )
    if triggers:
        msg = f"[A3 min-adaptation] random-QR provides only {adaptation_steps} adaptation step(s) (total_steps={max_steps}, warmup_steps={stacr_cfg.warmup_steps}, freeze_after_warmup={stacr_cfg.freeze_basis_after_warmup}) < min_adaptation_steps={stacr_cfg.min_adaptation_steps}. Falling back to SVD init to prevent silent quality regression. Override with --stacr-allow-unadapted-random-qr for diagnostic runs."
        warnings.warn(msg)
        if log_fn is not None:
            log_fn(msg)
        stacr_cfg.use_random_qr_init = False
    return stacr_cfg


def _resolve_effective_total_steps(
    config: ConvergenceConfig, steps_per_epoch: Optional[int] = None
) -> Optional[int]:
    """Resolve total optimizer steps for safety checks.

    Prefer explicit max_steps. Otherwise, derive the total step budget from the
    dataloader length and epoch count when available.
    """
    if getattr(config, "max_steps", None) is not None:
        return int(config.max_steps)
    if steps_per_epoch is None:
        return None
    epochs = getattr(config, "epochs", None)
    if epochs is None:
        return None
    return int(steps_per_epoch) * int(epochs)


def _resolve_runtime_stacr_basis_dtype(
    config: ConvergenceConfig, raw_value: Optional[str] = None
) -> str:
    """Resolve the runtime STAC-R basis dtype from config-level policy.

    `STACRConfig.basis_dtype="auto"` is intended to mirror the configured AMP dtype
    (BF16/FP16) and fall back to FP32 when AMP is off.
    Resolving that inside STACRLinear.__init__ is too early because autocast is not active
    during module construction. Resolve it from the configured AMP policy here instead.
    """
    value = raw_value if raw_value is not None else getattr(config, "stacr_basis_dtype", "auto")
    if not isinstance(value, str):
        return "float32"
    if value.lower() != "auto":
        return value
    baseline_cfg = getattr(config, "baseline_cfg", None)
    amp_mode = getattr(baseline_cfg, "amp", "none")
    device_value = getattr(config, "device", "cuda")
    try:
        device_type = torch.device(device_value).type
    except (TypeError, RuntimeError, ValueError):
        device_type = "cuda"
    (resolved_dtype, _) = resolve_amp_dtype(amp_mode, device=device_type)
    if resolved_dtype == torch.bfloat16:
        return "bfloat16"
    if resolved_dtype == torch.float16:
        return "float16"
    return "float32"


def _resolve_runtime_prac_proj_dtype(
    config: ConvergenceConfig, raw_value: Optional[str] = None
) -> str:
    """Resolve the runtime PRAC projection dtype from config-level policy.

    ``PRACConfig.proj_dtype="auto"`` mirrors the configured AMP dtype:
    BF16/FP16 under AMP, FP32 otherwise. The configured policy determines
    the projection dtype because autocast is inactive during module construction.
    """
    value = raw_value if raw_value is not None else getattr(config, "prac_proj_dtype", "auto")
    if not isinstance(value, str):
        return "float32"
    if value.lower() != "auto":
        return value
    baseline_cfg = getattr(config, "baseline_cfg", None)
    amp_mode = getattr(baseline_cfg, "amp", "none")
    device_value = getattr(config, "device", "cuda")
    try:
        device_type = torch.device(device_value).type
    except (TypeError, RuntimeError, ValueError):
        device_type = "cuda"
    (resolved_dtype, _) = resolve_amp_dtype(amp_mode, device=device_type)
    if resolved_dtype == torch.bfloat16:
        return "bfloat16"
    if resolved_dtype == torch.float16:
        return "float16"
    return "float32"


def _resolve_runtime_prac_activation_dtype(config: ConvergenceConfig) -> torch.dtype:
    """Resolve the dtype of saved ``x_proj`` activations under AMP policy.

    This follows the runtime autocast dtype, not the projection-buffer dtype.
    The distinction matters when ``prac_proj_dtype`` is explicitly pinned to a
    different dtype than the activation dtype, because forward then materializes
    a casted projection tensor ``P_act``.
    """
    baseline_cfg = getattr(config, "baseline_cfg", None)
    amp_mode = getattr(baseline_cfg, "amp", "none")
    device_value = getattr(config, "device", "cuda")
    try:
        device_type = torch.device(device_value).type
    except (TypeError, RuntimeError, ValueError):
        device_type = "cuda"
    (resolved_dtype, _) = resolve_amp_dtype(amp_mode, device=device_type)
    return resolved_dtype or torch.float32


def _maybe_plan_decoder_targets(
    model: nn.Module, config: ConvergenceConfig, targets: List[str]
) -> tuple[List[str], Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
    """Optionally run the Phase 3 decoder planner.

    Dry-run emits decisions but leaves wrapping unchanged. Enforced mode filters
    the candidate list before STAC-R replacement.
    """
    if not getattr(config, "stacr_use_planner", False):
        return (targets, None, None)
    if getattr(config, "use_tensor_parallel", False):
        raise NotImplementedError(
            "Phase 3 planner v0 is not supported with tensor parallel. Disable --tp or leave --stacr-use-planner off."
        )
    if config.method not in ("stacr", "stacr-v4"):
        raise ValueError(
            "--stacr-use-planner is only supported with --method stacr or --method stacr-v4."
        )
    if (
        getattr(config, "stacr_planner_enforce", False)
        and getattr(config, "stacr_adaptive_layer_rank", "none") == "heuristic"
    ):
        raise ValueError(
            "Phase 3 planner v0 does not support --stacr-adaptive-layer-rank heuristic together with --stacr-planner-enforce."
        )
    from stacr.protocols.planner import plan_stacr_layers, summarize_layer_decisions

    decisions = plan_stacr_layers(model, config, targets)
    summary = summarize_layer_decisions(
        decisions,
        target_mode=getattr(config, "stacr_decoder_target_mode", "all"),
        planner_enforced=bool(getattr(config, "stacr_planner_enforce", False)),
    )
    filtered_targets = targets
    if getattr(config, "stacr_planner_enforce", False):
        filtered_targets = [decision.name for decision in decisions if decision.keep]
    serialized = [decision.to_dict() for decision in decisions]
    return (filtered_targets, serialized, summary)


def _parse_stacr_per_layer_rank_schedule(raw: Optional[str]) -> List[Tuple[str, int]]:
    """Parse comma-separated per-layer rank overrides.

    Keys match exact module names, numeric replacement indices, shell-style
    wildcards, or suffixes such as ``c_proj``. Later matching entries override
    earlier ones so a broad rule can be followed by an exact correction.
    """
    if raw is None or raw.strip() == "":
        return []
    entries: List[Tuple[str, int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            (key, value) = item.split("=", 1)
        elif ":" in item:
            (key, value) = item.split(":", 1)
        else:
            raise ValueError(
                f"Invalid --stacr-per-layer-rank-schedule entry {item!r}; expected key=rank."
            )
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError("--stacr-per-layer-rank-schedule has an empty key.")
        try:
            rank = int(value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid --stacr-per-layer-rank-schedule rank {value!r}; expected a positive integer."
            ) from exc
        if rank < 1:
            raise ValueError("--stacr-per-layer-rank-schedule ranks must be >= 1.")
        entries.append((key, rank))
    return entries


def _rank_schedule_matches(key: str, module_name: str, layer_index: int) -> bool:
    if key.isdigit():
        return int(key) == layer_index
    if key == module_name:
        return True
    if "*" in key or "?" in key or "[" in key:
        return fnmatch.fnmatch(module_name, key)
    return module_name.endswith(key) or module_name.endswith(f".{key}")


def _resolve_stacr_rank_schedule_entry(
    schedule: List[Tuple[str, int]], module_name: str, layer_index: int
) -> Optional[int]:
    rank: Optional[int] = None
    for key, candidate_rank in schedule:
        if _rank_schedule_matches(key, module_name, layer_index):
            rank = candidate_rank
    return rank


def replace_linear_with_stacr(
    model: nn.Module, config: ConvergenceConfig, target_layers: Optional[List[str]] = None
) -> nn.Module:
    """Replace Linear layers (or Conv1D in GPT-2) with STACRLinear.

    Args:
        model: Model to modify
        config: Compression config (uses rank and stacr_* fields)
        target_layers: List of layer names to replace (None = auto-detect)

    Returns:
        Modified model

    Note:
        Handles both nn.Linear and HuggingFace Conv1D (used in GPT-2).
        Conv1D has transposed weights: (in_features, out_features) vs
        Linear's (out_features, in_features).
    """
    if config.method not in ("stacr", "stacr-v4"):
        return model
    replace_linear_with_stacr._skipped_min_features = 0
    rank_schedule = _parse_stacr_per_layer_rank_schedule(
        getattr(config, "stacr_per_layer_rank_schedule", None)
    )
    stacr_config = STACRConfig(
        rank=config.rank,
        residual_mode=getattr(config, "stacr_residual_mode", "block_sparse"),
        residual_budget=getattr(config, "stacr_residual_budget", 0.005),
        residual_threshold=getattr(config, "stacr_residual_threshold", 0.1),
        update_trigger=getattr(config, "stacr_update_trigger", "hybrid"),
        update_interval=getattr(config, "stacr_update_interval", 100),
        energy_threshold=getattr(config, "stacr_energy_threshold", 0.05),
        streaming_algorithm=getattr(config, "stacr_streaming_algorithm", "oja"),
        oja_learning_rate=getattr(config, "stacr_oja_learning_rate", 0.01),
        warmup_steps=getattr(config, "stacr_warmup_steps", 0),
        freeze_basis_after_warmup=getattr(config, "stacr_freeze_basis_after_warmup", True),
        quantize_coefficients=getattr(config, "stacr_quantize_coefficients", True),
        quantize_bits=getattr(config, "stacr_quantize_bits", 8),
        quantize_residuals=getattr(config, "stacr_quantize_residuals", True),
        residual_quantize_bits=getattr(config, "stacr_residual_quantize_bits", 4),
        randomized_svd_threshold=getattr(config, "stacr_randomized_svd_threshold", 1024),
        randomized_svd_oversampling=getattr(config, "stacr_randomized_svd_oversampling", 4),
        randomized_svd_niter=getattr(config, "stacr_randomized_svd_niter", 2),
        adaptive_oja_lr=getattr(config, "stacr_adaptive_oja_lr", False),
        adaptive_oja_scale=getattr(config, "stacr_adaptive_oja_scale", 5.0),
        adaptive_oja_max_lr=getattr(config, "stacr_adaptive_oja_max_lr", 0.1),
        basis_dtype=_resolve_runtime_stacr_basis_dtype(config),
        backward_mode=getattr(config, "stacr_backward_mode", "sparse"),
        residual_block_size=getattr(config, "stacr_residual_block_size", 16),
        use_mgs=getattr(config, "stacr_use_mgs", False),
        lazy_update_enabled=getattr(config, "stacr_lazy_update", True),
        lazy_update_threshold=getattr(config, "stacr_lazy_update_threshold", 0.01),
        use_fused_kernel=getattr(config, "stacr_use_fused_kernel", False),
        cpu_offload=getattr(config, "stacr_cpu_offload", False),
        rank_mode=getattr(config, "stacr_rank_mode", "fixed"),
        rank_variance_threshold=getattr(config, "stacr_rank_variance_threshold", 0.95),
        rank_min=getattr(config, "stacr_rank_min", 4),
        rank_max=getattr(config, "stacr_rank_max", 64),
        rank_adjust_interval=getattr(config, "stacr_rank_adjust_interval", 100),
        rank_sensitivity_low=getattr(config, "stacr_rank_sensitivity_low", 0.03),
        rank_sensitivity_high=getattr(config, "stacr_rank_sensitivity_high", 0.1),
        rank_step_fraction=getattr(config, "stacr_rank_step_fraction", 0.125),
        basis_sharing=getattr(config, "stacr_basis_sharing", "none"),
        streaming_topk=getattr(config, "stacr_streaming_topk", False),
        streaming_topk_sample_ratio=getattr(config, "stacr_streaming_topk_sample_ratio", 0.1),
        index_compression=getattr(config, "stacr_index_compression", "none"),
        skip_residual_enabled=getattr(config, "stacr_skip_residual", False),
        skip_residual_energy_threshold=getattr(config, "stacr_skip_residual_threshold", 0.05),
        buffer_reuse=getattr(config, "stacr_buffer_reuse", False),
        adaptive_layer_rank=getattr(config, "stacr_adaptive_layer_rank", "none"),
        residual_schedule=getattr(config, "stacr_residual_schedule", "none"),
        residual_start=getattr(config, "stacr_residual_start", 0.05),
        residual_end=getattr(config, "stacr_residual_end", 0.001),
        chunk_size=getattr(config, "stacr_chunk_size", 512),
        post_warmup_residual_mode=getattr(config, "stacr_post_warmup_mode", "full"),
        residual_warmup_steps=getattr(config, "stacr_residual_warmup_steps", 0),
        post_warmup_residual_threshold=getattr(config, "stacr_post_warmup_threshold", 0.1),
        refresh_trigger=getattr(config, "stacr_refresh_trigger", "off"),
        refresh_period=getattr(config, "stacr_refresh_period", 1000),
        refresh_energy_threshold=getattr(config, "stacr_refresh_energy_threshold", 0.1),
        cache_basis_per_optimizer_step=getattr(
            config, "stacr_cache_basis_per_optimizer_step", False
        ),
        oja_update_per_optimizer_step=getattr(config, "stacr_oja_update_per_optimizer_step", False),
        enable_low_rank_only_fast_path=getattr(
            config, "stacr_enable_low_rank_only_fast_path", True
        ),
        use_random_qr_init=getattr(config, "stacr_use_random_qr_init", False),
        min_adaptation_steps=getattr(config, "stacr_min_adaptation_steps", 50),
        allow_unadapted_random_qr=getattr(config, "stacr_allow_unadapted_random_qr", False),
    )
    stacr_config = apply_min_adaptation_rule(
        stacr_config,
        getattr(config, "effective_total_steps", None),
        log_fn=None if getattr(config, "quiet", False) else print,
    )
    replaced = 0
    layer_index = 0

    def _is_replaceable(module: nn.Module) -> bool:
        return isinstance(module, nn.Linear) or _is_hf_conv1d(module) or _is_pyg_linear(module)

    total_layers = sum(
        (
            1
            for (name, module) in model.named_modules()
            if _is_replaceable(module) and (target_layers is None or name in target_layers)
        )
    )
    if target_layers is not None and len(target_layers) > 0 and (total_layers == 0):
        raise RuntimeError(
            "STACR replacement found 0 matching layers. Check target layer names (e.g., GPT-2 uses transformer.h.*.mlp.c_fc/c_proj; GCNConv uses convs.*.lin)."
        )
    module_dict = dict(model.named_modules())
    for name, module in list(model.named_modules()):
        if not _is_replaceable(module):
            continue
        if target_layers is not None and name not in target_layers:
            continue
        parts = name.rsplit(".", 1)
        if len(parts) == 1:
            parent = model
            attr = parts[0]
        else:
            parent = module_dict[parts[0]]
            attr = parts[1]
        old_module = getattr(parent, attr)
        if isinstance(old_module, nn.Linear):
            in_features = old_module.in_features
            out_features = old_module.out_features
            has_bias = old_module.bias is not None
            is_conv1d = False
            is_pyg_linear = False
        elif _is_hf_conv1d(old_module):
            (in_features, out_features) = old_module.weight.shape
            has_bias = hasattr(old_module, "bias") and old_module.bias is not None
            is_conv1d = True
            is_pyg_linear = False
        elif _is_pyg_linear(old_module):
            in_features = getattr(old_module, "in_channels", None)
            out_features = getattr(old_module, "out_channels", None)
            if in_features is None or out_features is None:
                in_features = getattr(old_module, "in_features")
                out_features = getattr(old_module, "out_features")
            has_bias = hasattr(old_module, "bias") and old_module.bias is not None
            is_conv1d = False
            is_pyg_linear = True
        else:
            continue
        min_features = getattr(config, "stacr_min_features", 0)
        if min_features > 0 and in_features < min_features:
            if not config.quiet:
                print(f"  Skipped {name}: {in_features} features < min_features ({min_features})")
            if not hasattr(replace_linear_with_stacr, "_skipped_min_features"):
                replace_linear_with_stacr._skipped_min_features = 0
            replace_linear_with_stacr._skipped_min_features += 1
            continue
        layer_stacr_config = stacr_config
        small_layer_rank_scale = getattr(config, "stacr_small_layer_rank_scale", 1.0)
        small_layer_residual_scale = getattr(config, "stacr_small_layer_residual_scale", 1.0)
        if in_features < 1024 and (
            small_layer_rank_scale != 1.0 or small_layer_residual_scale != 1.0
        ):
            from copy import copy

            layer_stacr_config = copy(stacr_config)
            if small_layer_rank_scale != 1.0:
                adjusted_rank = max(4, int(stacr_config.rank * small_layer_rank_scale))
                layer_stacr_config.rank = adjusted_rank
                if not config.quiet:
                    print(
                        f"    Adjusted rank: {stacr_config.rank} -> {adjusted_rank} for small layer {name}"
                    )
            if small_layer_residual_scale != 1.0:
                adjusted_budget = stacr_config.residual_budget * small_layer_residual_scale
                layer_stacr_config.residual_budget = adjusted_budget
                if not config.quiet:
                    print(
                        f"    Adjusted residual: {stacr_config.residual_budget} -> {adjusted_budget} for small layer {name}"
                    )
        scheduled_rank = _resolve_stacr_rank_schedule_entry(
            rank_schedule, module_name=name, layer_index=layer_index
        )
        if scheduled_rank is not None:
            from copy import copy

            if layer_stacr_config is stacr_config:
                layer_stacr_config = copy(stacr_config)
            old_rank = layer_stacr_config.rank
            layer_stacr_config.rank = scheduled_rank
            layer_stacr_config.adaptive_layer_rank = "none"
            if not config.quiet:
                print(f"    Scheduled rank: {old_rank} -> {scheduled_rank} for layer {name}")
        new_linear = STACRLinear(
            in_features,
            out_features,
            layer_stacr_config,
            bias=has_bias,
            layer_index=layer_index,
            num_layers=total_layers,
        )
        target_device = new_linear.linear.weight.device
        target_dtype = new_linear.linear.weight.dtype
        if is_conv1d:
            new_linear.linear.weight.data = (
                old_module.weight.data.t().clone().to(device=target_device, dtype=target_dtype)
            )
        else:
            new_linear.linear.weight.data = old_module.weight.data.clone().to(
                device=target_device, dtype=target_dtype
            )
        if has_bias and new_linear.linear.bias is not None:
            new_linear.linear.bias.data = old_module.bias.data.clone().to(
                device=new_linear.linear.bias.device, dtype=new_linear.linear.bias.dtype
            )
        setattr(parent, attr, new_linear)
        replaced += 1
        layer_index += 1
        if not config.quiet:
            layer_type = "Conv1D" if is_conv1d else "PyG Linear" if is_pyg_linear else "Linear"
            print(
                f"  Replaced {name} ({layer_type}): {in_features}x{out_features} with STACRLinear"
            )
    if not config.quiet:
        print(
            f"Replaced {replaced} layers with STACRLinear (rank={config.rank}, residual={stacr_config.residual_budget})"
        )
    skipped_due_to_min_features = getattr(replace_linear_with_stacr, "_skipped_min_features", 0)
    expected_replaced = total_layers - skipped_due_to_min_features
    if target_layers and replaced != expected_replaced and (expected_replaced > 0):
        raise RuntimeError(
            f"STACR replacement incomplete: replaced {replaced} of {expected_replaced} matching layers (skipped {skipped_due_to_min_features} due to min_features). Check layer name patterns."
        )
    elif replaced == 0 and skipped_due_to_min_features > 0:
        if not config.quiet:
            print(
                f"  WARNING: All {skipped_due_to_min_features} target layers skipped due to min_features={min_features}"
            )
            print(f"  Model will run as BASELINE (no STACR compression)")
    return model


def _estimate_layer_storage_bytes(
    batch_tokens: int,
    in_features: int,
    method: str,
    rank: Optional[int],
    ranks: Optional[List[int]],
    reshape_shape: Optional[Tuple[int, ...]],
    factor_quantization_bits: Optional[int],
    activation_dtype: torch.dtype = torch.float32,
) -> Dict[str, Any]:
    """Estimate stored/original bytes for a single compressed layer."""
    import math as _math
    from stacr.memory_accounting import (
        bytes_per_elem,
        svd_stored_bytes,
        tt_stored_bytes,
        tt_uniform_ranks,
        tt_stored_elements,
        tucker_stored_bytes,
        tucker_stored_elements,
    )

    original_bytes = int(batch_tokens * in_features * bytes_per_elem(activation_dtype))
    if method == "svd":
        k = int(rank or 1)
        if factor_quantization_bits:
            per_elem = factor_quantization_bits / 8.0
            stored = (batch_tokens * k + k * in_features) * per_elem
            stored += k * 4
            stored += 2 * 4
            stored_bytes = int(_math.ceil(stored))
        else:
            stored_bytes = svd_stored_bytes(
                batch_tokens, in_features, k, torch.float32, include_meta=False
            ).total_bytes
        return {
            "method": method,
            "rank": k,
            "stored_bytes": stored_bytes,
            "original_bytes": original_bytes,
        }
    dims = (batch_tokens,) + tuple(reshape_shape or ())
    if method == "tt":
        if ranks is None:
            r = int(rank or 1)
            ranks = tt_uniform_ranks(dims, r)
        if factor_quantization_bits:
            per_elem = factor_quantization_bits / 8.0
            core_elems = tt_stored_elements(dims, ranks)
            stored = core_elems * per_elem
            stored += len(dims) * 4
            stored_bytes = int(_math.ceil(stored))
        else:
            stored_bytes = tt_stored_bytes(
                dims, ranks, torch.float32, include_meta=False
            ).total_bytes
        return {
            "method": method,
            "ranks": ranks,
            "stored_bytes": stored_bytes,
            "original_bytes": original_bytes,
        }
    if method == "tucker":
        if ranks is None:
            r = int(rank or 1)
            ranks = [r] * len(dims)
        if factor_quantization_bits:
            per_elem = factor_quantization_bits / 8.0
            core_elems = 1
            for r in ranks:
                core_elems *= r
            factor_elems = sum((d * r for (d, r) in zip(dims, ranks)))
            stored = (core_elems + factor_elems) * per_elem
            stored += (1 + len(dims)) * 4
            stored_bytes = int(_math.ceil(stored))
        else:
            stored_bytes = tucker_stored_bytes(
                dims, ranks, torch.float32, include_meta=False
            ).total_bytes
        return {
            "method": method,
            "ranks": ranks,
            "stored_bytes": stored_bytes,
            "original_bytes": original_bytes,
        }
    return {"method": method, "stored_bytes": None, "original_bytes": original_bytes}


def estimate_storage_for_model(
    model: nn.Module, config: ConvergenceConfig
) -> Optional[Dict[str, Any]]:
    """Estimate stored bytes and achieved ratio for all compressed layers."""
    from stacr.protocols.layers import CompressedLinear

    if config.method == "baseline":
        return None
    layers = [m for m in model.modules() if isinstance(m, CompressedLinear)]
    if not layers:
        return None
    batch_tokens = config.batch_size
    if config.task == "sst2":
        batch_tokens = config.batch_size * config.max_length
    activation_dtype = _resolve_runtime_prac_activation_dtype(config)
    total_stored = 0
    total_original = 0
    layer_details = []
    for layer in layers:
        layer_cfg = layer.config or {}
        detail = _estimate_layer_storage_bytes(
            batch_tokens=batch_tokens,
            in_features=layer.linear.in_features,
            method=layer_cfg.get("method", config.method),
            rank=layer_cfg.get("rank", config.rank),
            ranks=layer_cfg.get("ranks"),
            reshape_shape=tuple(layer_cfg.get("reshape_shape") or ()),
            factor_quantization_bits=layer_cfg.get(
                "factor_quantization_bits", config.factor_quantization_bits
            ),
            activation_dtype=activation_dtype,
        )
        if detail.get("stored_bytes") is not None:
            total_stored += detail["stored_bytes"]
        total_original += detail.get("original_bytes", 0)
        layer_details.append(detail)
    achieved_ratio = total_original / total_stored if total_stored > 0 else None
    return {
        "batch_tokens_est": batch_tokens,
        "layers": len(layers),
        "stored_bytes": total_stored or None,
        "original_bytes": total_original or None,
        "achieved_ratio": achieved_ratio,
        "layer_details": layer_details,
    }


def estimate_storage_for_stacr(
    model: nn.Module, config: ConvergenceConfig, actual_batch_tokens: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """Estimate stored bytes and achieved ratio for STAC-R compressed layers.

    Args:
        model: Model with STACRLinear layers
        config: Training configuration
        actual_batch_tokens: Override batch_tokens calculation. Use for GNN
            full-batch training where batch_tokens = num_nodes (not batch_size).
    """
    from stacr.memory_accounting import stacr_stored_bytes

    layers = [m for m in model.modules() if isinstance(m, STACRLinear)]
    if not layers:
        return None
    if actual_batch_tokens is not None:
        batch_tokens = actual_batch_tokens
    elif config.task == "sst2":
        batch_tokens = config.batch_size * config.max_length
    elif config.task == "wikitext103":
        batch_tokens = config.batch_size * config.max_length
    else:
        batch_tokens = config.batch_size
    total_stored = 0
    total_original = 0
    layer_details = []
    effective_total_steps = getattr(config, "effective_total_steps", None)
    if effective_total_steps is None:
        effective_total_steps = getattr(config, "max_steps", None)
    if effective_total_steps is not None:
        effective_total_steps = max(0, int(effective_total_steps))
    for layer in layers:
        in_features = layer.in_features
        allocated_rank = (
            layer.get_effective_rank()
            if hasattr(layer, "get_effective_rank")
            else layer.config.rank
        )
        active_rank = (
            layer.get_active_rank() if hasattr(layer, "get_active_rank") else allocated_rank
        )
        rank = active_rank
        warmup_steps = max(0, int(getattr(layer.config, "warmup_steps", 0)))
        residual_warmup_steps = max(0, int(getattr(layer.config, "residual_warmup_steps", 0)))
        configured_residual_budget = layer.config.residual_budget
        residual_mode = getattr(layer.config, "residual_mode", "topk")
        post_warmup_mode = getattr(layer.config, "post_warmup_residual_mode", "full")
        warmup_only_run = (
            effective_total_steps is not None and effective_total_steps <= warmup_steps
        )
        low_rank_only_enabled = bool(getattr(layer.config, "enable_low_rank_only_fast_path", True))
        low_rank_only_fast_path = False
        if not warmup_only_run:
            if low_rank_only_enabled and residual_mode == "none":
                low_rank_only_fast_path = True
            elif (
                low_rank_only_enabled and post_warmup_mode == "none" and (residual_warmup_steps > 0)
            ):
                if effective_total_steps is None:
                    low_rank_only_fast_path = True
                else:
                    activation_step = max(warmup_steps, residual_warmup_steps)
                    low_rank_only_fast_path = effective_total_steps > activation_step
        residual_budget: Optional[float]
        if warmup_only_run:
            residual_budget = None
        elif low_rank_only_fast_path:
            residual_budget = 0.0
        else:
            residual_budget = configured_residual_budget
        coeff_bits = None
        if (
            not warmup_only_run
            and (not low_rank_only_fast_path)
            and getattr(layer.config, "quantize_coefficients", False)
        ):
            coeff_bits = int(getattr(layer.config, "quantize_bits", 8))
        residual_bits = None
        if (
            not warmup_only_run
            and (not low_rank_only_fast_path)
            and getattr(layer.config, "quantize_residuals", False)
        ):
            residual_bits = int(getattr(layer.config, "residual_quantize_bits", 8))
        basis_dtype = _resolve_basis_dtype(getattr(layer.config, "basis_dtype", "float32"))
        activation_dtype = _resolve_runtime_prac_activation_dtype(config)
        original_bytes = (
            batch_tokens * in_features * torch.tensor([], dtype=activation_dtype).element_size()
        )
        storage_mode = (
            "baseline_warmup_only"
            if warmup_only_run
            else "low_rank_only_fast_path"
            if low_rank_only_fast_path
            else "stacr"
        )
        if warmup_only_run:
            stored_bytes = original_bytes
        else:
            breakdown = stacr_stored_bytes(
                batch_tokens,
                in_features,
                rank,
                residual_budget,
                activation_dtype,
                coeff_bits=coeff_bits,
                residual_bits=residual_bits,
                basis_dtype=basis_dtype,
                include_meta=False,
            )
            stored_bytes = breakdown.total_bytes
        total_stored += stored_bytes
        total_original += original_bytes
        layer_details.append(
            {
                "method": "stacr",
                "storage_mode": storage_mode,
                "rank": rank,
                "active_rank": active_rank,
                "allocated_rank": allocated_rank,
                "warmup_steps": warmup_steps,
                "residual_warmup_steps": residual_warmup_steps,
                "residual_mode": residual_mode,
                "configured_residual_budget": configured_residual_budget,
                "post_warmup_residual_mode": post_warmup_mode,
                "residual_budget": residual_budget,
                "low_rank_only_fast_path": low_rank_only_fast_path,
                "enable_low_rank_only_fast_path": low_rank_only_enabled,
                "coeff_bits": coeff_bits,
                "residual_bits": residual_bits,
                "basis_dtype": str(basis_dtype),
                "stored_bytes": stored_bytes,
                "original_bytes": original_bytes,
                "compression_ratio": original_bytes / stored_bytes if stored_bytes > 0 else None,
            }
        )
    achieved_ratio = total_original / total_stored if total_stored > 0 else None
    return {
        "batch_tokens_est": batch_tokens,
        "layers": len(layers),
        "stored_bytes": total_stored or None,
        "original_bytes": total_original or None,
        "achieved_ratio": achieved_ratio,
        "layer_details": layer_details,
    }


def estimate_storage_for_c1(
    model: nn.Module, config: ConvergenceConfig, actual_batch_tokens: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """Estimate stored bytes for compressed checkpoint boundaries."""
    from stacr.memory_accounting import stacr_ckpt_stored_bytes
    from stacr.protocols.planner import (
        plan_stacr_checkpoint_sites,
        summarize_checkpoint_site_decisions,
    )

    try:
        candidates = collect_c1_site_candidates(model, config)
    except ValueError:
        return None
    if not candidates:
        return None
    if actual_batch_tokens is not None:
        batch_tokens = actual_batch_tokens
    elif config.task in ("sst2", "wikitext103"):
        batch_tokens = config.batch_size * config.max_length
    else:
        batch_tokens = config.batch_size
    activation_dtype = _resolve_runtime_prac_activation_dtype(config)
    effective_total_steps = getattr(config, "effective_total_steps", None)
    if effective_total_steps is None:
        effective_total_steps = getattr(config, "max_steps", None)
    if effective_total_steps is not None:
        effective_total_steps = max(0, int(effective_total_steps))
    planner_mode = getattr(config, "stacr_ckpt_planner", "heuristic")
    planner_serialized = None
    planner_summary = None
    decisions_by_name: Dict[str, Any] = {}
    if planner_mode != "off":
        decisions = plan_stacr_checkpoint_sites(model, config, candidates=candidates)
        planner_serialized = [decision.to_dict() for decision in decisions]
        planner_summary = summarize_checkpoint_site_decisions(
            decisions,
            target=getattr(config, "stacr_ckpt_target", "decoder_block_input"),
            planner_mode=planner_mode,
            planner_enforced=planner_mode == "enforce",
        )
        decisions_by_name = {decision.name: decision for decision in decisions}
    total_stored = 0
    total_original = 0
    total_basis = 0
    active_sites = 0
    layer_details = []
    for candidate in candidates:
        spec = candidate.spec
        decision = decisions_by_name.get(spec.site_id)
        planner_reason = decision.reason if decision is not None else None
        planner_keep = decision.keep if decision is not None else True
        original_bytes = (
            batch_tokens
            * spec.hidden_size
            * torch.tensor([], dtype=activation_dtype).element_size()
        )
        warmup_only_run = (
            effective_total_steps is not None and effective_total_steps <= spec.warmup_steps
        )
        if planner_mode == "enforce" and (not planner_keep):
            storage_mode = "planner_dense_fallback"
            payload_bytes = original_bytes
            basis_persistent_bytes = 0
            residual_budget = None
            coeff_bits = None
            residual_bits = None
            active = False
        elif warmup_only_run:
            storage_mode = "baseline_warmup_only"
            payload_bytes = original_bytes
            basis_persistent_bytes = 0
            residual_budget = None
            coeff_bits = None
            residual_bits = None
            active = True
        else:
            coeff_bits = spec.quantize_bits if spec.quantize_coefficients else None
            residual_bits = (
                spec.residual_quantize_bits
                if spec.quantize_residuals and spec.residual_mode != "none"
                else None
            )
            residual_budget = spec.residual_budget if spec.residual_mode != "none" else 0.0
            breakdown = stacr_ckpt_stored_bytes(
                batch_tokens,
                spec.hidden_size,
                spec.rank,
                residual_budget=residual_budget,
                dtype=activation_dtype,
                coeff_bits=coeff_bits,
                residual_bits=residual_bits,
                basis_dtype=spec.basis_dtype,
                include_meta=False,
                include_basis=False,
            )
            payload_bytes = int(breakdown.total_bytes)
            basis_persistent_bytes = int(breakdown.persistent_cache_bytes)
            storage_mode = "stacr_checkpoint"
            active = True
        stored_bytes = payload_bytes + basis_persistent_bytes
        total_stored += stored_bytes
        total_original += original_bytes
        total_basis += basis_persistent_bytes
        if active:
            active_sites += 1
        layer_details.append(
            {
                "method": "stacr_ckpt",
                "site_id": spec.site_id,
                "module_type": spec.module_type,
                "hidden_size": spec.hidden_size,
                "batch_tokens_est": batch_tokens,
                "storage_mode": storage_mode,
                "rank": spec.rank,
                "warmup_steps": spec.warmup_steps,
                "residual_mode": spec.residual_mode,
                "residual_budget": residual_budget,
                "coeff_bits": coeff_bits,
                "residual_bits": residual_bits,
                "basis_dtype": str(spec.basis_dtype),
                "planner_reason": planner_reason,
                "planner_keep": planner_keep,
                "payload_bytes": payload_bytes,
                "basis_persistent_bytes": basis_persistent_bytes,
                "stored_bytes": stored_bytes,
                "original_bytes": original_bytes,
                "compression_ratio": original_bytes / stored_bytes if stored_bytes > 0 else None,
            }
        )
    achieved_ratio = total_original / total_stored if total_stored > 0 else None
    return {
        "batch_tokens_est": batch_tokens,
        "sites": len(candidates),
        "active_sites": active_sites,
        "stored_bytes": total_stored or None,
        "stored_bytes_payload": total_stored - total_basis or None,
        "basis_persistent_bytes_total": total_basis,
        "original_bytes": total_original or None,
        "achieved_ratio": achieved_ratio,
        "planner_mode": planner_mode,
        "planner_summary": planner_summary,
        "planner_decisions": planner_serialized,
        "layer_details": layer_details,
    }


def estimate_storage_for_prac(
    model: nn.Module, config: ConvergenceConfig, actual_batch_tokens: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """Estimate PRAC stored bytes including persistent projection buffers.

    Splits layer_details into saved activation and projection-buffer bytes.
    Dense reference bytes use the configured runtime activation dtype.
    """
    from stacr.protocols.prac import PRACLinear, prac_stored_bytes_per_layer

    layers = [m for m in model.modules() if isinstance(m, PRACLinear)]
    if not layers:
        return None
    if actual_batch_tokens is not None:
        batch_tokens = actual_batch_tokens
    elif config.task in ("sst2", "wikitext103"):
        batch_tokens = config.batch_size * config.max_length
    else:
        batch_tokens = config.batch_size
    act_dtype = _resolve_runtime_prac_activation_dtype(config)
    act_elem = torch.tensor([], dtype=act_dtype).element_size()
    total_stored = 0
    total_original = 0
    layer_details = []
    for layer in layers:
        proj_dtype = layer.projection.dtype
        saved_projection_dtype = getattr(layer, "_last_saved_projection_dtype", None)
        rank = layer.effective_rank
        in_features = layer.in_features
        per_layer = prac_stored_bytes_per_layer(
            batch_tokens=batch_tokens,
            in_features=in_features,
            rank=rank,
            proj_dtype=proj_dtype,
            act_dtype=act_dtype,
            saved_projection_dtype=saved_projection_dtype,
        )
        saved_activation_bytes = per_layer["saved_activation_bytes"]
        projection_buffer_bytes = per_layer["projection_buffer_bytes"]
        saved_projection_bytes = per_layer["saved_projection_bytes"]
        stored_bytes = per_layer["stored_bytes"]
        original_bytes = int(batch_tokens) * int(in_features) * act_elem
        total_stored += stored_bytes
        total_original += original_bytes
        layer_details.append(
            {
                "method": "prac_linear_only",
                "rank": rank,
                "in_features": in_features,
                "weight_layout": "linear(out,in)",
                "proj_dtype": str(proj_dtype),
                "act_dtype": str(act_dtype),
                "saved_projection_dtype": str(saved_projection_dtype or proj_dtype),
                "saved_activation_bytes": saved_activation_bytes,
                "projection_buffer_bytes": projection_buffer_bytes,
                "saved_projection_bytes": saved_projection_bytes,
                "stored_bytes": stored_bytes,
                "original_bytes": original_bytes,
                "compression_ratio": original_bytes / stored_bytes if stored_bytes > 0 else None,
            }
        )
    achieved_ratio = total_original / total_stored if total_stored > 0 else None
    return {
        "batch_tokens_est": batch_tokens,
        "layers": len(layers),
        "stored_bytes": total_stored or None,
        "original_bytes": total_original or None,
        "achieved_ratio": achieved_ratio,
        "layer_details": layer_details,
    }


def get_compression_targets_bert(model: nn.Module) -> List[str]:
    """Get FFN output projection layers for BERT compression."""
    targets = []
    for name, module in model.named_modules():
        if "output.dense" in name and "attention" not in name:
            targets.append(name)
    return targets


def _normalize_method_config(config: ConvergenceConfig) -> None:
    """Normalize method labels (svd-v2, tact-v3, prac) into base method + config."""
    label = config.method_label or config.method
    if label in ("prac", "prac_linear_only"):
        config.method = "prac"
        config.method_label = "prac_linear_only"
        return
    if label.endswith("-v2"):
        config.method_label = label
        config.method = label.replace("-v2", "")
        if config.factor_quantization_bits is None:
            config.factor_quantization_bits = 8
        return
    if label == "tact-v3":
        config.method_label = label
        _apply_tact_v3_config(config)
        return
    config.method_label = label
    config.method = label


def _apply_tact_v3_config(config: ConvergenceConfig) -> None:
    """Load TACT-V3 best config and apply to convergence config."""
    path = Path(config.tact_v3_config_path or "artifacts/tact_v3_best.json")
    if not path.exists():
        raise FileNotFoundError(f"TACT-V3 config not found: {path}")
    with open(path, "r") as f:
        payload = json.load(f)
    tasks = payload.get("tasks", payload)
    task_cfgs = tasks.get(config.task)
    if not task_cfgs:
        raise ValueError(f"TACT-V3 config missing task: {config.task}")
    ratio_key = f"{int(config.tact_v3_target_ratio)}x"
    entry = task_cfgs.get(ratio_key)
    if entry is None:
        available = ", ".join(sorted(task_cfgs.keys()))
        raise ValueError(f"TACT-V3 config missing ratio {ratio_key} (available: {available})")
    config.method = entry.get("method", config.method)
    config.rank = entry.get("rank", config.rank)
    if entry.get("reshape_shape") is not None:
        config.reshape_shape = tuple(entry.get("reshape_shape"))
    config.factor_quantization_bits = entry.get(
        "factor_quantization_bits", config.factor_quantization_bits
    )
    config.factor_native_backward = entry.get(
        "factor_native_backward", config.factor_native_backward
    )
    baseline_cfg = entry.get("baseline_cfg")
    if isinstance(baseline_cfg, dict):
        config.baseline_cfg = BaselineConfig(**baseline_cfg)


@dataclass
class BaselineState:
    amp_manager: Optional[AMPManager] = None
    offloader: Optional[SavedTensorOffloader] = None
    gact_controller: Optional[Any] = None
    c1_runtime: Optional[Any] = None
    status: str = "enabled"
    reason: Optional[str] = None


def apply_baseline(
    model: nn.Module, config: ConvergenceConfig, device: torch.device
) -> Tuple[nn.Module, BaselineState]:
    """Apply baseline configuration to model.

    Returns:
        (model, BaselineState)
    """
    baseline_cfg = config.baseline_cfg
    state = BaselineState()

    def _cleanup_c1_runtime() -> None:
        if state.c1_runtime is not None:
            clear_active_c1_runtime(clear_registry=True)
            state.c1_runtime = None

    if getattr(config, "use_data_parallel", False):
        dp_incompatible = []
        if device.type != "cuda":
            dp_incompatible.append("non-CUDA device")
        if torch.cuda.device_count() < 2:
            dp_incompatible.append("fewer than 2 CUDA devices")
        if config.task not in ("sst2", "wikitext103"):
            dp_incompatible.append(f"task={config.task}")
        if baseline_cfg.checkpointing != "none":
            dp_incompatible.append("checkpointing")
        if baseline_cfg.actnn != "none":
            dp_incompatible.append("actnn")
        if baseline_cfg.gact != "none":
            dp_incompatible.append("gact")
        if baseline_cfg.torchao != "none":
            dp_incompatible.append("torchao")
        if baseline_cfg.saved_tensors_offload != "none":
            dp_incompatible.append("saved_tensors_offload")
        if getattr(config, "stacr_compress_checkpoint", False):
            dp_incompatible.append("stacr_compress_checkpoint")
        if baseline_cfg.grad_accum_steps > 1:
            dp_incompatible.append("grad_accum_steps > 1")
        if dp_incompatible:
            _cleanup_c1_runtime()
            state.status = "skipped_incompatible"
            state.reason = f"DataParallel is currently limited to simple SST-2/WikiText runs without checkpointing/ActNN/GACT/TorchAO/offload/grad accumulation; incompatible with: {', '.join(dp_incompatible)}"
            return (model, state)
    if baseline_cfg.actnn != "none":
        (model, actnn_result) = try_enable_actnn(model, level=baseline_cfg.actnn)
        if not actnn_result.ok:
            _cleanup_c1_runtime()
            state.status = actnn_result.status.value
            state.reason = actnn_result.reason
            return (model, state)
    model = model.to(device)
    if getattr(config, "use_data_parallel", False) and torch.cuda.device_count() > 1:
        print(f"DataParallel: wrapping model across {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    if baseline_cfg.checkpointing != "none":
        try:
            mode = baseline_cfg.checkpointing
            if mode == "auto":
                mode = (
                    "transformer_hf" if config.task in ("sst2", "wikitext103") else "vision_stage"
                )
            if getattr(config, "stacr_compress_checkpoint", False):
                state.c1_runtime = enable_c1_checkpointing_for_model(model, config, mode)
            apply_checkpointing(model, mode=mode)
        except Exception as exc:
            _cleanup_c1_runtime()
            state.status = "skipped_incompatible"
            state.reason = f"Checkpointing failed: {exc}"
            return (model, state)
    if baseline_cfg.torchao != "none":
        (model, torchao_result) = try_enable_torchao(model, mode=baseline_cfg.torchao)
        if not torchao_result.ok:
            _cleanup_c1_runtime()
            state.status = torchao_result.status.value
            state.reason = torchao_result.reason
            return (model, state)
    if baseline_cfg.gact != "none":
        if baseline_cfg.grad_accum_steps > 1:
            _cleanup_c1_runtime()
            state.status = "skipped_incompatible"
            state.reason = "GACT does not support grad_accum_steps > 1"
            return (model, state)
        (model, gact_controller, gact_result) = try_enable_gact(model, level=baseline_cfg.gact)
        if not gact_result.ok:
            _cleanup_c1_runtime()
            state.status = gact_result.status.value
            state.reason = gact_result.reason
            return (model, state)
        gact_controller.install_hook()
        state.gact_controller = gact_controller
    if baseline_cfg.amp != "none":
        state.amp_manager = AMPManager(baseline_cfg.amp, device=str(device))
    if baseline_cfg.saved_tensors_offload != "none":
        state.offloader = SavedTensorOffloader(
            mode=baseline_cfg.saved_tensors_offload,
            min_bytes=baseline_cfg.saved_tensors_offload_min_bytes,
        )
    return (model, state)


def _safe_perplexity_from_loss(loss: float) -> float:
    if not math.isfinite(loss):
        return float("inf")
    if loss >= 700.0:
        return float("inf")
    return math.exp(loss)


def _summarize_c1_first_nonfinite_event(per_site: Dict[str, Any]) -> Dict[str, Any]:
    earliest_event = None
    earliest_site_id = None
    for site_id, site_stats in per_site.items():
        step = site_stats.get("first_nonfinite_step")
        if step is None:
            continue
        try:
            step_value = int(step)
        except (TypeError, ValueError):
            continue
        candidate = (step_value, str(site_id))
        if earliest_event is None or candidate < earliest_event:
            earliest_event = candidate
            earliest_site_id = str(site_id)
    if earliest_event is None or earliest_site_id is None:
        return {}
    site_stats = per_site[earliest_site_id]
    return {
        "c1_first_nonfinite_step": int(earliest_event[0]),
        "c1_first_nonfinite_site": earliest_site_id,
        "c1_first_nonfinite_reason": site_stats.get("first_nonfinite_reason"),
        "c1_first_nonfinite_stage": site_stats.get("first_nonfinite_stage"),
        "c1_first_nonfinite_input_max_abs": site_stats.get("first_nonfinite_input_max_abs"),
        "c1_first_nonfinite_coeff_max_abs": site_stats.get("first_nonfinite_coeff_max_abs"),
        "c1_first_nonfinite_reconstruction_max_abs": site_stats.get(
            "first_nonfinite_reconstruction_max_abs"
        ),
        "c1_first_nonfinite_residual_max_abs": site_stats.get("first_nonfinite_residual_max_abs"),
        "c1_first_nonfinite_basis_max_abs": site_stats.get("first_nonfinite_basis_max_abs"),
    }


def _safe_finite_mean(values: List[Any]) -> Optional[float]:
    finite_values = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    if not finite_values:
        return None
    return sum(finite_values) / len(finite_values)


def _summarize_c1_memory_phase_snapshot(
    memory_phases: Optional[Dict[str, Optional[int]]],
) -> Dict[str, int]:
    if not memory_phases:
        return {}
    summary: Dict[str, int] = {}
    for phase, value in memory_phases.items():
        if value is None:
            continue
        summary[f"c1_memory_allocated_{phase}_bytes"] = int(value)
    return summary


def _update_c1_memory_phase_peaks(
    config: "ConvergenceConfig", memory_phases: Optional[Dict[str, Optional[int]]]
) -> Dict[str, int]:
    peaks = dict(getattr(config, "_c1_debug_memory_phase_peaks", {}))
    if memory_phases:
        for phase, value in memory_phases.items():
            if value is None:
                continue
            peaks[phase] = max(int(value), int(peaks.get(phase, 0)))
    config._c1_debug_memory_phase_peaks = peaks
    return peaks


def _summarize_c1_memory_phase_peaks(config: "ConvergenceConfig") -> Dict[str, int]:
    peaks = getattr(config, "_c1_debug_memory_phase_peaks", {}) or {}
    summary = {
        f"c1_memory_peak_{phase}_allocated_bytes": int(value) for (phase, value) in peaks.items()
    }
    before_forward = peaks.get("before_forward")
    if before_forward is not None:
        for phase, value in peaks.items():
            if phase == "before_forward":
                continue
            summary[f"c1_memory_peak_{phase}_delta_from_before_forward_bytes"] = int(value) - int(
                before_forward
            )
    return summary


def _update_c1_finite_drift_tracker(
    tracker: Dict[str, Any],
    runtime_stats: Dict[str, Any],
    *,
    global_step: int,
    optimizer_step: int,
    loss_value: float,
    running_loss: float,
    step_perplexity: Optional[float],
    running_perplexity: Optional[float],
) -> Dict[str, Any]:
    event = tracker.get("event")
    slow_event = tracker.get("slow_event")
    per_site = runtime_stats.get("per_site", {})
    compressed_sites = int(runtime_stats.get("site_count_compressed", 0))
    current_site_energies: Dict[str, float] = {}
    candidate_site_id: Optional[str] = None
    candidate_site_energy: Optional[float] = None
    candidate_site_delta: Optional[float] = None
    candidate_site_ratio: Optional[float] = None
    prev_site_energies = tracker.get("prev_site_energy", {})
    for site_id, site_stats in per_site.items():
        if int(site_stats.get("compressed_pack_count", 0)) <= 0:
            continue
        energy = site_stats.get("last_energy")
        if not isinstance(energy, (int, float)) or not math.isfinite(float(energy)):
            continue
        energy_value = float(energy)
        site_key = str(site_id)
        current_site_energies[site_key] = energy_value
        prev_energy = prev_site_energies.get(site_key)
        delta = None if prev_energy is None else energy_value - float(prev_energy)
        ratio = None
        if prev_energy is not None:
            denom = max(abs(float(prev_energy)), 1e-08)
            ratio = energy_value / denom
        score = (float("-inf") if delta is None else float(delta), energy_value, site_key)
        current_best = (
            float("-inf") if candidate_site_delta is None else float(candidate_site_delta),
            float("-inf") if candidate_site_energy is None else float(candidate_site_energy),
            "" if candidate_site_id is None else candidate_site_id,
        )
        if score > current_best:
            candidate_site_id = site_key
            candidate_site_energy = energy_value
            candidate_site_delta = delta
            candidate_site_ratio = ratio
    recent_step_perplexities = tracker["recent_step_perplexities"]
    recent_running_perplexities = tracker["recent_running_perplexities"]
    recent_losses = tracker["recent_losses"]
    best_step_perplexity = tracker.get("best_step_perplexity")
    if best_step_perplexity is None:
        prior_step_candidates = [
            float(value)
            for value in recent_step_perplexities
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        ]
        if prior_step_candidates:
            best_step_perplexity = min(prior_step_candidates)
    signal = None
    signal_reference = None
    if (
        step_perplexity is not None
        and math.isfinite(step_perplexity)
        and (len(recent_step_perplexities) >= 3)
    ):
        reference = _safe_finite_mean(list(recent_step_perplexities))
        if reference is not None and step_perplexity >= max(reference * 3.0, reference + 25.0):
            signal = "step_perplexity_spike"
            signal_reference = reference
    if signal is None and (
        running_perplexity is not None
        and math.isfinite(running_perplexity)
        and (len(recent_running_perplexities) >= 3)
    ):
        reference = _safe_finite_mean(list(recent_running_perplexities))
        if reference is not None and running_perplexity >= max(reference * 1.5, reference + 3.0):
            signal = "running_perplexity_spike"
            signal_reference = reference
    if signal is None and math.isfinite(loss_value) and (len(recent_losses) >= 3):
        reference = _safe_finite_mean(list(recent_losses))
        if reference is not None and loss_value >= max(reference * 2.0, reference + 2.0):
            signal = "loss_spike"
            signal_reference = reference
    slow_signal = None
    slow_signal_reference = None
    if (
        slow_event is None
        and compressed_sites > 0
        and (step_perplexity is not None)
        and math.isfinite(step_perplexity)
        and (len(recent_step_perplexities) >= 3)
        and (best_step_perplexity is not None)
        and math.isfinite(float(best_step_perplexity))
        and (step_perplexity > float(best_step_perplexity) * 1.25)
    ):
        slow_signal = "step_perplexity_above_best_seen_advisory"
        slow_signal_reference = float(best_step_perplexity)
    if event is None and signal is not None and (compressed_sites > 0):
        if candidate_site_id is None and current_site_energies:
            (candidate_site_id, candidate_site_energy) = max(
                current_site_energies.items(), key=lambda item: (item[1], item[0])
            )
            candidate_site_delta = None
            candidate_site_ratio = None
        if candidate_site_id is not None:
            site_stats = per_site.get(candidate_site_id, {})
            event = {
                "c1_first_finite_drift_step": int(global_step + 1),
                "c1_first_finite_drift_optimizer_step": int(optimizer_step),
                "c1_first_finite_drift_site": str(candidate_site_id),
                "c1_first_finite_drift_reason": signal,
                "c1_first_finite_drift_reference": signal_reference,
                "c1_first_finite_drift_loss": float(loss_value),
                "c1_first_finite_drift_running_loss": float(running_loss),
                "c1_first_finite_drift_step_perplexity": step_perplexity,
                "c1_first_finite_drift_running_perplexity": running_perplexity,
                "c1_first_finite_drift_site_energy": candidate_site_energy,
                "c1_first_finite_drift_site_energy_delta": candidate_site_delta,
                "c1_first_finite_drift_site_energy_ratio": candidate_site_ratio,
                "c1_first_finite_drift_input_max_abs": site_stats.get("last_input_max_abs"),
                "c1_first_finite_drift_coeff_max_abs": site_stats.get("last_coeff_max_abs"),
                "c1_first_finite_drift_reconstruction_max_abs": site_stats.get(
                    "last_reconstruction_max_abs"
                ),
                "c1_first_finite_drift_residual_max_abs": site_stats.get("last_residual_max_abs"),
                "c1_first_finite_drift_basis_max_abs": site_stats.get("last_basis_max_abs"),
            }
            tracker["event"] = event
    if slow_event is None and slow_signal is not None:
        if candidate_site_id is None and current_site_energies:
            (candidate_site_id, candidate_site_energy) = max(
                current_site_energies.items(), key=lambda item: (item[1], item[0])
            )
            candidate_site_delta = None
            candidate_site_ratio = None
        if candidate_site_id is not None:
            site_stats = per_site.get(candidate_site_id, {})
            slow_event = {
                "c1_first_slow_drift_step": int(global_step + 1),
                "c1_first_slow_drift_optimizer_step": int(optimizer_step),
                "c1_first_slow_drift_site": str(candidate_site_id),
                "c1_first_slow_drift_reason": slow_signal,
                "c1_first_slow_drift_reference": slow_signal_reference,
                "c1_first_slow_drift_loss": float(loss_value),
                "c1_first_slow_drift_running_loss": float(running_loss),
                "c1_first_slow_drift_step_perplexity": step_perplexity,
                "c1_first_slow_drift_running_perplexity": running_perplexity,
                "c1_first_slow_drift_site_energy": candidate_site_energy,
                "c1_first_slow_drift_site_energy_delta": candidate_site_delta,
                "c1_first_slow_drift_site_energy_ratio": candidate_site_ratio,
                "c1_first_slow_drift_input_max_abs": site_stats.get("last_input_max_abs"),
                "c1_first_slow_drift_coeff_max_abs": site_stats.get("last_coeff_max_abs"),
                "c1_first_slow_drift_reconstruction_max_abs": site_stats.get(
                    "last_reconstruction_max_abs"
                ),
                "c1_first_slow_drift_residual_max_abs": site_stats.get("last_residual_max_abs"),
                "c1_first_slow_drift_basis_max_abs": site_stats.get("last_basis_max_abs"),
            }
            tracker["slow_event"] = slow_event
    recent_losses.append(float(loss_value))
    if step_perplexity is not None and math.isfinite(step_perplexity):
        recent_step_perplexities.append(float(step_perplexity))
        if best_step_perplexity is None:
            tracker["best_step_perplexity"] = float(step_perplexity)
        else:
            tracker["best_step_perplexity"] = min(
                float(best_step_perplexity), float(step_perplexity)
            )
    if running_perplexity is not None and math.isfinite(running_perplexity):
        recent_running_perplexities.append(float(running_perplexity))
    tracker["prev_site_energy"] = current_site_energies
    combined_event = dict(tracker.get("event") or {})
    combined_event.update(dict(tracker.get("slow_event") or {}))
    return combined_event


def _summarize_c1_first_finite_drift_event(
    config: "ConvergenceConfig",
    runtime_stats: Dict[str, Any],
    *,
    global_step: int,
    optimizer_step: int,
    loss_value: float,
    running_loss: float,
    step_perplexity: Optional[float],
    running_perplexity: Optional[float],
) -> Dict[str, Any]:
    tracker = getattr(config, "_c1_finite_drift_tracker", None)
    if tracker is None:
        tracker = {
            "recent_losses": deque(maxlen=5),
            "recent_step_perplexities": deque(maxlen=5),
            "recent_running_perplexities": deque(maxlen=5),
            "prev_site_energy": {},
            "event": None,
            "slow_event": None,
            "best_step_perplexity": None,
        }
        config._c1_finite_drift_tracker = tracker
    return _update_c1_finite_drift_tracker(
        tracker,
        runtime_stats,
        global_step=global_step,
        optimizer_step=optimizer_step,
        loss_value=loss_value,
        running_loss=running_loss,
        step_perplexity=step_perplexity,
        running_perplexity=running_perplexity,
    )


def _get_c1_first_finite_drift_event(config: "ConvergenceConfig") -> Dict[str, Any]:
    tracker = getattr(config, "_c1_finite_drift_tracker", None)
    if not isinstance(tracker, dict):
        return {}
    event = tracker.get("event")
    slow_event = tracker.get("slow_event")
    payload: Dict[str, Any] = {}
    if isinstance(event, dict):
        payload.update(event)
    if isinstance(slow_event, dict):
        payload.update(slow_event)
    return payload


def _summarize_c1_runtime_for_debug(runtime_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Build a compressed-checkpoint snapshot for stepwise debug metrics."""
    per_site = runtime_stats.get("per_site", {})
    fallback_reasons: Dict[str, int] = {}
    total_pack_count = 0
    total_unpack_count = 0
    total_compressed_pack_count = 0
    total_fallback_count = 0
    total_step_count = 0
    compressed_last_energies: List[float] = []
    compressed_energy_ewmas: List[float] = []
    for site_stats in per_site.values():
        total_pack_count += int(site_stats.get("pack_count", 0))
        total_unpack_count += int(site_stats.get("unpack_count", 0))
        total_compressed_pack_count += int(site_stats.get("compressed_pack_count", 0))
        total_fallback_count += int(site_stats.get("fallback_count", 0))
        total_step_count += int(site_stats.get("step_count", 0))
        if int(site_stats.get("compressed_pack_count", 0)) > 0:
            compressed_last_energies.append(float(site_stats.get("last_energy", 0.0)))
            compressed_energy_ewmas.append(float(site_stats.get("energy_ewma", 0.0)))
        for reason, count in site_stats.get("fallback_reasons", {}).items():
            fallback_reasons[reason] = fallback_reasons.get(reason, 0) + int(count)
    summary = {
        "c1_site_count_active": int(runtime_stats.get("site_count_active", 0)),
        "c1_site_count_engaged": int(runtime_stats.get("site_count_engaged", 0)),
        "c1_site_count_compressed": int(runtime_stats.get("site_count_compressed", 0)),
        "c1_total_pack_count": total_pack_count,
        "c1_total_unpack_count": total_unpack_count,
        "c1_total_compressed_pack_count": total_compressed_pack_count,
        "c1_total_fallback_count": total_fallback_count,
        "c1_total_step_count": total_step_count,
    }
    if compressed_last_energies:
        summary["c1_last_energy_max"] = max(compressed_last_energies)
        summary["c1_last_energy_mean"] = sum(compressed_last_energies) / len(
            compressed_last_energies
        )
    if compressed_energy_ewmas:
        summary["c1_energy_ewma_max"] = max(compressed_energy_ewmas)
        summary["c1_energy_ewma_mean"] = sum(compressed_energy_ewmas) / len(compressed_energy_ewmas)
    if fallback_reasons:
        summary["c1_fallback_reasons"] = dict(sorted(fallback_reasons.items()))
    summary.update(_summarize_c1_first_nonfinite_event(per_site))
    return summary


def _build_debug_step_metrics_payload(
    config: ConvergenceConfig,
    optimizer: optim.Optimizer,
    *,
    epoch: int,
    step: int,
    global_step: int,
    loss_value: float,
    running_accuracy: float,
    running_loss: float,
    memory_phases: Optional[Dict[str, Optional[int]]] = None,
) -> Dict[str, Any]:
    """Build an opt-in per-step payload for medium-horizon drift bug-hunts."""
    uses_stacr_step_counter = (
        config.method in ("stacr", "stacr-v4")
        or config.stacr_attention
        or getattr(config, "stacr_compress_checkpoint", False)
    )
    payload: Dict[str, Any] = {
        "record_type": "debug_step",
        "epoch": epoch,
        "step": step,
        "global_step": global_step,
        "optimizer_step": int(getattr(config, "_stacr_optimizer_steps_completed", 0))
        if uses_stacr_step_counter
        else int(getattr(config, "_prac_optimizer_steps_completed", 0))
        if config.method == "prac"
        else global_step + 1,
        "loss": loss_value,
        "running_loss": running_loss,
        "running_accuracy": running_accuracy,
        "lr": float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else 0.0,
    }
    if config.task == "wikitext103":
        payload["step_perplexity"] = _safe_perplexity_from_loss(loss_value)
        payload["running_perplexity"] = _safe_perplexity_from_loss(running_loss)
    runtime = get_active_c1_runtime()
    if runtime is not None:
        runtime_stats = runtime.stats()
        payload.update(_summarize_c1_runtime_for_debug(runtime_stats))
        payload.update(_get_c1_first_finite_drift_event(config))
    payload.update(_summarize_c1_memory_phase_snapshot(memory_phases))
    _update_c1_memory_phase_peaks(config, memory_phases)
    return payload


@torch.no_grad()
def validate(
    model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device, task: str
) -> Dict[str, float]:
    """Run validation and return metrics."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    total_tokens = 0
    for batch in val_loader:
        if task in ("sst2", "wikitext103"):
            inputs = {k: v.to(device) for (k, v) in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            if loss.dim() > 0:
                loss = loss.mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite validation loss")
            logits = outputs.logits
            labels = inputs["labels"]
            if task == "wikitext103":
                total_tokens += labels.numel()
                total_loss += loss.item() * labels.numel()
                total += labels.numel()
            else:
                total_loss += loss.item() * labels.size(0)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        else:
            (inputs, labels) = batch
            (inputs, labels) = (inputs.to(device), labels.to(device))
            logits = model(inputs)
            loss = criterion(logits, labels)
            if loss.dim() > 0:
                loss = loss.mean()
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    model.train()
    result = {"loss": total_loss / total}
    if task == "wikitext103":
        result["perplexity"] = _safe_perplexity_from_loss(result["loss"])
        result["accuracy"] = 0.0
    else:
        result["accuracy"] = correct / total * 100
    return result


def compute_grad_cosine(
    model: nn.Module,
    inputs: Any,
    labels: torch.Tensor,
    criterion: Optional[nn.Module],
    task: str,
    amp_manager: Optional[Any] = None,
) -> Dict[str, Any]:
    """Compute gradient cosine between compressed and baseline paths.

    Supports three compression surfaces:
      - STACRLinear: temporarily forces warmup_steps=999999 so the layer takes
        its dense baseline path on the second pass.
      - PRACLinear: temporarily sets _disable_prac_compression=True so the
        layer falls through to its exact nn.Linear path.
      - CompressedLinear (vanilla SVD/TT/Tucker): temporarily sets
        config["method"]="none" so CompressedFunction saves the dense
        activation directly and computes the exact grad.
    """
    from stacr.diagnostics import compression_diagnostics
    from stacr.metrics import cosine_similarity

    diagnostic_scope = compression_diagnostics(model)
    diagnostic_entered = False
    original_configs = {}
    original_prac_bypass = {}
    original_compressed_methods = {}
    try:
        from stacr.protocols.layers import CompressedLinear
    except Exception:
        CompressedLinear = None
    try:
        diagnostic_scope.__enter__()
        diagnostic_entered = True
        model.zero_grad()
        if amp_manager:
            with amp_manager.autocast():
                if task in ("sst2", "wikitext103"):
                    outputs = model(**inputs)
                    loss = outputs.loss
                else:
                    logits = model(inputs)
                    loss = criterion(logits, labels)
            amp_manager.backward(loss)
        else:
            if task in ("sst2", "wikitext103"):
                outputs = model(**inputs)
                loss = outputs.loss
            else:
                logits = model(inputs)
                loss = criterion(logits, labels)
            loss.backward()
        compressed_grads = {
            n: p.grad.clone().float()
            for (n, p) in model.named_parameters()
            if p.grad is not None and "weight" in n
        }
        try:
            from stacr.protocols.prac import PRACLinear
        except Exception:
            PRACLinear = None
        for name, mod in model.named_modules():
            if isinstance(mod, STACRLinear):
                original_configs[name] = {
                    "residual_mode": mod.config.residual_mode,
                    "warmup_steps": mod.config.warmup_steps,
                }
                mod.config.warmup_steps = 999999
            elif PRACLinear is not None and isinstance(mod, PRACLinear):
                original_prac_bypass[name] = bool(getattr(mod, "_disable_prac_compression", False))
                mod._disable_prac_compression = True
            elif CompressedLinear is not None and isinstance(mod, CompressedLinear):
                cfg = mod.config if isinstance(mod.config, dict) else None
                if cfg is not None and cfg.get("method", "none") in ("svd", "tt", "tucker"):
                    original_compressed_methods[name] = cfg["method"]
                    cfg["method"] = "none"
        model.zero_grad()
        if amp_manager:
            with amp_manager.autocast():
                if task in ("sst2", "wikitext103"):
                    outputs = model(**inputs)
                    loss = outputs.loss
                else:
                    logits = model(inputs)
                    loss = criterion(logits, labels)
            amp_manager.backward(loss)
        else:
            if task in ("sst2", "wikitext103"):
                outputs = model(**inputs)
                loss = outputs.loss
            else:
                logits = model(inputs)
                loss = criterion(logits, labels)
            loss.backward()
        baseline_grads = {
            n: p.grad.clone().float()
            for (n, p) in model.named_parameters()
            if p.grad is not None and "weight" in n
        }
        cosines = {}
        for name in compressed_grads:
            if name in baseline_grads:
                g1 = compressed_grads[name].flatten()
                g2 = baseline_grads[name].flatten()
                cosines[name] = float(cosine_similarity(g1, g2))
        if not cosines:
            return {
                "grad_cosine_mean": float("nan"),
                "grad_cosine_min": float("nan"),
                "grad_cosine_by_layer": {},
            }
        return {
            "grad_cosine_mean": sum(cosines.values()) / len(cosines),
            "grad_cosine_min": min(cosines.values()),
            "grad_cosine_by_layer": cosines,
        }
    finally:
        for name, mod in model.named_modules():
            if isinstance(mod, STACRLinear) and name in original_configs:
                mod.config.residual_mode = original_configs[name]["residual_mode"]
                mod.config.warmup_steps = original_configs[name]["warmup_steps"]
            elif name in original_prac_bypass:
                mod._disable_prac_compression = original_prac_bypass[name]
            elif (
                CompressedLinear is not None
                and isinstance(mod, CompressedLinear)
                and (name in original_compressed_methods)
                and isinstance(mod.config, dict)
            ):
                mod.config["method"] = original_compressed_methods[name]
        model.zero_grad(set_to_none=True)
        if diagnostic_entered:
            diagnostic_scope.__exit__(*sys.exc_info())


def _flatten_last_dim(tensor: torch.Tensor, features: int) -> torch.Tensor:
    """Flatten all sample/token dimensions while preserving the feature axis."""
    return tensor.reshape(-1, features)


def _effective_stacr_residual_mode(module: STACRLinear) -> str:
    """Resolve the STAC-R residual mode active at the current schedule step."""
    residual_mode = getattr(module.config, "residual_mode", "none")
    post_warmup_mode = getattr(module.config, "post_warmup_residual_mode", "full")
    residual_warmup = int(getattr(module.config, "residual_warmup_steps", 0))
    schedule_step = int(getattr(module, "layer_state", {}).get("_schedule_step", 0))
    if post_warmup_mode != "full" and residual_warmup > 0 and (schedule_step >= residual_warmup):
        if post_warmup_mode == "none":
            return "none"
    return residual_mode


def _quantize_dequantize_coefficients(
    coefficients: torch.Tensor, module: STACRLinear
) -> Tuple[torch.Tensor, bool]:
    if not bool(getattr(module.config, "quantize_coefficients", False)):
        return (coefficients, False)
    bits = int(getattr(module.config, "quantize_bits", 8))
    if bits != 8:
        return (coefficients, False)
    (q, scale) = quantize_int8(coefficients)
    return (dequantize_int8(q, scale).to(coefficients.dtype), True)


def _quantize_dequantize_residual_values(
    values: torch.Tensor, module: STACRLinear
) -> Tuple[torch.Tensor, bool]:
    if values.numel() == 0 or not bool(getattr(module.config, "quantize_residuals", False)):
        return (values, False)
    bits = int(getattr(module.config, "residual_quantize_bits", 8))
    if bits == 8:
        (q, scale) = quantize_int8(values)
        return (dequantize_int8(q, scale).to(values.dtype), True)
    if bits == 4:
        (q, scale, orig_len) = quantize_int4(values)
        return (dequantize_int4(q, scale, orig_len).to(values.dtype), True)
    return (values, False)


def _stacr_reconstruct_activation(
    module: STACRLinear, activation_2d: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Reconstruct a STAC-R layer input using the current basis and residual policy.

    This measures a reconstruction policy, not the actual saved backward factors.
    Top-k and block selection use global quotas; training uses chunk quotas.
    Coefficient quantization follows configuration even for the unquantized
    low-rank-only fast path. Warmup and residual skip gates are not replayed;
    unsupported selectors are labeled as low-rank-only diagnostics. Use
    compute_grad_cosine for paired gradients from the actual backward path.
    """
    basis = module.effective_basis if getattr(module, "basis_initialized", False) else None
    if basis is None or basis.numel() == 0:
        return (
            activation_2d,
            {
                "reconstruction_scope": "baseline_or_uninitialized_basis",
                "rank": 0,
                "residual_mode": "baseline",
                "residual_budget": 0.0,
                "residual_entries_kept": activation_2d.numel(),
                "coefficients_quantized": False,
                "residuals_quantized": False,
            },
        )
    active_rank = max(
        1,
        min(
            int(getattr(module, "layer_state", {}).get("active_rank", basis.shape[1])),
            basis.shape[1],
        ),
    )
    basis = basis[:, :active_rank].to(device=activation_2d.device, dtype=activation_2d.dtype)
    coefficients = activation_2d.mm(basis)
    low_rank_exact = coefficients.mm(basis.t())
    (coefficients_recon, coeff_q) = _quantize_dequantize_coefficients(coefficients, module)
    low_rank_recon = coefficients_recon.mm(basis.t())
    residual_mode = _effective_stacr_residual_mode(module)
    residual_budget = float(getattr(module.config, "residual_budget", 0.0))
    layer_state = getattr(module, "layer_state", {})
    if str(getattr(module.config, "residual_schedule", "none")) != "none":
        residual_budget = float(layer_state.get("current_residual_budget", residual_budget))
    residual_budget = max(0.0, min(1.0, residual_budget))
    residual = activation_2d - low_rank_exact
    residual_recon = torch.zeros_like(residual)
    residual_entries_kept = 0
    residuals_quantized = False
    if residual_mode == "none" or residual_budget <= 0.0:
        pass
    elif residual_mode == "topk":
        k = min(int(residual.numel() * residual_budget), residual.numel())
        if k > 0:
            flat = residual.reshape(-1)
            (_, idx) = torch.topk(flat.abs(), k)
            (values, residuals_quantized) = _quantize_dequantize_residual_values(flat[idx], module)
            residual_recon.reshape(-1)[idx] = values
            residual_entries_kept = int(idx.numel())
    elif residual_mode == "threshold":
        flat = residual.reshape(-1)
        mask = flat.abs() > float(getattr(module.config, "residual_threshold", 0.1))
        idx = mask.nonzero(as_tuple=False).reshape(-1)
        if idx.numel() > 0:
            (values, residuals_quantized) = _quantize_dequantize_residual_values(flat[idx], module)
            residual_recon.reshape(-1)[idx] = values
            residual_entries_kept = int(idx.numel())
    elif residual_mode == "block_sparse":
        (b_flat, in_features) = residual.shape
        block_size = max(1, int(getattr(module.config, "residual_block_size", 16)))
        k_elements = min(int(residual.numel() * residual_budget), residual.numel())
        if k_elements > 0:
            blocks_per_row = (in_features + block_size - 1) // block_size
            pad_features = blocks_per_row * block_size - in_features
            residual_padded = (
                torch.nn.functional.pad(residual, (0, pad_features)) if pad_features else residual
            )
            block_view = residual_padded.contiguous().view(b_flat, blocks_per_row, block_size)
            block_scores = (block_view * block_view).sum(dim=2)
            k_blocks = min(max(1, math.ceil(k_elements / block_size)), block_scores.numel())
            (_, block_idx) = torch.topk(block_scores.reshape(-1), k_blocks)
            rows = torch.div(block_idx, blocks_per_row, rounding_mode="floor")
            block_in_row = block_idx % blocks_per_row
            offsets = torch.arange(block_size, device=residual.device)
            cols = block_in_row[:, None] * block_size + offsets[None, :]
            valid = cols < in_features
            flat_idx = rows[:, None] * in_features + cols
            flat_idx = flat_idx[valid]
            flat = residual.reshape(-1)
            (values, residuals_quantized) = _quantize_dequantize_residual_values(
                flat[flat_idx], module
            )
            residual_recon.reshape(-1)[flat_idx] = values
            residual_entries_kept = int(flat_idx.numel())
    else:
        residual_mode = f"{residual_mode}_low_rank_only_diagnostic"
    return (
        low_rank_recon + residual_recon,
        {
            "reconstruction_scope": "stacr_current_basis_residual_policy",
            "rank": int(active_rank),
            "residual_mode": residual_mode,
            "residual_budget": residual_budget,
            "residual_entries_kept": residual_entries_kept,
            "coefficients_quantized": bool(coeff_q),
            "residuals_quantized": bool(residuals_quantized),
        },
    )


def _prac_reconstruct_activation(
    module: nn.Module, activation_2d: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    projection = module.projection.to(device=activation_2d.device, dtype=activation_2d.dtype)
    projected = activation_2d.mm(projection)
    reconstructed = projected.mm(projection.t())
    return (
        reconstructed,
        {
            "reconstruction_scope": "prac_linear_random_projection",
            "rank": int(getattr(module, "effective_rank", projection.shape[1])),
            "residual_mode": "none",
            "residual_budget": 0.0,
            "residual_entries_kept": 0,
            "coefficients_quantized": False,
            "residuals_quantized": False,
        },
    )


def _svd_lowrank_reconstruct_activation(
    activation_2d: torch.Tensor, rank: int, oversampling: int = 16, niter: int = 2
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Reconstruct an activation matrix with a vanilla low-rank SVD diagnostic.

    This is intentionally a measurement-only path for matched-surface cosine
    comparisons. It uses ``torch.svd_lowrank`` so GPT-2 c_proj inputs remain
    feasible on a 32 GiB card; the paper labels it as an SVD diagnostic rather
    than as a training-time implementation.
    """
    (sample_dim, feature_dim) = activation_2d.shape
    max_rank = min(sample_dim, feature_dim)
    active_rank = max(1, min(int(rank), int(max_rank)))
    q = max(active_rank, active_rank + max(0, int(oversampling)))
    q = min(q, int(max_rank))
    niter = max(0, int(niter))
    work = activation_2d.to(dtype=torch.float32)
    try:
        (u, s, v) = torch.svd_lowrank(work, q=q, niter=niter)
        reconstructed = (u[:, :active_rank] * s[:active_rank]).mm(v[:, :active_rank].t())
        svd_backend = "torch.svd_lowrank"
    except RuntimeError:
        (u, s, vh) = torch.linalg.svd(work, full_matrices=False)
        reconstructed = (u[:, :active_rank] * s[:active_rank]).mm(vh[:active_rank, :])
        svd_backend = "torch.linalg.svd"
        q = active_rank
    return (
        reconstructed.to(dtype=activation_2d.dtype),
        {
            "reconstruction_scope": "svd_lowrank_diagnostic_same_activation",
            "rank": int(active_rank),
            "residual_mode": "none",
            "residual_budget": 0.0,
            "residual_entries_kept": 0,
            "coefficients_quantized": False,
            "residuals_quantized": False,
            "svd_backend": svd_backend,
            "svd_q": int(q),
            "svd_niter": int(niter),
        },
    )


def _summarize_per_layer_grad_fidelity(records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not records:
        return None
    by_layer: Dict[str, Dict[str, Any]] = {}
    finite_products: List[float] = []
    finite_recon_rel: List[float] = []
    finite_grad_cos: List[float] = []
    for record in records:
        layer = str(record.get("layer_name"))
        slot = by_layer.setdefault(
            layer,
            {
                "layer_name": layer,
                "measurements": 0,
                "upstream_grad_norm_fro_sum": 0.0,
                "reconstruction_error_fro_sum": 0.0,
                "reconstruction_rel_fro_sum": 0.0,
                "lemma1_product_bound_sum": 0.0,
                "grad_rel_fro_sum": 0.0,
                "grad_cosine_sum": 0.0,
                "rank": record.get("rank"),
                "residual_mode": record.get("residual_mode"),
            },
        )
        slot["measurements"] += 1
        for src, dst in [
            ("upstream_grad_norm_fro", "upstream_grad_norm_fro_sum"),
            ("reconstruction_error_fro", "reconstruction_error_fro_sum"),
            ("reconstruction_rel_fro", "reconstruction_rel_fro_sum"),
            ("lemma1_product_bound", "lemma1_product_bound_sum"),
            ("grad_rel_fro", "grad_rel_fro_sum"),
            ("grad_cosine", "grad_cosine_sum"),
        ]:
            value = record.get(src)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                slot[dst] += float(value)
        for key, target in [
            ("lemma1_product_bound", finite_products),
            ("reconstruction_rel_fro", finite_recon_rel),
            ("grad_cosine", finite_grad_cos),
        ]:
            value = record.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                target.append(float(value))
    layer_summaries = []
    for slot in by_layer.values():
        n = max(1, int(slot["measurements"]))
        layer_summaries.append(
            {
                "layer_name": slot["layer_name"],
                "measurements": n,
                "rank": slot.get("rank"),
                "residual_mode": slot.get("residual_mode"),
                "upstream_grad_norm_fro_mean": slot["upstream_grad_norm_fro_sum"] / n,
                "reconstruction_error_fro_mean": slot["reconstruction_error_fro_sum"] / n,
                "reconstruction_rel_fro_mean": slot["reconstruction_rel_fro_sum"] / n,
                "lemma1_product_bound_mean": slot["lemma1_product_bound_sum"] / n,
                "grad_rel_fro_mean": slot["grad_rel_fro_sum"] / n,
                "grad_cosine_mean": slot["grad_cosine_sum"] / n,
            }
        )
    layer_summaries.sort(key=lambda row: row["lemma1_product_bound_mean"], reverse=True)
    return {
        "num_records": len(records),
        "num_layers": len(by_layer),
        "num_measurement_steps": len({(r.get("epoch"), r.get("global_step")) for r in records}),
        "lemma1_product_bound_mean": sum(finite_products) / len(finite_products)
        if finite_products
        else None,
        "reconstruction_rel_fro_mean": sum(finite_recon_rel) / len(finite_recon_rel)
        if finite_recon_rel
        else None,
        "grad_cosine_mean": sum(finite_grad_cos) / len(finite_grad_cos)
        if finite_grad_cos
        else None,
        "top_lemma1_product_layers": layer_summaries[:8],
    }


def compute_per_layer_grad_fidelity(
    model: nn.Module,
    inputs: Any,
    labels: torch.Tensor,
    criterion: Optional[nn.Module],
    task: str,
    amp_manager: Optional[Any] = None,
    reconstruction_override: Optional[str] = None,
    svd_rank: Optional[int] = None,
    svd_oversampling: int = 16,
    svd_niter: int = 2,
) -> List[Dict[str, Any]]:
    """Capture per-layer ``||G_l||`` and reconstruction-error telemetry.

    The hook payloads are staged on CPU to avoid keeping all GPT-2 MLP
    activations live on GPU. Reconstruction and ``G^T E`` products are computed
    one layer at a time on the layer's device.
    """
    try:
        from stacr.protocols.prac import PRACLinear
    except Exception:
        PRACLinear = None
    wrapped_modules = []
    for name, module in model.named_modules():
        if isinstance(module, STACRLinear) or (
            PRACLinear is not None and isinstance(module, PRACLinear)
        ):
            wrapped_modules.append((name, module))
    if not wrapped_modules:
        return []
    was_training = model.training
    model.eval()
    captured: Dict[str, Dict[str, torch.Tensor]] = {}
    handles = []

    def make_forward_hook(layer_name: str):

        def hook(_module, module_inputs, _output):
            if not module_inputs:
                return
            activation = module_inputs[0]
            if isinstance(activation, torch.Tensor) and layer_name not in captured:
                captured.setdefault(layer_name, {})["activation"] = activation.detach().to(
                    "cpu", dtype=torch.float32
                )

        return hook

    def make_backward_hook(layer_name: str):

        def hook(_module, _grad_input, grad_output):
            if not grad_output:
                return
            upstream = grad_output[0]
            if isinstance(upstream, torch.Tensor):
                captured.setdefault(layer_name, {})["upstream_grad"] = upstream.detach().to(
                    "cpu", dtype=torch.float32
                )

        return hook

    for layer_name, module in wrapped_modules:
        handles.append(module.register_forward_hook(make_forward_hook(layer_name)))
        handles.append(module.register_full_backward_hook(make_backward_hook(layer_name)))
    try:
        model.zero_grad(set_to_none=True)
        if amp_manager:
            with amp_manager.autocast():
                if task in ("sst2", "wikitext103"):
                    outputs = model(**inputs)
                    loss = outputs.loss
                else:
                    logits = model(inputs)
                    loss = criterion(logits, labels)
        elif task in ("sst2", "wikitext103"):
            outputs = model(**inputs)
            loss = outputs.loss
        else:
            logits = model(inputs)
            loss = criterion(logits, labels)
        if loss.dim() > 0:
            loss = loss.mean()
        loss.backward()
    finally:
        for handle in handles:
            handle.remove()
    records: List[Dict[str, Any]] = []
    module_by_name = dict(wrapped_modules)
    with torch.no_grad():
        for layer_name, payload in captured.items():
            if "activation" not in payload or "upstream_grad" not in payload:
                continue
            module = module_by_name.get(layer_name)
            if module is None:
                continue
            device = (
                module.linear.weight.device if hasattr(module, "linear") else torch.device("cpu")
            )
            if reconstruction_override == "svd_lowrank":
                in_features = int(module.in_features)
                method = "svd_lowrank"
            elif isinstance(module, STACRLinear):
                in_features = int(module.in_features)
                method = "stacr"
            else:
                in_features = int(module.in_features)
                method = "prac"
            activation = _flatten_last_dim(payload["activation"], in_features).to(
                device=device, dtype=torch.float32
            )
            upstream = (
                payload["upstream_grad"]
                .reshape(-1, payload["upstream_grad"].shape[-1])
                .to(device=device, dtype=torch.float32)
            )
            if activation.shape[0] != upstream.shape[0]:
                continue
            if reconstruction_override == "svd_lowrank":
                rank = int(
                    svd_rank if svd_rank is not None else getattr(module, "effective_rank", 8)
                )
                (reconstructed, recon_meta) = _svd_lowrank_reconstruct_activation(
                    activation, rank=rank, oversampling=svd_oversampling, niter=svd_niter
                )
            elif isinstance(module, STACRLinear):
                (reconstructed, recon_meta) = _stacr_reconstruct_activation(module, activation)
            else:
                (reconstructed, recon_meta) = _prac_reconstruct_activation(module, activation)
            error = activation - reconstructed
            activation_norm = activation.norm()
            error_norm = error.norm()
            upstream_norm = upstream.norm()
            d_w_true = upstream.t().mm(activation)
            d_w_error = upstream.t().mm(error)
            d_w_hat = d_w_true - d_w_error
            d_w_true_norm = d_w_true.norm()
            d_w_error_norm = d_w_error.norm()
            d_w_hat_norm = d_w_hat.norm()
            denom = d_w_true_norm * d_w_hat_norm
            grad_cosine = (
                (d_w_true.flatten() @ d_w_hat.flatten() / denom).item()
                if denom.item() > 1e-12
                else float("nan")
            )
            grad_rel = (d_w_error_norm / (d_w_true_norm + 1e-12)).item()
            records.append(
                {
                    "layer_name": layer_name,
                    "method": method,
                    "rank": recon_meta.get("rank"),
                    "residual_mode": recon_meta.get("residual_mode"),
                    "residual_budget": recon_meta.get("residual_budget"),
                    "reconstruction_scope": recon_meta.get("reconstruction_scope"),
                    "residual_entries_kept": recon_meta.get("residual_entries_kept"),
                    "coefficients_quantized": recon_meta.get("coefficients_quantized"),
                    "residuals_quantized": recon_meta.get("residuals_quantized"),
                    "svd_backend": recon_meta.get("svd_backend"),
                    "svd_q": recon_meta.get("svd_q"),
                    "svd_niter": recon_meta.get("svd_niter"),
                    "activation_shape": list(activation.shape),
                    "upstream_grad_shape": list(upstream.shape),
                    "activation_norm_fro": activation_norm.item(),
                    "upstream_grad_norm_fro": upstream_norm.item(),
                    "reconstruction_error_fro": error_norm.item(),
                    "reconstruction_rel_fro": (error_norm / (activation_norm + 1e-12)).item(),
                    "lemma1_product_bound": (upstream_norm * error_norm).item(),
                    "dW_true_norm_fro": d_w_true_norm.item(),
                    "dW_error_norm_fro": d_w_error_norm.item(),
                    "grad_rel_fro": grad_rel,
                    "grad_cosine": grad_cosine,
                }
            )
    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    return records


def _summarize_activation_spectrum(records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not records:
        return None
    finite_explained = [
        float(r["explained_variance_rank"])
        for r in records
        if isinstance(r.get("explained_variance_rank"), (int, float))
        and math.isfinite(float(r["explained_variance_rank"]))
    ]
    finite_stable = [
        float(r["stable_rank"])
        for r in records
        if isinstance(r.get("stable_rank"), (int, float)) and math.isfinite(float(r["stable_rank"]))
    ]
    finite_rhs = [
        float(r["learned_vs_random_rhs"])
        for r in records
        if isinstance(r.get("learned_vs_random_rhs"), (int, float))
        and math.isfinite(float(r["learned_vs_random_rhs"]))
    ]
    finite_basis = [
        float(r["current_basis_explained_variance"])
        for r in records
        if isinstance(r.get("current_basis_explained_variance"), (int, float))
        and math.isfinite(float(r["current_basis_explained_variance"]))
    ]
    finite_random = [
        float(r["random_orthogonal_energy_fraction"])
        for r in records
        if isinstance(r.get("random_orthogonal_energy_fraction"), (int, float))
        and math.isfinite(float(r["random_orthogonal_energy_fraction"]))
    ]
    by_layer: Dict[str, Dict[str, Any]] = {}
    for record in records:
        layer = str(record.get("layer_name"))
        slot = by_layer.setdefault(
            layer,
            {
                "layer_name": layer,
                "measurements": 0,
                "rank": record.get("rank"),
                "feature_dim": record.get("feature_dim"),
                "explained_variance_rank_sum": 0.0,
                "stable_rank_sum": 0.0,
                "learned_vs_random_rhs_sum": 0.0,
                "tail_energy_fraction_rank_sum": 0.0,
            },
        )
        slot["measurements"] += 1
        for key in [
            "explained_variance_rank",
            "stable_rank",
            "learned_vs_random_rhs",
            "tail_energy_fraction_rank",
        ]:
            value = record.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                slot[f"{key}_sum"] += float(value)
    layer_summaries = []
    for slot in by_layer.values():
        n = max(1, int(slot["measurements"]))
        layer_summaries.append(
            {
                "layer_name": slot["layer_name"],
                "measurements": n,
                "rank": slot.get("rank"),
                "feature_dim": slot.get("feature_dim"),
                "explained_variance_rank_mean": slot["explained_variance_rank_sum"] / n,
                "tail_energy_fraction_rank_mean": slot["tail_energy_fraction_rank_sum"] / n,
                "stable_rank_mean": slot["stable_rank_sum"] / n,
                "learned_vs_random_rhs_mean": slot["learned_vs_random_rhs_sum"] / n,
            }
        )
    layer_summaries.sort(key=lambda row: row["tail_energy_fraction_rank_mean"], reverse=True)
    return {
        "num_records": len(records),
        "num_layers": len(by_layer),
        "num_measurement_steps": len({(r.get("epoch"), r.get("global_step")) for r in records}),
        "explained_variance_rank_mean": sum(finite_explained) / len(finite_explained)
        if finite_explained
        else None,
        "tail_energy_fraction_rank_mean": 1.0 - sum(finite_explained) / len(finite_explained)
        if finite_explained
        else None,
        "stable_rank_mean": sum(finite_stable) / len(finite_stable) if finite_stable else None,
        "learned_vs_random_rhs_mean": sum(finite_rhs) / len(finite_rhs) if finite_rhs else None,
        "current_basis_explained_variance_mean": sum(finite_basis) / len(finite_basis)
        if finite_basis
        else None,
        "random_orthogonal_energy_fraction_mean": sum(finite_random) / len(finite_random)
        if finite_random
        else None,
        "top_tail_layers": layer_summaries[:8],
    }


def compute_activation_spectrum_telemetry(
    model: nn.Module, inputs: Any, task: str, rank: int = 8, oversampling: int = 16, niter: int = 2
) -> List[Dict[str, Any]]:
    """Capture activation spectrum telemetry for wrapped linear-layer inputs.

    This is a diagnostic path for the learned-vs-random PRAC-lite corollary.
    It computes exact Frobenius energy and approximate top singular values
    using ``torch.svd_lowrank`` with small ``q``. The activations are staged on
    CPU by hooks and processed one layer at a time to avoid retaining all MLP
    matrices on GPU.
    """
    try:
        from stacr.protocols.prac import PRACLinear
    except Exception:
        PRACLinear = None
    wrapped_modules = []
    for name, module in model.named_modules():
        if isinstance(module, STACRLinear) or (
            PRACLinear is not None and isinstance(module, PRACLinear)
        ):
            wrapped_modules.append((name, module))
    if not wrapped_modules:
        return []
    was_training = model.training
    model.eval()
    captured: Dict[str, torch.Tensor] = {}
    handles = []

    def make_forward_hook(layer_name: str):

        def hook(_module, module_inputs, _output):
            if not module_inputs or layer_name in captured:
                return
            activation = module_inputs[0]
            if isinstance(activation, torch.Tensor):
                captured[layer_name] = activation.detach().to("cpu", dtype=torch.float32)

        return hook

    for layer_name, module in wrapped_modules:
        handles.append(module.register_forward_hook(make_forward_hook(layer_name)))
    try:
        with torch.no_grad():
            if task in ("sst2", "wikitext103"):
                _ = model(**inputs)
            else:
                _ = model(inputs)
    finally:
        for handle in handles:
            handle.remove()
    records: List[Dict[str, Any]] = []
    module_by_name = dict(wrapped_modules)
    rank = max(1, int(rank))
    oversampling = max(0, int(oversampling))
    niter = max(0, int(niter))
    with torch.no_grad():
        for layer_name, activation_cpu in captured.items():
            module = module_by_name.get(layer_name)
            if module is None:
                continue
            if isinstance(module, STACRLinear):
                in_features = int(module.in_features)
                method = "stacr"
                device = module.linear.weight.device
            else:
                in_features = int(module.in_features)
                method = "prac"
                device = module.linear.weight.device
            activation = _flatten_last_dim(activation_cpu, in_features)
            (sample_dim, feature_dim) = activation.shape
            if sample_dim <= 0 or feature_dim <= 0:
                continue
            work = activation.to(device=device, dtype=torch.float32)
            total_energy_t = (work * work).sum()
            total_energy = float(total_energy_t.item())
            if total_energy <= 0.0 or not math.isfinite(total_energy):
                continue
            q = min(max(rank + oversampling, rank), sample_dim, feature_dim)
            try:
                (_, singular_values, _) = torch.svd_lowrank(work, q=q, niter=niter)
            except RuntimeError:
                singular_values = torch.linalg.svdvals(work)
                q = min(q, singular_values.numel())
                singular_values = singular_values[:q]
            singular_values = torch.sort(singular_values.float(), descending=True).values
            available_rank = min(rank, singular_values.numel())
            top_energy = singular_values[:available_rank].pow(2).sum()
            top1_energy = (
                singular_values[0].pow(2)
                if singular_values.numel()
                else torch.tensor(float("nan"), device=device)
            )
            explained = (
                float((top_energy / total_energy_t).item()) if total_energy > 0 else float("nan")
            )
            tail = max(0.0, 1.0 - explained) if math.isfinite(explained) else float("nan")
            stable_rank = (
                float((total_energy_t / (top1_energy + 1e-12)).item())
                if singular_values.numel()
                else float("nan")
            )
            random_tail = max(0.0, 1.0 - float(rank) / float(feature_dim))
            rhs = (
                math.sqrt(tail / random_tail)
                if random_tail > 0 and math.isfinite(tail)
                else float("nan")
            )
            current_basis_explained = None
            basis_orthogonality_error = None
            random_orthogonal_energy_fraction = None
            if isinstance(module, STACRLinear):
                basis = (
                    module.effective_basis if getattr(module, "basis_initialized", False) else None
                )
                if basis is not None and basis.numel() > 0:
                    basis = basis[:, : min(rank, basis.shape[1])].to(
                        device=device, dtype=torch.float32
                    )
                    if basis.numel() > 0:
                        projected = work.mm(basis)
                        current_basis_explained = float(
                            (projected.pow(2).sum() / total_energy_t).item()
                        )
                        gram = basis.t().mm(basis)
                        eye = torch.eye(gram.shape[0], device=device, dtype=gram.dtype)
                        basis_orthogonality_error = float((gram - eye).norm().item())
            else:
                projection = module.projection.to(device=device, dtype=torch.float32)
                if projection.numel() > 0:
                    (q_basis, _) = torch.linalg.qr(projection[:, : min(rank, projection.shape[1])])
                    projected = work.mm(q_basis)
                    random_orthogonal_energy_fraction = float(
                        (projected.pow(2).sum() / total_energy_t).item()
                    )
            records.append(
                {
                    "layer_name": layer_name,
                    "method": method,
                    "rank": rank,
                    "sample_dim": int(sample_dim),
                    "feature_dim": int(feature_dim),
                    "svd_q": int(q),
                    "svd_niter": int(niter),
                    "centered": False,
                    "total_energy_fro_sq": total_energy,
                    "top_singular_values": [
                        float(x)
                        for x in singular_values[: min(16, singular_values.numel())]
                        .detach()
                        .cpu()
                        .tolist()
                    ],
                    "explained_variance_rank": explained,
                    "tail_energy_fraction_rank": tail,
                    "stable_rank": stable_rank,
                    "random_projection_tail_fraction": random_tail,
                    "learned_vs_random_rhs": rhs,
                    "current_basis_explained_variance": current_basis_explained,
                    "basis_orthogonality_error_fro": basis_orthogonality_error,
                    "random_orthogonal_energy_fraction": random_orthogonal_energy_fraction,
                }
            )
            del work
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if was_training:
        model.train()
    return records


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    config: ConvergenceConfig,
    epoch: int,
    logger: Any,
    amp_manager: Optional[Any] = None,
    gact_controller: Optional[Any] = None,
    offloader: Optional[SavedTensorOffloader] = None,
    compressor: Optional[SavedTensorCompressor] = None,
    accum_steps: int = 1,
    grad_clip: float = 0.0,
    global_step_offset: int = 0,
) -> Dict[str, float]:
    """Train for one epoch.

    Args:
        global_step_offset: Total steps completed in previous epochs (for max_steps check).

    Returns:
        Dict with train metrics including 'steps_completed' for tracking total steps.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    total_tokens = 0
    processed_batches = 0
    accum_steps = max(1, int(accum_steps))
    optimizer.zero_grad(set_to_none=True)
    epoch_start = time.time()
    grad_fidelity_values: List[float] = []
    per_layer_grad_fidelity_records: List[Dict[str, Any]] = []
    svd_per_layer_grad_fidelity_records: List[Dict[str, Any]] = []
    activation_spectrum_records: List[Dict[str, Any]] = []
    is_stacr = config.method in ("stacr", "stacr-v4") or config.stacr_attention
    is_grad_fidelity_method = (
        is_stacr or config.method == "prac" or config.method in ("svd", "tt", "tucker")
    )
    uses_stacr_step_counter = is_stacr or getattr(config, "stacr_compress_checkpoint", False)
    if config.method == "prac" and (not hasattr(config, "_prac_optimizer_steps_completed")):
        config._prac_optimizer_steps_completed = 0
    if uses_stacr_step_counter and (not hasattr(config, "_stacr_optimizer_steps_completed")):
        config._stacr_optimizer_steps_completed = 0
    phase_profiler = _get_phase_profiler(config)
    for step, batch in enumerate(train_loader):
        global_step = global_step_offset + step
        if config.max_steps is not None and global_step >= config.max_steps:
            break
        debug_this_step = (
            getattr(config, "debug_step_metrics_every", 0) > 0
            and (global_step + 1) % int(config.debug_step_metrics_every) == 0
            and (not config.use_ddp or int(os.environ.get("RANK", 0)) == 0)
        )
        debug_memory_phases: Dict[str, Optional[int]] = {}
        if config.method == "prac":
            from stacr.protocols.prac import set_prac_global_step

            opt_step = int(getattr(config, "_prac_optimizer_steps_completed", 0))
            set_prac_global_step(model, opt_step)
        if is_stacr:
            opt_step = int(getattr(config, "_stacr_optimizer_steps_completed", 0))
            set_stacr_runtime_state(
                model,
                optimizer_step=opt_step,
                microbatch_index=step % accum_steps,
                accum_steps=accum_steps,
            )
        phase_step_cm = None
        if phase_profiler is not None:
            phase_step_cm = phase_profiler.step(global_step)
            phase_step_cm.__enter__()
        try:
            if config.task in ("sst2", "wikitext103"):
                inputs = {k: v.to(device) for (k, v) in batch.items()}
                if "attention_mask" in inputs:
                    total_tokens += int(inputs["attention_mask"].sum().item())
                elif "input_ids" in inputs:
                    total_tokens += inputs["input_ids"].numel()
            else:
                (inputs, labels) = batch
                (inputs, labels) = (inputs.to(device), labels.to(device))
            if config.measure_memory and step == 0:
                reset_peaks()
            if debug_this_step and config.measure_memory:
                debug_memory_phases["before_forward"] = _current_allocated_bytes(device)
            offload_ctx = offloader.context() if offloader else nullcontext()
            compress_ctx = compressor.context() if compressor else nullcontext()
            with compress_ctx:
                with offload_ctx:
                    with _phase_profile_context(config, "forward"):
                        if amp_manager:
                            with amp_manager.autocast():
                                if config.task in ("sst2", "wikitext103"):
                                    outputs = model(**inputs)
                                    loss = outputs.loss
                                    logits = outputs.logits
                                    labels = inputs["labels"]
                                else:
                                    logits = model(inputs)
                                    loss = criterion(logits, labels)
                        elif config.task in ("sst2", "wikitext103"):
                            outputs = model(**inputs)
                            loss = outputs.loss
                            logits = outputs.logits
                            labels = inputs["labels"]
                        else:
                            logits = model(inputs)
                            loss = criterion(logits, labels)
                    if loss.dim() > 0:
                        loss = loss.mean()
                    loss_value = loss.item()
                    if not math.isfinite(loss_value):
                        raise FloatingPointError(f"Nonfinite training loss at batch {global_step}")
                    if debug_this_step and config.measure_memory:
                        debug_memory_phases["after_forward"] = _current_allocated_bytes(device)
                    with _phase_profile_context(config, "backward"):
                        loss = loss / accum_steps
                        if amp_manager:
                            amp_manager.backward(loss)
                        else:
                            loss.backward()
                    if debug_this_step and config.measure_memory:
                        debug_memory_phases["after_backward"] = _current_allocated_bytes(device)
            do_step = (step + 1) % accum_steps == 0
            if do_step:
                with _phase_profile_context(config, "optimizer"):
                    if config.use_tensor_parallel:
                        from stacr.protocols.tensor_parallel import sync_replicated_grads

                        sync_replicated_grads(model, None)
                    if amp_manager and amp_manager.scaler is not None:
                        amp_manager.scaler.unscale_(optimizer)
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    if amp_manager:
                        amp_manager.step(optimizer)
                    else:
                        optimizer.step()
                    if config.method == "prac":
                        config._prac_optimizer_steps_completed = (
                            int(getattr(config, "_prac_optimizer_steps_completed", 0)) + 1
                        )
                    if uses_stacr_step_counter:
                        config._stacr_optimizer_steps_completed = (
                            int(getattr(config, "_stacr_optimizer_steps_completed", 0)) + 1
                        )
                    optimizer.zero_grad(set_to_none=True)
                    if gact_controller and hasattr(gact_controller, "iterate"):
                        if getattr(gact_controller, "auto_precision", False):

                            def _gact_replay_backprop_for_autoprec() -> None:
                                optimizer.zero_grad(set_to_none=True)
                                replay_offload_ctx = (
                                    offloader.context() if offloader else nullcontext()
                                )
                                replay_compress_ctx = (
                                    compressor.context() if compressor else nullcontext()
                                )
                                with replay_compress_ctx:
                                    with replay_offload_ctx:
                                        if amp_manager:
                                            with amp_manager.autocast():
                                                if config.task in ("sst2", "wikitext103"):
                                                    replay_outputs = model(**inputs)
                                                    replay_loss = replay_outputs.loss
                                                else:
                                                    replay_logits = model(inputs)
                                                    replay_loss = criterion(replay_logits, labels)
                                        elif config.task in ("sst2", "wikitext103"):
                                            replay_outputs = model(**inputs)
                                            replay_loss = replay_outputs.loss
                                        else:
                                            replay_logits = model(inputs)
                                            replay_loss = criterion(replay_logits, labels)
                                        if replay_loss.dim() > 0:
                                            replay_loss = replay_loss.mean()
                                        if amp_manager:
                                            amp_manager.backward(replay_loss)
                                        else:
                                            replay_loss.backward()

                            try:
                                gact_controller.iterate(_gact_replay_backprop_for_autoprec)
                            finally:
                                optimizer.zero_grad(set_to_none=True)
                        else:
                            gact_controller.iterate()
                    if compressor is not None and hasattr(compressor, "step"):
                        compressor.step()
                    if config.task in ("sst2", "wikitext103") and scheduler:
                        scheduler.step()
                if config.save_every_n_steps and (global_step + 1) % config.save_every_n_steps == 0:
                    _is_main_rank = (
                        not config.use_ddp
                        and (not config.use_tensor_parallel)
                        or int(os.environ.get("LOCAL_RANK", "0")) == 0
                    )
                    if _is_main_rank:
                        ckpt_dir = Path(logger.run_dir) / f"checkpoint_step_{global_step + 1}"
                        ckpt_dir.mkdir(parents=True, exist_ok=True)
                        save_target = model.module if hasattr(model, "module") else model
                        torch.save(save_target.state_dict(), ckpt_dir / "model.pt")
                        if not config.quiet:
                            print(
                                f"[step {global_step + 1}] Saved mid-training checkpoint to {ckpt_dir}"
                            )
            if debug_this_step and config.measure_memory:
                debug_memory_phases["after_optimizer_step" if do_step else "after_microbatch"] = (
                    _current_allocated_bytes(device)
                )
        finally:
            if phase_step_cm is not None:
                phase_step_cm.__exit__(*sys.exc_info())
                phase_step_cm = None
        if (
            config.monitor_grad_fidelity
            and is_grad_fidelity_method
            and (int(config.grad_fidelity_interval) > 0)
            and (global_step % int(config.grad_fidelity_interval) == 0)
        ):
            if accum_steps > 1 and (not do_step):
                pass
            else:
                if config.task == "sst2":
                    fidelity_inputs = inputs
                    fidelity_labels = inputs["labels"]
                else:
                    fidelity_inputs = inputs
                    fidelity_labels = labels
                grad_fidelity = compute_grad_cosine(
                    model, fidelity_inputs, fidelity_labels, criterion, config.task, amp_manager
                )
                grad_fidelity_values.append(grad_fidelity["grad_cosine_mean"])
                if not config.use_ddp or int(os.environ.get("RANK", 0)) == 0:
                    logger.log_metrics(
                        {
                            "epoch": epoch,
                            "step": step,
                            "global_step": global_step,
                            "grad_cosine": grad_fidelity["grad_cosine_mean"],
                            "grad_cosine_min": grad_fidelity["grad_cosine_min"],
                        }
                    )
                optimizer.zero_grad(set_to_none=True)
        if (
            config.monitor_per_layer_grad_fidelity
            and is_grad_fidelity_method
            and (int(config.per_layer_grad_fidelity_interval) > 0)
            and ((global_step + 1) % int(config.per_layer_grad_fidelity_interval) == 0)
        ):
            if accum_steps > 1 and (not do_step):
                pass
            else:
                if config.task == "sst2":
                    fidelity_inputs = inputs
                    fidelity_labels = inputs["labels"]
                else:
                    fidelity_inputs = inputs
                    fidelity_labels = labels
                per_layer_records = compute_per_layer_grad_fidelity(
                    model, fidelity_inputs, fidelity_labels, criterion, config.task, amp_manager
                )
                for record in per_layer_records:
                    record["epoch"] = int(epoch)
                    record["step"] = int(step)
                    record["global_step"] = int(global_step)
                    record["optimizer_step"] = int(
                        getattr(config, "_stacr_optimizer_steps_completed", 0)
                        if is_stacr
                        else getattr(config, "_prac_optimizer_steps_completed", 0)
                    )
                per_layer_grad_fidelity_records.extend(per_layer_records)
                per_layer_summary = _summarize_per_layer_grad_fidelity(per_layer_records)
                if not config.use_ddp or int(os.environ.get("RANK", 0)) == 0:
                    per_layer_path = Path(logger.run_dir) / "per_layer_grad_fidelity.jsonl"
                    with open(per_layer_path, "a", encoding="utf-8") as f:
                        for record in per_layer_records:
                            f.write(json.dumps(record, sort_keys=True) + "\n")
                    logger.log_metrics(
                        {
                            "epoch": epoch,
                            "step": step,
                            "global_step": global_step,
                            "record_type": "per_layer_grad_fidelity_summary",
                            "per_layer_grad_fidelity_summary": per_layer_summary,
                        }
                    )
                optimizer.zero_grad(set_to_none=True)
        if (
            config.monitor_svd_per_layer_grad_fidelity
            and is_grad_fidelity_method
            and (int(config.svd_per_layer_grad_fidelity_interval) > 0)
            and ((global_step + 1) % int(config.svd_per_layer_grad_fidelity_interval) == 0)
        ):
            if accum_steps > 1 and (not do_step):
                pass
            else:
                if config.task == "sst2":
                    fidelity_inputs = inputs
                    fidelity_labels = inputs["labels"]
                else:
                    fidelity_inputs = inputs
                    fidelity_labels = labels
                svd_records = compute_per_layer_grad_fidelity(
                    model,
                    fidelity_inputs,
                    fidelity_labels,
                    criterion,
                    config.task,
                    amp_manager,
                    reconstruction_override="svd_lowrank",
                    svd_rank=int(getattr(config, "svd_per_layer_rank", None) or config.rank),
                    svd_oversampling=int(getattr(config, "svd_per_layer_oversampling", 16)),
                    svd_niter=int(getattr(config, "svd_per_layer_niter", 2)),
                )
                for record in svd_records:
                    record["epoch"] = int(epoch)
                    record["step"] = int(step)
                    record["global_step"] = int(global_step)
                    record["optimizer_step"] = int(
                        getattr(config, "_stacr_optimizer_steps_completed", 0)
                        if is_stacr
                        else getattr(config, "_prac_optimizer_steps_completed", 0)
                    )
                svd_per_layer_grad_fidelity_records.extend(svd_records)
                svd_summary = _summarize_per_layer_grad_fidelity(svd_records)
                if not config.use_ddp or int(os.environ.get("RANK", 0)) == 0:
                    svd_path = Path(logger.run_dir) / "svd_per_layer_grad_fidelity.jsonl"
                    with open(svd_path, "a", encoding="utf-8") as f:
                        for record in svd_records:
                            f.write(json.dumps(record, sort_keys=True) + "\n")
                    logger.log_metrics(
                        {
                            "epoch": epoch,
                            "step": step,
                            "global_step": global_step,
                            "record_type": "svd_per_layer_grad_fidelity_summary",
                            "svd_per_layer_grad_fidelity_summary": svd_summary,
                        }
                    )
                optimizer.zero_grad(set_to_none=True)
        if (
            config.monitor_activation_spectrum
            and is_grad_fidelity_method
            and (int(config.activation_spectrum_interval) > 0)
            and ((global_step + 1) % int(config.activation_spectrum_interval) == 0)
        ):
            if accum_steps > 1 and (not do_step):
                pass
            else:
                spectrum_records = compute_activation_spectrum_telemetry(
                    model,
                    inputs,
                    config.task,
                    rank=int(getattr(config, "activation_spectrum_rank", config.rank)),
                    oversampling=int(getattr(config, "activation_spectrum_oversampling", 16)),
                    niter=int(getattr(config, "activation_spectrum_niter", 2)),
                )
                for record in spectrum_records:
                    record["epoch"] = int(epoch)
                    record["step"] = int(step)
                    record["global_step"] = int(global_step)
                    record["optimizer_step"] = int(
                        getattr(config, "_stacr_optimizer_steps_completed", 0)
                        if is_stacr
                        else getattr(config, "_prac_optimizer_steps_completed", 0)
                    )
                activation_spectrum_records.extend(spectrum_records)
                spectrum_summary = _summarize_activation_spectrum(spectrum_records)
                if not config.use_ddp or int(os.environ.get("RANK", 0)) == 0:
                    spectrum_path = Path(logger.run_dir) / "activation_spectrum.jsonl"
                    with open(spectrum_path, "a", encoding="utf-8") as f:
                        for record in spectrum_records:
                            f.write(json.dumps(record, sort_keys=True) + "\n")
                    logger.log_metrics(
                        {
                            "epoch": epoch,
                            "step": step,
                            "global_step": global_step,
                            "record_type": "activation_spectrum_summary",
                            "activation_spectrum_summary": spectrum_summary,
                        }
                    )
        processed_batches += 1
        total_loss += loss_value
        if config.task == "wikitext103":
            shift_preds = logits[:, :-1, :].argmax(dim=-1)
            shift_labels = labels[:, 1:]
            correct += (shift_preds == shift_labels).sum().item()
            total += shift_labels.numel()
        else:
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        runtime = get_active_c1_runtime()
        if runtime is not None:
            running_loss = total_loss / max(step + 1, 1)
            step_perplexity = None
            running_perplexity = None
            if config.task == "wikitext103":
                step_perplexity = _safe_perplexity_from_loss(loss_value)
                running_perplexity = _safe_perplexity_from_loss(running_loss)
            _summarize_c1_first_finite_drift_event(
                config,
                runtime.stats(),
                global_step=global_step,
                optimizer_step=int(getattr(config, "_stacr_optimizer_steps_completed", 0))
                if uses_stacr_step_counter
                else int(getattr(config, "_prac_optimizer_steps_completed", 0))
                if config.method == "prac"
                else global_step + 1,
                loss_value=loss_value,
                running_loss=running_loss,
                step_perplexity=step_perplexity,
                running_perplexity=running_perplexity,
            )
        if debug_this_step:
            running_accuracy = correct / total * 100 if total else 0.0
            running_loss = total_loss / max(step + 1, 1)
            logger.log_metrics(
                _build_debug_step_metrics_payload(
                    config,
                    optimizer,
                    epoch=epoch,
                    step=step,
                    global_step=global_step,
                    loss_value=loss_value,
                    running_accuracy=running_accuracy,
                    running_loss=running_loss,
                    memory_phases=debug_memory_phases,
                )
            )
        if step % 100 == 0 and (not config.use_ddp or int(os.environ.get("RANK", 0)) == 0):
            logger.log_metrics(
                {
                    "epoch": epoch,
                    "step": step,
                    "loss": loss_value,
                    "accuracy": correct / total * 100 if total else 0.0,
                }
            )
        if config.measure_memory and step == 0:
            log_memory_complete(logger, step)
    if processed_batches % accum_steps:
        # Scale a partial group by the configured accumulation count.
        # Flush its gradients even when max_steps stops before loader exhaustion.
        if config.use_tensor_parallel:
            from stacr.protocols.tensor_parallel import sync_replicated_grads

            sync_replicated_grads(model, None)
        if amp_manager and amp_manager.scaler is not None:
            amp_manager.scaler.unscale_(optimizer)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if amp_manager:
            amp_manager.step(optimizer)
        else:
            optimizer.step()
        if config.method == "prac":
            config._prac_optimizer_steps_completed = (
                int(getattr(config, "_prac_optimizer_steps_completed", 0)) + 1
            )
        if uses_stacr_step_counter:
            config._stacr_optimizer_steps_completed = (
                int(getattr(config, "_stacr_optimizer_steps_completed", 0)) + 1
            )
        optimizer.zero_grad(set_to_none=True)
        if gact_controller and hasattr(gact_controller, "iterate"):
            if getattr(gact_controller, "auto_precision", False):

                def _gact_replay_backprop_for_autoprec() -> None:
                    optimizer.zero_grad(set_to_none=True)
                    if amp_manager:
                        with amp_manager.autocast():
                            if config.task in ("sst2", "wikitext103"):
                                replay_outputs = model(**inputs)
                                replay_loss = replay_outputs.loss
                            else:
                                replay_logits = model(inputs)
                                replay_loss = criterion(replay_logits, labels)
                    elif config.task in ("sst2", "wikitext103"):
                        replay_outputs = model(**inputs)
                        replay_loss = replay_outputs.loss
                    else:
                        replay_logits = model(inputs)
                        replay_loss = criterion(replay_logits, labels)
                    if replay_loss.dim() > 0:
                        replay_loss = replay_loss.mean()
                    if amp_manager:
                        amp_manager.backward(replay_loss)
                    else:
                        replay_loss.backward()

                try:
                    gact_controller.iterate(_gact_replay_backprop_for_autoprec)
                finally:
                    optimizer.zero_grad(set_to_none=True)
            else:
                gact_controller.iterate()
        if compressor is not None and hasattr(compressor, "step"):
            compressor.step()
        if config.task in ("sst2", "wikitext103") and scheduler:
            scheduler.step()
    if config.task == "cifar10" and scheduler:
        scheduler.step()
    if processed_batches == 0:
        warnings.warn("Empty dataloader - no batches processed in epoch")
        throughput_key = (
            "tokens_per_s" if config.task in ("sst2", "wikitext103") else "samples_per_s"
        )
        return {
            "train_loss": float("nan"),
            "train_accuracy": 0.0,
            "epoch_time_s": 0.0,
            throughput_key: 0.0,
            "grad_fidelity_values": [],
            "per_layer_grad_fidelity_records": [],
            "svd_per_layer_grad_fidelity_records": [],
            "activation_spectrum_records": [],
            "steps_completed": 0,
        }
    steps_this_epoch = processed_batches
    epoch_time_s = max(time.time() - epoch_start, 1e-06)
    throughput = (total_tokens if config.task in ("sst2", "wikitext103") else total) / epoch_time_s
    throughput_key = "tokens_per_s" if config.task in ("sst2", "wikitext103") else "samples_per_s"
    return {
        "train_loss": total_loss / max(steps_this_epoch, 1),
        "train_accuracy": correct / max(total, 1) * 100,
        "epoch_time_s": epoch_time_s,
        throughput_key: throughput,
        "grad_fidelity_values": grad_fidelity_values,
        "per_layer_grad_fidelity_records": per_layer_grad_fidelity_records,
        "svd_per_layer_grad_fidelity_records": svd_per_layer_grad_fidelity_records,
        "activation_spectrum_records": activation_spectrum_records,
        "steps_completed": steps_this_epoch,
        "_raw_total_loss": total_loss,
        "_raw_correct": correct,
        "_raw_total": total,
        "_raw_total_tokens": total_tokens,
        "_raw_steps": steps_this_epoch,
    }


def run_convergence_experiment(config: ConvergenceConfig, logger: Any) -> Dict[str, Any]:
    """Main entry point for convergence training."""
    from stacr.adaptive import get_basis_pool

    # Independent runs in one process must not inherit fitted bases.
    get_basis_pool().clear()
    set_all_seeds(config.seed, strict=bool(getattr(config, "deterministic", False)))
    _normalize_method_config(config)
    if config.method == "prac":
        config._prac_optimizer_steps_completed = 0
    if (
        config.method in ("stacr", "stacr-v4")
        or config.stacr_attention
        or getattr(config, "stacr_compress_checkpoint", False)
    ):
        config._stacr_optimizer_steps_completed = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    ddp_is_main = True
    ddp_train_sampler = None
    (tp_local_rank, tp_world_size, tp_group) = (0, 1, None)
    if config.use_ddp:
        ddp_local_rank = int(os.environ.get("LOCAL_RANK", 0))
        ddp_world_size = int(os.environ.get("WORLD_SIZE", 1))
        ddp_is_main = int(os.environ.get("RANK", 0)) == 0
        device = torch.device(f"cuda:{ddp_local_rank}")
        config.device = str(device)
        if ddp_is_main:
            print(f"[DDP] world_size={ddp_world_size}, device={device}")
    elif config.use_tensor_parallel:
        tp_local_rank = int(os.environ.get("LOCAL_RANK", 0))
        tp_world_size = int(os.environ.get("WORLD_SIZE", 1))
        tp_is_main = int(os.environ.get("RANK", 0)) == 0
        tp_group = None
        device = torch.device(f"cuda:{tp_local_rank}")
        config.device = str(device)
        if tp_is_main:
            print(f"[TP] world_size={tp_world_size}, scope={config.tp_scope}, device={device}")
    else:
        device = torch.device(config.device)

    def _update_max(current: Optional[int], value: Optional[int]) -> Optional[int]:
        if value is None:
            return current
        if current is None:
            return value
        return max(current, value)

    gact_controller = None
    max_allocated = None
    max_reserved = None
    max_rss = None
    compressor = None
    throughput_values: List[float] = []
    all_grad_fidelity_values: List[float] = []
    all_per_layer_grad_fidelity_records: List[Dict[str, Any]] = []
    all_svd_per_layer_grad_fidelity_records: List[Dict[str, Any]] = []
    all_activation_spectrum_records: List[Dict[str, Any]] = []
    storage_estimate = None
    activation_profile = None
    effective_total_steps: Optional[int] = getattr(config, "max_steps", None)
    planner_layer_decisions = None
    planner_layer_decision_summary = None
    baseline_state: Optional[BaselineState] = None
    try:
        if config.task == "sst2":
            model = create_bert_sst2(config.model)
            (train_loader, val_loader, _) = setup_sst2_data(config)
            effective_total_steps = _resolve_effective_total_steps(config, len(train_loader))
            config.effective_total_steps = effective_total_steps
            criterion = None
            if config.method not in ("baseline",):
                targets = get_compression_targets_bert(model)
                targets = _exclude_checkpointed_decoder_targets_for_c1(model, config, targets)
                if config.method in ("stacr", "stacr-v4"):
                    model = replace_linear_with_stacr(model, config, targets)
                    storage_estimate = estimate_storage_for_stacr(model, config)
                elif config.method == "prac":
                    from stacr.protocols.prac import PRACConfig, replace_linear_with_prac

                    prac_cfg = PRACConfig(
                        rank=config.rank,
                        update_proj_gap=int(config.prac_update_proj_gap),
                        seed_base=int(config.prac_seed_base),
                        proj_dtype=_resolve_runtime_prac_proj_dtype(config),
                        quiet=config.quiet,
                    )
                    replace_linear_with_prac(model, prac_cfg, target_layers=targets)
                    model.to(device)
                    storage_estimate = estimate_storage_for_prac(model, config)
                else:
                    model = replace_linear_with_compressed(model, config, targets)
                    storage_estimate = estimate_storage_for_model(model, config)
        elif config.task == "wikitext103":
            try:
                model_name = config.model or "gpt2"
                model = create_gpt2_lm(model_name)
                model.config.use_cache = False
                param_count = sum((p.numel() for p in model.parameters()))
                if not config.quiet:
                    print(f"Using {model_name} for WikiText-103 ({param_count:,} params)")
                (train_loader, val_loader, tokenizer) = setup_wikitext_data(config)
                effective_total_steps = _resolve_effective_total_steps(config, len(train_loader))
                config.effective_total_steps = effective_total_steps
                model.config.pad_token_id = tokenizer.pad_token_id
            except ImportError as e:
                raise RuntimeError("Language modeling requires the train dependencies") from e
            criterion = None
            if config.use_tensor_parallel:
                from stacr.protocols.tp_gpt2 import (
                    apply_tensor_parallel_gpt2,
                    disable_tp_dropout,
                    get_tp_stacr_targets,
                    TPActivationProfiler,
                )

                model = model.to(device)
                apply_tensor_parallel_gpt2(
                    model, tp_local_rank, tp_world_size, tp_group, scope=config.tp_scope
                )
                n_disabled = disable_tp_dropout(model)
                if not config.quiet:
                    print(f"[TP] Disabled {n_disabled} Dropout modules (TP mode)")
            if config.method not in ("baseline",):
                if config.use_tensor_parallel and getattr(config, "stacr_use_planner", False):
                    raise NotImplementedError(
                        "Phase 3 planner v0 is not supported with tensor parallel. Disable --tp or leave --stacr-use-planner off."
                    )
                if config.use_tensor_parallel:
                    targets = get_tp_stacr_targets(model, config.tp_stacr_scope)
                else:
                    targets = get_compression_targets_gpt2(
                        model, getattr(config, "stacr_decoder_target_mode", "all")
                    )
                    targets = _exclude_checkpointed_decoder_targets_for_c1(model, config, targets)
                    (targets, planner_layer_decisions, planner_layer_decision_summary) = (
                        _maybe_plan_decoder_targets(model, config, targets)
                    )
                if not config.quiet:
                    print(
                        f"WikiText-103 compression targets: {len(targets)} layers (decoder_target_mode={getattr(config, 'stacr_decoder_target_mode', 'all')})"
                    )
                    if planner_layer_decision_summary is not None:
                        print(
                            f"Planner summary: kept={planner_layer_decision_summary['kept_count']} skipped={planner_layer_decision_summary['skipped_count']} forced={planner_layer_decision_summary['forced_count']} enforced={planner_layer_decision_summary['planner_enforced']}"
                        )
                if config.method in ("stacr", "stacr-v4"):
                    model = replace_linear_with_stacr(model, config, targets)
                    model.to(device)
                    storage_estimate = estimate_storage_for_stacr(model, config)
                elif config.method == "prac":
                    from stacr.protocols.prac import PRACConfig, replace_linear_with_prac

                    prac_cfg = PRACConfig(
                        rank=config.rank,
                        update_proj_gap=int(config.prac_update_proj_gap),
                        seed_base=int(config.prac_seed_base),
                        proj_dtype=_resolve_runtime_prac_proj_dtype(config),
                        quiet=config.quiet,
                    )
                    replace_linear_with_prac(model, prac_cfg, target_layers=targets)
                    model.to(device)
                    storage_estimate = estimate_storage_for_prac(model, config)
                elif config.method in ("svd", "tt", "tucker"):
                    model = replace_linear_with_compressed(model, config, targets)
                    model.to(device)
                    storage_estimate = estimate_storage_for_model(model, config)
                else:
                    raise NotImplementedError(
                        f"Method '{config.method}' not supported for GPT-2 Conv1D layers. Use --method svd/tt/tucker, --method stacr, --method prac, or --method baseline for WikiText-103."
                    )
            if config.use_tensor_parallel and config.tp_profile_activations:
                profiler = TPActivationProfiler(model)
                model.eval()
                profile_loader = DataLoader(
                    train_loader.dataset,
                    batch_size=config.batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=getattr(train_loader, "pin_memory", False),
                    collate_fn=getattr(train_loader, "collate_fn", None),
                    drop_last=False,
                )
                profile_batch = next(iter(profile_loader))
                profile_inputs = {k: v.to(device) for (k, v) in profile_batch.items()}
                with torch.no_grad(), profiler:
                    model(**profile_inputs)
                activation_profile = profiler.summary()
                if tp_local_rank == 0 and (not config.quiet):
                    profiler.print_summary(rank=tp_local_rank)
                model.train()
        else:
            raise ValueError(f"Unknown task: {config.task}")
        stacr_attention_report = None
        if config.stacr_attention:
            attn_rank = getattr(config, "stacr_attention_rank", None) or config.rank
            attn_residual_mode = (
                getattr(config, "stacr_attention_residual_mode", None) or config.stacr_residual_mode
            )
            stacr_config = STACRConfig(
                rank=attn_rank,
                residual_mode=attn_residual_mode,
                residual_budget=config.stacr_residual_budget,
                residual_threshold=config.stacr_residual_threshold,
                update_trigger=config.stacr_update_trigger,
                update_interval=config.stacr_update_interval,
                energy_threshold=config.stacr_energy_threshold,
                streaming_algorithm=config.stacr_streaming_algorithm,
                oja_learning_rate=config.stacr_oja_learning_rate,
                warmup_steps=config.stacr_warmup_steps,
                freeze_basis_after_warmup=config.stacr_freeze_basis_after_warmup,
                quantize_coefficients=config.stacr_quantize_coefficients,
                quantize_bits=config.stacr_quantize_bits,
                quantize_residuals=config.stacr_quantize_residuals,
                residual_quantize_bits=config.stacr_residual_quantize_bits,
                basis_dtype=_resolve_runtime_stacr_basis_dtype(config),
                backward_mode=getattr(config, "stacr_backward_mode", "sparse"),
                use_mgs=config.stacr_use_mgs,
                lazy_update_enabled=config.stacr_lazy_update,
                lazy_update_threshold=config.stacr_lazy_update_threshold,
                use_fused_kernel=config.stacr_use_fused_kernel,
                cpu_offload=getattr(config, "stacr_cpu_offload", False),
                use_random_qr_init=getattr(config, "stacr_use_random_qr_init", False),
                min_adaptation_steps=getattr(config, "stacr_min_adaptation_steps", 50),
                allow_unadapted_random_qr=getattr(config, "stacr_allow_unadapted_random_qr", False),
                rank_mode=config.stacr_rank_mode,
                rank_variance_threshold=config.stacr_rank_variance_threshold,
                rank_min=config.stacr_rank_min,
                rank_max=config.stacr_rank_max,
                basis_sharing=config.stacr_basis_sharing,
                streaming_topk=config.stacr_streaming_topk,
                streaming_topk_sample_ratio=config.stacr_streaming_topk_sample_ratio,
                index_compression=config.stacr_index_compression,
                skip_residual_enabled=config.stacr_skip_residual,
                skip_residual_energy_threshold=config.stacr_skip_residual_threshold,
                buffer_reuse=config.stacr_buffer_reuse,
                adaptive_layer_rank=getattr(config, "stacr_adaptive_layer_rank", "none"),
                residual_schedule=getattr(config, "stacr_residual_schedule", "none"),
                residual_start=getattr(config, "stacr_residual_start", 0.05),
                residual_end=getattr(config, "stacr_residual_end", 0.001),
                chunk_size=config.stacr_chunk_size,
                post_warmup_residual_mode=getattr(config, "stacr_post_warmup_mode", "full"),
                residual_warmup_steps=getattr(config, "stacr_residual_warmup_steps", 0),
                post_warmup_residual_threshold=getattr(config, "stacr_post_warmup_threshold", 0.1),
            )
            stacr_config = apply_min_adaptation_rule(
                stacr_config,
                effective_total_steps,
                log_fn=None if getattr(config, "quiet", False) else print,
            )
            stacr_attention_report = apply_stacr_to_attention(
                model,
                stacr_config,
                enable_gpt2_c_attn_split=config.stacr_gpt2_split_c_attn,
                mask_aware=True,
                joint_qk=config.stacr_joint_qk,
            )
            if not config.quiet:
                print(
                    f"STAC-R Attention: replaced {stacr_attention_report.replaced_qkv} Q/K/V projections"
                )
                if stacr_attention_report.joint_qk_pairs > 0:
                    print(f"  {stacr_attention_report.joint_qk_pairs} Q/K pairs use a shared basis")
                if stacr_attention_report.warnings:
                    for w in stacr_attention_report.warnings:
                        print(f"  Warning: {w}")
        (model, baseline_state) = apply_baseline(model, config, device)
        if baseline_state.status != "enabled":
            return {
                "status": baseline_state.status,
                "note": "baseline_unavailable",
                "error": baseline_state.reason,
            }
        if getattr(config, "stacr_compress_checkpoint", False) and storage_estimate is None:
            storage_estimate = estimate_storage_for_c1(model, config)
        if getattr(config, "phase_profile_full_run", False):
            phase_profiler = PhaseProfiler(
                device=str(device), out_path=str(logger.profile_path), enabled=True
            )
            attached_layers = _attach_phase_profiler_to_stacr_layers(model, phase_profiler)
            setattr(config, "_phase_profiler", phase_profiler)
            phase_profiler.write_metadata(
                extra={
                    "experiment": "convergence_training",
                    "phase_profile_full_run": True,
                    "task": config.task,
                    "model": config.model,
                    "method": config.method,
                    "method_label": config.method_label,
                    "rank": config.rank,
                    "stacr_layers_attached": attached_layers,
                    "note": "per-step forward/backward/optimizer plus STAC-R layer subphases",
                }
            )
            if not config.quiet:
                print(
                    f"[PhaseProfiler] enabled full-run trace; attached_stacr_layers={attached_layers}"
                )
        if config.use_ddp:
            model = wrap_model_ddp(model, ddp_local_rank)
            if ddp_is_main:
                print(f"[DDP] Model wrapped with DistributedDataParallel")
            if train_loader is not None:
                collate_fn = getattr(train_loader, "collate_fn", None)
                (train_loader, ddp_train_sampler) = make_distributed_loader(
                    train_loader.dataset,
                    config.batch_size,
                    config.seed,
                    num_workers=config.num_workers,
                    collate_fn=collate_fn,
                )
        gact_controller = baseline_state.gact_controller
        amp_manager = baseline_state.amp_manager
        offloader = baseline_state.offloader
        accum_steps = config.baseline_cfg.grad_accum_steps
        galore_config_str = config.baseline_cfg.galore
        galore_optimizer = None
        galore_result = None
        galore_targets = _parse_galore_target_modules(config.baseline_cfg.galore_target_modules)
        if galore_config_str and galore_config_str.lower() != "none":
            galore_config = parse_galore_config_string(galore_config_str)
            if galore_config is None:
                raise ValueError("Invalid GaLore configuration")
            (galore_optimizer, galore_result) = try_enable_galore(
                model=model,
                lr=config.lr,
                weight_decay=config.weight_decay,
                rank=galore_config.rank,
                update_proj_gap=galore_config.update_proj_gap,
                scale=galore_config.scale,
                proj_type=galore_config.proj_type,
                target_modules=galore_targets,
            )
            if galore_result.ok:
                logger.log_metrics({"galore_status": "enabled", "galore_meta": galore_result.meta})
            else:
                raise RuntimeError(
                    f"GaLore was requested but could not start: {galore_result.reason}"
                )
        from transformers import get_linear_schedule_with_warmup

        if galore_optimizer is not None and galore_result.ok:
            optimizer = galore_optimizer
        elif getattr(config, "use_8bit_optimizer", False):
            from stacr.protocols.baselines.bnb_optimizer import try_create_8bit_optimizer

            bnb_result = try_create_8bit_optimizer(
                model.parameters(), lr=config.lr, weight_decay=config.weight_decay
            )
            if bnb_result.ok:
                optimizer = bnb_result.optimizer
                logger.log_metrics(
                    {
                        "8bit_optimizer": "enabled",
                        "8bit_estimated_savings_mb": bnb_result.meta.get("estimated_savings_mb", 0),
                    }
                )
            else:
                raise RuntimeError(
                    f"8-bit AdamW was requested but could not start: {bnb_result.reason}"
                )
        else:
            optimizer = optim.AdamW(
                model.parameters(), lr=config.lr, weight_decay=config.weight_decay
            )
        accum_steps = config.baseline_cfg.grad_accum_steps if config.baseline_cfg else 1
        batches_per_epoch = len(train_loader)
        optimizer_steps_per_epoch = (batches_per_epoch + accum_steps - 1) // accum_steps
        num_training_steps = optimizer_steps_per_epoch * config.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * num_training_steps),
            num_training_steps=num_training_steps,
        )
        best_accuracy = 0.0
        best_perplexity = float("inf")
        start_time = time.time()
        epochs_completed = 0
        total_steps_completed = 0
        total_optimizer_attempts = 0
        if config.method in ("stacr", "stacr-v4") or config.stacr_attention:
            if train_loader is not None:
                total_training_steps = len(train_loader) * config.epochs
            else:
                total_training_steps = None
            for module in model.modules():
                if hasattr(module, "layer_state") and isinstance(module.layer_state, dict):
                    if total_training_steps is not None:
                        module.layer_state["total_steps"] = total_training_steps
        for epoch in range(config.epochs):
            if config.use_ddp and ddp_train_sampler is not None:
                ddp_train_sampler.set_epoch(epoch)
            if config.measure_memory and config.task != "ogbn-arxiv":
                reset_peaks()
            train_metrics = train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scheduler,
                device,
                config,
                epoch,
                logger,
                amp_manager=amp_manager,
                gact_controller=gact_controller,
                offloader=offloader,
                compressor=compressor,
                accum_steps=accum_steps,
                grad_clip=config.grad_clip,
                global_step_offset=total_steps_completed,
            )
            if config.use_ddp:
                train_metrics = _ddp_reduce_train_metrics(train_metrics, device)
            val_model = model.module if isinstance(model, (torch.nn.DataParallel, DDP)) else model
            if config.use_ddp and (not ddp_is_main):
                val_metrics = {"loss": 0.0, "accuracy": 0.0}
                if config.task == "wikitext103":
                    val_metrics["perplexity"] = float("inf")
            else:
                val_metrics = validate(val_model, val_loader, criterion, device, config.task)
            completed = train_metrics.get("steps_completed", 0)
            total_steps_completed += completed
            total_optimizer_attempts += (completed + accum_steps - 1) // accum_steps
            throughput_key = (
                "tokens_per_s" if config.task in ("sst2", "wikitext103") else "samples_per_s"
            )
            if throughput_key in train_metrics:
                throughput_values.append(float(train_metrics[throughput_key]))
            if "grad_fidelity_values" in train_metrics:
                all_grad_fidelity_values.extend(train_metrics["grad_fidelity_values"])
            if "per_layer_grad_fidelity_records" in train_metrics:
                all_per_layer_grad_fidelity_records.extend(
                    train_metrics["per_layer_grad_fidelity_records"]
                )
            if "svd_per_layer_grad_fidelity_records" in train_metrics:
                all_svd_per_layer_grad_fidelity_records.extend(
                    train_metrics["svd_per_layer_grad_fidelity_records"]
                )
            if "activation_spectrum_records" in train_metrics:
                all_activation_spectrum_records.extend(train_metrics["activation_spectrum_records"])
            if not config.use_ddp or ddp_is_main:
                logger.log_metrics(
                    {
                        "epoch": epoch,
                        "train_loss": train_metrics["train_loss"],
                        "train_accuracy": train_metrics["train_accuracy"],
                        "train_epoch_time_s": train_metrics.get("epoch_time_s"),
                        throughput_key: train_metrics.get(throughput_key),
                        "val_loss": val_metrics["loss"],
                        "val_accuracy": val_metrics["accuracy"],
                    }
                )
            if config.measure_memory:
                peaks = read_peaks()
                if (config.use_ddp or config.use_tensor_parallel) and dist.is_initialized():
                    alloc = peaks.get("max_allocated_bytes", 0) or 0
                    reserved = peaks.get("max_reserved_bytes", 0) or 0
                    rss = peaks.get("cpu_rss_bytes", 0) or 0
                    mem_t = torch.tensor([alloc, reserved, rss], device=device, dtype=torch.long)
                    dist.all_reduce(mem_t, op=dist.ReduceOp.MAX)
                    peaks["max_allocated_bytes"] = mem_t[0].item()
                    peaks["max_reserved_bytes"] = mem_t[1].item()
                    peaks["cpu_rss_bytes"] = mem_t[2].item()
                if not config.use_ddp or ddp_is_main:
                    logger.log_profile({"type": "epoch_peaks", "epoch": epoch, **peaks})
                max_allocated = _update_max(max_allocated, peaks.get("max_allocated_bytes"))
                max_reserved = _update_max(max_reserved, peaks.get("max_reserved_bytes"))
                max_rss = _update_max(max_rss, peaks.get("cpu_rss_bytes"))
            epochs_completed = epoch + 1
            if val_metrics["accuracy"] > best_accuracy:
                best_accuracy = val_metrics["accuracy"]
            if config.task == "wikitext103" and "perplexity" in val_metrics:
                if val_metrics["perplexity"] < best_perplexity:
                    best_perplexity = val_metrics["perplexity"]
            if config.max_steps is not None and total_steps_completed >= config.max_steps:
                break
        total_time = time.time() - start_time
        throughput_mean = (
            sum(throughput_values) / len(throughput_values) if throughput_values else None
        )
        galore_summary = galore_result.to_dict() if galore_result else None
        c1_runtime_stats = None
        if baseline_state is not None and getattr(baseline_state, "c1_runtime", None) is not None:
            c1_runtime_stats = baseline_state.c1_runtime.stats()
        stacr_flags = None
        if config.method in ("stacr", "stacr-v4") or config.stacr_attention:
            stacr_flags = {
                "rank": config.rank,
                "residual_mode": config.stacr_residual_mode,
                "residual_budget": config.stacr_residual_budget,
                "streaming_topk": getattr(config, "stacr_streaming_topk", False),
                "streaming_topk_sample_ratio": getattr(
                    config, "stacr_streaming_topk_sample_ratio", 0.1
                ),
                "chunk_size": getattr(config, "stacr_chunk_size", 512),
                "index_compression": getattr(config, "stacr_index_compression", "none"),
                "skip_residual_enabled": getattr(config, "stacr_skip_residual", False),
                "skip_residual_energy_threshold": getattr(
                    config, "stacr_skip_residual_threshold", 0.05
                ),
                "buffer_reuse": getattr(config, "stacr_buffer_reuse", False),
                "warmup_steps": config.stacr_warmup_steps,
                "post_warmup_mode": getattr(config, "stacr_post_warmup_mode", "full"),
                "residual_warmup_steps": getattr(config, "stacr_residual_warmup_steps", 0),
                "post_warmup_threshold": getattr(config, "stacr_post_warmup_threshold", 0.1),
                "enable_low_rank_only_fast_path": getattr(
                    config, "stacr_enable_low_rank_only_fast_path", True
                ),
                "refresh_trigger": getattr(config, "stacr_refresh_trigger", "off"),
                "refresh_period": getattr(config, "stacr_refresh_period", 1000),
                "refresh_energy_threshold": getattr(config, "stacr_refresh_energy_threshold", 0.1),
                "cache_basis_per_optimizer_step": getattr(
                    config, "stacr_cache_basis_per_optimizer_step", False
                ),
                "oja_update_per_optimizer_step": getattr(
                    config, "stacr_oja_update_per_optimizer_step", False
                ),
                "refresh_count_total": 0,
                "refresh_count_by_layer": {},
                "refresh_count_by_reason": {},
                "microbatch_cache_enabled": getattr(
                    config, "stacr_cache_basis_per_optimizer_step", False
                ),
                "microbatch_cache_hits": 0,
                "microbatch_cache_misses": 0,
                "b1_route_layer_count": 0,
                "b1_route_step_count": 0,
                "b1_generic_fallback_count": 0,
                "freeze_basis_after_warmup": getattr(
                    config, "stacr_freeze_basis_after_warmup", True
                ),
                "use_mgs": config.stacr_use_mgs,
                "lazy_update": config.stacr_lazy_update,
                "lazy_update_threshold": config.stacr_lazy_update_threshold,
                "use_fused_kernel": config.stacr_use_fused_kernel,
                "quantize_coefficients": getattr(config, "stacr_quantize_coefficients", True),
                "quantize_bits": getattr(config, "stacr_quantize_bits", 8),
                "quantize_residuals": getattr(config, "stacr_quantize_residuals", True),
                "residual_quantize_bits": getattr(config, "stacr_residual_quantize_bits", 4),
                "attention_enabled": config.stacr_attention,
                "attention_replaced_qkv": stacr_attention_report.replaced_qkv
                if stacr_attention_report
                else 0,
                "rank_mode": getattr(config, "stacr_rank_mode", "fixed"),
                "rank_variance_threshold": getattr(config, "stacr_rank_variance_threshold", 0.95),
                "rank_min": getattr(config, "stacr_rank_min", 4),
                "rank_max": getattr(config, "stacr_rank_max", 64),
                "rank_adjust_interval": getattr(config, "stacr_rank_adjust_interval", 100),
                "rank_sensitivity_low": getattr(config, "stacr_rank_sensitivity_low", 0.03),
                "rank_sensitivity_high": getattr(config, "stacr_rank_sensitivity_high", 0.1),
                "rank_step_fraction": getattr(config, "stacr_rank_step_fraction", 0.125),
                "planner_enabled": getattr(config, "stacr_use_planner", False),
                "planner_enforced": getattr(config, "stacr_planner_enforce", False),
                "force_compress_layers": getattr(config, "stacr_force_compress_layers", None),
            }
        elif getattr(config, "stacr_compress_checkpoint", False):
            stacr_flags = {}
        prac_flags = None
        if config.method == "prac":
            prac_flags = {
                "rank": config.rank,
                "update_proj_gap": int(config.prac_update_proj_gap),
                "seed_base": int(config.prac_seed_base),
                "proj_dtype": _resolve_runtime_prac_proj_dtype(config),
                "proj_dtype_raw": str(getattr(config, "prac_proj_dtype", "auto")),
            }
        grad_fidelity_stats = None
        valid_grad_fidelity = [
            v for v in all_grad_fidelity_values if isinstance(v, (int, float)) and math.isfinite(v)
        ]
        if valid_grad_fidelity:
            sorted_values = sorted(valid_grad_fidelity)
            n = len(sorted_values)
            grad_fidelity_stats = {
                "grad_cosine_mean": sum(valid_grad_fidelity) / n,
                "grad_cosine_median": sorted_values[n // 2]
                if n % 2 == 1
                else (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2,
                "grad_cosine_min": min(valid_grad_fidelity),
                "grad_cosine_max": max(valid_grad_fidelity),
                "num_measurements": n,
            }
        per_layer_grad_fidelity_summary = _summarize_per_layer_grad_fidelity(
            all_per_layer_grad_fidelity_records
        )
        svd_per_layer_grad_fidelity_summary = _summarize_per_layer_grad_fidelity(
            all_svd_per_layer_grad_fidelity_records
        )
        activation_spectrum_summary = _summarize_activation_spectrum(
            all_activation_spectrum_records
        )
        effective_rank_stats = None
        effective_rank_by_layer = None
        stacr_skip_stats = None
        stacr_refresh_stats = None
        stacr_microbatch_cache_stats = None
        stacr_b1_route_stats = None
        stacr_b3_rank_stats = None
        stacr_b4_fused_stats = None
        if config.method in ("stacr", "stacr-v4") or config.stacr_attention:
            effective_ranks = {}
            active_ranks = {}
            skipped_residuals_by_layer = {}
            skipped_residuals_total = 0
            refresh_count_by_layer = {}
            refresh_total = 0
            refresh_reason_counts = {}
            microbatch_hits_total = 0
            microbatch_misses_total = 0
            b1_route_layer_count = 0
            b1_route_step_total = 0
            b1_generic_fallback_total = 0
            b3_reason_code_counts = {}
            b3_rank_trajectory = {}
            b4_fused_route_counts = {}
            b4_numerical_parity_error = {}
            for name, mod in model.named_modules():
                if isinstance(mod, STACRLinear):
                    effective_ranks[name] = mod.get_effective_rank()
                    active_ranks[name] = mod.get_active_rank()
                    layer_state = getattr(mod, "layer_state", {})
                    skipped = int(layer_state.get("skipped_residuals", 0))
                    if skipped:
                        skipped_residuals_by_layer[name] = skipped
                    skipped_residuals_total += skipped
                    refresh_count = int(layer_state.get("refresh_count_total", 0))
                    if refresh_count:
                        refresh_count_by_layer[name] = refresh_count
                    refresh_total += refresh_count
                    for reason, count in layer_state.get("refresh_count_by_reason", {}).items():
                        refresh_reason_counts[reason] = int(
                            refresh_reason_counts.get(reason, 0)
                        ) + int(count)
                    microbatch_hits_total += int(layer_state.get("microbatch_cache_hits", 0))
                    microbatch_misses_total += int(layer_state.get("microbatch_cache_misses", 0))
                    b1_route_steps = int(layer_state.get("b1_route_step_count", 0))
                    if b1_route_steps > 0:
                        b1_route_layer_count += 1
                    b1_route_step_total += b1_route_steps
                    b1_generic_fallback_total += int(
                        layer_state.get("b1_generic_fallback_count", 0)
                    )
                    for reason, count in layer_state.get("rank_reason_code_counts", {}).items():
                        b3_reason_code_counts[reason] = int(
                            b3_reason_code_counts.get(reason, 0)
                        ) + int(count)
                    trajectory = layer_state.get("rank_trajectory")
                    if trajectory:
                        b3_rank_trajectory[name] = trajectory
                    route_counts = layer_state.get("b4_fused_kernel_route_counts") or {}
                    if route_counts:
                        for route, count in route_counts.items():
                            b4_fused_route_counts[route] = int(
                                b4_fused_route_counts.get(route, 0)
                            ) + int(count)
                    else:
                        route = layer_state.get("b4_fused_kernel_route")
                        if route is not None:
                            b4_fused_route_counts[route] = (
                                int(b4_fused_route_counts.get(route, 0)) + 1
                            )
                    parity_error = layer_state.get("b4_numerical_parity_error")
                    if parity_error is not None:
                        b4_numerical_parity_error[name] = parity_error
            if effective_ranks:
                rank_values = list(effective_ranks.values())
                effective_rank_by_layer = effective_ranks
                effective_rank_stats = {
                    "rank_min": min(rank_values),
                    "rank_max": max(rank_values),
                    "rank_mean": sum(rank_values) / len(rank_values),
                    "num_layers": len(rank_values),
                    "rank_mode": config.stacr_rank_mode,
                }
                active_rank_values = list(active_ranks.values())
                stacr_b3_rank_stats = {
                    "b3_controller": config.stacr_rank_mode,
                    "b3_rank_by_layer_final": active_ranks,
                    "b3_rank_trajectory": b3_rank_trajectory,
                    "b3_reason_code_counts": b3_reason_code_counts,
                    "active_rank_stats": {
                        "rank_min": min(active_rank_values),
                        "rank_max": max(active_rank_values),
                        "rank_mean": sum(active_rank_values) / len(active_rank_values),
                        "num_layers": len(active_rank_values),
                    },
                }
            if skipped_residuals_total > 0 or skipped_residuals_by_layer:
                stacr_skip_stats = {
                    "skipped_residuals_total": skipped_residuals_total,
                    "skipped_residuals_by_layer": skipped_residuals_by_layer,
                }
            if refresh_total > 0 or refresh_count_by_layer or refresh_reason_counts:
                stacr_refresh_stats = {
                    "refresh_count_total": refresh_total,
                    "refresh_count_by_layer": refresh_count_by_layer,
                    "refresh_count_by_reason": refresh_reason_counts,
                }
            if microbatch_hits_total > 0 or microbatch_misses_total > 0:
                stacr_microbatch_cache_stats = {
                    "microbatch_cache_enabled": getattr(
                        config, "stacr_cache_basis_per_optimizer_step", False
                    ),
                    "microbatch_cache_hits": microbatch_hits_total,
                    "microbatch_cache_misses": microbatch_misses_total,
                }
            if b1_route_layer_count > 0 or b1_route_step_total > 0 or b1_generic_fallback_total > 0:
                stacr_b1_route_stats = {
                    "b1_route_layer_count": b1_route_layer_count,
                    "b1_route_step_count": b1_route_step_total,
                    "b1_generic_fallback_count": b1_generic_fallback_total,
                }
            if b4_fused_route_counts or b4_numerical_parity_error:
                stacr_b4_fused_stats = {
                    "b4_fused_kernel_route_counts": b4_fused_route_counts,
                    "b4_numerical_parity_error": b4_numerical_parity_error,
                }
        if stacr_flags is not None:
            if stacr_refresh_stats is not None:
                stacr_flags.update(stacr_refresh_stats)
            if stacr_microbatch_cache_stats is not None:
                stacr_flags.update(stacr_microbatch_cache_stats)
            if stacr_b1_route_stats is not None:
                stacr_flags.update(stacr_b1_route_stats)
            if stacr_b3_rank_stats is not None:
                stacr_flags.update(stacr_b3_rank_stats)
            if stacr_b4_fused_stats is not None:
                stacr_flags.update(stacr_b4_fused_stats)
        if stacr_flags is not None and c1_runtime_stats is not None:
            c1_per_site = c1_runtime_stats.get("per_site", {})
            c1_rank_value = int(getattr(config, "stacr_ckpt_rank", 0) or 0)
            if c1_rank_value <= 0 and c1_per_site:
                first_site = next(iter(c1_per_site.values()))
                c1_rank_value = int(first_site.get("rank", 0))
            c1_basis_dtype = getattr(config, "stacr_ckpt_basis_dtype", "auto")
            c1_refresh_energy_threshold_mode = getattr(
                config, "stacr_ckpt_refresh_energy_threshold_mode", "auto"
            )
            c1_refresh_energy_threshold = float(
                getattr(config, "stacr_ckpt_refresh_energy_threshold", 0.15)
            )
            if c1_per_site:
                first_site = next(iter(c1_per_site.values()))
                c1_basis_dtype = first_site.get("basis_dtype", c1_basis_dtype)
                c1_refresh_energy_threshold = float(
                    first_site.get("refresh_energy_threshold", c1_refresh_energy_threshold)
                )
            c1_energy_values = [
                float(site.get("energy_ewma"))
                for site in c1_per_site.values()
                if int(site.get("compressed_pack_count", 0)) > 0
                and isinstance(site.get("energy_ewma"), (int, float))
                and math.isfinite(site.get("energy_ewma"))
            ]
            c1_raw_bytes_total = sum(
                (int(site.get("raw_bytes_total", 0)) for site in c1_per_site.values())
            )
            c1_stored_payload_bytes_total = sum(
                (int(site.get("stored_bytes_total", 0)) for site in c1_per_site.values())
            )
            c1_basis_persistent_bytes_total = sum(
                (int(site.get("basis_persistent_bytes", 0)) for site in c1_per_site.values())
            )
            c1_stored_bytes_total = c1_stored_payload_bytes_total + c1_basis_persistent_bytes_total
            c1_steps_observed = max(
                (int(site.get("step_count", 0)) for site in c1_per_site.values()), default=0
            )
            c1_stored_boundary_bytes_total_amortized = (
                c1_stored_payload_bytes_total + c1_basis_persistent_bytes_total * c1_steps_observed
            )
            c1_fallback_reasons = {}
            for site in c1_per_site.values():
                for reason, count in site.get("fallback_reasons", {}).items():
                    c1_fallback_reasons[str(reason)] = int(
                        c1_fallback_reasons.get(str(reason), 0)
                    ) + int(count)
            stacr_flags.update(
                {
                    "c1_enabled": True,
                    "c1_checkpoint_mode": getattr(config.baseline_cfg, "checkpointing", "none"),
                    "c1_target": getattr(config, "stacr_ckpt_target", "decoder_block_input"),
                    "c1_site_count_total": int(c1_runtime_stats.get("site_count_total", 0)),
                    "c1_site_count_registered": int(
                        c1_runtime_stats.get("site_count_registered", 0)
                    ),
                    "c1_site_count_active": int(c1_runtime_stats.get("site_count_active", 0)),
                    "c1_site_count_engaged": int(c1_runtime_stats.get("site_count_engaged", 0)),
                    "c1_site_count_compressed": int(
                        c1_runtime_stats.get("site_count_compressed", 0)
                    ),
                    "c1_rank": c1_rank_value,
                    "c1_warmup_steps": int(getattr(config, "stacr_ckpt_warmup_steps", 500)),
                    "c1_residual_mode": getattr(config, "stacr_ckpt_residual_mode", "none"),
                    "c1_residual_budget": float(
                        getattr(config, "stacr_ckpt_residual_budget", 0.005)
                    ),
                    "c1_basis_dtype": c1_basis_dtype,
                    "c1_refresh_trigger": getattr(
                        config, "stacr_ckpt_refresh_trigger", "residual_energy"
                    ),
                    "c1_refresh_energy_threshold_mode": c1_refresh_energy_threshold_mode,
                    "c1_refresh_energy_threshold": c1_refresh_energy_threshold,
                    "c1_quantize_coefficients": bool(
                        getattr(config, "stacr_ckpt_quantize_coefficients", False)
                    ),
                    "c1_quantize_residuals": bool(
                        getattr(config, "stacr_ckpt_quantize_residuals", True)
                    ),
                    "c1_steps_observed": c1_steps_observed,
                    "c1_raw_boundary_bytes_total": c1_raw_bytes_total,
                    "c1_stored_boundary_payload_bytes_total": c1_stored_payload_bytes_total,
                    "c1_basis_persistent_bytes_total": c1_basis_persistent_bytes_total,
                    "c1_stored_boundary_bytes_total": c1_stored_bytes_total,
                    "c1_stored_boundary_bytes_total_amortized": c1_stored_boundary_bytes_total_amortized,
                    "c1_net_saved_boundary_payload_bytes_total": c1_raw_bytes_total
                    - c1_stored_payload_bytes_total,
                    "c1_net_saved_bytes_total": c1_raw_bytes_total
                    - c1_stored_boundary_bytes_total_amortized,
                    "c1_basis_refresh_total": sum(
                        (int(site.get("basis_refresh_count", 0)) for site in c1_per_site.values())
                    ),
                    "c1_basis_init_refresh_total": sum(
                        (
                            int(site.get("basis_init_refresh_count", 0))
                            for site in c1_per_site.values()
                        )
                    ),
                    "c1_basis_warmup_refresh_total": sum(
                        (
                            int(site.get("basis_warmup_refresh_count", 0))
                            for site in c1_per_site.values()
                        )
                    ),
                    "c1_basis_trigger_refresh_total": sum(
                        (
                            int(site.get("basis_trigger_refresh_count", 0))
                            for site in c1_per_site.values()
                        )
                    ),
                    "c1_reconstruction_energy_mean": sum(c1_energy_values) / len(c1_energy_values)
                    if c1_energy_values
                    else None,
                    "c1_reconstruction_energy_min": min(c1_energy_values)
                    if c1_energy_values
                    else None,
                    "c1_reconstruction_energy_max": max(c1_energy_values)
                    if c1_energy_values
                    else None,
                    "c1_fallback_site_count": sum(
                        (
                            1
                            for site in c1_per_site.values()
                            if int(site.get("fallback_count", 0)) > 0
                        )
                    ),
                    "c1_fallback_reasons": c1_fallback_reasons,
                    "c1_duplicate_pack_count": sum(
                        (int(site.get("duplicate_pack_count", 0)) for site in c1_per_site.values())
                    ),
                    "c1_pack_hook_fire_count_by_site": {
                        site_id: int(site.get("pack_count", 0))
                        for (site_id, site) in c1_per_site.items()
                    },
                    "c1_planner_mode": c1_runtime_stats.get("planner_mode"),
                    "c1_planner_summary": c1_runtime_stats.get("planner_summary"),
                    "c1_per_site": c1_per_site,
                }
            )
            stacr_flags.update(_summarize_c1_first_nonfinite_event(c1_per_site))
            stacr_flags.update(_summarize_c1_memory_phase_peaks(config))
            stacr_flags.update(_get_c1_first_finite_drift_event(config))
            planner_decisions = c1_runtime_stats.get("planner_decisions")
            if planner_decisions is not None:
                stacr_flags["c1_planner_decisions"] = planner_decisions
        params_bytes = sum((p.numel() * p.element_size() for p in model.parameters()))
        optimizer_name = optimizer.__class__.__name__
        optimizer_name_lower = optimizer_name.lower()
        if "8bit" in optimizer_name_lower or "paged" in optimizer_name_lower:
            optimizer_multiplier = 0.5
        elif "adam" in optimizer_name_lower:
            optimizer_multiplier = 2
        elif "sgd" in optimizer_name_lower:
            optimizer_multiplier = 1
        else:
            optimizer_multiplier = 1
        optimizer_bytes_estimate = params_bytes * optimizer_multiplier
        if config.method == "prac":
            storage_estimate = estimate_storage_for_prac(model, config)
        activation_bytes_estimate = None
        if storage_estimate is not None:
            activation_bytes_estimate = storage_estimate.get("stored_bytes")
        memory_breakdown = {
            "params_bytes": params_bytes,
            "optimizer_bytes_estimate": optimizer_bytes_estimate,
            "optimizer_type": optimizer_name,
            "optimizer_multiplier": optimizer_multiplier,
            "activation_bytes_estimate": activation_bytes_estimate,
            "total_static_estimate": params_bytes
            + optimizer_bytes_estimate
            + (activation_bytes_estimate or 0),
        }
        if (
            (config.use_ddp or config.use_tensor_parallel)
            and dist.is_initialized()
            and (max_allocated is not None)
        ):
            mem_t = torch.tensor(
                [max_allocated or 0, max_reserved or 0, max_rss or 0],
                device=device,
                dtype=torch.long,
            )
            dist.all_reduce(mem_t, op=dist.ReduceOp.MAX)
            (max_allocated, max_reserved, max_rss) = mem_t.tolist()
        _is_main = (
            not config.use_ddp
            and (not config.use_tensor_parallel)
            or int(os.environ.get("LOCAL_RANK", "0")) == 0
        )
        if config.save_final_model and _is_main:
            save_dir = Path(logger.run_dir) / "checkpoint"
            save_dir.mkdir(parents=True, exist_ok=True)
            if hasattr(model, "module"):
                torch.save(model.module.state_dict(), save_dir / "model.pt")
            else:
                torch.save(model.state_dict(), save_dir / "model.pt")
            if not config.quiet:
                print(f"Saved final model checkpoint to {save_dir / 'model.pt'}")
        return {
            "status": "success",
            "note": "convergence_training",
            "results": {
                "task": config.task,
                "method": config.method_label,
                "method_base": config.method,
                "baseline": config.baseline,
                "best_accuracy": best_accuracy,
                "final_accuracy": val_metrics["accuracy"],
                "best_perplexity": best_perplexity if config.task == "wikitext103" else None,
                "final_perplexity": val_metrics.get("perplexity")
                if config.task == "wikitext103"
                else None,
                "epochs_completed": epochs_completed,
                "training_batches_completed": total_steps_completed,
                "optimizer_step_attempts": total_optimizer_attempts,
                "total_time_s": total_time,
                "throughput_mean": throughput_mean,
                "throughput_units": "tokens_per_s"
                if config.task in ("sst2", "wikitext103")
                else "samples_per_s",
                "peak_allocated_bytes": max_allocated,
                "peak_reserved_bytes": max_reserved,
                "cpu_rss_bytes": max_rss,
                "compressor_stats": compressor.get_stats() if compressor else None,
                "storage_estimate": storage_estimate,
                "galore": galore_summary,
                "stacr_flags": stacr_flags,
                "prac_flags": prac_flags,
                "grad_fidelity_stats": grad_fidelity_stats,
                "per_layer_grad_fidelity_summary": per_layer_grad_fidelity_summary,
                "svd_per_layer_grad_fidelity_summary": svd_per_layer_grad_fidelity_summary,
                "activation_spectrum_summary": activation_spectrum_summary,
                "effective_rank_stats": effective_rank_stats,
                "effective_rank_by_layer": effective_rank_by_layer,
                "stacr_skip_stats": stacr_skip_stats,
                "workspace_pool_stats": get_workspace_pool().get_stats()
                if config.stacr_buffer_reuse
                else None,
                "memory_breakdown": memory_breakdown,
                "activation_profile": activation_profile,
                "stacr_layer_decisions": planner_layer_decisions,
                "stacr_layer_decision_summary": planner_layer_decision_summary,
            },
            "comparison_config": {
                "note": "Compare memory/accuracy to plain baseline with SAME lr, epochs, batch_size, seed",
                "this_run_lr": config.lr,
                "this_run_epochs": config.epochs,
                "this_run_batch_size": config.batch_size,
                "this_run_grad_clip": config.grad_clip,
            },
        }
    finally:
        if baseline_state is not None and getattr(baseline_state, "c1_runtime", None) is not None:
            clear_active_c1_runtime(clear_registry=True)
        if gact_controller is not None:
            gact_controller.uninstall_hook()
