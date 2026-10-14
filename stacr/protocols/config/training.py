"""
Training configuration dataclasses.

Defines configurations for:
- Optimizer settings (SGD, Adam, etc.)
- Training loop parameters (epochs, batch size)
- OOM handling and recovery
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Literal


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""

    name: str = "sgd"
    lr: float = 0.01
    weight_decay: float = 0.0
    momentum: float = 0.9
    nesterov: bool = False
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-08

    def get_optimizer_kwargs(self) -> Dict[str, Any]:
        """Get kwargs dict for optimizer construction."""
        kwargs = {"lr": self.lr, "weight_decay": self.weight_decay}
        if self.name == "sgd":
            kwargs["momentum"] = self.momentum
            kwargs["nesterov"] = self.nesterov
        elif self.name in ("adam", "adamw"):
            kwargs["betas"] = self.betas
            kwargs["eps"] = self.eps
        return kwargs


@dataclass
class TrainingConfig:
    """Training loop configuration."""

    epochs: int = 3
    max_steps: Optional[int] = None
    batch_size: int = 128
    eval_batch_size: Optional[int] = None
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss: str = "cross_entropy"
    lr_scheduler: Optional[str] = None
    lr_decay_epochs: list = field(default_factory=lambda: [100, 150])
    lr_decay_factor: float = 0.1
    warmup_epochs: int = 0
    max_oom_retries: int = 3
    oom_backoff_factor: float = 0.5
    enable_checkpointing: bool = False
    enable_amp: bool = False
    grad_accumulation_steps: int = 1
    eval_frequency: int = 1
    log_frequency: int = 10
    early_stopping: bool = False
    patience: int = 5
    min_delta: float = 0.001

    def __post_init__(self):
        """Set defaults for optional fields."""
        if self.eval_batch_size is None:
            self.eval_batch_size = self.batch_size


@dataclass
class BaselineConfig:
    """Baseline method configuration.

    Configures composable baseline toggles: checkpointing, AMP, grad_accum, offload.
    These can be combined (e.g., checkpointing + amp) for realistic baseline comparisons.
    """

    name: str = "baseline"
    checkpointing: str = "none"
    amp: str = "none"
    grad_accum_steps: int = 1
    saved_tensors_offload: str = "none"
    saved_tensors_offload_min_bytes: int = 1000000
    actnn: str = "none"
    gact: str = "none"
    torchao: str = "none"
    galore: str = "none"
    galore_target_modules: str = "none"
    checkpoint_segments: int = 2
    amp_dtype: str = "float16"

    def get_description(self) -> str:
        """Get human-readable description of baseline configuration."""
        parts = []
        if self.checkpointing != "none":
            parts.append(f"ckpt={self.checkpointing}")
        if self.amp != "none":
            parts.append(f"amp={self.amp}")
        if self.grad_accum_steps > 1:
            parts.append(f"accum={self.grad_accum_steps}")
        if self.actnn != "none":
            parts.append(f"actnn={self.actnn}")
        if self.gact != "none":
            parts.append(f"gact={self.gact}")
        if self.torchao != "none":
            parts.append(f"torchao={self.torchao}")
        if self.galore != "none":
            parts.append(f"galore={self.galore}")
        if self.galore_target_modules not in ("none", "", None):
            parts.append(f"galore_targets={self.galore_target_modules}")
        if self.saved_tensors_offload != "none":
            parts.append(f"offload={self.saved_tensors_offload}")
        if not parts:
            return "Standard (no optimization)"
        return " + ".join(parts)

    def is_default(self) -> bool:
        """Check if this is the default (no optimizations) configuration."""
        return (
            self.checkpointing == "none"
            and self.amp == "none"
            and (self.grad_accum_steps == 1)
            and (self.saved_tensors_offload == "none")
            and (self.actnn == "none")
            and (self.gact == "none")
            and (self.torchao == "none")
            and (self.galore == "none")
            and (self.galore_target_modules in ("none", "", None))
        )

    @classmethod
    def from_legacy(cls, baseline_str: str) -> "BaselineConfig":
        """Create BaselineConfig from a single baseline method name."""
        if baseline_str in ("none", "baseline"):
            return cls()
        elif baseline_str == "checkpointing":
            return cls(checkpointing="transformer_hf")
        elif baseline_str == "amp":
            return cls(amp="auto")
        elif baseline_str == "grad_accum":
            return cls(grad_accum_steps=2)
        elif baseline_str == "actnn":
            return cls(actnn="L2")
        elif baseline_str == "gact":
            return cls(gact="L1")
        elif baseline_str == "torchao":
            return cls(torchao="auto")
        elif baseline_str == "galore":
            return cls(galore="rank=128,update_gap=200,scale=0.25")
        else:
            return cls(name=baseline_str)
