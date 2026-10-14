from __future__ import annotations

import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SUBTRACK_COMMIT = "aabebe310e95a709454d32ce4c697d09114edac9"
SUBTRACK_URL = "https://github.com/criticalml-uw/SubTrack"
SUBTRACK_SOURCE_ENV = "STACR_SUBTRACK_SOURCE"
INSTALL_COMMAND = (
    f"git clone {SUBTRACK_URL}.git third_party/SubTrack\n"
    f"git -C third_party/SubTrack fetch origin {SUBTRACK_COMMIT}\n"
    f"git -C third_party/SubTrack checkout {SUBTRACK_COMMIT}\n"
    "Set STACR_SUBTRACK_SOURCE to the absolute third_party/SubTrack directory.\n"
    "Install this project's [benchmarks,comparators] extras for its dependencies."
)


def activate_subtrack_source():
    """Expose an explicitly selected upstream checkout, including its utils package.

    Upstream setup.py omits utils even though the optimizer imports it. Using
    the complete checkout avoids modifying or replacing upstream optimizer code.
    """
    configured = os.environ.get(SUBTRACK_SOURCE_ENV)
    if not configured:
        return None
    source = Path(configured).expanduser().resolve()
    if (
        not (source / "low_rank_torch" / "__init__.py").is_file()
        or not (source / "utils" / "file_utils.py").is_file()
    ):
        raise RuntimeError(
            f"{SUBTRACK_SOURCE_ENV} must point to a complete SubTrack checkout: {source}"
        )
    existing = sys.modules.get("low_rank_torch")
    if existing is not None and getattr(existing, "__file__", None):
        if not Path(existing.__file__).resolve().is_relative_to(source):
            raise RuntimeError(
                "A different low_rank_torch package is already loaded; start a fresh process with STACR_SUBTRACK_SOURCE set"
            )
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    return source


def subtrack_available() -> bool:
    activate_subtrack_source()
    return importlib.util.find_spec("low_rank_torch") is not None


def require_subtrack() -> None:
    if not subtrack_available():
        raise RuntimeError(
            f"SubTrack++ is optional. Install the pinned upstream optimizer with:\n{INSTALL_COMMAND}"
        )


def subtrack_revision():
    """Return installed source revision when pip or the source checkout records it."""
    source = activate_subtrack_source()
    if source is not None:
        result = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={source.as_posix()}",
                "-C",
                str(source),
                "rev-parse",
                "HEAD",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    try:
        distribution = importlib.metadata.distribution("subtrackpp-torch")
        direct = distribution.read_text("direct_url.json")
        if direct:
            commit = json.loads(direct).get("vcs_info", {}).get("commit_id")
            if commit:
                return commit
    except importlib.metadata.PackageNotFoundError:
        pass
    spec = importlib.util.find_spec("low_rank_torch")
    if spec is not None and spec.origin:
        source = Path(spec.origin).resolve().parent.parent
        if (source / ".git").exists():
            result = subprocess.run(
                [
                    "git",
                    "-c",
                    f"safe.directory={source.as_posix()}",
                    "-C",
                    str(source),
                    "rev-parse",
                    "HEAD",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                return result.stdout.strip()
    return None


def require_pinned_subtrack():
    require_subtrack()
    actual = subtrack_revision()
    if actual != SUBTRACK_COMMIT:
        raise RuntimeError(
            f"The SubTrack++ benchmark requires revision {SUBTRACK_COMMIT}; "
            f"installed revision is {actual or 'unverified'}.\n{INSTALL_COMMAND}"
        )
    source = activate_subtrack_source()
    if source is not None:
        changed = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={source.as_posix()}",
                "-C",
                str(source),
                "diff",
                "HEAD",
                "--quiet",
                "--",
                "low_rank_torch",
                "utils",
            ],
            check=False,
        )
        if changed.returncode:
            raise RuntimeError(
                "SubTrack optimizer source has local changes; use a clean pinned checkout for the benchmark"
            )


def create_subtrack_optimizer(model, config):
    """Build the upstream optimizer for ordinary and STAC-R-wrapped linears.

    The dedicated ``subtrack_composition`` benchmark preserves the upstream BERT
    experiment's exact module-selection rule and training recipe.
    """
    require_subtrack()
    try:
        from low_rank_torch import LowRankAdamW
    except ImportError as exc:
        raise RuntimeError(
            "SubTrack++ dependencies are incomplete; install stacr[comparators] and the pinned upstream optimizer."
        ) from exc
    from stacr.surgery import classify
    from stacr.adaptive import STACRLinear as AdaptiveSTACRLinear
    from torch.nn import Parameter

    selected, selected_names, seen = [], [], set()
    requested = config.get("subtrack_target_modules")
    for name, module in model.named_modules():
        if isinstance(module, AdaptiveSTACRLinear):
            # Select by the wrapper's logical name, but optimize the original
            # leaf parameter. A Conv1D facade's .weight is a transposed view.
            weight = getattr(module.linear, "source_weight", None)
            if weight is None:
                weight = module.linear.weight
        else:
            weight = getattr(module, "weight", None)
        if (
            not isinstance(weight, Parameter)
            or not weight.is_leaf
            or weight.ndim != 2
            or not weight.requires_grad
            or id(weight) in seen
        ):
            continue
        if requested:
            matches = any(name == target or name.endswith("." + target) for target in requested)
        else:
            matches = classify(name) in ("mlp", "attention")
        if matches:
            selected.append(weight)
            selected_names.append(name + ".weight")
            seen.add(id(weight))
    if not selected:
        raise ValueError(
            "SubTrack++ selected no matrix weights; set subtrack_target_modules explicitly"
        )
    rank = int(config.get("subtrack_rank", 4))
    if rank < 1 or any(rank > min(p.shape) for p in selected):
        raise ValueError("SubTrack++ rank must fit every selected matrix")
    ordinary = [p for p in model.parameters() if p.requires_grad and id(p) not in seen]
    projected = {
        "params": selected,
        "rank": rank,
        "scale": float(config.get("subtrack_scale", 4.0)),
        "proj_type": config.get("subtrack_proj_type", "std"),
        "subspace_update_method": "subtrack",
        "subspace_update_interval": int(config.get("subtrack_update_interval", 500)),
        "st_init_step_size": float(config.get("subtrack_step_size", 0.001)),
        "st_step_size_scheduler": "constant",
        "st_step_size_coef": 1.0,
        "st_noise_sigma2": 0.0,
        "st_subspace_coef": 1.0,
        "adaptive_optimizer": bool(config.get("subtrack_adaptive_optimizer", True)),
        "recovery_scaling": bool(config.get("subtrack_recovery_scaling", True)),
        "norm_growth_limit": float(config.get("subtrack_norm_growth_limit", 1.01)),
        "norm_growth_limiter_off": False,
    }
    optimizer = LowRankAdamW(
        [{"params": ordinary}, projected],
        lr=float(config.get("lr", 1e-5)),
        betas=(0.908, 0.99),
        eps=1e-8,
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    optimizer.stacr_projected_parameter_names = selected_names
    optimizer.stacr_upstream_commit = subtrack_revision()
    return optimizer
