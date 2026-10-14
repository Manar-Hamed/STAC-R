"""Runtime rank controllers for streamed activation subspaces.

The controller operates on the active prefix rank used for a forward pass while
keeping the allocated basis shape fixed. This avoids mid-training basis-buffer
resizes and lets the runtime adapt rank at safe boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Optional


@dataclass(frozen=True)
class RankDecision:
    active_rank: int
    reason_code: str
    changed: bool
    metric_value: Optional[float] = None


def _clamp_rank(
    rank: int,
    rank_min: int,
    rank_max: int,
    max_available_rank: int,
) -> int:
    rank_min = max(1, int(rank_min))
    rank_max = max(rank_min, int(rank_max))
    max_available_rank = max(1, int(max_available_rank))
    return min(max_available_rank, max(rank_min, min(int(rank), rank_max)))


def should_adjust_rank(
    schedule_step: int,
    last_adjust_step: int,
    adjust_interval: int,
) -> bool:
    """Gate rank changes so the controller does not thrash every forward."""
    adjust_interval = max(1, int(adjust_interval))
    if last_adjust_step < 0:
        return True
    return (int(schedule_step) - int(last_adjust_step)) >= adjust_interval


def choose_variance_rank(
    coeff_energy_by_rank: Optional[Iterable[float]],
    current_rank: int,
    rank_min: int,
    rank_max: int,
    max_available_rank: int,
    variance_threshold: float,
) -> RankDecision:
    """Choose smallest prefix rank capturing the requested coefficient energy."""
    current_rank = _clamp_rank(current_rank, rank_min, rank_max, max_available_rank)
    if coeff_energy_by_rank is None:
        return RankDecision(
            active_rank=current_rank,
            reason_code="variance_cold_start",
            changed=False,
            metric_value=None,
        )

    energies = []
    for value in coeff_energy_by_rank:
        energy = float(value)
        if not math.isfinite(energy):
            raise ValueError(f"Invalid coefficient energy: {value!r}")
        energies.append(max(0.0, energy))
    if not energies:
        return RankDecision(
            active_rank=current_rank,
            reason_code="variance_empty",
            changed=False,
            metric_value=0.0,
        )

    energies = energies[:max_available_rank]
    total_energy = float(sum(energies))
    if total_energy <= 0.0:
        target_rank = _clamp_rank(rank_min, rank_min, rank_max, max_available_rank)
        return RankDecision(
            active_rank=target_rank,
            reason_code="variance_zero_energy",
            changed=(target_rank != current_rank),
            metric_value=0.0,
        )

    threshold = min(max(float(variance_threshold), 0.0), 1.0)
    target_energy = total_energy * threshold
    running = 0.0
    prefix_rank = 1
    for idx, value in enumerate(energies, start=1):
        running += value
        prefix_rank = idx
        if running >= target_energy:
            break

    target_rank = _clamp_rank(prefix_rank, rank_min, rank_max, max_available_rank)
    if target_rank > current_rank:
        reason = "variance_grow"
    elif target_rank < current_rank:
        reason = "variance_shrink"
    else:
        reason = "variance_keep"
    return RankDecision(
        active_rank=target_rank,
        reason_code=reason,
        changed=(target_rank != current_rank),
        metric_value=(running / total_energy if total_energy > 0 else 0.0),
    )


def choose_sensitivity_rank(
    residual_energy: float,
    current_rank: int,
    rank_min: int,
    rank_max: int,
    max_available_rank: int,
    low_threshold: float,
    high_threshold: float,
    step_fraction: float,
) -> RankDecision:
    """Adjust active rank up/down from residual-energy feedback."""
    current_rank = _clamp_rank(current_rank, rank_min, rank_max, max_available_rank)
    low_threshold = float(low_threshold)
    high_threshold = float(high_threshold)
    if high_threshold < low_threshold:
        low_threshold, high_threshold = high_threshold, low_threshold

    residual_energy = float(residual_energy)
    step_fraction = max(0.01, float(step_fraction))
    rank_delta = max(1, int(round(current_rank * step_fraction)))

    if residual_energy >= high_threshold:
        target_rank = _clamp_rank(current_rank + rank_delta, rank_min, rank_max, max_available_rank)
        return RankDecision(
            active_rank=target_rank,
            reason_code="sensitivity_grow"
            if target_rank != current_rank
            else "sensitivity_keep_high",
            changed=(target_rank != current_rank),
            metric_value=residual_energy,
        )

    if residual_energy <= low_threshold:
        target_rank = _clamp_rank(current_rank - rank_delta, rank_min, rank_max, max_available_rank)
        return RankDecision(
            active_rank=target_rank,
            reason_code="sensitivity_shrink"
            if target_rank != current_rank
            else "sensitivity_keep_low",
            changed=(target_rank != current_rank),
            metric_value=residual_energy,
        )

    return RankDecision(
        active_rank=current_rank,
        reason_code="sensitivity_keep_band",
        changed=False,
        metric_value=residual_energy,
    )


def choose_runtime_rank(
    rank_mode: str,
    current_rank: int,
    rank_min: int,
    rank_max: int,
    max_available_rank: int,
    layer_state: Mapping[str, object],
    variance_threshold: float = 0.95,
    sensitivity_low: float = 0.03,
    sensitivity_high: float = 0.10,
    step_fraction: float = 0.125,
) -> RankDecision:
    """Dispatch to the configured runtime controller."""
    if rank_mode == "fixed":
        target_rank = _clamp_rank(current_rank, rank_min, rank_max, max_available_rank)
        return RankDecision(
            active_rank=target_rank,
            reason_code="fixed",
            changed=(target_rank != current_rank),
            metric_value=None,
        )

    if rank_mode == "variance":
        return choose_variance_rank(
            coeff_energy_by_rank=layer_state.get("last_coeff_energy_by_rank"),
            current_rank=current_rank,
            rank_min=rank_min,
            rank_max=rank_max,
            max_available_rank=max_available_rank,
            variance_threshold=variance_threshold,
        )

    if rank_mode == "sensitivity":
        return choose_sensitivity_rank(
            residual_energy=float(layer_state.get("last_residual_energy", 0.0)),
            current_rank=current_rank,
            rank_min=rank_min,
            rank_max=rank_max,
            max_available_rank=max_available_rank,
            low_threshold=sensitivity_low,
            high_threshold=sensitivity_high,
            step_fraction=step_fraction,
        )

    raise ValueError(f"Unsupported rank_mode: {rank_mode}")
