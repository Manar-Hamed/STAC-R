"""Byte accounting and gradient-error telemetry for deployment and mechanism studies."""

from __future__ import annotations

import math

import torch

from .metrics import cosine_similarity


@torch.no_grad()
def gradient_bridge(
    activation: torch.Tensor,
    reconstruction: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    spectral: bool = True,
) -> dict:
    """Measure the activation-to-weight-gradient error inequality.

    Spectral mode computes the matrix 2-norm of the upstream gradient. Setting
    spectral=False uses its Frobenius norm for a cheaper, weaker bound. Dense
    reconstructions are used only for this diagnostic, outside the training path.
    Undefined relative bounds are returned as None when the exact gradient is zero.
    """
    if activation.shape != reconstruction.shape or activation.ndim < 2:
        raise ValueError("Activation and reconstruction must have the same matrix-like shape")
    a = activation.detach().reshape(-1, activation.shape[-1]).double()
    reconstructed = reconstruction.detach().reshape_as(a).double()
    g = grad_output.detach().reshape(-1, grad_output.shape[-1]).double()
    if a.shape[0] != g.shape[0] or a.device != g.device or reconstructed.device != a.device:
        raise ValueError("Activations and upstream gradients must share row count and device")
    if not all(bool(torch.isfinite(t).all()) for t in (a, reconstructed, g)):
        raise ValueError("Diagnostic inputs must be finite")
    error = a - reconstructed
    exact = g.t() @ a
    estimated = g.t() @ reconstructed
    delta = g.t() @ error
    a_norm = float(torch.linalg.vector_norm(a))
    error_norm = float(torch.linalg.vector_norm(error))
    exact_norm = float(torch.linalg.vector_norm(exact))
    delta_norm = float(torch.linalg.vector_norm(delta))
    upstream_norm = float(torch.linalg.matrix_norm(g, ord=2 if spectral else "fro"))
    epsilon = delta_norm / exact_norm if exact_norm else None
    bridge_upper = upstream_norm * error_norm / exact_norm if exact_norm else None
    cosine = float(cosine_similarity(exact, estimated))
    sharp_bound = (
        math.sqrt(max(0.0, 1 - epsilon**2)) if epsilon is not None and epsilon < 1 else None
    )
    bound_from_activation = (
        math.sqrt(max(0.0, 1 - bridge_upper**2))
        if bridge_upper is not None and bridge_upper < 1
        else None
    )
    proxy = (1 - cosine) / (1 + cosine) if cosine > -1 else None
    return {
        "activation_relative_error": error_norm / a_norm if a_norm else None,
        "gradient_relative_error": epsilon,
        "cosine": cosine,
        "gradient_error_upper": bridge_upper,
        "kappa": upstream_norm * a_norm / exact_norm if exact_norm else None,
        "sharp_cosine_bound": sharp_bound,
        "activation_cosine_bound": bound_from_activation,
        "bridge_holds": delta_norm <= upstream_norm * error_norm + 1e-10 * max(1, exact_norm),
        "cosine_bound_holds": cosine + 1e-10 >= sharp_bound if sharp_bound is not None else None,
        "cosine_error_proxy": proxy,
        "upstream_norm": "spectral" if spectral else "frobenius",
    }


def residual_proxy_gain(cosine_on: float, cosine_off: float) -> float | None:
    """Return gamma_epsilon=1-(epsilon_on/epsilon_off)^2 for epsilon=(1-c)/(1+c).

    This is the cosine-based proxy used in the residual table, not retained energy.
    """
    if not all(math.isfinite(c) and -1 < c <= 1 for c in (cosine_on, cosine_off)):
        raise ValueError("Cosines must be finite and in (-1, 1]")
    on = (1 - cosine_on) / (1 + cosine_on)
    off = (1 - cosine_off) / (1 + cosine_off)
    return 1 - (on / off) ** 2 if off else None


def surface_ledger(
    wrapped_bytes: int,
    payload_bytes: int,
    baseline_peak_bytes: int,
    *,
    basis_bytes: int = 0,
    workspace_bytes: int | None = None,
) -> dict:
    """Report the activation-byte SSP and separately measured persistent/workspace costs.

    This does not simulate allocation lifetimes or predict an exact CUDA peak.
    """
    from .memory import saved_surface_predictor

    if min(wrapped_bytes, payload_bytes, basis_bytes) < 0 or (
        workspace_bytes is not None and workspace_bytes < 0
    ):
        raise ValueError("Byte counts must be nonnegative")
    saving = wrapped_bytes - payload_bytes
    return {
        "wrapped_bytes": wrapped_bytes,
        "payload_bytes": payload_bytes,
        "activation_saving_bytes": saving,
        "ssp": saved_surface_predictor(saving, baseline_peak_bytes),
        "basis_bytes": basis_bytes,
        "workspace_bytes": workspace_bytes,
        "net_after_basis_bytes": saving - basis_bytes,
        "net_after_workspace_bytes": saving - basis_bytes - workspace_bytes
        if workspace_bytes is not None
        else None,
    }
