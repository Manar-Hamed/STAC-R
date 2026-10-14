"""Read-only compression diagnostics with stochastic model layers disabled."""

from contextlib import contextmanager
import copy

import torch

from .adaptive import STACRLinear as AdaptiveLinear
from .adaptive_checkpointing import get_global_recomputation_guard
from .layers import STACRLinear


def _snapshot(value):
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return value


@contextmanager
def compression_diagnostics(model):
    """Enable compressed gradients while dropout stays off, then restore state.

    Adaptive layers must already have a fitted basis (a training forward is
    sufficient). Diagnostic forwards do not update bases, runtime counters or
    rank controllers. This context does not save or restore parameter gradients.
    """
    layers = [
        module for module in model.modules() if isinstance(module, (STACRLinear, AdaptiveLinear))
    ]
    for layer in layers:
        if (
            isinstance(layer, AdaptiveLinear)
            and layer.config.residual_mode != "sparse_only"
            and not layer.basis_initialized
        ):
            raise ValueError(
                "Initialize adaptive bases with a training forward before computing fidelity"
            )
    modes = {module: module.training for module in model.modules()}
    snapshots = []
    for layer in layers:
        buffers = {
            key: tensor if key == "basis" or tensor is None else tensor.detach().clone()
            for key, tensor in layer._buffers.items()
        }
        if isinstance(layer, AdaptiveLinear):
            snapshots.append(
                (
                    layer,
                    buffers,
                    (layer.config, copy.deepcopy(layer.config)),
                    _snapshot(layer.layer_state),
                )
            )
        else:
            snapshots.append(
                (layer, buffers, (layer.freeze_after_warmup, layer.warmup_steps), None)
            )
    try:
        model.eval()
        for layer in layers:
            layer.training = True
            if isinstance(layer, STACRLinear):
                layer.freeze_after_warmup, layer.warmup_steps = True, 0
        with get_global_recomputation_guard().recomputation_context():
            yield
    finally:
        for layer, buffers, config, state in snapshots:
            for key, tensor in buffers.items():
                setattr(layer, key, tensor)
            if isinstance(layer, AdaptiveLinear):
                original_config, values = config
                original_config.__dict__.clear()
                original_config.__dict__.update(values.__dict__)
                layer.config = original_config
                layer.layer_state.clear()
                layer.layer_state.update(state)
            else:
                layer.freeze_after_warmup, layer.warmup_steps = config
        for module, training in modes.items():
            module.training = training
