"""Checkpoint contexts that preserve adaptive basis updates during replay."""

import threading
from contextlib import contextmanager, nullcontext


class RecomputationGuard:
    def __init__(self):
        self._local = threading.local()

    @property
    def is_recomputing(self):
        return getattr(self._local, "recomputing", False)

    @contextmanager
    def forward_context(self, records):
        previous = getattr(self._local, "recording", None)
        self._local.recording = records
        try:
            yield
        finally:
            self._local.recording = previous

    def record_layer(self, layer, state):
        records = getattr(self._local, "recording", None)
        if records is not None and not self.is_recomputing:
            records.append((id(layer), state))

    def replay_layer(self, layer):
        replay = getattr(self._local, "replay", None)
        if replay is None:
            return None
        identity, state = next(replay)
        if identity != id(layer):
            raise RuntimeError("STAC-R layer order changed during checkpoint replay")
        return state

    @contextmanager
    def recomputation_context(self, records=None):
        previous = self.is_recomputing
        previous_replay = getattr(self._local, "replay", None)
        self._local.recomputing = True
        self._local.replay = None if records is None else iter(records)
        try:
            yield
        finally:
            self._local.recomputing = previous
            self._local.replay = previous_replay


_guard = RecomputationGuard()


def get_global_recomputation_guard():
    return _guard


def checkpoint_contexts():
    """Pass as context_fn to torch.utils.checkpoint.checkpoint(use_reentrant=False)."""
    records = []
    return _guard.forward_context(records), _guard.recomputation_context(records)


def checkpoint(function, *args, **kwargs):
    """Nonreentrant checkpoint with adaptive-update replay protection."""
    from torch.utils.checkpoint import checkpoint as torch_checkpoint

    if kwargs.pop("use_reentrant", False):
        raise ValueError("Adaptive checkpointing requires use_reentrant=False")
    kwargs.setdefault("context_fn", checkpoint_contexts)
    return torch_checkpoint(function, *args, use_reentrant=False, **kwargs)
