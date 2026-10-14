"""
Peak memory profiling by execution phase.

Provides per-phase memory profiling with explicit CUDA vs CPU RSS modes.
Never silently reports CUDA zeros on CPU - explicit mode selection required.
"""

import contextlib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peaks():
    if torch.cuda.is_available():
        cuda_sync()
        torch.cuda.reset_peak_memory_stats()


def read_peaks() -> Dict[str, Any]:
    """
    Read peak memory statistics with explicit device mode.

    Returns nulls for CUDA fields on CPU and records CPU RSS separately.

    Returns:
        Dictionary with:
        - device_mode: "cuda" or "cpu" (explicit mode indicator)
        - max_allocated_bytes: int or None (None on CPU)
        - max_reserved_bytes: int or None (None on CPU)
        - cpu_rss_bytes: int or None (None on CUDA)
        - note: Optional warning message

    Example:
        >>> peaks = read_peaks()
        >>> if peaks["device_mode"] == "cuda":
        ...     print(f"Peak allocated: {peaks['max_allocated_bytes']} bytes")
        >>> else:
        ...     print(f"RSS: {peaks['cpu_rss_bytes']} bytes")

    Return-value conventions:
        CUDA fields are None on CPU. Plotting code should check device_mode
        before accessing device-specific measurements.
    """
    if not torch.cuda.is_available():
        try:
            import psutil

            rss = psutil.Process(os.getpid()).memory_info().rss
        except Exception:
            rss = None
        return {
            "device_mode": "cpu",
            "max_allocated_bytes": None,
            "max_reserved_bytes": None,
            "cpu_rss_bytes": rss,
            "note": "cuda_unavailable",
        }
    cuda_sync()
    return {
        "device_mode": "cuda",
        "max_allocated_bytes": torch.cuda.max_memory_allocated(),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(),
        "cpu_rss_bytes": None,
    }


def capture_environment() -> Dict[str, Any]:
    """Capture environment and allocator configuration for reproducibility."""
    env = {"torch_version": torch.__version__, "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        env.update(
            {
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version()
                if torch.backends.cudnn.is_available()
                else None,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(),
                "device_capability": torch.cuda.get_device_capability(),
                "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            }
        )
        try:
            env["allocator_backend"] = torch.cuda.memory.get_allocator_backend()
        except Exception:
            env["allocator_backend"] = "unknown"
        env["tf32_matmul_allow"] = torch.backends.cuda.matmul.allow_tf32
        env["tf32_cudnn_allow"] = torch.backends.cudnn.allow_tf32
    env["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    env["CUDA_LAUNCH_BLOCKING"] = os.environ.get("CUDA_LAUNCH_BLOCKING")
    return env


@dataclass
class PhaseResult:
    """Memory and timing result for a single phase."""

    time_ms: float = 0.0
    start_allocated_bytes: int = 0
    start_reserved_bytes: int = 0
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0
    end_allocated_bytes: int = 0
    end_reserved_bytes: int = 0
    delta_peak_allocated_bytes: int = 0
    delta_peak_reserved_bytes: int = 0
    start_rss_bytes: int = 0
    peak_rss_bytes: int = 0
    end_rss_bytes: int = 0
    call_count: int = 1


@dataclass
class StepProfile:
    """Profile for a single training step with per-phase breakdown."""

    step: int
    device: str
    mode: str
    global_peak_allocated_bytes: int = 0
    global_peak_reserved_bytes: int = 0
    global_peak_rss_bytes: int = 0
    time_ms: float = 0.0
    phases: Dict[str, PhaseResult] = field(default_factory=dict)


class PhaseProfiler:
    """
    Allocator-aware peak-memory profiler with per-phase instrumentation.

    Usage:
        profiler = PhaseProfiler(device="cuda:0", out_path="profile.jsonl")
        profiler.write_metadata()

        with profiler.step(step_idx=0):
            with profiler.phase("forward_compute"):
                output = model(input)
            with profiler.phase("backward_total"):
                loss.backward()
            with profiler.phase("optimizer"):
                optimizer.step()

    Key features:
    - Explicit CUDA vs CPU RSS mode (no silent zeros)
    - Per-phase peak tracking with aggregation for repeated phases
    - Environment/allocator config logging
    - Structured JSON line output
    """

    def __init__(self, device: str = "auto", out_path: Optional[str] = None, enabled: bool = True):
        """
        Initialize the profiler.

        Args:
            device: "cuda", "cuda:N", "cpu", or "auto" (detect from torch.cuda.is_available)
            out_path: Path to write profile.jsonl (None = don't write)
            enabled: If False, profiler becomes a no-op
        """
        self.enabled = enabled
        self.out_path = Path(out_path) if out_path else None
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.mode = "cuda" if self.device.startswith("cuda") else "cpu_rss"
        if self.mode == "cuda" and (not torch.cuda.is_available()):
            raise RuntimeError(
                f"PhaseProfiler: device='{device}' requested but CUDA is unavailable. Use device='cpu' for CPU RSS profiling to avoid silent zeros."
            )
        self._current_step: Optional[StepProfile] = None
        self.last_step: Optional[StepProfile] = None
        self._step_start_time: float = 0.0
        self._phase_stack: List[Dict[str, Any]] = []
        self._step_start_allocated: int = 0
        self._step_start_reserved: int = 0
        self._step_start_rss: int = 0
        self._step_manual_peak_allocated: int = 0
        self._step_manual_peak_reserved: int = 0
        self._step_manual_peak_rss: int = 0

    def write_metadata(self, extra: Optional[Dict[str, Any]] = None) -> None:
        """Write environment metadata as first line of profile.jsonl."""
        if not self.enabled or not self.out_path:
            return
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        if self.out_path.exists():
            try:
                with open(self.out_path, "r") as f:
                    first = f.readline()
                if first:
                    payload = json.loads(first)
                    if isinstance(payload, dict) and payload.get("type") == "meta":
                        return
            except Exception:
                pass
        meta = {
            "type": "meta",
            "profiler_mode": self.mode,
            "device": self.device,
            **capture_environment(),
        }
        if extra:
            meta.update(extra)
        with open(self.out_path, "a") as f:
            f.write(json.dumps(meta) + "\n")

    def _get_cuda_memory(self) -> tuple:
        """Get current CUDA memory stats."""
        if self.mode != "cuda":
            return (0, 0)
        cuda_sync()
        return (torch.cuda.memory_allocated(), torch.cuda.memory_reserved())

    def _get_cuda_peaks(self) -> tuple:
        """Get peak CUDA memory since last reset."""
        if self.mode != "cuda":
            return (0, 0)
        cuda_sync()
        return (torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())

    def _get_rss(self) -> int:
        """Get current process RSS."""
        try:
            import psutil

            return psutil.Process(os.getpid()).memory_info().rss
        except Exception:
            return 0

    def _reset_cuda_peaks(self) -> None:
        """Reset CUDA peak memory tracking."""
        if self.mode == "cuda":
            cuda_sync()
            torch.cuda.reset_peak_memory_stats()

    def _update_phase_manual_peak(self, phase_state: Dict[str, Any]) -> None:
        """Update manual peaks for an active phase using a current snapshot."""
        if self.mode == "cuda":
            (alloc, reserved) = self._get_cuda_memory()
            phase_state["manual_peak_allocated"] = max(phase_state["manual_peak_allocated"], alloc)
            phase_state["manual_peak_reserved"] = max(phase_state["manual_peak_reserved"], reserved)
        else:
            rss = self._get_rss()
            phase_state["manual_peak_rss"] = max(phase_state["manual_peak_rss"], rss)

    @contextlib.contextmanager
    def step(self, step_idx: int):
        """
        Context manager for a training step.

        Tracks global peak memory across all phases and writes step profile on exit.
        """
        if not self.enabled:
            yield
            return
        self._current_step = StepProfile(step=step_idx, device=self.device, mode=self.mode)
        self._step_start_time = time.perf_counter()
        (self._step_start_allocated, self._step_start_reserved) = self._get_cuda_memory()
        self._step_start_rss = self._get_rss()
        self._step_manual_peak_allocated = self._step_start_allocated
        self._step_manual_peak_reserved = self._step_start_reserved
        self._step_manual_peak_rss = self._step_start_rss
        self._phase_stack = []
        self._reset_cuda_peaks()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - self._step_start_time) * 1000.0
            (end_alloc, end_res) = self._get_cuda_memory()
            end_rss = self._get_rss()
            if self.mode == "cuda":
                phase_allocs = [p.peak_allocated_bytes for p in self._current_step.phases.values()]
                phase_res = [p.peak_reserved_bytes for p in self._current_step.phases.values()]
                self._current_step.global_peak_allocated_bytes = max(
                    [self._step_manual_peak_allocated, end_alloc] + phase_allocs
                )
                self._current_step.global_peak_reserved_bytes = max(
                    [self._step_manual_peak_reserved, end_res] + phase_res
                )
            else:
                end_rss = self._get_rss()
                self._current_step.global_peak_rss_bytes = max(
                    self._step_start_rss,
                    end_rss,
                    max((p.peak_rss_bytes for p in self._current_step.phases.values()), default=0),
                )
            self._current_step.time_ms = elapsed_ms
            self.last_step = self._current_step
            self._write_step()
            self._current_step = None

    @contextlib.contextmanager
    def phase(self, name: str, reset_peaks: bool = True, **extra):
        """
        Context manager for a phase within a step.

        Multiple calls to the same phase name within a step are aggregated:
        - peak = max(peaks across calls)
        - time = sum(times across calls)

        Args:
            name: Phase name (e.g., "forward_compute", "compress", "backward_total", "optimizer")
            **extra: Additional metadata to attach to the phase
        """
        if not self.enabled or self._current_step is None:
            yield
            return
        parent_state = self._phase_stack[-1] if self._phase_stack else None
        if parent_state:
            self._update_phase_manual_peak(parent_state)
        start_time = time.perf_counter()
        (start_alloc, start_res) = self._get_cuda_memory()
        start_rss = self._get_rss()
        if reset_peaks:
            self._reset_cuda_peaks()
        phase_state = {
            "name": name,
            "start_allocated": start_alloc,
            "start_reserved": start_res,
            "start_rss": start_rss,
            "manual_peak_allocated": start_alloc,
            "manual_peak_reserved": start_res,
            "manual_peak_rss": start_rss,
            "had_nested": False,
        }
        self._phase_stack.append(phase_state)
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            (end_alloc, end_res) = self._get_cuda_memory()
            end_rss = self._get_rss()
            if self.mode == "cuda":
                (peak_alloc, peak_res) = self._get_cuda_peaks()
            else:
                (peak_alloc, peak_res) = (0, 0)
            phase_state["manual_peak_allocated"] = max(
                phase_state["manual_peak_allocated"], end_alloc
            )
            phase_state["manual_peak_reserved"] = max(phase_state["manual_peak_reserved"], end_res)
            phase_state["manual_peak_rss"] = max(phase_state["manual_peak_rss"], end_rss)
            if phase_state["had_nested"]:
                peak_alloc = max(peak_alloc, phase_state["manual_peak_allocated"])
                peak_res = max(peak_res, phase_state["manual_peak_reserved"])
            result = PhaseResult(
                time_ms=elapsed_ms,
                start_allocated_bytes=start_alloc,
                start_reserved_bytes=start_res,
                peak_allocated_bytes=peak_alloc,
                peak_reserved_bytes=peak_res,
                end_allocated_bytes=end_alloc,
                end_reserved_bytes=end_res,
                delta_peak_allocated_bytes=peak_alloc - start_alloc,
                delta_peak_reserved_bytes=peak_res - start_res,
                start_rss_bytes=start_rss,
                peak_rss_bytes=max(phase_state["manual_peak_rss"], end_rss),
                end_rss_bytes=end_rss,
                call_count=1,
            )
            if name in self._current_step.phases:
                existing = self._current_step.phases[name]
                existing.time_ms += result.time_ms
                existing.peak_allocated_bytes = max(
                    existing.peak_allocated_bytes, result.peak_allocated_bytes
                )
                existing.peak_reserved_bytes = max(
                    existing.peak_reserved_bytes, result.peak_reserved_bytes
                )
                existing.delta_peak_allocated_bytes = max(
                    existing.delta_peak_allocated_bytes, result.delta_peak_allocated_bytes
                )
                existing.delta_peak_reserved_bytes = max(
                    existing.delta_peak_reserved_bytes, result.delta_peak_reserved_bytes
                )
                existing.peak_rss_bytes = max(existing.peak_rss_bytes, result.peak_rss_bytes)
                existing.call_count += 1
            else:
                self._current_step.phases[name] = result
            self._phase_stack.pop()
            if self._phase_stack:
                parent = self._phase_stack[-1]
                parent["had_nested"] = True
                parent["manual_peak_allocated"] = max(
                    parent["manual_peak_allocated"], result.peak_allocated_bytes, end_alloc
                )
                parent["manual_peak_reserved"] = max(
                    parent["manual_peak_reserved"], result.peak_reserved_bytes, end_res
                )
                parent["manual_peak_rss"] = max(
                    parent["manual_peak_rss"], result.peak_rss_bytes, end_rss
                )
            self._step_manual_peak_allocated = max(
                self._step_manual_peak_allocated, result.peak_allocated_bytes, end_alloc
            )
            self._step_manual_peak_reserved = max(
                self._step_manual_peak_reserved, result.peak_reserved_bytes, end_res
            )
            self._step_manual_peak_rss = max(
                self._step_manual_peak_rss, result.peak_rss_bytes, end_rss
            )

    def _write_step(self) -> None:
        """Write current step profile to output file."""
        if not self.out_path or not self._current_step:
            return
        step_dict = {
            "type": "step",
            "step": self._current_step.step,
            "device": self._current_step.device,
            "mode": self._current_step.mode,
            "global": {"time_ms": self._current_step.time_ms},
            "phases": {},
        }
        if self._current_step.mode == "cuda":
            step_dict["global"]["peak_allocated_bytes"] = (
                self._current_step.global_peak_allocated_bytes
            )
            step_dict["global"]["peak_reserved_bytes"] = (
                self._current_step.global_peak_reserved_bytes
            )
        else:
            step_dict["global"]["peak_rss_bytes"] = self._current_step.global_peak_rss_bytes
        for name, phase in self._current_step.phases.items():
            phase_dict = asdict(phase)
            if self._current_step.mode == "cuda":
                for k in ["start_rss_bytes", "peak_rss_bytes", "end_rss_bytes"]:
                    phase_dict.pop(k, None)
            else:
                for k in [
                    "start_allocated_bytes",
                    "start_reserved_bytes",
                    "peak_allocated_bytes",
                    "peak_reserved_bytes",
                    "end_allocated_bytes",
                    "end_reserved_bytes",
                    "delta_peak_allocated_bytes",
                    "delta_peak_reserved_bytes",
                ]:
                    phase_dict.pop(k, None)
            step_dict["phases"][name] = phase_dict
        with open(self.out_path, "a") as f:
            f.write(json.dumps(step_dict) + "\n")

    def get_last_step(self) -> Optional[StepProfile]:
        """Get the last completed step profile (for testing)."""
        return self.last_step


@contextlib.contextmanager
def CudaProfile(section: str):
    """
    Measures peak CUDA memory and elapsed time for a code block.
    Returns a dict with section name, peaks, and elapsed_ms.
    """
    enabled = torch.cuda.is_available()
    if not enabled:
        import warnings

        warnings.warn(
            f"CudaProfile '{section}' called but CUDA is unavailable. Memory peaks will report zeros; use a CPU profiler if CPU metrics are needed.",
            UserWarning,
        )
    if enabled:
        cuda_sync()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        yield
    finally:
        if enabled:
            cuda_sync()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        peaks = read_peaks()
        CudaProfile.last_result = {"section": section, "elapsed_ms": elapsed_ms, **peaks}
        if not enabled:
            CudaProfile.last_result["note"] = "cuda_unavailable"


CudaProfile.last_result: Optional[Dict[str, float]] = None
