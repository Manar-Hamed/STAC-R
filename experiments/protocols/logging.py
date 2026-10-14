from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class TrainConfig:
    epochs: int = 3
    batch_size: int = 128
    lr: float = 0.01
    momentum: float = 0.9
    num_workers: int = 2


def _json_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def _jsonl_append(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(payload, default=str) + "\n")


class RunLogger:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = run_dir / "config.json"
        self.metrics_path = run_dir / "metrics.jsonl"
        self.profile_path = run_dir / "profile.jsonl"
        self.final_path = run_dir / "final.json"
        self.metrics_path.touch(exist_ok=True)
        self.profile_path.touch(exist_ok=True)

    def write_config(self, payload: Dict[str, Any]) -> None:
        _json_dump(self.config_path, payload)

    def log_metrics(self, payload: Dict[str, Any]) -> None:
        _jsonl_append(self.metrics_path, payload)

    def log_profile(self, payload: Dict[str, Any]) -> None:
        _jsonl_append(self.profile_path, payload)

    def write_final(self, payload: Dict[str, Any]) -> None:
        _json_dump(self.final_path, payload)


class NullLogger:
    """No-op logger for DDP non-main ranks. Silently drops all writes."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.config_path = run_dir / "config.json"
        self.metrics_path = run_dir / "metrics.jsonl"
        self.profile_path = run_dir / "profile.jsonl"
        self.final_path = run_dir / "final.json"

    def write_config(self, payload: Dict[str, Any]) -> None:
        pass

    def log_metrics(self, payload: Dict[str, Any]) -> None:
        pass

    def log_profile(self, payload: Dict[str, Any]) -> None:
        pass

    def write_final(self, payload: Dict[str, Any]) -> None:
        pass
