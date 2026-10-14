from __future__ import annotations

import csv
import random
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml


class _OverrideLoader(yaml.SafeLoader):
    # Copy the resolver lists: changing SafeLoader itself would alter file parsing
    # and other libraries that use PyYAML in the same process.
    yaml_implicit_resolvers = {
        initial: [(tag, pattern) for tag, pattern in resolvers if tag != "tag:yaml.org,2002:bool"]
        for initial, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
    }


_OverrideLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool", re.compile(r"^(?:true|false)$", re.IGNORECASE), list("tTfF")
)


def parse_override(value: str):
    """Parse a CLI value, treating only true/false as implicit booleans.

    Enum values such as on/off and yes/no stay strings, including inside lists
    and mappings. Numbers, nulls and explicit quoted strings retain YAML parsing.
    """
    return yaml.load(value, Loader=_OverrideLoader)


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        result = yaml.safe_load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"Configuration {path} must be a YAML mapping")
    return result


def write_csv(path: str | Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row))
        )
        writer.writeheader()
        writer.writerows(rows)


def device_from_config(value: str = "auto") -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    return device
