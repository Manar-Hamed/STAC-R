from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def default_data_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data-dir", default=str(default_data_dir()))
    p.add_argument("--output-dir", default="generated_figures")
    p.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    return p


def read_csv(data_dir: str | Path, name: str) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / name)


def output_path(output_dir: str | Path, stem: str, suffix: str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{stem}.{suffix}"


def clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.7)
