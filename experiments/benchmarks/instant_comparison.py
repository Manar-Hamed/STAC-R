"""INSTANT classification and causal-language-model protocols."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from . import common
from .common import RESULTS, emit
from comparators.instant import REVISION

SEEDS = [42, 123, 456]
COHORTS = {
    "sst2": {
        "model": "bert-base-uncased",
        "task": "sst2",
        "bs": 32,
        "seq": 128,
        "epochs": 3,
        "steps": -1,
        "seeds": SEEDS,
        "oversampling": [None, 15, 25],
    },
    "mrpc": {
        "model": "bert-base-uncased",
        "task": "mrpc",
        "bs": 32,
        "seq": 128,
        "epochs": 3,
        "steps": -1,
        "seeds": SEEDS,
        "oversampling": [None, 15, 25],
    },
    "gpt2_large": {
        "model": "gpt2-large",
        "task": "wikitext103",
        "bs": 4,
        "seq": 512,
        "epochs": 1,
        "steps": 2000,
        "seeds": [42, 123],
        "oversampling": [None, 15, 25],
    },
    "gpt2_medium": {
        "model": "gpt2-medium",
        "task": "wikitext103",
        "bs": 2,
        "seq": 512,
        "epochs": 1,
        "steps": 500,
        "seeds": [42],
        "oversampling": [None, 15],
        "calib_batches": 5,
    },
    "gpt2_medium_smoke": {
        "model": "gpt2-medium",
        "task": "wikitext103",
        "bs": 2,
        "seq": 512,
        "epochs": 1,
        "steps": 200,
        "seeds": [42],
        "oversampling": [None, 15],
        "calib_batches": 2,
    },
}


def enumerate_cells(selectors=None):
    cells = []
    for cohort in selectors or list(COHORTS):
        if cohort not in COHORTS:
            raise ValueError(f"Unknown cohort {cohort!r}; choose {list(COHORTS)}")
        spec = COHORTS[cohort]
        causal = spec["task"] == "wikitext103"
        module = "train_language" if causal else "train_classification"
        for oversampling in spec["oversampling"]:
            method = "baseline" if oversampling is None else f"instant_ov{oversampling}"
            for seed in spec["seeds"]:
                output = RESULTS / f"{cohort}_{method}_seed{seed}"
                command = [
                    sys.executable,
                    "-m",
                    f"comparators.instant.{module}",
                    "--model_name",
                    common.OPTIONS.get("model_id", spec["model"]),
                    "--dataset",
                    spec["task"],
                    "--seed",
                    str(seed),
                    "--batch-size",
                    str(common.OPTIONS.get("bs", spec["bs"])),
                    "--max-length",
                    str(common.OPTIONS.get("seq", spec["seq"])),
                    "--epochs",
                    str(spec["epochs"]),
                    "--learning-rate",
                    "2e-5",
                    "--n-last-layers",
                    "0",
                    "--max-steps",
                    str(common.OPTIONS.get("steps", spec["steps"])),
                    "--output_dir",
                    str(output),
                ]
                if oversampling is not None:
                    command += [
                        "--compress",
                        "--over_sampling",
                        str(oversampling),
                        "--var",
                        "0.95",
                        "--calib_batches",
                        str(spec.get("calib_batches", 5)),
                        "--calib_iter",
                        "50",
                    ]
                cells.append(
                    {
                        "cohort": cohort,
                        "method": method,
                        "seed": seed,
                        "command": command,
                        "output": str(output),
                        "upstream_commit": REVISION,
                    }
                )
    return cells


def main():
    for cell in enumerate_cells(sys.argv[1:]):
        subprocess.run(cell["command"], check=True)
        summaries = sorted(Path(cell["output"]).rglob("final.json"))
        if len(summaries) != 1:
            raise RuntimeError(
                f"Expected one final.json under {cell['output']}; found {len(summaries)}"
            )
        summary = json.loads(summaries[0].read_text(encoding="utf-8"))
        emit(dict(cell, status="ok", result=summary), RESULTS / "instant_comparison.jsonl")


if __name__ == "__main__":
    main()
