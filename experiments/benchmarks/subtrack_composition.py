"""BERT/SST-2 with the upstream SubTrack++ optimizer and STAC-R composition."""

from __future__ import annotations

import json
import subprocess
import sys

from . import common
from .common import RESULTS, emit
from comparators.subtrack_adapter import SUBTRACK_COMMIT, require_pinned_subtrack

SEEDS = [42, 123, 456]
COHORTS = {
    "bert_base": ("bert-base-uncased", ("baseline", "subtrack", "subtrack_stacr")),
    "bert_large": ("bert-large-uncased", ("subtrack_stacr",)),
}


def enumerate_cells(selectors=None):
    cells = []
    for cohort in selectors or list(COHORTS):
        if cohort not in COHORTS:
            raise ValueError(f"Unknown cohort {cohort!r}; choose {list(COHORTS)}")
        model, arms = COHORTS[cohort]
        seeds = common.OPTIONS.get("seeds") or (SEEDS if cohort == "bert_base" else SEEDS[:2])
        for method in arms:
            for seed in seeds:
                output = RESULTS / f"{cohort}_{method}_seed{seed}"
                command = [
                    sys.executable,
                    "-m",
                    "comparators.subtrack_protocol",
                    "--model_name_or_path",
                    common.OPTIONS.get("model_id", model),
                    "--task_name",
                    "sst2",
                    "--max_length",
                    str(common.OPTIONS.get("seq", 128)),
                    "--seed",
                    str(seed),
                    "--per_device_train_batch_size",
                    str(common.OPTIONS.get("bs", 32)),
                    "--learning_rate",
                    "2e-5" if method == "baseline" else "1e-5",
                    "--num_train_epochs",
                    "3",
                    "--disable_wandb",
                    "--output_dir",
                    str(output),
                ]
                if "steps" in common.OPTIONS:
                    command += ["--max_train_steps", str(common.OPTIONS["steps"])]
                if method != "baseline":
                    command += [
                        "--enable_low_rank",
                        "--lora_all_modules",
                        "--lora_r",
                        "4",
                        "--low_rank_scale",
                        "4",
                        "--subspace_update_interval",
                        "500",
                        "--subspace_update_method",
                        "subtrack",
                        "--st_init_step_size",
                        "0.001",
                        "--adaptive_optimizer",
                        "--recovery_scaling",
                    ]
                if method == "subtrack_stacr":
                    command += [
                        "--enable_stacr",
                        "--stacr_rank",
                        "8",
                        "--stacr_residual_budget",
                        "0.005",
                    ]
                cells.append(
                    {
                        "cohort": cohort,
                        "method": method,
                        "seed": seed,
                        "command": command,
                        "output": str(output),
                        "upstream_commit": SUBTRACK_COMMIT,
                    }
                )
    return cells


def main():
    require_pinned_subtrack()
    for cell in enumerate_cells(sys.argv[1:]):
        subprocess.run(cell["command"], check=True)
        summary_path = common.Path(cell["output"]) / "run_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        emit(dict(cell, status="ok", result=summary), RESULTS / "subtrack_composition.jsonl")


if __name__ == "__main__":
    main()
