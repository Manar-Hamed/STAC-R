"""Summarize STAC-R measurements and recompute published numerical cohorts."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "paper_data"


EXCLUDED = {
    "failed_execution",
    "smoke_only",
    "superseded_basis_implementation",
    "superseded_rng_unpinned_fidelity",
    "incomplete_training_stream",
    "diagnostic_probe",
    "environment_record",
}

QUALITY_EXCLUDED = {
    "quality_quarantined_seed_instability",
    "quality_not_evaluated_architecture_probe",
    "quality_requires_underupdate_control",
    "short_horizon_quality",
    "quality_overlapping_training_prefix",
    "quality_invalid_allocation_only_backward",
    "quality_random_label_memory_probe",
}

METRICS = (
    "peak_GiB",
    "time_s",
    "val_ppl",
    "accuracy",
    "mean_cosine_full_model",
    "mean_cosine_local_wrapped",
    "worst_unwrapped_cosine",
    "stored_MiB",
    "tokens_per_s",
)

PROTOCOL = (
    "tag",
    "cohort",
    "model",
    "bs",
    "bs_per_rank",
    "seq",
    "steps",
    "steps_ran",
    "steps_measured",
    "world_size",
    "dtype",
    "amp",
    "eight_bit",
    "dataset",
    "attn_impl",
    "attn_requested",
    "optimizer_actual",
    "optimizer_state_dtype",
    "alloc_conf",
)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def read_jsonl(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(
                    line,
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError("Non-finite JSON: " + value)
                    ),
                )


def load(data=DATA):
    return list(read_jsonl(Path(data) / "results" / "records.jsonl"))


def select_records(records, experiment=None, platform=None):
    available = {name for record in records for name in record["experiments"]}
    if experiment is not None and experiment not in available:
        raise ValueError("Unknown experiment: " + experiment)
    return [
        record
        for record in records
        if (platform is None or record["platform"] == platform)
        and (experiment is None or experiment in record["experiments"])
    ]


def summarize(records, experiment=None, include_flagged=False):
    groups = defaultdict(list)
    for record in records:
        row = record["row"]
        protocol = {key: row[key] for key in PROTOCOL if key in row}
        # Compression settings must remain distinct even if a run reused its name.
        method = row.get("method", row.get("variant", row.get("comparator", "unknown")))
        method_config = {
            key: row[key]
            for key in (
                "stacr_cfg",
                "rank",
                "rank_used",
                "residual_budget",
                "selector",
                "n_checkpointed_blocks",
                "n_wrapped_layers",
                "n_offloaded_blocks",
                "n_frozen_params",
                "ckpt_mode",
            )
            if key in row
        }
        series = experiment or record["experiments"][0]
        key = canonical([record["platform"], series, protocol, method, method_config])
        groups[key].append(record)
    results = []
    for key, group in sorted(groups.items()):
        platform, series, protocol, method, method_config = json.loads(key)
        flags = Counter(flag for record in group for flag in record["flags"])
        out = {
            "platform": platform,
            "experiment": series,
            "protocol": protocol,
            "method": method,
            "method_config": method_config,
            "records": len(group),
            "seeds": sorted(
                {r["row"]["seed"] for r in group if isinstance(r["row"].get("seed"), int)}
            ),
            "flags": dict(flags),
            "metrics": {},
        }
        for metric in METRICS:
            by_seed = defaultdict(list)
            for record in group:
                row, validity = record["row"], set(record["flags"])
                if validity & {"failed_execution", "environment_record"}:
                    continue
                if not include_flagged and (
                    validity & EXCLUDED
                    or (metric in {"val_ppl", "accuracy"} and validity & QUALITY_EXCLUDED)
                ):
                    continue
                value = row.get(metric)
                if isinstance(value, (float, int)) and not isinstance(value, bool):
                    # Repeated executions of a seed do not inflate the seed count.
                    seed = row.get("seed", record["record_id"])
                    by_seed[seed].append(float(value))
            values = [statistics.mean(samples) for samples in by_seed.values()]
            if values:
                out["metrics"][metric] = {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else None,
                    "n_independent_seeds_or_unseeded_runs": len(values),
                    "n_runs": sum(len(samples) for samples in by_seed.values()),
                    "min": min(values),
                    "max": max(values),
                }
        results.append(out)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List experiment names and measurement counts")
    cohorts = commands.add_parser(
        "cohorts", help="Recompute cohort statistics and rounded table values"
    )
    cohorts.add_argument("--cohort", help="Limit to one cohort, such as bert_sst2 or pythia_10k")
    cohorts.add_argument("--output", type=Path)
    summary = commands.add_parser(
        "summary", help="Aggregate runs with the same hardware and protocol"
    )
    summary.add_argument("--experiment")
    summary.add_argument("--platform", choices=("rtx5090", "rtx5090_cluster", "a100", "h100"))
    summary.add_argument(
        "--include-flagged",
        action="store_true",
        help="Include diagnostic or excluded measurements; retain their flags",
    )
    summary.add_argument(
        "--output", type=Path, help="Write JSON or CSV according to the file extension"
    )
    args = parser.parse_args()
    if args.command == "cohorts":
        if __package__:
            from .cohorts import recompute
        else:
            from cohorts import recompute
        result = recompute(args.data, args.cohort)
    else:
        records = load(args.data)
        if args.command == "list":
            counts = Counter(name for record in records for name in record["experiments"])
            result = [
                {"experiment": name, "records": count} for name, count in sorted(counts.items())
            ]
        else:
            result = summarize(
                select_records(records, args.experiment, args.platform),
                args.experiment,
                args.include_flagged,
            )
    if getattr(args, "output", None):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix.lower() == ".csv" and args.command == "summary":
            rows = []
            for row in result:
                out = {
                    key: canonical(value) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                    if key != "metrics"
                }
                for metric, stats in row["metrics"].items():
                    out.update({metric + "_" + key: value for key, value in stats.items()})
                rows.append(out)
            fields = sorted({key for row in rows for key in row})
            with args.output.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        else:
            args.output.write_text(
                json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        print(str(args.output))
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    if args.command == "cohorts" and not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
