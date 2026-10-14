"""Recompute seed means, sample standard deviations and paired per-seed ratios."""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def _stats(values):
    values = list(values)
    return (
        {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else None,
            "n": len(values),
        }
        if values
        else None
    )


def _numeric(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def recompute(data, selected=None):
    groups = defaultdict(list)
    with (Path(data) / "results" / "cohorts.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sample = {
                "method": row["method"],
                "seed": int(row["seed"]),
                "timing_excluded": row["timing_excluded"].lower() == "true",
            }
            for metric in ("accuracy", "ppl", "peak_gib", "time_s", "tokens_per_s", "cosine"):
                value = _numeric(row[metric])
                if value is not None:
                    sample[metric] = value
            groups[row["cohort"]].append(sample)

    def method_summary(samples):
        by_method = defaultdict(list)
        for row in samples:
            by_method[row["method"]].append(row)
        result = {}
        for method, sample in by_method.items():
            stats = {}
            for metric in ("accuracy", "ppl", "peak_gib", "time_s", "tokens_per_s", "cosine"):
                per_seed = defaultdict(list)
                for row in sample:
                    if metric == "time_s" and row.get("timing_excluded"):
                        continue
                    value = row.get(metric)
                    if value is not None:
                        per_seed[row["seed"]].append(value)
                value = _stats(statistics.mean(v) for v in per_seed.values())
                if value:
                    stats[metric] = value
            result[method] = {
                "seeds": sorted({r["seed"] for r in sample}),
                "metrics": stats,
            }
        return result

    def paired(samples, reference, method):
        by_method = defaultdict(lambda: defaultdict(list))
        for row in samples:
            by_method[row["method"]][row["seed"]].append(row)
        seeds = sorted(set(by_method[reference]) & set(by_method[method]))
        collected = defaultdict(list)
        for seed in seeds:
            base_rows, sample_rows = by_method[reference][seed], by_method[method][seed]
            for metric in ("accuracy", "ppl", "peak_gib", "time_s", "tokens_per_s"):
                if metric == "time_s" and any(
                    r.get("timing_excluded") for r in base_rows + sample_rows
                ):
                    continue
                base = [r[metric] for r in base_rows if r.get(metric) is not None]
                sample = [r[metric] for r in sample_rows if r.get(metric) is not None]
                if base and sample:
                    b, value = statistics.mean(base), statistics.mean(sample)
                    collected["reference_" + metric].append(b)
                    collected["method_" + metric].append(value)
                    collected[metric + "_difference"].append(value - b)
                    if b:
                        collected[metric + "_ratio"].append(value / b)
                        collected[metric + "_delta_pct"].append(100 * (value / b - 1))
        return {
            "reference": reference,
            "method": method,
            "paired_seeds": seeds,
            "metrics": {key: _stats(values) for key, values in collected.items()},
        }

    output = {}
    for name, samples in groups.items():
        if selected and name != selected:
            continue
        methods = method_summary(samples)
        pairs = (
            [("baseline", name) for name in methods if name != "baseline"]
            if "baseline" in methods
            else []
        )
        if name == "galore_composition":
            pairs += [("galore_alone", "stacr_galore"), ("galore_alone", "turbofast_galore")]
        if name == "matched_bytes":
            pairs = [
                (family + "_none_r" + str(rank), family + "_topk")
                for family, rank in [
                    ("bert_base_3ep", 139),
                    ("bert_large_3ep", 182),
                    ("gpt2_large_10k", 144),
                    ("gpt2_xl_5k", 178),
                ]
            ]
        output[name] = {
            "methods": methods,
            "paired_comparisons": [
                paired(samples, reference, method) for reference, method in pairs
            ],
        }

    checks = []

    def check(cohort, table, row_label, column, observed):
        with (data / table).open(encoding="utf-8-sig", newline="") as handle:
            source = list(csv.DictReader(handle))
        row = next(r for r in source if next(iter(r.values())) == row_label)
        value = row[column]
        numeric_text = value.split()[0]
        expected = float(numeric_text)
        decimals = len(numeric_text.partition(".")[2]) if "." in numeric_text else 0
        tolerance = 0.500001 * 10**-decimals
        checks.append(
            {
                "cohort": cohort,
                "table": table,
                "row": row_label,
                "column": column,
                "expected": expected,
                "computed": observed,
                "rounding_tolerance": tolerance,
                "passed": observed is not None and abs(observed - expected) <= tolerance,
            }
        )

    if "pythia_10k" in output:
        result = output["pythia_10k"]
        for method, label in [
            ("baseline", "10k baseline 5-seed"),
            ("stacr", "10k STAC-R 5-seed"),
            ("turbo_fast", "10k Turbo-fast 5-seed"),
        ]:
            check(
                "pythia_10k",
                "pythia_longhorizon.csv",
                label,
                "ppl",
                result["methods"][method]["metrics"]["ppl"]["mean"],
            )
            if method != "turbo_fast":
                check(
                    "pythia_10k",
                    "pythia_longhorizon.csv",
                    label,
                    "ppl_std",
                    result["methods"][method]["metrics"]["ppl"]["std"],
                )
    if "galore_composition" in output:
        result = output["galore_composition"]
        for method, label in [
            ("baseline", "baseline"),
            ("galore_alone", "GaLore"),
            ("stacr_galore", "STAC-R + GaLore"),
            ("turbofast_galore", "Turbo-fast + GaLore"),
        ]:
            for column, metric, statistic in [
                ("ppl_mean", "ppl", "mean"),
                ("ppl_std", "ppl", "std"),
                ("peak_gib", "peak_gib", "mean"),
                ("time_s", "time_s", "mean"),
            ]:
                check(
                    "galore_composition",
                    "composition.csv",
                    label,
                    column,
                    result["methods"][method]["metrics"][metric][statistic],
                )
        for pair in result["paired_comparisons"]:
            if pair["reference"] == "galore_alone":
                label = (
                    "STAC-R marginal over GaLore"
                    if pair["method"] == "stacr_galore"
                    else "Turbo-fast marginal over GaLore"
                )
                for column, metric, statistic in [
                    ("ppl_mean", "ppl_difference", "mean"),
                    ("ppl_std", "ppl_difference", "std"),
                    ("peak_gib", "peak_gib_difference", "mean"),
                    ("time_s", "time_s_difference", "mean"),
                    ("peak_delta_pct", "peak_gib_delta_pct", "mean"),
                ]:
                    check(
                        "galore_composition",
                        "composition.csv",
                        label,
                        column,
                        pair["metrics"][metric][statistic],
                    )
    if "bert_sst2" in output:
        pair = output["bert_sst2"]["paired_comparisons"][0]
        for column, metric in [
            ("memory_delta_pct", "peak_gib_delta_pct"),
            ("time_x", "time_s_ratio"),
            ("quality_delta", "accuracy_difference"),
        ]:
            check(
                "bert_sst2",
                "quality_cost_summary.csv",
                "BERT-base SST-2",
                column,
                pair["metrics"][metric]["mean"],
            )
    if "matched_bytes" in output:
        labels = {
            "bert_base_3ep_topk": "BERT-base SST-2",
            "bert_large_3ep_topk": "BERT-large SST-2",
            "gpt2_large_10k_topk": "GPT-2 Large WT103",
            "gpt2_xl_5k_topk": "GPT-2 XL WT103",
        }
        for pair in output["matched_bytes"]["paired_comparisons"]:
            label = labels[pair["method"]]
            for column, metric in [
                ("hybrid_tok_s", "method_tokens_per_s"),
                ("pure_lr_tok_s", "reference_tokens_per_s"),
                ("gain_pct", "tokens_per_s_delta_pct"),
            ]:
                check(
                    "matched_bytes",
                    "matched_budget_throughput.csv",
                    label,
                    column,
                    pair["metrics"][metric]["mean"],
                )
            check(
                "matched_bytes",
                "matched_budget_throughput.csv",
                label,
                "seeds",
                len(pair["paired_seeds"]),
            )
    if "residual_gamma" in output:
        result = output["residual_gamma"]
        derived = []
        for rank in (8, 32, 128, 307):
            on = result["methods"][f"rank{rank}_block_sparse"]["metrics"]["cosine"]["mean"]
            off = result["methods"][f"rank{rank}_none"]["metrics"]["cosine"]["mean"]
            epsilon_on, epsilon_off = (1 - on) / (1 + on), (1 - off) / (1 + off)
            row = {
                "rank": rank,
                "cosine_on": on,
                "cosine_off": off,
                "epsilon_on": epsilon_on,
                "gamma_epsilon": 1 - (epsilon_on / epsilon_off) ** 2,
            }
            derived.append(row)
            for column in ("cosine_on", "cosine_off", "epsilon_on", "gamma_epsilon"):
                check("residual_gamma", "residual_gamma.csv", str(rank), column, row[column])
        result["derived_rows"] = derived
    if selected and selected not in output:
        raise ValueError("Unknown recomputable cohort: " + selected)
    return {
        "cohorts": output,
        "published_cell_checks": checks,
        "all_checks_passed": all(c["passed"] for c in checks),
    }
