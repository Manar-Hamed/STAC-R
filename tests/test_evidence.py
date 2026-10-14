import pytest

from scripts.cohorts import recompute
from scripts.evidence import DATA, load, select_records, summarize


def record(seed, peak, *, flags=(), ppl=10.0, seq=128):
    return {
        "record_id": f"{seed}-{peak}-{seq}",
        "platform": "a100",
        "experiments": ["checkpoint_frontier"],
        "flags": list(flags),
        "row": {
            "method": "baseline",
            "seed": seed,
            "peak_GiB": peak,
            "val_ppl": ppl,
            "seq": seq,
        },
    }


def test_summary_averages_repeated_runs_within_seed():
    summary = summarize([record(1, 10), record(1, 30), record(2, 50)])[0]
    metric = summary["metrics"]["peak_GiB"]
    assert metric["mean"] == 35
    assert metric["std"] == pytest.approx(30 / 2**0.5)
    assert metric["n_runs"] == 3
    assert metric["n_independent_seeds_or_unseeded_runs"] == 2


def test_quality_flags_do_not_remove_valid_memory_measurements():
    rows = [
        record(1, 10, flags=["quality_overlapping_training_prefix"], ppl=999),
        record(2, 20, flags=["failed_execution"], ppl=999),
        record(3, 30, ppl=12),
    ]
    summary = summarize(rows)[0]
    assert summary["metrics"]["peak_GiB"]["mean"] == 20
    assert summary["metrics"]["val_ppl"]["mean"] == 12
    assert summary["flags"]["quality_overlapping_training_prefix"] == 1
    included = summarize(rows, include_flagged=True)[0]
    assert included["metrics"]["val_ppl"]["n_runs"] == 2


def test_shapes_and_platforms_are_separate_groups():
    rows = [record(1, 10), record(2, 20, seq=256), record(3, 30)]
    rows[-1]["platform"] = "h100"
    assert len(summarize(rows)) == 3
    assert len(select_records(rows, "checkpoint_frontier", "a100")) == 2
    with pytest.raises(ValueError, match="Unknown experiment"):
        select_records(rows, "missing")


def test_packaged_cohorts_recompute_published_values():
    result = recompute(DATA)
    assert len(result["cohorts"]) == 7
    assert len(result["published_cell_checks"]) == 66
    assert result["all_checks_passed"]
    assert len(load()) == 3144


def test_timing_exclusion_preserves_quality_seed_count():
    result = recompute(DATA, "pythia_10k")["cohorts"]["pythia_10k"]
    for method in result["methods"].values():
        assert method["metrics"]["ppl"]["n"] == 5
        assert method["metrics"]["time_s"]["n"] == 4
    for pair in result["paired_comparisons"]:
        assert len(pair["paired_seeds"]) == 5
        assert pair["metrics"]["time_s_ratio"]["n"] == 4


def test_unknown_cohort_fails_explicitly():
    with pytest.raises(ValueError, match="Unknown recomputable cohort"):
        recompute(DATA, "missing")
