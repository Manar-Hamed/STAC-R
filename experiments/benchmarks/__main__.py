"""Run a named experiment or print every planned measurement without loading data."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import inspect
import io
import json
import os
from pathlib import Path
import sys
import uuid

from .registry import EXPERIMENTS


SPECIAL_CELLS = {
    "bert_sst2": ("run",),
    "lora": ("run_lora",),
    "qlora": ("run",),
    "allocator_stability": ("trace",),
    "single_model_fidelity": ("run",),
    "large_model_fidelity": ("single_run",),
    "peak_anatomy": ("phase_run", "ce_run"),
    "distributed_data": ("one_run",),
    "distributed_sharding": ("one_run",),
}

SPECIAL_OVERRIDES = {
    "bert_sst2": {"steps", "bs"},
    "lora": {"steps", "bs", "seq", "model_id"},
    "qlora": {"steps", "bs", "seq"},
    "allocator_stability": {"steps", "bs", "seq", "model_id"},
    "single_model_fidelity": {"steps", "bs", "seq", "model_id"},
    "large_model_fidelity": {"steps", "bs", "seq", "model_id"},
    "distributed_data": {"steps", "bs", "seq", "model_id", "measure_from"},
    "distributed_sharding": {"steps", "bs", "seq", "model_id", "measure_from"},
    "peak_anatomy": set(),
    "subtrack_composition": {"steps", "bs", "seq", "model_id"},
    "instant_comparison": {"steps", "bs", "seq", "model_id"},
}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if callable(value):
        return {"selected_block_indices_0_to_127": [i for i in range(128) if value(i)]}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class _PlanResult(dict):
    def __missing__(self, key):
        return 1.0


def _fake_result(parameters):
    result = _PlanResult(parameters)
    result.update(
        peak_bytes=1024**3,
        peak_GiB=1.0,
        time_s=1.0,
        val_ppl=1.0,
        n_wrapped_layers=1,
        n_checkpointed_blocks=0,
        attn_impl="planned",
        peak_phase="planned",
        mean_cosine_local_wrapped=1.0,
        steps_measured=max(1, parameters.get("steps", 1) - parameters.get("measure_from", 0)),
        ssp={"SSP_pct": 0.0},
        peak_GiB_all_ranks=[1.0],
    )
    return result


def _change_seed_lists(module, seeds):
    changed = False
    if hasattr(module, "SEEDS"):
        module.SEEDS = seeds
        changed = True
    if hasattr(module, "SEED") and len(seeds) == 1:
        module.SEED = seeds[0]
        changed = True
    cohorts = getattr(module, "COHORTS", {})
    for spec in cohorts.values() if isinstance(cohorts, dict) else cohorts:
        if isinstance(spec, dict) and "seeds" in spec:
            spec["seeds"] = seeds
            changed = True
    return changed


def _special_parameters(module, signature, args, kwargs):
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    params = dict(bound.arguments)
    for constant, key in (("MODEL", "model_id"), ("BS", "bs"), ("SEQ", "seq"), ("STEPS", "steps")):
        if hasattr(module, constant):
            params.setdefault(key, getattr(module, constant))
    if "spec" in params:
        params.update({k: v for k, v in params.pop("spec").items() if k != "arms"})
    if hasattr(module, "MEASURE_FROM"):
        params.setdefault("measure_from", module.MEASURE_FROM)
    params.setdefault(
        "optimizer", "adamw" if module.__name__.endswith("distributed_sharding") else "adamw8bit"
    )
    params.setdefault("dataset", "wikitext")
    return params


def _plan_distributed(module, name, selectors, capture):
    if name == "distributed_data":
        if selectors:
            raise ValueError("distributed_data has no cohort selectors")
        for method, options in module.ARMS:
            for seed in module.SEEDS:
                capture(module.one_run, method, seed, **options)
    else:
        selected = selectors or list(module.COHORTS)
        for name in selected:
            if name not in module.COHORTS:
                raise ValueError(f"Unknown cohort {name!r}; choose {list(module.COHORTS)}")
            spec = module.COHORTS[name]
            for method, options in module.arm_filter(spec["arms"]):
                for seed in spec["seeds"]:
                    capture(module.one_run, method, seed, spec, **options)


def _plan(module, name, selectors):
    """Execute only grid construction; replace every expensive cell with a recorder."""
    from unittest.mock import patch
    from . import common

    cells = []
    if hasattr(module, "enumerate_cells"):
        return [
            {"experiment": name, "cell": i + 1, "parameters": cell}
            for i, cell in enumerate(module.enumerate_cells(selectors))
        ]

    def record(function, *args, **kwargs):
        params = _special_parameters(module, inspect.signature(function), args, kwargs)
        params = common.resolve_parameters(params)
        cells.append(
            {
                "experiment": name,
                "cell": len(cells) + 1,
                "function": function.__name__,
                "parameters": _jsonable(params),
            }
        )
        return _fake_result(params)

    if name in ("distributed_data", "distributed_sharding"):
        _plan_distributed(module, name, selectors, record)
        return cells

    def emitted(result, path):
        if result.get("status", "ok") not in ("ok",):
            raise RuntimeError(f"Could not build complete plan: {result}")

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(module, "emit", emitted))
        for attr in ("run_config", "gradient_fidelity", *SPECIAL_CELLS.get(name, ())):
            if hasattr(module, attr):
                fn = getattr(module, attr)
                stack.enter_context(
                    patch.object(module, attr, lambda *a, _fn=fn, **kw: record(_fn, *a, **kw))
                )
        if name == "rate_distortion":
            stack.enter_context(patch.object(module, "payload_bytes", lambda *args: (1, 1, 1)))
        if name == "swiglu_surfaces":

            def cfg_for(model_id, surface, config, rank):
                cfg = dict(
                    config, rank=rank, residual_budget=0.005, residual_bits=8, chunk_rows=512
                )
                if cfg.pop("_down_only", False):
                    cfg["extra_names"] = ["<all model MLP down_proj modules; resolved at run time>"]
                return cfg

            stack.enter_context(patch.object(module, "cfg_for", cfg_for))
        with contextlib.redirect_stdout(io.StringIO()):
            if name == "hardware_runtime":
                stack.enter_context(patch.object(module, "gpu_gib", lambda: 80.0))
                module.phase1_runtime(Path("planned"))
                module.phase2_scale(Path("planned"))
                for cell in cells:
                    cell["planning_gpu_capacity_GiB"] = 80
            else:
                module.main()
    if not cells:
        raise ValueError("No measurement cells matched the requested selectors")
    return cells


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list", help="List available scientific experiments")
    listing.add_argument("--json", action="store_true")
    run = sub.add_parser("run", help="Run one experiment; --dry-run prints its full cell plan")
    run.add_argument("experiment", choices=sorted(EXPERIMENTS))
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--plan-output", type=Path)
    run.add_argument("--output", type=Path)
    run.add_argument(
        "--append",
        action="store_true",
        help="Append a distinct run ID to an existing output directory",
    )
    run.add_argument("--steps", type=int)
    run.add_argument("--seeds", type=int, nargs="+")
    run.add_argument("--model", dest="model_id")
    run.add_argument("--batch-size", dest="bs", type=int)
    run.add_argument("--seq-length", dest="seq", type=int)
    run.add_argument("--eval-batches", type=int)
    run.add_argument("--measure-from", type=int)
    run.add_argument("--device", choices=("cuda", "cpu"))
    run.add_argument("--optimizer", choices=("adamw8bit", "adamw"))
    run.add_argument("--dataset", choices=("wikitext", "openwebtext"))
    run.add_argument(
        "--validation-protocol",
        choices=("disjoint", "shared-prefix"),
        help="shared-prefix reproduces overlapping OpenWebText train/evaluation blocks",
    )
    run.add_argument(
        "--attention", dest="attn_impl", choices=("eager", "sdpa", "flash_attention_2")
    )
    run.add_argument(
        "--cohort",
        action="append",
        default=[],
        help="Experiment-specific cohort or mode; repeat for multiple selectors",
    )
    args = parser.parse_args(argv)
    if args.command == "list":
        if args.json:
            print(json.dumps(EXPERIMENTS, indent=2))
        else:
            for name, meta in sorted(EXPERIMENTS.items()):
                print(f"{name:27s} {meta['description']}")
        return 0

    from . import common

    name = args.experiment
    standard = EXPERIMENTS[name]["standard"] and name not in SPECIAL_CELLS
    override_keys = (
        "steps",
        "model_id",
        "bs",
        "seq",
        "eval_batches",
        "measure_from",
        "device",
        "dataset",
        "validation_protocol",
        "attn_impl",
    )
    overrides = {key: getattr(args, key) for key in override_keys if getattr(args, key) is not None}
    if args.optimizer:
        overrides["eight_bit"] = args.optimizer == "adamw8bit"
    unsupported = set(overrides) - SPECIAL_OVERRIDES.get(name, set())
    if unsupported and not standard:
        parser.error(
            f"{name} uses a specialized protocol and does not support these shared overrides: "
            f"{', '.join(sorted(unsupported))}. Select a cohort; its full settings are visible with --dry-run."
        )
    output = args.output or Path("outputs") / "benchmarks" / f"{name}-{uuid.uuid4().hex[:10]}"
    if not args.dry_run and output.exists() and any(output.glob("*.jsonl")) and not args.append:
        parser.error(
            f"{output} already contains results; choose a new --output or explicitly use --append"
        )
    common.configure(output=output, seeds=args.seeds, **overrides)
    module = importlib.import_module(f"{__package__}.{name}")
    module = importlib.reload(module)
    module.RESULTS = output
    for constant, key in (
        ("MODEL", "model_id"),
        ("BS", "bs"),
        ("SEQ", "seq"),
        ("STEPS", "steps"),
        ("MEASURE_FROM", "measure_from"),
    ):
        if key in overrides and hasattr(module, constant):
            setattr(module, constant, overrides[key])
    if "steps" in overrides and hasattr(module, "MEASURE_FROM") and "measure_from" not in overrides:
        module.MEASURE_FROM = min(module.MEASURE_FROM, max(0, overrides["steps"] - 1))
    if args.seeds and not _change_seed_lists(module, args.seeds):
        parser.error(f"{name} does not expose a seed list; its seed is fixed")
    source = Path(module.__file__).read_text(encoding="utf-8")
    if args.cohort and "sys.argv" not in source:
        parser.error(f"{name} has no cohort selector")
    old_argv = sys.argv
    sys.argv = [f"python -m experiments.benchmarks run {name}", *args.cohort]
    try:
        plan = _plan(module, name, args.cohort)
        if args.dry_run:
            payload = {"experiment": name, "cell_count": len(plan), "cells": plan}
            formatted = json.dumps(payload, indent=2)
            if args.plan_output:
                args.plan_output.parent.mkdir(parents=True, exist_ok=True)
                args.plan_output.write_text(formatted + "\n", encoding="utf-8")
                print(f"{len(plan)} cells: {args.plan_output}")
            else:
                print(formatted)
            return 0
        output.mkdir(parents=True, exist_ok=True)
        (output / f"plan-{common.RUN_ID}.json").write_text(
            json.dumps({"experiment": name, "cell_count": len(plan), "cells": plan}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        failures = []
        for attr in SPECIAL_CELLS.get(name, ()):
            function = getattr(module, attr)
            signature = inspect.signature(function)

            def measured(*a, _function=function, _signature=signature, **kw):
                bound = _signature.bind(*a, **kw)
                bound.apply_defaults()
                for key, value in overrides.items():
                    if key in bound.arguments:
                        bound.arguments[key] = value
                if "spec" in bound.arguments:
                    spec = dict(bound.arguments["spec"])
                    spec.update({key: value for key, value in overrides.items() if key in spec})
                    if "steps" in overrides and "measure_from" not in overrides:
                        spec["measure_from"] = min(spec["measure_from"], max(0, spec["steps"] - 1))
                    bound.arguments["spec"] = spec
                protocol = _special_parameters(module, _signature, bound.args, bound.kwargs)
                result = _function(*bound.args, **bound.kwargs)
                result.setdefault("protocol", _jsonable(protocol))
                return result

            setattr(module, attr, measured)
        original_emit = module.emit

        def tracked_emit(record, path):
            if record.get("status", "ok") not in ("ok", "OOM", "OutOfMemoryError"):
                failures.append(record)
            return original_emit(record, path)

        module.emit = tracked_emit
        print(f"Experiment: {name}\nOutput: {output}\nRun ID: {common.RUN_ID}", flush=True)
        module.main()
        if failures:
            print(
                f"{len(failures)} cells failed; their error records are in {output}",
                file=sys.stderr,
            )
            return 1
        return 0
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
