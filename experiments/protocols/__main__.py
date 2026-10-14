"""Run a named transformer training protocol."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from importlib.resources import files
from pathlib import Path


def profiles():
    return json.loads(files("configs").joinpath("protocols.json").read_text(encoding="utf-8"))[
        "profiles"
    ]


def make_config(raw):
    from .training import ConvergenceConfig
    from stacr.protocols.config.training import BaselineConfig

    allowed = {f.name for f in dataclasses.fields(ConvergenceConfig)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unrecognized protocol fields: {sorted(unknown)}")
    cfg = dict(raw)
    if isinstance(cfg.get("baseline_cfg"), dict):
        cfg["baseline_cfg"] = BaselineConfig(**cfg["baseline_cfg"])
    resolved = ConvergenceConfig(**cfg)
    if resolved.task not in ("sst2", "wikitext103", "ogbn-arxiv"):
        raise ValueError("This runner supports sst2, wikitext103 and ogbn-arxiv")
    if resolved.use_ddp and resolved.use_tensor_parallel:
        raise ValueError("Select DDP or tensor parallelism for a protocol")
    for name in ("epochs", "batch_size", "max_length", "rank"):
        if getattr(resolved, name) < 1:
            raise ValueError(f"{name} must be positive")
    if resolved.max_steps is not None and resolved.max_steps < 1:
        raise ValueError("max_steps must be positive or null")
    if resolved.baseline_cfg.grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be positive")
    for name in (
        "use_neighbor_sampling",
        "use_random_node_loader",
        "use_stacr_graph",
        "conv_v2",
        "conv_checkpoint",
        "conv_block_checkpoint",
        "conv_stage_checkpoint",
        "conv_saved_hooks",
    ):
        if getattr(resolved, name):
            raise ValueError(f"{name} is outside the supported experiment protocols")
    if resolved.conv_compression_mode != "none":
        raise ValueError("Convolution compression is outside the supported experiment protocols")
    return resolved


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "openwebtext":
        from .openwebtext_transfer import main as transfer_main

        return transfer_main(argv[1:])
    if argv and argv[0] == "bounds":
        from scripts.theorem_probe import main as bounds_main

        return bounds_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("openwebtext", help="Run the paired transfer smoke protocol")
    commands.add_parser("bounds", help="Measure per-layer gradient bounds on saved checkpoints")
    commands.add_parser("list", help="List training configurations")
    run = commands.add_parser("run")
    run.add_argument("profile", help="Configuration name from list")
    run.add_argument("--seeds", nargs="+", type=int)
    run.add_argument(
        "--steps", type=int, help="Override processed training-batch limit; changes the protocol"
    )
    run.add_argument("--device", help="Override cpu or cuda device")
    run.add_argument("--output", default="outputs/protocols")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    catalog = profiles()
    if args.command == "list":
        for name, spec in catalog.items():
            print(f"{name:42} {spec.get('description', spec.get('group', ''))}")
        return
    if args.profile not in catalog:
        parser.error(f"Unknown profile {args.profile!r}; run list")
    profile = catalog[args.profile]
    from stacr.utils import parse_override

    overrides = {}
    for value in args.set:
        key, separator, text = value.partition("=")
        if not separator or not key:
            parser.error("--set requires KEY=VALUE")
        overrides[key] = parse_override(text)
    if args.steps is not None:
        if args.steps < 1:
            parser.error("steps must be positive")
        overrides["max_steps"] = args.steps
    if args.device:
        overrides["device"] = args.device
    plans = []
    for seed in args.seeds or profile.get("seeds", [42]):
        raw = {**profile["config"], **overrides, "seed": seed}
        cfg = make_config(raw)
        plans.append(dataclasses.asdict(cfg))
    if args.dry_run:
        print(
            json.dumps(
                {
                    "profile": args.profile,
                    "world_size": profile.get("world_size", 1),
                    "overrides": overrides,
                    "runs": plans,
                },
                indent=2,
            )
        )
        return

    import torch
    import torch.distributed as dist
    from .logging import RunLogger, NullLogger
    from .training import run_convergence_experiment
    from stacr.cli import doctor

    expected_world = int(profile.get("world_size", 1))
    actual_world = int(os.environ.get("WORLD_SIZE", "1"))
    if expected_world != actual_world:
        parser.error(
            f"This profile needs WORLD_SIZE={expected_world}; launch with torchrun --nproc_per_node={expected_world}"
        )
    initialized = False
    if actual_world > 1 and not dist.is_initialized():
        if not torch.cuda.is_available():
            parser.error("Distributed measurement profiles require CUDA")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")
        initialized = True
    try:
        for raw in plans:
            cfg = make_config(raw)
            if str(cfg.device).startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError(
                    "This profile requires CUDA; use --device cpu for a small execution check"
                )
            run_dir = Path(args.output) / args.profile / f"seed{cfg.seed}_{time.time_ns()}"
            # Rank-zero owns the logs. Other ranks need no filesystem writes.
            is_main = not dist.is_initialized() or dist.get_rank() == 0
            logger = RunLogger(run_dir) if is_main else NullLogger(run_dir)
            logger.write_config(
                {
                    "profile": args.profile,
                    "config": raw,
                    "overrides": overrides,
                    "environment": doctor(),
                }
            )
            try:
                if cfg.task == "ogbn-arxiv":
                    from .graph import run_graph_experiment

                    result = run_graph_experiment(cfg, logger)
                else:
                    result = run_convergence_experiment(cfg, logger)
            except Exception as exc:
                logger.write_final(
                    {"status": "error", "type": type(exc).__name__, "error": str(exc)}
                )
                raise
            logger.write_final(result)
            if result.get("status") != "success":
                raise RuntimeError(f"Protocol did not complete successfully: {result}")
            if is_main:
                print(
                    json.dumps(
                        {
                            "profile": args.profile,
                            "seed": cfg.seed,
                            "output": str(run_dir),
                            "result": result,
                        },
                        default=str,
                    )
                )
    finally:
        if initialized:
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
