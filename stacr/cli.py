"""Command line entry point; model libraries load only for commands that need them."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


def doctor():
    import torch

    libraries = {}
    for name in (
        "stacr",
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "bitsandbytes",
        "peft",
        "galore-torch",
    ):
        try:
            libraries[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            libraries[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.system(),
        "libraries": libraries,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpus": [
            {
                "name": torch.cuda.get_device_name(i),
                "memory_gib": torch.cuda.get_device_properties(i).total_memory / 1024**3,
            }
            for i in range(torch.cuda.device_count())
        ],
    }


def resolve_config(name):
    from importlib.resources import files

    path = Path(name)
    if path.is_file():
        return path
    resource = files("configs").joinpath(name if name.endswith(".yaml") else f"{name}.yaml")
    if resource.is_file():
        return resource
    raise ValueError(f"Config {name!r} not found; pass a YAML path or a packaged preset name")


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "reproduce":
        from experiments.protocols.__main__ import main as protocol_main

        return protocol_main(args[1:])
    if args and args[0] == "benchmarks":
        from experiments.benchmarks.__main__ import main as benchmark_main

        return benchmark_main(args[1:])
    parser = argparse.ArgumentParser(
        description="STAC-R training, integration checks and experiments."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Report library versions and available GPUs")
    sub.add_parser("benchmarks", help="List or run measurement grids")
    sub.add_parser("reproduce", help="List or run training protocols")
    for name in ("train", "fidelity"):
        run = sub.add_parser(
            name, help="Train a configuration" if name == "train" else "Compare paired gradients"
        )
        run.add_argument("--config", required=True)
        run.add_argument("--steps", type=int)
        run.add_argument("--seed", type=int)
        run.add_argument("--device")
        run.add_argument("--demo", action="store_true")
        run.add_argument(
            "--dry-run",
            action="store_true",
            help="Print resolved settings without loading data/models",
        )
        run.add_argument(
            "--output", help="CSV output; resolved settings are saved alongside it as JSON"
        )
        run.add_argument(
            "--set",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="Override a YAML setting",
        )
    parsed = parser.parse_args(args)
    if parsed.command == "doctor":
        print(json.dumps(doctor(), indent=2))
        return
    from .training import gradient_fidelity, run_config, validate_config
    from .utils import load_yaml, parse_override, write_csv

    config = load_yaml(resolve_config(parsed.config))
    for item in parsed.set:
        key, separator, value = item.partition("=")
        if not separator or not key:
            parser.error("--set needs KEY=VALUE")
        config[key] = parse_override(value)
    for key in ("steps", "seed", "device"):
        value = getattr(parsed, key)
        if value is not None:
            config[key] = value
    if parsed.demo:
        config["steps"] = min(int(config.get("steps", 5)), 5)
        config["max_examples"] = min(int(config.get("max_examples", 64)), 64)
        config["max_eval_examples"] = min(int(config.get("max_eval_examples", 16)), 16)
    config = validate_config(config)
    if parsed.dry_run:
        print(json.dumps(config, indent=2))
        return
    output = Path(
        parsed.output
        or f"outputs/{Path(parsed.config).stem}_{parsed.command}_seed{config['seed']}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    result = gradient_fidelity(config) if parsed.command == "fidelity" else run_config(config)
    write_csv(output, [result])
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    raise SystemExit(main())
