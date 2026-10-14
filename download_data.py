from __future__ import annotations

import argparse


KNOWN_DATASETS = {"wt103", "sst2", "pythia", "gpt2", "openwebtext"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public datasets used by the artifact.")
    parser.add_argument("--datasets", nargs="+", default=["wt103", "sst2"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    unknown = sorted(set(args.datasets) - KNOWN_DATASETS)
    if unknown:
        raise ValueError(f"Unknown dataset key(s): {unknown}. Known keys: {sorted(KNOWN_DATASETS)}")
    if args.dry_run:
        print({"datasets": args.datasets, "status": "dry_run"})
        return
    from transformers import AutoTokenizer

    from stacr.datasets import load_hf_dataset

    for name in args.datasets:
        if name == "wt103":
            load_hf_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        elif name == "sst2":
            load_hf_dataset("glue", "sst2", split="train")
        elif name == "pythia":
            load_hf_dataset("wikitext", "wikitext-103-raw-v1", split="train")
            AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b")
        elif name == "gpt2":
            load_hf_dataset("wikitext", "wikitext-103-raw-v1", split="train")
            AutoTokenizer.from_pretrained("gpt2")
        elif name == "openwebtext":
            load_hf_dataset("openwebtext", split="train", streaming=True)
    print({"datasets": args.datasets, "status": "available"})


if __name__ == "__main__":
    main()
