from __future__ import annotations

from pathlib import Path

import torch

# Hub ids are tried in order. The canonical short ids were namespaced upstream, so a
# current hub client needs the namespaced form while older pins still accept the short one.
DATASET_ALIASES: dict[str, tuple[str, ...]] = {
    "wikitext": ("Salesforce/wikitext", "wikitext"),
    "glue": ("nyu-mll/glue", "glue"),
    "openwebtext": ("Skylion007/openwebtext", "openwebtext"),
}


def load_hf_dataset(name: str, subset: str | None = None, split: str = "train", **kwargs):
    from datasets import load_dataset

    last: Exception | None = None
    for repo in DATASET_ALIASES.get(name, (name,)):
        try:
            if subset:
                return load_dataset(repo, subset, split=split, **kwargs)
            return load_dataset(repo, split=split, **kwargs)
        except Exception as exc:  # fall through to the next alias
            last = exc
    raise RuntimeError(
        f"Could not load dataset {name!r} (subset={subset!r}, split={split!r})"
    ) from last


def tokenize_text_dataset(dataset, tokenizer, text_key: str, max_length: int):
    def encode(batch):
        return tokenizer(
            batch[text_key],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    return dataset.map(encode, batched=True, remove_columns=dataset.column_names)


def packed_text_blocks(
    rows, tokenizer, seq_len: int, count: int, *, text_key: str = "text", skip: int = 0
) -> list[dict]:
    """Pack contiguous causal-LM blocks without padding targets or document separators.

    ``skip`` counts complete blocks, for held-out data in a single training split.
    """
    if seq_len < 2 or count < 1 or skip < 0:
        raise ValueError("seq_len >= 2, count >= 1 and skip >= 0 are required")
    buffer: list[int] = []
    result = []
    skipped = 0
    for row in rows:
        text = row[text_key]
        if not text or not text.strip():
            continue
        encoded = tokenizer(text, truncation=False, add_special_tokens=False)
        buffer.extend(encoded["input_ids"])
        offset = 0
        while len(buffer) - offset >= seq_len:
            ids = buffer[offset : offset + seq_len]
            offset += seq_len
            if skipped < skip:
                skipped += 1
                continue
            tensor = torch.tensor(ids, dtype=torch.long)
            result.append(
                {
                    "input_ids": tensor,
                    "attention_mask": torch.ones_like(tensor),
                    "labels": tensor.clone(),
                }
            )
            if len(result) == count:
                return result
        buffer = buffer[offset:]
    if not result:
        raise ValueError("The selected text split contains no complete token blocks")
    return result


def build_examples(config: dict, tokenizer, *, validation: bool = False) -> list[dict]:
    """Prepare deterministic CPU examples for the named dataset or local text files."""
    task = config.get("task", "language_modeling")
    count = int(
        config.get(
            "max_eval_examples" if validation else "max_examples", 128 if validation else 2048
        )
    )
    seq = int(config.get("seq_len", 128 if task == "sequence_classification" else 1024))
    if count < 1:
        raise ValueError("Example counts must be positive")
    if task == "sequence_classification":
        subset = str(config.get("dataset", "sst2"))
        if subset not in ("sst2", "mrpc"):
            raise ValueError("Classification datasets: sst2, mrpc")
        raw = load_hf_dataset(
            "glue",
            subset,
            split="validation" if validation else "train",
            revision=config.get("dataset_revision"),
        )
        result = []
        for row in raw.select(range(min(count, len(raw)))):
            texts = (row["sentence"],) if subset == "sst2" else (row["sentence1"], row["sentence2"])
            encoded = tokenizer(*texts, truncation=True, padding="max_length", max_length=seq)
            result.append(
                {
                    **{k: torch.tensor(v) for k, v in encoded.items()},
                    "labels": torch.tensor(row["label"], dtype=torch.long),
                }
            )
        return result
    dataset = str(config.get("dataset", "wikitext-103"))
    kwargs = {"streaming": True}
    if config.get("dataset_revision"):
        kwargs["revision"] = config["dataset_revision"]
    skip = 0
    if dataset in ("wikitext", "wikitext-103", "wt103"):
        rows = load_hf_dataset(
            "wikitext",
            "wikitext-103-raw-v1",
            split="validation" if validation else "train",
            **kwargs,
        )
    elif dataset == "openwebtext":
        rows = load_hf_dataset("openwebtext", split="train", **kwargs)
        if validation:
            skip = int(config.get("max_examples", 2048))
    elif dataset == "text":
        key = "validation_file" if validation else "train_file"
        if not config.get(key):
            raise ValueError(f"dataset=text requires {key}")
        if validation and Path(config[key]).resolve() == Path(config["train_file"]).resolve():
            raise ValueError("Training and validation text files must be different")
        with Path(config[key]).open(encoding="utf-8") as stream:
            return packed_text_blocks(({"text": line} for line in stream), tokenizer, seq, count)
    else:
        raise ValueError(f"Unsupported dataset {dataset!r}; use wikitext-103, openwebtext or text")
    return packed_text_blocks(rows, tokenizer, seq, count, skip=skip)
