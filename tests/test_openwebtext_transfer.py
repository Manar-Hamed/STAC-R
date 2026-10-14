from types import SimpleNamespace

import pytest
import torch

from experiments.protocols.openwebtext_transfer import _loaders


def test_paired_order_is_rng_independent_and_validation_blocks_are_disjoint(monkeypatch):
    datasets = pytest.importorskip("datasets")

    class Corpus:
        def __init__(self, rows):
            self.rows = rows
            self.column_names = list(rows[0]) if rows else []
            self.as_tensors = False

        def map(self, function, **kwargs):
            batch = {key: [row[key] for row in self.rows] for key in self.column_names}
            output = function(batch)
            return Corpus([dict(zip(output, row)) for row in zip(*output.values())])

        def set_format(self, **kwargs):
            self.as_tensors = True

        def select(self, indices):
            result = Corpus([self.rows[i] for i in indices])
            result.as_tensors = self.as_tensors
            return result

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            return {key: torch.tensor(value) for key, value in self.rows[index].items()}

    monkeypatch.setattr(
        datasets,
        "load_dataset",
        lambda *args, **kwargs: Corpus([{"text": str(i)} for i in range(20)]),
    )

    def tokenizer(texts, **kwargs):
        return {
            "input_ids": [[int(text)] * 4 for text in texts],
            "attention_mask": [[1] * 4 for _ in texts],
        }

    args = SimpleNamespace(
        dataset="fixture",
        fallback_dataset="",
        split="train",
        max_length=4,
        batch_size=2,
        validation_blocks=4,
        data_order="paired",
    )
    train, validation, metadata = _loaders(args, tokenizer, 123)
    first = torch.cat([batch["input_ids"][:, 0] for batch in train]).tolist()
    torch.rand(107)
    train_again, _, _ = _loaders(args, tokenizer, 123)
    assert first == torch.cat([batch["input_ids"][:, 0] for batch in train_again]).tolist()
    heldout = torch.cat([batch["input_ids"][:, 0] for batch in validation]).tolist()
    assert set(first).isdisjoint(heldout)
    assert metadata["training_blocks"] == 16 and heldout == [16, 17, 18, 19]
