import copy
from types import SimpleNamespace

import pytest
import torch

from stacr.datasets import packed_text_blocks
from stacr.training import _evaluate, run_config, train_model, validate_config


class Tokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": [int(x) for x in text.split()]}


def test_packed_data_has_disjoint_validation_and_no_padding_targets():
    texts = [{"text": "0 1 2 3 4"}, {"text": "5 6 7 8 9 10 11"}]
    train = packed_text_blocks(texts, Tokenizer(), 3, 2)
    valid = packed_text_blocks(texts, Tokenizer(), 3, 2, skip=2)
    assert train[-1]["input_ids"].tolist() == [3, 4, 5]
    assert valid[0]["input_ids"].tolist() == [6, 7, 8]
    assert torch.equal(valid[0]["labels"], valid[0]["input_ids"])


def test_empty_text_rejected():
    with pytest.raises(ValueError, match="no complete"):
        packed_text_blocks([{"text": "1"}], Tokenizer(), 4, 1)


def test_misspelled_method_is_rejected():
    with pytest.raises(ValueError, match="Unknown method"):
        validate_config({"method": "stcar"})


def test_composed_method_selects_real_optimizer():
    assert validate_config({"method": "stacr_galore"})["optimizer"] == "galore"
    assert validate_config({"method": "turbofast"})["turbo_after_step"] == 50


def test_synthetic_default_is_turbo_and_hf_default_remains_baseline():
    synthetic = validate_config({})
    assert synthetic["method"] == "turbofast"
    assert synthetic["turbo_after_step"] == 50
    hf = validate_config({"task": "language_modeling", "model_name": "tiny"})
    assert hf["method"] == "baseline"


@pytest.mark.parametrize(
    "method", ["stacr", "stacr_galore", "subtrack_stacr", "stacr_checkpointing"]
)
def test_named_full_methods_resolve_persistent_full_compression(method):
    config = validate_config({"method": method})
    assert config["turbo_after_step"] is None
    assert validate_config({**config, "turbo_after_step": 7})["turbo_after_step"] == 7


@pytest.mark.parametrize("method", ["stacr", "turbofast"])
def test_named_training_method_keeps_its_basis_schedule_after_step_50(method):
    from stacr import set_step
    from stacr.training import configure_model

    torch.manual_seed(71)
    model = torch.nn.Sequential(torch.nn.Linear(8, 4))
    layers, _ = configure_model(model, {"method": method, "rank": 2})
    set_step(layers, 49)
    model(torch.randn(12, 8)).sum().backward()
    previous_basis = layers[0].basis.clone()
    set_step(layers, 50)
    model(torch.randn(12, 8)).sum().backward()
    assert layers[0]._turbo() == (method == "turbofast")
    if method == "turbofast":
        torch.testing.assert_close(layers[0].basis, previous_basis, rtol=0, atol=0)
    else:
        assert not torch.equal(layers[0].basis, previous_basis)
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_training_smoke_returns_actual_cpu_measurement():
    row = run_config(
        {
            "task": "synthetic",
            "method": "stacr",
            "device": "cpu",
            "steps": 3,
            "features": 16,
            "hidden": 12,
            "rank": 4,
            "batch_size": 8,
            "measure_from": 1,
            "seed": 7,
        }
    )
    assert row["steps_measured"] == 2
    assert row["wrapped_layers"] == 2
    assert row["peak_gib"] is None
    assert row["seconds"] > 0


def test_accumulation_matches_equivalent_batch():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 2)

        def forward(self, inputs, labels):
            return SimpleNamespace(loss=(self.linear(inputs) - labels).square().mean())

    torch.manual_seed(42)
    one = Model()
    two = copy.deepcopy(one)
    data = {"inputs": torch.randn(6, 4), "labels": torch.randn(6, 2)}
    micro = [{k: v[:3] for k, v in data.items()}, {k: v[3:] for k, v in data.items()}]
    cfg = {
        "task": "synthetic",
        "method": "baseline",
        "steps": 2,
        "device": "cpu",
        "batch_size": 6,
        "lr": 0.001,
    }
    train_model(one, [data], cfg)
    train_model(two, micro, {**cfg, "gradient_accumulation_steps": 2, "batch_size": 3})
    for a, b in zip(one.parameters(), two.parameters()):
        assert torch.allclose(a, b, atol=1e-7)


def test_validation_counts_only_shifted_unmasked_targets():
    class Model(torch.nn.Module):
        def forward(self, **batch):
            return SimpleNamespace(loss=torch.tensor(float(batch["input_ids"][0, 0])))

    model = Model().eval()
    batches = [
        {"input_ids": torch.tensor([[1, 0, 0, 0]]), "labels": torch.tensor([[1, 1, 1, -100]])},
        {"input_ids": torch.tensor([[3, 0, 0, 0]]), "labels": torch.tensor([[1, 1, -100, -100]])},
    ]
    result = _evaluate(model, batches, torch.device("cpu"), "language_modeling", None)
    assert result["nll"] == pytest.approx(5 / 3)
    assert not model.training


@pytest.mark.parametrize("method", ["baseline", "stacr"])
def test_recorded_transformer_protocol_runs_offline(monkeypatch, tmp_path, method):
    transformers = pytest.importorskip("transformers")
    from experiments.protocols import training
    from experiments.protocols.logging import RunLogger
    from stacr.protocols.config.training import BaselineConfig

    torch.manual_seed(42)
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=32,
            n_positions=16,
            n_embd=16,
            n_layer=1,
            n_head=2,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
    )
    ids = torch.randint(0, 32, (4, 8))
    samples = [
        {"input_ids": x, "attention_mask": torch.ones_like(x), "labels": x.clone()} for x in ids
    ]
    loader = torch.utils.data.DataLoader(samples, batch_size=2)
    monkeypatch.setattr(training, "create_gpt2_lm", lambda *args, **kwargs: model)
    monkeypatch.setattr(
        training,
        "setup_wikitext_data",
        lambda cfg: (loader, loader, SimpleNamespace(pad_token_id=0)),
    )
    config = training.ConvergenceConfig(
        task="wikitext103",
        model="tiny-gpt2",
        method=method,
        method_label=method,
        rank=4,
        device="cpu",
        epochs=1,
        max_steps=2,
        batch_size=2,
        max_length=8,
        lr=1e-4,
        num_workers=0,
        quiet=True,
        use_8bit_optimizer=False,
        baseline_cfg=BaselineConfig(),
        stacr_basis_dtype="float32",
    )
    result = training.run_convergence_experiment(config, RunLogger(tmp_path))
    assert result["status"] == "success"
    assert result["results"]["epochs_completed"] == 1
    assert result["results"]["final_perplexity"] > 0


@pytest.mark.parametrize("limit,accum", [(2, 1), (3, 2), (2, 4)])
def test_protocol_counts_processed_batches_and_flushes_tail(tmp_path, limit, accum):
    from experiments.protocols import training
    from experiments.protocols.logging import RunLogger

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 2)

        def forward(self, input_ids, labels):
            logits = self.linear(input_ids)
            return SimpleNamespace(
                logits=logits, loss=torch.nn.functional.cross_entropy(logits, labels)
            )

    torch.manual_seed(9)
    model = Model()
    expected = copy.deepcopy(model)
    batches = [{"input_ids": torch.randn(2, 3), "labels": torch.tensor([0, 1])} for _ in range(5)]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    reference = torch.optim.SGD(expected.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    cfg = training.ConvergenceConfig(
        task="sst2", method="baseline", max_steps=limit, quiet=True, measure_memory=False
    )
    metrics = training.train_epoch(
        model,
        batches,
        None,
        optimizer,
        scheduler,
        torch.device("cpu"),
        cfg,
        0,
        RunLogger(tmp_path),
        accum_steps=accum,
        grad_clip=0,
    )
    losses = []
    for i, batch in enumerate(batches[:limit]):
        loss = expected(**batch).loss
        losses.append(float(loss.detach()))
        (loss / accum).backward()
        if (i + 1) % accum == 0 or i + 1 == limit:
            reference.step()
            reference.zero_grad(set_to_none=True)
    assert metrics["steps_completed"] == limit
    assert metrics["train_loss"] == pytest.approx(sum(losses) / limit)
    assert scheduler.last_epoch == (limit + accum - 1) // accum
    for actual, wanted in zip(model.parameters(), expected.parameters()):
        torch.testing.assert_close(actual, wanted)


def test_generic_adaptive_turbo_and_microbatch_runtime():
    from stacr.training import configure_model

    cfg = {
        "task": "synthetic",
        "method": "turbofast",
        "engine": "adaptive",
        "rank": 2,
        "turbo_after_step": 2,
        "compression": {"basis_dtype": "float32"},
    }
    layers, _ = configure_model(torch.nn.Sequential(torch.nn.Linear(8, 4)), cfg)
    assert layers[0].config.residual_mode == "block_sparse"
    assert layers[0].config.post_warmup_residual_mode == "none"
    assert layers[0].config.residual_warmup_steps == 2
    result = run_config(
        {
            **cfg,
            "steps": 3,
            "device": "cpu",
            "features": 8,
            "hidden": 4,
            "gradient_accumulation_steps": 2,
        }
    )
    assert result["steps"] == 3


@pytest.mark.parametrize("engine", ["projection", "adaptive"])
def test_generic_fidelity_preserves_parameter_mapping(monkeypatch, engine):
    from stacr import training

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(8, 4)
            self.dropout = torch.nn.Dropout(0.5)

        def forward(self, inputs, labels):
            return SimpleNamespace(
                loss=(self.dropout(self.linear(inputs)) - labels).square().mean()
            )

    batch = {"inputs": torch.randn(6, 8), "labels": torch.randn(6, 4)}
    monkeypatch.setattr(training, "_hf_inputs", lambda _: (Model(), [batch], []))
    result = training.gradient_fidelity(
        {
            "task": "synthetic",
            "model_name": "tiny",
            "steps": 2,
            "device": "cpu",
            "engine": engine,
            "rank": 2,
            "residual_budget": 0,
            "compression": {
                "basis_dtype": "float32",
                **({"randomized_svd_threshold": 4} if engine == "adaptive" else {}),
            },
        }
    )
    assert result["wrapped_layers"] == 1
    assert -1 <= result["local_mean_cosine"] <= 1
    assert result["worst_unwrapped_cosine"] > 1 - 1e-10
