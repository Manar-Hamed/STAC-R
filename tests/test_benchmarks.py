"""Offline checks for experiment planning and measurement protocol integrity."""

import json
from types import SimpleNamespace

import pytest
import torch

from experiments.benchmarks import common
from experiments.benchmarks.__main__ import main
from experiments.benchmarks.registry import EXPERIMENTS


@pytest.fixture(autouse=True)
def reset_options():
    common.configure()
    yield
    common.configure()


@pytest.mark.parametrize("turbo_after_step", [None, 2])
def test_scientific_projection_schedule_is_explicit_at_step_50(turbo_after_step):
    from stacr import set_step

    config = {"surface": "all", "min_features": 1, "rank": 2}
    if turbo_after_step is not None:
        config["turbo_after_step"] = turbo_after_step
    resolved = common.resolve_parameters({"stacr_cfg": config})["stacr_cfg"]
    assert resolved["turbo_after_step"] == turbo_after_step
    model = torch.nn.Sequential(torch.nn.Linear(8, 4))
    layers = common.stacr_on_blocks(model, config)
    set_step(layers, 50)
    model(torch.randn(6, 8)).sum().backward()
    assert layers[0]._turbo() == (turbo_after_step is not None)
    assert ("turbo_after_step" in config) == (turbo_after_step is not None)


@pytest.mark.parametrize("name", sorted(EXPERIMENTS))
def test_every_benchmark_has_an_offline_plan(name, tmp_path):
    path = tmp_path / f"{name}.json"
    assert main(["run", name, "--dry-run", "--plan-output", str(path)]) == 0
    plan = json.loads(path.read_text())
    assert plan["cell_count"] > 0
    assert len(plan["cells"]) == plan["cell_count"]
    assert all(cell["parameters"] for cell in plan["cells"])


def test_resolved_plan_uses_requested_steps_shapes_and_seeds(tmp_path):
    path = tmp_path / "plan.json"
    main(
        [
            "run",
            "composed_frontier",
            "--cohort",
            "cf_gpt2m",
            "--dry-run",
            "--steps",
            "3",
            "--batch-size",
            "1",
            "--seq-length",
            "64",
            "--seeds",
            "11",
            "--plan-output",
            str(path),
        ]
    )
    cells = json.loads(path.read_text())["cells"]
    for cell in cells:
        params = cell["parameters"]
        assert (params["steps"], params["bs"], params["seq"], params["seed"]) == (3, 1, 64, 11)
        assert params["measure_from"] < params["steps"]


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.1))
        self.config = SimpleNamespace(_attn_implementation="eager")

    def forward(self, input_ids, labels):
        return SimpleNamespace(loss=(self.weight - input_ids.float().mean()).square())


def test_nonfinite_benchmark_loss_stops_before_optimizer_update(monkeypatch):
    model = TinyModel()
    before = model.weight.detach().clone()
    original = model.forward
    model.forward = lambda **batch: SimpleNamespace(loss=original(**batch).loss * float("nan"))
    monkeypatch.setattr(common, "load_model", lambda *a, **kw: model)
    ids = torch.ones(1, 4, dtype=torch.long)
    monkeypatch.setattr(common, "get_batches", lambda *a, **kw: [{"input_ids": ids, "labels": ids}])
    with pytest.raises(FloatingPointError, match="Nonfinite training loss at batch 0"):
        common.run_config(
            model_id="tiny",
            bs=1,
            seq=4,
            steps=1,
            seed=42,
            method="baseline",
            amp=False,
            eight_bit=False,
            device="cpu",
            measure_from=0,
            eval_batches=1,
        )
    torch.testing.assert_close(model.weight, before)
    assert model.weight.grad is None


@pytest.mark.parametrize("protocol, expected_skip", [("disjoint", 6), ("shared-prefix", 0)])
def test_openwebtext_validation_offset_and_measured_steps(monkeypatch, protocol, expected_skip):
    calls = []
    monkeypatch.setattr(common, "load_model", lambda *a, **kw: TinyModel())

    def batches(model_id, split, bs, seq, n_batches, dataset, device, skip_blocks=0):
        calls.append((split, skip_blocks))
        ids = torch.ones(bs, seq, dtype=torch.long)
        return [{"input_ids": ids, "labels": ids.clone()} for _ in range(n_batches)]

    monkeypatch.setattr(common, "get_batches", batches)
    result = common.run_config(
        model_id="tiny",
        bs=2,
        seq=4,
        steps=3,
        seed=42,
        method="baseline",
        amp=False,
        eight_bit=False,
        dataset="openwebtext",
        device="cpu",
        measure_from=1,
        eval_batches=1,
        validation_protocol=protocol,
    )
    assert calls == [("train", 0), ("validation", expected_skip)]
    assert result["steps_ran"] == 3
    assert result["steps_measured"] == 2
    assert result["validation_overlaps_training"] == (protocol == "shared-prefix")
    assert result["peak_source"] == "unavailable"


def test_short_dataset_is_an_error(monkeypatch):
    monkeypatch.setattr(common, "_tokenizer", lambda _: SimpleNamespace(name_or_path="unit-test"))
    monkeypatch.setattr(common, "_stream_blocks", lambda *a: [[1, 2, 3, 4]])
    common._DATA_CACHE.clear()
    with pytest.raises(ValueError, match="supplied 1 complete batches"):
        common.get_batches("tiny", "train", 1, 4, 2, device="cpu")


def test_importing_single_model_probe_does_not_replace_global_forward():
    import importlib
    from stacr.layers import STACRLinear

    original = STACRLinear.forward
    module = importlib.import_module("experiments.benchmarks.single_model_fidelity")
    importlib.reload(module)
    assert STACRLinear.forward is original


def test_gradient_probe_activates_compression_even_if_loader_returns_eval(monkeypatch):
    class ProbeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = torch.nn.Module()
            self.mlp.fc1 = torch.nn.Linear(8, 4)

        def forward(self, input_ids, labels):
            return SimpleNamespace(loss=self.mlp.fc1(input_ids).square().mean())

    monkeypatch.setattr(common, "load_model", lambda *a, **kw: ProbeModel().eval())
    generator = torch.Generator().manual_seed(321)
    inputs = torch.randn(2, 5, 8, generator=generator)
    monkeypatch.setattr(
        common,
        "get_batches",
        lambda *a, **kw: [{"input_ids": inputs, "labels": inputs} for _ in range(2)],
    )
    result = common.gradient_fidelity(
        model_id="tiny",
        bs=2,
        seq=5,
        seed=42,
        steps=2,
        stacr_cfg={"rank": 1, "residual_budget": 0, "surface": "mlp", "min_features": 1},
        device="cpu",
    )
    assert result["compression_training_layers"] == 1
    assert result["initialized_bases"] == 1
    assert result["mean_loss_gap"] < 1e-6
    assert result["mean_cosine_local_wrapped"] < 0.99
