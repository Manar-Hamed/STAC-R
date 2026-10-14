import copy
import gc
import weakref
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from stacr.protocols.checkpointing import (
    CheckpointBoundaryCodec,
    STACRCheckpointSpec,
    ResolvedCheckpointSite,
    _deserialize_packed_record,
    _serialize_packed_record,
    stacr_checkpoint,
)

DEVICES = ["cpu"] + (
    [pytest.param("cuda", marks=pytest.mark.cuda)] if torch.cuda.is_available() else []
)


def boundary_spec(**overrides):
    spec = STACRCheckpointSpec(
        site_id="block",
        module_type="GPT2Block",
        model_family="gpt2",
        hidden_size=32,
        rank=8,
        residual_mode="none",
        residual_budget=0,
        basis_dtype=torch.float32,
        quantize_coefficients=False,
        quantize_bits=8,
        quantize_residuals=False,
        residual_quantize_bits=4,
        warmup_steps=1,
        refresh_trigger="off",
        refresh_period=1,
        refresh_energy_threshold=0.15,
        target_arg_index=0,
        min_numel=1,
    )
    return replace(spec, **overrides)


@pytest.mark.parametrize(
    "hidden,refresh,quantized",
    [(32, "off", False), (64, "off", True), (32, "residual_energy", False)],
)
@pytest.mark.parametrize("device", DEVICES)
def test_boundary_protocol_runs_four_steps_and_validation_preserves_state(
    monkeypatch, tmp_path, hidden, refresh, quantized, device
):
    transformers = pytest.importorskip("transformers")
    from experiments.protocols import training
    from experiments.protocols.logging import RunLogger
    from stacr.protocols.config.training import BaselineConfig

    torch.manual_seed(13)
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=48,
            n_positions=16,
            n_embd=hidden,
            n_layer=2,
            n_head=4,
            resid_pdrop=0,
            embd_pdrop=0,
            attn_pdrop=0,
            use_cache=False,
            _attn_implementation="eager",
        )
    )
    ids = torch.randint(0, 48, (8, 8))
    loader = torch.utils.data.DataLoader(
        [
            {
                "input_ids": tokens,
                "attention_mask": torch.ones_like(tokens),
                "labels": tokens.clone(),
            }
            for tokens in ids
        ],
        batch_size=2,
    )
    monkeypatch.setattr(training, "create_gpt2_lm", lambda *args, **kwargs: model)
    monkeypatch.setattr(
        training,
        "setup_wikitext_data",
        lambda cfg: (loader, loader, SimpleNamespace(pad_token_id=0)),
    )
    captured = {}
    enable = training.enable_c1_checkpointing_for_model

    def capture_runtime(*args, **kwargs):
        runtime = enable(*args, **kwargs)
        captured["runtime"] = runtime
        return runtime

    monkeypatch.setattr(training, "enable_c1_checkpointing_for_model", capture_runtime)
    validate = training.validate

    def validate_without_state_changes(*args, **kwargs):
        codecs = captured["runtime"].registry.items()
        stats = copy.deepcopy({name: codec.stats() for name, codec in codecs.items()})
        buffers = {name: copy.deepcopy(codec.state_dict()) for name, codec in codecs.items()}
        result = validate(*args, **kwargs)
        assert {name: codec.stats() for name, codec in codecs.items()} == stats
        for name, codec in codecs.items():
            for key, value in codec.state_dict().items():
                assert torch.equal(value, buffers[name][key])
        captured["validated"] = True
        return result

    monkeypatch.setattr(training, "validate", validate_without_state_changes)
    cfg = training.ConvergenceConfig(
        task="wikitext103",
        model="tiny-gpt2",
        method="baseline",
        method_label="baseline",
        device=device,
        epochs=1,
        max_steps=4,
        batch_size=2,
        max_length=8,
        lr=1e-4,
        num_workers=0,
        quiet=True,
        use_8bit_optimizer=False,
        baseline_cfg=BaselineConfig(checkpointing="transformer_hf"),
        stacr_compress_checkpoint=True,
        stacr_ckpt_rank=8,
        stacr_ckpt_warmup_steps=1,
        stacr_ckpt_min_numel=1,
        stacr_ckpt_planner="off",
        stacr_ckpt_basis_dtype="float32",
        stacr_ckpt_refresh_trigger=refresh,
        stacr_ckpt_quantize_coefficients=quantized,
    )
    result = training.run_convergence_experiment(cfg, RunLogger(tmp_path))
    assert result["status"] == "success", result
    assert captured["validated"]
    flags = result["results"]["stacr_flags"]
    assert flags["c1_steps_observed"] == 4
    assert flags["c1_site_count_engaged"] == 2
    if refresh == "off":
        assert flags["c1_site_count_compressed"] == 2
    assert torch.isfinite(torch.tensor(result["results"]["final_perplexity"]))
    assert not captured["runtime"].is_active


@pytest.mark.parametrize("device", DEVICES)
def test_boundary_record_reconstruction_survives_later_basis_refresh(device):
    torch.manual_seed(16)
    codec = CheckpointBoundaryCodec(boundary_spec(refresh_trigger="period", refresh_period=2))
    codec.pack_tensor(torch.randn(2, 8, 32, device=device), training=True)
    record = codec.pack_tensor(torch.randn(2, 8, 32, device=device), training=True)
    assert not record.is_dense_fallback
    expected = codec.unpack_record(record).clone()
    codec.pack_tensor(torch.randn(2, 8, 32, device=device), training=True)
    refreshed = codec.pack_tensor(torch.randn(2, 8, 32, device=device), training=True)
    assert refreshed.is_dense_fallback
    meta, tensors = _serialize_packed_record(record)
    restored = _deserialize_packed_record(meta, tuple(tensors))
    torch.testing.assert_close(codec.unpack_record(restored), expected)


@pytest.mark.parametrize("device", DEVICES)
def test_compressed_checkpoint_does_not_retain_dense_boundary_reference(device):
    spec = boundary_spec(warmup_steps=0)
    codec = CheckpointBoundaryCodec(spec)
    block = torch.nn.Linear(32, 32).to(device)
    source = torch.randn(2, 8, 32, device=device, requires_grad=True)
    boundary = source * 2
    reference = weakref.ref(boundary)
    resolved = ResolvedCheckpointSite(block, spec, codec, boundary, 0)
    output = stacr_checkpoint(block.forward, boundary, resolved=resolved)
    del boundary, resolved
    gc.collect()
    assert reference() is None, "Checkpoint context retains the dense boundary after compression"
    output.square().sum().backward()
    assert source.grad is not None
