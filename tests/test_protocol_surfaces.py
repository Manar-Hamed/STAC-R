import copy

import pytest
import torch

transformers = pytest.importorskip("transformers")
DEVICES = ["cpu"] + (
    [pytest.param("cuda", marks=pytest.mark.cuda)] if torch.cuda.is_available() else []
)


def model_and_batch(family):
    torch.manual_seed(12)
    ids = torch.randint(0, 32, (2, 8))
    if family == "bert":
        model = transformers.BertForSequenceClassification(
            transformers.BertConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                hidden_dropout_prob=0,
                attention_probs_dropout_prob=0,
                num_labels=2,
                _attn_implementation="eager",
            )
        )
        batch = dict(
            input_ids=ids, attention_mask=torch.ones_like(ids), labels=torch.tensor([0, 1])
        )
    else:
        model = transformers.GPT2LMHeadModel(
            transformers.GPT2Config(
                vocab_size=32,
                n_positions=16,
                n_embd=16,
                n_layer=1,
                n_head=2,
                resid_pdrop=0,
                embd_pdrop=0,
                attn_pdrop=0,
                use_cache=False,
                _attn_implementation="eager",
            )
        )
        batch = dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone())
    return model, batch


@pytest.mark.parametrize("family", ["bert", "gpt2"])
@pytest.mark.parametrize("method", ["svd", "tt", "tucker"])
def test_factorization_protocol_forward_and_backward(family, method):
    pytest.importorskip("tensorly")
    from experiments.protocols.training import (
        ConvergenceConfig,
        get_compression_targets_bert,
        get_compression_targets_gpt2,
        replace_linear_with_compressed,
    )
    from stacr.protocols.layers import CompressedLinear

    model, batch = model_and_batch(family)
    expected = model(**batch).logits.detach()
    targets = (
        get_compression_targets_bert(model)
        if family == "bert"
        else get_compression_targets_gpt2(model)
    )
    replace_linear_with_compressed(
        model, ConvergenceConfig(method=method, rank=2, quiet=True), targets
    )
    assert sum(isinstance(module, CompressedLinear) for module in model.modules()) == len(targets)
    output = model(**batch)
    torch.testing.assert_close(output.logits, expected, atol=1e-6, rtol=1e-5)
    output.loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


@pytest.mark.parametrize("method", ["svd", "tt", "tucker"])
def test_factorization_does_not_add_bias_to_biasless_projections(method):
    pytest.importorskip("tensorly")
    from experiments.protocols.training import ConvergenceConfig, replace_linear_with_compressed

    model = torch.nn.Sequential(torch.nn.Linear(16, 8, bias=False))
    inputs = torch.randn(4, 16)
    expected = model(inputs).detach()
    replace_linear_with_compressed(
        model, ConvergenceConfig(method=method, rank=2, quiet=True), ["0"]
    )
    assert model[0].linear.bias is None
    torch.testing.assert_close(model(inputs), expected)
    model(inputs).sum().backward()
    assert torch.isfinite(model[0].linear.weight.grad).all()


@pytest.mark.parametrize("family,joint", [("bert", False), ("bert", True), ("gpt2", False)])
@pytest.mark.parametrize("device", DEVICES)
def test_attention_protocol_forward_backward_and_shared_basis(family, joint, device):
    from stacr.adaptive import STACRConfig, STACRLinear
    from stacr.protocols.attention import JointQKSTACRLinear, apply_stacr_to_attention

    model, batch = model_and_batch(family)
    model = model.to(device)
    batch = {name: tensor.to(device) for name, tensor in batch.items()}
    expected = model(**batch).logits.detach()
    report = apply_stacr_to_attention(
        model,
        STACRConfig(rank=2, basis_dtype="float32"),
        enable_gpt2_c_attn_split=True,
        joint_qk=joint,
        mask_aware=True,
    )
    assert report.replaced_qkv == 3
    output = model(**batch)
    torch.testing.assert_close(output.logits, expected, atol=1e-6, rtol=1e-5)
    output.loss.backward()
    layers = [module for module in model.modules() if isinstance(module, STACRLinear)]
    assert len(layers) == 3
    assert all(int(layer.step_counter) == 1 for layer in layers)
    assert all(torch.isfinite(layer.linear.weight.grad).all() for layer in layers)
    if joint:
        pair = next(module for module in model.modules() if isinstance(module, JointQKSTACRLinear))
        assert pair.q_stacr.effective_basis is pair.k_stacr.effective_basis
        restored, _ = model_and_batch(family)
        restored = restored.to(device)
        apply_stacr_to_attention(restored, STACRConfig(rank=2), joint_qk=True, mask_aware=True)
        restored.load_state_dict(copy.deepcopy(model.state_dict()))
        assert torch.isfinite(restored(**batch).loss)


def test_attention_mask_reduces_causality_without_retaining_padding():
    from stacr.protocols.attention import _extract_token_mask

    allowed = torch.ones(4, 4, dtype=torch.bool).tril()[None, None]
    allowed[..., -1] = False
    expected = torch.tensor([[True, True, True, False]])
    assert torch.equal(_extract_token_mask(allowed, 4, torch.device("cpu")), expected)
    additive = torch.zeros_like(allowed, dtype=torch.float32).masked_fill(~allowed, -1e9)
    assert torch.equal(_extract_token_mask(additive, 4, torch.device("cpu")), expected)
    binary = expected.float()
    assert torch.equal(_extract_token_mask(binary, 4, torch.device("cpu")), expected)
