# Adapted from INSTANT, copyright (c) 2024 Trung-Hieu Tran. MIT; see LICENSE.txt.
# Upstream revision: 8a04bd43c19cc52984a49fb1b7fd1dae14fb2ffc.
import torch
import torch.nn as nn

_GPT2_NUM_LAYERS = {"gpt2": 12, "gpt2-medium": 24, "gpt2-large": 36, "gpt2-xl": 48}


def add_activate_to_module():
    if not hasattr(nn.Module, "activate"):
        nn.Module.activate = True

    def enable_compression(self):
        self.activate = True
        for module in self.children():
            module.enable_compression()
        return self

    nn.Module.enable_compression = enable_compression

    def disable_compression(self):
        self.activate = False
        for module in self.children():
            module.disable_compression()
        return self

    nn.Module.disable_compression = disable_compression


def add_compression_tensor_to_module():
    if not hasattr(nn.Module, "CompressionTensor_x"):
        nn.Module.CompressionTensor_x = None
    if not hasattr(nn.Module, "CompressionTensor_gy"):
        nn.Module.CompressionTensor_gy = None

    def update_compression(self, P_dict, Q_dict):
        for name, module in self.named_modules():
            if name in P_dict and name in Q_dict:
                module.CompressionTensor_x = P_dict[name].detach().requires_grad_(False)
                module.CompressionTensor_gy = Q_dict[name].detach().requires_grad_(False)
        return self

    nn.Module.update_compression = update_compression


def convert_conv1d_to_linear(model):
    """Convert GPT-2 Conv1D modules to nn.Linear for INSTANT compatibility.

    GPT-2 uses transformers.pytorch_utils.Conv1D (weight shape [in, out])
    instead of nn.Linear (weight shape [out, in]). INSTANT's custom_op only
    handles nn.Linear, so we convert before applying compression.
    """
    try:
        from transformers.pytorch_utils import Conv1D
    except ImportError:
        return model
    module_dict = dict(model.named_modules())
    replacements = []
    for name, module in module_dict.items():
        if isinstance(module, Conv1D):
            in_features = module.weight.shape[0]
            out_features = module.weight.shape[1]
            linear = nn.Linear(in_features, out_features, bias=module.bias is not None)
            linear.weight.data = module.weight.data.T.contiguous()
            if module.bias is not None:
                linear.bias.data = module.bias.data.clone()
            replacements.append((name, linear))
    for name, linear in replacements:
        parts = name.rsplit(".", 1)
        parent = module_dict[parts[0]] if len(parts) > 1 else model
        attr = parts[1] if len(parts) > 1 else parts[0]
        setattr(parent, attr, linear)
    if replacements:
        print(f"[GPT-2] Converted {len(replacements)} Conv1D -> nn.Linear")
    return model


def freeze_layers(model, n_last_layers, model_name):
    """Freezes all layers except the last n transformer layers and the head.

    Supports BERT, DistilBERT, and GPT-2 model families.
    n_last_layers=0 means full fine-tuning (all params trainable).
    """
    if n_last_layers == 0:
        return
    if model_name == "bert-base-uncased":
        total_layers = 12
        n_last_layers = min(n_last_layers, total_layers)
        trainable_layer_indices = list(range(total_layers - n_last_layers, total_layers))
        for name, param in model.named_parameters():
            is_trainable = any(
                (f"bert.encoder.layer.{i}" in name for i in trainable_layer_indices)
            ) or name.startswith("classifier")
            param.requires_grad = is_trainable
    elif model_name == "distilbert-base-uncased":
        total_layers = 6
        n_last_layers = min(n_last_layers, total_layers)
        trainable_layer_indices = list(range(total_layers - n_last_layers, total_layers))
        for name, param in model.named_parameters():
            is_trainable = any(
                (f"distilbert.transformer.layer.{i}" in name for i in trainable_layer_indices)
            ) or name.startswith("classifier")
            param.requires_grad = is_trainable
    elif model_name.startswith("gpt2"):
        total_layers = _GPT2_NUM_LAYERS.get(model_name, 12)
        n_last_layers = min(n_last_layers, total_layers)
        trainable_layer_indices = list(range(total_layers - n_last_layers, total_layers))
        for name, param in model.named_parameters():
            is_trainable = (
                any((f"transformer.h.{i}." in name for i in trainable_layer_indices))
                or name.startswith("lm_head")
                or name == "transformer.ln_f.weight"
                or (name == "transformer.ln_f.bias")
            )
            param.requires_grad = is_trainable


freeze_bert_layers = freeze_layers


def get_svd_layers(model, model_name=None):
    """Get names of Linear layers eligible for INSTANT compression."""
    from .callback import _skip_layer

    modules_compressed = []
    for name, module in model.named_modules():
        if any((param.requires_grad for param in module.parameters())):
            if isinstance(module, nn.Linear):
                if model_name and _skip_layer(name, model_name):
                    continue
                elif model_name is None:
                    if "classifier" in name or "bert.embeddings" in name or "bert.pooler" in name:
                        continue
                modules_compressed.append(name)
    return modules_compressed


def print_status(model):
    for name, param in model.named_parameters():
        status = "frozen" if not param.requires_grad else "trainable"
        print(f"{name}: {status}")
