# Adapted from INSTANT, copyright (c) 2024 Trung-Hieu Tran. MIT; see LICENSE.txt.
# Upstream revision: 8a04bd43c19cc52984a49fb1b7fd1dae14fb2ffc.
from transformers import TrainerCallback
from itertools import islice
import gc
import torch.nn as nn
import torch
from .svd import SVD_expected_value
from .utils import log_svd_shapes

P_dict = {}
Q_dict = {}
layer_inputs = {}
layer_gradients = {}
hook_handles_input = []
hook_handles_grad = []


def _get_layer_prefix(model_name):
    """Get the encoder layer prefix for a given model."""
    if model_name == "bert-base-uncased":
        return "bert.encoder.layer."
    elif model_name == "distilbert-base-uncased":
        return "distilbert.transformer.layer."
    elif model_name.startswith("gpt2"):
        return "transformer.h."
    else:
        raise ValueError(f"Unknown model for INSTANT hooks: {model_name}")


def _skip_layer(name, model_name):
    """Check if a layer should be excluded from compression hooks."""
    if model_name == "bert-base-uncased":
        return "classifier" in name or "bert.embeddings" in name or "bert.pooler" in name
    elif model_name == "distilbert-base-uncased":
        return "classifier" in name or "distilbert.embeddings" in name or "pre_classifier" in name
    elif model_name.startswith("gpt2"):
        return (
            "lm_head" in name
            or "transformer.wte" in name
            or "transformer.wpe" in name
            or ("transformer.ln_f" in name)
        )
    return False


def save_input_hook(name, module, input, output):
    input = input[0].detach()
    if input.dim() == 3:
        X_XT = torch.matmul(input, input.transpose(1, 2))
        average_X_XT = torch.mean(X_XT, dim=0)
    else:
        raise ValueError(
            f"Unexpected tensor dimensions for layer {name}: got {input.dim()}, expected 3"
        )
    if name not in layer_inputs:
        layer_inputs[name] = average_X_XT
    else:
        layer_inputs[name] += average_X_XT
    del X_XT, average_X_XT
    torch.cuda.empty_cache()


def save_grad_hook(name, module, grad_input, grad_output):
    grad_output = grad_output[0].detach()
    if grad_output.dim() == 3:
        gradX_gradXT = torch.matmul(grad_output, grad_output.transpose(1, 2))
        average_gradX_gradXT = torch.mean(gradX_gradXT, dim=0)
    else:
        raise ValueError(
            f"Unexpected tensor dimensions for layer {name}: got {grad_output.dim()}, expected 3"
        )
    if name not in layer_gradients:
        layer_gradients[name] = average_gradX_gradXT
    else:
        layer_gradients[name] += average_gradX_gradXT
    del gradX_gradXT, average_gradX_gradXT
    torch.cuda.empty_cache()


def register_hooks(model, model_name):
    global hook_handles_input, hook_handles_grad
    for name, module in model.named_modules():
        if any((param.requires_grad for param in module.parameters())):
            if isinstance(module, nn.Linear):
                if _skip_layer(name, model_name):
                    continue
                handle_forward = module.register_forward_hook(
                    lambda module, input, output, name=name: save_input_hook(
                        name, module, input, output
                    )
                )
                handle_backward = module.register_full_backward_hook(
                    lambda module, grad_input, grad_output, name=name: save_grad_hook(
                        name, module, grad_input, grad_output
                    )
                )
                hook_handles_input.append(handle_forward)
                hook_handles_grad.append(handle_backward)


def unregister_hooks():
    global hook_handles_input, hook_handles_grad
    for handle in hook_handles_input:
        handle.remove()
    for handle in hook_handles_grad:
        handle.remove()
    hook_handles_input = []
    hook_handles_grad = []
    layer_inputs.clear()
    layer_gradients.clear()
    P_dict.clear()
    Q_dict.clear()
    gc.collect()
    torch.cuda.empty_cache()


class CustomLoggingCallback(TrainerCallback):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_iter = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        train_dataloader = kwargs.get("train_dataloader")
        if train_dataloader is not None:
            self.train_iter = iter(train_dataloader)

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step % self.args.calib_iter == 0:
            if self.args.compress:
                print(f"[CALIBRATION] Step {state.global_step}, Epoch {state.epoch}")
                model = kwargs.get("model")
                if model is None:
                    print("In Calibration CallBack: Model not found in kwargs!")
                    return
                loss_fn = nn.CrossEntropyLoss()
                optimizer = kwargs.get("optimizer")
                if optimizer is None:
                    print("Optimizer not found in kwargs!")
                model.disable_compression()
                register_hooks(model, self.args.model_name)
                is_causal_lm = self.args.model_name.startswith("gpt2")
                calib_iter = self.args.calib_batches
                calib_batches = []
                if self.train_iter is None:
                    train_dataloader = kwargs.get("train_dataloader")
                    if train_dataloader is not None:
                        self.train_iter = iter(train_dataloader)
                    else:
                        print(
                            "[CALIBRATION] Warning: train_iter not available, skipping calibration"
                        )
                        unregister_hooks()
                        return
                for batch in islice(self.train_iter, calib_iter):
                    calib_batches.append(batch)
                for calib_batch in calib_batches:
                    num_splits = self.args.num_split
                    inputs = {
                        k: v.to(model.device)
                        for (k, v) in calib_batch.items()
                        if k in ["input_ids", "attention_mask", "token_type_ids", "labels"]
                    }
                    splits = {k: torch.chunk(v, num_splits, dim=0) for (k, v) in inputs.items()}
                    for i in range(num_splits):
                        optimizer.zero_grad()
                        sub_inputs = {k: splits[k][i] for k in splits}
                        outputs = model(**sub_inputs)
                        if is_causal_lm:
                            loss = outputs.loss
                        else:
                            logits = outputs.logits
                            labels = sub_inputs["labels"]
                            loss = loss_fn(logits, labels)
                        loss.backward()
                    del outputs, loss
                    gc.collect()
                    torch.cuda.empty_cache()
                del calib_batches
                gc.collect()
                torch.cuda.empty_cache()
                layer_prefix = _get_layer_prefix(self.args.model_name)
                average_X_XT = None
                average_grad_X_XT = None
                for layer_name, input_sum in layer_inputs.items():
                    if layer_name.startswith(layer_prefix):
                        average_X_XT = input_sum / calib_iter / num_splits
                        P_dict[layer_name] = SVD_expected_value(
                            average_X_XT, self.args.var, self.args.over_sampling
                        )
                for layer_name, grad_sum in layer_gradients.items():
                    if layer_name.startswith(layer_prefix):
                        average_grad_X_XT = grad_sum / calib_iter
                        Q_dict[layer_name] = SVD_expected_value(
                            average_grad_X_XT, self.args.var, self.args.over_sampling
                        )
                output_file_U = f"{self.args.output_dir}/P_dict.json"
                output_file_U_grad = f"{self.args.output_dir}/Q_dict.json"
                log_svd_shapes(state.global_step, P_dict, output_file=output_file_U)
                log_svd_shapes(state.global_step, Q_dict, output_file=output_file_U_grad)
                model.update_compression(P_dict, Q_dict)
                model.to(model.device)
                model.enable_compression()
                del average_X_XT, average_grad_X_XT
                unregister_hooks()
                gc.collect()
                torch.cuda.empty_cache()
                calib_steps = list(range(self.args.calib_batches))
                if state.global_step % self.args.calib_iter not in calib_steps:
                    next(self.train_iter)
