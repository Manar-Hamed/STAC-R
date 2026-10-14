# Adapted from INSTANT, copyright (c) 2024 Trung-Hieu Tran. MIT; see LICENSE.txt.
# Upstream revision: 8a04bd43c19cc52984a49fb1b7fd1dae14fb2ffc.
import math
import torch
import numpy as np

dataset_to_num_labels = {
    "cola": 2,
    "sst2": 2,
    "mrpc": 2,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
    "mnli": 3,
}
dataset_fields = {
    "cola": ["sentence"],
    "sst2": ["sentence"],
    "mrpc": ["sentence1", "sentence2"],
    "qqp": ["question1", "question2"],
    "mnli": ["premise", "hypothesis"],
    "qnli": ["question", "sentence"],
    "rte": ["sentence1", "sentence2"],
    "wnli": ["sentence1", "sentence2"],
}
dataset_best_metrics = {
    "cola": "matthews_correlation",
    "sst2": "accuracy",
    "mrpc": "f1",
    "qqp": "f1",
    "mnli": "accuracy",
    "qnli": "accuracy",
    "rte": "accuracy",
    "wnli": "accuracy",
}


def compute_metrics_with_args(args):
    from evaluate import load

    def compute_metrics(eval_pred):
        (predictions, labels) = eval_pred
        preds = predictions.argmax(axis=-1) if predictions.shape[-1] > 1 else predictions.squeeze()
        best_metric = dataset_best_metrics.get(args.dataset, "accuracy")
        metric = load(best_metric)
        return metric.compute(predictions=preds, references=labels)

    return compute_metrics


def load_wikitext103(tokenizer, max_length=512):
    """Load and tokenize WikiText-103 for causal language modeling.

    Returns train and validation datasets with input_ids, attention_mask,
    and labels (= input_ids, for CLM loss computed by the model).
    Uses contiguous chunking (no padding waste).
    """
    from datasets import load_dataset

    raw = load_dataset("wikitext", "wikitext-103-raw-v1")

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)

    tokenized = raw.map(tokenize_fn, batched=True, remove_columns=["text"])

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"]) // max_length * max_length
        result = {
            k: [concatenated[k][i : i + max_length] for i in range(0, total_length, max_length)]
            for k in concatenated.keys()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized.map(group_texts, batched=True)
    lm_datasets.set_format("torch")
    return (lm_datasets["train"], lm_datasets["validation"])


def compute_perplexity(eval_pred):
    """Compute perplexity from CausalLM logits and labels."""
    (logits, labels) = eval_pred
    shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
    shift_labels = labels[:, 1:].reshape(-1)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(torch.from_numpy(shift_logits).float(), torch.from_numpy(shift_labels).long())
    ppl = math.exp(min(loss.item(), 100))
    return {"perplexity": ppl, "loss": loss.item()}
