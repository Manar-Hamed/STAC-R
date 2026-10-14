from __future__ import annotations


def load_model_and_tokenizer(model_name: str, task: str, **options):
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

    revision = options.get("revision")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=revision, local_files_only=options.get("local_files_only", False)
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has neither a padding token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {"revision": revision, "local_files_only": options.get("local_files_only", False)}
    if options.get("attn_implementation"):
        kwargs["attn_implementation"] = options["attn_implementation"]
    if options.get("dtype") is not None:
        kwargs["torch_dtype"] = options["dtype"]
    if task == "sequence_classification":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, **kwargs
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    return model, tokenizer
