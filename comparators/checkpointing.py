from __future__ import annotations


def enable_gradient_checkpointing(model) -> bool:
    """Enable model-native gradient checkpointing when available."""
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        return True
    return False
