"""Full-batch ogbn-arxiv/GCN activation-surface limitation experiment.

Linear activation compression leaves graph message-passing storage unchanged.
The reported aggregate accuracies do not specify every training setting; this
configuration therefore supports evaluation of the regime without promising
exact agreement with those aggregate values.
"""

from __future__ import annotations
import dataclasses
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import torch
from torch import nn, optim
from .training import ConvergenceConfig, replace_linear_with_stacr
from stacr.adaptive import STACRLinear, set_stacr_runtime_state
from stacr.protocols.utils import set_all_seeds


def setup_ogbn_arxiv_data(config: ConvergenceConfig) -> Tuple[Any, Any, Dict]:
    """Load ogbn-arxiv dataset for node classification.

    ogbn-arxiv is a citation network:
    - 169,343 nodes (papers)
    - 1,166,243 edges (citations)
    - 128-dim node features (word2vec of title/abstract)
    - 40 classes (arXiv subject areas)

    Returns:
        (data, split_idx, dataset_info)
    """
    try:
        from ogb.nodeproppred import PygNodePropPredDataset
    except ImportError as e:
        raise ImportError("ogbn-arxiv requires: pip install ogb torch-geometric") from e
    from torch_geometric.data import Data
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage

    if not config.quiet:
        print("Loading ogbn-arxiv dataset...")
    # OGB caches PyG objects. Keep the weights-only loader enabled and scope the
    # known container classes to this load instead of changing process defaults.
    registered = set(torch.serialization.get_safe_globals())
    cache_types = [Data, DataEdgeAttr, DataTensorAttr, GlobalStorage]
    with torch.serialization.safe_globals([cls for cls in cache_types if cls not in registered]):
        dataset = PygNodePropPredDataset(
            name="ogbn-arxiv", root=str(Path(os.environ.get("STACR_CACHE", ".cache/stacr")) / "ogb")
        )
        data = dataset[0]
        split_idx = dataset.get_idx_split()
    info = {
        "num_nodes": data.num_nodes,
        "num_edges": data.num_edges,
        "num_features": dataset.num_features,
        "num_classes": dataset.num_classes,
        "train_size": len(split_idx["train"]),
        "val_size": len(split_idx["valid"]),
        "test_size": len(split_idx["test"]),
    }
    if not config.quiet:
        print(
            f"ogbn-arxiv: {info['num_nodes']} nodes, {info['num_edges']} edges, {info['num_classes']} classes"
        )
    return (data, split_idx, info)


def create_gcn_model(
    in_features: int,
    hidden_features: int,
    out_features: int,
    num_layers: int = 3,
    dropout: float = 0.5,
) -> nn.Module:
    """Create a GCN model for node classification.

    Architecture follows standard GCN:
    - GCNConv layers with ReLU activation
    - Dropout between layers
    - No activation after final layer (logits)

    Args:
        in_features: Input feature dimension (128 for ogbn-arxiv)
        hidden_features: Hidden layer dimension (256 typical)
        out_features: Number of classes (40 for ogbn-arxiv)
        num_layers: Number of GCN layers
        dropout: Dropout probability
    """
    try:
        from torch_geometric.nn import GCNConv
    except ImportError as e:
        raise ImportError("GCN requires: pip install torch-geometric") from e
    import torch.nn.functional as F

    class GCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            self.convs.append(GCNConv(in_features, hidden_features))
            self.bns.append(nn.BatchNorm1d(hidden_features))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_features, hidden_features))
                self.bns.append(nn.BatchNorm1d(hidden_features))
            self.convs.append(GCNConv(hidden_features, out_features))
            self.dropout = dropout

        def forward(self, x, edge_index):
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, edge_index)
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[-1](x, edge_index)
            return x

        def reset_parameters(self):
            for conv in self.convs:
                conv.reset_parameters()
            for bn in self.bns:
                bn.reset_parameters()

    model = GCN()
    return model


def get_compression_targets_gnn(model: nn.Module) -> List[str]:
    """Get linear layers in GNN for compression.

    GCNConv internally uses linear transformations.
    We target the weight matrices in each conv layer.
    """
    targets = []
    seen = set()
    for name, module in model.named_modules():
        if "convs" in name and hasattr(module, "lin"):
            target = f"{name}.lin"
            if target not in seen:
                targets.append(target)
                seen.add(target)
        cls_name = module.__class__.__name__
        mod_name = module.__class__.__module__
        if cls_name == "Linear" and "torch_geometric" in mod_name:
            if name not in seen:
                targets.append(name)
                seen.add(name)
    return targets


def train_gnn_epoch(
    model: nn.Module,
    data: Any,
    train_idx: torch.Tensor,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Train GNN for one epoch (full-batch).

    Full-batch training processes all nodes at once.
    This is memory-intensive but standard for small graphs.
    """
    import torch.nn.functional as F

    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_idx], data.y[train_idx].squeeze())
    loss.backward()
    optimizer.step()
    pred = out[train_idx].argmax(dim=1)
    correct = (pred == data.y[train_idx].squeeze()).sum().item()
    acc = correct / len(train_idx)
    return {"loss": loss.item(), "accuracy": acc * 100}


@torch.no_grad()
def evaluate_gnn(
    model: nn.Module, data: Any, split_idx: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """Evaluate GNN on train/val/test splits."""
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    results = {}
    for split in ["train", "valid", "test"]:
        idx = split_idx[split]
        correct = (pred[idx] == data.y[idx].squeeze()).sum().item()
        results[f"{split}_acc"] = correct / len(idx) * 100
    return results


def run_graph_experiment(config, logger):
    if config.method not in {"baseline", "stacr"}:
        raise ValueError("The full-batch graph protocol supports baseline or stacr")
    if config.use_ddp or config.use_tensor_parallel:
        raise ValueError("The full-batch graph protocol runs on one device")
    from stacr.adaptive import get_basis_pool

    get_basis_pool().clear()
    set_all_seeds(config.seed)
    device = torch.device(config.device)
    data, split_idx, info = setup_ogbn_arxiv_data(config)
    model = create_gcn_model(info["num_features"], 256, info["num_classes"], 3, 0.5)
    config.effective_total_steps = min(config.epochs, config.max_steps or config.epochs)
    if config.method == "stacr":
        replace_linear_with_stacr(model, config, get_compression_targets_gnn(model))
    model.to(device)
    data = data.to(device)
    split_idx = {k: v.to(device) for k, v in split_idx.items()}
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for step in range(config.effective_total_steps):
        set_stacr_runtime_state(model, optimizer_step=step, microbatch_index=0, accum_steps=1)
        metrics = train_gnn_epoch(model, data, split_idx["train"], optimizer, device)
        metrics.update(evaluate_gnn(model, data, split_idx))
        metrics["epoch"] = step + 1
        logger.log_metrics(metrics)
        history.append(metrics)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    if not history:
        raise ValueError("epochs/max_steps must allow at least one graph update")
    best = max(history, key=lambda row: row["valid_acc"])
    result = {
        "status": "success",
        "results": {
            "epochs_completed": len(history),
            "best_validation_accuracy": best["valid_acc"],
            "test_accuracy_at_best_validation": best["test_acc"],
            "final_validation_accuracy": history[-1]["valid_acc"],
            "seconds": elapsed,
            "peak_gib": torch.cuda.max_memory_allocated(device) / 2**30
            if device.type == "cuda"
            else None,
            "wrapped_layers": sum(isinstance(m, STACRLinear) for m in model.modules()),
            "dataset": info,
        },
    }
    if config.save_final_model:
        folder = Path(logger.run_dir) / "checkpoint"
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), folder / "model.pt")
    return result
