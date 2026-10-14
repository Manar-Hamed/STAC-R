import torch
import pytest


def test_graph_cache_loads_with_weights_only_and_scoped_classes(monkeypatch, tmp_path):
    pytest.importorskip("torch_geometric")
    ogb = pytest.importorskip("ogb.nodeproppred")
    from torch_geometric.data import Data
    from experiments.protocols import graph
    from experiments.protocols.training import ConvergenceConfig

    cached = tmp_path / "graph.pt"
    torch.save(
        Data(
            x=torch.ones(3, 2),
            y=torch.tensor([[0], [1], [0]]),
            edge_index=torch.tensor([[0, 1], [1, 2]]),
        ),
        cached,
    )
    safe_before = set(torch.serialization.get_safe_globals())

    class CachedDataset:
        num_features = 2
        num_classes = 2

        def __init__(self, name, root):
            assert name == "ogbn-arxiv"
            assert root == str(tmp_path / "ogb")
            self.data = torch.load(cached, weights_only=True)

        def __getitem__(self, index):
            assert index == 0
            return self.data

        def get_idx_split(self):
            return {
                "train": torch.tensor([0]),
                "valid": torch.tensor([1]),
                "test": torch.tensor([2]),
            }

    monkeypatch.setenv("STACR_CACHE", str(tmp_path))
    monkeypatch.setattr(ogb, "PygNodePropPredDataset", CachedDataset)
    data, split, info = graph.setup_ogbn_arxiv_data(ConvergenceConfig(quiet=True))
    assert data.num_nodes == 3
    assert split["valid"].tolist() == [1]
    assert info["num_features"] == 2
    assert set(torch.serialization.get_safe_globals()) == safe_before


@pytest.mark.parametrize("method", ["baseline", "stacr"])
def test_full_batch_graph_protocol_on_tiny_graph(monkeypatch, tmp_path, method):
    pytest.importorskip("torch_geometric")
    from torch_geometric.data import Data
    from experiments.protocols import graph
    from experiments.protocols.logging import RunLogger
    from experiments.protocols.training import ConvergenceConfig

    torch.manual_seed(42)
    count = 20
    nodes = torch.arange(count)
    edges = torch.stack([nodes.repeat(2), torch.cat([(nodes + 1) % count, (nodes - 1) % count])])
    data = Data(x=torch.randn(count, 8), y=torch.randint(0, 3, (count, 1)), edge_index=edges)
    splits = {"train": nodes[:12], "valid": nodes[12:16], "test": nodes[16:]}
    monkeypatch.setattr(
        graph,
        "setup_ogbn_arxiv_data",
        lambda cfg: (data, splits, {"num_features": 8, "num_classes": 3}),
    )
    cfg = ConvergenceConfig(
        task="ogbn-arxiv",
        model="gcn",
        method=method,
        rank=4,
        epochs=2,
        device="cpu",
        quiet=True,
        stacr_min_features=1,
        stacr_basis_dtype="float32",
    )
    result = graph.run_graph_experiment(cfg, RunLogger(tmp_path))
    assert result["status"] == "success"
    assert result["results"]["epochs_completed"] == 2
    assert result["results"]["wrapped_layers"] == (3 if method == "stacr" else 0)
    assert 0 <= result["results"]["best_validation_accuracy"] <= 100
