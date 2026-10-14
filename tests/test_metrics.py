import torch
import pytest

from stacr.metrics import cosine_similarity, relative_mse


def test_metrics_identity():
    x = torch.randn(4, 8)
    assert float(cosine_similarity(x, x)) > 0.999
    assert float(relative_mse(x, x)) == 0.0


def test_cosine_zero_gradients_are_not_false_agreement():
    zero, nonzero = torch.zeros(5), torch.ones(5)
    assert cosine_similarity(zero, zero) == 1
    assert cosine_similarity(zero, nonzero) == 0
    assert cosine_similarity(nonzero, zero) == 0


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_cosine_retains_direction_for_extreme_magnitudes(scale):
    a = torch.tensor([scale, 0.0], dtype=torch.float64)
    b = torch.tensor([0.0, scale], dtype=torch.float64)
    assert cosine_similarity(a, b) == 0
    assert cosine_similarity(a, a) == 1


def test_nonfinite_cosine_raises_instead_of_reporting_agreement():
    with pytest.raises(ValueError, match="finite"):
        cosine_similarity(torch.tensor([float("nan")]), torch.ones(1))
