import pytest

from stacr.rank_controller import (
    choose_runtime_rank,
    choose_sensitivity_rank,
    choose_variance_rank,
    should_adjust_rank,
)


def test_should_adjust_rank_initial_step():
    assert should_adjust_rank(schedule_step=0, last_adjust_step=-1, adjust_interval=100) is True


def test_should_adjust_rank_respects_interval():
    assert should_adjust_rank(schedule_step=50, last_adjust_step=0, adjust_interval=100) is False
    assert should_adjust_rank(schedule_step=100, last_adjust_step=0, adjust_interval=100) is True


def test_choose_variance_rank_shrinks_to_energy_prefix():
    decision = choose_variance_rank(
        coeff_energy_by_rank=[9.0, 1.0, 0.2, 0.1],
        current_rank=4,
        rank_min=1,
        rank_max=4,
        max_available_rank=4,
        variance_threshold=0.90,
    )
    assert decision.active_rank == 2
    assert decision.reason_code == "variance_shrink"


def test_choose_variance_rank_cold_start_keeps_current_rank():
    decision = choose_variance_rank(
        coeff_energy_by_rank=None,
        current_rank=8,
        rank_min=4,
        rank_max=16,
        max_available_rank=16,
        variance_threshold=0.95,
    )
    assert decision.active_rank == 8
    assert decision.reason_code == "variance_cold_start"


def test_choose_sensitivity_rank_grows_above_high_band():
    decision = choose_sensitivity_rank(
        residual_energy=0.30,
        current_rank=8,
        rank_min=4,
        rank_max=16,
        max_available_rank=16,
        low_threshold=0.03,
        high_threshold=0.10,
        step_fraction=0.25,
    )
    assert decision.active_rank == 10
    assert decision.reason_code == "sensitivity_grow"


def test_choose_sensitivity_rank_shrinks_below_low_band():
    decision = choose_sensitivity_rank(
        residual_energy=0.01,
        current_rank=8,
        rank_min=4,
        rank_max=16,
        max_available_rank=16,
        low_threshold=0.03,
        high_threshold=0.10,
        step_fraction=0.25,
    )
    assert decision.active_rank == 6
    assert decision.reason_code == "sensitivity_shrink"


def test_choose_runtime_rank_dispatches_variance():
    decision = choose_runtime_rank(
        rank_mode="variance",
        current_rank=8,
        rank_min=4,
        rank_max=16,
        max_available_rank=16,
        layer_state={"last_coeff_energy_by_rank": [10.0, 1.0, 0.1, 0.1, 0.1]},
        variance_threshold=0.9,
    )
    assert decision.active_rank == 4
    assert decision.reason_code.startswith("variance_")


def test_choose_runtime_rank_dispatches_sensitivity():
    decision = choose_runtime_rank(
        rank_mode="sensitivity",
        current_rank=8,
        rank_min=4,
        rank_max=16,
        max_available_rank=16,
        layer_state={"last_residual_energy": 0.20},
        sensitivity_low=0.03,
        sensitivity_high=0.10,
        step_fraction=0.25,
    )
    assert decision.active_rank == 10
    assert decision.reason_code == "sensitivity_grow"


def test_choose_variance_rank_respects_max_available_prefix():
    decision = choose_variance_rank(
        coeff_energy_by_rank=[1.0, 1.0, 100.0],
        current_rank=3,
        rank_min=1,
        rank_max=8,
        max_available_rank=2,
        variance_threshold=0.9,
    )
    assert decision.active_rank == 2
    assert decision.reason_code in {"variance_keep", "variance_shrink", "variance_grow"}


def test_choose_variance_rank_rejects_invalid_energy():
    with pytest.raises(ValueError, match="Invalid coefficient energy"):
        choose_variance_rank(
            coeff_energy_by_rank=[1.0, float("nan")],
            current_rank=2,
            rank_min=1,
            rank_max=8,
            max_available_rank=8,
            variance_threshold=0.9,
        )


def test_choose_runtime_rank_fixed_reports_change_when_clamped():
    decision = choose_runtime_rank(
        rank_mode="fixed",
        current_rank=32,
        rank_min=4,
        rank_max=16,
        max_available_rank=16,
        layer_state={},
    )
    assert decision.active_rank == 16
    assert decision.changed is True


def test_choose_runtime_rank_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unsupported rank_mode"):
        choose_runtime_rank(
            rank_mode="bogus",
            current_rank=8,
            rank_min=4,
            rank_max=16,
            max_available_rank=16,
            layer_state={},
        )
