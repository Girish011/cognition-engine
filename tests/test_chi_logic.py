"""CHI pure-logic tests (no model; mlx is stubbed off Apple Silicon)."""

import pytest

from cognition_engine.chi import (  # noqa: E402
    DRIFT_BASELINE_NEUTRAL,
    DRIFT_BASELINE_TURNS,
    CHIMonitor,
)
from cognition_engine.steering.injector import compute_adaptive_alpha  # noqa: E402


@pytest.fixture
def monitor():
    return CHIMonitor(model=None, tokenizer=None, original_task="task")


def test_drift_baseline_phase_is_neutral(monitor):
    for _ in range(DRIFT_BASELINE_TURNS):
        assert monitor._compute_drift_health(0.4) == DRIFT_BASELINE_NEUTRAL
    assert monitor.drift_baseline_mean == pytest.approx(0.4)


def test_drift_health_tracks_drop_below_baseline(monitor):
    for v in (0.40, 0.42, 0.38):
        monitor._compute_drift_health(v)
    assert monitor._compute_drift_health(0.45) == 1.0  # above baseline: no drift
    assert monitor._compute_drift_health(0.10) < 0.2  # far below baseline


@pytest.mark.parametrize(
    "chi,level",
    [(0.95, 1), (0.7, 2), (0.4, 3), (0.1, 4)],
)
def test_intervention_levels(monitor, chi, level):
    monitor.chi_history.append(chi)
    assert monitor.get_intervention_level() == level


def test_adaptive_alpha_is_monotonic_and_capped():
    assert compute_adaptive_alpha(0.95) == 0.0
    alphas = [compute_adaptive_alpha(c / 10) for c in range(9, 0, -1)]
    assert alphas == sorted(alphas)
    assert max(alphas) <= 1.5 * 0.5 + 1e-9
