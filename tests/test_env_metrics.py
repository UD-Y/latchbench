"""RJ45 metric compute functions on synthetic per-episode recordings (no Isaac Sim)."""

import math

import numpy as np

from patchbench.metrics import compute_mean, compute_nanmean, compute_rate, compute_stage_funnel


def episodes(*values):
    """Recorder layout: one (1, 1) array per finished episode."""
    return [np.array([[v]], dtype=np.float32) for v in values]


def test_funnel_counts_furthest_stage():
    f = compute_stage_funnel(episodes(0, 1, 2, 5, 5, 3, 4, 5))
    assert f == {"reached": 7 / 8, "contacted": 6 / 8, "aligned": 5 / 8, "seated": 4 / 8, "latched": 3 / 8}


def test_funnel_is_monotone_and_handles_no_episodes():
    f = compute_stage_funnel(episodes(*np.random.default_rng(0).integers(0, 6, 50)))
    vals = list(f.values())
    assert all(a >= b for a, b in zip(vals, vals[1:]))
    assert compute_stage_funnel([]) == dict.fromkeys(f, 0.0)


def test_time_to_latch_ignores_unlatched():
    assert compute_nanmean(episodes(float("nan"), 4.0, 6.0)) == 5.0
    assert math.isnan(compute_nanmean(episodes(float("nan"))))


def test_mean_and_rate():
    assert compute_mean(episodes(1.0, 3.0)) == 2.0
    assert compute_rate(episodes(0, 1, 0, 2)) == 0.5
    assert math.isnan(compute_mean([]))
