"""Arena metrics for the RJ45 task: stage funnel, peak axial force, time to latch, servo overload events, wrong port.

Pattern: ``isaaclab_arena/metrics/success_rate.py`` (record once per episode in ``record_pre_reset``, skipping the
initial all-env reset) with the compute functions kept pure numpy so they are unit-testable. The recorded values come
from ``patchbench.predicates.StageTracker.summary()`` through the task's per-env state (``patchbench.task.get_state``).
"""

from __future__ import annotations

from dataclasses import MISSING

import numpy as np

STAGE_NAMES = ("reached", "contacted", "aligned", "seated", "latched")


# ----------------------------------------------------------------------------------------- compute (pure numpy)


def _per_episode(recorded: list[np.ndarray]) -> np.ndarray:
    """One value per episode (each recorded array holds one row: the episode's summary)."""
    if not recorded:
        return np.zeros(0)
    return np.concatenate([np.asarray(r, dtype=float).reshape(-1) for r in recorded])


def compute_stage_funnel(recorded: list[np.ndarray]) -> dict[str, float]:
    """Fraction of episodes whose furthest stage is at least each stage (``predicates.funnel``)."""
    furthest = _per_episode(recorded)
    n = max(1, furthest.size)
    return {name: float((furthest >= k + 1).sum()) / n for k, name in enumerate(STAGE_NAMES)}


def compute_mean(recorded: list[np.ndarray]) -> float:
    v = _per_episode(recorded)
    return float(v.mean()) if v.size else float("nan")


def compute_nanmean(recorded: list[np.ndarray]) -> float:
    """Mean over episodes where the value is defined (time to latch: latched episodes only); NaN if none."""
    v = _per_episode(recorded)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


def compute_rate(recorded: list[np.ndarray]) -> float:
    v = _per_episode(recorded)
    return float((v > 0).mean()) if v.size else float("nan")


# ----------------------------------------------------------------------------------------- Arena / Isaac Lab glue

try:  # the classes below need Isaac Lab; the compute functions above do not
    from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
    from isaaclab.utils.configclass import configclass
    from isaaclab_arena.metrics.metric_base import MetricBase
    from isaaclab_arena.metrics.metric_term_cfg import MetricTermCfg
except ImportError:  # pragma: no cover - CPU-only test env
    RecorderTerm = RecorderTermCfg = MetricBase = MetricTermCfg = None

if MetricBase is not None:

    class EpisodeSummaryRecorder(RecorderTerm):
        """Records one ``StageTracker.summary()`` entry per finished episode, just before its reset."""

        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self.name, self.key = cfg.name, cfg.key
            self.first_reset = True

        def record_pre_reset(self, env_ids):
            if self.first_reset:  # the initial reset of all envs ends no episode
                self.first_reset = False
                return None, None
            from patchbench.task import get_state

            value = get_state(self._env).tracker.summary()[self.key][env_ids]
            return self.name, value.float().unsqueeze(-1)

    @configclass
    class EpisodeSummaryRecorderCfg(RecorderTermCfg):
        class_type: type = EpisodeSummaryRecorder
        name: str = MISSING
        key: str = MISSING

    class _SummaryMetric(MetricBase):
        key: str
        compute = staticmethod(compute_mean)

        def get_recorder_term_cfg(self):
            return EpisodeSummaryRecorderCfg(name=self.recorder_term_name, key=self.key)

        def get_metric_term_cfg(self):
            return MetricTermCfg(compute_metric_func=self.compute, params={}, recorder_term_name=self.recorder_term_name)

    class StageFunnelMetric(_SummaryMetric):
        """Fraction of episodes reaching at least each stage: reached, contacted, aligned, seated, latched."""

        name = "stage_funnel"
        recorder_term_name = "rj45_furthest_stage"
        key = "furthest_stage"
        compute = staticmethod(compute_stage_funnel)

    class PeakAxialForceMetric(_SummaryMetric):
        """Mean over episodes of the peak axial plug-jack force on the target port [N] (control-rate samples)."""

        name = "peak_axial_force_N"
        recorder_term_name = "rj45_peak_axial_force"
        key = "peak_axial_force"

    class TimeToLatchMetric(_SummaryMetric):
        """Mean time from reset to latch over latched episodes [s]; NaN when none latched."""

        name = "mean_time_to_latch_s"
        recorder_term_name = "rj45_time_to_latch"
        key = "time_to_latch"
        compute = staticmethod(compute_nanmean)

    class OverloadEventMetric(_SummaryMetric):
        """Mean number of STS3215 overload-protection onsets (any joint) per episode."""

        name = "overload_events"
        recorder_term_name = "rj45_overload_events"
        key = "overload_events"

    class WrongPortRateMetric(_SummaryMetric):
        """Fraction of episodes in which the plug entered a non-target port."""

        name = "wrong_port"
        recorder_term_name = "rj45_wrong_port"
        key = "wrong_port"
        compute = staticmethod(compute_rate)
