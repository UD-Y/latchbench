"""results_io: Wilson CI, episode selection, section 6.2 schema (CPU only, no Isaac)."""

import csv
import json
import math

import pytest

from patchbench import results_io as rio


def test_wilson_hand_values():
    lo, hi = rio.wilson_ci(7, 10)  # textbook value: [0.3968, 0.8922]
    assert lo == pytest.approx(0.3968, abs=1e-4) and hi == pytest.approx(0.8922, abs=1e-4)
    lo, hi = rio.wilson_ci(0, 16)  # upper = z^2 / (n + z^2)
    assert lo == 0.0 and hi == pytest.approx(3.8415 / 19.8415, abs=1e-4)
    lo, hi = rio.wilson_ci(16, 16)
    assert hi == 1.0 and lo == pytest.approx(16 / 19.8415, abs=1e-4)
    assert rio.wilson_ci(0, 0) == (0.0, 1.0)


def _ep(env_id, k, stage, ttl=float("nan"), wrong=False, ov=0, peak=1.0):
    return {"env_id": env_id, "episode_in_env": k, "furthest_stage": stage, "success": stage == 5,
            "wrong_port": wrong, "time_to_latch": ttl, "overload_events": ov, "peak_axial_force": peak,
            "tier": "loose", "target_port": env_id % 3, "episode_length": 100}


def test_select_episodes_uses_schedule_slots():
    # 2 envs, n=3 -> slots: (env0,k0)=0, (env1,k0)=1, (env0,k1)=2; env1's 2nd episode (slot 3) is dropped.
    recs = [_ep(1, 0, 1), _ep(0, 0, 2), _ep(1, 1, 5), _ep(0, 1, 3), _ep(0, 2, 4)]
    sel = rio.select_episodes(recs, n=3, num_envs=2)
    assert [e["episode"] for e in sel] == [0, 1, 2]
    assert [e["furthest_stage"] for e in sel] == [2, 1, 3]


def test_summarize_hand_computed():
    eps = [_ep(0, 0, 5, ttl=4.0, peak=3.0), _ep(1, 0, 5, ttl=6.0, ov=1, peak=5.0), _ep(2, 0, 3, wrong=True),
           _ep(3, 0, 1)]
    r = rio.summarize(eps, policy="telemetry_spiral", tier="loose", seed=0, extra={"pose_noise_mm": 1.0})
    assert list(r)[:13] == ["policy", "tier", "n", "seed", "funnel", "success", "ci95", "mean_time_to_latch_s",
                            "wrong_port", "overload_events", "git_sha", "so101_contact_version", "physics"]
    assert r["n"] == 4 and r["physics"] == "physx" and r["pose_noise_mm"] == 1.0
    assert r["funnel"] == {"reached": 1.0, "contacted": 0.75, "aligned": 0.75, "seated": 0.5, "latched": 0.5}
    assert r["success"] == 0.5
    lo, hi = rio.wilson_ci(2, 4)
    assert r["ci95"] == [round(lo, 4), round(hi, 4)]
    assert r["mean_time_to_latch_s"] == 5.0
    assert r["wrong_port"] == 0.25 and r["overload_events"] == 0.25
    assert r["mean_peak_axial_force_n"] == pytest.approx(2.5)


def test_summarize_no_latch_and_empty():
    r = rio.summarize([_ep(0, 0, 2)], policy="open_loop", tier="spec", seed=1)
    assert r["success"] == 0.0 and r["mean_time_to_latch_s"] is None
    r = rio.summarize([], policy="open_loop", tier="spec", seed=1)
    assert r["n"] == 0 and r["ci95"] == [0.0, 1.0]


def test_write_and_load(tmp_path):
    eps = [_ep(0, 0, 5, ttl=2.0), _ep(1, 0, 0)]
    r = rio.summarize(eps, policy="open_loop", tier="loose", seed=0)
    p = rio.write_result(tmp_path, r, eps)
    assert p == tmp_path / "open_loop" / "loose.json"
    assert json.loads(p.read_text())["success"] == 0.5
    rows = list(csv.DictReader(open(tmp_path / "open_loop" / "loose_episodes.csv")))
    assert [row["stage"] for row in rows] == ["latched", "none"]
    (tmp_path / "open_loop" / "notes.json").write_text('{"x": 1}')  # not a result: ignored
    assert [x["tier"] for x in rio.load_results(tmp_path)] == ["loose"]
    png = rio.plot_funnel([r], tmp_path / "open_loop" / "funnel.png")
    assert png.stat().st_size > 1000


def test_nan_time_to_latch_ignored():
    eps = [_ep(0, 0, 5, ttl=float("nan")), _ep(1, 0, 5, ttl=3.0)]
    r = rio.summarize(eps, policy="x", tier="loose", seed=0)
    assert r["mean_time_to_latch_s"] == 3.0 and not math.isnan(r["success"])
