"""leaderboard.py / gap closure from fake result JSONs (CPU only)."""

import importlib.util
import json
from pathlib import Path

import pytest

from latchbench import results_io as rio

ROOT = Path(__file__).resolve().parents[1]


def _fake(d: Path, policy: str, tier: str, success: float, n: int = 100):
    lo, hi = rio.wilson_ci(round(success * n), n)
    r = {"policy": policy, "tier": tier, "n": n, "seed": 0,
         "funnel": {"reached": 1.0, "contacted": 0.9, "aligned": max(success, 0.5), "seated": success,
                    "latched": success},
         "success": success, "ci95": [lo, hi], "mean_time_to_latch_s": 5.0 if success else None, "wrong_port": 0.0,
         "overload_events": 0.0, "git_sha": "abc", "servosense_version": "0.1.0", "physics": "physx"}
    (d / policy).mkdir(parents=True, exist_ok=True)
    (d / policy / f"{tier}.json").write_text(json.dumps(r))


def test_gap_closure_point_and_ci():
    g = rio.gap_closure((20, 100), (50, 100), (80, 100))
    assert g["value"] == pytest.approx(0.5)
    lo, hi = g["ci95"]
    assert lo < 0.5 < hi and 0.2 < lo and hi < 0.8


def test_gap_closure_guard_small_denominator():
    g = rio.gap_closure((50, 100), (52, 100), (52, 100))
    assert g["value"] is None and g["ci95"] is None and "not reported" in g["note"]
    g = rio.gap_closure((60, 100), (50, 100), (40, 100))  # oracle worse than open loop
    assert g["value"] is None


def test_leaderboard_build(tmp_path):
    for p, s in [("open_loop", 0.2), ("telemetry_spiral", 0.5), ("oracle_spiral", 0.8)]:
        _fake(tmp_path, p, "loose", s)
    _fake(tmp_path, "open_loop", "medium", 0.1)  # medium lacks the other two: no gap row
    _fake(tmp_path, "expert", "loose", 0.9)
    spec = importlib.util.spec_from_file_location("leaderboard", ROOT / "scripts" / "leaderboard.py")
    lb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lb)
    md, png = lb.build(tmp_path)
    text = md.read_text()
    assert "| open_loop | 0.20 [" in text and "| medium |" not in text.split("## Gap closure")[1].split("##")[0]
    assert "**0.50** [" in text
    # policy order: scripted three first, then expert
    rows = [line.split("|")[1].strip() for line in text.split("## Gap closure")[0].splitlines() if line.startswith("| ")]
    assert rows[1:] == ["open_loop", "telemetry_spiral", "oracle_spiral", "expert"]
    assert png is not None and png.stat().st_size > 1000


def test_leaderboard_empty(tmp_path):
    assert "Needs open_loop" in rio.leaderboard_markdown([])
