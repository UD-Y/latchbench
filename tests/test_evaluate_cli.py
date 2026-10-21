"""scripts/evaluate.py orchestration without Isaac Sim: argv, schedule quota, raw JSONL -> results + leaderboard."""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("evaluate", ROOT / "scripts" / "evaluate.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["evaluate"] = m
    spec.loader.exec_module(m)
    return m


ev = _load()


def test_quota_fills_schedule():
    assert ev.episode_quota(16, 16) == [1] * 16
    assert ev.episode_quota(100, 16) == [7] * 4 + [6] * 12
    assert sum(ev.episode_quota(100, 16)) == 100
    assert ev.episode_quota(3, 8) == [1, 1, 1, 0, 0, 0, 0, 0]


def test_args_and_argv(tmp_path):
    a = ev.parse_args(["--policy", "open_loop", "--tiers", "loose,spec", "--n", "16", "--out", str(tmp_path)])
    assert a.tiers == ["loose", "spec"] and a.name == "open_loop" and not a.enable_cameras
    c = ev.worker_config(a, "spec")
    argv = ev.arena_base_argv(c) + ev.arena_env_argv(c)
    assert argv[:4] == ["--num_envs", "16", "--seed", "0"]
    i = argv.index(ev.ENV_NAME)
    assert argv[i - 1] == ev.ENV_CLASS and argv[i + 1:i + 3] == ["--tier", "spec"]
    env_argv = ev.arena_env_argv(c, policy_adds_noise=True)
    assert env_argv[env_argv.index("--port_est_noise_mm") + 1] == "0.0"  # noise applied once, by the policy
    env_argv = ev.arena_env_argv(c, policy_adds_noise=False)
    assert env_argv[env_argv.index("--port_est_noise_mm") + 1] == "1.0"
    b = ev.parse_args(["--policy", "lerobot", "--policy.path", "user/latchbench-groot-n17", "--out", str(tmp_path)])
    assert b.enable_cameras and b.name == "lerobot_latchbench-groot-n17"
    assert "--enable_cameras" in ev.arena_base_argv(ev.worker_config(b, "loose"))


def _fake_worker(cfg, cfg_path, log_path):
    """Writes what Arena's recorder would: core fields + StageTracker summary, more episodes than scheduled."""
    rows = []
    for env_id in range(cfg["num_envs"]):
        for k in range(3):
            stage = 5 if (env_id + k) % 2 == 0 else 3
            rows.append({"job_name": "x", "env_id": env_id, "episode_in_env": k, "success": stage == 5,
                         "episode_length": 300, "furthest_stage": stage, "wrong_port": False,
                         "peak_axial_force": 4.0, "time_to_latch": 5.0 if stage == 5 else None, "overload_events": 0,
                         "tier": cfg["tier"], "target_port": env_id % 3})
    Path(cfg["jsonl"]).write_text("".join(json.dumps(r) + "\n" for r in rows))
    Path(cfg["meta"]).write_text(json.dumps({"wall_time_s": 1.0, "sim_steps": 900}))


def test_end_to_end_without_isaac(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "launch_worker", _fake_worker)
    ev.main(["--policy", "open_loop", "--tiers", "loose", "--n", "6", "--num_envs", "4", "--out", str(tmp_path)])
    r = json.loads((tmp_path / "open_loop" / "loose.json").read_text())
    # slots 0..5 = (env0,k0) (env1,k0) (env2,k0) (env3,k0) (env0,k1) (env1,k1): stages 5 3 5 3 3 5
    assert r["n"] == 6 and r["success"] == 0.5 and r["funnel"]["aligned"] == 1.0
    assert r["pose_noise_mm"] == 1.0 and r["sim_steps"] == 900 and r["physics"] == "physx"
    assert (tmp_path / "open_loop" / "loose_episodes.csv").exists()
    assert (tmp_path / "open_loop" / "funnel.png").exists()
    assert "| open_loop | 0.50 [" in (tmp_path / "LEADERBOARD.md").read_text()
    assert (tmp_path / "leaderboard.png").exists()


def test_dry_run(tmp_path, capsys):
    ev.main(["--policy", "telemetry_spiral", "--tiers", "loose,medium", "--out", str(tmp_path), "--dry_run"])
    out = capsys.readouterr().out
    assert out.count("arena argv:") == 2 and "--tier medium" in out
