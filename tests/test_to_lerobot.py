"""End to end: 3 synthetic raw episodes (random 240x320 images) -> LeRobot v3 -> LeRobotDataset load.

Needs lerobot==0.6.1, so it only runs in the LeRobot venv; elsewhere it is skipped:
    ~/work/.venv-lerobot/bin/python -m pytest -q -s tests/test_to_lerobot.py
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from latchbench import raw_format as rf

pytest.importorskip("lerobot.datasets.lerobot_dataset")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "to_lerobot.py"
H, W, T = 240, 320, 20


def make_synthetic(root: Path, n: int = 3, failed: int = 1) -> rf.Manifest:
    rng = np.random.default_rng(0)
    m = rf.Manifest(info={"generator": "synthetic test"})
    for k in range(n + failed):
        port, ok = k % 3, k < n
        ep = rf.RawEpisode(
            front=rng.integers(0, 256, (T, H, W, 3), dtype=np.uint8),
            wrist=rng.integers(0, 256, (T, H, W, 3), dtype=np.uint8),
            state=rng.normal(0, 30, (T, 6)), action=rng.normal(0, 30, (T, 6)),
            servo_load=rng.integers(-1000, 1001, (T, 6)), servo_current=rng.integers(0, 400, (T, 6)),
            task=rf.task_string(port), tier="loose", port=port, success=ok, seed=k,
        )
        i = m.next_index()
        m.add(i, rf.write_episode(root, i, ep), ep)
    m.save(root)
    return m


def test_convert_and_load(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    raw, root, repo = tmp_path / "raw", tmp_path / "ds", "local/latchbench-test"
    make_synthetic(raw)
    cmd = [sys.executable, str(SCRIPT), "--raw", str(raw), "--repo_id", repo, "--root", str(root)]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    print(out)
    assert "wrote 3 episodes" in out

    ds = LeRobotDataset(repo, root=root)
    assert ds.num_episodes == 3 and ds.num_frames == 3 * T and ds.fps == rf.FPS
    assert ds.meta.robot_type == "so101_follower"
    assert ds.features["observation.state"]["names"][0] == "shoulder_pan.pos"
    item = ds[T + 1]  # second episode
    assert tuple(item["observation.images.front"].shape) == (3, H, W)
    assert tuple(item["observation.images.wrist"].shape) == (3, H, W)
    assert tuple(item["observation.state"].shape) == (6,) and tuple(item["action"].shape) == (6,)
    assert tuple(item["observation.servo_load"].shape) == (6,)
    assert item["task"] == "insert the cable into the middle port"

    # Numeric features are copied exactly (images are lossy-encoded video and are not compared).
    second = rf.read_episode(raw / "episodes" / "ep_000001.npz")
    np.testing.assert_allclose(item["observation.state"].numpy(), second.state[1], rtol=0, atol=1e-6)
    np.testing.assert_array_equal(item["observation.servo_load"].numpy(), second.servo_load[1])

    side = [json.loads(line) for line in (root / "meta" / "latchbench_episodes.jsonl").read_text().splitlines()]
    assert [s["port"] for s in side] == [0, 1, 2] and all(s["tier"] == "loose" for s in side)
    assert not (tmp_path / "ds.partial").exists()

    # Second run reuses the complete dataset.
    out2 = subprocess.run(cmd + ["--no_check"], check=True, capture_output=True, text=True).stdout
    assert "skipped" in out2


def test_state_with_load(tmp_path):
    """--state_with_load: observation.state = 6 positions + 6 signed Present_Load counts, positions first."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    raw, root, repo = tmp_path / "raw", tmp_path / "ds", "local/latchbench-test-load"
    make_synthetic(raw, n=2, failed=0)
    cmd = [sys.executable, str(SCRIPT), "--raw", str(raw), "--repo_id", repo, "--root", str(root),
           "--state_with_load", "--no_check"]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    ds = LeRobotDataset(repo, root=root)
    names = ds.features["observation.state"]["names"]
    assert len(names) == 12 and names[0] == "shoulder_pan.pos" and names[6] == "shoulder_pan.load"
    item = ds[T + 3]
    second = rf.read_episode(raw / "episodes" / "ep_000001.npz")
    np.testing.assert_allclose(item["observation.state"][:6].numpy(), second.state[3], rtol=0, atol=1e-5)
    np.testing.assert_array_equal(item["observation.state"][6:].numpy(), second.servo_load[3].astype(np.float32))
    np.testing.assert_array_equal(item["observation.servo_load"].numpy(), second.servo_load[3])
