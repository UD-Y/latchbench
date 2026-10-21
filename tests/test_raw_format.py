"""Raw episode format: round trip, validation, manifest resume. numpy only (no Isaac Sim, no LeRobot)."""

import json

import numpy as np
import pytest

from latchbench import raw_format as rf


def synthetic_episode(rng, t=12, hw=(24, 32), port=1, tier="loose", success=True) -> rf.RawEpisode:
    return rf.RawEpisode(
        front=rng.integers(0, 256, (t, *hw, 3), dtype=np.uint8),
        wrist=rng.integers(0, 256, (t, *hw, 3), dtype=np.uint8),
        state=rng.normal(0, 30, (t, 6)),
        action=rng.normal(0, 30, (t, 6)),
        servo_load=rng.integers(-1000, 1001, (t, 6)),
        servo_current=rng.integers(0, 400, (t, 6)),
        task=rf.task_string(port), tier=tier, port=port, success=success, seed=7,
        extra={"time_to_latch_s": 3.2},
    )


def test_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    ep = synthetic_episode(rng)
    path = rf.write_episode(tmp_path, 0, ep)
    back = rf.read_episode(path)
    for k in rf.ARRAY_KEYS:
        a, b = getattr(ep, k), getattr(back, k)
        assert a.dtype == b.dtype and np.array_equal(a, b), k
    assert back.task == "insert the cable into the middle port"
    assert (back.tier, back.port, back.success, back.seed) == ("loose", 1, True, 7)
    assert back.extra == {"time_to_latch_s": 3.2}
    assert back.state.dtype == np.float32 and back.servo_load.dtype == np.int32
    assert not list(tmp_path.rglob("*.tmp"))


def test_manifest_resume_and_filter(tmp_path):
    rng = np.random.default_rng(1)
    m = rf.load_manifest(tmp_path)
    assert m.episodes == [] and m.next_index() == 0
    for port, ok in [(0, True), (1, False), (2, True)]:
        i = m.next_index()
        ep = synthetic_episode(rng, port=port, success=ok)
        m.add(i, rf.write_episode(tmp_path, i, ep), ep)
    m.save(tmp_path)

    m2 = rf.load_manifest(tmp_path)
    assert m2.next_index() == 3
    assert m2.count(tier="loose") == 2 and m2.count(port=1, success=None) == 1
    got = [e["port"] for e, _ in rf.iter_episodes(tmp_path)]
    assert got == [0, 2]
    entry = json.loads((tmp_path / "manifest.json").read_text())["episodes"][0]
    assert entry["file"] == "episodes/ep_000000.npz" and entry["num_frames"] == 12


@pytest.mark.parametrize("field,bad", [
    ("state", np.zeros((12, 5))),
    ("front", np.zeros((11, 24, 32, 3))),
    ("servo_load", np.full((12, 6), 1500)),
    ("servo_current", np.full((12, 6), -1)),
    ("action", np.full((12, 6), np.nan)),
    ("tier", "tight"),
    ("port", 3),
])
def test_validation_rejects(field, bad):
    rng = np.random.default_rng(2)
    kw = synthetic_episode(rng).__dict__.copy()
    kw[field] = bad
    with pytest.raises(ValueError):
        rf.RawEpisode(**kw)


def test_manifest_rejects_other_joint_names(tmp_path):
    rf.Manifest(joint_names=["a"] * 6).save(tmp_path)
    with pytest.raises(ValueError):
        rf.load_manifest(tmp_path)
