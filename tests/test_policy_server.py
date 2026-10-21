"""ZMQ policy protocol: both ends in one process with the random dummy policy (no Isaac, no LeRobot)."""

import socket
import threading

import numpy as np
import pytest

zmq = pytest.importorskip("zmq")

from latchbench import policy_server as ps  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server():
    port = _free_port()
    pol = ps.RandomPolicy(chunk_size=4, scale_deg=0.0)
    th = threading.Thread(target=ps.serve, args=(pol, f"tcp://127.0.0.1:{port}"), kwargs={"log": lambda *_: None},
                          daemon=True)
    th.start()
    yield pol, f"tcp://127.0.0.1:{port}"
    th.join(timeout=0.1)


def test_pack_roundtrip():
    a = {"x": np.arange(6, dtype=np.float32).reshape(2, 3), "img": np.zeros((1, 4, 5, 3), np.uint8)}
    hdr, out = ps.unpack(ps.pack({"cmd": "infer", "task": ["t"]}, a))
    assert hdr["cmd"] == "infer" and hdr["task"] == ["t"]
    for k in a:
        np.testing.assert_array_equal(out[k], a[k])
        assert out[k].dtype == a[k].dtype
    frames = ps.pack({"cmd": "x"}, a)
    with pytest.raises(ValueError):
        ps.unpack(frames[:-1])


def test_chunked_actions_and_reset(server):
    pol, ep = server
    c = ps.PolicyClient(ep, timeout_s=5, n_action_steps=3)
    assert c.info["chunk_size"] == 4 and c.info["action_dim"] == 6 and c.n_action_steps == 3
    b = 3
    imgs = {k: np.random.rand(b, 24, 32, 3).astype(np.float32) for k in ps.IMAGE_KEYS}
    calls = []
    orig = pol.infer
    pol.infer = lambda obs, task: (calls.append((obs[ps.STATE_KEY].shape[0], list(task), obs[ps.IMAGE_KEYS[0]].dtype))
                                   or orig(obs, task))
    s0 = np.tile(np.arange(6, dtype=np.float32), (b, 1))
    tasks = ["insert the cable into the left port", "insert the cable into the middle port",
             "insert the cable into the right port"]
    a = c.get_action(s0, imgs, tasks)
    assert a.shape == (b, 6) and np.allclose(a, s0)  # scale 0: chunk = state
    assert calls == [(3, tasks, np.uint8)]
    # next 2 steps come from the queue; the 4th triggers one batched request
    c.get_action(s0 + 1, imgs, tasks)
    c.get_action(s0 + 1, imgs, tasks)
    assert len(calls) == 1
    a = c.get_action(s0 + 10, imgs, tasks)
    assert len(calls) == 2 and np.allclose(a, s0 + 10)
    # resetting env 1 re-requests only env 1
    c.reset([1])
    assert pol.resets[-1] == [1]
    a = c.get_action(s0 + 20, imgs, tasks)
    assert calls[-1][0] == 1 and calls[-1][1] == [tasks[1]]
    assert np.allclose(a[1], s0[1] + 20) and np.allclose(a[0], s0[0] + 10)
    c.shutdown_server()
    c.close()


def test_server_error_is_reported(server):
    pol, ep = server
    c = ps.PolicyClient(ep, timeout_s=5)
    with pytest.raises(RuntimeError, match="unknown cmd"):
        c.request({"cmd": "bogus"})
    pol.infer = lambda obs, task: np.zeros((1, 2, 6), np.float32)  # wrong batch size
    with pytest.raises(RuntimeError, match="expected"):
        c.get_action(np.zeros((2, 6), np.float32), {}, "t")
    c.shutdown_server()


def test_timeout_without_server():
    with pytest.raises(TimeoutError):
        ps.PolicyClient(f"tcp://127.0.0.1:{_free_port()}", timeout_s=0.2, connect_timeout_s=0.3)
