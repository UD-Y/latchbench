"""ZMQ bridge between the Isaac Sim venv (client) and a LeRobot policy in the separate lerobot venv (server).

Isaac Sim and LeRobot need different torch builds, so they never share a process (CLAUDE.md section 8, rule 7).
This module imports only numpy, pyzmq and stdlib at top level; torch/lerobot are imported lazily by the server.

Protocol v1 (REQ/REP, one outstanding request; specified in docs/BENCHMARK.md):

    request  = [header_json, *array_buffers]       header = {"v": 1, "cmd": ..., "arrays": [spec, ...], ...}
    reply    = [header_json, *array_buffers]       header = {"v": 1, "ok": true|false, "error": str?, ...}
    spec     = {"key": str, "dtype": numpy dtype str, "shape": [int, ...]}   one C-contiguous buffer per spec

    cmd "hello":  -> {"policy": str, "action_dim": 6, "chunk_size": H, "image_keys": [...], "state_dim": 6 | 12}
                  state_dim 12 = 6 joint positions [deg] + 6 signed Present_Load counts (to_lerobot.py --state_with_load)
    cmd "infer":  arrays observation.state (B, 6) float32 [deg], observation.images.<cam> (B, H, W, 3) uint8,
                  header "task": [str] * B, "env_ids": [int] * B   ->  arrays action (B, H, 6) float32 [deg]
    cmd "reset":  header "env_ids": [int] | null                    -> {}
    cmd "shutdown":                                                 -> {} and the server exits

Run the server (lerobot venv):

    python -m patchbench.policy_server --policy.path <hf repo or dir> [--port 5555] [--device cuda]
    python -m patchbench.policy_server --random   # dummy policy for protocol tests
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

PROTOCOL_VERSION = 1
DEFAULT_ENDPOINT = "tcp://127.0.0.1:5555"
JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
STATE_KEY = "observation.state"
IMAGE_KEYS = ("observation.images.front", "observation.images.wrist")
ACTION_KEY = "action"


# --------------------------------------------------------------------------------------------------- wire format
def pack(header: dict[str, Any], arrays: dict[str, np.ndarray] | None = None) -> list[bytes]:
    """Header dict + named arrays -> multipart frames."""
    arrays = arrays or {}
    specs, bufs = [], []
    for k, a in arrays.items():
        a = np.ascontiguousarray(a)
        specs.append({"key": k, "dtype": a.dtype.str, "shape": list(a.shape)})
        bufs.append(a.tobytes())
    header = dict(header, v=PROTOCOL_VERSION, arrays=specs)
    return [json.dumps(header).encode()] + bufs


def unpack(frames: list[bytes]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Multipart frames -> (header, arrays). Raises ``ValueError`` on version or framing mismatch."""
    header = json.loads(bytes(frames[0]))
    if header.get("v") != PROTOCOL_VERSION:
        raise ValueError(f"protocol version {header.get('v')} != {PROTOCOL_VERSION}")
    specs = header.pop("arrays", [])
    if len(specs) != len(frames) - 1:
        raise ValueError(f"{len(specs)} array specs but {len(frames) - 1} buffers")
    arrays = {
        s["key"]: np.frombuffer(bytes(buf), dtype=np.dtype(s["dtype"])).reshape(s["shape"])
        for s, buf in zip(specs, frames[1:])
    }
    return header, arrays


# --------------------------------------------------------------------------------------------------- server side
class ServerPolicy(Protocol):
    name: str
    chunk_size: int

    def infer(self, obs: dict[str, np.ndarray], task: list[str]) -> np.ndarray:
        """Observations (batch B) -> action chunk ``(B, chunk_size, 6)`` float32 in degrees."""

    def reset(self, env_ids: list[int] | None) -> None: ...


@dataclass
class RandomPolicy:
    """Dummy policy: current state + uniform noise of ``scale_deg`` (for protocol tests, no torch needed)."""

    chunk_size: int = 16
    scale_deg: float = 2.0
    seed: int = 0
    name: str = "random"
    resets: list = field(default_factory=list)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def infer(self, obs: dict[str, np.ndarray], task: list[str]) -> np.ndarray:
        state = obs[STATE_KEY].astype(np.float32)
        noise = self.rng.uniform(-self.scale_deg, self.scale_deg, (state.shape[0], self.chunk_size, state.shape[1]))
        return (state[:, None, :] + noise).astype(np.float32)

    def reset(self, env_ids: list[int] | None) -> None:
        self.resets.append(env_ids)


class LeRobotPolicy:
    """A pretrained LeRobot policy (e.g. GR00T N1.7 fine-tune) with its saved pre/post-processors.

    Runs in the lerobot==0.6.1 venv. Images arrive as uint8 HWC and are given to the preprocessor as float CHW in
    [0, 1] (LeRobot dataset convention); states and actions stay in the dataset's units (degrees).
    """

    def __init__(self, path: str, device: str = "cuda", n_action_steps: int | None = None):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self.torch = torch
        cfg = PreTrainedConfig.from_pretrained(path)
        self.name = f"lerobot:{cfg.type}:{path}"
        self.policy = get_policy_class(cfg.type).from_pretrained(path)
        self.policy.to(device).eval()
        self.device = device
        dev = {"device": device}
        self.pre, self.post = make_pre_post_processors(
            self.policy.config,
            pretrained_path=path,
            preprocessor_overrides={"device_processor": dev},
            postprocessor_overrides={"device_processor": dev},
        )
        cs = n_action_steps or getattr(self.policy.config, "n_action_steps", None) or 1
        self.chunk_size = int(cs)
        self.image_keys = [k for k in self.policy.config.input_features if k.startswith("observation.images.")]
        st = self.policy.config.input_features.get(STATE_KEY)
        self.state_dim = int(st.shape[0]) if st is not None and st.shape else len(JOINT_NAMES)

    def infer(self, obs: dict[str, np.ndarray], task: list[str]) -> np.ndarray:
        torch = self.torch
        batch: dict[str, Any] = {STATE_KEY: torch.from_numpy(obs[STATE_KEY].astype(np.float32))}
        for k in self.image_keys:
            img = torch.from_numpy(np.ascontiguousarray(obs[k]))  # (B, H, W, 3) uint8
            batch[k] = img.permute(0, 3, 1, 2).float() / 255.0
        batch["task"] = list(task)
        with torch.inference_mode():
            batch = self.pre(batch)
            chunk = self.policy.predict_action_chunk(batch)
            if chunk.ndim == 2:
                chunk = chunk.unsqueeze(1)
            # Decode the whole (B, H, D) chunk in one call: GR00T N1.7 relative actions are decoded against the state
            # cached by this preprocess call and cannot be decoded one step at a time (GrootN17ActionDecodeStep).
            acts = self.post(chunk)
            if acts.ndim == 2:
                acts = acts.unsqueeze(1)
            acts = acts[:, : self.chunk_size]
        return acts.float().cpu().numpy()

    def reset(self, env_ids: list[int] | None) -> None:
        self.policy.reset()  # LeRobot policies keep one internal queue; we always request whole chunks


def serve(policy: ServerPolicy, endpoint: str = "tcp://*:5555", *, max_requests: int | None = None, log=print) -> None:
    """Serve ``policy`` on ``endpoint`` until a ``shutdown`` request (or ``max_requests``)."""
    import zmq

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(endpoint)
    log(f"[policy_server] {policy.name} on {endpoint} (chunk {policy.chunk_size})")
    served, logged_task = 0, False
    try:
        while max_requests is None or served < max_requests:
            frames = sock.recv_multipart(copy=False)
            served += 1
            try:
                hdr, arrays = unpack([f.bytes for f in frames])
                cmd = hdr.get("cmd")
                if cmd == "hello":
                    reply = pack({"ok": True, "policy": policy.name, "chunk_size": policy.chunk_size,
                                  "action_dim": len(JOINT_NAMES),
                                  "state_dim": int(getattr(policy, "state_dim", len(JOINT_NAMES))),
                                  "image_keys": list(getattr(policy, "image_keys", IMAGE_KEYS)),
                                  "joint_names": list(JOINT_NAMES)})
                elif cmd == "infer":
                    task = hdr.get("task") or [""] * arrays[STATE_KEY].shape[0]
                    if not logged_task:  # once: which instructions the sim sends (per env, by target port)
                        log(f"[policy_server] first infer: env_ids {hdr.get('env_ids')} tasks {task}")
                        logged_task = True
                    t0 = time.perf_counter()
                    act = np.asarray(policy.infer(arrays, task), dtype=np.float32)
                    b = arrays[STATE_KEY].shape[0]
                    if act.ndim != 3 or act.shape[0] != b or act.shape[2] != len(JOINT_NAMES):
                        raise ValueError(f"policy returned {act.shape}, expected ({b}, H, {len(JOINT_NAMES)})")
                    reply = pack({"ok": True, "infer_s": time.perf_counter() - t0}, {ACTION_KEY: act})
                elif cmd == "reset":
                    policy.reset(hdr.get("env_ids"))
                    reply = pack({"ok": True})
                elif cmd == "shutdown":
                    sock.send_multipart(pack({"ok": True}))
                    log("[policy_server] shutdown requested")
                    return
                else:
                    raise ValueError(f"unknown cmd {cmd!r}")
            except Exception as e:  # report to the client instead of dying; REP must always answer
                reply = pack({"ok": False, "error": f"{type(e).__name__}: {e}"})
            sock.send_multipart(reply)
    finally:
        sock.close(0)


# --------------------------------------------------------------------------------------------------- client side
class PolicyClient:
    """Sim-side client (numpy only). Keeps one action queue per env and asks the server, in one batched request,
    for a new chunk for every env whose queue is empty.

    ``n_action_steps`` (default: the server's chunk size) limits how many actions of each chunk are executed before
    re-planning.
    """

    def __init__(self, endpoint: str = DEFAULT_ENDPOINT, *, timeout_s: float = 60.0, n_action_steps: int | None = None,
                 connect_timeout_s: float = 600.0):
        import zmq

        self._zmq = zmq
        self.endpoint = endpoint
        self.timeout_ms = int(timeout_s * 1000)
        self._ctx = zmq.Context.instance()
        self._sock = None
        self._connect()
        self.info = self._hello(connect_timeout_s)
        self.chunk_size = int(self.info["chunk_size"])
        self.n_action_steps = min(n_action_steps or self.chunk_size, self.chunk_size)
        self._queues: list[np.ndarray] | None = None
        self._pos: np.ndarray | None = None

    def _connect(self):
        zmq = self._zmq
        if self._sock is not None:
            self._sock.close(0)
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._sock.connect(self.endpoint)

    def _hello(self, wait_s: float) -> dict[str, Any]:
        """Retry ``hello`` until the server answers (model loading can take minutes)."""
        deadline = time.monotonic() + wait_s
        while True:
            try:
                hdr, _ = self.request({"cmd": "hello"})
                return hdr
            except TimeoutError:
                if time.monotonic() > deadline:
                    raise
                self._connect()  # a timed-out REQ socket cannot send again

    def request(self, header: dict[str, Any], arrays: dict[str, np.ndarray] | None = None):
        self._sock.send_multipart(pack(header, arrays))
        try:
            frames = self._sock.recv_multipart()
        except self._zmq.Again:
            self._connect()
            raise TimeoutError(f"no reply from policy server at {self.endpoint} within {self.timeout_ms} ms")
        hdr, out = unpack(frames)
        if not hdr.get("ok", False):
            raise RuntimeError(f"policy server error: {hdr.get('error')}")
        return hdr, out

    def reset(self, env_ids: list[int] | None = None) -> None:
        if self._queues is not None:
            ids = range(len(self._queues)) if env_ids is None else env_ids
            for i in ids:
                self._pos[i] = self.n_action_steps  # empty
        self.request({"cmd": "reset", "env_ids": None if env_ids is None else [int(i) for i in env_ids]})

    def get_action(self, state_deg: np.ndarray, images: dict[str, np.ndarray], task: list[str] | str) -> np.ndarray:
        """``state_deg (B, state_dim)``, images ``key -> (B, H, W, 3) uint8``, task per env -> actions ``(B, 6)`` [deg]."""
        b = state_deg.shape[0]
        if isinstance(task, str):
            task = [task] * b
        if self._queues is None:
            self._queues = [np.zeros((self.chunk_size, len(JOINT_NAMES)), np.float32) for _ in range(b)]
            self._pos = np.full(b, self.n_action_steps, dtype=np.int64)
        need = np.nonzero(self._pos >= self.n_action_steps)[0]
        if need.size:
            arrays = {STATE_KEY: np.asarray(state_deg[need], np.float32)}
            for k, img in images.items():
                arrays[k] = _to_uint8_hwc(np.asarray(img)[need])
            _, out = self.request({"cmd": "infer", "task": [task[i] for i in need], "env_ids": need.tolist()}, arrays)
            chunk = out[ACTION_KEY]
            if chunk.shape[1] < self.n_action_steps:
                raise ValueError(f"server chunk {chunk.shape[1]} < n_action_steps {self.n_action_steps}")
            for j, i in enumerate(need):
                self._queues[i] = chunk[j]
                self._pos[i] = 0
        act = np.stack([self._queues[i][self._pos[i]] for i in range(b)])
        self._pos += 1
        return act

    def shutdown_server(self) -> None:
        try:
            self.request({"cmd": "shutdown"})
        except (TimeoutError, RuntimeError):
            pass

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close(0)
            self._sock = None


def _to_uint8_hwc(img: np.ndarray) -> np.ndarray:
    """Accept (B, H, W, 3|4) uint8 or float in [0, 1]; return (B, H, W, 3) uint8."""
    img = img[..., :3]
    if img.dtype != np.uint8:
        img = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return np.ascontiguousarray(img)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="PatchBench LeRobot policy server (run in the lerobot venv)")
    ap.add_argument("--policy.path", dest="policy_path", default=None, help="HF repo id or local checkpoint dir")
    ap.add_argument("--random", action="store_true", help="serve a random dummy policy instead")
    ap.add_argument("--chunk_size", type=int, default=16, help="random policy chunk size")
    ap.add_argument("--n_action_steps", type=int, default=None, help="actions per chunk to return (lerobot policy)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    args = ap.parse_args(argv)
    if args.random:
        policy: ServerPolicy = RandomPolicy(chunk_size=args.chunk_size)
    elif args.policy_path:
        policy = LeRobotPolicy(args.policy_path, device=args.device, n_action_steps=args.n_action_steps)
    else:
        ap.error("--policy.path or --random is required")
    serve(policy, args.endpoint)


if __name__ == "__main__":
    main()
