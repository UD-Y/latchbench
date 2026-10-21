"""Raw episode format shared by ``scripts/gen_data.py`` (Isaac Sim venv) and ``scripts/to_lerobot.py`` (LeRobot venv).

numpy + stdlib only, so both venvs can import it. Layout of a raw directory::

    <root>/manifest.json            # format version, fps, joint names, one entry per episode
    <root>/episodes/ep_000000.npz   # one compressed npz per episode

Per-episode arrays (T = frames at ``fps``; joint order = ``JOINT_NAMES``, the LeRobot SO-101 names):

=================  ==================  =======  ======================================================================
key                shape               dtype    meaning
=================  ==================  =======  ======================================================================
``front``          (T, H, W, 3)        uint8    static front camera RGB  -> ``observation.images.front``
``wrist``          (T, H, W, 3)        uint8    wrist camera RGB         -> ``observation.images.wrist``
``state``          (T, 6)              float32  joint positions [deg], LeRobot ``use_degrees=True`` convention
``action``         (T, 6)              float32  joint goal positions [deg] sent at step t (same units as state)
``servo_load``     (T, 6)              int32    Present_Load decoded to signed (0.1 % duty, +-1000)
``servo_current``  (T, 6)              int32    Present_Current (6.5 mA per count, unsigned as LeRobot reads it)
=================  ==================  =======  ======================================================================

Scalars (``task``, ``tier``, ``port``, ``success``, ``seed``) live in the npz as a JSON string under ``meta`` and are
copied into the manifest entry, so the converter can filter without opening every file.

Gripper: LeRobot's SO-101 calibration reports the gripper in 0-100 range even with ``use_degrees=True``. We record the
sim jaw angle in degrees for every joint; the sim->real joint-zero and gripper mapping needs a real calibration file
(docs/DATA.md) and is not faked here.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1
FPS = 30  # control rate of the env and the dataset (CLAUDE.md 6.3)
JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
TIERS = ("loose", "medium", "spec")
PORT_WORDS = ("left", "middle", "right")  # port_0 = left (+Y), see assets hand-over
IMAGE_KEYS = ("front", "wrist")
FLOAT_KEYS = ("state", "action")
INT_KEYS = ("servo_load", "servo_current")
ARRAY_KEYS = IMAGE_KEYS + FLOAT_KEYS + INT_KEYS
MANIFEST = "manifest.json"
EPISODE_DIR = "episodes"


def task_string(port: int) -> str:
    """Language instruction for a target port (CLAUDE.md 6.2)."""
    return f"insert the cable into the {PORT_WORDS[port]} port"


@dataclass
class RawEpisode:
    front: np.ndarray
    wrist: np.ndarray
    state: np.ndarray
    action: np.ndarray
    servo_load: np.ndarray
    servo_current: np.ndarray
    task: str
    tier: str
    port: int
    success: bool
    seed: int = -1
    extra: dict = field(default_factory=dict)  # free-form, e.g. time_to_latch_s, peak_axial_force_n

    def __post_init__(self):
        self.front = np.ascontiguousarray(self.front, dtype=np.uint8)
        self.wrist = np.ascontiguousarray(self.wrist, dtype=np.uint8)
        self.state = np.asarray(self.state, dtype=np.float32)
        self.action = np.asarray(self.action, dtype=np.float32)
        self.servo_load = np.asarray(np.rint(self.servo_load), dtype=np.int32)
        self.servo_current = np.asarray(np.rint(self.servo_current), dtype=np.int32)
        self.port, self.success, self.seed = int(self.port), bool(self.success), int(self.seed)
        validate(self)

    @property
    def num_frames(self) -> int:
        return int(self.state.shape[0])

    def meta(self) -> dict:
        return {"task": self.task, "tier": self.tier, "port": self.port, "success": self.success, "seed": self.seed,
                "num_frames": self.num_frames, "image_hw": list(self.front.shape[1:3]), "extra": self.extra}


def validate(ep: RawEpisode) -> None:
    t = ep.state.shape[0]
    if t < 1:
        raise ValueError("episode has no frames")
    for k in IMAGE_KEYS:
        a = getattr(ep, k)
        if a.ndim != 4 or a.shape[0] != t or a.shape[-1] != 3:
            raise ValueError(f"{k}: expected (T={t}, H, W, 3) uint8, got {a.shape}")
    for k in FLOAT_KEYS + INT_KEYS:
        a = getattr(ep, k)
        if a.shape != (t, len(JOINT_NAMES)):
            raise ValueError(f"{k}: expected ({t}, {len(JOINT_NAMES)}), got {a.shape}")
    for k in FLOAT_KEYS:
        if not np.isfinite(getattr(ep, k)).all():
            raise ValueError(f"{k}: non-finite values")
    if np.abs(ep.servo_load).max() > 1000:
        raise ValueError("servo_load outside +-1000 (expected decoded Present_Load)")
    if ep.servo_current.min() < 0:
        raise ValueError("servo_current must be unsigned register counts")
    if ep.tier not in TIERS:
        raise ValueError(f"tier {ep.tier!r} not in {TIERS}")
    if not 0 <= ep.port < len(PORT_WORDS):
        raise ValueError(f"port {ep.port} out of range")
    if not ep.task:
        raise ValueError("empty task string")


def _atomic_write(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            write(f)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def episode_path(root: str | Path, index: int) -> Path:
    return Path(root) / EPISODE_DIR / f"ep_{index:06d}.npz"


def write_episode(root: str | Path, index: int, ep: RawEpisode) -> Path:
    """Write one episode atomically (a crash never leaves a truncated npz under the final name)."""
    path = episode_path(root, index)
    arrays = {k: getattr(ep, k) for k in ARRAY_KEYS}
    _atomic_write(path, lambda f: np.savez_compressed(f, meta=np.array(json.dumps(ep.meta())), **arrays))
    return path


def read_episode(path: str | Path) -> RawEpisode:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        arrays = {k: z[k] for k in ARRAY_KEYS}
    return RawEpisode(**arrays, task=meta["task"], tier=meta["tier"], port=meta["port"], success=meta["success"],
                      seed=meta.get("seed", -1), extra=meta.get("extra", {}))


@dataclass
class Manifest:
    fps: int = FPS
    joint_names: list = field(default_factory=lambda: list(JOINT_NAMES))
    format_version: int = FORMAT_VERSION
    episodes: list = field(default_factory=list)  # [{"index", "file", **RawEpisode.meta()}]
    info: dict = field(default_factory=dict)  # generator settings (git sha, physics, resolution, ...)

    def next_index(self) -> int:
        return 1 + max((e["index"] for e in self.episodes), default=-1)

    def add(self, index: int, path: Path, ep: RawEpisode) -> None:
        self.episodes.append({"index": index, "file": f"{EPISODE_DIR}/{path.name}", **ep.meta()})

    def count(self, tier: str | None = None, port: int | None = None, success: bool | None = True) -> int:
        return sum(1 for e in self.episodes
                   if (tier is None or e["tier"] == tier) and (port is None or e["port"] == port)
                   and (success is None or e["success"] == success))

    def save(self, root: str | Path) -> None:
        data = json.dumps(asdict(self), indent=1).encode()
        _atomic_write(Path(root) / MANIFEST, lambda f: f.write(data))


def load_manifest(root: str | Path) -> Manifest:
    """Load ``root/manifest.json``; an absent manifest is an empty one (lets gen_data resume)."""
    path = Path(root) / MANIFEST
    if not path.exists():
        return Manifest()
    m = Manifest(**json.loads(path.read_text()))
    if m.format_version != FORMAT_VERSION:
        raise ValueError(f"{path}: format_version {m.format_version}, this code reads {FORMAT_VERSION}")
    if tuple(m.joint_names) != JOINT_NAMES:
        raise ValueError(f"{path}: joint names {m.joint_names} != {JOINT_NAMES}")
    return m


def iter_episodes(root: str | Path, success_only: bool = True):
    """Yield (manifest entry, RawEpisode) in index order."""
    m = load_manifest(root)
    for e in sorted(m.episodes, key=lambda e: e["index"]):
        if success_only and not e["success"]:
            continue
        yield e, read_episode(Path(root) / e["file"])
