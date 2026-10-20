"""Raw episodes (patchbench.raw_format) -> LeRobot v3 dataset. Runs in the separate LeRobot venv (~/work/.venv-lerobot),
never in the Isaac Sim venv (torch conflicts).

    python scripts/to_lerobot.py --raw ~/data/raw/patchbench_v0 --repo_id $HF_USER/patchbench-so101-v0 [--push]

Features (CLAUDE.md 6.3; names match a real SO-101 recorded with LeRobot, ``<motor>.pos``):
  observation.state           float32 (6,)  joint positions [deg]; (12,) with --state_with_load: + signed Present_Load
  action                      float32 (6,)  joint goal positions [deg]
  observation.images.front    video (H, W, 3)
  observation.images.wrist    video (H, W, 3)
  observation.servo_load      int32 (6,)    Present_Load, signed, 0.1 % duty
  observation.servo_current   int32 (6,)    Present_Current, 6.5 mA per count
  task                        language instruction ("insert the cable into the left port")

Per-episode metadata LeRobot has no column for (tier, port, seed, raw file, ...) goes to
``meta/patchbench_episodes.jsonl`` (one line per dataset episode_index); it is uploaded with the dataset.

The dataset is built in ``<root>.partial`` and moved to ``<root>`` only after ``finalize()``, so an interrupted run
never leaves something that looks finished. An existing complete dataset is reused (``--overwrite`` rebuilds it).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path


# patchbench.raw_format is numpy-only; import it from the repo even when patchbench is not installed in this venv.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from patchbench import raw_format as rf  # noqa: E402

ROBOT_TYPE = "so101_follower"


def build_features(height: int, width: int, state_with_load: bool = False) -> dict:
    def joints(suffix: str) -> list[str]:
        return [f"{j}.{suffix}" for j in rf.JOINT_NAMES]

    video = {"dtype": "video", "shape": (height, width, 3), "names": ["height", "width", "channels"]}
    return {
        # --state_with_load: GR00T N1.7 reads only observation.state, so the load registers are appended to it
        # (positions first: relative actions use the first 6 state dims). observation.servo_load stays as well.
        "observation.state": ({"dtype": "float32", "shape": (12,), "names": joints("pos") + joints("load")}
                              if state_with_load else {"dtype": "float32", "shape": (6,), "names": joints("pos")}),
        "action": {"dtype": "float32", "shape": (6,), "names": joints("pos")},
        "observation.images.front": dict(video),
        "observation.images.wrist": dict(video),
        "observation.servo_load": {"dtype": "int32", "shape": (6,), "names": joints("load")},
        "observation.servo_current": {"dtype": "int32", "shape": (6,), "names": joints("current")},
    }


def default_root(repo_id: str) -> Path:
    from lerobot.utils.constants import HF_LEROBOT_HOME

    return Path(HF_LEROBOT_HOME) / repo_id


def is_complete(root: Path, n_expected: int) -> bool:
    info = root / "meta" / "info.json"
    if not info.exists():
        return False
    return json.loads(info.read_text()).get("total_episodes") == n_expected


def convert(raw: Path, repo_id: str, root: Path, vcodec: str, encoder_threads: int | None,
            max_episodes: int | None, state_with_load: bool = False) -> int:
    import numpy as np
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    manifest = rf.load_manifest(raw)
    if manifest.fps != rf.FPS:
        raise ValueError(f"raw fps {manifest.fps} != {rf.FPS}")
    entries = sorted((e for e in manifest.episodes if e["success"]), key=lambda e: e["index"])[:max_episodes]
    if not entries:
        raise SystemExit(f"{raw}: no successful episodes to convert")
    hw = {tuple(e["image_hw"]) for e in entries}
    if len(hw) != 1:
        raise ValueError(f"mixed image sizes in {raw}: {hw}")
    (h, w), = hw

    partial = root.with_name(root.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.parent.mkdir(parents=True, exist_ok=True)
    ds = LeRobotDataset.create(
        repo_id=repo_id, fps=manifest.fps, features=build_features(h, w, state_with_load), root=partial, robot_type=ROBOT_TYPE,
        use_videos=True, image_writer_threads=4, rgb_encoder=RGBEncoderConfig(vcodec=vcodec),
        encoder_threads=encoder_threads,
    )
    sidecar = []
    t0 = time.time()
    frames = 0
    for k, e in enumerate(entries):
        ep = rf.read_episode(raw / e["file"])
        for t in range(ep.num_frames):
            ds.add_frame({
                "observation.state": (np.concatenate([ep.state[t], ep.servo_load[t]]).astype(np.float32)
                                      if state_with_load else ep.state[t]),
                "action": ep.action[t],
                "observation.images.front": ep.front[t],
                "observation.images.wrist": ep.wrist[t],
                "observation.servo_load": ep.servo_load[t],
                "observation.servo_current": ep.servo_current[t],
                "task": ep.task,
            })
        ds.save_episode()
        frames += ep.num_frames
        sidecar.append({"episode_index": k, "raw_index": e["index"], "raw_file": e["file"], "tier": ep.tier,
                        "port": ep.port, "seed": ep.seed, "success": ep.success, "num_frames": ep.num_frames,
                        **ep.extra})
        print(f"[to_lerobot] episode {k + 1}/{len(entries)} ({ep.tier}, port {ep.port}, {ep.num_frames} frames) "
              f"{frames / (time.time() - t0):.0f} frames/s", flush=True)
    ds.finalize()
    with open(partial / "meta" / "patchbench_episodes.jsonl", "w") as f:
        for row in sidecar:
            f.write(json.dumps(row) + "\n")
    (partial / "meta" / "patchbench_raw_info.json").write_text(json.dumps(manifest.info, indent=1))
    shutil.rmtree(root, ignore_errors=True)
    os.replace(partial, root)
    return len(entries)


def check(repo_id: str, root: Path) -> None:
    """LeRobot's own loader: open the dataset, decode one frame, print features and shapes."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, root=root)
    print(f"[check] {repo_id} @ {root}: {ds.num_episodes} episodes, {ds.num_frames} frames, fps {ds.fps}, "
          f"robot_type {ds.meta.robot_type}")
    for k, ft in ds.features.items():
        print(f"  {k:28s} {ft['dtype']:8s} {tuple(ft['shape'])}")
    item = ds[0]
    for k, v in item.items():
        shape = tuple(v.shape) if hasattr(v, "shape") else repr(v)
        print(f"  frame0 {k:28s} {getattr(v, 'dtype', type(v).__name__)} {shape}")
    img = item["observation.images.front"]
    assert img.shape[0] == 3 and float(img.min()) >= 0.0 and float(img.max()) <= 1.0, "unexpected image tensor"


def push(repo_id: str, root: Path, private: bool) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, root=root)
    ds.push_to_hub(tags=["patchbench", "so101", "isaac-lab-arena", "rj45"], private=private,
                   upload_large_folder=True)
    print(f"[push] https://huggingface.co/datasets/{repo_id}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True, help="raw dir with manifest.json (from scripts/gen_data.py)")
    ap.add_argument("--repo_id", required=True, help="e.g. $HF_USER/patchbench-so101-v0")
    ap.add_argument("--root", type=Path, default=None, help="dataset dir (default $HF_LEROBOT_HOME/<repo_id>)")
    ap.add_argument("--vcodec", default="libsvtav1", help="LeRobot video codec (libsvtav1 default; h264 is faster)")
    ap.add_argument("--encoder_threads", type=int, default=None)
    ap.add_argument("--max_episodes", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true", help="rebuild even if a complete dataset exists")
    ap.add_argument("--no_check", action="store_true")
    ap.add_argument("--state_with_load", action="store_true",
                    help="observation.state = 6 positions [deg] + 6 signed Present_Load counts (12,)")
    ap.add_argument("--push", action="store_true", help="push to the Hub (token from HF_TOKEN or `hf auth login`)")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    root = (args.root or default_root(args.repo_id)).expanduser()
    n_expected = rf.load_manifest(args.raw).count(success=True)
    if args.max_episodes is not None:
        n_expected = min(n_expected, args.max_episodes)
    if not args.overwrite and is_complete(root, n_expected):
        print(f"[to_lerobot] {root} already holds {n_expected} episodes; skipped (use --overwrite to rebuild)")
    else:
        t0 = time.time()
        n = convert(args.raw, args.repo_id, root, args.vcodec, args.encoder_threads, args.max_episodes,
                    args.state_with_load)
        print(f"[to_lerobot] wrote {n} episodes to {root} in {time.time() - t0:.0f} s")
    if not args.no_check:
        check(args.repo_id, root)
    if args.push:
        push(args.repo_id, root, args.private)


if __name__ == "__main__":
    main()
