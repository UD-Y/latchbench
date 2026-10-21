"""Offline sanity check of a LeRobot policy on recorded episodes: predicted action chunks vs the expert's actions.

Feeds dataset frames (raw episodes from scripts/gen_data.py) through the same ``LeRobotPolicy`` wrapper the policy
server uses, and compares each predicted chunk with the recorded joint goals ``action[t : t + H]`` [deg]. Baseline:
"hold" = predict the current joint state for every step. A policy that learned the data beats "hold" by a wide margin
on the frames it was trained on; errors of tens of degrees point to a pipeline bug, not a weak policy.
Runs in the lerobot venv (GPU; not next to Isaac Sim on the laptop):

    ~/work/.venv-lerobot/bin/python scripts/check_policy_offline.py --policy.path UD77/latchbench-groot-n17 \
        --raw ~/data/raw/latchbench_v0 --episodes 0,250,500 --frames 0,30,60,90,120
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from latchbench import raw_format as rf  # noqa: E402
from latchbench.policy_server import IMAGE_KEYS, JOINT_NAMES, STATE_KEY, LeRobotPolicy  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy.path", dest="policy_path", required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--episodes", default="0,250,500", help="manifest indices")
    ap.add_argument("--frames", default="0,30,60,90,120", help="frame indices within each episode")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    pol = LeRobotPolicy(args.policy_path, device=args.device)
    m = {e["index"]: e for e in rf.load_manifest(args.raw).episodes}
    err_pol, err_hold, err_pol0 = [], [], []
    for idx in [int(v) for v in args.episodes.split(",")]:
        e = m[idx]
        ep = rf.read_episode(args.raw / e["file"])
        frames = [t for t in (int(v) for v in args.frames.split(",")) if t < ep.num_frames]
        state = ep.state[frames]
        if pol.state_dim == 12:  # checkpoint trained with to_lerobot.py --state_with_load
            state = np.concatenate([state, ep.servo_load[frames].astype(np.float32)], axis=1)
        obs = {STATE_KEY: state, IMAGE_KEYS[0]: ep.front[frames], IMAGE_KEYS[1]: ep.wrist[frames]}
        pol.reset(None)
        chunk = pol.infer(obs, [ep.task] * len(frames))  # (B, H, 6) deg
        for b, t in enumerate(frames):
            h = min(chunk.shape[1], ep.num_frames - t)
            ref = ep.action[t : t + h]
            err_pol.append(np.abs(chunk[b, :h] - ref).mean(0))
            err_pol0.append(np.abs(chunk[b, 0] - ref[0]))
            err_hold.append(np.abs(ep.state[t][None] - ref).mean(0))
            print(f"ep {idx} ({e['tier']}, port {e['port']}) t={t:3d}  state {np.round(ep.state[t], 1)}\n"
                  f"   expert a[t]  {np.round(ep.action[t], 1)}\n   policy a[t]  {np.round(chunk[b, 0], 1)}")
    fmt = lambda a: "  ".join(f"{n}={v:.2f}" for n, v in zip(JOINT_NAMES, np.mean(a, 0)))  # noqa: E731
    print("\nmean |error| [deg] per joint")
    print("  policy, step 0   ", fmt(err_pol0))
    print("  policy, chunk    ", fmt(err_pol))
    print("  hold baseline    ", fmt(err_hold))
    print(f"overall: policy chunk {np.mean(err_pol):.2f} deg vs hold {np.mean(err_hold):.2f} deg")


if __name__ == "__main__":
    main()
