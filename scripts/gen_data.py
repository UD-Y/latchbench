"""Expert rollouts with cameras -> raw episodes (latchbench.raw_format: one npz per episode + manifest.json).

Runs in the Arena venv. Only successful (latched) episodes are saved. Resumable: per (tier, port) it counts the
successes already in ``--out/manifest.json`` and generates only the rest.

    flock -o ~/work/.isaac.lock python scripts/gen_data.py --tiers loose,medium --episodes_per_port 150 \
        --num_envs 8 --height 480 --width 640 --out ~/data/raw/latchbench_v0

The top-level call never starts Isaac Sim itself: it runs one child process per (tier, port) in sequence (the env
is built for one tier and one target port), so the whole call holds the Isaac lock once. Extra arguments after
``--`` are passed to the environment (after its name), e.g. ``-- --some_env_flag x``. Policy flags (e.g.
``--expert_controller oracle_spiral``; the expert's ``--pose_noise_mm`` defaults to 0) go before ``--``.

Recorded per control step t (30 Hz), before ``env.step``:
  state   = robot joint positions [deg], LeRobot joint order (sim Rotation..Jaw == shoulder_pan..gripper)
  action  = absolute joint position targets the sim received for step t [deg] (portable across action terms)
  front / wrist = ``camera_obs`` terms (``--front_key`` / ``--wrist_key``), uint8 RGB, resized to --height/--width
  servo_load / servo_current = servosense telemetry registers (signed Present_Load, Present_Current counts)
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from latchbench import raw_format as rf  # noqa: E402

ENV_CLASS = "latchbench.environment:RJ45PanelEnv"
ENV_NAME = "rj45_panel"


def gen_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group("gen_data")
    g.add_argument("--out", type=Path, required=True, help="raw output dir (manifest.json + episodes/)")
    g.add_argument("--episodes_per_port", type=int, default=1, help="successful episodes per (tier, port)")
    g.add_argument("--height", type=int, default=240)
    g.add_argument("--width", type=int, default=320)
    g.add_argument("--front_key", default="external_camera_rgb", help="camera_obs term for observation.images.front")
    g.add_argument("--wrist_key", default="camera_ego_rgb", help="camera_obs term for observation.images.wrist")
    g.add_argument("--max_attempts_factor", type=float, default=3.0,
                   help="give up on a (tier, port) after this many x episodes_per_port attempts")
    g.add_argument("--writers", type=int, default=4, help="background npz writer threads")
    return parser


# ----------------------------------------------------------------------------------------------- parent (no Isaac)
def parent_main() -> None:
    argv = sys.argv[1:]
    env_extra: list[str] = []
    if "--" in argv:
        k = argv.index("--")
        argv, env_extra = argv[:k], argv[k + 1:]
    ap = gen_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--tiers", default="loose")
    ap.add_argument("--ports", default="0,1,2")
    ap.add_argument("--num_envs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy_type", default="expert")
    args, passthrough = ap.parse_known_args(argv)

    tiers = [t for t in args.tiers.split(",") if t]
    ports = [int(p) for p in args.ports.split(",") if p]
    t0 = time.time()
    for ti, tier in enumerate(tiers):
        for port in ports:
            have = rf.load_manifest(args.out).count(tier=tier, port=port)
            if have >= args.episodes_per_port:
                print(f"[gen_data] {tier} port {port}: {have}/{args.episodes_per_port} already; skipped", flush=True)
                continue
            seed = args.seed + 1000 * rf.TIERS.index(tier) + 100 * port + have  # distinct per run and resume
            cmd = [sys.executable, os.path.abspath(__file__), "--_child",
                   "--out", str(args.out), "--episodes_per_port", str(args.episodes_per_port),
                   "--height", str(args.height), "--width", str(args.width),
                   "--front_key", args.front_key, "--wrist_key", args.wrist_key,
                   "--max_attempts_factor", str(args.max_attempts_factor), "--writers", str(args.writers),
                   "--enable_cameras", "--num_envs", str(args.num_envs), "--seed", str(seed),
                   "--policy_type", args.policy_type, *passthrough,
                   "--external_environment_class_path", ENV_CLASS, ENV_NAME,
                   "--tier", tier, "--port", str(port),
                   "--camera_height", str(args.height), "--camera_width", str(args.width), *env_extra]
            print(f"[gen_data] {tier} port {port}: have {have}, running: {' '.join(cmd)}", flush=True)
            subprocess.run(cmd, check=True)
    m = rf.load_manifest(args.out)
    for tier in tiers:
        print(f"[gen_data] {tier}: " + ", ".join(f"port {p} {m.count(tier=tier, port=p)}" for p in ports))
    print(f"[gen_data] done in {time.time() - t0:.0f} s -> {args.out}")


# ------------------------------------------------------------------------------------------- child (one Isaac app)
def child_main() -> None:
    sys.argv.remove("--_child")
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    parser = gen_args(get_isaaclab_arena_cli_parser())
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    with SimulationAppContext(args_cli):
        run_child(parser)


def run_child(parser: argparse.ArgumentParser) -> None:
    import torch
    import torch.nn.functional as F
    from isaaclab.managers import SceneEntityCfg
    from isaaclab_arena.evaluation.policy_runner import get_policy_cls
    from isaaclab_arena.evaluation.policy_runner_cli import (
        add_policy_cli_args,
        add_policy_runner_arguments,
        build_policy_from_cli,
    )
    from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser
    from isaaclab_so101.mapping import SIM_JOINT_NAMES
    from servosense.isaaclab.observations import servo_current, servo_load

    try:
        import latchbench.policies  # noqa: F401  (registers expert / scripted policies)
    except ImportError as e:
        print(f"[gen_data] WARNING: latchbench.policies not importable ({e})")

    add_policy_runner_arguments(parser)
    args_cli, _ = parser.parse_known_args()
    policy_cls = get_policy_cls(args_cli.policy_type)
    parser = get_isaaclab_arena_environments_cli_parser(parser)
    parser = add_policy_cli_args(parser, policy_cls)
    args_cli, hydra_overrides = parser.parse_known_args()
    args_cli.enable_cameras = True
    tier, port = args_cli.tier, int(args_cli.port)
    target = args_cli.episodes_per_port
    out: Path = args_cli.out.expanduser()

    env = get_arena_builder_from_cli(args_cli, hydra_overrides=hydra_overrides).make_registered()
    uenv = env.unwrapped
    policy = build_policy_from_cli(policy_cls, args_cli)
    n = uenv.num_envs
    robot = uenv.scene["robot"]
    joint_ids = robot.find_joints(list(SIM_JOINT_NAMES), preserve_order=True)[0]
    telem_cfg = SceneEntityCfg("robot")
    rad2deg = 180.0 / math.pi
    cams = uenv.observation_manager.active_terms.get("camera_obs", [])
    for key in (args_cli.front_key, args_cli.wrist_key):
        if key not in cams:
            raise SystemExit(f"[gen_data] camera term {key!r} not in camera_obs {cams}; set --front_key/--wrist_key")
    task = rf.task_string(port)
    if (instr := uenv.get_language_instruction()) and instr != task:
        print(f"[gen_data] WARNING: env instruction {instr!r} != {task!r}; recording {task!r}")

    def torch_of(x):
        return x.torch if hasattr(x, "torch") else x

    def image(obs, key) -> np.ndarray:
        img = obs["camera_obs"][key]
        if img.dtype != torch.uint8:  # float in [0, 1]
            img = (img.clamp(0, 1) * 255).round().to(torch.uint8)
        img = img[..., :3]
        if img.shape[1:3] != (args_cli.height, args_cli.width):
            x = img.permute(0, 3, 1, 2).float()
            x = F.interpolate(x, size=(args_cli.height, args_cli.width), mode="area")
            img = x.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
        return img.cpu().numpy()

    manifest = rf.load_manifest(out)
    manifest.info.setdefault("runs", [])
    have = manifest.count(tier=tier, port=port)
    max_attempts = math.ceil(args_cli.max_attempts_factor * target)
    attempts = successes = frames_total = 0
    buf: list[dict[str, list]] = [None] * n  # type: ignore[list-item]
    pending: list[tuple[int, Future, rf.RawEpisode]] = []
    pool = ThreadPoolExecutor(max_workers=args_cli.writers)

    def new_buf():
        return {k: [] for k in rf.ARRAY_KEYS}

    def drain(block: bool) -> None:
        for item in list(pending):
            idx, fut, ep = item
            if block or fut.done():
                manifest.add(idx, fut.result(), ep)
                pending.remove(item)
        manifest.save(out)

    obs, _ = env.reset()
    policy.reset()
    policy.set_task_description(task)
    for i in range(n):
        buf[i] = new_buf()
    print(f"[gen_data] {tier} port {port}: have {have}/{target}, num_envs {n}, cameras {cams}", flush=True)
    t_start = time.time()
    while have + successes < target and attempts < max_attempts:
        with torch.inference_mode():
            front, wrist = image(obs, args_cli.front_key), image(obs, args_cli.wrist_key)
            state = (torch_of(robot.data.joint_pos)[:, joint_ids] * rad2deg).cpu().numpy()
            load = servo_load(uenv, telem_cfg)[:, joint_ids].round().to(torch.int32).cpu().numpy()
            current = servo_current(uenv, telem_cfg)[:, joint_ids].round().to(torch.int32).cpu().numpy()
            actions = policy.get_action(env, obs)
            obs, _, terminated, truncated, _ = env.step(actions)
            # The targets the sim received for this step (absolute, whatever the action term).
            target_deg = (torch_of(robot.data.joint_pos_target)[:, joint_ids] * rad2deg).cpu().numpy()
            done = (terminated | truncated).cpu().numpy()
            success = uenv.termination_manager.get_term("success").cpu().numpy() & terminated.cpu().numpy()
        frames_total += n
        for i in range(n):
            b = buf[i]
            b["front"].append(front[i]); b["wrist"].append(wrist[i])
            b["state"].append(state[i]); b["action"].append(target_deg[i])
            b["servo_load"].append(load[i]); b["servo_current"].append(current[i])
        for i in np.flatnonzero(done):
            attempts += 1
            b, buf[i] = buf[i], new_buf()
            # The env auto-reset after this step, so its read-back target may already be the reset pose: drop the
            # last frame rather than record an action the sim may not have received.
            for v in b.values():
                v.pop()
            if not b["state"]:
                continue
            if success[i] and have + successes < target:
                ep = rf.RawEpisode(**{k: np.stack(v) for k, v in b.items()}, task=task, tier=tier, port=port,
                                   success=True, seed=int(args_cli.seed), extra={"num_envs": n, "env_id": int(i)})
                idx = manifest.next_index() + len(pending)
                pending.append((idx, pool.submit(rf.write_episode, out, idx, ep), ep))
                successes += 1
            print(f"[gen_data] {tier} port {port}: attempt {attempts} {'latched' if success[i] else 'failed'} "
                  f"({len(b['state'])} frames); saved {have + successes}/{target}; "
                  f"{frames_total / (time.time() - t_start):.1f} env-frames/s", flush=True)
        if done.any():
            policy.reset(env_ids=torch.as_tensor(np.flatnonzero(done), device=uenv.device))
            drain(block=False)
    drain(block=True)
    pool.shutdown()
    wall = time.time() - t_start
    run = {"tier": tier, "port": port, "num_envs": n, "height": args_cli.height, "width": args_cli.width,
           "attempts": attempts, "successes": successes, "env_frames": frames_total, "wall_s": round(wall, 1),
           "env_frames_per_s": round(frames_total / max(wall, 1e-6), 2), "seed": int(args_cli.seed),
           "policy_type": args_cli.policy_type, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    manifest.info["runs"].append(run)
    manifest.save(out)
    print(f"[gen_data] RUN {run}", flush=True)
    env.close()
    if have + successes < target:
        raise SystemExit(f"[gen_data] {tier} port {port}: only {have + successes}/{target} latched after "
                         f"{attempts} attempts (max {max_attempts})")


if __name__ == "__main__":
    child_main() if "--_child" in sys.argv else parent_main()
