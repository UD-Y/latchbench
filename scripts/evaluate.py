"""Evaluator CLI: runs a policy per tier and writes results/<policy>/<tier>.json (CLAUDE.md section 6.2).

    # laptop, state-only (hold the shared Isaac lock)
    flock ~/work/.isaac.lock python scripts/evaluate.py --policy open_loop --tiers loose --n 16 --num_envs 16
    # Brev: a LeRobot policy served from the lerobot venv over ZMQ (cameras on)
    python scripts/evaluate.py --policy lerobot --policy.path $HF_USER/patchbench-groot-n17 --tiers loose --n 100 --num_envs 8

Each tier runs in its own Isaac Sim process (one tier = one panel USD), sequentially, through Arena's builder, policy
registry and per-episode recorder. Episodes follow a fixed schedule: global episode ``i < n`` is episode
``i // num_envs`` of env ``i % num_envs``, so results do not favour short episodes; the same ``--seed`` and
``--num_envs`` reproduce the same episodes. After all tiers, results/LEADERBOARD.md is rebuilt.

Run it with the Arena venv's python. ``--dry_run`` prints the worker commands without starting Isaac Sim.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # run from a checkout without installing

from patchbench import results_io as rio  # noqa: E402

SCRIPTED = ("open_loop", "telemetry_spiral", "oracle_spiral", "expert")
ENV_CLASS = "patchbench.environment:RJ45PanelEnv"
ENV_NAME = "rj45_panel"
DEFAULT_LEROBOT_PYTHON = Path.home() / "work" / ".venv-lerobot" / "bin" / "python"
# Candidate PolicyCfg field names for the policy's private noise RNG seed (Arena already owns --seed).
POLICY_SEED_FIELDS = ("noise_seed", "policy_seed")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, choices=SCRIPTED + ("lerobot",))
    ap.add_argument("--policy.path", dest="policy_path", default=None,
                    help="lerobot only: HF repo id or checkpoint dir ('random' = dummy server policy)")
    ap.add_argument("--name", default=None, help="results subdirectory (default: --policy; lerobot: lerobot_<repo>)")
    ap.add_argument("--tiers", default="loose,medium,spec")
    ap.add_argument("--n", type=int, default=100, help="episodes per tier")
    ap.add_argument("--num_envs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pose_noise_mm", type=float, default=None,
                    help="port-pose estimate noise sigma [mm] (default 1; expert 0, CLAUDE.md section 7)")
    ap.add_argument("--port", default="random", help="target port {0,1,2,random}")
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--enable_cameras", action="store_true", help="forced on for --policy lerobot")
    ap.add_argument("--endpoint", default="tcp://127.0.0.1:5555", help="lerobot policy server endpoint")
    ap.add_argument("--lerobot_python", default=os.environ.get("LEROBOT_PYTHON", str(DEFAULT_LEROBOT_PYTHON)))
    ap.add_argument("--n_action_steps", type=int, default=None, help="lerobot: actions executed per chunk")
    ap.add_argument("--server_device", default="cuda")
    ap.add_argument("--no_server", action="store_true", help="lerobot: connect to an already running server")
    ap.add_argument("--no_leaderboard", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--env_arg", action="append", default=[],
                    help="extra env flag passed after the env name, e.g. --env_arg=--foo=1 (repeatable)")
    ap.add_argument("--policy_arg", action="append", default=[],
                    help="extra policy cfg flag, e.g. --policy_arg=--expert_controller=oracle_spiral (repeatable)")
    ap.add_argument("--_worker", default=None, help=argparse.SUPPRESS)
    # debug: run the worker machinery on another Arena env / registered policy (no StageTracker results)
    ap.add_argument("--_env", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--_arena_policy", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.policy == "lerobot":
        if not args.policy_path:
            ap.error("--policy lerobot needs --policy.path")
        args.enable_cameras = True
    if args.pose_noise_mm is None:
        args.pose_noise_mm = 0.0 if args.policy == "expert" else 1.0
    args.tiers = [t for t in args.tiers.split(",") if t]
    if args.name is None:
        args.name = args.policy if args.policy != "lerobot" else "lerobot_" + _slug(args.policy_path)
    return args


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s.rstrip("/").split("/")[-1])


def episode_quota(n: int, num_envs: int) -> list[int]:
    """Episodes each env must finish so that every schedule slot ``i < n`` is filled."""
    return [(n - j + num_envs - 1) // num_envs if j < n else 0 for j in range(num_envs)]


def worker_config(args: argparse.Namespace, tier: str) -> dict:
    d = Path(args.out) / args.name / "raw"
    return {
        "policy": args.policy,
        "name": args.name,
        "tier": tier,
        "n": args.n,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "pose_noise_mm": args.pose_noise_mm,
        "port": args.port,
        "enable_cameras": args.enable_cameras,
        "endpoint": args.endpoint,
        "n_action_steps": args.n_action_steps,
        "env_args": list(args.env_arg),
        "policy_args": list(args.policy_arg),
        "env_name": args._env,
        "arena_policy": args._arena_policy,
        "jsonl": str(d / f"{tier}.jsonl"),
        "meta": str(d / f"{tier}.meta.json"),
    }


def arena_base_argv(cfg: dict) -> list[str]:
    """Arena CLI before the policy flags (app/Isaac Lab flags); headless unless --viz is given."""
    argv = ["--num_envs", str(cfg["num_envs"]), "--seed", str(cfg["seed"])]
    if cfg["enable_cameras"]:
        argv.append("--enable_cameras")
    return argv


def arena_env_argv(cfg: dict, policy_adds_noise: bool = True) -> list[str]:
    """Env subcommand flags. The port-pose estimate noise is applied once: by the policy when its cfg has
    ``pose_noise_mm`` (scripted controllers), otherwise by the env's ``--port_est_noise_mm`` (obs ``port_pos_est``)."""
    if cfg.get("env_name"):  # debug: built-in Arena env
        return [cfg["env_name"], *cfg["env_args"]]
    env_noise = 0.0 if policy_adds_noise else cfg["pose_noise_mm"]
    return ["--external_environment_class_path", ENV_CLASS, ENV_NAME, "--tier", cfg["tier"],
            "--port_est_noise_mm", str(env_noise), "--port", str(cfg["port"]), *cfg["env_args"]]


def launch_worker(cfg: dict, cfg_path: Path, log_path: Path) -> None:
    """One Isaac Sim process for one tier; stdout/stderr tee'd to ``log_path``."""
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES", ACCEPT_EULA="Y")
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    cmd = [sys.executable, str(Path(__file__).resolve()), "--policy", cfg["policy"], "--_worker", str(cfg_path)]
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            log.write(line)
            if line.startswith("[evaluate]") or "Error" in line or "Traceback" in line:
                print(line, end="", flush=True)
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"worker for tier {cfg['tier']} exited with {rc}; see {log_path}")


def start_policy_server(args: argparse.Namespace) -> subprocess.Popen | None:
    if args.policy != "lerobot" or args.no_server:
        return None
    py = Path(args.lerobot_python)
    if not py.exists():
        raise FileNotFoundError(f"lerobot venv python not found: {py} (set --lerobot_python or $LEROBOT_PYTHON)")
    cmd = [str(py), str(ROOT / "patchbench" / "policy_server.py"), "--endpoint", args.endpoint,
           "--device", args.server_device]
    cmd += ["--random"] if args.policy_path == "random" else ["--policy.path", args.policy_path]
    if args.n_action_steps:
        cmd += ["--n_action_steps", str(args.n_action_steps)]
    log = open(Path(args.out) / args.name / "policy_server.log", "w")
    print("[evaluate] policy server:", " ".join(cmd), flush=True)
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)


def stop_policy_server(proc: subprocess.Popen | None, endpoint: str) -> None:
    if proc is None:
        return
    try:
        from patchbench.policy_server import PolicyClient

        PolicyClient(endpoint, timeout_s=5, connect_timeout_s=5).shutdown_server()
        proc.wait(timeout=20)
    except Exception:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()


def finish_tier(args: argparse.Namespace, cfg: dict) -> dict:
    """Raw per-episode JSONL -> selected episodes -> <tier>.json + CSV."""
    records = rio.read_jsonl(cfg["jsonl"])
    missing = [k for k in ("furthest_stage",) if records and k not in records[0]]
    if missing:
        raise KeyError(f"episode records lack {missing}: the env must record StageTracker.summary() per episode "
                       f"(got keys {sorted(records[0])})")
    eps = rio.select_episodes(records, args.n, args.num_envs)
    if len(eps) < args.n:
        print(f"[evaluate] WARNING: {cfg['tier']}: only {len(eps)}/{args.n} scheduled episodes recorded", flush=True)
    meta = json.loads(Path(cfg["meta"]).read_text()) if Path(cfg["meta"]).exists() else {}
    extra = {"pose_noise_mm": args.pose_noise_mm, "num_envs": args.num_envs, "port": args.port,
             "cameras": bool(args.enable_cameras), **{k: meta[k] for k in ("wall_time_s", "sim_steps") if k in meta}}
    if args.policy == "lerobot":
        extra["policy_path"] = args.policy_path
    res = rio.summarize(eps, policy=args.name, tier=cfg["tier"], seed=args.seed, extra=extra)
    path = rio.write_result(args.out, res, eps)
    print(f"[evaluate] {args.name}/{cfg['tier']}: success {res['success']:.3f} {res['ci95']} n={res['n']} "
          f"funnel {res['funnel']} -> {path}", flush=True)
    return res


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args._worker:
        return worker(json.loads(Path(args._worker).read_text()))
    raw = Path(args.out) / args.name / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    cfgs = [worker_config(args, t) for t in args.tiers]
    if args.dry_run:
        for c in cfgs:
            print(json.dumps(c))
            print("  arena argv:", " ".join(arena_base_argv(c) + ["--policy_type", c["policy"], "<policy cfg flags>"]
                                            + arena_env_argv(c)))
        return
    server = start_policy_server(args)
    try:
        for c in cfgs:
            cfg_path = raw / f"{c['tier']}.worker.json"
            cfg_path.write_text(json.dumps(c, indent=1))
            print(f"[evaluate] {args.name}: tier {c['tier']}, n={args.n}, num_envs={args.num_envs}", flush=True)
            launch_worker(c, cfg_path, raw / f"{c['tier']}.log")
            finish_tier(args, c)
    finally:
        stop_policy_server(server, args.endpoint)
    mine = [r for r in rio.load_results(args.out) if r["policy"] == args.name]
    if mine:
        png = rio.plot_funnel(mine, Path(args.out) / args.name / "funnel.png")
        print(f"[evaluate] funnel plot -> {png}")
    if not args.no_leaderboard:
        sys.path.insert(0, str(ROOT / "scripts"))
        import leaderboard

        md, _ = leaderboard.build(args.out)
        print(f"[evaluate] leaderboard -> {md}")


# ------------------------------------------------------------------------------------------------ Isaac Sim worker
def worker(cfg: dict) -> None:
    """Runs inside the Arena venv, one Isaac Sim app per tier (mirrors isaaclab_arena/evaluation/policy_runner.py)."""
    base = arena_base_argv(cfg)
    sys.argv = [sys.argv[0], *base, *arena_env_argv(cfg)]  # Arena's env parser reads sys.argv itself

    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    parser = get_isaaclab_arena_cli_parser()
    args_cli, _ = parser.parse_known_args(sys.argv[1:])
    with SimulationAppContext(args_cli):
        import torch

        from isaaclab_arena.evaluation.policy_runner_cli import add_policy_cli_args, add_policy_runner_arguments
        from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

        policy_flags: list[str] = []
        policy_cls = None
        if cfg["policy"] in SCRIPTED:
            from isaaclab_arena.assets.registries import PolicyRegistry
            from isaaclab_arena.evaluation.policy_runner import get_policy_cls

            policy_type = cfg.get("arena_policy") or cfg["policy"]
            if not cfg.get("arena_policy"):
                import patchbench.policies  # noqa: F401  (registers open_loop / telemetry_spiral / oracle_spiral / expert)
            policy_cls = get_policy_cls(policy_type)
            fields = set(getattr(PolicyRegistry().get_policy_cfg_type(policy_cls), "__dataclass_fields__", {}))
            if "pose_noise_mm" in fields:
                policy_flags += ["--pose_noise_mm", str(cfg["pose_noise_mm"])]
            for f in POLICY_SEED_FIELDS:
                if f in fields:
                    policy_flags += [f"--{f}", str(cfg["seed"])]
            policy_flags = ["--policy_type", policy_type, *policy_flags, *cfg.get("policy_args", [])]

        policy_adds_noise = "--pose_noise_mm" in policy_flags
        sys.argv = [sys.argv[0], *base, *policy_flags, *arena_env_argv(cfg, policy_adds_noise)]
        add_policy_runner_arguments(parser)
        parser = get_isaaclab_arena_environments_cli_parser(parser)
        if policy_cls is not None:
            parser = add_policy_cli_args(parser, policy_cls)
        args_cli, unknown = parser.parse_known_args(sys.argv[1:])
        if cfg["enable_cameras"]:
            args_cli.enable_cameras = True
        assert args_cli.example_environment == (cfg.get("env_name") or ENV_NAME), args_cli.example_environment
        assert cfg.get("env_name") or "--tier" not in unknown, f"{ENV_NAME} does not accept --tier: {unknown}"
        if unknown:
            print(f"[evaluate] WARNING: unused CLI args {unknown}", flush=True)

        builder = get_arena_builder_from_cli(args_cli)
        _ensure_stage_recorder(builder)
        env = builder.make_registered()
        rec = env.unwrapped.episode_recorder
        rec.set_job_name(f"{cfg['name']}/{cfg['tier']}")
        rec.set_output_path(cfg["jsonl"])

        if policy_cls is not None:
            from isaaclab_arena.evaluation.policy_runner_cli import build_policy_from_cli

            policy = build_policy_from_cli(policy_cls, args_cli)
        else:
            policy = _make_remote_lerobot_policy(cfg, device=env.unwrapped.device)

        t0 = time.time()
        steps = rollout(env, policy, episode_quota(cfg["n"], env.unwrapped.num_envs))
        wall = time.time() - t0
        Path(cfg["meta"]).write_text(json.dumps({"wall_time_s": round(wall, 1), "sim_steps": steps,
                                                 "num_envs": env.unwrapped.num_envs}))
        print(f"[evaluate] tier {cfg['tier']}: {steps} steps x {env.unwrapped.num_envs} envs in {wall:.0f} s "
              f"({steps * env.unwrapped.num_envs / max(wall, 1e-6):.0f} env-steps/s)", flush=True)
        policy.close()
        env.close()
        del torch


def rollout(env, policy, quota: list[int]) -> int:
    """Step until every env has finished its quota of episodes; returns the number of env steps."""
    import torch

    num_envs = env.unwrapped.num_envs
    quota_t = torch.tensor(quota, device="cpu")
    done = torch.zeros(num_envs, dtype=torch.long)
    max_len = int(getattr(env.unwrapped, "max_episode_length", 1000))
    max_steps = (max(quota) + 1) * max_len + 10
    obs, _ = env.reset()
    policy.reset()
    policy.set_task_description(env.unwrapped.get_language_instruction())
    steps, last_print = 0, 0
    total = int(quota_t.sum())
    while bool((done < quota_t).any()):
        with torch.inference_mode():
            actions = policy.get_action(env, obs)
            obs, _, terminated, truncated, _ = env.step(actions)
        steps += 1
        ended = (terminated | truncated).nonzero().flatten()
        if ended.numel():
            policy.reset(env_ids=ended)
            done[ended.cpu()] += 1
            finished = int(torch.minimum(done, quota_t).sum())
            if finished - last_print >= max(1, total // 10) or finished == total:
                print(f"[evaluate] episodes {finished}/{total} (step {steps})", flush=True)
                last_print = finished
        if steps > max_steps:
            raise RuntimeError(f"rollout exceeded {max_steps} steps with {done.tolist()} / {quota} episodes done")
    return steps


def _ensure_stage_recorder(builder) -> None:
    """Make sure each finished episode records the StageTracker summary.

    If the environment did not add its own episode recorder term, add one that reads the first ``StageTracker``
    attribute found on the env (or its task) at reset time.
    """
    terms = builder.arena_env.episode_recorder_terms
    if terms:
        return
    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg

    terms["patchbench_stage"] = EpisodeRecorderTermCfg(func=_record_stage_summary)
    print("[evaluate] env has no episode recorder term; added patchbench_stage", flush=True)


def _find_tracker(env):
    from patchbench.predicates import StageTracker

    for holder in (env, getattr(env, "cfg", None)):
        for v in vars(holder).values() if holder is not None else ():
            if isinstance(v, StageTracker):
                return v
    return None


def _record_stage_summary(env, env_id: int) -> dict:
    tracker = _find_tracker(env)
    if tracker is None:
        return {}
    out = {k: v[env_id].item() for k, v in tracker.summary().items()}
    return {k: (None if isinstance(v, float) and v != v else v) for k, v in out.items()}


def _flatten(d, prefix=""):
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            yield from _flatten(v, key + ".")
        else:
            yield key, v


def _make_remote_lerobot_policy(cfg: dict, device):
    """Arena policy that forwards observations to the ZMQ policy server (patchbench/policy_server.py)."""
    import numpy as np
    import torch

    from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg
    from patchbench.policy_server import IMAGE_KEYS, PolicyClient

    class RemoteLeRobotPolicy(PolicyBase):
        """State in degrees (all 6 joints, LeRobot use_degrees convention), RGB uint8 images, per-env task strings.

        Observation keys are found by name in the flattened obs dict: state = ``policy.joint_pos`` (rad);
        images = the 4-D tensors whose key contains ``front`` / ``wrist``.
        """

        def __init__(self):
            super().__init__(PolicyCfg())
            self.client = PolicyClient(cfg["endpoint"], n_action_steps=cfg["n_action_steps"])
            print(f"[evaluate] connected to policy server: {self.client.info}", flush=True)
            self.keys = None

        def _resolve(self, flat):
            state = next((k for k in ("policy.joint_pos", "policy.joint_positions", "joint_pos") if k in flat), None)
            assert state is not None, f"no joint_pos in obs keys {sorted(flat)}"
            imgs = {}
            for cam in IMAGE_KEYS:
                short = cam.rsplit(".", 1)[-1]
                cand = [k for k, v in flat.items() if short in k and getattr(v, "ndim", 0) == 4]
                assert cand, f"no {short} image in obs keys {sorted(flat)} (cameras enabled?)"
                imgs[cam] = sorted(cand, key=lambda k: ("rgb" not in k, k))[0]
            print(f"[evaluate] obs mapping: state={state} images={imgs}", flush=True)
            return state, imgs

        def get_action(self, env, observation):
            flat = dict(_flatten(observation))
            if self.keys is None:
                self.keys = self._resolve(flat)
            sk, ik = self.keys
            state = np.degrees(flat[sk][:, :6].detach().cpu().numpy().astype(np.float32))
            images = {cam: flat[k].detach().cpu().numpy() for cam, k in ik.items()}
            u = env.unwrapped
            tasks = u.get_language_instructions() if hasattr(u, "get_language_instructions") else self.task_description
            act_deg = self.client.get_action(state, images, tasks or "")
            return torch.from_numpy(np.radians(act_deg)).to(device=device, dtype=torch.float32)

        def reset(self, env_ids=None):
            ids = None if env_ids is None else [int(i) for i in torch.as_tensor(env_ids).flatten().tolist()]
            self.client.reset(ids)

        def close(self):
            self.client.close()

    return RemoteLeRobotPolicy()


if __name__ == "__main__":
    main()
