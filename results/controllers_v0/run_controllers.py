"""Integration runner for the scripted controllers (P6): N episodes per (policy, tier), stage funnel + controller flags.

Builds the env exactly like Arena's policy_runner (same CLI parsers, ``make_registered``), but instantiates the policy
directly and snapshots the stage tracker just before every reset, so each finished episode is recorded exactly.

    cd ~/work/IsaacLab-Arena && source .venv/bin/activate && export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
    flock ~/work/.isaac.lock python ~/work/patchbench/results/controllers_v0/run_controllers.py --headless \
        --num_envs 8 --policies expert --episodes 20 --pnoise 0 \
        --external_environment_class_path patchbench.environment:RJ45PanelEnv rj45_panel --tier loose

Writes ``results/controllers_v0/<policy>_<tier>_n<N>_s<sigma>.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

OUT = Path(__file__).resolve().parent


def main():
    parser = get_isaaclab_arena_cli_parser()
    parser.add_argument("--policies", default="expert")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--pnoise", type=float, default=None, help="pose_noise_mm for the policy (default: cfg)")
    parser.add_argument("--policy_seed", type=int, default=0)
    parser.add_argument("--expert_controller", default="open_loop")
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--friction_mode", default=None, help="diagnosis only: override physics.SERVO_FRICTION_MODE")
    parser.add_argument("--trace_env", type=int, default=-1, help="print a per-step trace for this env id")
    args, _ = parser.parse_known_args()
    with SimulationAppContext(args):
        try:
            run(parser)
        except Exception:
            traceback.print_exc()
            sys.stdout.flush()
            raise


def run(parser):
    import torch
    from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

    import patchbench.policies as P
    from patchbench.controllers import PHASES
    from patchbench.predicates import STAGES, funnel
    from patchbench.task import get_state

    parser = get_isaaclab_arena_environments_cli_parser(parser)
    args, _ = parser.parse_known_args()
    if args.friction_mode:
        from patchbench import physics

        physics.SERVO_FRICTION_MODE = args.friction_mode
        print(f"[diag] SERVO_FRICTION_MODE overridden -> {args.friction_mode}", flush=True)
    env = get_arena_builder_from_cli(args).make_registered()
    tier = getattr(args, "tier", "?")
    E = env.unwrapped.num_envs
    classes = {"open_loop": P.OpenLoopPolicy, "telemetry_spiral": P.TelemetrySpiralPolicy,
               "oracle_spiral": P.OracleSpiralPolicy, "expert": P.ExpertPolicy}
    for name in args.policies.split(","):
        cls = classes[name]
        cfg_type = {P.OpenLoopPolicy: P.OpenLoopPolicyCfg, P.TelemetrySpiralPolicy: P.TelemetrySpiralPolicyCfg,
                    P.OracleSpiralPolicy: P.OracleSpiralPolicyCfg, P.ExpertPolicy: P.ExpertPolicyCfg}[cls]
        kw = {"policy_seed": args.policy_seed}
        if args.pnoise is not None:
            kw["pose_noise_mm"] = args.pnoise
        if cls is P.ExpertPolicy:
            kw["expert_controller"] = args.expert_controller
        policy = cls(cfg_type(**kw))
        with torch.inference_mode():  # so101-contact's actuator reset writes in place into inference tensors
            obs, _ = env.reset()
        policy.reset()
        st = get_state(env)
        episodes: list[dict] = []
        orig_reset = st.tracker.reset

        def snap(env_ids=None, _orig=orig_reset):
            ids = range(E) if env_ids is None else torch.as_tensor(env_ids).flatten().tolist()
            c = getattr(policy, "controller", None)
            summ = st.tracker.summary()
            for i in ids:
                if c is None or int(c.phase[i]) == 0 and steps_in_ep[i] == 0:
                    continue
                rec = {k: (v[i].item()) for k, v in summ.items()}
                rec.update(env_id=i, target_port=int(st.target_port[i]), steps=int(steps_in_ep[i]),
                           phase=PHASES[int(c.phase[i])])
                for f in ("contact_detected", "dropped_in", "latch_detected", "backstop_hit", "no_contact", "gave_up",
                          "restarts"):
                    if hasattr(c, f):
                        rec[f] = getattr(c, f)[i].item()
                rec["meas_depth_mm"] = float(c.meas_depth[i]) * 1e3
                rec["meas_lat_mm"] = (c.meas_lat[i] * 1e3).tolist()
                rec["noise_mm"] = (c.noise[i] * 1e3).tolist()
                episodes.append(rec)
            _orig(env_ids)

        st.tracker.reset = snap
        steps_in_ep = [0] * E
        t0 = time.time()
        for step in range(args.max_steps):
            with torch.inference_mode():
                act = policy.get_action(env, obs)
                obs, _, term, trunc, _ = env.step(act)
            steps_in_ep = [s + 1 for s in steps_in_ep]
            c = policy.controller
            if args.trace_env >= 0 and step % 3 == 0:
                i = args.trace_env
                sig = float(getattr(c, "signal", torch.zeros(E))[i])
                print(f"[trace] t={steps_in_ep[i]/30:5.2f} ph={PHASES[int(c.phase[i])]:11s} cmd_d={c.depth[i]*1e3:6.2f} "
                      f"meas_d={c.meas_depth[i]*1e3:6.2f} lat={[round(x*1e3,2) for x in c.meas_lat[i].tolist()]} "
                      f"cmd_lat={[round(x*1e3,2) for x in c.lat[i].tolist()]} sig={sig:7.2f} "
                      f"stage={int(st.tracker.furthest[i])} F={st.contact_force_w[i].norm():.2f} "
                      f"latch={float(st.latch_angle()[i])*57.3:5.1f}", flush=True)
                if step % 15 == 0:
                    o = policy.adapter.observe()
                    plug = st.plug_pose_w()
                    tip_b, _ = st.to_base(plug[:, :3], plug[:, 3:7])
                    fk, _ = c.servo.measured_pose(o.joint_pos, o.tool_p, o.tool_R)
                    from patchbench.controllers.ik import quat_xyzw_to_matrix
                    Rp = quat_xyzw_to_matrix(plug[:, 3:7])[i]
                    pos_w, quat_w = st.target_mouth_w()
                    Rm = quat_xyzw_to_matrix(quat_w)[i]
                    ax = torch.rad2deg(torch.acos((Rp[:, 0] * Rm[:, 0]).sum().clamp(-1, 1)))
                    rel = Rm.T @ Rp  # plug axes in the mouth frame
                    print(f"[ang] axis_err_deg={ax:.2f} plug_x_in_mouth={[round(x, 4) for x in rel[:, 0].tolist()]} "
                          f"plug_z_in_mouth={[round(x, 4) for x in rel[:, 2].tolist()]}", flush=True)
                    print(f"[dbg] q={[round(x, 4) for x in o.joint_pos[i].tolist()]} goal={[round(x, 4) for x in c.servo.goal[i].tolist()]} "
                          f"off_mm={[round(x*1e3, 2) for x in c.servo.offset[i].tolist()]} fk_vs_sim_mm={(fk[i]-tip_b[i]).norm()*1e3:.3f} "
                          f"load={o.load[i].tolist()}", flush=True)
            done = (term | trunc).nonzero().flatten()
            if len(done):
                policy.reset(done)
                for i in done.tolist():
                    steps_in_ep[i] = 0
            if len(episodes) >= args.episodes:
                break
        st.tracker.reset = orig_reset
        eps = episodes[: args.episodes]
        furthest = torch.tensor([e["furthest_stage"] for e in eps])
        res = {
            "policy": name, "tier": tier, "n": len(eps), "pose_noise_mm": policy.config.pose_noise_mm,
            "num_envs": E, "funnel": funnel(furthest), "success": float((furthest >= 5).float().mean()),
            "wall_s": round(time.time() - t0, 1), "episodes": eps,
        }
        if cls is P.ExpertPolicy:
            res["expert_controller"] = args.expert_controller
        sig = res["pose_noise_mm"]
        path = OUT / f"{name}_{tier}_n{len(eps)}_s{sig:g}.json"
        path.write_text(json.dumps(res, indent=1))
        print(f"[result] {name} {tier} sigma={sig} n={len(eps)} funnel={res['funnel']} wall={res['wall_s']}s -> {path}",
              flush=True)
        stages = {s: sum(1 for e in eps if e["furthest_stage"] == k) for k, s in enumerate(STAGES)}
        print(f"[result] furthest-stage histogram: {stages}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
