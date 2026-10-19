"""Arena env v0 checks for ``rj45_panel`` (headless): hold, camera frames, joint-space push, physics throughput.

Builds the env exactly as policy_runner does, then
  * holds the start pose (plug 30 mm in front of the middle port) for ``--hold_s`` and prints where the plug is
    relative to the target mouth, the latch angle, the contact force and the servo telemetry;
  * ``--frames DIR``: saves the front and wrist camera images (needs ``--enable_cameras``);
  * ``--push``: drives the plug tip along the target port's axis with the closed-form 5-DOF IK (isaaclab_so101.ee_pose)
    plus an integral correction for the 7 mm TCP offset, from -15 mm to seat + 3 mm at 10 mm/s, and logs stage /
    depth / latch / axial force (a joint-space sanity check, not one of the benchmark controllers);
  * ``--bench N``: times N control steps after the hold and reports physics steps/s.

    cd ~/work/IsaacLab-Arena && source .venv/bin/activate && export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
    flock ~/work/.isaac.lock python ~/work/patchbench/results/arena_env_v0/check_env.py --push \
        --external_environment_class_path patchbench.environment:RJ45PanelEnv rj45_panel --tier loose --no_randomize
"""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

parser = get_isaaclab_arena_cli_parser()
parser.add_argument("--hold_s", type=float, default=1.0)
parser.add_argument("--frames", type=str, default=None)
parser.add_argument("--push", action="store_true")
parser.add_argument("--bench", type=int, default=0)
parser.add_argument("--gain", type=float, default=0.08, help="push: integral gain on the tip error, per step")
parser.add_argument("--speed", type=float, default=0.005, help="push: insertion speed [m/s]")
parser.add_argument("--json", type=str, default=None, help="write the printed summary here")


def main(args):
    import numpy as np
    import torch
    from isaaclab_so101.constants import PAN_AXIS_XY, SIM_JOINT_NAMES
    from isaaclab_so101.ee_pose import natural_ik

    from patchbench import physics
    from patchbench.environment import start_joint_pose
    from patchbench.objects import DIMS, PLUG_GRASP_BEHIND, wrist_roll_for
    from patchbench.predicates import STAGES, port_coordinates
    from patchbench.task import get_state
    from isaaclab_arena_environments.cli import get_arena_builder_from_cli

    builder = get_arena_builder_from_cli(args)
    env = builder.make_registered()
    u = env.unwrapped
    obs, _ = env.reset()
    st = get_state(u)
    E, dev = u.num_envs, u.device
    out = {"num_envs": E, "tier": st.p.tier, "robot_joints": st.robot.joint_names, "robot_bodies": st.robot.body_names}
    q0 = start_joint_pose(st.p.panel_pos)
    hold = torch.tensor([[q0[n] for n in SIM_JOINT_NAMES]], device=dev).repeat(E, 1)

    def report(tag):
        st.update(u)
        ports = st.ports()
        tip = st.plug_pose_w()[:, :3]
        c = port_coordinates(tip, ports)[torch.arange(E, device=dev), st.target_port]
        pol = obs["policy"]
        r = {
            "target_port": st.target_port.tolist(),
            "tip_in_mouth_mm(depth,width,height)": (c * 1e3).round(decimals=2).tolist(),
            "latch_deg": torch.rad2deg(st.latch_angle()).round(decimals=2).tolist(),
            "contact_force_N": st.contact_force_w.norm(dim=-1).round(decimals=3).tolist(),
            "furthest_stage": [STAGES[k] for k in st.tracker.furthest.tolist()],
            "servo_load_env0": pol["servo_load"][0].tolist(),
            "servo_current_env0": pol["servo_current"][0].tolist(),
            "joint_err_deg_env0": torch.rad2deg(pol["joint_pos"][0] - hold[0]).round(decimals=2).tolist(),
            "plug_tip_pos_b_env0": pol["plug_tip_pos"][0].round(decimals=4).tolist(),
            "port_pos_est_b_env0": pol["port_pos_est"][0].round(decimals=4).tolist(),
            "eef_pos_b_env0": pol["eef_pos"][0].round(decimals=4).tolist(),
        }
        print(f"[check_env] {tag}:", json.dumps(r))
        out[tag] = r

    n_hold = int(args.hold_s / u.step_dt)
    for _ in range(n_hold):
        obs, *_ = env.step(hold)
    report("hold")

    if args.frames:
        from PIL import Image

        d = Path(args.frames)
        d.mkdir(parents=True, exist_ok=True)
        cams = obs["camera_obs"]
        for key, name in (("external_camera_rgb", "front"), ("camera_ego_rgb", "wrist")):
            img = cams[key][0].cpu().numpy()
            Image.fromarray(img[..., :3].astype(np.uint8)).save(d / f"{st.p.tier}_{name}.png")
        print(f"[check_env] frames -> {d}")

    if args.bench:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.bench):
            env.step(hold)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        sps = args.bench * physics.DECIMATION / dt
        out["bench"] = {"control_steps": args.bench, "wall_s": dt, "physics_steps_per_s": sps,
                        "env_physics_steps_per_s": sps * E, "real_time_factor": sps * physics.PHYSICS_DT,
                        "friction_mode": getattr(args, "servo_friction_mode", physics.SERVO_FRICTION_MODE)}  # fmt: skip
        print("[check_env] bench:", json.dumps(out["bench"]))

    if args.push:
        # Phase A (free space): servo the tip to 5 mm in front of the mouth, integrating the tip error into the
        # TCP command (7 mm TCP offset, P-only servo sag) until it is within 0.2 mm laterally. Phase B: push along
        # the axis at 10 mm/s to seat + 3 mm with the correction frozen, then hold 1.5 s.
        L = DIMS.plug_length + PLUG_GRASP_BEHIND
        speed, pre, end = args.speed, -0.005, DIMS.seat_depth + 0.003
        settled = 0
        axis = torch.tensor([0.0, -1.0, 0.0])  # nominal insertion axis in base (env +X); use with --no_randomize
        corr = torch.zeros(E, 3)
        log, phase, t_push = [], "align", None
        n_max = int(6.0 / u.step_dt) + int((end - pre) / speed / u.step_dt) + int(1.5 / u.step_dt)
        for k in range(n_max):
            pol = obs["policy"]
            port_b = obs["privileged"]["port_pos"].cpu()  # true mouth (sanity check only)
            tip = pol["plug_tip_pos"].cpu()
            depth = (tip - port_b) @ axis
            if phase == "align":
                depth_cmd = pre
                err = port_b + pre * axis - tip
                corr += args.gain * err
                lat = (err - (err @ axis)[:, None] * axis).norm(dim=-1)
                settled = settled + 1 if bool((lat < 2e-4).all()) else 0
                if settled >= 15 or k * u.step_dt > 6.0:
                    phase, t_push = "push", k
                    print(f"[check_env] aligned after {k * u.step_dt:.2f} s, lateral err {lat.tolist()} m")
            else:
                depth_cmd = min(end, pre + speed * (k - t_push) * u.step_dt)
                err = port_b + depth_cmd * axis - tip
                err -= (err @ axis)[:, None] * axis
                free = (depth < -0.001).float()[:, None]  # keep correcting lateral error until the tip enters
                corr += args.gain * err * free
            tcp = port_b + depth_cmd * axis - L * axis + corr
            act = hold.clone()
            for e in range(E):
                x, y, z = tcp[e].tolist()
                psi = math.atan2(y - PAN_AXIS_XY[1], x - PAN_AXIS_XY[0])
                q = natural_ik(x, y, z, 0.0, wrist_roll_for(psi))
                if q is not None:
                    act[e, :5] = torch.tensor(q, device=dev)
            obs, _, term, trunc, _ = env.step(act)
            st.update(u)
            ar = torch.arange(E, device=dev)
            c = port_coordinates(st.plug_pose_w()[:, :3], st.ports())[ar, st.target_port]
            ax = -(st.port_force_w[ar, st.target_port] * st.ports().mouth_rot[ar, st.target_port, :, 0]).sum(-1)
            log.append((k * u.step_dt, depth_cmd * 1e3, float(c[0, 0]) * 1e3, float(c[0, 1]) * 1e3,
                        float(c[0, 2]) * 1e3, math.degrees(float(st.latch_angle()[0])), float(ax[0]),
                        int(st.tracker.furthest[0]), float(obs["policy"]["servo_load"][0, 1]),
                        float(obs["policy"]["servo_load"][0, 2])))  # fmt: skip
            if k % 15 == 0:
                print("[check_env] push t=%.2f cmd=%.1f depth=%.2f w=%.2f h=%.2f latch=%.1f axial=%.2f stage=%d "
                      "load_pitch=%d load_elbow=%d" % log[-1])  # fmt: skip
            if bool(term[0] | trunc[0]):
                ended = {n: bool(u.termination_manager.get_term(n)[0]) for n in u.termination_manager.active_terms}
                print(f"[check_env] episode ended at t={k * u.step_dt:.2f} s: {ended}")
                out["push_end"] = ended
                break
            if phase == "push" and (k - t_push) * u.step_dt > (end - pre) / speed + 1.5:
                break
        arr = np.array(log)
        out["push"] = {"max_stage": STAGES[int(arr[:, 7].max())], "max_depth_mm": float(arr[:, 2].max()),
                       "min_latch_deg": float(arr[:, 5].min()), "peak_axial_N": float(arr[:, 6].max())}  # fmt: skip
        if args.json:
            np.savetxt(Path(args.json).with_suffix(".push.csv"), arr, delimiter=",", fmt="%.4g",
                       header="t,cmd_mm,depth_mm,width_mm,height_mm,latch_deg,axial_N,stage,load_pitch,load_elbow")  # fmt: skip
        print("[check_env] push summary:", json.dumps(out["push"]))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
    env.close()


if __name__ == "__main__":
    args_cli, _ = parser.parse_known_args()
    with SimulationAppContext(args_cli):
        from isaaclab_arena_environments.cli import get_isaaclab_arena_environments_cli_parser

        full = get_isaaclab_arena_environments_cli_parser(parser)
        args_cli = full.parse_args()
        try:
            main(args_cli)
        except Exception:
            traceback.print_exc()
            raise
