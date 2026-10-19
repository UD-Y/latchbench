"""SO-101 on STS3215 actuators: gravity hold, 30 deg step and press test (headless, PhysX, 1 env, no cameras).

Spawns two SO-101s (isaaclab_so101) side by side: one with all six joints on so101-contact's STS3215Actuator
(LeRobot settings), one with isaaclab_so101's implicit PD as the baseline.

(a) hold the home pose 5 s: steady-state sag per joint and Present_Load, vs the implicit-PD baseline
(b) 30 deg step on shoulder_lift (sim joint "Pitch"): q(t), duty(t), current(t)
(c) press the gripper into a fixed sphere: Present_Load on shoulder_lift / elbow_flex vs measured contact force

    cd ~/work/IsaacLab-Arena && source .venv/bin/activate && export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
    python ~/work/patchbench/scripts/hold_test.py [--friction_mode explicit|physx_joint] [--out DIR]

Writes PNGs and summary.json to patchbench/results/hold_test/.
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--friction_mode", default="explicit", choices=["explicit", "physx_joint"])
parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results/hold_test"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
simulation_app = AppLauncher(args).app

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sensors import ContactSensor, ContactSensorCfg  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402
from isaaclab_so101.assets import SO101_CFG  # noqa: E402
from isaaclab_so101.constants import HOME_JOINT_POS, TCP_OFFSET  # noqa: E402

from so101_contact.isaaclab.actuator import SO101_SIM_TO_LEROBOT, STS3215Actuator, so101_sts3215_cfg  # noqa: E402

DT = 1 / 240  # physics; telemetry and commands at 30 Hz (decimation 8)
DECIMATION = 8
SIM_JOINTS = list(SO101_SIM_TO_LEROBOT)
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)


def t(x):
    return x.torch if hasattr(x, "torch") else x


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args.device))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2000.0))

    servo_cfg = SO101_CFG.replace(prim_path="/World/servo", actuators=so101_sts3215_cfg(args.friction_mode, physics_dt=DT))
    servo_cfg.spawn.activate_contact_sensors = True
    servo = Articulation(servo_cfg)
    base_cfg = SO101_CFG.replace(prim_path="/World/baseline")
    base_cfg.init_state.pos = (0.0, 1.0, 0.0)
    baseline = Articulation(base_cfg)
    ball = RigidObject(
        RigidObjectCfg(
            prim_path="/World/ball",
            spawn=sim_utils.SphereCfg(
                radius=0.03,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                activate_contact_sensors=True,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -3.0, 0.5)),
        )
    )
    contact = ContactSensor(ContactSensorCfg(prim_path="/World/ball", filter_prim_paths_expr=["/World/servo/gripper", "/World/servo/jaw"]))
    sim.reset()
    groups = STS3215Actuator.bind_all(servo)
    print("[hold_test] STS3215 groups:", {g.cfg.joint_names_expr[0]: g.joint_names for g in groups})
    jid = [servo.joint_names.index(n) for n in SIM_JOINTS]
    jid_b = [baseline.joint_names.index(n) for n in SIM_JOINTS]
    home = torch.tensor([[HOME_JOINT_POS[n] for n in SIM_JOINTS]], device=sim.device)
    gripper_body = servo.find_bodies("gripper")[0][0]

    def telemetry():
        load = torch.zeros(len(SIM_JOINTS))
        cur = torch.zeros(len(SIM_JOINTS))
        duty = torch.zeros(len(SIM_JOINTS))
        current = torch.zeros(len(SIM_JOINTS))
        ext = torch.zeros(len(SIM_JOINTS))
        for g in groups:
            for k, name in enumerate(g.joint_names):
                i = SIM_JOINTS.index(name)
                load[i] = g.telemetry["Present_Load"][0, k].item()
                cur[i] = g.telemetry["Present_Current"][0, k].item()
                duty[i] = g.duty[0, k].item()
                current[i] = g.current[0, k].item()
                ext[i] = g.external_torque[0, k].item()
        return load, cur, duty, current, ext

    def tcp():
        pos = t(servo.data.body_link_pos_w)[0, gripper_body]
        quat = t(servo.data.body_link_quat_w)[0, gripper_body]
        return pos + quat_apply(quat.unsqueeze(0), torch.tensor([TCP_OFFSET], device=sim.device))[0]

    def step(goal, goal_b=None):
        servo.actuators.target_command.set_position_index(value=goal, joint_ids=jid)
        baseline.actuators.target_command.set_position_index(value=goal if goal_b is None else goal_b, joint_ids=jid_b)
        for art in (servo, baseline):
            art.write_data_to_sim()
        sim.step(render=False)
        for art in (servo, baseline):
            art.update(DT)
        ball.update(DT)
        contact.update(DT)

    def reset_home():
        for art, ids in ((servo, jid), (baseline, jid_b)):
            q = t(art.data.default_joint_pos).clone()
            q[:, ids] = home
            art.write_joint_position_to_sim_index(position=q)
            art.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
            art.reset()

    summary = {"friction_mode": args.friction_mode, "physics_dt": DT}
    t0 = time.time()

    # (a) gravity hold ------------------------------------------------------------------------------------------
    reset_home()
    log_a = []
    for i in range(int(5.0 / DT)):
        step(home)
        if i % DECIMATION == 0:
            q = t(servo.data.joint_pos)[0, jid].cpu()
            log_a.append((i * DT, *q.tolist(), *telemetry()[0].tolist()))
    q_s = t(servo.data.joint_pos)[0, jid].cpu()
    q_b = t(baseline.data.joint_pos)[0, jid_b].cpu()
    load, cur, duty, current, ext = telemetry()
    grav = -t(servo.data.gravity_compensation_forces)[0, jid].cpu()
    qd_s = t(servo.data.joint_vel)[0, jid].cpu()
    a = {
        "sag_deg_sts3215": dict(zip(SIM_JOINTS, np.degrees((q_s - home[0].cpu()).numpy()).round(2).tolist())),
        "sag_deg_implicit_pd": dict(zip(SIM_JOINTS, np.degrees((q_b - home[0].cpu()).numpy()).round(2).tolist())),
        "present_load": dict(zip(SIM_JOINTS, load.int().tolist())),
        "present_current": dict(zip(SIM_JOINTS, cur.int().tolist())),
        "external_torque_estimate": dict(zip(SIM_JOINTS, ext.numpy().round(4).tolist())),
        "gravity_torque": dict(zip(SIM_JOINTS, grav.numpy().round(4).tolist())),
        "max_abs_joint_vel_end": float(qd_s.abs().max()),
    }
    summary["hold"] = a
    print("[hold_test] (a) hold:", json.dumps(a, indent=1))

    # (b) 30 deg shoulder_lift step -------------------------------------------------------------------------------
    pitch = SIM_JOINTS.index("Pitch")
    goal = home.clone()
    goal[0, pitch] += math.radians(30.0)
    log_b = []
    for i in range(int(2.0 / DT)):
        step(goal)
        q = t(servo.data.joint_pos)[0, jid[pitch]].item()
        qb = t(baseline.data.joint_pos)[0, jid_b[pitch]].item()
        _, _, duty, current, ext = telemetry()
        log_b.append((i * DT, q, qb, duty[pitch].item(), current[pitch].item(), ext[pitch].item()))
    lb = np.array(log_b)
    err = np.abs(lb[:, 1] - goal[0, pitch].item())
    final = abs(lb[-1, 1] - goal[0, pitch].item())
    band = max(math.radians(1.0), 1.5 * final)
    outside = np.where(err > band)[0]
    settle = float(lb[outside[-1] + 1, 0]) if len(outside) else 0.0
    overshoot = float(np.degrees(max(0.0, (lb[:, 1] - goal[0, pitch].item()).max())))
    b = {"settle_time_s(1deg_band)": settle, "final_error_deg": float(np.degrees(final)),
         "overshoot_deg": overshoot, "peak_duty": float(np.abs(lb[:, 3]).max()),
         "peak_current_A": float(np.abs(lb[:, 4]).max())}  # fmt: skip
    summary["step"] = b
    print("[hold_test] (b) step:", b)

    # (c) press test ---------------------------------------------------------------------------------------------
    reset_home()
    for _ in range(int(1.0 / DT)):
        step(home)
    p0 = tcp()
    probe = home.clone()
    probe[0, pitch] += 0.15
    for _ in range(int(1.0 / DT)):
        step(probe)
    d = tcp() - p0
    d = d / d.norm()
    reset_home()
    for _ in range(int(1.0 / DT)):
        step(home)
    p0 = tcp()
    radius, gap = 0.03, 0.01
    pose = t(ball.data.default_root_pose).clone()
    pose[0, :3] = p0 + d * (gap + radius)
    ball.write_root_pose_to_sim_index(root_pose=pose)
    elbow = SIM_JOINTS.index("Elbow")
    log_c = []
    seconds = 8.0
    for i in range(int(seconds / DT)):
        g = home.clone()
        g[0, pitch] += 0.3 * min(1.0, i * DT / (seconds - 1.0))
        step(g)
        if i % DECIMATION == 0:
            f = t(contact.data.net_forces_w)[0].norm(dim=-1).sum().item()
            load, cur, duty, _, ext = telemetry()
            log_c.append((i * DT, f, load[pitch].item(), load[elbow].item(), cur[pitch].item(), cur[elbow].item(),
                          ext[pitch].item(), ext[elbow].item(), duty[pitch].item()))  # fmt: skip
    lc = np.array(log_c)
    touching = lc[:, 1] > 0.2
    c = {"max_contact_force_N": float(lc[:, 1].max()), "contact_frames": int(touching.sum()),
         "load_pitch_free_to_max": [float(lc[~touching, 2].mean()) if (~touching).any() else None,
                                    float(lc[:, 2][np.argmax(lc[:, 1])])],
         "load_elbow_free_to_max": [float(lc[~touching, 3].mean()) if (~touching).any() else None,
                                    float(lc[:, 3][np.argmax(lc[:, 1])])],
         "press_direction_w": d.cpu().numpy().round(3).tolist()}  # fmt: skip
    if touching.sum() > 5:
        c["spearman_like_corr_load_pitch_vs_force"] = float(np.corrcoef(lc[touching, 1], lc[touching, 2])[0, 1])
    summary["press"] = c
    summary["wall_s"] = time.time() - t0
    print("[hold_test] (c) press:", c)

    np.savetxt(OUT / "hold.csv", np.array(log_a), delimiter=",", header="t," + ",".join(
        [f"q_{n}" for n in SIM_JOINTS] + [f"load_{n}" for n in SIM_JOINTS]))  # fmt: skip
    np.savetxt(OUT / "step.csv", lb, delimiter=",", header="t,q_sts,q_pd,duty,current,ext")
    np.savetxt(OUT / "press.csv", lc, delimiter=",",
               header="t,force,load_pitch,load_elbow,cur_pitch,cur_elbow,ext_pitch,ext_elbow,duty_pitch")  # fmt: skip
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    plots(np.array(log_a), lb, lc, goal[0, pitch].item())
    print(f"[hold_test] wrote {OUT}")


def plots(la, lb, lc, goal):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True, constrained_layout=True)
    ax[0].plot(lb[:, 0], np.degrees(lb[:, 1]), label="STS3215 actuator")
    ax[0].plot(lb[:, 0], np.degrees(lb[:, 2]), "--", label="implicit PD (isaaclab_so101)")
    ax[0].axhline(np.degrees(goal), c="gray", ls=":", label="goal")
    ax[0].set_ylabel("shoulder_lift q [deg]")
    ax[0].legend(fontsize=8)
    ax[1].plot(lb[:, 0], lb[:, 3])
    ax[1].set_ylabel("duty [-]")
    ax[2].plot(lb[:, 0], lb[:, 4])
    ax[2].set_ylabel("current [A]")
    ax[2].set_xlabel("time [s]")
    fig.suptitle(f"30° shoulder_lift step, LeRobot P=16 ({args.friction_mode} friction)")
    fig.savefig(OUT / "step_response.png", dpi=130)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    ax[0].plot(lc[:, 0], lc[:, 1], c="k", label="contact force [N]")
    ax0 = ax[0].twinx()
    ax0.plot(lc[:, 0], lc[:, 2], label="Present_Load shoulder_lift")
    ax0.plot(lc[:, 0], lc[:, 3], label="Present_Load elbow_flex")
    ax0.set_ylabel("Present_Load [counts]")
    ax[0].set_xlabel("time [s]")
    ax[0].set_ylabel("contact force [N]")
    ax0.legend(fontsize=8, loc="upper left")
    ax[1].scatter(lc[:, 1], lc[:, 2], s=8, label="shoulder_lift")
    ax[1].scatter(lc[:, 1], lc[:, 3], s=8, label="elbow_flex")
    ax[1].set_xlabel("gripper contact force [N]")
    ax[1].set_ylabel("Present_Load [counts, 30 Hz, ±1 noise]")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.suptitle("Gripper pressed into a fixed sphere by a slow shoulder_lift ramp")
    fig.savefig(OUT / "press.png", dpi=130)

    fig, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
    n = len(SIM_JOINTS)
    for k, name in enumerate(SIM_JOINTS):
        ax.plot(la[:, 0], la[:, 1 + n + k], label=f"{SO101_SIM_TO_LEROBOT[name]}")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("Present_Load [counts]")
    ax.set_title("Gravity hold at the home pose")
    ax.legend(fontsize=7, ncol=3)
    fig.savefig(OUT / "hold_load.png", dpi=130)


if __name__ == "__main__":
    from isaaclab.sim import SimulationContext

    import traceback

    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
