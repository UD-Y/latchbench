"""Physics step rate of N SO-101s on STS3215 actuators, per servo friction mode (headless, no cameras).

``physx_joint`` writes the friction budget to PhysX through the CPU every physics step; this measures what that
costs against ``explicit``. Run once per mode (one Isaac process at a time):

    python scripts/bench_friction_mode.py --friction_mode physx_joint --num_envs 16
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--friction_mode", default="physx_joint", choices=["explicit", "physx_joint"])
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--seconds", type=float, default=10.0, help="simulated seconds to time")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
simulation_app = AppLauncher(args).app

import json  # noqa: E402
import time  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import torch  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab_so101.assets import SO101_CFG  # noqa: E402
from isaaclab_so101.constants import HOME_JOINT_POS  # noqa: E402

from so101_contact.isaaclab.actuator import SO101_SIM_TO_LEROBOT, STS3215Actuator, so101_sts3215_cfg  # noqa: E402

DT, DECIMATION = 1 / 240, 8


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args.device))
    side = int(args.num_envs**0.5 + 0.999)
    for i in range(args.num_envs):
        sim_utils.create_prim(f"/World/env_{i}", "Xform", translation=(0.6 * (i % side), 0.6 * (i // side), 0.0))
    cfg = SO101_CFG.replace(prim_path="/World/env_.*/Robot",
                            actuators=so101_sts3215_cfg(args.friction_mode, physics_dt=DT))  # fmt: skip
    robot = Articulation(cfg)
    sim.reset()
    STS3215Actuator.bind_all(robot)
    names = list(SO101_SIM_TO_LEROBOT)
    jid = [robot.joint_names.index(n) for n in names]
    home = torch.tensor([[HOME_JOINT_POS[n] for n in names]], device=sim.device).repeat(args.num_envs, 1)
    gen = torch.Generator(device="cpu").manual_seed(0)

    def run(steps):
        goal = home.clone()
        for i in range(steps):
            if i % DECIMATION == 0 and i % (DECIMATION * 15) == 0:  # new random goal every 0.5 s
                goal = home + (torch.rand(home.shape, generator=gen).to(sim.device) - 0.5) * 0.6
            robot.actuators.target_command.set_position_index(value=goal, joint_ids=jid)
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(DT)

    run(240)  # warm-up
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    steps = int(args.seconds / DT)
    t0 = time.perf_counter()
    run(steps)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    wall = time.perf_counter() - t0
    q = robot.data.joint_pos.torch if hasattr(robot.data.joint_pos, "torch") else robot.data.joint_pos
    out = {"friction_mode": args.friction_mode, "num_envs": args.num_envs, "physics_steps": steps, "wall_s": wall,
           "physics_steps_per_s": steps / wall, "env_steps_per_s": steps * args.num_envs / wall,
           "realtime_factor": steps * DT / wall, "finite": bool(torch.isfinite(q).all())}  # fmt: skip
    print("[bench]", json.dumps(out))


if __name__ == "__main__":
    import traceback

    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
