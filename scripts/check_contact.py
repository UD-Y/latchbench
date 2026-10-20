"""Virtual-wrist contact tuning: RJ45 plug into port_1, no robot (headless PhysX, no cameras).

The plug hangs on a compliant 6-DOF "virtual wrist", a serial chain rail -> x, y, z prismatic -> rx, ry, rz revolute
-> plug body, all in port_1's jack frame (+X insertion, +Z latch side), with implicit drives: 2000 N/m laterally
and axially, 5 N*m/rad in rotation (CLAUDE.md P3). The x target moves at 10 mm/s from 15 mm in front of the mouth
to seat_depth + 3 mm, holds, then pulls back 3 mm (retention check). Each env is one variant
(tier x lateral offset x tilt); seeds re-run all envs with a new in-plane offset direction and tilt axis.

Contact sensors on the plug body and latch, filtered against the panel, give the plug-jack force at physics rate.
One run = one physics config (dt, solver position iterations, contact offset; rest offset 0) on top of Isaac Lab's
Factory solver settings (isaaclab_tasks/contrib/factory/factory_env_cfg.py: TGS, 1 velocity iteration,
gpu_max_num_partitions=1, bounce 0.2, friction offset 0.01, correlation 0.00625, max depenetration 5 m/s).

    python scripts/check_contact.py --dt 0.0020833 --pos_iters 32 --contact_offset 0.0005 --run_id c_480_32_05
Writes results/contact_tuning/<run_id>/{summary.json, traces.csv.gz, force_vs_depth.png, jam_map.png}.

Watch one aligned insertion (1 env, laptop: nothing else Isaac-related running):
    python scripts/check_contact.py --viz kit --tiers medium --offsets_mm 0 --tilts_deg 0 --seeds 1 \
        --pos_iters 32 --cutaway --slowmo 10 --hold_open
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--dt", type=float, default=1 / 480)
parser.add_argument("--pos_iters", type=int, default=32)
parser.add_argument("--vel_iters", type=int, default=1)
parser.add_argument("--contact_offset", type=float, default=0.5e-3, help="m")
parser.add_argument("--rest_offset", type=float, default=0.0, help="m")
parser.add_argument("--tiers", default="loose,medium")
parser.add_argument("--offsets_mm", default="0,0.2,0.5,1.0")
parser.add_argument("--tilts_deg", default="0,1,2")
parser.add_argument("--seeds", type=int, default=5)
parser.add_argument("--friction", type=float, default=0.35, help="static = dynamic friction, plastic on plastic")
parser.add_argument("--run_id", default=None)
parser.add_argument("--out", type=Path, default=None, help="default results/contact_tuning (results/gui with --viz kit)")
parser.add_argument("--slowmo", type=float, default=0.0, help="GUI: play N x slower than real time (0 = as fast as possible)")
parser.add_argument("--hold_open", action="store_true", help="GUI: keep the window open after the run")
parser.add_argument("--cutaway", action="store_true", help="hide env 0's panel on the camera side of the latch slot")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
GUI = "kit" in str(args.visualizer or "")  # headless unless --viz kit
simulation_app = AppLauncher(args).app

import math  # noqa: E402
import time  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import torch  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sensors import ContactSensor, ContactSensorCfg  # noqa: E402
from isaaclab.sim.utils.stage import get_current_stage  # noqa: E402
from isaaclab_physx.physics import PhysxCfg  # noqa: E402
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

from patchbench.assets.build_assets import USD_DIR, jack_geometry  # noqa: E402
from patchbench.contact_probe import CHAIN, D_LIN, D_ROT, DIMS, K_LIN, K_ROT, START, Probe  # noqa: E402

TIERS = args.tiers.split(",")
OFFSETS = [float(x) * 1e-3 for x in args.offsets_mm.split(",")]
TILTS = [math.radians(float(x)) for x in args.tilts_deg.split(",")]
VARIANTS = [(tier, o, tl) for tier in TIERS for o in OFFSETS for tl in TILTS]
N = len(VARIANTS)
PANEL_Z = 0.10
FLIP = Gf.Quatd(0.0, Gf.Vec3d(1.0, 0.0, 0.0))  # port_1 jack frame in the panel frame (latch side down)
RUN_ID = args.run_id or f"dt{round(1 / args.dt)}_it{args.pos_iters}_co{args.contact_offset * 1e3:g}"
OUT = (args.out or Path(__file__).resolve().parents[1] / "results" / ("gui" if GUI else "contact_tuning")) / RUN_ID


def origin(i):
    return (0.12 * (i % 6), 0.12 * (i // 6), 0.0)


def build_scene(stage: Usd.Stage):
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    mat = UsdShade.Material.Define(stage, "/World/plastic")
    m = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    m.CreateStaticFrictionAttr(args.friction)
    m.CreateDynamicFrictionAttr(args.friction)
    m.CreateRestitutionAttr(0.0)
    for i, (tier, _, _) in enumerate(VARIANTS):
        env = f"/World/envs/env_{i}"
        o = origin(i)
        UsdGeom.Xform.Define(stage, env)
        panel = UsdGeom.Xform.Define(stage, f"{env}/Panel")
        panel.GetPrim().GetReferences().AddReference(str(USD_DIR / tier / f"panel_3port_{tier}.usd"))
        panel.AddTranslateOp().Set(Gf.Vec3d(o[0], o[1], PANEL_Z))

        rig = UsdGeom.Xform.Define(stage, f"{env}/Rig")
        rig.AddTranslateOp().Set(Gf.Vec3d(o[0], o[1], PANEL_Z))
        rig.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(FLIP)
        UsdPhysics.ArticulationRootAPI.Apply(rig.GetPrim())
        art = PhysxSchema.PhysxArticulationAPI.Apply(rig.GetPrim())
        art.CreateSolverPositionIterationCountAttr(args.pos_iters)
        art.CreateSolverVelocityIterationCountAttr(args.vel_iters)
        art.CreateEnabledSelfCollisionsAttr(False)
        links = ["rail"] + [f"l_{n}" for n in CHAIN[:-1]]
        for name in links:
            p = UsdGeom.Xform.Define(stage, f"{env}/Rig/{name}").GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(p)
            mass = UsdPhysics.MassAPI.Apply(p)
            mass.CreateMassAttr(0.1 if name == "rail" else 1e-3)
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-7, 1e-7, 1e-7))
        fixed = UsdPhysics.FixedJoint.Define(stage, f"{env}/Rig/rail_fixed")
        fixed.CreateBody1Rel().SetTargets([f"{env}/Rig/rail"])
        fixed.CreateLocalPos0Attr(Gf.Vec3f(o[0], o[1], PANEL_Z))
        fixed.CreateLocalRot0Attr(Gf.Quatf(FLIP))
        plug = UsdGeom.Xform.Define(stage, f"{env}/Rig/Plug")
        plug.GetPrim().GetReferences().AddReference(str(USD_DIR / tier / "plug.usd"))
        plug.AddTranslateOp().Set(Gf.Vec3d(START, 0.0, 0.0))
        bodies = links + ["Plug/body"]
        for k, n in enumerate(CHAIN):
            cls = UsdPhysics.PrismaticJoint if k < 3 else UsdPhysics.RevoluteJoint
            j = cls.Define(stage, f"{env}/Rig/{n}")
            j.CreateBody0Rel().SetTargets([f"{env}/Rig/{bodies[k]}"])
            j.CreateBody1Rel().SetTargets([f"{env}/Rig/{bodies[k + 1]}"])
            j.CreateAxisAttr("XYZ"[k % 3])
            if k < 3:
                lim = (-0.05, 0.05) if k == 0 else (-0.005, 0.005)
            else:
                lim = (-30.0, 30.0)  # deg
            j.CreateLowerLimitAttr(lim[0])
            j.CreateUpperLimitAttr(lim[1])
            UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear" if k < 3 else "angular")
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(env)).Bind(
            mat, UsdShade.Tokens.weakerThanDescendants, "physics"
        )
    # Collision offsets on every collider; contact reports + depenetration limit on the plug links.
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            c = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            c.CreateContactOffsetAttr(args.contact_offset)
            c.CreateRestOffsetAttr(args.rest_offset)
        path = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and (path.endswith("/Plug/body") or path.endswith("/Plug/latch")):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateMaxDepenetrationVelocityAttr(5.0)


def main():
    physics = PhysxCfg(
        solver_type=1, max_position_iteration_count=255, max_velocity_iteration_count=1, bounce_threshold_velocity=0.2,
        friction_offset_threshold=0.01, friction_correlation_distance=0.00625, gpu_max_rigid_contact_count=2**23,
        gpu_max_rigid_patch_count=2**23, gpu_collision_stack_size=2**28, gpu_max_num_partitions=1,
    )  # fmt: skip
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=args.dt, device=args.device, physics=physics))
    build_scene(get_current_stage())
    actuators = {
        "lin": ImplicitActuatorCfg(joint_names_expr=["x", "y", "z"], stiffness=K_LIN, damping=D_LIN),
        "rot": ImplicitActuatorCfg(joint_names_expr=["rx", "ry", "rz"], stiffness=K_ROT, damping=D_ROT),
        "latch": ImplicitActuatorCfg(joint_names_expr=["latch_joint"], stiffness=None, damping=None),
    }
    rig = Articulation(ArticulationCfg(prim_path="/World/envs/env_.*/Rig", spawn=None, actuators=actuators,
                                       init_state=ArticulationCfg.InitialStateCfg(joint_pos={"x": START})))  # fmt: skip
    filt = ["/World/envs/env_.*/Panel"]
    s_body = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_.*/Rig/Plug/body", filter_prim_paths_expr=filt,
                                            track_friction_forces=True, max_contact_data_count_per_prim=64))
    s_latch = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_.*/Rig/Plug/latch", filter_prim_paths_expr=filt,
                                             track_friction_forces=True, max_contact_data_count_per_prim=64))
    sim.reset()
    if GUI:  # side view of env 0's port_1 from +Y, plug start to back stop
        sim.set_camera_view([-0.006, 0.07, PANEL_Z + 0.01], [-0.006, 0.0, PANEL_Z])
    if args.cutaway:
        cutaway(get_current_stage(), VARIANTS[0][0])
    dev = sim.device
    pace_t0, pace_sim = time.perf_counter(), 0.0
    names = rig.joint_names
    ji = {n: names.index(n) for n in CHAIN + ["latch_joint"]}
    probe = Probe(TIERS, OFFSETS, TILTS, args.dt, dev)
    assert probe.variants == VARIANTS

    def f_of(sensor):
        """Normal + friction force of the panel on one plug link, world frame, (N, 3).

        ``force_matrix_w`` holds normal forces only; along the latch most of the axial load is friction."""
        total = 0
        for name in ("force_matrix_w", "friction_forces_w"):
            fm = getattr(sensor.data, name)
            fm = fm.torch if hasattr(fm, "torch") else fm
            total = total + fm.reshape(N, -1, 3).sum(1)
        return total

    rot_w = torch.tensor([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]], device=dev)  # jack frame -> world (FLIP)
    t_wall, steps_done = 0.0, 0
    for seed in range(args.seeds):
        q_init = probe.initial_targets(seed)
        tgt = torch.zeros(N, rig.num_joints, device=dev)
        for n in CHAIN:
            tgt[:, ji[n]] = q_init[n]
        q0 = tgt.clone()
        rig.write_joint_position_to_sim_index(position=q0)
        rig.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q0))
        rig.reset()
        s_body.reset()
        s_latch.reset()
        probe.begin(seed)
        t0 = time.perf_counter()
        for k in range(probe.n_steps):
            tgt[:, ji["x"]] = probe.x_target(k * args.dt)
            rig.actuators.target_command.set_position_index(value=tgt, full_data=True)
            rig.write_data_to_sim()
            sim.step(render=GUI)
            if GUI and args.slowmo > 0:
                pace_sim += args.dt
                lag = args.slowmo * pace_sim - (time.perf_counter() - pace_t0)
                if lag > 0:
                    time.sleep(lag)
            rig.update(args.dt)
            s_body.update(args.dt)
            s_latch.update(args.dt)

            q = rig.data.joint_pos.torch
            f_j = (f_of(s_body) + f_of(s_latch)) @ rot_w  # world -> jack frame (rot_w is symmetric)
            probe.step(k, {n: q[:, ji[n]] for n in CHAIN}, q[:, ji["latch_joint"]],
                       rig.data.joint_vel.torch[:, ji["x"]], f_j)  # fmt: skip
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_wall += time.perf_counter() - t0
        steps_done += probe.n_steps
        n_latched = probe.end(rig.data.joint_pos.torch[:, ji["x"]])
        print(f"[check_contact] seed {seed}: latched {n_latched}/{N}, exploded {int(probe.exploded.sum())}")
    config = {"engine": "physx", "dt": args.dt, "pos_iters": args.pos_iters, "vel_iters": args.vel_iters,
              "contact_offset_m": args.contact_offset, "rest_offset_m": args.rest_offset,
              "friction": args.friction, "factory_base": "isaaclab_tasks/contrib/factory/factory_env_cfg.py"}
    probe.save(OUT, RUN_ID, config, steps_done / max(t_wall, 1e-9), args.seeds,
               {"wall_s_per_sim_s": t_wall / (steps_done * args.dt)})  # fmt: skip
    if GUI and args.hold_open:
        print("[check_contact] done; close the window to exit")
        while simulation_app.is_running():
            sim.render()


def cutaway(stage, tier):
    """Make env 0's panel prims on the +Y side of the latch slot invisible (colliders stay active)."""
    half_slot = jack_geometry(DIMS, tier).slot_width / 2
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/envs/env_0/Panel")):
        if prim.IsA(UsdGeom.Gprim) and cache.ComputeWorldBound(prim).ComputeAlignedRange().GetMin()[1] >= half_slot - 1e-6:
            UsdGeom.Imageable(prim).MakeInvisible()


if __name__ == "__main__":
    import traceback

    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
