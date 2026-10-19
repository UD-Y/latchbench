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
parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "contact_tuning")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
simulation_app = AppLauncher(args).app

import gzip  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sensors import ContactSensor, ContactSensorCfg  # noqa: E402
from isaaclab.sim.utils.stage import get_current_stage  # noqa: E402
from isaaclab_physx.physics import PhysxCfg  # noqa: E402
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

from patchbench.assets.build_assets import USD_DIR, RJ45Dims, jack_geometry  # noqa: E402

DIMS = RJ45Dims()
TIERS = args.tiers.split(",")
OFFSETS = [float(x) * 1e-3 for x in args.offsets_mm.split(",")]
TILTS = [math.radians(float(x)) for x in args.tilts_deg.split(",")]
VARIANTS = [(tier, o, tl) for tier in TIERS for o in OFFSETS for tl in TILTS]
N = len(VARIANTS)
GEOM = {tier: jack_geometry(DIMS, tier) for tier in TIERS}
PANEL_Z = 0.10
FLIP = Gf.Quatd(0.0, Gf.Vec3d(1.0, 0.0, 0.0))  # port_1 jack frame in the panel frame (latch side down)
START = -0.015  # plug tip 15 mm in front of the mouth
SPEED = 0.010  # m/s
OVERTRAVEL = 0.003  # target goes to seat_depth + 3 mm
K_LIN, D_LIN = 2000.0, 9.0  # N/m, N*s/m (~critical for ~10 g)
K_ROT, D_ROT = 5.0, 0.01  # N*m/rad, N*m*s/rad
CHAIN = ["x", "y", "z", "rx", "ry", "rz"]
TRACE_DT = 0.004
RUN_ID = args.run_id or f"dt{round(1 / args.dt)}_it{args.pos_iters}_co{args.contact_offset * 1e3:g}"
OUT = args.out / RUN_ID


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
    dev = sim.device
    names = rig.joint_names
    ji = {n: names.index(n) for n in CHAIN + ["latch_joint"]}
    seat = torch.tensor([GEOM[t].seat_depth for t, _, _ in VARIANTS], device=dev)
    snap = torch.tensor([GEOM[t].snap_depth for t, _, _ in VARIANTS], device=dev)
    lip = torch.tensor([GEOM[t].lip_start for t, _, _ in VARIANTS], device=dev)
    clear = torch.tensor([GEOM[t].clearance for t, _, _ in VARIANTS], device=dev)
    off = torch.tensor([o for _, o, _ in VARIANTS], device=dev)
    tilt = torch.tensor([tl for _, _, tl in VARIANTS], device=dev)
    dt = args.dt
    t_push = (DIMS.seat_depth + OVERTRAVEL - START) / SPEED
    phases = (0.3, 0.3 + t_push, 0.6 + t_push, 0.9 + t_push, 1.1 + t_push)  # settle, push, hold, pull, hold
    n_steps = int(phases[-1] / dt)
    trace_every = max(1, round(TRACE_DT / dt))
    hw, hh = DIMS.plug_width / 2, DIMS.plug_body_height / 2
    corners = torch.tensor([[0.0, sy * hw, sz * hh] for sy in (-1, 1) for sz in (-1, 1)], device=dev)

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
    rows, results = [], []
    t_wall, steps_done = 0.0, 0
    for seed in range(args.seeds):
        g = torch.Generator().manual_seed(seed)
        theta = torch.rand(N, generator=g).to(dev) * 2 * math.pi if seed else torch.zeros(N, device=dev)
        phi = torch.rand(N, generator=g).to(dev) * 2 * math.pi if seed else torch.full((N,), math.pi / 2, device=dev)
        tgt = torch.zeros(N, rig.num_joints, device=dev)
        tgt[:, ji["x"]] = START
        tgt[:, ji["y"]] = off * torch.cos(theta)
        tgt[:, ji["z"]] = off * torch.sin(theta)
        tgt[:, ji["ry"]] = tilt * torch.cos(phi)  # tilt about an in-plane axis
        tgt[:, ji["rz"]] = tilt * torch.sin(phi)
        q0 = tgt.clone()
        rig.write_joint_position_to_sim_index(position=q0)
        rig.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q0))
        rig.reset()
        s_body.reset()
        s_latch.reset()
        # per-env episode stats
        deflect_min = torch.zeros(N, device=dev)
        latched = torch.zeros(N, dtype=torch.bool, device=dev)
        peak_axial = torch.zeros(N, device=dev)
        after_snap_min = torch.full((N,), float("inf"), device=dev)
        max_pen = torch.zeros(N, device=dev)
        max_depth = torch.full((N,), -1.0, device=dev)
        exploded = torch.zeros(N, dtype=torch.bool, device=dev)
        depth_end_hold = torch.zeros(N, device=dev)
        t0 = time.perf_counter()
        for k in range(n_steps):
            t = k * dt
            if t < phases[0]:
                x_t = START
            elif t < phases[1]:
                x_t = START + SPEED * (t - phases[0])
            elif t < phases[2]:
                x_t = DIMS.seat_depth + OVERTRAVEL
            else:
                x_t = DIMS.seat_depth + OVERTRAVEL - min(0.003, SPEED * (t - phases[2]))
            tgt[:, ji["x"]] = x_t
            rig.actuators.target_command.set_position_index(value=tgt, full_data=True)
            rig.write_data_to_sim()
            sim.step(render=False)
            rig.update(dt)
            s_body.update(dt)
            s_latch.update(dt)

            q = rig.data.joint_pos.torch
            depth, ly, lz = q[:, ji["x"]], q[:, ji["y"]], q[:, ji["z"]]
            latch = q[:, ji["latch_joint"]]
            f_w = f_of(s_body) + f_of(s_latch)
            f_j = f_w @ rot_w  # world -> jack frame (rot_w is symmetric)
            axial = -f_j[:, 0]
            f_lat = f_j[:, 1:].norm(dim=-1)
            f_wrist = K_LIN * (x_t - depth) - D_LIN * rig.data.joint_vel.torch[:, ji["x"]]
            # penetration proxy: front-face corners (rotated by the wrist angles) beyond the opening, once inside
            rx, ry, rz = q[:, ji["rx"]], q[:, ji["ry"]], q[:, ji["rz"]]
            cx, cy, cz = corners[:, 0], corners[:, 1], corners[:, 2]
            py = ly[:, None] + cy * torch.cos(rz)[:, None] + cx * torch.sin(rz)[:, None] - cz * rx[:, None]
            pz = lz[:, None] + cz * torch.cos(ry)[:, None] - cx * torch.sin(ry)[:, None] + cy * rx[:, None]
            gw, gh = hw + clear, hh + clear
            pen = torch.maximum(py.abs() - gw[:, None], pz.abs() - gh[:, None]).amax(1)
            inside = depth > lip
            max_pen = torch.where(inside, torch.maximum(max_pen, pen.clamp(min=0)), max_pen)
            max_pen = torch.maximum(max_pen, (depth - seat).clamp(min=0))  # through the back stop
            max_depth = torch.maximum(max_depth, depth)
            deflect_min = torch.minimum(deflect_min, latch)
            in_deflect = (depth > lip) & (depth < snap)
            peak_axial = torch.where(in_deflect, torch.maximum(peak_axial, axial), peak_axial)
            after = (depth >= snap) & (depth < seat - 1e-4) & (deflect_min < math.radians(-3))
            after_snap_min = torch.where(after, torch.minimum(after_snap_min, axial), after_snap_min)
            bad = ~torch.isfinite(q).all(1) | (ly.abs() > 0.0045) | (lz.abs() > 0.0045) | (latch.abs() > math.radians(25))
            exploded |= bad
            if t < phases[2]:
                depth_end_hold = depth.clone()
                latched = (deflect_min < math.radians(-3)) & (latch > math.radians(-2)) & (depth >= seat - 1.14e-3)
            if k % trace_every == 0 and seed < 2:
                vals = torch.stack([depth, ly, lz, latch, axial, f_lat, f_wrist], 1).cpu().numpy()
                for i in range(N):
                    tier, o, tl = VARIANTS[i]
                    if seed == 0 or (o == 0 and tl == 0):
                        rows.append((tier, o * 1e3, math.degrees(tl), seed, round(t, 4), x_t * 1e3,
                                     *(vals[i, :3] * 1e3), math.degrees(vals[i, 3]), *vals[i, 4:]))  # fmt: skip
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_wall += time.perf_counter() - t0
        steps_done += n_steps
        retained = rig.data.joint_pos.torch[:, ji["x"]] >= seat - 1.14e-3
        for i, (tier, o, tl) in enumerate(VARIANTS):
            pk = float(peak_axial[i])
            aft = float(after_snap_min[i])
            results.append({
                "tier": tier, "offset_mm": o * 1e3, "tilt_deg": math.degrees(tl), "seed": seed,
                "latched": bool(latched[i]), "retained": bool(retained[i] & latched[i]),
                "max_deflection_deg": math.degrees(float(deflect_min[i])), "peak_axial_N": pk,
                "after_snap_min_axial_N": aft if math.isfinite(aft) else None,
                "signature": bool(latched[i]) and pk > 0.3 and math.isfinite(aft) and aft < 0.5 * pk,
                "max_penetration_mm": float(max_pen[i]) * 1e3, "max_depth_mm": float(max_depth[i]) * 1e3,
                "depth_end_hold_mm": float(depth_end_hold[i]) * 1e3, "exploded": bool(exploded[i]),
                "tunnel": bool(max_depth[i] > seat[i] + 0.5e-3),
            })  # fmt: skip
        print(f"[check_contact] seed {seed}: latched {int(latched.sum())}/{N}, exploded {int(exploded.sum())}")
    save(rows, results, steps_done / max(t_wall, 1e-9))


def save(rows, results, steps_per_s):
    OUT.mkdir(parents=True, exist_ok=True)
    header = "tier,offset_mm,tilt_deg,seed,t,target_mm,depth_mm,lat_y_mm,lat_z_mm,latch_deg,axial_N,lateral_N,wrist_N"
    with gzip.open(OUT / "traces.csv.gz", "wt") as fh:
        fh.write(header + "\n")
        for r in rows:
            fh.write(",".join(f"{v:.5g}" if isinstance(v, float) else str(v) for v in r) + "\n")
    per = {}
    for r in results:
        key = f"{r['tier']}|{r['offset_mm']:g}|{r['tilt_deg']:g}"
        per.setdefault(key, []).append(r)
    variants = {}
    for key, rs in per.items():
        variants[key] = {
            "latch_rate": sum(r["latched"] for r in rs) / len(rs),
            "retained_rate": sum(r["retained"] for r in rs) / len(rs),
            "signature_rate": sum(r["signature"] for r in rs) / len(rs),
            "peak_axial_N_mean": float(np.mean([r["peak_axial_N"] for r in rs])),
            "max_penetration_mm": max(r["max_penetration_mm"] for r in rs),
            "exploded": sum(r["exploded"] for r in rs),
            "tunnel": sum(r["tunnel"] for r in rs),
        }
    aligned = {t: [r for r in results if r["tier"] == t and r["offset_mm"] == 0 and r["tilt_deg"] == 0] for t in TIERS}
    k_tip = DIMS.latch_k_theta / DIMS.latch_length**2
    defl = [abs(r["max_deflection_deg"]) for rs in aligned.values() for r in rs]
    beam_tip_force = DIMS.latch_k_theta * math.radians(float(np.mean(defl)) if defl else 0.0) / DIMS.latch_length
    peaks = [r["peak_axial_N"] for rs in aligned.values() for r in rs]
    summary = {
        "run_id": RUN_ID,
        "config": {"dt": args.dt, "pos_iters": args.pos_iters, "vel_iters": args.vel_iters,
                   "contact_offset_m": args.contact_offset, "rest_offset_m": args.rest_offset,
                   "friction": args.friction, "factory_base": "isaaclab_tasks/contrib/factory/factory_env_cfg.py"},
        "num_envs": N, "seeds": args.seeds, "sim_steps_per_s": steps_per_s,
        "explosions": sum(r["exploded"] for r in results), "tunnels": sum(r["tunnel"] for r in results),
        "max_penetration_mm": max(r["max_penetration_mm"] for r in results),
        "aligned": {t: {"latch_rate": sum(r["latched"] for r in rs) / max(1, len(rs)),
                        "signature_rate": sum(r["signature"] for r in rs) / max(1, len(rs)),
                        "retained_rate": sum(r["retained"] for r in rs) / max(1, len(rs)),
                        "peak_axial_N": [round(r["peak_axial_N"], 3) for r in rs],
                        "max_penetration_mm": max([r["max_penetration_mm"] for r in rs], default=None)}
                    for t, rs in aligned.items()},
        "beam_estimate": {"k_tip_N_per_m": k_tip, "mean_max_deflection_deg": float(np.mean(defl)) if defl else None,
                          "beam_tip_force_N": beam_tip_force,
                          "peak_axial_over_beam_tip_force": float(np.mean(peaks)) / beam_tip_force
                          if beam_tip_force > 0 else None},
        "variants": variants,
        "episodes": results,
    }  # fmt: skip
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print("[check_contact] summary", json.dumps({k: summary[k] for k in ("run_id", "sim_steps_per_s", "explosions",
          "tunnels", "max_penetration_mm", "aligned", "beam_estimate")}, indent=1))  # fmt: skip
    plots(rows, variants)


def plots(rows, variants):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.array([r[3:] for r in rows], dtype=float)
    tiers = np.array([r[0] for r in rows])
    fig, ax = plt.subplots(2, len(TIERS), figsize=(5.5 * len(TIERS), 6.5), sharex=True, squeeze=False,
                           constrained_layout=True)  # fmt: skip
    for c, tier in enumerate(TIERS):
        g = GEOM[tier]
        sel = (tiers == tier) & (np.array([r[1] for r in rows]) == 0) & (np.array([r[2] for r in rows]) == 0)
        for seed in sorted(set(arr[sel, 0].astype(int))):
            s = sel & (arr[:, 0] == seed)
            push = s & (arr[:, 1] <= 0.3 + (DIMS.seat_depth + OVERTRAVEL - START) / SPEED)
            ax[0, c].plot(arr[push, 3], arr[push, 7], lw=1, label=f"seed {seed}")
            ax[1, c].plot(arr[push, 3], arr[push, 6], lw=1)
        for a in ax[:, c]:
            for x, lbl in ((g.lip_start, "lip"), (g.catch_depth, "catch"), (g.snap_depth, "snap"), (g.seat_depth, "seat")):
                a.axvline(x * 1e3, c="0.75", lw=0.8)
                a.text(x * 1e3, a.get_ylim()[1] if a is ax[1, c] else 0, lbl, fontsize=7, rotation=90, va="top")
        ax[0, c].set_title(f"{tier} (clearance {g.clearance * 1e3:.3f} mm/side)")
        ax[0, c].set_ylabel("axial contact force [N]")
        ax[1, c].set_ylabel("latch angle from rest [deg]")
        ax[1, c].set_xlabel("plug tip depth [mm]")
        ax[0, c].set_xlim(-2, DIMS.seat_depth * 1e3 + 1.5)
        ax[0, c].legend(fontsize=7)
    fig.suptitle(f"Virtual-wrist insertion, aligned plug, {RUN_ID}")
    fig.savefig(OUT / "force_vs_depth.png", dpi=130)

    fig, ax = plt.subplots(1, len(TIERS), figsize=(4.5 * len(TIERS), 3.4), squeeze=False, constrained_layout=True)
    for c, tier in enumerate(TIERS):
        m = np.array([[variants[f"{tier}|{o * 1e3:g}|{math.degrees(tl):g}"]["latch_rate"] for tl in TILTS]
                      for o in OFFSETS])  # fmt: skip
        im = ax[0, c].imshow(m, vmin=0, vmax=1, cmap="RdYlGn", origin="lower")
        for (a, b), v in np.ndenumerate(m):
            ax[0, c].text(b, a, f"{v:.1f}", ha="center", va="center", fontsize=8)
        ax[0, c].set_xticks(range(len(TILTS)), [f"{math.degrees(t):g}" for t in TILTS])
        ax[0, c].set_yticks(range(len(OFFSETS)), [f"{o * 1e3:g}" for o in OFFSETS])
        ax[0, c].set_xlabel("tilt [deg]")
        ax[0, c].set_ylabel("lateral offset [mm]")
        ax[0, c].set_title(f"{tier}: latch rate")
    fig.colorbar(im, ax=ax[0, -1], shrink=0.8)
    fig.savefig(OUT / "jam_map.png", dpi=130)


if __name__ == "__main__":
    import traceback

    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
