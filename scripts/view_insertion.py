"""Headless visual check of the RJ45 assets: push the plug into port_1 of a panel and watch the latch.

One env, PhysX, ONE camera at 240x320 (laptop limit, CLAUDE.md section 2) looking along the panel face from the
side, with the panel geometry between the camera and port_1's latch slot hidden (visual cut-away; colliders stay
active). The plug rides a prismatic joint (fixed rail -> slide -> plug body -> latch_joint, one fixed-base
articulation) whose drive target moves at 10 mm/s from 20 mm in front of the mouth to seat_depth + 1 mm, holds,
then pulls back 3 mm to test retention.

Physics is NOT tuned here (Block 2 does that); this only reports what the untuned assets do.

Run (Arena venv, nothing else Isaac-related running):
    python scripts/view_insertion.py [--tier medium] [--out results/assets_v0]
Writes <out>/{before,deflection,snapped,pulled}.png, latch_log.csv, latch_angle.png and summary.json.
"""

import argparse
import json
import math
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--tier", default="medium", choices=["loose", "medium", "spec"])
parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "assets_v0")
parser.add_argument("--dt", type=float, default=1.0 / 120.0, help="physics dt (Arena default 1/120)")
parser.add_argument("--speed", type=float, default=0.010, help="insertion speed, m/s")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(enable_cameras=True)  # headless unless --viz kit
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sensors.camera import Camera, CameraCfg  # noqa: E402
from isaaclab.sim.utils.stage import get_current_stage  # noqa: E402

from latchbench.assets.build_assets import USD_DIR, RJ45Dims, jack_geometry  # noqa: E402

DIMS = RJ45Dims()
GEOM = jack_geometry(DIMS, args_cli.tier)
PANEL_POS = (0.0, 0.0, 0.10)  # panel origin = port_1 mouth, 10 cm above the ground
FLIP_WXYZ = Gf.Quatd(0.0, Gf.Vec3d(1.0, 0.0, 0.0))  # port_1 mouth orientation (latch side down)
START_GAP = 0.020  # plug tip 20 mm in front of the mouth
SLIDE_K, SLIDE_D = 2000.0, 50.0  # N/m, N*s/m: overdamped carriage, lag ~ D*v/K = 0.25 mm at 10 mm/s


def build_scene(stage: Usd.Stage) -> None:
    tier_dir = USD_DIR / args_cli.tier
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))
    light.func("/World/light", light)

    panel = UsdGeom.Xform.Define(stage, "/World/Panel")
    panel.GetPrim().GetReferences().AddReference(str(tier_dir / f"panel_3port_{args_cli.tier}.usd"))
    panel.AddTranslateOp().Set(Gf.Vec3d(*PANEL_POS))

    # Rig frame = port_1 mouth frame. Slide joint position = insertion depth of the plug tip.
    rig = UsdGeom.Xform.Define(stage, "/World/Rig")
    rig.AddTranslateOp().Set(Gf.Vec3d(*PANEL_POS))
    rig.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(FLIP_WXYZ)
    UsdPhysics.ArticulationRootAPI.Apply(rig.GetPrim())
    rail = UsdGeom.Xform.Define(stage, "/World/Rig/rail").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(rail)
    UsdPhysics.MassAPI.Apply(rail).CreateMassAttr(0.1)
    fixed = UsdPhysics.FixedJoint.Define(stage, "/World/Rig/rail_fixed")
    fixed.CreateBody1Rel().SetTargets([rail.GetPath()])
    fixed.CreateLocalPos0Attr(Gf.Vec3f(*PANEL_POS))
    fixed.CreateLocalRot0Attr(Gf.Quatf(FLIP_WXYZ))

    plug = UsdGeom.Xform.Define(stage, "/World/Rig/Plug")
    plug.GetPrim().GetReferences().AddReference(str(tier_dir / "plug.usd"))
    plug.AddTranslateOp().Set(Gf.Vec3d(-START_GAP, 0.0, 0.0))

    slide = UsdPhysics.PrismaticJoint.Define(stage, "/World/Rig/slide")
    slide.CreateBody0Rel().SetTargets([rail.GetPath()])
    slide.CreateBody1Rel().SetTargets(["/World/Rig/Plug/body"])
    slide.CreateAxisAttr("X")
    slide.CreateLowerLimitAttr(-0.05)
    slide.CreateUpperLimitAttr(0.05)
    UsdPhysics.DriveAPI.Apply(slide.GetPrim(), "linear")

    # Cut-away: hide panel geometry entirely on the camera side (+Y world) of port_1's latch slot.
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    hidden = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Panel")):
        if prim.IsA(UsdGeom.Gprim):
            rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if rng.GetMin()[1] >= GEOM.slot_width / 2 - 1e-6:
                UsdGeom.Imageable(prim).MakeInvisible()
                hidden += 1
    print(f"[view_insertion] cut-away: hid {hidden} panel prims on the camera side")


def main() -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=args_cli.dt, device=args_cli.device))
    stage = get_current_stage()
    build_scene(stage)

    rig = Articulation(
        ArticulationCfg(
            prim_path="/World/Rig",
            spawn=None,
            init_state=ArticulationCfg.InitialStateCfg(
                pos=PANEL_POS, rot=(1.0, 0.0, 0.0, 0.0), joint_pos={"slide": -START_GAP, "latch_joint": 0.0}
            ),
            actuators={
                "slide": ImplicitActuatorCfg(joint_names_expr=["slide"], stiffness=SLIDE_K, damping=SLIDE_D),
                # None = keep the stiffness/damping authored in plug.usd (k_theta from section 5)
                "latch": ImplicitActuatorCfg(joint_names_expr=["latch_joint"], stiffness=None, damping=None),
            },
        )
    )
    cam = Camera(
        CameraCfg(
            prim_path="/World/cam",
            update_period=0,
            height=240,
            width=320,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.005, 10.0)),
        )
    )

    sim.reset()
    names = rig.joint_names
    i_slide, i_latch = names.index("slide"), names.index("latch_joint")
    print(f"[view_insertion] joints {names}; fixed base: {rig.is_fixed_base}")
    q0 = rig.data.joint_pos.torch.clone()
    q0[:, i_slide], q0[:, i_latch] = -START_GAP, 0.0
    rig.write_joint_position_to_sim_index(position=q0)
    rig.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q0))
    rig.reset()
    # Side view along -Y, centred between the plug start and the back stop, at port height.
    x_mid = -0.006  # frame spans ~[-30, +18] mm: plug tip at start .. back stop
    eye = torch.tensor([[x_mid, 0.055, PANEL_POS[2]]], device=sim.device)
    tgt = torch.tensor([[x_mid, 0.0, PANEL_POS[2]]], device=sim.device)
    cam.set_world_poses_from_view(eye, tgt)

    dt = sim.get_physics_dt()
    t_push = (START_GAP + DIMS.seat_depth + 1e-3) / args_cli.speed
    t_hold, t_pull = 1.0, 1.5
    n_steps = int((0.5 + t_push + t_hold + t_pull) / dt)
    log, frames = [], {}
    target = -START_GAP
    t0 = time.time()
    for k in range(n_steps):
        t = k * dt
        if t < 0.5:
            target = -START_GAP
        elif t < 0.5 + t_push:
            target = -START_GAP + args_cli.speed * (t - 0.5)
        elif t < 0.5 + t_push + t_hold:
            target = DIMS.seat_depth + 1e-3
        else:  # pull back 3 mm over 0.3 s, then hold: tests whether the catch retains the plug
            target = DIMS.seat_depth + 1e-3 - min(3e-3, 0.01 * (t - 0.5 - t_push - t_hold))
        pos_t = rig.data.joint_pos.torch.clone()
        pos_t[:, i_slide] = target
        pos_t[:, i_latch] = 0.0
        rig.actuators.target_command.set_position_index(value=pos_t, full_data=True)
        rig.write_data_to_sim()
        sim.step()
        rig.update(dt)
        cam.update(dt)
        q = rig.data.joint_pos.torch[0].cpu()
        log.append((t, target, float(q[i_slide]), math.degrees(float(q[i_latch]))))
        if k % 2 == 0:
            rgb = cam.data.output["rgb"]
            rgb = rgb.torch if hasattr(rgb, "torch") else rgb
            frames[k] = rgb[0, ..., :3].cpu().numpy().astype(np.uint8)
    wall = time.time() - t0
    print(f"[view_insertion] {n_steps} steps in {wall:.1f} s")

    save(np.array(log), frames, dt, t_push, t_hold)


def save(log: np.ndarray, frames: dict, dt: float, t_push: float, t_hold: float) -> None:
    from PIL import Image

    out = args_cli.out
    out.mkdir(parents=True, exist_ok=True)
    t, target, depth, ang = log.T
    np.savetxt(
        out / "latch_log.csv",
        log,
        delimiter=",",
        header="t_s,target_depth_m,depth_m,latch_deg_from_rest",
        comments="",
        fmt="%.6f",
    )
    end_push = int((0.5 + t_push) / dt)
    end_hold = int((0.5 + t_push + t_hold) / dt)
    k_min = int(np.argmin(ang[:end_hold]))
    deflected = ang[k_min] < -3.0
    after = np.nonzero(np.abs(ang[k_min:end_hold]) < 2.0)[0]
    k_snap = k_min + int(after[0]) if deflected and after.size else None
    nearest = lambda k: min(frames, key=lambda f: abs(f - k))  # noqa: E731
    picks = {"before": int(0.4 / dt), "deflection": k_min, "snapped": end_hold - 1, "pulled": len(log) - 1}
    for name, k in picks.items():
        Image.fromarray(frames[nearest(k)]).save(out / f"{name}.png")

    pulled_depth = float(depth[-1])
    summary = {
        "tier": args_cli.tier,
        "dt": dt,
        "speed_m_s": args_cli.speed,
        "max_deflection_deg": float(ang[k_min]),
        "max_deflection_depth_mm": float(depth[k_min] * 1e3),
        "deflected_gt_3deg": bool(deflected),
        "snapped_back_within_2deg": k_snap is not None,
        "snap_depth_mm": float(depth[k_snap] * 1e3) if k_snap is not None else None,
        "expected_snap_depth_mm": GEOM.snap_depth * 1e3,
        "latch_deg_end_of_hold": float(ang[end_hold - 1]),
        "depth_end_of_push_mm": float(depth[end_push - 1] * 1e3),
        "depth_end_of_hold_mm": float(depth[end_hold - 1] * 1e3),
        "max_depth_mm": float(depth.max() * 1e3),
        "seat_depth_mm": GEOM.seat_depth * 1e3,
        "depth_after_pull_mm": pulled_depth * 1e3,
        "pull_target_depth_mm": float(target[-1] * 1e3),
        "retained_after_pull": bool(pulled_depth >= GEOM.seat_depth - 1.14e-3),
        "nan": bool(np.isnan(log).any()),
        "frames": {k: int(nearest(v)) for k, v in picks.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
        ax[0].plot(t, depth * 1e3, label="plug tip depth")
        ax[0].plot(t, target * 1e3, "--", label="drive target")
        for y, lbl in ((GEOM.seat_depth, "seat"), (GEOM.snap_depth, "expected snap"), (GEOM.catch_depth, "catch")):
            ax[0].axhline(y * 1e3, color="0.7", lw=0.8)
            ax[0].text(t[0], y * 1e3, lbl, fontsize=7, va="bottom")
        ax[0].set_ylabel("depth [mm]")
        ax[0].legend(fontsize=8)
        ax[1].plot(t, ang)
        ax[1].axhline(-3, color="0.7", lw=0.8)
        ax[1].axhline(-2, color="0.85", lw=0.8)
        ax[1].set_ylabel("latch angle from rest [deg]")
        ax[1].set_xlabel("t [s]")
        fig.suptitle(f"view_insertion, tier {args_cli.tier}, untuned PhysX, dt {dt:.4f} s")
        fig.tight_layout()
        fig.savefig(out / "latch_angle.png", dpi=120)
    except ImportError:
        print("[view_insertion] matplotlib missing; skipped latch_angle.png")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
