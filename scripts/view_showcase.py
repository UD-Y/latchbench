"""Open the Newton showcase scene (showcase_usd.py output) in the Isaac Sim viewer and loop the animation.

Laptop GUI rule (CLAUDE.md section 2): nothing else Isaac-related running; close the window when done.

    python scripts/view_showcase.py --viz kit
Orbit with Alt + left mouse, zoom with the wheel; Space pauses. ``--seconds N`` closes after N s (headless tests).
"""

import argparse
import time
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--scene", type=Path, default=ROOT / "results/newton_showcase/scene/showcase.usdc")
parser.add_argument("--sim", type=Path, default=ROOT / "results/newton_showcase/sim_loose_port1.npz")
parser.add_argument("--seconds", type=float, default=0.0, help="close after this many seconds (0 = until closed)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux  # noqa: E402


def main():
    if not args.scene.exists():
        raise SystemExit(f"{args.scene} missing: run python scripts/showcase_usd.py first")
    ctx = omni.usd.get_context()
    ctx.open_stage(str(args.scene))
    stage = ctx.get_stage()
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(600.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun")
    sun.CreateIntensityAttr(2500.0)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(35.0, 0.0, 40.0))
    # hero camera (render_showcase.py preset), in the target port's jack frame
    sim = np.load(args.sim, allow_pickle=True)
    jack = np.asarray(sim["panel_w"]) @ np.diag([1.0, -1.0, -1.0, 1.0])
    eye = (jack @ np.r_[-0.10, -0.48, -0.26, 1.0])[:3]
    tgt = (jack @ np.r_[-0.13, 0.0, 0.02, 1.0])[:3]
    cam = UsdGeom.Camera.Define(stage, "/World/ViewCam")
    cam.CreateFocalLengthAttr(28.0)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.001, 20.0))
    look = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, tgt)), Gf.Vec3d(0, 0, 1))
    UsdGeom.Xformable(cam).AddTransformOp().Set(look.GetInverse())
    try:
        from omni.kit.viewport.utility import get_active_viewport

        vp = get_active_viewport()
        if vp is not None:
            vp.camera_path = "/World/ViewCam"
    except ImportError:
        pass
    tl = omni.timeline.get_timeline_interface()
    tl.set_looping(True)
    tl.play()
    print(f"[view_showcase] playing {args.scene.name}: {len(sim['robot_body_q'])} frames at {float(sim['fps']):g} fps "
          "(looping); close the window to exit", flush=True)  # fmt: skip
    t0 = time.time()
    while app.is_running():
        app.update()
        if args.seconds and time.time() - t0 > args.seconds:
            break
    print(f"[view_showcase] timeline at {tl.get_current_time():.2f} s", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
