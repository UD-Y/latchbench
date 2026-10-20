"""RTX still of the RJ45 look layers (usd/look/*.usd): panel + plug in front of / inside port_1. No physics.

Laptop: pipeline check only (CLAUDE.md section 2: one camera, <= 320 x 240). Brev L40S: social-media stills,
e.g. ``--res 1920x1080 --spp 256``.

    python scripts/render_look.py --tier medium --depth_mm -6 --res 320x240 --spp 32 --out results/look/check.png
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--tier", default="medium")
parser.add_argument("--depth_mm", type=float, default=-6.0, help="plug tip depth in port_1 (negative = in front)")
parser.add_argument("--res", default="320x240")
parser.add_argument("--spp", type=int, default=32, help="path-tracing samples per pixel")
parser.add_argument("--cam", default="-0.055,-0.045,0.030", help="camera position [m], panel frame")
parser.add_argument("--target", default="0.0,0.0,-0.002", help="look-at point [m], panel frame")
parser.add_argument("--focal_mm", type=float, default=35.0)
parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "look" / "check.png")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux  # noqa: E402

from patchbench.assets.build_assets import USD_DIR  # noqa: E402

LOOK = USD_DIR / "look"


def main():
    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    panel = stage.DefinePrim("/World/Panel")
    panel.GetReferences().AddReference(str(LOOK / f"panel_3port_{args.tier}_look.usd"))
    # plug in port_1's jack frame: same 180 deg flip about X as the port (latch side down), tip at depth along +X
    plug = UsdGeom.Xform.Define(stage, "/World/Plug")
    plug.GetPrim().GetReferences().AddReference(str(LOOK / "plug_look.usd"))
    plug.AddTranslateOp().Set(Gf.Vec3d(args.depth_mm * 1e-3, 0.0, 0.0))
    plug.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(0.0, Gf.Vec3d(1.0, 0.0, 0.0)))
    # studio lighting: soft dome + key rect light + rim light
    dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
    dome.CreateIntensityAttr(200.0)
    dome.CreateColorAttr(Gf.Vec3f(0.92, 0.95, 1.0))
    key = UsdLux.RectLight.Define(stage, "/World/Key")
    key.CreateWidthAttr(0.15)
    key.CreateHeightAttr(0.15)
    key.CreateIntensityAttr(12000.0)
    kx = UsdGeom.Xformable(key)
    kx.AddTranslateOp().Set(Gf.Vec3d(-0.12, -0.10, 0.12))
    kx.AddRotateXYZOp().Set(Gf.Vec3f(45.0, -35.0, 0.0))
    rim = UsdLux.SphereLight.Define(stage, "/World/Rim")
    rim.CreateRadiusAttr(0.02)
    rim.CreateIntensityAttr(12000.0)
    UsdGeom.Xformable(rim).AddTranslateOp().Set(Gf.Vec3d(-0.05, 0.12, 0.06))

    s = carb.settings.get_settings()
    s.set("/rtx/rendermode", "PathTracing")
    s.set("/rtx/pathtracing/spp", args.spp)
    s.set("/rtx/pathtracing/totalSpp", args.spp)
    s.set("/rtx/pathtracing/optixDenoiser/enabled", True)
    w, h = (int(v) for v in args.res.split("x"))
    cam = rep.create.camera(position=tuple(float(v) for v in args.cam.split(",")),
                            look_at=tuple(float(v) for v in args.target.split(",")),
                            focal_length=args.focal_mm, clipping_range=(0.001, 10.0))  # fmt: skip
    rp = rep.create.render_product(cam, (w, h))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach(rp)
    for _ in range(3):
        app.update()
    rep.orchestrator.step(rt_subframes=args.spp)
    img = np.asarray(rgb.get_data())[..., :3]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(args.out)
    print(f"[render_look] wrote {args.out} {img.shape} mean {img.mean():.1f}")


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
