"""Path-traced frames of the Newton showcase scene (showcase_usd.py) -> PNG sequence (+ mp4 with ffmpeg if present).

Laptop: pipeline check only (CLAUDE.md section 2: <= 320 x 240, a few frames). Brev L40S: the clip, e.g.
    python scripts/render_showcase.py --res 1920x1080 --spp 128 --preset hero --out_dir results/newton_showcase/frames_hero

Camera presets are defined in the jack frame of the target port (x = insertion, z = latch side = down in the world):
``hero`` shows the arm, cable and panel from the front-left; ``macro`` frames the plug entering the port.
"""

import argparse
import shutil
import subprocess
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--scene", type=Path, default=ROOT / "results/newton_showcase/scene/showcase.usdc")
parser.add_argument("--sim", type=Path, default=ROOT / "results/newton_showcase/sim_loose_port1.npz")
parser.add_argument("--preset", choices=["hero", "macro"], default="hero")
parser.add_argument("--frames", default="all", help="'all', 'a:b' or 'a:b:step' or comma list")
parser.add_argument("--res", default="320x240")
parser.add_argument("--spp", type=int, default=32)
parser.add_argument("--out_dir", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux  # noqa: E402

PRESETS = {  # camera / look-at offsets in the jack frame [m] (x insertion, y lateral, z latch side = world down)
    "hero": ((-0.10, -0.48, -0.26), (-0.13, 0.0, 0.02), 28.0),
    "macro": ((-0.050, -0.045, -0.025), (-0.004, 0.0, 0.0), 50.0),
}


def frame_list(n: int) -> list[int]:
    if args.frames == "all":
        return list(range(n))
    if ":" in args.frames:
        return list(range(n))[slice(*[int(v) if v else None for v in args.frames.split(":")])]
    return [int(v) for v in args.frames.split(",")]


def main():
    sim = np.load(args.sim, allow_pickle=True)
    jack = np.asarray(sim["panel_w"]) @ np.diag([1.0, -1.0, -1.0, 1.0])  # port_1 jack frame in the world
    n = len(sim["robot_body_q"])
    ctx = omni.usd.get_context()
    ctx.open_stage(str(args.scene))
    stage = ctx.get_stage()
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(120.0)
    dome.CreateColorAttr(Gf.Vec3f(0.92, 0.95, 1.0))
    mouth = jack[:3, 3]
    for name, off, inten, size in (("Key", (-0.35, 0.30, -0.45), 14000.0, 0.3), ("Rim", (0.25, -0.30, -0.25), 9000.0, 0.15)):
        light = UsdLux.RectLight.Define(stage, f"/World/Lights/{name}")
        light.CreateWidthAttr(size)
        light.CreateHeightAttr(size)
        light.CreateIntensityAttr(inten)
        pos = (jack @ np.r_[off, 1.0])[:3]
        m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*pos), Gf.Vec3d(*mouth), Gf.Vec3d(0, 0, 1)).GetInverse()
        UsdGeom.Xformable(light).AddTransformOp().Set(m)  # RectLight emits along -Z, like a camera looks

    s = carb.settings.get_settings()
    s.set("/rtx/rendermode", "PathTracing")
    s.set("/rtx/pathtracing/spp", args.spp)
    s.set("/rtx/pathtracing/totalSpp", args.spp)
    s.set("/rtx/pathtracing/optixDenoiser/enabled", True)
    cam_off, tgt_off, focal = PRESETS[args.preset]
    cam_pos = (jack @ np.r_[cam_off, 1.0])[:3]
    tgt = (jack @ np.r_[tgt_off, 1.0])[:3]
    w, h = (int(v) for v in args.res.split("x"))
    cam = rep.create.camera(position=tuple(map(float, cam_pos)), look_at=tuple(map(float, tgt)), focal_length=focal,
                            clipping_range=(0.001, 20.0))  # fmt: skip
    rp = rep.create.render_product(cam, (w, h))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach(rp)
    tl = omni.timeline.get_timeline_interface()
    tl.stop()
    fps = float(sim["fps"])
    out = args.out_dir or ROOT / "results/newton_showcase" / f"frames_{args.preset}_{w}x{h}"
    out.mkdir(parents=True, exist_ok=True)
    frames = frame_list(n)
    for f in frames:
        tl.set_current_time(f / fps)
        for _ in range(2):
            app.update()
        rep.orchestrator.step(rt_subframes=args.spp, delta_time=0.0)
        img = np.asarray(rgb.get_data())[..., :3]
        Image.fromarray(img).save(out / f"{f:04d}.png")
        print(f"[render_showcase] frame {f} -> {out / f'{f:04d}.png'} mean {img.mean():.1f}", flush=True)
    if len(frames) > 10 and shutil.which("ffmpeg"):
        mp4 = out.with_suffix(".mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(int(fps)), "-start_number", str(frames[0]),
                        "-i", str(out / "%04d.png"), "-pix_fmt", "yuv420p", "-crf", "18", str(mp4)], check=False)  # fmt: skip
        print(f"[render_showcase] wrote {mp4}")


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
