"""Annotated videos and collages from ``evaluate.py --save_frames_every k`` grids (4x4 front-camera JPEGs).

Each cell gets its env id, target port and the furthest stage its first episode reached (from <tier>_episodes.csv);
latched cells get a green border and the time to latch. Frames are named by sim step (30 Hz), so with k = 2 the
video plays in real time at 15 fps. Needs Pillow and ffmpeg (no Isaac Sim).

    # one policy -> <run>/<name>_<tier>.mp4 + <name>_<tier>_collage.png
    python scripts/eval_frames_video.py single results/hero_loose16/telemetry_spiral --title "telemetry spiral"
    # two policies side by side (same seeds) -> one mp4 + collage
    python scripts/eval_frames_video.py pair results/hero_loose16/open_loop results/hero_loose16/telemetry_spiral \
        --titles "open-loop" "telemetry spiral" --out results/hero_loose16/open_loop_vs_telemetry_loose
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PORT = {"0": "left", "1": "middle", "2": "right"}
COL = {"latched": (60, 220, 90), "seated": (60, 200, 200), "aligned": (255, 150, 40),
       "contacted": (255, 220, 60), "reached": (200, 200, 200), "none": (130, 130, 130)}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FPS_SIM = 30


def font(size: int):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


class Run:
    """One evaluate.py output dir: frames_<tier>/ grids + <tier>_episodes.csv + <tier>.json."""

    def __init__(self, run_dir: str, tier: str):
        self.dir = Path(run_dir)
        self.frames = {int(Path(p).stem): p for p in glob.glob(str(self.dir / f"frames_{tier}" / "*.jpg"))}
        assert self.frames, f"no frames in {self.dir / f'frames_{tier}'}"
        rows = list(csv.DictReader(open(self.dir / f"{tier}_episodes.csv")))
        self.ep = {int(r["env_id"]): r for r in rows if r["episode_in_env"] == "0"}
        self.summary = json.loads((self.dir / f"{tier}.json").read_text())
        self.steps = sorted(self.frames)
        self.n_latched = sum(r["stage"] == "latched" for r in self.ep.values())

    def frame(self, step: int) -> Image.Image:
        """Annotated grid at ``step`` (the last saved grid once this run has ended)."""
        s = max([t for t in self.steps if t <= step] or [self.steps[0]])
        im = Image.open(self.frames[s]).convert("RGB")
        cols = 4
        cw, ch = im.width // cols, im.height // ((len(self.ep) + cols - 1) // cols)
        d, f = ImageDraw.Draw(im), font(15)
        for e, r in self.ep.items():
            x, y = (e % cols) * cw, (e // cols) * ch
            stage = r["stage"]
            ended = step >= int(r["episode_length"]) - 1
            d.rectangle([x, y, x + cw - 1, y + ch - 1], outline=COL[stage] if stage != "none" else (60, 60, 60),
                        width=3 if stage == "latched" else 2)
            label = f"env {e}  -> {PORT[r['target_port']]} port"
            d.rectangle([x + 3, y + 3, x + 8 + d.textlength(label, font=f), y + 22], fill=(0, 0, 0))
            d.text((x + 6, y + 4), label, fill=(240, 240, 240), font=f)
            if stage == "latched":
                msg = f"LATCHED at {float(r['time_to_latch']):.1f} s" + (" (env then resets)" if ended else "")
                bg, fg = (0, 90, 30), (255, 255, 255)
            elif stage != "none":
                msg, bg, fg = f"best: {stage}", (0, 0, 0), COL[stage]
            else:
                continue
            d.rectangle([x + 3, y + ch - 26, x + 8 + d.textlength(msg, font=f), y + ch - 4], fill=bg)
            d.text((x + 6, y + ch - 25), msg, fill=fg, font=f)
        return im


def compose(runs: list[Run], titles: list[str], step: int, width: int) -> Image.Image:
    """Panels side by side, each with a title bar ``<title>: k/n latched``; sim time at the top right."""
    gap, head = 12, 44
    pw = (width - gap * (len(runs) - 1)) // len(runs)
    panels = []
    for run, title in zip(runs, titles):
        im = run.frame(step)
        panels.append(im.resize((pw, round(im.height * pw / im.width)), Image.LANCZOS))
    out = Image.new("RGB", (width, head + max(p.height for p in panels)), (18, 18, 18))
    d, f = ImageDraw.Draw(out), font(26)
    for i, (p, run, title) in enumerate(zip(panels, runs, titles)):
        x = i * (pw + gap)
        out.paste(p, (x, head))
        d.text((x + 10, 8), f"{title}: {run.n_latched}/{len(run.ep)} latched", fill=(235, 235, 235), font=f)
    t = f"t = {step / FPS_SIM:4.1f} s"
    d.text((width - 12 - d.textlength(t, font=f), 8), t, fill=(255, 210, 60), font=f)
    return out


def write_video(runs: list[Run], titles: list[str], mp4: Path, width: int, fps: int) -> None:
    steps = sorted(set().union(*(r.steps for r in runs)))
    with tempfile.TemporaryDirectory() as tmp:
        for i, s in enumerate(steps):
            im = compose(runs, titles, s, width)
            im = im.crop((0, 0, im.width - im.width % 2, im.height - im.height % 2))  # yuv420p needs even sizes
            im.save(os.path.join(tmp, f"{i:05d}.jpg"), quality=90)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i",
                        os.path.join(tmp, "%05d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                        "-movflags", "+faststart", str(mp4)], check=True)
    print(f"[eval_frames_video] {len(steps)} frames -> {mp4}")


def write_collage(runs: list[Run], titles: list[str], steps: list[int], png: Path, caption: str) -> None:
    """One row per time step (all runs side by side in each row)."""
    rows = [compose(runs, titles, s, 1920) for s in steps]
    lab, top = 0, 50
    out = Image.new("RGB", (1920, top + sum(r.height + lab for r in rows)), (18, 18, 18))
    d = ImageDraw.Draw(out)
    d.text((12, 12), caption, fill=(235, 235, 235), font=font(24))
    y = top
    for s, r in zip(steps, rows):
        out.paste(r, (0, y + lab))
        y += r.height + lab
    out.save(png, optimize=True)
    print(f"[eval_frames_video] collage -> {png}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    one = sub.add_parser("single")
    one.add_argument("run")
    one.add_argument("--title", default=None)
    pair = sub.add_parser("pair")
    pair.add_argument("runs", nargs=2)
    pair.add_argument("--titles", nargs=2, default=None)
    pair.add_argument("--out", required=True, help="output path prefix (.mp4 and _collage.png are appended)")
    for p in (one, pair):
        p.add_argument("--tier", default="loose")
        p.add_argument("--fps", type=int, default=15, help="15 = real time for --save_frames_every 2")
        p.add_argument("--collage_steps", default="165", help="sim steps shown in the collage (one row each)")
        p.add_argument("--caption", default=None)
    args = ap.parse_args()

    dirs = [args.run] if args.mode == "single" else args.runs
    runs = [Run(d, args.tier) for d in dirs]
    titles = ([args.title] if args.mode == "single" else args.titles) or [None] * len(runs)
    titles = [t or r.summary["policy"] for t, r in zip(titles, runs)]
    prefix = Path(args.out) if args.mode == "pair" else runs[0].dir / f"{runs[0].summary['policy']}_{args.tier}"
    width = 1280 if len(runs) == 1 else 1920
    write_video(runs, titles, prefix.with_suffix(".mp4"), width, args.fps)
    caption = args.caption or (f"PatchBench, tier {args.tier}, {len(runs[0].ep)} envs (front camera), "
                               f"port-pose noise {runs[0].summary.get('pose_noise_mm', '?')} mm")
    steps = [int(s) for s in args.collage_steps.split(",") if s]
    write_collage(runs, titles, steps, Path(f"{prefix}_collage.png"), caption)


if __name__ == "__main__":
    main()
