"""Engine-neutral virtual-wrist insertion probe: schedule, per-step metrics, summary and plots (pure torch).

Shared by ``scripts/check_contact.py`` (PhysX) and ``scripts/check_contact_newton.py`` (Newton), so both engines run
the same insertion and are scored by the same code. An engine provides, each physics step, the wrist joint positions
(jack frame: ``x`` = plug-tip depth, ``y``/``z`` lateral, ``rx``/``ry``/``rz`` about the plug tip), the latch angle
from rest, the x joint velocity, and the panel-on-plug contact force (body + latch, normal + friction) in the jack frame.

Insertion (CLAUDE.md P3): x target moves at 10 mm/s from 15 mm in front of the mouth to seat_depth + 3 mm, holds,
pulls back 3 mm (retention check), holds. Each env is one variant (tier x lateral offset x tilt); seed 0 offsets
along +Z (latch side) and tilts about Y; later seeds draw the in-plane offset direction and tilt axis.
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import numpy as np
import torch

from patchbench.assets.build_assets import RJ45Dims, jack_geometry

DIMS = RJ45Dims()
START = -0.015  # m, plug tip 15 mm in front of the mouth
SPEED = 0.010  # m/s
OVERTRAVEL = 0.003  # m, target goes to seat_depth + 3 mm
K_LIN, D_LIN = 2000.0, 9.0  # N/m, N*s/m (~critical for ~10 g)
K_ROT, D_ROT = 5.0, 0.01  # N*m/rad, N*m*s/rad
CHAIN = ["x", "y", "z", "rx", "ry", "rz"]
TRACE_DT = 0.004  # s
TRACE_HEADER = "tier,offset_mm,tilt_deg,seed,t,target_mm,depth_mm,lat_y_mm,lat_z_mm,latch_deg,axial_N,lateral_N,wrist_N"
WRIST_LIMITS = {"x": (-0.05, 0.05), "y": (-0.005, 0.005), "z": (-0.005, 0.005), "rot_deg": (-30.0, 30.0)}


class Probe:
    """One run: variants x seeds at one physics dt. Call ``begin(seed)``, then ``step(...)`` once per physics step
    for ``n_steps`` steps, then ``end(final_depth)``; finally ``save(...)``."""

    def __init__(self, tiers, offsets_m, tilts_rad, dt: float, device):
        self.tiers, self.offsets, self.tilts = list(tiers), list(offsets_m), list(tilts_rad)
        self.variants = [(t, o, tl) for t in self.tiers for o in self.offsets for tl in self.tilts]
        self.n = len(self.variants)
        self.geom = {t: jack_geometry(DIMS, t) for t in self.tiers}
        self.dt, self.dev = dt, device
        g = [self.geom[t] for t, _, _ in self.variants]
        tens = lambda vals: torch.tensor(vals, dtype=torch.float32, device=device)  # noqa: E731
        self.seat = tens([x.seat_depth for x in g])
        self.snap = tens([x.snap_depth for x in g])
        self.lip = tens([x.lip_start for x in g])
        self.clear = tens([x.clearance for x in g])
        self.off = tens([o for _, o, _ in self.variants])
        self.tilt = tens([tl for _, _, tl in self.variants])
        t_push = (DIMS.seat_depth + OVERTRAVEL - START) / SPEED
        self.phases = (0.3, 0.3 + t_push, 0.6 + t_push, 0.9 + t_push, 1.1 + t_push)  # settle, push, hold, pull, hold
        self.n_steps = int(self.phases[-1] / dt)
        self.trace_every = max(1, round(TRACE_DT / dt))
        hw, hh = DIMS.plug_width / 2, DIMS.plug_body_height / 2
        self.hw, self.hh = hw, hh
        self.corners = tens([[0.0, sy * hw, sz * hh] for sy in (-1, 1) for sz in (-1, 1)])
        self.rows, self.results = [], []

    # ------------------------------------------------------------------ schedule
    def x_target(self, t: float) -> float:
        p = self.phases
        if t < p[0]:
            return START
        if t < p[1]:
            return START + SPEED * (t - p[0])
        if t < p[2]:
            return DIMS.seat_depth + OVERTRAVEL
        return DIMS.seat_depth + OVERTRAVEL - min(0.003, SPEED * (t - p[2]))

    def initial_targets(self, seed: int) -> dict[str, torch.Tensor]:
        """Wrist joint targets (= initial positions) per env for one seed; x is START."""
        n, dev = self.n, self.dev
        g = torch.Generator().manual_seed(seed)
        theta = torch.rand(n, generator=g).to(dev) * 2 * math.pi if seed else torch.zeros(n, device=dev)
        phi = torch.rand(n, generator=g).to(dev) * 2 * math.pi if seed else torch.full((n,), math.pi / 2, device=dev)
        zero = torch.zeros(n, device=dev)
        return {
            "x": torch.full((n,), START, device=dev),
            "y": self.off * torch.cos(theta),
            "z": self.off * torch.sin(theta),
            "rx": zero,
            "ry": self.tilt * torch.cos(phi),  # tilt about an in-plane axis
            "rz": self.tilt * torch.sin(phi),
        }

    # ------------------------------------------------------------------ episode
    def begin(self, seed: int) -> None:
        n, dev = self.n, self.dev
        self.seed = seed
        self.deflect_min = torch.zeros(n, device=dev)
        self.latched = torch.zeros(n, dtype=torch.bool, device=dev)
        self.peak_axial = torch.zeros(n, device=dev)
        self.after_snap_min = torch.full((n,), float("inf"), device=dev)
        self.max_pen = torch.zeros(n, device=dev)
        self.max_depth = torch.full((n,), -1.0, device=dev)
        self.exploded = torch.zeros(n, dtype=torch.bool, device=dev)
        self.depth_end_hold = torch.zeros(n, device=dev)

    def step(self, k: int, q: dict[str, torch.Tensor], latch: torch.Tensor, xd: torch.Tensor, f_j: torch.Tensor):
        """``q``: wrist joint positions by CHAIN name, (N,); ``latch``: latch angle from rest [rad], + = tip up;
        ``xd``: x joint velocity; ``f_j``: panel force on plug body + latch in the jack frame, (N, 3)."""
        t = k * self.dt
        x_t = self.x_target(t)
        depth, ly, lz = q["x"], q["y"], q["z"]
        axial = -f_j[:, 0]
        f_lat = f_j[:, 1:].norm(dim=-1)
        f_wrist = K_LIN * (x_t - depth) - D_LIN * xd
        # penetration proxy: front-face corners (rotated by the wrist angles) beyond the opening, once inside
        rx, ry, rz = q["rx"], q["ry"], q["rz"]
        cx, cy, cz = self.corners[:, 0], self.corners[:, 1], self.corners[:, 2]
        py = ly[:, None] + cy * torch.cos(rz)[:, None] + cx * torch.sin(rz)[:, None] - cz * rx[:, None]
        pz = lz[:, None] + cz * torch.cos(ry)[:, None] - cx * torch.sin(ry)[:, None] + cy * rx[:, None]
        gw, gh = self.hw + self.clear, self.hh + self.clear
        pen = torch.maximum(py.abs() - gw[:, None], pz.abs() - gh[:, None]).amax(1)
        inside = depth > self.lip
        self.max_pen = torch.where(inside, torch.maximum(self.max_pen, pen.clamp(min=0)), self.max_pen)
        self.max_pen = torch.maximum(self.max_pen, (depth - self.seat).clamp(min=0))  # through the back stop
        self.max_depth = torch.maximum(self.max_depth, depth)
        self.deflect_min = torch.minimum(self.deflect_min, latch)
        in_deflect = (depth > self.lip) & (depth < self.snap)
        self.peak_axial = torch.where(in_deflect, torch.maximum(self.peak_axial, axial), self.peak_axial)
        after = (depth >= self.snap) & (depth < self.seat - 1e-4) & (self.deflect_min < math.radians(-3))
        self.after_snap_min = torch.where(after, torch.minimum(self.after_snap_min, axial), self.after_snap_min)
        allq = torch.stack([q[n] for n in CHAIN] + [latch], 1)
        bad = ~torch.isfinite(allq).all(1) | (ly.abs() > 0.0045) | (lz.abs() > 0.0045) | (latch.abs() > math.radians(25))
        self.exploded |= bad
        if t < self.phases[2]:
            self.depth_end_hold = depth.clone()
            self.latched = (
                (self.deflect_min < math.radians(-3)) & (latch > math.radians(-2)) & (depth >= self.seat - 1.14e-3)
            )
        if k % self.trace_every == 0 and self.seed < 2:
            vals = torch.stack([depth, ly, lz, latch, axial, f_lat, f_wrist], 1).cpu().numpy()
            for i in range(self.n):
                tier, o, tl = self.variants[i]
                if self.seed == 0 or (o == 0 and tl == 0):
                    self.rows.append((tier, o * 1e3, math.degrees(tl), self.seed, round(t, 4), x_t * 1e3,
                                      *(vals[i, :3] * 1e3), math.degrees(vals[i, 3]), *vals[i, 4:]))  # fmt: skip

    def end(self, final_depth: torch.Tensor) -> int:
        """Close the episode; returns the number of latched envs."""
        retained = final_depth >= self.seat - 1.14e-3
        for i, (tier, o, tl) in enumerate(self.variants):
            pk = float(self.peak_axial[i])
            aft = float(self.after_snap_min[i])
            self.results.append({
                "tier": tier, "offset_mm": o * 1e3, "tilt_deg": math.degrees(tl), "seed": self.seed,
                "latched": bool(self.latched[i]), "retained": bool(retained[i] & self.latched[i]),
                "max_deflection_deg": math.degrees(float(self.deflect_min[i])), "peak_axial_N": pk,
                "after_snap_min_axial_N": aft if math.isfinite(aft) else None,
                "signature": bool(self.latched[i]) and pk > 0.3 and math.isfinite(aft) and aft < 0.5 * pk,
                "max_penetration_mm": float(self.max_pen[i]) * 1e3, "max_depth_mm": float(self.max_depth[i]) * 1e3,
                "depth_end_hold_mm": float(self.depth_end_hold[i]) * 1e3, "exploded": bool(self.exploded[i]),
                "tunnel": bool(self.max_depth[i] > self.seat[i] + 0.5e-3),
            })  # fmt: skip
        return int(self.latched.sum())

    # ------------------------------------------------------------------ outputs
    def save(self, out: Path, run_id: str, config: dict, steps_per_s: float, seeds: int, extra: dict | None = None):
        out.mkdir(parents=True, exist_ok=True)
        with gzip.open(out / "traces.csv.gz", "wt") as fh:
            fh.write(TRACE_HEADER + "\n")
            for r in self.rows:
                fh.write(",".join(f"{v:.5g}" if isinstance(v, float) else str(v) for v in r) + "\n")
        results = self.results
        per = {}
        for r in results:
            per.setdefault(f"{r['tier']}|{r['offset_mm']:g}|{r['tilt_deg']:g}", []).append(r)
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
        aligned = {t: [r for r in results if r["tier"] == t and r["offset_mm"] == 0 and r["tilt_deg"] == 0]
                   for t in self.tiers}  # fmt: skip
        k_tip = DIMS.latch_k_theta / DIMS.latch_length**2
        defl = [abs(r["max_deflection_deg"]) for rs in aligned.values() for r in rs]
        beam_tip_force = DIMS.latch_k_theta * math.radians(float(np.mean(defl)) if defl else 0.0) / DIMS.latch_length
        peaks = [r["peak_axial_N"] for rs in aligned.values() for r in rs]
        summary = {
            "run_id": run_id,
            "config": config,
            "num_envs": self.n, "seeds": seeds, "sim_steps_per_s": steps_per_s,
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
            **(extra or {}),
            "variants": variants,
            "episodes": results,
        }  # fmt: skip
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        print("[contact_probe] summary", json.dumps({k: summary[k] for k in ("run_id", "sim_steps_per_s", "explosions",
              "tunnels", "max_penetration_mm", "aligned", "beam_estimate")}, indent=1))  # fmt: skip
        self.plots(out, run_id, variants)
        return summary

    def plots(self, out: Path, run_id: str, variants: dict):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows, tiers_ = self.rows, self.tiers
        arr = np.array([r[3:] for r in rows], dtype=float)
        tiers = np.array([r[0] for r in rows])
        fig, ax = plt.subplots(2, len(tiers_), figsize=(5.5 * len(tiers_), 6.5), sharex=True, squeeze=False,
                               constrained_layout=True)  # fmt: skip
        for c, tier in enumerate(tiers_):
            g = self.geom[tier]
            sel = (tiers == tier) & (np.array([r[1] for r in rows]) == 0) & (np.array([r[2] for r in rows]) == 0)
            for seed in sorted(set(arr[sel, 0].astype(int))):
                s = sel & (arr[:, 0] == seed)
                push = s & (arr[:, 1] <= self.phases[1])
                ax[0, c].plot(arr[push, 3], arr[push, 7], lw=1, label=f"seed {seed}")
                ax[1, c].plot(arr[push, 3], arr[push, 6], lw=1)
            for a in ax[:, c]:
                for x, lbl in ((g.lip_start, "lip"), (g.catch_depth, "catch"), (g.snap_depth, "snap"),
                               (g.seat_depth, "seat")):  # fmt: skip
                    a.axvline(x * 1e3, c="0.75", lw=0.8)
                    a.text(x * 1e3, a.get_ylim()[1] if a is ax[1, c] else 0, lbl, fontsize=7, rotation=90, va="top")
            ax[0, c].set_title(f"{tier} (clearance {g.clearance * 1e3:.3f} mm/side)")
            ax[0, c].set_ylabel("axial contact force [N]")
            ax[1, c].set_ylabel("latch angle from rest [deg]")
            ax[1, c].set_xlabel("plug tip depth [mm]")
            ax[0, c].set_xlim(-2, DIMS.seat_depth * 1e3 + 1.5)
            ax[0, c].legend(fontsize=7)
        fig.suptitle(f"Virtual-wrist insertion, aligned plug, {run_id}")
        fig.savefig(out / "force_vs_depth.png", dpi=130)
        plt.close(fig)

        fig, ax = plt.subplots(1, len(tiers_), figsize=(4.5 * len(tiers_), 3.4), squeeze=False, constrained_layout=True)
        for c, tier in enumerate(tiers_):
            m = np.array([[variants[f"{tier}|{o * 1e3:g}|{math.degrees(tl):g}"]["latch_rate"] for tl in self.tilts]
                          for o in self.offsets])  # fmt: skip
            im = ax[0, c].imshow(m, vmin=0, vmax=1, cmap="RdYlGn", origin="lower")
            for (a, b), v in np.ndenumerate(m):
                ax[0, c].text(b, a, f"{v:.1f}", ha="center", va="center", fontsize=8)
            ax[0, c].set_xticks(range(len(self.tilts)), [f"{math.degrees(t):g}" for t in self.tilts])
            ax[0, c].set_yticks(range(len(self.offsets)), [f"{o * 1e3:g}" for o in self.offsets])
            ax[0, c].set_xlabel("tilt [deg]")
            ax[0, c].set_ylabel("lateral offset [mm]")
            ax[0, c].set_title(f"{tier}: latch rate")
        fig.colorbar(im, ax=ax[0, -1], shrink=0.8)
        fig.savefig(out / "jam_map.png", dpi=130)
        plt.close(fig)
