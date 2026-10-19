"""Compare contact-tuning runs and pick the cheapest config that passes gate G1.

G1 (CLAUDE.md section 9): on loose and medium, the aligned plug latches with a rise -> peak -> drop axial force
signature in every seed, with zero explosions or tunnelling and penetration < 0.1 mm.
Cost = wall seconds per simulated second of the 24-env batch (lower is cheaper).

    python scripts/contact_report.py [--root results/contact_tuning]
Writes <root>/REPORT.md.
"""

import argparse
import json
from pathlib import Path

PEN_MAX_MM = 0.1


def load(root: Path) -> list[dict]:
    runs = []
    for p in sorted(root.glob("*/summary.json")):
        s = json.loads(p.read_text())
        if not {"loose", "medium"} <= set(s["aligned"]):
            continue  # single-tier runs (e.g. spec) are reported separately
        c = s["config"]
        s["cost"] = 1.0 / (s["sim_steps_per_s"] * c["dt"])
        al = s["aligned"]
        s["aligned_ok"] = all(al[t]["latch_rate"] == 1.0 and al[t]["signature_rate"] == 1.0 for t in ("loose", "medium"))
        s["aligned_pen"] = max(al[t]["max_penetration_mm"] for t in ("loose", "medium"))
        s["passes"] = s["aligned_ok"] and s["explosions"] == 0 and s["tunnels"] == 0 and s["aligned_pen"] < PEN_MAX_MM
        runs.append(s)
    return runs


def table(runs: list[dict]) -> str:
    head = ("| run | dt | pos it | contact offset mm | wall s / sim s | aligned latch+signature (loose, medium) | "
            "peak axial N (loose, medium) | explosions | tunnels | max pen. aligned / all mm | "
            "latch rate all variants (loose, medium) | G1 |")  # fmt: skip
    lines = [head, "|" + "---|" * 12]
    for s in sorted(runs, key=lambda r: r["cost"]):
        c, al, v = s["config"], s["aligned"], s["variants"]
        rate = {t: sum(x["latch_rate"] for k, x in v.items() if k.startswith(t + "|"))
                / max(1, sum(1 for k in v if k.startswith(t + "|"))) for t in ("loose", "medium")}  # fmt: skip
        pk = {t: sum(al[t]["peak_axial_N"]) / max(1, len(al[t]["peak_axial_N"])) for t in ("loose", "medium")}
        lines.append(
            f"| {s['run_id']} | 1/{round(1 / c['dt'])} | {c['pos_iters']} | {c['contact_offset_m'] * 1e3:g} | "
            f"{s['cost']:.2f} | {al['loose']['signature_rate']:.1f}, {al['medium']['signature_rate']:.1f} | "
            f"{pk['loose']:.2f}, {pk['medium']:.2f} | {s['explosions']} | {s['tunnels']} | "
            f"{s['aligned_pen']:.3f} / {s['max_penetration_mm']:.3f} | {rate['loose']:.2f}, {rate['medium']:.2f} | "
            f"{'PASS' if s['passes'] else 'fail'} |"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "contact_tuning")
    args = ap.parse_args()
    runs = [r for r in load(args.root) if r["run_id"] != "smoke"]
    chosen = "c480_it32_co0.5"  # patchbench/physics.py
    passing = sorted((r for r in runs if r["passes"]), key=lambda r: r["cost"])
    text = ["# Contact tuning report", "", table(runs), ""]
    if passing:
        best = passing[0]
        text += [f"Cheapest passing config: **{best['run_id']}** ({best['cost']:.2f} wall s per sim s, 24 envs).",
                 f"Beam check: peak axial / beam tip force = {best['beam_estimate']['peak_axial_over_beam_tip_force']:.2f}."]  # fmt: skip
    else:
        text += ["No config passes G1."]
    for p in sorted(args.root.glob("spec_*/summary.json")):
        s = json.loads(p.read_text())
        a = s["aligned"]["spec"]
        text += ["", f"Spec tier, {s['run_id']}: aligned latch {a['latch_rate']:.1f}, signature {a['signature_rate']:.1f}, "
                 f"peak axial {a['peak_axial_N'][0]:.2f} N, explosions {s['explosions']}, tunnels {s['tunnels']}, "
                 f"max penetration all variants {s['max_penetration_mm']:.3f} mm."]
    text += ["", f"Chosen for patchbench/physics.py: **{chosen}** (32 iterations: jam penetration 0.12 mm vs 0.73 mm "
             "at 16 under a 32 N load; see the physics.py docstring)."]
    out = "\n".join(text) + "\n"
    (args.root / "REPORT.md").write_text(out)
    print(out)
    print(json.dumps({"best": passing[0]["run_id"] if passing else None}))


if __name__ == "__main__":
    main()
