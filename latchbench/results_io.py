"""Evaluation results: per-episode records -> ``results/<policy>/<tier>.json`` (CLAUDE.md section 6.2) + CSV, CIs, plots.

No Isaac imports. Episode records are flat dicts, one per finished episode, as written by the env's episode recorder
term (``StageTracker.summary()`` for that env plus ``tier``/``target_port``) and Arena's core term (``env_id``,
``episode_in_env``, ``episode_length``).
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Iterable

import torch

from latchbench.predicates import STAGES, funnel

TIERS = ("loose", "medium", "spec")
# Policies in leaderboard / plot order; anything else is appended alphabetically.
POLICY_ORDER = ("open_loop", "telemetry_spiral", "oracle_spiral", "expert")
Z95 = 1.959963984540054  # two-sided 95 % normal quantile

CSV_FIELDS = (
    "episode",
    "env_id",
    "episode_in_env",
    "tier",
    "target_port",
    "furthest_stage",
    "stage",
    "success",
    "wrong_port",
    "peak_axial_force",
    "time_to_latch",
    "overload_events",
    "episode_length",
)


def wilson_ci(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for ``k`` successes out of ``n`` (``(0, 1)`` when ``n == 0``)."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _num(x: Any) -> float:
    return float("nan") if x is None else float(x)


def _mean(xs: Iterable[float]) -> float | None:
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else None


def git_sha(repo: str | Path | None = None) -> str:
    """Short git SHA of ``repo`` (default: the latchbench checkout), ``+dirty`` if tracked files are modified."""
    repo = Path(repo) if repo is not None else Path(__file__).resolve().parents[1]
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True
        ).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def servosense_version() -> str:
    try:
        from importlib.metadata import version

        return version("servosense")
    except Exception:
        try:
            import servosense

            return getattr(servosense, "__version__", "unknown")
        except ImportError:
            return "unknown"


def select_episodes(records: list[dict[str, Any]], n: int, num_envs: int) -> list[dict[str, Any]]:
    """Keep exactly the episodes of the fixed schedule: global episode ``i < n`` runs in env ``i % num_envs`` as that
    env's ``i // num_envs``-th episode.

    Selecting by schedule slot instead of "first n to finish" avoids biasing towards short (early-terminating)
    episodes. Records without ``env_id``/``episode_in_env`` are taken in order.
    """
    if not records or "env_id" not in records[0] or "episode_in_env" not in records[0]:
        return [dict(r, episode=i) for i, r in enumerate(records[:n])]
    out = []
    for r in records:
        i = int(r["episode_in_env"]) * num_envs + int(r["env_id"])
        if i < n:
            out.append(dict(r, episode=i))
    out.sort(key=lambda r: r["episode"])
    return out


def summarize(
    episodes: list[dict[str, Any]],
    *,
    policy: str,
    tier: str,
    seed: int,
    physics: str = "physx",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """CLAUDE.md section 6.2 result dict from per-episode records.

    ``overload_events`` is the mean number of servo overload events per episode; ``wrong_port`` the fraction of
    episodes that entered a non-target port; ``mean_time_to_latch_s`` is over latched episodes only (``None`` if none).
    """
    n = len(episodes)
    furthest = torch.tensor([int(e["furthest_stage"]) for e in episodes], dtype=torch.long)
    fun = {k: round(v, 4) for k, v in funnel(furthest).items()} if n else {s: 0.0 for s in STAGES[1:]}
    k = int((furthest >= len(STAGES) - 1).sum()) if n else 0
    lo, hi = wilson_ci(k, n)
    ttl = _mean(_num(e.get("time_to_latch")) for e in episodes if int(e["furthest_stage"]) >= len(STAGES) - 1)
    res = {
        "policy": policy,
        "tier": tier,
        "n": n,
        "seed": seed,
        "funnel": fun,
        "success": round(k / n, 4) if n else 0.0,
        "ci95": [round(lo, 4), round(hi, 4)],
        "mean_time_to_latch_s": None if ttl is None else round(ttl, 3),
        "wrong_port": round(sum(bool(e.get("wrong_port", False)) for e in episodes) / n, 4) if n else 0.0,
        "overload_events": round(sum(int(e.get("overload_events", 0)) for e in episodes) / n, 4) if n else 0.0,
        "git_sha": git_sha(),
        "servosense_version": servosense_version(),
        "physics": physics,
    }
    peak = _mean(_num(e.get("peak_axial_force")) for e in episodes)
    res["mean_peak_axial_force_n"] = None if peak is None else round(peak, 3)
    if extra:
        res.update(extra)
    return res


def write_result(out_dir: str | Path, result: dict[str, Any], episodes: list[dict[str, Any]]) -> Path:
    """Write ``<out>/<policy>/<tier>.json`` and ``<tier>_episodes.csv``; returns the JSON path."""
    d = Path(out_dir) / result["policy"]
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{result['tier']}.json"
    path.write_text(json.dumps(result, indent=1) + "\n")
    with open(d / f"{result['tier']}_episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for e in episodes:
            row = {k: e.get(k) for k in CSV_FIELDS}
            row["stage"] = STAGES[int(e["furthest_stage"])]
            row["tier"] = e.get("tier", result["tier"])
            w.writerow(row)
    return path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_results(results_dir: str | Path) -> list[dict[str, Any]]:
    """All ``results/<policy>/<tier>.json`` files that follow the section 6.2 schema (directory and file name must
    match the ``policy`` and ``tier`` fields, so other JSONs under results/ are ignored)."""
    out = []
    for p in sorted(Path(results_dir).glob("*/*.json")):
        try:
            r = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (isinstance(r, dict) and {"policy", "tier", "n", "success", "funnel"} <= r.keys()
                and r["policy"] == p.parent.name and r["tier"] == p.stem):
            out.append(r)
    return out


def policy_sort_key(policy: str) -> tuple[int, str]:
    return (POLICY_ORDER.index(policy), "") if policy in POLICY_ORDER else (len(POLICY_ORDER), policy)


def tier_sort_key(tier: str) -> tuple[int, str]:
    return (TIERS.index(tier), "") if tier in TIERS else (len(TIERS), tier)


def plot_funnel(results: list[dict[str, Any]], path: str | Path, title: str | None = None) -> Path:
    """Stage funnel (fraction reaching each stage) per tier for one policy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = list(STAGES[1:])
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=120)
    for r in sorted(results, key=lambda r: tier_sort_key(r["tier"])):
        ys = [r["funnel"].get(s, 0.0) for s in stages]
        ax.plot(stages, ys, marker="o", label=f"{r['tier']} (n={r['n']}, success {r['success']:.2f})")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("fraction of episodes reaching stage")
    ax.set_title(title or f"Stage funnel: {results[0]['policy']}" if results else "Stage funnel")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return Path(path)


# ----------------------------------------------------------------------------------------------- leaderboard / gap
GAP_POLICIES = ("open_loop", "telemetry_spiral", "oracle_spiral")
MIN_GAP = 0.05  # oracle - open_loop below this: gap_closure is not reported (ratio of noise)


def _successes(r: dict[str, Any]) -> tuple[int, int]:
    return int(round(r["success"] * r["n"])), int(r["n"])


def gap_closure(
    open_loop: tuple[int, int],
    telemetry: tuple[int, int],
    oracle: tuple[int, int],
    *,
    min_gap: float = MIN_GAP,
    draws: int = 20000,
    seed: int = 0,
) -> dict[str, Any]:
    """``(telemetry - open_loop) / (oracle - open_loop)`` from ``(successes, n)`` pairs, with a 95 % interval.

    The interval is Monte Carlo over independent Jeffreys posteriors Beta(k + 1/2, n - k + 1/2) of the three success
    rates (draws whose denominator is below ``1e-3`` are discarded). If the point estimate of the denominator is below
    ``min_gap`` the ratio is not reported (``value`` is ``None``) because it would be dominated by noise.
    """
    import numpy as np

    p = [k / n if n else float("nan") for k, n in (open_loop, telemetry, oracle)]
    den = p[2] - p[0]
    out: dict[str, Any] = {"open_loop": p[0], "telemetry_spiral": p[1], "oracle_spiral": p[2], "oracle_gap": den}
    if not all(n > 0 for _, n in (open_loop, telemetry, oracle)) or den < min_gap:
        out.update(value=None, ci95=None, note=f"oracle - open_loop = {den:.3f} < {min_gap}: not reported")
        return out
    rng = np.random.default_rng(seed)
    o, t, r = (rng.beta(k + 0.5, n - k + 0.5, size=draws) for k, n in (open_loop, telemetry, oracle))
    d = r - o
    ok = np.abs(d) > 1e-3
    g = (t[ok] - o[ok]) / d[ok]
    lo, hi = np.percentile(g, [2.5, 97.5])
    out.update(value=round((p[1] - p[0]) / den, 4), ci95=[round(float(lo), 4), round(float(hi), 4)], note=None)
    return out


def gap_closure_by_tier(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """gap_closure for every tier where all three scripted policies have results."""
    by = {(r["policy"], r["tier"]): r for r in results}
    tiers = sorted({r["tier"] for r in results}, key=tier_sort_key)
    return {
        t: gap_closure(*(_successes(by[(p, t)]) for p in GAP_POLICIES))
        for t in tiers
        if all((p, t) in by for p in GAP_POLICIES)
    }


def _fmt_ci(ci) -> str:
    return "" if ci is None else f" [{ci[0]:.2f}, {ci[1]:.2f}]"


def leaderboard_markdown(results: list[dict[str, Any]]) -> str:
    """Markdown leaderboard: success table, gap closure, per-tier stage funnels."""
    policies = sorted({r["policy"] for r in results}, key=policy_sort_key)
    tiers = sorted({r["tier"] for r in results}, key=tier_sort_key)
    by = {(r["policy"], r["tier"]): r for r in results}
    L = ["# LatchBench leaderboard", "",
         "Success = latched (CLAUDE.md section 6.1). Cells: success rate [Wilson 95 % CI] (n). "
         "Generated by `scripts/leaderboard.py` from `results/*/*.json`; do not edit by hand.", ""]
    L += ["| policy | " + " | ".join(tiers) + " |", "|---|" + "---|" * len(tiers)]
    for p in policies:
        cells = []
        for t in tiers:
            r = by.get((p, t))
            cells.append("-" if r is None else f"{r['success']:.2f}{_fmt_ci(r.get('ci95'))} ({r['n']})")
        L.append(f"| {p} | " + " | ".join(cells) + " |")
    gaps = gap_closure_by_tier(results)
    L += ["", "## Gap closure", "",
          "`gap_closure = (telemetry_spiral - open_loop) / (oracle_spiral - open_loop)`; 95 % interval from "
          "independent Jeffreys posteriors. Not reported when oracle - open_loop < "
          f"{MIN_GAP} (ratio of noise).", ""]
    if gaps:
        L += ["| tier | open_loop | telemetry_spiral | oracle_spiral | gap_closure |", "|---|---|---|---|---|"]
        for t, g in gaps.items():
            v = g["note"] if g["value"] is None else f"**{g['value']:.2f}**{_fmt_ci(g['ci95'])}"
            L.append(f"| {t} | {g['open_loop']:.2f} | {g['telemetry_spiral']:.2f} | {g['oracle_spiral']:.2f} | {v} |")
    else:
        L.append("_Needs open_loop, telemetry_spiral and oracle_spiral results on the same tier._")
    L += ["", "## Stage funnels", ""]
    stages = list(STAGES[1:])
    for t in tiers:
        L += [f"### {t}", "", "| policy | " + " | ".join(stages) + " | wrong port | overload / ep | time to latch [s] |",
              "|---|" + "---|" * (len(stages) + 3)]
        for p in policies:
            r = by.get((p, t))
            if r is None:
                continue
            ttl = r.get("mean_time_to_latch_s")
            L.append(f"| {p} | " + " | ".join(f"{r['funnel'].get(s, 0.0):.2f}" for s in stages)
                     + f" | {r.get('wrong_port', 0.0):.2f} | {r.get('overload_events', 0.0):.2f} | "
                     + ("-" if ttl is None else f"{ttl:.1f}") + " |")
        L.append("")
    L += ["## Provenance", "", "| policy | tier | git_sha | servosense | physics | seed | pose noise [mm] |",
          "|---|---|---|---|---|---|---|"]
    for p in policies:
        for t in tiers:
            r = by.get((p, t))
            if r is not None:
                L.append(f"| {p} | {t} | {r.get('git_sha', '?')} | {r.get('servosense_version', '?')} | "
                         f"{r.get('physics', '?')} | {r.get('seed', '?')} | {r.get('pose_noise_mm', '?')} |")
    return "\n".join(L) + "\n"


def plot_leaderboard(results: list[dict[str, Any]], path: str | Path) -> Path:
    """Grouped bars: success per tier per policy with Wilson 95 % error bars."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    policies = sorted({r["policy"] for r in results}, key=policy_sort_key)
    tiers = sorted({r["tier"] for r in results}, key=tier_sort_key)
    by = {(r["policy"], r["tier"]): r for r in results}
    w = 0.8 / max(1, len(policies))
    fig, ax = plt.subplots(figsize=(7.0, 3.8), dpi=120)
    for i, p in enumerate(policies):
        xs, ys, err = [], [], [[], []]
        for j, t in enumerate(tiers):
            r = by.get((p, t))
            if r is None:
                continue
            lo, hi = r.get("ci95") or (r["success"], r["success"])
            xs.append(j - 0.4 + w * (i + 0.5))
            ys.append(r["success"])
            err[0].append(max(0.0, r["success"] - lo))
            err[1].append(max(0.0, hi - r["success"]))
        ax.bar(xs, ys, w, yerr=err, capsize=3, label=p)
    gaps = gap_closure_by_tier(results)
    labels = [t + (f"\ngap closure {gaps[t]['value']:.2f}" if gaps.get(t, {}).get("value") is not None else "")
              for t in tiers]
    ax.set_xticks(range(len(tiers)), labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("success (latched)")
    ax.set_title("LatchBench: RJ45 insertion success by clearance tier")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return Path(path)
