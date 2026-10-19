"""Builds results/LEADERBOARD.md and leaderboard.png from results/*/*.json.

    python scripts/leaderboard.py [--results results/]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a checkout without installing

from patchbench import results_io as rio  # noqa: E402


def build(results_dir: str | Path) -> tuple[Path, Path | None]:
    results_dir = Path(results_dir)
    results = rio.load_results(results_dir)
    md = results_dir / "LEADERBOARD.md"
    md.write_text(rio.leaderboard_markdown(results))
    png = rio.plot_leaderboard(results, results_dir / "leaderboard.png") if results else None
    return md, png


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default=str(Path(__file__).resolve().parents[1] / "results"))
    args = ap.parse_args(argv)
    md, png = build(args.results)
    print(md.read_text())
    print(f"wrote {md}" + (f" and {png}" if png else ""))


if __name__ == "__main__":
    main()
