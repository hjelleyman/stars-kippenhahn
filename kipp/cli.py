"""Command-line entry point: `python -m kipp PLOTFILE [options]`."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

import matplotlib.pyplot as plt

from kipp.decode import decode_all
from kipp.io import load_plot
from kipp.render import plot_kippenhahn

__all__ = ["main"]

_XAXIS_CHOICES = ("model", "index", "age", "collapse")
_MAX_BAD_MODELS_SHOWN = 20


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kipp", description="Render a shaded Kippenhahn diagram from a STARS plot file."
    )
    parser.add_argument("plotfile", help="Path to a STARS plot file.")
    parser.add_argument("-o", "--output", default=None, help="Write the figure to this PNG path instead of showing it.")
    parser.add_argument("--xaxis", choices=_XAXIS_CHOICES, default="model", help="Quantity to use for the x axis.")
    parser.add_argument("--n-mass", type=int, default=800, help="Mass-grid resolution for rasterisation.")
    parser.add_argument("--no-semiconv", action="store_true", help="Merge semiconvective cells into the convective shading.")
    parser.add_argument("--dots", action="store_true", help="Overlay the legacy per-slot |conv| scatter.")
    parser.add_argument("--dpi", type=float, default=150, help="Resolution (dots per inch) used when saving with -o.")
    parser.add_argument("--dump-intervals", default=None, help="Write decoded intervals as JSON to this path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse `argv`, render the requested Kippenhahn diagram, and return an
    exit code (0 on success). `-o` saves a PNG; without it the figure is
    shown interactively. `--dump-intervals` additionally writes the decoded
    intervals as JSON.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        data = load_plot(args.plotfile)
    except OSError as exc:
        print(f"kipp: error reading {args.plotfile!r}: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"kipp: error parsing {args.plotfile!r}: {exc}", file=sys.stderr)
        return 1

    intervals_per_model, bad_rows = decode_all(
        data["conv"], data["M"], conv_env=data.get("conv_env")
    )
    n_models = len(data["model"])
    bad_models = [int(data["model"][i]) for i in bad_rows[:_MAX_BAD_MODELS_SHOWN]]
    print(
        f"n_models={n_models} n_bad_rows={len(bad_rows)} "
        f"bad_models(first {_MAX_BAD_MODELS_SHOWN})={bad_models}"
    )

    if args.dump_intervals:
        payload = [
            {
                "model": int(data["model"][i]),
                "intervals": [[iv.lo, iv.hi, iv.kind] for iv in intervals],
            }
            for i, intervals in enumerate(intervals_per_model)
        ]
        with open(args.dump_intervals, "w") as f:
            json.dump(payload, f)

    ax = plot_kippenhahn(
        data,
        xaxis=args.xaxis,
        n_mass=args.n_mass,
        semiconv=not args.no_semiconv,
        show_dots=args.dots,
        intervals_per_model=intervals_per_model,
    )

    if args.output:
        ax.figure.savefig(args.output, dpi=args.dpi)
        plt.close(ax.figure)
    else:
        plt.show()

    return 0
