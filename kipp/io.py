"""Load STARS `plot` files.

Plot files are whitespace-separated, one model per line. The real data file
this package targets has 74 columns per line except for one ragged line
with only 73 -- since only the first 23 columns carry fields this package
cares about, the file is read line by line (never with `np.loadtxt`, which
requires a rectangular array) and only those columns are kept.
"""
from __future__ import annotations

import os

import numpy as np

__all__ = ["load_plot"]

_MIN_COLUMNS = 23
_CONV_START, _CONV_END = 11, 23  # conv1..12 occupy columns 11..22 inclusive


def load_plot(path: str | os.PathLike) -> dict[str, np.ndarray]:
    """Read a STARS plot file into a dict of numpy arrays.

    Returns 1-D float arrays `model`, `age`, `M`, `He_core`, `CO_core`
    (length n_models) and a 2-D float array `conv` of shape (n_models, 12).
    Rows are kept in file order.

    Raises ValueError, naming the 1-indexed line number, if a non-blank line
    has fewer than 23 whitespace-separated fields.
    """
    rows: list[list[float]] = []
    with open(path) as f:
        for lineno, line in enumerate(f, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < _MIN_COLUMNS:
                raise ValueError(
                    f"line {lineno}: expected at least {_MIN_COLUMNS} columns, "
                    f"got {len(fields)}"
                )
            rows.append([float(v) for v in fields[:_MIN_COLUMNS]])

    arr = np.array(rows, dtype=np.float64)

    return {
        "model": arr[:, 0],
        "age": arr[:, 1],
        "M": arr[:, 5],
        "He_core": arr[:, 6],
        "CO_core": arr[:, 7],
        "conv": arr[:, _CONV_START:_CONV_END],
    }
