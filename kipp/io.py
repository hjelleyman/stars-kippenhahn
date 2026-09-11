"""Load STARS `plot` files.

Plot files are whitespace-separated, one model per line. The real data file
this package targets has 74 columns per line, except that STARS writes
fixed-width fields and a large R_conv-env (column 72) overflows into
M_conv-env (column 71), producing a single merged token such as
`19.83829100.25554` and a 73-token line. The file is therefore read line by
line (never with `np.loadtxt`) and the merged token is split back apart.
"""
from __future__ import annotations

import os
import re

import numpy as np

__all__ = ["load_plot"]

_MIN_COLUMNS = 23
_CONV_START, _CONV_END = 11, 23  # conv1..12 occupy columns 11..22 inclusive
_CONV_ENV = 70                    # M_conv-env: mass coordinate of envelope base
# a 5-decimal number immediately followed by another number (field overflow)
_MERGED = re.compile(r"^(-?\d+\.\d{5})(-?\d+\.\d+)$")


def _leading_float(token: str) -> float:
    """Parse a field, recovering the first number from a fixed-width overflow
    merge; nan if the token is not numeric at all."""
    try:
        return float(token)
    except ValueError:
        m = _MERGED.match(token)
        return float(m.group(1)) if m else float("nan")


def load_plot(path: str | os.PathLike) -> dict[str, np.ndarray]:
    """Read a STARS plot file into a dict of numpy arrays.

    Returns 1-D float arrays `model`, `age`, `M`, `He_core`, `CO_core`,
    `conv_env` (length n_models) and a 2-D float array `conv` of shape
    (n_models, 12). `conv_env` is the mass coordinate of the base of the
    convective envelope (equal to `M` when there is none) and is nan where
    the column is absent. Rows are kept in file order.

    Raises ValueError, naming the 1-indexed line number, if a non-blank line
    has fewer than 23 whitespace-separated fields.
    """
    rows: list[list[float]] = []
    conv_env: list[float] = []
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
            conv_env.append(
                _leading_float(fields[_CONV_ENV]) if len(fields) > _CONV_ENV
                else float("nan")
            )

    arr = np.array(rows, dtype=np.float64)

    return {
        "model": arr[:, 0],
        "age": arr[:, 1],
        "M": arr[:, 5],
        "He_core": arr[:, 6],
        "CO_core": arr[:, 7],
        "conv": arr[:, _CONV_START:_CONV_END],
        "conv_env": np.array(conv_env, dtype=np.float64),
    }
