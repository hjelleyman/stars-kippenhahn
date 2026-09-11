"""Rasterise decoded convective/semiconvective intervals onto a mass grid.

Pure functions, no global state. Uses numpy vectorised slicing (via
np.searchsorted) so no per-cell Python loop is ever executed; the only
Python-level loops are over models and over intervals per model.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def time_edges(x: Iterable[float]) -> np.ndarray:
    """Convert 1-D cell-centre coordinates into n+1 cell edges.

    `x` must be strictly increasing or strictly decreasing, length >= 1.
    Edges are midpoints between neighbouring centres, extrapolated by half
    a step at both ends. A single centre `x0` yields `[x0 - 0.5, x0 + 0.5]`.

    Raises ValueError if `x` is empty or not strictly monotone.
    """
    arr = np.asarray(x, dtype=float)
    n = arr.shape[0]
    if n == 0:
        raise ValueError("time_edges requires at least one point, got 0")

    if n == 1:
        x0 = arr[0]
        return np.array([x0 - 0.5, x0 + 0.5])

    diffs = np.diff(arr)
    if not (np.all(diffs > 0) or np.all(diffs < 0)):
        raise ValueError(
            "time_edges requires a strictly increasing or strictly "
            "decreasing sequence"
        )

    midpoints = (arr[:-1] + arr[1:]) / 2.0
    first_edge = arr[0] - diffs[0] / 2.0
    last_edge = arr[-1] + diffs[-1] / 2.0
    return np.concatenate(([first_edge], midpoints, [last_edge]))


def rasterise(
    intervals_per_model: Sequence[Sequence[object]],
    m_max: float,
    n_mass: int = 800,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterise per-model intervals onto a uniform mass grid.

    `intervals_per_model` is a list (length n_models) of lists of objects
    with `.lo`, `.hi`, `.kind` ("conv" or "semi") attributes.

    Returns `(m_edges, conv, semi)`: `m_edges` has length n_mass + 1 and
    spans [0, m_max]; `conv` and `semi` are boolean arrays of shape
    (n_models, n_mass). A cell is marked if its centre lies in [lo, hi) of
    an interval of the matching kind. Overlapping conv/semi intervals may
    mark the same cell in both grids; disjointness is never assumed.
    """
    m_edges = np.linspace(0.0, m_max, n_mass + 1)
    centres = (m_edges[:-1] + m_edges[1:]) / 2.0

    n_models = len(intervals_per_model)
    conv = np.zeros((n_models, n_mass), dtype=bool)
    semi = np.zeros((n_models, n_mass), dtype=bool)

    for row_idx, intervals in enumerate(intervals_per_model):
        conv_row = conv[row_idx]
        semi_row = semi[row_idx]
        for interval in intervals:
            lo = interval.lo
            hi = interval.hi
            if hi <= lo:
                continue
            i0 = int(np.searchsorted(centres, lo, side="left"))
            i1 = int(np.searchsorted(centres, hi, side="left"))
            if i1 <= i0:
                continue
            if interval.kind == "conv":
                conv_row[i0:i1] = True
            elif interval.kind == "semi":
                semi_row[i0:i1] = True

    return m_edges, conv, semi
