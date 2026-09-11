"""Render shaded Kippenhahn diagrams from decoded STARS plot data.

Pure rendering: `plot_kippenhahn` never calls `plt.show()` and works under
the `Agg` backend. `data` is the dict returned by `kipp.io.load_plot`.
"""
from __future__ import annotations

from typing import Any

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from kipp.decode import decode_all
from kipp.rasterise import rasterise, time_edges

__all__ = ["plot_kippenhahn"]

_XLABELS = {
    "model": "Model number",
    "index": "Model index",
    "age": "Age (yr)",
    "collapse": "log10(time to end of run / yr)",
}


def _strictly_increasing(x: np.ndarray) -> np.ndarray:
    """Nudge a non-strictly-monotone array forward into strict increase.

    STARS' `model`/`age` columns are usually non-decreasing but real runs
    also contain short backtracks (a rejected timestep retried near the end
    of the run produces a model whose age is *lower* than the previous
    line's, not just equal to it) -- so this cannot assume non-decreasing
    input. Instead it walks left to right and whenever an element would not
    exceed its (possibly already-nudged) predecessor, bumps it up to
    `previous + eps` where `eps` is a perturbation far smaller than any
    physically meaningful gap (`1e-9 * max(1, |previous|)`). Elements that
    are already strictly greater than their predecessor are left untouched.
    """
    arr = np.asarray(x, dtype=float).copy()
    n = arr.shape[0]
    for i in range(1, n):
        if arr[i] <= arr[i - 1]:
            eps = 1e-9 * max(1.0, abs(arr[i - 1]))
            arr[i] = arr[i - 1] + eps
    return arr


def _compute_x(data: dict[str, Any], xaxis: str) -> tuple[np.ndarray, str, bool]:
    """Return (x, xlabel, invert) for the requested xaxis mode."""
    if xaxis == "model":
        x = np.asarray(data["model"], dtype=float)
        x = _strictly_increasing(x)
        return x, _XLABELS[xaxis], False
    if xaxis == "index":
        x = np.arange(len(data["model"]), dtype=float)
        return x, _XLABELS[xaxis], False
    if xaxis == "age":
        x = np.asarray(data["age"], dtype=float)
        x = _strictly_increasing(x)
        return x, _XLABELS[xaxis], False
    if xaxis == "collapse":
        age = np.asarray(data["age"], dtype=float)
        age_inc = _strictly_increasing(age)
        t_end = age_inc[-1]
        pos_diffs = np.diff(age_inc)
        pos_diffs = pos_diffs[pos_diffs > 0]
        dt_floor = float(pos_diffs.min()) if pos_diffs.size else 1.0
        x = np.log10(t_end - age_inc + dt_floor)
        return x, _XLABELS[xaxis], True
    raise ValueError(
        f"xaxis must be one of {sorted(_XLABELS)}, got {xaxis!r}"
    )


def plot_kippenhahn(
    data: dict[str, Any],
    *,
    xaxis: str = "model",
    n_mass: int = 800,
    semiconv: bool = True,
    show_dots: bool = False,
    ax: matplotlib.axes.Axes | None = None,
    conv_color: str = "#1f4fd8",
    semi_color: str = "#9ab4f0",
    title: str | None = None,
    intervals_per_model: list[list[Any]] | None = None,
) -> matplotlib.axes.Axes:
    """Draw a shaded Kippenhahn diagram of `data` onto `ax` (or a new one).

    `data` is the dict from `kipp.io.load_plot`. Pass `intervals_per_model`
    (the first element of `decode_all`'s result) to skip decoding when the
    caller has already done it. Layers bottom to top:
    semiconvective shading (if `semiconv`), convective shading, CO-core
    fill, He-core line, total-mass line. `show_dots=True` overlays the
    legacy per-slot |conv| scatter for validation. Returns the Axes; never
    calls `plt.show()`.
    """
    x, xlabel, invert = _compute_x(data, xaxis)

    M = np.asarray(data["M"], dtype=float)
    He_core = np.asarray(data["He_core"], dtype=float)
    CO_core = np.asarray(data["CO_core"], dtype=float)
    m_max = float(M.max()) * 1.001

    if intervals_per_model is None:
        intervals_per_model, _bad_rows = decode_all(
            data["conv"], M, conv_env=data.get("conv_env")
        )
    m_edges, conv, semi = rasterise(intervals_per_model, m_max=m_max, n_mass=n_mass)

    if not semiconv:
        conv = conv | semi
        semi = np.zeros_like(semi)

    # For "collapse", x was built from a strictly increasing age array via a
    # monotonically decreasing transform, so x itself is already strictly
    # decreasing and time_edges handles it directly -- no extra transform
    # of the edges is needed.
    x_edges = time_edges(x)

    if ax is None:
        _fig, ax = plt.subplots()

    legend_handles: list[Any] = []

    if semiconv:
        semi_mask = np.ma.masked_where(~semi, np.ones_like(semi, dtype=float))
        ax.pcolormesh(
            x_edges,
            m_edges,
            semi_mask.T,
            shading="flat",
            cmap=ListedColormap([semi_color]),
            rasterized=True,
            zorder=1,
        )
        legend_handles.append(Patch(color=semi_color, label="Semiconvective"))

    conv_mask = np.ma.masked_where(~conv, np.ones_like(conv, dtype=float))
    ax.pcolormesh(
        x_edges,
        m_edges,
        conv_mask.T,
        shading="flat",
        cmap=ListedColormap([conv_color]),
        rasterized=True,
        zorder=2,
    )
    legend_handles.append(Patch(color=conv_color, label="Convective"))

    ax.fill_between(
        x, 0, CO_core, color="0.7", alpha=0.5, zorder=3, label="CO core"
    )
    legend_handles.append(Patch(color="0.7", alpha=0.5, label="CO core"))

    (he_line,) = ax.plot(
        x, He_core, color="0.3", linestyle="--", zorder=4, label="He core"
    )
    legend_handles.append(he_line)

    (mass_line,) = ax.plot(
        x, M, color="black", lw=1.3, zorder=5, label="Total mass"
    )
    legend_handles.append(mass_line)

    if show_dots:
        conv_vals = np.abs(np.asarray(data["conv"], dtype=float))
        eps = 1e-4
        pad_mask = (conv_vals < eps) | (conv_vals >= M[:, None] - eps)
        x_dots = np.repeat(x, conv_vals.shape[1])
        y_dots = conv_vals.ravel()
        keep = ~pad_mask.ravel()
        ax.scatter(
            x_dots[keep],
            y_dots[keep],
            s=2,
            c="grey",
            alpha=0.5,
            zorder=6,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Mass coordinate (M$_\odot$)")
    ax.set_ylim(0, m_max)
    lo, hi = float(x_edges[0]), float(x_edges[-1])
    ax.set_xlim(min(lo, hi), max(lo, hi))
    if invert:
        ax.invert_xaxis()
    if title is not None:
        ax.set_title(title)
    ax.legend(handles=legend_handles, loc="upper right")

    return ax
