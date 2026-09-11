"""Tests for kipp.render, written before the implementation (TDD).

Uses the Agg backend (set before importing pyplot) so figures render
headlessly; assertions are on artists / array data / saved-file existence,
never on pixels.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PathCollection, PolyCollection, QuadMesh
from matplotlib.lines import Line2D

from kipp.render import _strictly_increasing, plot_kippenhahn

# ---------------------------------------------------------------------------
# Synthetic data fixture
# ---------------------------------------------------------------------------
# 6 models, constant total mass 10, conv row built from the decode
# worked examples so decode_all always yields a conv core plus a semi shell:
# [-3.0, 3.5, pad(=+-M)...] -> start must be conv (since a '-' boundary while
# in rad is illegal) -> [conv(0, 3.0), semi(3.0, 3.5)] to... actually with
# pure padding after, the walk ends in semi and needs a closing boundary to
# the surface. Use decode.py's own contract: pad entries are 0.0 or +-M, and
# are dropped before the walk runs, so the walk only sees [-3.0, 3.5] and
# emits Interval(0, 3.0, "conv"), Interval(3.0, 3.5, "semi") -- semi does NOT
# extend to the surface since the row only supplies two real boundaries.


def _make_data(n=6):
    model = np.arange(1, n + 1, dtype=float)
    age = np.cumsum(np.full(n, 100.0))  # strictly increasing
    M = np.full(n, 10.0)
    He_core = np.zeros(n)
    CO_core = np.linspace(0.5, 1.5, n)
    pad = M[0]  # +-M padding
    row = [-3.0, 3.5] + [pad] * 10
    conv = np.tile(row, (n, 1))
    return {
        "model": model,
        "age": age,
        "M": M,
        "He_core": He_core,
        "CO_core": CO_core,
        "conv": conv,
    }


def _quadmeshes(ax):
    return [c for c in ax.collections if isinstance(c, QuadMesh)]


def _path_collections(ax):
    return [c for c in ax.collections if isinstance(c, PathCollection)]


# ---------------------------------------------------------------------------
# _strictly_increasing helper
# ---------------------------------------------------------------------------


def test_strictly_increasing_no_repeats_unchanged():
    x = np.array([1.0, 2.0, 3.0])
    out = _strictly_increasing(x)
    np.testing.assert_allclose(out, x)


def test_strictly_increasing_nudges_repeats():
    x = np.array([1.0, 2.0, 2.0, 2.0, 3.0])
    out = _strictly_increasing(x)
    assert np.all(np.diff(out) > 0)
    # values stay very close to the originals
    np.testing.assert_allclose(out, x, atol=1e-4)


def test_strictly_increasing_all_equal():
    x = np.array([5.0, 5.0, 5.0])
    out = _strictly_increasing(x)
    assert np.all(np.diff(out) > 0)


def test_strictly_increasing_handles_real_backtracks():
    # STARS retries near the end of a run can make age genuinely decrease
    # (not just repeat) for a few models before resuming its climb -- e.g.
    # the real sample file has a model whose age drops by ~1.04 yr before
    # the next model picks back up above the pre-backtrack value.
    x = np.array([10.0, 20.0, 19.0, 21.0, 22.0])
    out = _strictly_increasing(x)
    assert np.all(np.diff(out) > 0)
    # points that were already fine are untouched; only the backtrack (and
    # anything it would otherwise collide with) moves, and only by a
    # negligible amount.
    np.testing.assert_allclose(out[[0, 1, 3, 4]], x[[0, 1, 3, 4]])
    assert out[2] == pytest.approx(x[1], abs=1e-4)


# ---------------------------------------------------------------------------
# plot_kippenhahn
# ---------------------------------------------------------------------------


def test_returns_axes():
    data = _make_data()
    ax = plot_kippenhahn(data)
    assert isinstance(ax, plt.Axes)
    plt.close(ax.figure)


def test_semiconv_true_gives_two_quadmeshes():
    data = _make_data()
    ax = plot_kippenhahn(data, semiconv=True)
    assert len(_quadmeshes(ax)) == 2
    plt.close(ax.figure)


def test_semiconv_false_gives_one_quadmesh():
    data = _make_data()
    ax = plot_kippenhahn(data, semiconv=False)
    assert len(_quadmeshes(ax)) == 1
    plt.close(ax.figure)


def test_total_mass_line_present_with_right_ydata():
    data = _make_data()
    ax = plot_kippenhahn(data)
    lines = [
        ln
        for ln in ax.get_lines()
        if ln.get_color() in ("black", "#000000", (0, 0, 0, 1))
    ]
    assert lines, "expected a black total-mass line"
    np.testing.assert_allclose(lines[0].get_ydata(), data["M"])
    plt.close(ax.figure)


def test_legend_has_entries():
    data = _make_data()
    ax = plot_kippenhahn(data)
    legend = ax.get_legend()
    assert legend is not None
    assert len(legend.get_texts()) > 0
    plt.close(ax.figure)


@pytest.mark.parametrize(
    "xaxis,label",
    [
        ("model", "Model number"),
        ("index", "Model index"),
        ("age", "Age (yr)"),
        ("collapse", "log10(time to end of run / yr)"),
    ],
)
def test_xlabel_per_mode(xaxis, label):
    data = _make_data()
    ax = plot_kippenhahn(data, xaxis=xaxis)
    assert ax.get_xlabel() == label
    plt.close(ax.figure)


def test_collapse_inverts_x_axis():
    data = _make_data()
    ax = plot_kippenhahn(data, xaxis="collapse")
    xlim = ax.get_xlim()
    assert xlim[0] > xlim[1]
    plt.close(ax.figure)


def test_unknown_xaxis_raises_value_error():
    data = _make_data()
    with pytest.raises(ValueError):
        plot_kippenhahn(data, xaxis="bogus")


def test_show_dots_adds_path_collection():
    data = _make_data()
    ax_without = plot_kippenhahn(data, show_dots=False)
    ax_with = plot_kippenhahn(data, show_dots=True)
    assert len(_path_collections(ax_without)) == 0
    assert len(_path_collections(ax_with)) >= 1
    plt.close(ax_without.figure)
    plt.close(ax_with.figure)


def test_passing_ax_draws_into_it():
    data = _make_data()
    fig, ax = plt.subplots()
    returned = plot_kippenhahn(data, ax=ax)
    assert returned is ax
    plt.close(fig)


def test_repeated_model_numbers_do_not_raise():
    data = _make_data()
    data["model"] = np.array([1.0, 2.0, 3.0, 4.0, 4.0, 4.0], dtype=float)
    ax = plot_kippenhahn(data, xaxis="model")
    plt.close(ax.figure)


def test_repeated_ages_do_not_raise():
    data = _make_data()
    data["age"] = np.array([0.0, 100.0, 200.0, 300.0, 300.0, 400.0], dtype=float)
    ax = plot_kippenhahn(data, xaxis="age")
    plt.close(ax.figure)


def test_ylim_upper_matches_m_max():
    data = _make_data()
    ax = plot_kippenhahn(data)
    m_max = data["M"].max() * 1.001
    ylim = ax.get_ylim()
    assert ylim[0] == pytest.approx(0.0)
    assert ylim[1] == pytest.approx(m_max, rel=1e-3)
    plt.close(ax.figure)


def test_savefig_under_agg(tmp_path):
    data = _make_data()
    ax = plot_kippenhahn(data, show_dots=True)
    out = tmp_path / "kipp.png"
    ax.figure.savefig(out)
    plt.close(ax.figure)
    assert out.exists()
    assert out.stat().st_size > 1024


# --- envelope column is threaded through to the decoder ---------------------


def test_conv_env_is_passed_to_decoder(monkeypatch):
    synthetic_data = _make_data()
    import kipp.render as render_mod

    seen = {}

    def fake_decode_all(conv, M, **kwargs):
        seen.update(kwargs)
        return [[] for _ in range(len(M))], []

    monkeypatch.setattr(render_mod, "decode_all", fake_decode_all)
    data = dict(synthetic_data)
    data["conv_env"] = np.full(len(data["M"]), 7.5)
    plot_kippenhahn(data)
    assert "conv_env" in seen
    np.testing.assert_array_equal(seen["conv_env"], data["conv_env"])


def test_precomputed_intervals_skip_decoding(monkeypatch):
    synthetic_data = _make_data()
    import kipp.render as render_mod

    def boom(*a, **k):
        raise AssertionError("decode_all must not be called")

    monkeypatch.setattr(render_mod, "decode_all", boom)
    intervals = [[] for _ in range(len(synthetic_data["M"]))]
    ax = plot_kippenhahn(synthetic_data, intervals_per_model=intervals)
    assert ax is not None


# --- polish: layout and unknown-region bands ----------------------------------


def _twelve_central_shells():
    return [0.00608, -0.00341, 0.0183, -0.01831, 0.04689, -0.04687,
            0.15772, -0.15773, 0.23986, -0.23985, 0.35412, -0.35413]


def test_legend_sits_below_the_axes():
    ax = plot_kippenhahn(_make_data())
    leg = ax.get_legend()
    bb = leg.get_bbox_to_anchor().transformed(ax.transAxes.inverted())
    assert bb.y1 <= 0.0


def test_co_core_fill_is_drawn_beneath_the_shading():
    ax = plot_kippenhahn(_make_data())
    meshes = _quadmeshes(ax)
    fills = [c for c in ax.collections if isinstance(c, PolyCollection)]
    assert fills, "CO core fill missing"
    assert max(f.get_zorder() for f in fills) < min(m.get_zorder() for m in meshes)


def test_default_title_names_the_zams_mass():
    ax = plot_kippenhahn(_make_data())
    assert "10.0" in ax.get_title()


def test_truncated_rows_get_an_unknown_band_mesh():
    data = _make_data()
    data["conv"][2] = _twelve_central_shells()
    data["M"][:] = 17.18642
    data["conv"][[0, 1, 3, 4, 5], 2:] = 17.18642 * np.tile([1, -1], 5)
    data["conv_env"] = np.full(6, 7.757)
    n_without = len(_quadmeshes(plot_kippenhahn(_make_data())))
    ax = plot_kippenhahn(data)
    assert len(_quadmeshes(ax)) == n_without + 1
    assert any("not in file" in t.get_text().lower() for t in ax.get_legend().get_texts())
