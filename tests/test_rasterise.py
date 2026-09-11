"""Tests for kipp.rasterise: time_edges and rasterise.

These tests define their own tiny stand-in for kipp.decode.Interval (a
dataclass with .lo, .hi, .kind) since kipp.decode may not exist yet.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pytest

from kipp.rasterise import rasterise, time_edges


@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float
    kind: str


# ---------------------------------------------------------------------------
# time_edges
# ---------------------------------------------------------------------------


def test_time_edges_uniform_increasing():
    edges = time_edges([0, 1, 2])
    np.testing.assert_allclose(edges, [-0.5, 0.5, 1.5, 2.5])


def test_time_edges_nonuniform_increasing():
    edges = time_edges([0, 1, 3])
    np.testing.assert_allclose(edges, [-0.5, 0.5, 2.0, 4.0])


def test_time_edges_strictly_decreasing():
    # e.g. log time-to-collapse
    edges = time_edges([5, 4, 2])
    # midpoints: (5+4)/2=4.5, (4+2)/2=3, extrapolate ends by half-step
    # first gap = 5-4=1 -> half step 0.5 -> edge0 = 5+0.5=5.5
    # last gap = 4-2=2 -> half step 1 -> edge_last = 2-1=1
    np.testing.assert_allclose(edges, [5.5, 4.5, 3.0, 1.0])
    # monotone decreasing
    assert np.all(np.diff(edges) < 0)


def test_time_edges_length_one():
    edges = time_edges([3.0])
    np.testing.assert_allclose(edges, [2.5, 3.5])


def test_time_edges_length_zero_raises():
    with pytest.raises(ValueError):
        time_edges([])


def test_time_edges_non_monotone_raises():
    with pytest.raises(ValueError):
        time_edges([0, 2, 1])
    with pytest.raises(ValueError):
        time_edges([0, 0, 1])  # flat / repeated is not strictly monotone


# ---------------------------------------------------------------------------
# rasterise
# ---------------------------------------------------------------------------


def test_m_edges_shape_and_bounds():
    m_edges, conv, semi = rasterise([[]], m_max=10.0, n_mass=10)
    assert m_edges.shape == (11,)
    assert m_edges[0] == pytest.approx(0.0)
    assert m_edges[-1] == pytest.approx(10.0)


def test_output_shapes_and_dtype():
    intervals_per_model = [[], [Interval(0, 1, "conv")]]
    m_edges, conv, semi = rasterise(intervals_per_model, m_max=10.0, n_mass=10)
    assert conv.shape == (2, 10)
    assert semi.shape == (2, 10)
    assert conv.dtype == bool
    assert semi.dtype == bool


def test_single_conv_interval_marks_correct_cells():
    # m_max=10, n_mass=10 -> cell width 1, centres 0.5, 1.5, ..., 9.5
    intervals_per_model = [[Interval(2, 4, "conv")]]
    m_edges, conv, semi = rasterise(intervals_per_model, m_max=10.0, n_mass=10)
    expected = np.zeros(10, dtype=bool)
    expected[[2, 3]] = True
    np.testing.assert_array_equal(conv[0], expected)
    np.testing.assert_array_equal(semi[0], np.zeros(10, dtype=bool))


def test_half_open_interval_boundary():
    # cell centres at 0.5, 1.5, 2.5, ... with interval [1.5, 2.5)
    # centre 1.5 (index1) should be marked (lo included), centre 2.5 (index2) not (hi excluded)
    intervals_per_model = [[Interval(1.5, 2.5, "conv")]]
    m_edges, conv, semi = rasterise(intervals_per_model, m_max=10.0, n_mass=10)
    assert conv[0, 1] == True  # noqa: E712
    assert conv[0, 2] == False  # noqa: E712


def test_semi_interval_marks_semi_only():
    intervals_per_model = [[Interval(2, 4, "semi")]]
    m_edges, conv, semi = rasterise(intervals_per_model, m_max=10.0, n_mass=10)
    expected = np.zeros(10, dtype=bool)
    expected[[2, 3]] = True
    np.testing.assert_array_equal(semi[0], expected)
    np.testing.assert_array_equal(conv[0], np.zeros(10, dtype=bool))


def test_overlapping_conv_and_semi_both_marked():
    intervals_per_model = [[Interval(2, 4, "conv"), Interval(3, 5, "semi")]]
    m_edges, conv, semi = rasterise(intervals_per_model, m_max=10.0, n_mass=10)
    conv_expected = np.zeros(10, dtype=bool)
    conv_expected[[2, 3]] = True
    semi_expected = np.zeros(10, dtype=bool)
    semi_expected[[3, 4]] = True
    np.testing.assert_array_equal(conv[0], conv_expected)
    np.testing.assert_array_equal(semi[0], semi_expected)
    # cell 3 is marked in both -> confirms no disjointness assumption
    assert conv[0, 3] and semi[0, 3]


def test_empty_interval_list_all_false():
    intervals_per_model = [[]]
    m_edges, conv, semi = rasterise(intervals_per_model, m_max=10.0, n_mass=10)
    assert not conv.any()
    assert not semi.any()


def test_interval_beyond_m_max_clipped():
    intervals_per_model = [[Interval(8, 20, "conv")]]
    m_edges, conv, semi = rasterise(intervals_per_model, m_max=10.0, n_mass=10)
    expected = np.zeros(10, dtype=bool)
    expected[[8, 9]] = True
    np.testing.assert_array_equal(conv[0], expected)


def test_interval_hi_lte_lo_marks_nothing():
    intervals_per_model = [[Interval(5, 5, "conv"), Interval(6, 4, "semi")]]
    m_edges, conv, semi = rasterise(intervals_per_model, m_max=10.0, n_mass=10)
    assert not conv.any()
    assert not semi.any()


def test_n_mass_one():
    intervals_per_model = [[Interval(0, 10, "conv")], []]
    m_edges, conv, semi = rasterise(intervals_per_model, m_max=10.0, n_mass=1)
    assert m_edges.shape == (2,)
    assert conv.shape == (2, 1)
    assert conv[0, 0] == True  # noqa: E712
    assert conv[1, 0] == False  # noqa: E712


def test_performance_smoke():
    rng = np.random.default_rng(0)
    n_models = 4000
    n_intervals = 6
    m_max = 20.0
    intervals_per_model = []
    for _ in range(n_models):
        row = []
        for _ in range(n_intervals):
            lo, hi = sorted(rng.uniform(0, m_max, size=2))
            kind = "conv" if rng.random() < 0.5 else "semi"
            row.append(Interval(float(lo), float(hi), kind))
        intervals_per_model.append(row)

    start = time.perf_counter()
    m_edges, conv, semi = rasterise(intervals_per_model, m_max=m_max, n_mass=800)
    elapsed = time.perf_counter() - start

    assert conv.shape == (n_models, 800)
    assert semi.shape == (n_models, 800)
    assert elapsed < 2.0
