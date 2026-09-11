"""Tests for kipp.io, written before the implementation (TDD).

Synthetic STARS `plot`-file fixtures are built in tmp_path: whitespace
separated, >=23 columns, with one deliberately ragged line (73 vs 74
columns) to prove trailing columns are harmless, and a separate fixture with
a too-short line to prove load_plot raises ValueError naming the offending
line number.
"""
from pathlib import Path

import numpy as np
import pytest

from kipp.io import load_plot

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_DATA = REPO_ROOT / "data" / "plot.STANDARD_SINGLE_DOUBLEPREC"


def _row(model, age, m, he_core, co_core, conv, n_cols=74):
    """Build one whitespace-separated plot-file line with `n_cols` total
    columns. Columns 0,1,5,6,7 carry the named fields, 11..22 carry `conv`
    (12 values), and everything else is filler so ragged trailing columns
    can be exercised without touching the fields load_plot actually reads.
    """
    assert len(conv) == 12
    cols = ["0.0"] * n_cols
    cols[0] = str(model)
    cols[1] = str(age)
    cols[5] = str(m)
    cols[6] = str(he_core)
    cols[7] = str(co_core)
    cols[11:23] = [str(v) for v in conv]
    return " ".join(cols[:n_cols])


def test_load_plot_basic_fields_and_shapes(tmp_path):
    conv_a = list(range(1, 13))
    conv_b = [-v for v in conv_a]
    conv_c = [0.0] * 12
    lines = [
        _row(1, 0.0, 10.0, 1.0, 0.5, conv_a, n_cols=74),
        _row(2, 100.0, 20.0, 2.0, 1.5, conv_b, n_cols=74),
        _row(3, 200.0, 30.0, 3.0, 2.5, conv_c, n_cols=74),
    ]
    path = tmp_path / "plot.synthetic"
    path.write_text("\n".join(lines) + "\n")

    data = load_plot(path)

    assert set(data.keys()) == {"model", "age", "M", "He_core", "CO_core", "conv"}
    for key in ("model", "age", "M", "He_core", "CO_core"):
        arr = data[key]
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3,)
        assert arr.dtype == np.float64

    assert data["conv"].shape == (3, 12)
    assert data["conv"].dtype == np.float64

    # Row order preserved.
    np.testing.assert_array_equal(data["model"], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(data["age"], [0.0, 100.0, 200.0])
    np.testing.assert_array_equal(data["M"], [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(data["He_core"], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(data["CO_core"], [0.5, 1.5, 2.5])
    np.testing.assert_array_equal(data["conv"][0], conv_a)
    np.testing.assert_array_equal(data["conv"][1], conv_b)
    np.testing.assert_array_equal(data["conv"][2], conv_c)


def test_load_plot_ragged_trailing_columns_are_harmless(tmp_path):
    # One line has 74 columns, the other only 73 (mirrors the real data
    # file's line 1206) -- both should load fine since only the first 23
    # columns are ever read.
    conv_a = list(range(1, 13))
    conv_b = list(range(13, 25))
    lines = [
        _row(1, 0.0, 10.0, 1.0, 0.5, conv_a, n_cols=74),
        _row(2, 100.0, 20.0, 2.0, 1.5, conv_b, n_cols=73),
    ]
    path = tmp_path / "plot.ragged"
    path.write_text("\n".join(lines) + "\n")

    data = load_plot(path)

    assert data["model"].shape == (2,)
    assert data["conv"].shape == (2, 12)
    np.testing.assert_array_equal(data["conv"][0], conv_a)
    np.testing.assert_array_equal(data["conv"][1], conv_b)


def test_load_plot_too_few_columns_raises_value_error_with_line_number(tmp_path):
    good = _row(1, 0.0, 10.0, 1.0, 0.5, list(range(12)), n_cols=74)
    bad = " ".join(["1.0"] * 20)  # only 20 columns, short of the required 23
    path = tmp_path / "plot.bad"
    path.write_text("\n".join([good, bad]) + "\n")

    with pytest.raises(ValueError, match=r"\b2\b"):
        load_plot(path)


def test_load_plot_too_few_columns_line_number_is_1_indexed(tmp_path):
    bad = " ".join(["1.0"] * 20)
    good = _row(1, 0.0, 10.0, 1.0, 0.5, list(range(12)), n_cols=74)
    path = tmp_path / "plot.bad_first_line"
    # Bad line is first, so it must be reported as line 1.
    path.write_text("\n".join([bad, good]) + "\n")

    with pytest.raises(ValueError, match=r"\b1\b"):
        load_plot(path)


@pytest.mark.skipif(not REAL_DATA.exists(), reason="real data file not present")
def test_load_plot_real_data_sanity():
    data = load_plot(REAL_DATA)
    assert data["model"].shape == (3704,)
    assert data["conv"].shape == (3704, 12)
    assert data["M"][0] == pytest.approx(20.73143)
