"""Tests for kipp.cli, written before the implementation (TDD).

`main(argv)` is invoked directly (never via subprocess) against a synthetic
plot file written to tmp_path.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from kipp.cli import main

N_COLS = 74


def _row(model, age, m, he_core, co_core, conv, n_cols=N_COLS):
    assert len(conv) == 12
    cols = ["0.0"] * n_cols
    cols[0] = str(model)
    cols[1] = str(age)
    cols[5] = str(m)
    cols[6] = str(he_core)
    cols[7] = str(co_core)
    cols[11:23] = [str(v) for v in conv]
    return " ".join(cols[:n_cols])


def _make_plot_file(tmp_path, n=6, include_bad_row=False):
    m_total = 10.0
    lines = []
    for i in range(1, n + 1):
        model = i
        age = 100.0 * i
        conv = [-3.0, 3.5] + [m_total] * 10
        lines.append(_row(model, age, m_total, 0.0, 0.5, conv))
    if include_bad_row:
        # SPEC.md illegal example: [3.0, -5.0, 7.0] admits no legal walk.
        bad_conv = [3.0, -5.0, 7.0] + [m_total] * 9
        lines.append(_row(n + 1, 100.0 * (n + 1), m_total, 0.0, 0.5, bad_conv))
    path = tmp_path / "plot.synthetic"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_output_flag_writes_png(tmp_path):
    plotfile = _make_plot_file(tmp_path)
    out = tmp_path / "out.png"
    rc = main([str(plotfile), "-o", str(out)])
    assert rc == 0
    assert out.exists()
    assert out.stat().st_size > 1024


def test_dump_intervals_writes_documented_json(tmp_path):
    plotfile = _make_plot_file(tmp_path)
    out_png = tmp_path / "out.png"
    out_json = tmp_path / "intervals.json"
    rc = main(
        [str(plotfile), "-o", str(out_png), "--dump-intervals", str(out_json)]
    )
    assert rc == 0
    assert out_json.exists()
    payload = json.loads(out_json.read_text())
    assert isinstance(payload, list)
    assert len(payload) == 6
    for entry in payload:
        assert set(entry.keys()) == {"model", "intervals"}
        assert isinstance(entry["model"], int)
        assert isinstance(entry["intervals"], list)
        for iv in entry["intervals"]:
            assert len(iv) == 3
            lo, hi, kind = iv
            assert isinstance(lo, float)
            assert isinstance(hi, float)
            assert kind in ("conv", "semi")
    # Every synthetic row decodes to conv(0,3.0), semi(3.0,3.5).
    first = payload[0]
    assert first["model"] == 1
    assert first["intervals"] == [[0.0, 3.0, "conv"], [3.0, 3.5, "semi"]]


def test_xaxis_collapse_works(tmp_path):
    plotfile = _make_plot_file(tmp_path)
    out = tmp_path / "out.png"
    rc = main([str(plotfile), "-o", str(out), "--xaxis", "collapse"])
    assert rc == 0
    assert out.exists()


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--no-semiconv"],
        ["--dots"],
        ["--n-mass", "50"],
        ["--dpi", "50"],
        ["--no-semiconv", "--dots", "--n-mass", "50", "--dpi", "50"],
    ],
)
def test_optional_flags_accepted(tmp_path, extra_args):
    plotfile = _make_plot_file(tmp_path)
    out = tmp_path / "out.png"
    rc = main([str(plotfile), "-o", str(out), *extra_args])
    assert rc == 0
    assert out.exists()


def test_summary_line_mentions_counts(tmp_path, capsys):
    plotfile = _make_plot_file(tmp_path, n=6, include_bad_row=True)
    out = tmp_path / "out.png"
    rc = main([str(plotfile), "-o", str(out)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "n_models" in captured.out
    assert "n_bad_rows" in captured.out
    # 7 rows total, 1 bad (the illegal boundary row appended last).
    assert "7" in captured.out
    assert "1" in captured.out


def test_missing_file_returns_nonzero_or_raises_system_exit(tmp_path):
    missing = tmp_path / "does_not_exist.plot"
    try:
        rc = main([str(missing), "-o", str(tmp_path / "out.png")])
    except SystemExit as exc:
        assert exc.code != 0
    else:
        assert rc != 0
