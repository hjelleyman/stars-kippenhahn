"""End-to-end checks against the real STARS plot file in data/.

These assert physics-level facts about a 20.7 Msun STARS run rather than
exact numbers, so they should survive small changes to eps or rendering.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from kipp import decode_all, load_plot, plot_kippenhahn
from kipp.cli import main
from kipp.decode import is_truncated

DATA = Path(__file__).resolve().parents[1] / "data" / "plot.STANDARD_SINGLE_DOUBLEPREC"

pytestmark = pytest.mark.skipif(not DATA.exists(), reason="sample data file not present")


@pytest.fixture(scope="module")
def decoded():
    data = load_plot(DATA)
    intervals, bad = decode_all(data["conv"], data["M"], conv_env=data["conv_env"])
    return data, intervals, bad


def _conv(intervals):
    return [iv for iv in intervals if iv.kind == "conv"]


def test_every_model_decodes(decoded):
    _, _, bad = decoded
    assert bad == []


def test_intervals_are_inside_the_star_and_disjoint(decoded):
    data, intervals, _ = decoded
    for k, row in enumerate(intervals):
        for iv in row:
            assert 0.0 <= iv.lo < iv.hi <= data["M"][k] + 1e-9
        for a, b in zip(row, row[1:]):
            assert a.hi <= b.lo + 1e-12


def test_zams_has_a_single_convective_core_near_nine_solar_masses(decoded):
    _, intervals, _ = decoded
    core = _conv(intervals[0])
    assert len(core) == 1
    assert core[0].lo < 0.01
    assert 8.5 < core[0].hi < 9.5


def test_core_shrinks_through_the_main_sequence_and_vanishes(decoded):
    _, intervals, _ = decoded
    top = [max((iv.hi for iv in _conv(row) if iv.lo < 0.01), default=np.nan)
           for row in intervals[:700]]
    assert top[0] > top[200] > top[400] > top[500]
    assert np.isnan(top[650])


def test_intermediate_zone_appears_after_the_main_sequence(decoded):
    _, intervals, _ = decoded
    hits = [k for k in range(600, 1600)
            for iv in _conv(intervals[k]) if iv.lo > 5 and iv.hi > 10]
    assert hits and 650 < hits[0] < 750


def test_helium_burning_core_grows_inside_the_helium_core(decoded):
    data, intervals, _ = decoded
    top = {k: max((iv.hi for iv in _conv(intervals[k]) if iv.lo < 0.01), default=np.nan)
           for k in (1800, 2000, 2200)}
    assert top[1800] < top[2000] < top[2200]
    for k, t in top.items():
        assert t < data["He_core"][k]


def test_convective_envelope_exists_throughout_the_supergiant_phase(decoded):
    data, intervals, _ = decoded
    for k in range(2000, len(intervals), 25):
        env = [iv for iv in _conv(intervals[k]) if iv.hi >= data["M"][k] - 0.05]
        assert env, f"row {k} has no convective envelope"
        assert 7.5 < env[-1].lo < 9.5, f"row {k}: envelope base {env[-1].lo}"


def test_truncated_rows_never_extend_a_central_shell_to_the_surface(decoded):
    data, intervals, _ = decoded
    for k, row in enumerate(intervals):
        if is_truncated(data["conv"][k], data["M"][k]):
            for iv in _conv(row):
                assert not (iv.lo < 5 and iv.hi >= data["M"][k] - 0.05)


@pytest.mark.parametrize("xaxis", ["model", "index", "age", "collapse"])
def test_renders_under_every_x_axis(decoded, xaxis):
    data, intervals, _ = decoded
    ax = plot_kippenhahn(data, xaxis=xaxis, intervals_per_model=intervals, n_mass=200)
    assert ax.get_xlabel()


def test_cli_end_to_end(tmp_path, capsys):
    out = tmp_path / "k.png"
    dump = tmp_path / "iv.json"
    rc = main([str(DATA), "-o", str(out), "--xaxis", "collapse", "--n-mass", "200",
               "--dump-intervals", str(dump)])
    assert rc == 0
    assert out.stat().st_size > 10_000
    assert dump.exists()
    assert "n_bad_rows=0" in capsys.readouterr().out
