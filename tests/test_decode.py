"""Tests for kipp.decode, written before the implementation (TDD).

The first group are worked examples: real rows from the sample plot file
(ZAMS, TAMS, post-MS) decoded by hand from the STARS manual's sign
convention, plus small synthetic rows covering each branch of the state
machine. Then padding-boundary precision, zero-width interval dropping, sort
order, Interval immutability, decode_all's error-handling contract, tie
handling, and truncation.
"""
import dataclasses
import numpy as np

import pytest

from kipp.decode import Interval, DecodeError, decode_row, decode_all


def test_zams_row():
    # Worked example 1:
    values = [
        0.00064, -0.00064, 9.01792, -8.96118,
        20.73143, -20.73143, 20.73143, -20.73143,
        20.73143, -20.73143, 20.73143, -20.73143,
    ]
    m_total = 20.73143
    result = decode_row(values, m_total)
    assert result == [
        Interval(0.00064, 8.96118, "conv"),
        Interval(8.96118, 9.01792, "semi"),
    ]


def test_tams_row():
    # Worked example 2: purely two real boundaries, rest padding.
    values = [2.19412, 3.11245, 19.85237, -19.85237, 0, 0, 0, 0, 0, 0, 0, 0]
    m_total = 19.8524
    result = decode_row(values, m_total)
    assert result == [Interval(2.19412, 3.11245, "semi")]


def test_post_ms_row():
    # Worked example 3: corrected: with eps=1e-4 and M=19.83848, NONE of the
    # 19.8x entries are padding (19.83684 is only 0.00164 below M, which is
    # well outside eps). Sorted abs values after +10.10233 are
    # +19.82018, -19.82024, -19.83681, +19.83684, giving an extra
    # rad -> semi -> conv -> semi run out to the surface.
    values = [
        5.96927, -5.97847, -7.32142, 7.52217,
        8.41099, -8.76458, -9.85256, 10.10233,
        19.82018, -19.82024, 19.83684, -19.83681,
    ]
    m_total = 19.83848
    result = decode_row(values, m_total)
    assert result == [
        Interval(5.96927, 5.97847, "semi"),
        Interval(5.97847, 7.32142, "conv"),
        Interval(7.32142, 7.52217, "semi"),
        Interval(8.41099, 8.76458, "semi"),
        Interval(8.76458, 9.85256, "conv"),
        Interval(9.85256, 10.10233, "semi"),
        Interval(19.82018, 19.82024, "semi"),
        Interval(19.82024, 19.83681, "conv"),
        Interval(19.83681, 19.83684, "semi"),
    ]


def test_all_padding_row():
    values = [0.0] * 12
    m_total = 20.0
    assert decode_row(values, m_total) == []


def test_all_padding_row_at_surface():
    values = [20.0, -20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    m_total = 20.0
    assert decode_row(values, m_total) == []


def test_ambiguous_conv_or_semi_prefers_conv():
    # Worked example 5:
    values = [-3.0, -5.0] + [0.0] * 10
    m_total = 100.0
    result = decode_row(values, m_total, prefer="conv")
    assert result == [
        Interval(0.0, 3.0, "conv"),
        Interval(3.0, 5.0, "semi"),
        Interval(5.0, m_total, "conv"),
    ]


def test_ambiguous_conv_or_semi_prefers_semi():
    values = [-3.0, -5.0] + [0.0] * 10
    m_total = 100.0
    result = decode_row(values, m_total, prefer="semi")
    assert result == [
        Interval(0.0, 3.0, "semi"),
        Interval(3.0, 5.0, "conv"),
        Interval(5.0, m_total, "semi"),
    ]


def test_unambiguous_conv_start():
    # Worked example 6: sign pattern -,+ forces start=conv (rad is illegal
    # after a leading '-', and conv is the only legal start left after
    # walking the full sequence).
    values = [-3.0, 5.0] + [0.0] * 10
    m_total = 100.0
    result = decode_row(values, m_total)
    assert result == [
        Interval(0.0, 3.0, "conv"),
        Interval(3.0, 5.0, "semi"),
    ]


def test_all_plus_prefers_rad_start():
    # Worked example 7: three '+' boundaries. Both rad and semi starts are
    # legal (an all-'+' sequence never touches conv), but rad is always
    # chosen over semi/conv whenever it is a legal candidate.
    values = [3.0, 5.0, 7.0] + [0.0] * 9
    m_total = 100.0
    result = decode_row(values, m_total)
    assert result == [
        Interval(3.0, 5.0, "semi"),
        Interval(7.0, m_total, "semi"),
    ]


def test_lone_negative_boundary_prefers_conv():
    # Worked example 8:
    values = [-3.0] + [0.0] * 11
    m_total = 100.0
    result = decode_row(values, m_total, prefer="conv")
    assert result == [
        Interval(0.0, 3.0, "conv"),
        Interval(3.0, m_total, "semi"),
    ]


def test_lone_negative_boundary_prefers_semi():
    values = [-3.0] + [0.0] * 11
    m_total = 100.0
    result = decode_row(values, m_total, prefer="semi")
    assert result == [
        Interval(0.0, 3.0, "semi"),
        Interval(3.0, m_total, "conv"),
    ]


def test_illegal_sequence_raises_decode_error():
    # Worked example 9: [3.0, -5.0, 7.0] is illegal from every start state.
    values = [3.0, -5.0, 7.0] + [0.0] * 9
    m_total = 100.0
    with pytest.raises(DecodeError):
        decode_row(values, m_total)


def test_padding_eps_entry_exactly_equal_to_m_total_is_padding():
    m_total = 12.34567
    values = [m_total, -m_total] + [0.0] * 10
    assert decode_row(values, m_total) == []


def test_padding_eps_entry_5e4_below_m_total_is_real_boundary():
    # 5e-4 below M_total is outside eps=1e-4, so it must be treated as a
    # genuine boundary, not padding.
    m_total = 12.34567
    near_surface = m_total - 5e-4
    values = [-near_surface] + [0.0] * 11
    result = decode_row(values, m_total, prefer="conv")
    assert result == [
        Interval(0.0, near_surface, "conv"),
        Interval(near_surface, m_total, "semi"),
    ]


def test_zero_width_intervals_are_dropped():
    # Two boundaries at the same magnitude collapse the middle region.
    values = [0.00064, -0.00064, 9.01792, -8.96118] + [0.0] * 8
    m_total = 20.73143
    result = decode_row(values, m_total)
    # The [0.00064, 0.00064] semi sliver must not appear.
    assert all(iv.hi - iv.lo > 0 for iv in result)
    assert Interval(0.00064, 0.00064, "semi") not in result


def test_output_sorted_by_lo():
    values = [
        5.96927, -5.97847, -7.32142, 7.52217,
        8.41099, -8.76458, -9.85256, 10.10233,
        19.82018, -19.82024, 19.83684, -19.83681,
    ]
    m_total = 19.83848
    result = decode_row(values, m_total)
    los = [iv.lo for iv in result]
    assert los == sorted(los)


def test_interval_is_frozen_and_hashable():
    iv = Interval(1.0, 2.0, "conv")
    with pytest.raises(dataclasses.FrozenInstanceError):
        iv.lo = 5.0
    # Must be hashable (usable in sets / as dict keys).
    {iv: "ok"}
    assert hash(iv) == hash(Interval(1.0, 2.0, "conv"))


def test_decode_error_is_a_value_error():
    assert issubclass(DecodeError, ValueError)


def test_decode_all_skip_collects_bad_rows():
    good = [-3.0, -5.0] + [0.0] * 10
    bad = [3.0, -5.0, 7.0] + [0.0] * 9
    conv = [good, bad, good]
    m = [100.0, 100.0, 100.0]
    intervals, bad_rows = decode_all(conv, m, on_error="skip")
    assert bad_rows == [1]
    assert intervals[1] == []
    assert intervals[0] != []
    assert intervals[2] != []
    assert len(intervals) == 3


def test_decode_all_raise_includes_row_index_in_message():
    good = [-3.0, -5.0] + [0.0] * 10
    bad = [3.0, -5.0, 7.0] + [0.0] * 9
    conv = [good, bad, good]
    m = [100.0, 100.0, 100.0]
    with pytest.raises(DecodeError, match=r"\brow 1\b"):
        decode_all(conv, m, on_error="raise")


def test_decode_all_raise_row_index_matches_offending_row():
    bad = [3.0, -5.0, 7.0] + [0.0] * 9
    good = [-3.0, -5.0] + [0.0] * 10
    conv = [good, good, bad]
    m = [100.0, 100.0, 100.0]
    with pytest.raises(DecodeError, match=r"\brow 2\b"):
        decode_all(conv, m, on_error="raise")


# --- tie handling -----------------------------------------------------------
# STARS prints 5 decimals, so the two sides of a thin semiconvective skin often
# print with identical |m|. Their relative order is then unknowable from the
# file and the decoder must pick whichever order yields a legal walk.


def test_tied_pair_in_illegal_file_order_is_reordered():
    # real row 1000 (model 33210), M=19.84230. File order of the tie is (+, -)
    # which is illegal after a conv state; (-, +) is legal.
    vals = [6.05284, -6.05874, -8.16246, -9.18424, -11.09175, 11.48362,
            19.84084, -19.84085, 19.8422, -19.8422, 19.84229, -19.84229]
    out = decode_row(vals, 19.84230)
    kinds = [(round(i.lo, 5), round(i.hi, 5), i.kind) for i in out]
    assert kinds == [
        (6.05284, 6.05874, "semi"),
        (6.05874, 8.16246, "conv"),
        (8.16246, 9.18424, "semi"),
        (9.18424, 11.09175, "conv"),
        (11.09175, 11.48362, "semi"),
        (19.84084, 19.84085, "semi"),
        (19.84085, 19.8422, "conv"),
    ]


def test_two_tied_pairs_each_resolved_independently():
    # real row 3686 (model 35858), M=17.18638: ties at 1.0087 and 1.03454 need
    # opposite orders ((-,+) then (+,-)) to stay legal.
    vals = [0.86166, -0.86167, 1.0087, -1.0087, 1.03454, -1.03454,
            2.53863, -2.53858, 4.93251, -4.9327, 7.28093, -7.27498]
    out = decode_row(vals, 17.18638)
    conv = [(round(i.lo, 5), round(i.hi, 5)) for i in out if i.kind == "conv"]
    assert conv == [(0.86167, 1.0087), (1.03454, 2.53858), (4.9327, 7.27498)]


def test_near_tie_within_print_precision_is_treated_as_tie():
    # 5.0 and 5.00001 differ by one unit in the 5th decimal: one skin whose
    # printed order is meaningless. In sorted order (+5.0, -5.00001) no start
    # state is legal; swapping the tie makes start=semi legal:
    # semi(0,3) conv(3,5.00001) rad(5,7) semi(7,10).
    out = decode_row([-3.0, 5.0, -5.00001, 7.0], 10.0)
    assert all(i.hi > i.lo for i in out)
    assert [(i.lo, i.hi, i.kind) for i in out] == [
        (0.0, 3.0, "semi"), (3.0, 5.00001, "conv"), (7.0, 10.0, "semi")]


def test_genuinely_illegal_row_still_raises():
    with pytest.raises(DecodeError):
        decode_row([3.0, -5.0, 7.0], 10.0)


# --- truncation: all 12 slots in use -----------------------------------------
# STARS can only report 12 boundaries. Late in the evolution the star has
# more, and the list is cut off at the outside. The region above the last
# reported boundary is then unknown and must not be extended to the surface.


def _twelve_central_shells():
    # six thin conv skins near the centre, all 12 slots used, walk ends conv
    return [0.00608, -0.00341, 0.0183, -0.01831, 0.04689, -0.04687,
            0.15772, -0.15773, 0.23986, -0.23985, 0.35412, -0.35413]


def test_truncated_row_does_not_extend_to_surface():
    out = decode_row(_twelve_central_shells(), 17.18642)
    assert out, "central shells must still be decoded"
    assert max(i.hi for i in out) < 1.0


def test_full_row_that_ends_radiative_is_unaffected():
    vals = [1.0, -1.1, -2.0, 2.1, 3.0, -3.1, -4.0, 4.1, 5.0, -5.1, -6.0, 6.1]
    out = decode_row(vals, 10.0)
    assert [(i.lo, i.hi) for i in out if i.kind == "conv"] == [
        (1.1, 2.0), (3.1, 4.0), (5.1, 6.0)]


def test_non_truncated_row_still_extends_to_surface():
    out = decode_row([3.0, -3.5, 10.0, -10.0, 10.0, -10.0], 10.0)
    assert out[-1] == Interval(3.5, 10.0, "conv")


def test_decode_all_restores_envelope_on_truncated_rows_from_conv_env():
    conv = np.array([_twelve_central_shells(),
                     [3.0, -3.5] + [10.0, -10.0] * 5])
    M = np.array([17.18642, 10.0])
    conv_env = np.array([7.757, 3.0])   # base of convective envelope
    ivs, bad = decode_all(conv, M, conv_env=conv_env)
    assert bad == []
    assert ivs[0][-1] == Interval(7.757, 17.18642, "conv")
    # non-truncated rows trust the boundary columns, not conv_env
    assert ivs[1][-1] == Interval(3.5, 10.0, "conv")


def test_decode_all_ignores_conv_env_when_no_envelope_or_nan():
    conv = np.array([_twelve_central_shells()] * 2)
    M = np.array([17.18642, 17.18642])
    conv_env = np.array([17.18642, np.nan])   # == M means "no envelope"
    ivs, _ = decode_all(conv, M, conv_env=conv_env)
    assert all(max(i.hi for i in r) < 1.0 for r in ivs)


def test_restored_envelope_is_clipped_to_top_of_decoded_intervals():
    # conv_env sits a hair below the last decoded skin (real row 3033):
    # the boundary columns win and the envelope starts where they end.
    row = [0.01865, -0.01865, 0.0187, -0.01872, 0.27064, -0.27058,
           5.38347, -5.38378, 7.07116, -7.0691, 7.75921, -7.75946]
    ivs, _ = decode_all(np.array([row]), np.array([17.18666]),
                        conv_env=np.array([7.75935]))
    assert ivs[0][-1] == Interval(7.75946, 17.18666, "conv")


def test_restore_skips_rows_whose_envelope_was_already_decoded():
    # real row 3500: envelope conv(7.758, 17.155) is in the columns; the
    # truncation only removed the radiative skin above it. Nothing to add.
    row = [5.03828, -5.0383, 6.35398, -6.35212, 6.35411, -6.37025, 7.29116,
           -7.2903, 7.75741, -7.75768, 17.15507, -17.15498]
    ivs, _ = decode_all(np.array([row]), np.array([17.1864]),
                        conv_env=np.array([7.7574]))
    conv = [(round(i.lo, 3), round(i.hi, 3)) for i in ivs[0] if i.kind == "conv"]
    assert conv == [(5.038, 6.352), (6.37, 7.29), (7.758, 17.155)]


# --- unknown regions on truncated rows ---------------------------------------


def test_unknown_regions_none_for_normal_rows():
    from kipp.decode import unknown_regions
    conv = np.array([[3.0, -3.5] + [10.0, -10.0] * 5])
    assert unknown_regions(conv, np.array([10.0])) == [None]


def test_unknown_region_spans_last_boundary_to_surface_when_no_envelope():
    from kipp.decode import unknown_regions
    conv = np.array([_twelve_central_shells()])
    (region,) = unknown_regions(conv, np.array([17.18642]))
    assert region == (0.35413, 17.18642)


def test_unknown_region_stops_at_restored_envelope_base():
    from kipp.decode import unknown_regions
    conv = np.array([_twelve_central_shells()])
    (region,) = unknown_regions(conv, np.array([17.18642]),
                                conv_env=np.array([7.757]))
    assert region == (0.35413, 7.757)


def test_unknown_region_absent_when_envelope_already_decoded():
    from kipp.decode import unknown_regions
    # real row 3500: truncation removed only the thin radiative skin above
    # the envelope; what is missing is above 17.15507.
    row = [5.03828, -5.0383, 6.35398, -6.35212, 6.35411, -6.37025, 7.29116,
           -7.2903, 7.75741, -7.75768, 17.15507, -17.15498]
    (region,) = unknown_regions(np.array([row]), np.array([17.1864]),
                                conv_env=np.array([7.7574]))
    assert region == (17.15507, 17.1864)
