"""Tests for kipp.decode, written before the implementation (TDD).

Every worked example from SPEC.md's "decode.py" section is encoded here as a
test, plus additional coverage for padding-boundary precision, zero-width
interval dropping, sort order, Interval immutability, and decode_all's
error-handling contract.

Note: the third worked example in SPEC.md ("post-MS row") contains an
arithmetic slip about which entries count as padding. The corrected version
(none of the 19.8x entries are padding at eps=1e-4) is what's encoded below;
see the comment on test_post_ms_row for the full derivation.
"""
import dataclasses

import pytest

from kipp.decode import Interval, DecodeError, decode_row, decode_all


def test_zams_row():
    # SPEC.md example 1.
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
    # SPEC.md example 2: purely two real boundaries, rest padding.
    values = [2.19412, 3.11245, 19.85237, -19.85237, 0, 0, 0, 0, 0, 0, 0, 0]
    m_total = 19.8524
    result = decode_row(values, m_total)
    assert result == [Interval(2.19412, 3.11245, "semi")]


def test_post_ms_row():
    # SPEC.md example 3, corrected: with eps=1e-4 and M=19.83848, NONE of the
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
    # SPEC.md example 5.
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
    # SPEC.md example 6: sign pattern -,+ forces start=conv (rad is illegal
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
    # SPEC.md example 7: three '+' boundaries. Both rad and semi starts are
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
    # SPEC.md example 8.
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
    # SPEC.md example 9: [3.0, -5.0, 7.0] is illegal from every start state.
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
