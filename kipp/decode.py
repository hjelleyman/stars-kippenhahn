"""Decode STARS `plot`-file boundary columns into mass-coordinate intervals.

Physics (STARS manual p.20): each of the 12 `conv1..12` values is the mass
coordinate of a boundary between two adjacent regions of a star. Only
rad<->semi and semi<->conv transitions occur (since grad - grad_ad is
continuous), so a signed sequence of boundary values can be walked, from the
centre outward, as a small state machine:

- a positive value is a rad/semi boundary (toggles between rad and semi)
- a negative value is a conv/semi boundary (toggles between conv and semi)

Unused slots are padded with 0.0 or +/-M_total (the surface); both are
dropped before decoding.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Iterable, Sequence

__all__ = ["Interval", "DecodeError", "decode_row", "decode_all", "is_truncated", "unknown_regions"]

_STATES = ("rad", "semi", "conv")
_SLOTS = 12


def _boundaries(values: Iterable[float], m_total: float, eps: float) -> list[tuple[float, int]]:
    """(|m|, sign) for every non-padding entry, sorted by |m|. Padding is
    0.0 or +/-m_total (within eps)."""
    pairs: list[tuple[float, int]] = []
    for raw in values:
        v = float(raw)
        av = abs(v)
        if av < eps or av >= m_total - eps:
            continue
        pairs.append((av, 1 if v > 0 else -1))
    pairs.sort(key=lambda p: p[0])
    return pairs


def is_truncated(values: Iterable[float], m_total: float, eps: float = 1e-4) -> bool:
    """True if every one of the 12 slots holds a real boundary, i.e. the
    star has more boundaries than STARS can report and the outer ones are
    missing."""
    return len(_boundaries(values, m_total, eps)) == _SLOTS


@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float
    kind: str  # "conv" or "semi"


class DecodeError(ValueError):
    """Raised when a row's boundary values admit no legal state walk."""


def _state_priority(prefer: str) -> list[str]:
    """Return start-state candidates in preference order.

    `rad` is always tried first when it is a legal start (a radiative
    centre/exterior is the least remarkable configuration); `prefer` then
    breaks any remaining tie between `conv` and `semi`.
    """
    order: list[str] = []
    for state in ("rad", prefer, "conv", "semi"):
        if state in _STATES and state not in order:
            order.append(state)
    return order


def _tie_groups(pairs: Sequence[tuple[float, int]], tol: float) -> list[list[tuple[float, int]]]:
    """Split abs-sorted (|m|, sign) pairs into runs whose |m| agree within
    `tol`. Values inside a run are the two sides of one thin skin printed at
    the same precision; their relative order is not knowable from the file.
    """
    groups: list[list[tuple[float, int]]] = []
    for pair in pairs:
        if groups and pair[0] - groups[-1][-1][0] <= tol:
            groups[-1].append(pair)
        else:
            groups.append([pair])
    return groups


def _step(state: str, sign: int) -> str | None:
    """Apply one boundary to `state`; None if the transition is illegal."""
    if sign > 0:
        return None if state == "conv" else ("semi" if state == "rad" else "rad")
    return None if state == "rad" else ("conv" if state == "semi" else "semi")


def _search(
    state: str,
    groups: Sequence[Sequence[tuple[float, int]]],
    gi: int,
) -> list[tuple[float, int, str]] | None:
    """Depth-first search for a legal ordering of every tie group from group
    `gi` onward. Returns [(|m|, sign, state_after)] or None.
    """
    if gi == len(groups):
        return []
    for order in permutations(groups[gi]):
        st = state
        prefix: list[tuple[float, int, str]] = []
        for av, sign in order:
            nxt = _step(st, sign)
            if nxt is None:
                break
            st = nxt
            prefix.append((av, sign, st))
        else:
            rest = _search(st, groups, gi + 1)
            if rest is not None:
                return prefix + rest
    return None


def decode_row(
    values: Iterable[float],
    m_total: float,
    *,
    eps: float = 1e-4,
    prefer: str = "conv",
    tie_tol: float = 1.5e-5,
) -> list[Interval]:
    """Decode one row's boundary values into a list of non-radiative
    Intervals, sorted by `lo`.

    Boundaries whose |m| agree within `tie_tol` (one unit in STARS' 5-decimal
    output) are treated as unordered: the decoder picks whichever order of
    each tied group yields a legal walk.

    Raises DecodeError if no start state (rad/semi/conv) and no tie ordering
    produces a fully legal walk through the non-padding boundary values.
    """
    pairs = _boundaries(values, m_total, eps)
    # STARS reports at most 12 boundaries. When every slot is in use the
    # list has been cut off at the outside, so the region above the last
    # boundary is unknown rather than "whatever state the walk ends in".
    truncated = len(pairs) == _SLOTS
    groups = _tie_groups(pairs, tie_tol)

    walked = None
    for start in _state_priority(prefer):
        walked = _search(start, groups, 0)
        if walked is not None:
            break
    if walked is None:
        raise DecodeError(
            f"no legal start state (rad/semi/conv) for boundaries {values!r}"
        )

    states = [start] + [st for _, _, st in walked]
    boundaries = [0.0] + [av for av, _, _ in walked] + [m_total]
    if truncated:
        states = states[:-1]

    intervals: list[Interval] = []
    for i, state in enumerate(states):
        lo, hi = boundaries[i], boundaries[i + 1]
        if state != "rad" and hi - lo > 0:
            intervals.append(Interval(lo, hi, state))

    intervals.sort(key=lambda iv: iv.lo)
    return intervals


def decode_all(
    conv,
    M,
    *,
    conv_env=None,
    eps: float = 1e-4,
    prefer: str = "conv",
    on_error: str = "skip",
) -> tuple[list[list[Interval]], list[int]]:
    """Decode every row of `conv` (shape (n_models, 12)) against the matching
    total mass in `M` (length n_models).

    `conv_env` (optional, length n_models): mass coordinate of the base of
    the convective envelope, from the plot file's M_conv-env column. It is
    used only for truncated rows (all 12 slots in use), where the envelope
    boundaries have fallen off the end of the list; a finite value below
    `M - eps` restores the envelope as a conv interval up to the surface.
    Non-truncated rows trust the boundary columns alone.

    `on_error="skip"`: a row that fails to decode contributes `[]` to the
    result and its index is appended to `bad_rows`.
    `on_error="raise"`: re-raise DecodeError with the row index in the
    message.
    """
    if on_error not in ("skip", "raise"):
        raise ValueError(f"on_error must be 'skip' or 'raise', got {on_error!r}")

    results: list[list[Interval]] = []
    bad_rows: list[int] = []
    for i, (row, m_total) in enumerate(zip(conv, M)):
        m_total = float(m_total)
        try:
            intervals = decode_row(row, m_total, eps=eps, prefer=prefer)
        except DecodeError as exc:
            if on_error == "raise":
                raise DecodeError(f"row {i}: {exc}") from exc
            results.append([])
            bad_rows.append(i)
            continue
        if conv_env is not None and is_truncated(row, m_total, eps):
            intervals = _restore_envelope(intervals, float(conv_env[i]), m_total, eps)
        results.append(intervals)
    return results, bad_rows


def _restore_envelope(
    intervals: list[Interval], base: float, m_total: float, eps: float,
    tol: float = 0.5,
) -> list[Interval]:
    """Append a conv interval [base, m_total) if the envelope whose base is
    `base` fell off the end of the truncated list. If a decoded conv
    interval already starts at or above `base - tol`, the envelope is
    present and nothing is added. The boundary columns are authoritative
    where they exist, so a restored envelope starts no lower than the last
    decoded interval."""
    if not (base == base):   # nan: column unavailable
        return intervals
    if any(iv.kind == "conv" and iv.lo >= base - tol for iv in intervals):
        return intervals
    base = max(base, max((iv.hi for iv in intervals), default=0.0))
    if base >= m_total - eps:   # "no envelope" (STARS writes base == M)
        return intervals
    return intervals + [Interval(base, m_total, "conv")]


def unknown_regions(
    conv, M, *, conv_env=None, eps: float = 1e-4
) -> list[tuple[float, float] | None]:
    """For each row, the mass range whose structure is *not in the file*:
    None for normal rows; for truncated rows (all 12 slots used) the range
    from the outermost reported boundary up to the restored envelope base
    (when `conv_env` supplies one below the surface and above that boundary)
    or otherwise the surface.
    """
    out: list[tuple[float, float] | None] = []
    for i, (row, m_total) in enumerate(zip(conv, M)):
        m_total = float(m_total)
        pairs = _boundaries(row, m_total, eps)
        if len(pairs) < _SLOTS:
            out.append(None)
            continue
        lo = pairs[-1][0]
        hi = m_total
        if conv_env is not None:
            base = float(conv_env[i])
            if base == base and lo < base < m_total - eps:
                hi = base
        out.append((lo, hi))
    return out
