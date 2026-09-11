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

__all__ = ["Interval", "DecodeError", "decode_row", "decode_all"]

_STATES = ("rad", "semi", "conv")


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
    pairs: list[tuple[float, int]] = []
    for raw in values:
        v = float(raw)
        av = abs(v)
        if av < eps or av >= m_total - eps:
            continue
        pairs.append((av, 1 if v > 0 else -1))
    pairs.sort(key=lambda p: p[0])
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
    eps: float = 1e-4,
    prefer: str = "conv",
    on_error: str = "skip",
) -> tuple[list[list[Interval]], list[int]]:
    """Decode every row of `conv` (shape (n_models, 12)) against the matching
    total mass in `M` (length n_models).

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
        try:
            results.append(decode_row(row, float(m_total), eps=eps, prefer=prefer))
        except DecodeError as exc:
            if on_error == "raise":
                raise DecodeError(f"row {i}: {exc}") from exc
            results.append([])
            bad_rows.append(i)
    return results, bad_rows
