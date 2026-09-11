# Plan: Shaded Kippenhahn diagram from STARS `plot` files

Goal: turn the `M_conv1..12` boundary columns of a STARS plot file into filled
convective regions on a (time × mass) diagram, matching the reference figure
(blue-shaded zones, total mass / He core / CO core lines on top).

Core idea: **don't reconstruct shapes**. Decode each timestep into a list of
convective mass intervals, paint those as vertical strips on a raster, and let
the shapes emerge. Zone identity across time is never needed.

Working inputs:
- `~/Downloads/plot.STANDARD_SINGLE_DOUBLEPREC` — 20 M☉ star, 3704 models, 74 cols
- `~/Downloads/Claude.py` — previous attempt (correct approach, broken decoding)
- Kaitiaki repo (cloned to `./kaitiaki`) — only needed for the column map

---

## Phase 0 — Decode the boundary columns (30 min)

**Resolved by the STARS manual (stars.pdf p.20–21).** Columns 12–23 are the
mass co-ordinates of region boundaries. Three region types exist —
radiative, semiconvective (0 < ∇−∇_ad < DR, DR set in `data`), convective —
and the sign says which boundary it is:

- negative → convective ↔ semiconvective
- positive (no sign) → (semi)convective ↔ radiative

Because ∇−∇_ad is continuous the only transitions are rad ↔ semi ↔ conv.
This makes the decoder a deterministic state machine, not a heuristic.

0.1 **Load with plain numpy**, `usecols=range(23)` (row 1206 is ragged: 73
    cols instead of 74). Columns 0-indexed: 0 model#, 1 age, 5 M, 6 He_core,
    7 CO_core, 11..22 M_conv1..12.

0.2 **Clean each row**: drop padding — entries with |m| ≥ M_total − ε
    (surface) and 0.0 entries. Keep (|m|, sign) for the rest, sort by |m|.

0.3 **State machine**, walking outward from the centre:
    - `+` flips rad ↔ semi, `−` flips semi ↔ conv.
    - Start state: try `rad`, `semi`, `conv`; keep the one(s) for which every
      transition is legal (a `+` is illegal from `conv`, a `−` from `rad`).
      In practice this is unique except for all-`−` rows, where prefer `conv`
      at the centre (physical prior). Log any row with zero or >1 legal
      starts as a diagnostic — these are the rows to eyeball.
    - Emit intervals `[(m_lo, m_hi, kind)]`, kind ∈ {semi, conv}.

0.4 **Verify against physics** on ~12 hand-picked models (ZAMS → TAMS → He
    burning → C burning → end). Already checked by hand and correct:
    - ZAMS `+0.0006 −0.0006 −8.96 +9.02` → conv 0–8.96, semi skin to 9.02
    - TAMS `+2.19 +3.11` → purely semiconvective remnant
    - post-MS `+5.97 −5.98 −7.32 +7.52 +8.41 −8.77 −9.85 +10.10` → two
      convective shells with semi skins, radiative between
    Expectations for the rest: He-burning core 0→3–5 M☉ inside the He core;
    convective envelope during the mass-loss phase; cluster of thin shell
    zones near the centre after model ~3000.

0.5 **Semiconvection handling**: two masks fall out for free. Render conv =
    solid, semi = lighter, `--no-semiconv` flag to merge into one colour like
    the reference figure.

Deliverable: `decode.py` with `decode_row(row, M_total) -> list[(lo, hi,
kind)]` plus a diagnostic dump printing the decoded layering for the
hand-picked models and any rows the state machine flagged.

---

## Phase 1 — Rasterise (30 min)

1.1 Mass grid: `n_mass` cells (default 800) from 0 to max(M)·1.001. Use cell
    *edges*, mark a cell convective if its centre lies inside any interval.
    Vectorise: for each interval, `np.searchsorted` on the edges → slice
    assignment. No Python loop over cells.

1.2 Time axis options (x): model index, model number, age, or
    log10(time-to-collapse) like the reference. Build cell edges as midpoints
    between consecutive x values so the adaptive timestep doesn't distort
    strip widths. For log-time-to-collapse, compute `t_end − age` with a
    floor so the last model doesn't hit log(0).

1.3 Two boolean grids: `strict[n_models, n_mass]`, `extended[...]`.

1.4 Sanity metrics printed on run: fraction of models with ≥1 zone, max zone
    count per model, total convective mass at ZAMS (expect ≈ 9 M☉).

Deliverable: `rasterise(rows, n_mass, xaxis) -> (x_edges, m_edges, strict,
extended)`.

---

## Phase 2 — Render (30 min)

2.1 `pcolormesh(x_edges, m_edges, mask.T)` with a masked array so
    non-convective cells are transparent. `shading='flat'`, `rasterized=True`
    (3700 × 800 cells is fine for PNG; matters for PDF export).

2.2 Layer order: extended (light) → strict (solid) → He-core / CO-core fills
    or lines → total mass line on top. Mirror the reference: blue zones, grey
    band for the CO core region, thin black total-mass line.

2.3 Axes: mass in M☉ on y; x per option chosen. Title with M_ZAMS, Z if
    available. Legend via proxy artists.

2.4 CLI: `kipp.py PLOTFILE [-o out.png] [--xaxis model|age|collapse]
    [--n-mass N] [--no-semiconv] [--dpi]`. Never call `plt.show()` when `-o`
    given (headless-safe). Drop `plt.style.use('krytic')` — not portable.

Deliverable: `kipp.py`, PNG output for the sample file.

---

## Phase 3 — Validate against physics and the reference (30 min)

3.1 Overlay check: draw the old Kaitiaki-style |M_conv| dots on top of the
    shading at low alpha. Every dot should sit on the edge of a shaded region.
    Any dot floating in white or buried inside solid blue means the decoding
    is wrong for that row → go back to Phase 0 with those rows as test cases.

3.2 Feature checklist against the screenshot / reference:
    - [ ] ZAMS core ~9 M☉ shrinking to ~5.8 by model ~500
    - [ ] core detaches from centre and pinches out at ~model 620
    - [ ] intermediate zone 6–13 M☉ around models 700–1500 (the boomerang)
    - [ ] envelope zone reaching down to ~8 M☉ during the mass-loss phase
    - [ ] He-burning core inside the He core (green fill) from model ~2200
    - [ ] cluster of thin shell zones near the centre after model ~3000
    - [ ] no zones above the total-mass line, none below 0

3.3 Robustness pass on edge cases in the data: zero-padded vs
    ±M-padded rows, the ragged line, single-model zone dropouts (should just
    appear as a one-strip gap — do *not* morphologically close by default;
    offer `--close-gap N` as an opt-in cosmetic only).

Deliverable: annotated comparison PNG (ours vs reference side by side).

---

## Phase 4 — Package for your friend (20 min)

4.1 Make it a drop-in for Kaitiaki: a function taking a `kaitiaki.file.plot`
    object (uses `.get('M_conv1')` etc.) *and* a numpy fallback for people
    without Kaitiaki. Keep the two paths sharing the same decode/rasterise
    core.

4.2 Optional: `zones(rows) -> list of intervals per model` exported as JSON /
    CSV, since "a list of shapes" was the original ask — even if the plot
    doesn't need it, they may want it for analysis (e.g. zone lifetimes).

4.3 Short README: what the sign convention turned out to be (with the
    evidence rows), the CLI, and a note on why the slot-tracking approach
    can't work.

4.4 Send: `kipp.py`, `decode.py`, README, example PNG. Offer to open a PR to
    Kaitiaki if they want it upstream.

---

## Fallback if the state machine flags many rows

The convention is documented, so a per-row decode should be exact. If the
diagnostic in 0.3 flags a meaningful fraction of rows (no legal start state,
or transitions that violate rad↔semi↔conv), that points to something
version-specific in this STARS build. Then: treat flagged rows as missing
and let neighbouring strips carry the picture (a one-model gap is invisible
at 3700 models), and report the flagged model numbers so your friend can
check them against `out`. Only if flags cluster in a whole phase do we fall
back to the Claude.py connected-components approach on a conservative mask.

## Explicitly out of scope

- True polygon/shape reconstruction from the point cloud. Not needed for the
  figure and not needed for zone statistics either (intervals per model give
  those directly).
- Zone tracking / identity across time (splits, merges). Can be layered on
  later via `scipy.ndimage.label` on the raster if they ever want it.

## Rough total: ~2 hours. The convention is documented, so the remaining
risk is cosmetic (matching the reference look), not correctness.
