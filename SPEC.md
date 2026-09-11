# kipp — interface contract

Package layout (all under `kipp/`, tests under `tests/`, run with `python -m pytest`):

```
kipp/__init__.py        exports: load_plot, decode_row, decode_all, rasterise, time_edges, plot_kippenhahn
kipp/io.py              load_plot
kipp/decode.py          Interval, decode_row, decode_all, DecodeError
kipp/rasterise.py       time_edges, rasterise
kipp/render.py          plot_kippenhahn
kipp/cli.py             main(argv)
```

Python 3.14, numpy, matplotlib. No scipy, no kaitiaki. Pure functions, no global state,
no `plt.show()` anywhere except the CLI when `-o` is not given.

Data file: `data/plot.STANDARD_SINGLE_DOUBLEPREC` (STARS `plot` file, whitespace
separated, one model per line, 74 columns, BUT line 1206 has 73 columns — loaders
must cope). Column meanings (0-indexed):

| idx   | name      | meaning |
|-------|-----------|---------|
| 0     | model     | model number (int) |
| 1     | age       | age in years |
| 5     | M         | total mass, M☉ |
| 6     | He_core   | H-exhausted core mass |
| 7     | CO_core   | He-exhausted core mass |
| 11–22 | conv1..12 | signed mass coordinates of region boundaries (see decode) |

## io.py

```python
def load_plot(path) -> dict[str, np.ndarray]
```
Returns keys `model, age, M, He_core, CO_core` (1-D float arrays, length n_models)
and `conv` (2-D float array, shape (n_models, 12)). Reads only the first 23 columns
so the ragged line is harmless. Rows must stay in file order. Raise `ValueError`
with the offending line number if a row has fewer than 23 columns.

## decode.py

Physics (STARS manual p.20): each of the 12 values is the mass coordinate of a
boundary between two adjacent regions of the star. Region types: `rad`, `semi`,
`conv`. Only rad↔semi and semi↔conv transitions exist (∇−∇_ad is continuous).
Sign encodes boundary type:

- value < 0  → boundary between `conv` and `semi`
- value > 0  → boundary between `semi` and `rad`

Padding: unused slots hold `0.0` or `±M_total` (the surface). Treat any entry with
`abs(v) < eps` or `abs(v) >= M_total - eps` as padding; default `eps = 1e-4` (padding equals ±M to 5 d.p.; real near-surface boundaries sit ≥ 5e-4 below M).

```python
@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float
    kind: str        # "conv" or "semi"

class DecodeError(ValueError): ...

def decode_row(values, m_total, *, eps=1e-4, prefer="conv") -> list[Interval]
```
Algorithm:
1. Drop padding, take `(abs(v), sign(v))`, sort by abs.
2. Walk from the centre. State ∈ {rad, semi, conv}. A `+` boundary flips
   rad↔semi, a `−` boundary flips semi↔conv. A `+` seen while in `conv`, or a
   `−` seen while in `rad`, is illegal.
3. Choose the start state: the set of states in {rad, semi, conv} for which the
   whole walk is legal. If exactly one → use it. If several → use `prefer`
   ("conv" wins over "semi" wins over "rad"). If none → raise `DecodeError`.
4. Emit one `Interval` per non-`rad` region: from the boundary where it started
   (0.0 for the centre) to the boundary where it ended (`m_total` for the
   surface if the walk ends in a non-rad state). Skip zero-width intervals
   (hi - lo <= 0).
5. Return intervals sorted by `lo`.

Worked examples (must be tests):

- `[0.00064, -0.00064, 9.01792, -8.96118, 20.73, -20.73, 20.73, -20.73, 20.73, -20.73, 20.73, -20.73]`, M=20.73143
  → `[Interval(0.00064, 8.96118, "conv"), Interval(8.96118, 9.01792, "semi")]`
  (the semi interval [0.00064, 0.00064] is zero-width and dropped; the centre
  region [0, 0.00064] is rad because the first boundary is `+` and start must be rad)
- `[2.19412, 3.11245, 19.85237, -19.85237, ...pad..., 0, 0]`, M=19.8524
  → `[Interval(2.19412, 3.11245, "semi")]`
- `[5.96927, -5.97847, -7.32142, 7.52217, 8.41099, -8.76458, -9.85256, 10.10233, 19.82018, -19.82024, 19.83684, -19.83681]`, M=19.83848
  → semi(5.96927,5.97847), conv(5.97847,7.32142), semi(7.32142,7.52217),
    semi(8.41099,8.76458), conv(8.76458,9.85256), semi(9.85256,10.10233),
    semi(19.82018,19.82024), conv(19.82024,19.83681), semi(19.83681,19.83684)
    (none of the 19.8x entries are padding: the closest is 0.0016 below M, well
    above eps=1e-4. The outermost 0.0016 M☉ is radiative.)
- all-padding row → `[]`
- `[-3.0, -5.0, pad...]` (ambiguous: start conv or semi) with prefer="conv"
  → `[conv(0,3), semi(3,5)]`; with prefer="semi" → `[semi(0,3), conv(3,5)]`
- `[-3.0, 5.0, pad...]` → start must be conv: `[conv(0,3), semi(3,5)]`
- `[3.0, 5.0, 7.0, pad...]`: start rad → semi(3,5), rad(5,7), semi(7, M) →
  `[semi(3,5), semi(7,M)]`
- `[-3.0, pad...]` alone: legal starts conv (→semi to surface) and semi (→conv to
  surface); prefer conv → `[conv(0,3), semi(3,M)]`
- illegal: `[3.0, -5.0, 7.0]`? start rad: +→semi, −→conv, +→ILLEGAL. start semi:
  +→rad, −→ILLEGAL. start conv: + ILLEGAL. → raise DecodeError.

```python
def decode_all(conv, M, *, eps=1e-4, prefer="conv", on_error="skip") -> tuple[list[list[Interval]], list[int]]
```
Vectorised-enough loop over rows. `on_error="skip"` → rows that raise get `[]`
and their row index is appended to the returned `bad_rows` list; `"raise"` →
re-raise with the row index in the message.

## rasterise.py

```python
def time_edges(x) -> np.ndarray
```
Given 1-D cell-centre coordinates (strictly increasing or strictly decreasing,
length n ≥ 2), return n+1 edges: midpoints between neighbours, extrapolated
half a step at both ends. n == 1 → `[x-0.5, x+0.5]`.

```python
def rasterise(intervals_per_model, m_max, n_mass=800) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```
Returns `(m_edges, conv, semi)` with `m_edges` of length n_mass+1 spanning
`[0, m_max]`, and `conv`, `semi` boolean arrays shape `(n_models, n_mass)`. A cell
is marked if its centre lies in `[lo, hi)` of an interval of that kind. Use
`np.searchsorted` on cell centres per interval; do not loop over cells. A cell
can be both conv and semi only if intervals overlap, which decode never produces,
but rasterise must not assume that.

## render.py

```python
def plot_kippenhahn(data, *, xaxis="model", n_mass=800, semiconv=True,
                    show_dots=False, ax=None, conv_color="#1f4fd8",
                    semi_color="#9ab4f0", title=None) -> matplotlib.axes.Axes
```
`data` is the dict from `load_plot`. `xaxis` ∈ {"model", "index", "age",
"collapse"} where collapse = log10(t_end − age + dt_floor), dt_floor = the smallest
positive timestep so the last model doesn't hit log(0); for "collapse" the x axis
must be inverted (time-to-collapse decreasing to the right). Layers, bottom to
top: semi (pcolormesh, masked), conv (pcolormesh, masked), CO core fill (light
grey, 0→CO_core), He core line, total mass line (black). `show_dots=True` draws
`abs(conv)` values as small grey dots (the legacy Kaitiaki view) for validation.
Legend via proxy artists. Uses `ax` if given else creates a figure. No
`plt.show()`. Must work with the `Agg` backend.

## cli.py

```
python -m kipp PLOTFILE [-o OUT.png] [--xaxis model|index|age|collapse]
                        [--n-mass N] [--no-semiconv] [--dots] [--dpi D]
                        [--dump-intervals OUT.json]
```
`-o` → savefig, no show. No `-o` → plt.show(). `--dump-intervals` writes
`[{"model": int, "intervals": [[lo, hi, kind], ...]}, ...]`. Print a one-line
summary: n_models, n_bad_rows, list of bad model numbers (first 20).

## Testing conventions

- pytest, one test file per module, tests written BEFORE implementation.
- Tests must not depend on the real data file except in `tests/test_integration.py`,
  which is allowed to load `data/plot.STANDARD_SINGLE_DOUBLEPREC` and assert
  physics-level facts (ZAMS convective core top between 8.5 and 9.5 M☉, zero bad
  rows or a documented small number, etc.).
- Render tests use `matplotlib.use("Agg")` and assert on artists / saved PNG
  existence, not on pixels.
