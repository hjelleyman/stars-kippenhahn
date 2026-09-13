# Contributing

Bug reports and pull requests are welcome, especially plot files from other
STARS runs that decode badly — the decoder in `kipp/decode.py` has only been
exercised on a handful of models.

## Development

```bash
git clone https://github.com/hjelleyman/stars-kippenhahn
cd stars-kippenhahn
python -m pip install -e ".[test]"
python -m pytest -q
```

Tests live in `tests/`, one file per module, and were written before the
code they test. Keep that habit: add a failing test, then make it pass.
`tests/test_integration.py` needs `data/plot.STANDARD_SINGLE_DOUBLEPREC`
and skips when it is missing.

If you hit a row that raises `DecodeError`, please open an issue with the
row's 12 `M_conv` values and its total mass — that is all the decoder sees.
