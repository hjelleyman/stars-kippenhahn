"""kipp: shaded Kippenhahn diagrams from Cambridge STARS plot files."""
from kipp.io import load_plot
from kipp.decode import decode_row, decode_all
from kipp.rasterise import rasterise, time_edges
from kipp.render import plot_kippenhahn

__all__ = [
    "load_plot",
    "decode_row",
    "decode_all",
    "rasterise",
    "time_edges",
    "plot_kippenhahn",
]

__version__ = "0.1.0"
