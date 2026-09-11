import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import label, binary_closing, binary_fill_holes, sum as ndi_sum
import kaitiaki

plt.style.use('krytic')

direc = "."
plot_file = f"{direc}/plot.STANDARD_SINGLE_DOUBLEPREC"


# --------------------------------------------------------------------------
# Why this needed a rewrite, not just a tweak
# --------------------------------------------------------------------------
# The M_conv1..12 columns are six *slots*, not six persistent zone
# identities. The code fills them in whatever order it currently finds
# convective regions -- it does not promise that "slot 3" refers to the
# same physical zone from one saved model to the next. When zones split,
# merge, appear, or disappear, the zone that used to live in slot 1 can
# reappear in slot 3 at the very next saved model.
#
# Because of that, treating each slot as a continuous (bottom(t), top(t))
# time series and handing it to fill_between (as both the original script
# and my first fix did) is unsound: whenever a zone's slot changes,
# fill_between happily draws a straight line connecting two boundary
# values that have nothing to do with each other, producing a shape whose
# extent at time t depends on bookkeeping, not physics.
#
# The fix is to stop trusting slot identity entirely:
#   1. Rasterise "is this mass shell convective" as a plain boolean field,
#      taking the union over all six slots at each timestep. This throws
#      away slot identity and just asks a physically meaningful question.
#   2. Use connected-component labelling (scipy.ndimage.label) on the
#      resulting 2D (model x mass) image to recover the actual contiguous
#      regions. This is the "clustering / image recognition" step: a
#      region's true shape depends on its neighbours in time as well as
#      mass, which is exactly what connected-component labelling on an
#      image gives you for free.
#   3. Morphologically close single-timestep numerical dropouts (a zone
#      that is physically continuous but happens to read as exactly zero
#      for one saved model) and discard components too small to be a real
#      sustained feature (single-pixel numerical noise).
#   4. Render with pcolormesh using the *actual* (non-uniform) ages as
#      cell edges, so STARS' adaptive timestepping doesn't distort the
#      picture the way imshow-with-linear-extent did in the very first
#      version of this script.


def rasterize_convective(plot_obj, n_mass=400):
    """
    Build a boolean (n_models, n_mass) grid: True wherever the given mass
    shell is inside *any* of the six M_conv slot pairs at that timestep.

    Slot identity is deliberately ignored here -- see module docstring.
    """
    mass = np.asarray(plot_obj.get('M'), dtype=float)
    n_models = len(mass)
    max_mass = np.nanmax(mass)

    mass_edges = np.linspace(0, max_mass * 1.001, n_mass + 1)
    mass_centres = 0.5 * (mass_edges[:-1] + mass_edges[1:])

    grid = np.zeros((n_models, n_mass), dtype=bool)

    for k in range(1, 13, 2):
        bottom = np.abs(np.asarray(plot_obj.get(f'M_conv{k}'), dtype=float))
        top = np.abs(np.asarray(plot_obj.get(f'M_conv{k + 1}'), dtype=float))

        valid = np.isfinite(bottom) & np.isfinite(top) & (top > bottom)
        # A (0, 0) pair means "no zone in this slot" -- (0,0) also
        # satisfies top>bottom being False already, but guard explicitly
        # in case of floating point noise right at the boundary.
        valid &= ~((bottom <= 1e-10) & (top <= 1e-10))

        rows = np.nonzero(valid)[0]
        for j in rows:
            grid[j, (mass_centres >= bottom[j]) & (mass_centres <= top[j])] = True

    return grid, mass_edges


def clean_and_label(grid, close_gap=1, min_pixels=6, fill_holes=True):
    """
    Morphologically clean the raw boolean grid and return a labelled image
    (0 = not convective, 1..N = distinct connected convective regions).

    close_gap: size of the structuring element used for binary_closing,
        which bridges over short (numerical-glitch) dropouts where a
        genuinely continuous zone briefly reads as absent for one or two
        saved models. Set to 0 to disable.
    min_pixels: connected components smaller than this (in total raster
        cell count) are discarded as noise rather than real, sustained
        convective regions.
    fill_holes: fill any fully-enclosed gaps inside a region (e.g. a
        single stray False cell surrounded by True on all sides).
    """
    cleaned = grid.copy()

    if close_gap > 0:
        # 8-connected structuring element sized to bridge `close_gap`
        # consecutive missing timesteps without merging genuinely distinct,
        # well-separated regions.
        size = 2 * close_gap + 1
        structure = np.ones((size, size), dtype=bool)
        cleaned = binary_closing(cleaned, structure=structure)

    if fill_holes:
        cleaned = binary_fill_holes(cleaned)

    structure8 = np.ones((3, 3), dtype=bool)
    labelled, n_features = label(cleaned, structure=structure8)

    if n_features > 0 and min_pixels > 0:
        sizes = ndi_sum(np.ones_like(labelled), labelled, index=np.arange(1, n_features + 1))
        keep = np.zeros(n_features + 1, dtype=bool)
        keep[0] = False
        keep[1:] = sizes >= min_pixels
        labelled = np.where(keep[labelled], labelled, 0)
        # relabel to keep IDs contiguous after dropping small blobs
        labelled, n_features = label(labelled > 0, structure=structure8)

    return labelled, n_features


def build_time_edges(x):
    """Cell edges for pcolormesh from an array of cell-centre x-values,
    correctly handling non-uniform spacing (STARS' adaptive timestep)."""
    x = np.asarray(x, dtype=float)
    edges = np.empty(len(x) + 1)
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0] - (edges[1] - x[0])
    edges[-1] = x[-1] + (x[-1] - edges[-2])
    return edges


def plot_kippenhahn(plot_file, outpath=None, xaxis='age',
                     n_mass=400, close_gap=1, min_pixels=6):
    plot = kaitiaki.file.plot(plot_file)

    dt = np.asarray(plot.get('timestep'), dtype=float)
    age = np.asarray(plot.get('age'), dtype=float)
    mass = np.asarray(plot.get('M'), dtype=float)
    He_mass = np.asarray(plot.get('He_core'), dtype=float)
    CO_mass = np.asarray(plot.get('CO_core'), dtype=float)

    if xaxis == 'model':
        x = dt  # np.arange(1, len(age) + 1, dtype=float)
        xlabel = "Model number"
    else:
        x = age
        xlabel = "Age (yr)"

    x_edges = build_time_edges(x)

    grid, mass_edges = rasterize_convective(plot, n_mass=n_mass)
    labelled, n_regions = clean_and_label(grid, close_gap=close_gap, min_pixels=min_pixels)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x, mass, color='black', lw=1.3, zorder=5, label='Total mass')

    ax.fill_between(x, CO_mass, He_mass,
                     color='red', lw=0, zorder=3,
                     label='He core', alpha=0.3)

    ax.fill_between(x, 0, CO_mass,
                     color='green', lw=0, zorder=3,
                     label='CO core', alpha=0.3)

    # Shade every retained connected convective region. All regions get
    # the same colour/alpha (standard Kippenhahn convention) -- the
    # labelling is used to clean the mask, not to colour-code zones. To
    # colour-code instead, swap the two lines below for a discrete
    # colormap keyed on `labelled`.
    mask = labelled > 0
    if mask.any():
        masked = np.ma.masked_where(~mask.T, mask.T.astype(float))
        cmap = ListedColormap(['0.35'])
        ax.pcolormesh(x_edges, mass_edges, masked, cmap=cmap,
                       alpha=0.45, shading='flat', zorder=2)
        # proxy artist so this shows up in the legend
        ax.fill_between([], [], [], color='0.35', alpha=0.45, label='Convective region')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Mass coordinate (M$_\odot$)")
    ax.set_ylim(0, mass.max() * 1.05)
    ax.set_xlim(x_edges.min(), x_edges.max())
    ax.set_title("Kippenhahn Diagram: Convective Zones Shaded")
    ax.legend(loc='upper right')

    fig.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=200)
    plt.show()

    return fig, ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("plotfile", nargs="?", default=plot_file)
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--xaxis", choices=["age", "model"], default="age")
    parser.add_argument("--n-mass", type=int, default=400,
                         help="Mass-grid resolution for rasterisation (default: 400).")
    parser.add_argument("--close-gap", type=int, default=1,
                         help="Bridge dropouts up to this many timesteps long (default: 1; 0 disables).")
    parser.add_argument("--min-pixels", type=int, default=6,
                         help="Discard connected regions smaller than this many raster cells (default: 6).")
    args = parser.parse_args()

    plot_kippenhahn(args.plotfile, outpath=args.output, xaxis=args.xaxis,
                     n_mass=args.n_mass, close_gap=args.close_gap,
                     min_pixels=args.min_pixels)