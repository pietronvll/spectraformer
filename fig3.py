import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import xarray as xr

from pathlib import Path
from matplotlib import rcParams

from typing import List, Tuple, Sequence, Optional, Dict, Any, Union
from scipy.optimize import curve_fit
from scipy.special import wofz
from scipy.signal import savgol_filter

rcParams['font.size'] = 24

is_filter = False
filter_window_length = 15  # must be odd
filter_polyorder = 3

# -----------------------------
# Paths and data discovery
# -----------------------------
maindir = Path(__file__).parent.resolve()
unmixed_dir = maindir / "data/unmixed_spatial"

# -----------------------------
# Core plotting logic
# -----------------------------
def plot_spatial_dataset_mean(ds: xr.Dataset, annotations_letters: List[str]):
    fig = plt.figure(figsize=(12.5, 12.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.1)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    wave_number = ds["wave_number"].values

    # Determine reduction dimensions (everything except wave_number)
    reduce_dims = [d for d in ds["spectra"].dims if d != "wave_number"]

    # -------------------------
    # Mask (union over space)
    # -------------------------
    mask_any = ds["mask"].any(dim=reduce_dims).values.astype(bool)

    mask_intervals = []
    start = None
    for i, v in enumerate(mask_any):
        if v and start is None:
            start = i
        elif not v and start is not None:
            mask_intervals.append((start, i - 1))
            start = None
    if start is not None:
        mask_intervals.append((start, len(mask_any) - 1))

    for s, e in mask_intervals:
        for ax in (ax1, ax2):
            ax.axvspan(
                wave_number[s], wave_number[e],
                color="gray", alpha=0.1, linewidth=0
            )

    # -------------------------
    # Top panel
    # -------------------------
    for key, label, color in [
        ("spectra", "Data", "C0"),
        ("predicted_spectra", "Model output", "C1"),
    ]:
        da = ds[key]

        mean = da.mean(dim=reduce_dims).values
        std = da.std(dim=reduce_dims).values

        ax1.plot(wave_number, mean, color=color, lw=1.2, label=label)
        ax1.fill_between(
            wave_number,
            mean - std,
            mean + std,
            color=color,
            alpha=0.4,
            linewidth=0
        )

    ax1.legend(frameon=True, fontsize='small')
    ax1.set_ylabel("Intensity, a.u.")
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(300))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(50))
    # ax1.grid(True)

    # -------------------------
    # Bottom panel
    # -------------------------
    diff = ds["predicted_difference"]
    if is_filter:
        diff = diff.copy()
        diff.values = savgol_filter(
            diff.values,
            window_length=filter_window_length,
            polyorder=filter_polyorder,
            axis=-1,
            mode='interp'
        )

    diff_mean = diff.mean(dim=reduce_dims).values
    diff_std = diff.std(dim=reduce_dims).values

    ax2.plot(wave_number, diff_mean, color="C2", lw=1.2, label="Spectral subtraction")
    ax2.fill_between(
        wave_number,
        diff_mean - diff_std,
        diff_mean + diff_std,
        color="C2",
        alpha=0.4,
        linewidth=0
    )

    ax2.axhline(0, color='k', alpha=0.4)
    ax2.legend(frameon=True, fontsize='small')
    ax2.set_xlabel("Raman shift, cm$^{-1}$")
    ax2.set_ylabel("Intensity, a.u.")
    # ymin_cur, ymax_cur = ax2.get_ylim()
    # ymin_req, ymax_req = -0.08, 0.18

    # if ymin_cur > ymin_req or ymax_cur < ymax_req:
    #     ax2.set_ylim(
    #         min(ymin_cur, ymin_req),
    #         max(ymax_cur, ymax_req)
    #     )
    ax2.set_ylim(-0.08, 0.32)
    # ax2.grid(True)
    annotations = []
    for ax, label in zip([ax1, ax2], annotations_letters):
        annotation = ax.annotate(label, xy=(0.01, 0.99), xycoords='axes fraction', ha='left', va='top', fontsize='large', fontweight='bold')
        annotations.append(annotation)

    fig.align_ylabels([ax1, ax2])
    # plt.tight_layout()

    return fig

if __name__ == "__main__":
    graphene_path = maindir / "data/unmixed_spatial/min70_highf/buffer+graphene/main/unmixed_by_min70_highf_main_8x8_Mixed_8x8.nc"
    buffer_path = maindir / "data/unmixed_spatial/min70_highf/buffer+graphene/RUN3REC2_Buffer_20241011_1/5_5_1/unmixed_by_min70_highf_RUN3REC2_Buffer_20241011_1_11x11_pt1.nc"
    
    graphene_ds = xr.load_dataset(graphene_path)
    buffer_ds = xr.load_dataset(buffer_path)
    
    fig1 = plot_spatial_dataset_mean(graphene_ds, annotations_letters=['(a)', '(c)'])
    fig2 = plot_spatial_dataset_mean(buffer_ds, annotations_letters=['(b)', '(d)'])
    plt.show()
    
    fig1.savefig(maindir / "temp/fig3ac_graphene_lowres.png", dpi=90, transparent=True, bbox_inches='tight')
    fig2.savefig(maindir / "temp/fig3bd_buffer_lowres.png", dpi=90, transparent=True, bbox_inches='tight')