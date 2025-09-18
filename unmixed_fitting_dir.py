#!/usr/bin/env python3
"""
Voigt multi-peak fitting for many spectra with parallel workers.
Improvements:
 - safer parameter bounds and defaults
 - robust handling of failed fits
 - logging instead of silent excepts
 - control over number of workers ( detects cluster env vars )
 - chunksize for executor.map to reduce overhead
"""

from __future__ import annotations
import logging
from pathlib import Path
import os
from typing import List, Tuple, Sequence, Optional

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import wofz
import xarray as xr
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Optional: if available, psutil gives physical core count
try:
    import psutil
except Exception:
    psutil = None

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def detect_cpu_counts() -> dict:
    """Return useful CPU counts and relevant env vars for cluster environments."""
    logical = os.cpu_count() or 1
    physical = psutil.cpu_count(logical=False) if psutil is not None else None
    slurm = os.environ.get("SLURM_CPUS_ON_NODE")
    omp = os.environ.get("OMP_NUM_THREADS")
    pbs = os.environ.get("PBS_NUM_PPN")
    return {"logical": logical, "physical": physical, "SLURM_CPUS_ON_NODE": slurm, "OMP_NUM_THREADS": omp, "PBS_NUM_PPN": pbs}


def choose_workers(n_tasks: int, prefer_env: bool = True) -> int:
    """
    Choose a safe number of worker processes.
    prefer_env: if True, prefer scheduler env vars (e.g. SLURM) when present.
    """
    info = detect_cpu_counts()
    if prefer_env:
        for key in ("SLURM_CPUS_ON_NODE", "OMP_NUM_THREADS", "PBS_NUM_PPN"):
            val = info.get(key)
            if val:
                try:
                    n = max(1, min(n_tasks, int(val)))
                    logger.info(f"Choosing {n} workers from env var {key}={val}")
                    return n
                except Exception:
                    pass
    # fallback
    logical = info["logical"]
    n = max(1, min(n_tasks, logical))
    logger.info(f"Choosing {n} workers from logical CPU count ({logical})")
    return n


def voigt_profile(x: np.ndarray, center: float, width: float, area: float, gamma_ratio: float,
                  min_width: float = 1e-3) -> np.ndarray:
    """
    Normalized Voigt profile scaled by `area`.
    - width is interpreted as FWHM (for the Gaussian part).
    - gamma_ratio is gamma / width (so gamma = width * gamma_ratio / 2).
    - min_width avoids division by zero.
    """
    # enforce a minimum width to avoid division-by-zero
    width = float(max(width, min_width))
    sigma = width / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gamma = width * gamma_ratio / 2.0

    denom = sigma * np.sqrt(2.0)
    z = ((x - center) + 1j * gamma) / denom
    voigt = np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))

    # normalize to unit area before scaling
    norm = np.trapz(voigt, x)
    if not np.isfinite(norm) or norm <= 0:
        # fallback to Gaussian with safe sigma
        sigma = max(sigma, 1e-6)
        voigt = np.exp(-0.5 * ((x - center) / sigma) ** 2)
        norm = np.trapz(voigt, x)
        if not np.isfinite(norm) or norm <= 0:
            # ultimate fallback: delta-like peak
            voigt = np.zeros_like(x, dtype=float)
            idx = np.argmin(np.abs(x - center))
            voigt[idx] = 1.0
            norm = 1.0
    voigt = voigt / norm
    return area * voigt


def multi_voigt_free_gamma(x: np.ndarray, *flat_params: float) -> np.ndarray:
    """
    Sum of N Voigt peaks. flat_params should be multiple of 4: (center, width, area, gamma_ratio) per peak.
    """
    x = np.asarray(x)
    y_model = np.zeros_like(x, dtype=float)
    n_params = len(flat_params)
    if n_params % 4 != 0:
        raise ValueError("flat_params length must be a multiple of 4")
    for i in range(0, n_params, 4):
        c, w, A, r = flat_params[i:i + 4]
        y_model += voigt_profile(x, c, w, A, r)
    return y_model


def build_initial_guesses_and_bounds(x: np.ndarray, y: np.ndarray, centers: Sequence[float],
                                     width: float, amp_max: Optional[float], gamma_ratio: float,
                                     center_window: float, min_width: float = 1e-3, max_width: float = 400.0
                                     ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Build p0 and (lb, ub) arrays for curve_fit.
    min_width must be > 0 to avoid problems in voigt_profile.
    """
    GAUSS_AREA_FACTOR = 1.0644670194312262  # area factor converting gaussian height*FWHM to area (approx)
    if amp_max is None:
        ymax = np.nanmax(y) if getattr(y, "size", 0) else 1.0
    else:
        ymax = amp_max

    p0_list = []
    lb_list = []
    ub_list = []

    for c0 in centers:
        # center
        p0_list.append(float(c0))
        lb_list.append(float(c0 - center_window))
        ub_list.append(float(c0 + center_window))
        # width (FWHM)
        p0_list.append(float(width))
        lb_list.append(float(min_width))
        ub_list.append(float(max_width))
        # area (positive)
        height_guess = float(max(1e-12, ymax))
        A0 = max(1e-12, height_guess * width * GAUSS_AREA_FACTOR)
        p0_list.append(A0)
        lb_list.append(0.0)
        ub_list.append(1e6 * A0)
        # gamma_ratio [0, 1] (kept conservative)
        p0_list.append(float(gamma_ratio))
        lb_list.append(0.0)
        ub_list.append(1.0)

    p0 = np.array(p0_list, dtype=float)
    lb = np.array(lb_list, dtype=float)
    ub = np.array(ub_list, dtype=float)
    return p0, (lb, ub)


def is_peak_present(x, y, center, window=30.0, rel_height=0.15) -> bool:
    """Return True if a peak around `center` appears to be present.
    Uses local max vs global max (cheap heuristic).
    """
    mask = (x >= center - window) & (x <= center + window)
    if not np.any(mask):
        return False
    local_max = np.nanmax(y[mask])
    global_max = np.nanmax(y)
    # guard against pathological global_max (e.g. all zeros)
    if not np.isfinite(global_max) or global_max <= 0:
        return False
    return local_max >= rel_height * global_max


def fit_one_spectrum(task_args):
    """
    Fit one spectrum with a special rule:
    - Last peak ("2D") is only fitted if present.
    - If absent, it's skipped during fitting and zeros are appended afterwards.
    """
    x, y, rough_centers, rough_width, rough_amp_max, rough_gamma_ratio, center_window, min_width, max_width = task_args

    n_peaks = len(peak_names)

    # --- 1. Check if last peak (2D) is present ---
    last_peak_idx = n_peaks - 1
    fit_last_peak = True
    if peak_names[last_peak_idx] == "2D":
        fit_last_peak = is_peak_present(x, y, rough_centers[last_peak_idx], window=30.0, rel_height=0.15)

    # --- 2. Decide which peaks to fit ---
    if fit_last_peak:
        centers_to_use = rough_centers
        peaks_to_fit = peak_names
    else:
        centers_to_use = rough_centers[:-1]   # exclude 2D
        peaks_to_fit = peak_names[:-1]

    # --- 3. Build guesses/bounds for chosen peaks ---
    p0, bounds = build_initial_guesses_and_bounds(
        x=x, y=y,
        centers=centers_to_use,
        width=rough_width,
        amp_max=rough_amp_max,
        gamma_ratio=rough_gamma_ratio,
        center_window=center_window,
        min_width=max(min_width, 1e-6),
        max_width=max_width
    )

    # --- 4. Fit chosen peaks ---
    try:
        popt, _ = curve_fit(
            multi_voigt_free_gamma, x, y,
            p0=p0, bounds=bounds, maxfev=400
        )
    except Exception:
        popt = np.full_like(p0, np.nan)

    # --- 5. Construct full result vector ---
    full_result = np.zeros(n_peaks * 4, dtype=float)

    # fill fitted peaks
    j = 0
    for i in range(len(peaks_to_fit)):
        full_result[i*4:(i+1)*4] = popt[j:j+4]
        j += 4

    # if last peak was absent → leave zeros in its slot
    # (already zero-initialized)

    return full_result





def fit_all_spectra(ds: xr.Dataset, var: str = 'predicted_difference', n_peaks: Optional[int] = None,
                    min_width: float = 1.0, max_width: float = 400.0, chunksize: int = 16):
    """
    Fit all spectra from an xarray Dataset `ds` variable `var`.
    Returns:
      - results: numpy array shaped (*spatial_shape, n_peaks*4)
      - spatial_dims: list of spatial dimension names (order preserved)
      - spatial_shape: tuple of sizes for those dims
    """
    x = ds['wave_number'].values
    # Get spatial dims in the same order as ds[var].dims excluding 'wave_number'
    var_dims = list(ds[var].dims)
    spatial_dims = [d for d in var_dims if d != 'wave_number']
    spatial_shape = tuple(ds.sizes[d] for d in spatial_dims)

    # flatten y to (n_spectra, n_wave)
    y_data = ds[var].values.reshape(-1, x.size)

    # choose n_peaks
    if n_peaks is None:
        # if not provided, infer from rough_centers global (fallback)
        n_peaks = len(rough_centers)

    args_list = [
        (x, y_data[i], rough_centers[:n_peaks], rough_width, rough_amp_max,
         rough_gamma_ratio, center_window, min_width, max_width)
        for i in range(y_data.shape[0])
    ]

    workers = choose_workers(n_tasks=len(args_list))
    # for large numbers of tasks, give a chunksize to reduce scheduling overhead
    chunksize = max(1, chunksize)

    logger.info(f"Starting parallel fit: {len(args_list)} spectra, {workers} workers, chunksize {chunksize}")
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # tqdm over the iterator from executor.map
        for res in tqdm(executor.map(fit_one_spectrum, args_list, chunksize=chunksize),
                        total=len(args_list), desc="Fitting spectra"):
            results.append(res)

    results = np.asarray(results, dtype=float)
    # reshape to spatial shape + (n_peaks * 4,)
    results = results.reshape(*spatial_shape, n_peaks * 4)
    return results, spatial_dims, spatial_shape


# --- defaults (you can tune these) ---
rough_centers = [1320, 1370, 1485, 1556, 1606, 2735]
rough_width = 40.0
min_width = 5.0         # must be > 0
max_width = 100.0
rough_amp_max = 0.09
rough_gamma_ratio = 0.5
center_window = 40.0
param_names = ['center', 'width', 'area', 'gamma_ratio']
peak_names = ['D1', 'D2', 'B', 'L', 'G', '2D']


if __name__ == "__main__":
    input_base = Path("data/unmixed_spatial/min67_highf/buffer+graphene")
    output_base = Path("data/unmixed_spatial_fitted/min67_highf/buffer+graphene")

    nc_files = list(input_base.rglob("*.nc"))
    logger.info("Found %d netCDF files under %s", len(nc_files), input_base)

    for elem in nc_files:
        try:
            unmixed_datapath = str(elem)
            ds_unmixed = xr.load_dataset(unmixed_datapath)
            logger.info("Loaded %s; vars: %s", unmixed_datapath, list(ds_unmixed.data_vars))

            # make sure the variable exists
            if "predicted_difference" not in ds_unmixed:
                logger.warning("Skipping %s: variable 'predicted_difference' not found", unmixed_datapath)
                continue

            fit_results, spatial_dims, spatial_shape = fit_all_spectra(
                ds_unmixed,
                var="predicted_difference",
                n_peaks=len(peak_names),
                min_width=min_width,
                max_width=max_width,
                chunksize=1
            )

            # reshape to (spatial_dims..., peak, param)
            fit_results = fit_results.reshape(*spatial_shape, len(peak_names), 4)

            fit_da = xr.DataArray(
                fit_results,
                dims=spatial_dims + ["peak", "param"],
                coords={**{dim: ds_unmixed.coords[dim] for dim in spatial_dims},
                        "peak": peak_names,
                        "param": param_names},
            )

            ds_out = xr.Dataset({"fitting_results": fit_da})

            # --- Build mirrored output path ---
            rel_path = elem.relative_to(input_base)  # path relative to input_base
            out_dir = output_base / rel_path.parent   # replicate folder structure
            out_dir.mkdir(parents=True, exist_ok=True)

            out_name = elem.stem + "_fitting_results.nc"
            output_path = out_dir / out_name

            ds_out.to_netcdf(output_path)
            logger.info("Fitting results saved to %s", output_path)

        except Exception as exc:
            # log the exception and continue with next file
            logger.exception("Failed processing %s: %s", elem, exc)
            continue

    logger.info("All done.")
