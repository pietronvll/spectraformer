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


def voigt_profile(x: np.ndarray, center: float, voigt_fwhm: float, area: float, 
                 lorentz_frac: float, min_width: float = 1e-10) -> np.ndarray:
    """
    Area-normalized Voigt profile with exact physical parameterization.
    
    Parameters:
    -----------
    voigt_fwhm : float
        Full width at half maximum of the composite Voigt profile
    lorentz_frac : float
        Fraction of Lorentzian character (0 = pure Gaussian, 1 = pure Lorentzian)
    
    Exact limits:
    - lorentz_frac = 0: Perfect Gaussian with FWHM = voigt_fwhm
    - lorentz_frac = 1: Perfect Lorentzian with FWHM = voigt_fwhm
    
    Scientific basis:
    Uses Olivero-Longbothum approximation for mixed profiles with exact endpoint handling:
    V ≈ 0.5346L + sqrt(0.2166L² + G²)
    where L = lorentz_frac*V, solved for G
    """
    if area <= 0 or not np.isfinite(area):
        return np.zeros_like(x, dtype=float)

    # Clamp to physical range and prevent numerical issues
    voigt_fwhm = max(float(voigt_fwhm), min_width)
    lorentz_frac = np.clip(float(lorentz_frac), 0.0, 1.0)

    # EXACT PURE COMPONENT HANDLING
    if lorentz_frac < 1e-8:  # Pure Gaussian branch
        sigma = voigt_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        gamma = 1e-15  # Numerical zero (prevents wofz singularity)
        
    elif lorentz_frac > 1.0 - 1e-8:  # Pure Lorentzian branch
        gamma = voigt_fwhm / 2.0  # Lorentzian HWHM = FWHM/2
        sigma = 1e-15  # Numerical zero
        
    else:  # Mixed profile: decompose Voigt FWHM
        lorentz_fwhm = lorentz_frac * voigt_fwhm
        
        # Solve for Gaussian FWHM using Olivero-Longbothum approximation
        # V = 0.5346*L + sqrt(0.2166*L² + G²) → G = sqrt((V - 0.5346L)² - 0.2166L²)
        term = voigt_fwhm - 0.5346 * lorentz_fwhm
        if term <= 0:
            # Numerical safeguard for extreme Lorentzian dominance
            gauss_fwhm = 1e-15
        else:
            gauss_fwhm_sq = term**2 - 0.2166 * lorentz_fwhm**2
            gauss_fwhm = np.sqrt(max(0.0, gauss_fwhm_sq))
        
        # Convert to distribution parameters
        sigma = gauss_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # Gaussian σ
        gamma = lorentz_fwhm / 2.0  # Lorentzian HWHM

    # Final numerical safeguards (prevent division by zero in wofz)
    sigma = max(sigma, 1e-15)
    gamma = max(gamma, 1e-15)

    # Core Voigt computation using Faddeeva function
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2.0))
    profile = np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))
    
    return area * profile


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
        logger.debug("curve_fit failed for centers %s: %s", centers_to_use, exc)
        popt = np.full_like(p0, np.nan)

    # --- 5. Construct full result vector ---
    full_result = np.full(n_peaks * 4, np.nan, dtype=float)

    # fill fitted peaks
    j = 0
    for i in range(len(peaks_to_fit)):
        full_result[i*4:(i+1)*4] = popt[j:j+4]
        j += 4

    # if last peak was absent → leave nans in its slot

    return full_result





def fit_all_spectra(ds: xr.Dataset, var: str = 'predicted_difference', n_peaks: Optional[int] = None,
                    min_width: float = 1.0, max_width: float = 400.0, chunksize: int = 16):
    x = ds['wave_number'].values
    var_dims = list(ds[var].dims)
    spatial_dims = [d for d in var_dims if d != 'wave_number']
    spatial_shape = tuple(ds.sizes[d] for d in spatial_dims)

    y_data = ds[var].values.reshape(-1, x.size)

    if "mask" not in ds:
        raise KeyError("Dataset must contain a 'mask' variable (bool array).")
    mask_data = ds["mask"].values.reshape(-1, x.size)

    if n_peaks is None:
        n_peaks = len(rough_centers)

    args_list = []
    for i in range(y_data.shape[0]):
        region_mask = (x >= 1646) & (x <= 2500)   # True only in the desired x-range
        x_masked = x[~region_mask] # keep points that are NOT in the main mask
        y_masked = y_data[i][~region_mask]
        if x_masked.size < n_peaks:  
            # safeguard: not enough points to fit
            x_masked = x
            y_masked = y_data[i]
        args_list.append(
            (x_masked, y_masked, rough_centers[:n_peaks], rough_width,
             rough_amp_max, rough_gamma_ratio, center_window, min_width, max_width)
        )

    workers = choose_workers(n_tasks=len(args_list))
    chunksize = max(1, chunksize)

    logger.info(f"Starting parallel fit: {len(args_list)} spectra, {workers} workers, chunksize {chunksize}")
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for res in tqdm(executor.map(fit_one_spectrum, args_list, chunksize=chunksize),
                        total=len(args_list), desc="Fitting spectra"):
            results.append(res)

    results = np.asarray(results, dtype=float)
    results = results.reshape(*spatial_shape, n_peaks * 4)
    return results, spatial_dims, spatial_shape


# --- defaults (you can tune these) ---
rough_centers = [1320, 1370, 1485, 1556, 1606, 2735]
rough_width = 40.0
min_width = 10.0         # must be > 0
max_width = 100.0
rough_amp_max = None # if None, use spectrum max
rough_gamma_ratio = 0.0
center_window = 40.0
param_names = ['center', 'width', 'area', 'gamma_ratio']
peak_names = ['D1', 'D2', 'B', 'L', 'G', '2D']


if __name__ == "__main__":
    input_base = Path("data/unmixed_spatial/min67_highf/buffer+graphene/RUN3REC2_Buffer_20241011_1/5_5_1")
    output_base = Path("temp/data/unmixed_spatial_fitted/min67_highf/buffer+graphene/RUN3REC2_Buffer_20241011_1/5_5_1")

    nc_files = list(input_base.rglob("*.nc"))
    logger.info("Found %d netCDF files under %s", len(nc_files), input_base)

    for elem in nc_files:
        try:
            unmixed_datapath = str(elem)
            ds_unmixed = xr.load_dataset(unmixed_datapath)
            logger.info("Loaded %s; vars: %s", unmixed_datapath, list(ds_unmixed.data_vars))
            # print(ds_unmixed)
            # print(ds_unmixed.coords)
            # exit()

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
                chunksize=16
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
