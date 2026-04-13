#!/.venv/bin/env python3
"""
SpectraFormer Multi-Map Raman Unmixing + Voigt Peak Fitting Pipeline
====================================================================

Complete end-to-end workflow for analyzing homogeneous material growth across substrate:

  1. Scan input folder for all .txt Raman spectral maps
  2. Load all maps and preprocess them using whitaker_hayes_with_outliers
  3. Aggregate all preprocessed maps into a single dataset for inference
  4. Run single SpectraFormer inference on combined dataset (min70_highf checkpoint)
  5. Compute median spectrum and standard deviation across all spectra
  6. Validity check: if max(median) < 0.05, skip fitting (no material grown), loss = 4.0
  7. [If valid] Fit median spectrum with multiple Voigt profiles
  8. Calculate loss function based on peak area ratios
  9. Save results with loss calculation and generate visualization

Multi-Map Analysis:
  This pipeline is designed for measuring material growth homogeneity across
  different substrate regions. It aggregates all .txt files in the input folder
  into a single large dataset and performs one inference pass followed by one
  fitting operation, ensuring consistent peak constraints across all data.

Required Arguments:
    --input     Path to folder containing .txt Raman spectral maps
    --output    Output directory for results (JSON, CSV, PNG)

Peak Parameters (Graphene - 4 peaks for loss calculation):
    B (1492), L (1564), G (1607), 2D (2735) cm⁻¹
    (D1, D2a, D2b, D2c included in fitting but not used in loss calculation)

Peak parameters and model checkpoint are hardcoded (see DEFAULT_PEAKS, DEFAULT_MODEL, DEFAULT_FITTING).
Modify these constants in the script to customize for your material system.

Usage Examples:
    # Standard run with all maps in folder
    python infer_and_fit_raman_map.py --input "data/raw_data/buffer+graphene/RUN3REC2_Buffer_20241011_1/5_5_1" --output "temp/fit_output/buffer+graphene/RUN3REC2_Buffer_20241011_1/5_5_1" 
    python infer_and_fit_raman_map.py --input "data/raw_data/SiC-high-f/6H_spectra_20250423/5s_5p" --output "temp/fit_output/SiC-high-f/6H_spectra_20250423/5s_5p" 
    python infer_and_fit_raman_map.py --input "data/raw_data/buffer+graphene/G1850A11" --output "temp/fit_output/buffer+graphene/G1850A11" 
    
    python infer_and_fit_raman_map.py --input "data/raw_data/buffer+graphene/20260410_buffer2" --output "temp/fit_output/buffer+graphene/20260410_buffer2"

Output Files:
    - fitting_results.json      : Peak parameters, uncertainties, quality metrics, AND loss function
    - fitting_parameters.csv    : CSV table of all peak parameters
    - fitting_visualization.png : 2-panel plot with spectrum, fit, uncertainty band, and components

Author: SpectraFormer Pipeline
Date: 2026
"""

import logging
import argparse
import json
import time
import copy
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit
from scipy.special import wofz
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from matplotlib import rcParams
rcParams['font.size'] = 24

# ============================================================================
# SETUP LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA LOADING
# ============================================================================

def parse_dataset(path: str) -> xr.DataArray:
    """
    Parse Raman spectral map from .txt file.
    
    Expected format (tab or space-delimited):
        X_0  X_1  ...  X_n  wave_number  intensity
        ...
    
    Last two columns are wave_number and intensity.
    All preceding columns are spatial coordinates.
    
    Parameters:
    -----------
    path : str
        Path to .txt file
    
    Returns:
    --------
    xr.DataArray
        Spectral map with dimensions X_0, X_1, ..., wave_number
    """
    _data = np.loadtxt(path, unpack=True)
    wave_number, _counts = _data[-2], _data[-1]
    num_coords = len(_data) - 2
    coords = [_data[idx] for idx in range(num_coords)]
    unique_coords = [np.unique(coord, return_inverse=True) for coord in coords]
    unique_coords += [np.unique(wave_number, return_inverse=True)]
    idxs, values = [coord[1] for coord in unique_coords], [coord[0] for coord in unique_coords]
    counts_shape = tuple([len(coord[0]) for coord in unique_coords])
    counts = np.zeros(counts_shape, dtype=_counts.dtype)

    for z in zip(*idxs, _counts):
        counts[tuple(z[:-1])] = z[-1]

    dimension_names = [f'X_{i}' for i in range(len(values) - 1)] + ['wave_number']
    counts = xr.DataArray(counts, coords=values, dims=dimension_names)
    counts.wave_number.attrs['units'] = 'cm^{-1}'
    return counts


def load_all_txt_files(folder_path: Path) -> List[xr.DataArray]:
    """
    Load all .txt files from a folder.
    
    Parameters:
    -----------
    folder_path : Path
        Path to folder containing .txt files
    
    Returns:
    --------
    List[xr.DataArray]
        List of parsed DataArrays from all .txt files
    """
    txt_files = sorted(folder_path.glob("*.txt"))
    if not txt_files:
        logger.error(f"No .txt files found in {folder_path}")
        return []
    
    dataarrays = []
    for txt_file in txt_files:
        logger.info(f"  Loading: {txt_file.name}")
        da = parse_dataset(str(txt_file))
        dataarrays.append(da)
    
    return dataarrays


def aggregate_preprocessed_maps(dataarrays: List[xr.DataArray]) -> xr.DataArray:
    """
    Aggregate multiple preprocessed maps into a single (spectra, wave_number) dataset.
    
    Concatenates along the 'spectra' dimension, creating a 2D array suitable for batch inference.
    
    Parameters:
    -----------
    dataarrays : List[xr.DataArray]
        List of DataArrays with dimensions (spectra, wave_number)
    
    Returns:
    --------
    xr.DataArray
        Aggregated array with dimensions (spectra, wave_number)
    """
    # Concatenate along spectra dimension
    aggregated = xr.concat(dataarrays, dim='spectra')
    
    logger.info(f"  ✓ Aggregated shape: {aggregated.shape}")
    logger.info(f"  ✓ Total spectra: {aggregated.sizes['spectra']}")
    
    return aggregated


# # ============================================================================
# # OUTLIER FILTERING (Modified Z-Score Method)
# # ============================================================================

# def modified_z_score(spectrum: np.ndarray) -> np.ndarray:
#     """
#     Calculates the modified z-scores of a given spectrum.
    
#     Modified z-score is more robust to outliers than standard z-score,
#     using median absolute deviation (MAD) instead of standard deviation.
    
#     Parameters:
#     -----------
#     spectrum : np.ndarray
#         Input spectrum values
    
#     Returns:
#     --------
#     np.ndarray
#         Modified z-scores for each point
#     """
#     median_val = np.median(spectrum)
#     mad = np.median(np.abs(spectrum - median_val))
#     if mad == 0:
#         return np.zeros_like(spectrum)
#     return 0.6745 * (spectrum - median_val) / mad


# def whitaker_hayes_modified_z_score(spectrum: np.ndarray) -> np.ndarray:
#     """
#     Calculates the Whitaker-Hayes modified z-scores of spectrum differences.
    
#     This detects spikes by looking at discontinuities in the spectrum.
    
#     Parameters:
#     -----------
#     spectrum : np.ndarray
#         Input spectrum values
    
#     Returns:
#     --------
#     np.ndarray
#         Absolute modified z-scores of spectrum first differences
#     """
#     return np.abs(modified_z_score(np.diff(spectrum)))


# def whitaker_hayes_spectrum(
#     intensity_values_array: np.ndarray,
#     kernel_size: int = 3,
#     threshold: float = 8.0
# ) -> np.ndarray:
#     """
#     Apply Whitaker-Hayes spike detection and removal to a single spectrum.
    
#     Iteratively identifies spikes based on modified z-scores and replaces
#     them with the median of non-spike neighbors.
    
#     Parameters:
#     -----------
#     intensity_values_array : np.ndarray
#         Single spectrum to process
#     kernel_size : int
#         Neighborhood size for median replacement
#     threshold : float
#         Modified z-score threshold above which points are considered spikes
    
#     Returns:
#     --------
#     np.ndarray
#         Spectrum with spikes removed
#     """
#     spectrum_array = copy.deepcopy(intensity_values_array)
#     spikes_original = whitaker_hayes_modified_z_score(spectrum_array) > threshold
    
#     # Pad spikes array to match spectrum length (diff reduces by 1)
#     spikes = np.zeros(len(spectrum_array), dtype=bool)
#     spikes[:-1] = spikes_original
    
#     iteration = 0
#     max_iterations = 100
    
#     while np.any(spikes) and iteration < max_iterations:
#         changes = False
#         for i in range(len(spikes)):
#             if spikes[i]:
#                 neighbours = np.arange(
#                     max(0, i - kernel_size),
#                     min(len(spectrum_array), i + kernel_size + 1)
#                 )
#                 non_spike_neighbours = spectrum_array[neighbours[~spikes[neighbours]]]
                
#                 if len(non_spike_neighbours) > 0:
#                     fixed_value = np.median(non_spike_neighbours)
#                     if np.isfinite(fixed_value):
#                         spectrum_array[i] = fixed_value
#                         spikes[i] = False
#                         changes = True
        
#         if not changes:
#             break
#         iteration += 1
    
#     return spectrum_array


# def filter_outliers_modified_z_score(
#     intensity_data: xr.DataArray,
#     kernel_size: int = 3,
#     threshold: float = 8.0
# ) -> xr.DataArray:
#     """
#     Apply modified z-score outlier filtering to all spectra in a DataArray.
    
#     Parameters:
#     -----------
#     intensity_data : xr.DataArray
#         Input spectral data with wave_number as last dimension
#     kernel_size : int
#         Neighborhood size for spike replacement
#     threshold : float
#         Modified z-score threshold for spike detection
    
#     Returns:
#     --------
#     xr.DataArray
#         Filtered spectral data with same dimensions and coordinates
#     """
#     filtered_data = np.apply_along_axis(
#         whitaker_hayes_spectrum,
#         axis=-1,
#         arr=intensity_data.values,
#         kernel_size=kernel_size,
#         threshold=threshold
#     )
    
#     return xr.DataArray(
#         filtered_data,
#         dims=intensity_data.dims,
#         coords=intensity_data.coords
#     )


# ============================================================================
# VOIGT PROFILE FITTING (Adapted from PL_fitting_individual_comprehensive_2.py)
# ============================================================================

def true_vectorized_voigt(
    x: np.ndarray, 
    centers: np.ndarray, 
    sigmas: np.ndarray,
    gammas: np.ndarray, 
    areas: np.ndarray
) -> np.ndarray:
    """
    Truly vectorized Voigt profile computation using broadcasting.
    
    Parameters:
    -----------
    x : np.ndarray
        Energy/frequency axis (shape: (n_energy,))
    centers : np.ndarray
        Peak centers (shape: (n_spectra,))
    sigmas : np.ndarray
        Gaussian std dev (shape: (n_spectra,))
    gammas : np.ndarray
        Lorentzian half-width (shape: (n_spectra,))
    areas : np.ndarray
        Peak areas (shape: (n_spectra,))
    
    Returns:
    --------
    np.ndarray
        Voigt profiles (shape: (n_spectra, n_energy))
    """
    # Reshape for broadcasting: (n_spectra, 1) vs (1, n_energy)
    x_reshaped = x.reshape(1, -1)
    centers_reshaped = centers.reshape(-1, 1)
    sigmas_reshaped = sigmas.reshape(-1, 1)
    gammas_reshaped = gammas.reshape(-1, 1)
    
    # Compute z for all spectra and energies at once
    z = ((x_reshaped - centers_reshaped) + 1j * gammas_reshaped) / (sigmas_reshaped * np.sqrt(2.0))
    
    # Compute Faddeeva function for all values
    w_values = wofz(z)
    
    # Compute profiles
    sqrt_2pi = np.sqrt(2.0 * np.pi)
    profiles = np.real(w_values) / (sigmas_reshaped * sqrt_2pi)
    
    # Scale by areas
    return profiles * areas.reshape(-1, 1)


def batch_voigt_profiles(
    x: np.ndarray,
    centers: np.ndarray,
    widths: np.ndarray,
    areas: np.ndarray,
    gamma_fracs: np.ndarray,
    min_width: float = 1e-10
) -> np.ndarray:
    """
    Vectorized computation of multiple Voigt profiles with shared energy axis.
    
    Parameters:
    -----------
    x : np.ndarray
        Energy axis (shape: (n_energy,))
    centers : np.ndarray 
        Peak centers (shape: (n_spectra,))
    widths : np.ndarray
        Voigt FWHM (shape: (n_spectra,))
    areas : np.ndarray
        Peak areas (shape: (n_spectra,))
    gamma_fracs : np.ndarray
        Lorentzian fractions (shape: (n_spectra,))
    min_width : float
        Minimum width for numerical stability
    
    Returns:
    --------
    np.ndarray
        Array of Voigt profiles (shape: (n_spectra, n_energy))
    """
    n_spectra = len(centers)
    if n_spectra == 0:
        return np.zeros((0, len(x)))
    
    sqrt_2ln2 = np.sqrt(2.0 * np.log(2.0))
    
    # Clamp parameters
    widths = np.maximum(widths, min_width)
    gamma_fracs = np.clip(gamma_fracs, 0.0, 1.0)
    
    sigma = np.empty(n_spectra, dtype=np.float64)
    gamma = np.empty(n_spectra, dtype=np.float64)
    
    # Pure Gaussian (gamma_frac ≈ 0)
    is_gaussian = gamma_fracs < 1e-8
    if np.any(is_gaussian):
        sigma[is_gaussian] = widths[is_gaussian] / (2.0 * sqrt_2ln2)
        gamma[is_gaussian] = 1e-15
    
    # Pure Lorentzian (gamma_frac ≈ 1)
    is_lorentz = gamma_fracs > (1.0 - 1e-8)
    if np.any(is_lorentz):
        gamma[is_lorentz] = widths[is_lorentz] / 2.0
        sigma[is_lorentz] = 1e-15
    
    # Mixed profiles (Olivero-Longbothum decomposition)
    is_mixed = ~(is_gaussian | is_lorentz)
    if np.any(is_mixed):
        V = widths[is_mixed]
        f = gamma_fracs[is_mixed]
        L = f * V
        term = V - 0.5346 * L
        G_sq = np.maximum(0.0, term**2 - 0.2166 * L**2)
        G = np.sqrt(G_sq)
        sigma[is_mixed] = G / (2.0 * sqrt_2ln2)
        gamma[is_mixed] = L / 2.0
    
    sigma = np.maximum(sigma, 1e-15)
    gamma = np.maximum(gamma, 1e-15)
    
    result = true_vectorized_voigt(x, centers, sigma, gamma, areas)
    return result


def voigt_profile(
    x: np.ndarray, 
    center: float, 
    voigt_fwhm: float, 
    area: float,
    lorentz_frac: float, 
    min_width: float = 1e-10
) -> np.ndarray:
    """
    Single area-normalized Voigt profile.
    
    Parameters:
    -----------
    x : np.ndarray
        Energy axis
    center : float
        Peak center
    voigt_fwhm : float
        Voigt FWHM
    area : float
        Peak area
    lorentz_frac : float
        Lorentzian fraction [0,1]
    min_width : float
        Minimum width for stability
    
    Returns:
    --------
    np.ndarray
        Voigt profile values at x
    """
    return batch_voigt_profiles(
        x.reshape(1, -1),
        np.array([center]),
        np.array([voigt_fwhm]),
        np.array([area]),
        np.array([lorentz_frac]),
        min_width
    )[0]


def multi_voigt_free_gamma(x: np.ndarray, *flat_params: float) -> np.ndarray:
    """
    Sum of N Voigt peaks.
    
    flat_params should be multiple of 4: (center, width, area, gamma_ratio) per peak.
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


def calculate_fwhm(energy: np.ndarray, spectrum: np.ndarray, baseline_mode: Optional[str] = None) -> float:
    """
    Calculate FWHM from half-maximum crossings.
    
    Parameters:
    -----------
    energy : np.ndarray
        Energy/frequency axis
    spectrum : np.ndarray
        Spectrum values
    baseline_mode : Optional[str]
        Baseline handling: None (half=peak/2), 'min' (half=(peak+min)/2), 'edge'
    
    Returns:
    --------
    float
        FWHM value or np.nan if invalid
    """
    x = np.asarray(energy, dtype=float)
    y = np.asarray(spectrum, dtype=float)

    if x.ndim != 1 or y.ndim != 1 or x.size < 3:
        return np.nan

    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        return np.nan

    # Sort if needed
    if not np.all(np.diff(x) > 0):
        idx = np.argsort(x)
        x = x[idx]
        y = y[idx]

    peak_idx = int(np.nanargmax(y))
    peak_val = float(y[peak_idx])

    # Baseline handling
    if baseline_mode is None:
        half = peak_val * 0.5
    elif baseline_mode == "min":
        half = 0.5 * (peak_val + np.min(y))
    elif baseline_mode == "edge":
        n_edge = max(1, int(0.01 * x.size))
        edge_vals = np.concatenate([y[:n_edge], y[-n_edge:]])
        half = 0.5 * (peak_val + float(np.median(edge_vals)))
    else:
        raise ValueError("baseline_mode must be None, 'min', or 'edge'")

    # Find left crossing
    left_x = None
    for i in range(peak_idx - 1, -1, -1):
        y0 = y[i]
        y1 = y[i + 1]
        if (y0 - half) * (y1 - half) < 0:
            # Linear interpolation for crossing point
            left_x = x[i] + (x[i + 1] - x[i]) * (half - y0) / (y1 - y0)
            break

    # Find right crossing
    right_x = None
    for i in range(peak_idx, x.size - 1):
        y0 = y[i]
        y1 = y[i + 1]
        if (y0 - half) * (y1 - half) < 0:
            right_x = x[i] + (x[i + 1] - x[i]) * (half - y0) / (y1 - y0)
            break

    if left_x is None or right_x is None:
        return np.nan

    return float(abs(right_x - left_x))


def build_initial_guesses_and_bounds(
    x: np.ndarray,
    y: np.ndarray,
    centers: List[float],
    widths: List[float],
    amp_maxs: List[Optional[float]],
    gamma_ratios: List[float],
    center_windows: List[float],
    min_widths: List[float],
    max_widths: List[float]
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Build initial parameter guesses and bounds for curve fitting."""
    p0_list, lb_list, ub_list = [], [], []
    
    for i, c0 in enumerate(centers):
        width = widths[i]
        amp_max = amp_maxs[i]
        gamma_ratio = gamma_ratios[i]
        center_window = center_windows[i]
        min_width = min_widths[i]
        max_width = max_widths[i]
        ymax = np.nanmax(y) if amp_max is None else amp_max
        
        if not np.isfinite(ymax) or ymax <= 0:
            ymax = 1.0
        
        # Center
        p0_list.append(float(c0))
        lb_list.append(float(c0 - center_window))
        ub_list.append(float(c0 + center_window))
        
        # Width (physical bounds)
        width = max(min_width, min(width, max_width))
        p0_list.append(float(width))
        lb_list.append(float(min_width))
        ub_list.append(float(max_width))
        
        # Area with physics-based initialization
        height_guess = float(max(1e-12, ymax))
        
        if gamma_ratio < 0.1:
            area_factor = np.sqrt(2.0 * np.log(2.0)) / 2.0
        elif gamma_ratio > 0.9:
            area_factor = 2.0 / np.pi
        else:
            area_factor = 0.5 + gamma_ratio * (np.sqrt(2.0 * np.log(2.0)) / 2.0 - 0.5 + 2.0 / np.pi)
        
        A0 = max(1e-12, height_guess * width * area_factor)
        p0_list.append(A0)
        lb_list.append(1e-6)
        ub_list.append(1e6 * A0)
        
        # Gamma ratio (physical bounds)
        gamma_ratio = np.clip(gamma_ratio, 0.0, 1.0)
        p0_list.append(float(gamma_ratio))
        lb_list.append(0.01)
        ub_list.append(0.99)
    
    p0 = np.array(p0_list, dtype=float)
    lb = np.array(lb_list, dtype=float)
    ub = np.array(ub_list, dtype=float)
    return p0, (lb, ub)


# ============================================================================
# INFERENCE
# ============================================================================

def load_model_and_predict(
    checkpoint_tag: str,
    dataset: xr.DataArray,
    ckpts_path: Path
) -> Dict[str, Any]:
    """
    Load SpectraFormer model and run inference on preprocessed dataset.
    
    Parameters:
    -----------
    checkpoint_tag : str
        Checkpoint name (with or without 'spectraformer:' prefix)
    dataset : xr.DataArray
        Preprocessed spectral data (stacked to 'spectra', 'wave_number' dims)
    ckpts_path : Path
        Path to checkpoints directory
    
    Returns:
    --------
    Dict[str, Any]
        Predictions dict with keys: spectra, predicted_spectra, predicted_difference, wave_number, mask
    """
    import jax
    import jax.numpy as jnp
    import optax
    import orbax.checkpoint as ocp
    from spectraformer.inference import predict
    from spectraformer.input_pipeline import batch_sampler
    from spectraformer.model import CustomTrainState, SpectraFormer

    # Build full checkpoint path
    full_tag = (
        f"spectraformer:{checkpoint_tag}"
        if not checkpoint_tag.startswith("spectraformer:")
        else checkpoint_tag
    )
    checkpoint_path = (ckpts_path / full_tag).resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint from: {checkpoint_path}")

    # Initialize checkpoint manager
    ckpt_options = ocp.CheckpointManagerOptions(
        read_only=True, save_interval_steps=0, create=False
    )
    ckpt_manager = ocp.CheckpointManager(checkpoint_path, options=ckpt_options)

    # Get config
    configs_dict = ckpt_manager.metadata()
    if configs_dict is None:
        raise ValueError(
            "Checkpoint does not contain configuration metadata. "
            "This checkpoint may have been created with an older version."
        )
    if "custom" in configs_dict:
        configs_dict = configs_dict["custom"]

    class Config:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)

    configs = Config(configs_dict)

    # Build learning rate schedule (required for checkpoint restoration)
    cosine_kwargs = []
    learning_rate = getattr(configs, "learning_rate", 1e-3)
    init_value = 0.1 * learning_rate
    peak_value = learning_rate
    warmup_steps = getattr(configs, "warmup_steps", 1000)
    decay_steps = getattr(configs, "decay_steps", 2000)
    decline_coeff = getattr(configs, "decline_coeff", 1)
    num_cycles = getattr(configs, "num_cycles", 100)

    for _ in range(num_cycles):
        end_value = decline_coeff * init_value
        cycle_dict = {
            "init_value": init_value,
            "peak_value": peak_value,
            "warmup_steps": warmup_steps,
            "decay_steps": decay_steps,
            "end_value": end_value,
        }
        cosine_kwargs.append(cycle_dict)
        init_value = end_value
        peak_value *= decline_coeff

    learning_rate_fn = optax.schedules.sgdr_schedule(cosine_kwargs=cosine_kwargs)

    learning_rate_decay = getattr(configs, "learning_rate_decay", "Constant")
    if learning_rate_decay == "Multiple cosine decay cycles":
        tx = optax.adam(learning_rate=learning_rate_fn)
    else:
        tx = optax.adam(learning_rate=configs.learning_rate)

    # Get mask windows
    mask_windows = list(
        zip(configs.masked_interval_starts, configs.masked_interval_ends)
    )

    # Create dummy batch
    dummy_example = next(batch_sampler(dataset, mask_windows, batch_size=1))

    # Initialize model
    model = SpectraFormer(
        num_heads=configs.num_heads,
        num_layers=configs.num_layers,
        embedding_dim=configs.embedding_dim,
    )

    root_key = jax.random.key(seed=configs.root_rng_seed)
    _, params_key, _ = jax.random.split(key=root_key, num=3)

    variables = model.init(
        params_key,
        dummy_example["masked_spectra"][0],
        dummy_example["wave_number"],
        dummy_example["mask"],
        training=False,
    )

    state = CustomTrainState.create(
        apply_fn=jax.jit(model.apply, static_argnames=("training",)),
        params=variables["params"],
        tx=tx,
        epoch=jnp.array(0, dtype=jnp.int32),
    )

    # Restore checkpoint
    latest_step = ckpt_manager.latest_step()
    try:
        state = ckpt_manager.restore(
            latest_step,
            args=ocp.args.StandardRestore(state),
        )
        logger.info(f"Successfully loaded checkpoint (NEW format)")
    except Exception as e_new:
        try:
            restored_dict = ckpt_manager.restore(
                latest_step,
                args=ocp.args.StandardRestore({"state": state})
            )
            state = restored_dict["state"]
            logger.info(f"Successfully loaded checkpoint (OLD format)")
        except Exception as e_old:
            raise ValueError(
                f"Failed to restore checkpoint in both formats.\n"
                f"New format error: {str(e_new)}\n"
                f"Old format error: {str(e_old)}"
            )

    # Run predictions
    logger.info("Running inference...")
    test_data = list(batch_sampler(dataset, mask_windows, shuffle=False, batch_size=1))
    predictions = [
        predict(
            state.apply_fn,
            {"params": state.params},
            datapoint,
            datapoint["mask"],
        )
        for datapoint in test_data
    ]

    return predictions, configs, mask_windows


# ============================================================================
# FITTING & LOSS CALCULATION
# ============================================================================

def fit_averaged_spectrum(
    x: np.ndarray,
    y: np.ndarray,
    centers: List[float],
    peak_names: List[str],
    widths: List[float],
    amp_maxs: List[Optional[float]],
    gamma_ratios: List[float],
    center_windows: List[float],
    min_widths: List[float],
    max_widths: List[float],
    maxfev: int = 20000,
    tolerances: Optional[Dict[str, float]] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """
    Fit averaged spectrum with multiple Voigt profiles.
    
    Parameters:
    -----------
    x : np.ndarray
        Raman shift axis (cm^-1)
    y : np.ndarray
        Averaged spectrum intensities
    centers, peak_names, widths, amp_maxs, gamma_ratios : List
        Peak parameters
    center_windows, min_widths, max_widths : List
        Fitting constraints
    maxfev : int
        Max function evaluations
    tolerances : Optional[Dict[str, float]]
        Convergence tolerances {ftol, xtol, gtol}
    
    Returns:
    --------
    Tuple[Optional[np.ndarray], Optional[np.ndarray], str]
        (popt, pcov, status_message) or (None, None, error_message)
    """
    if tolerances is None:
        tolerances = {"ftol": 1e-5, "xtol": 1e-5, "gtol": 1e-5}
    
    # Validate spectrum
    if not np.any(np.isfinite(y)) or np.all(y == 0) or np.all(np.isnan(y)):
        return None, None, "Spectrum is invalid (NaN, all zeros, or no finite values)"
    
    # Build initial guesses and bounds
    p0, bounds = build_initial_guesses_and_bounds(
        x, y, centers, widths, amp_maxs, gamma_ratios, center_windows, min_widths, max_widths
    )
    
    try:
        logger.info(f"Fitting {len(centers)} peaks with {len(p0)} parameters...")
        popt, pcov = curve_fit(
            multi_voigt_free_gamma, x, y,
            p0=p0, bounds=bounds, maxfev=maxfev,
            ftol=tolerances.get('ftol', 1e-5),
            xtol=tolerances.get('xtol', 1e-5),
            gtol=tolerances.get('gtol', 1e-5)
        )
        logger.info("Fitting successful!")
        # Logging all the fitting parameters per each peak in human readable format
        for i in range(0, len(popt), 4):
            c, w, A, r = popt[i:i + 4]
            peak_name = peak_names[i // 4] if i // 4 < len(peak_names) else f"Peak {i//4 + 1}"
            logger.info(f"      ✓ Peak {peak_name}: Center={c:.2f}, Width={w:.2f}, Area={A:.2f}, Gamma_frac={r:.3f}")
        return popt, pcov, "Success"
    except Exception as exc:
        error_msg = f"Fitting failed: {exc}"
        logger.warning(error_msg)
        return None, None, error_msg

def loss_term(value, target: float = 0.0):
    return ((value - target)**2)


def calculate_loss(popt: np.ndarray, peak_names: List[str], centers: List[float]) -> Dict[str, float]:
    """
    Calculate loss function based on peak area ratios.
    
    Loss function promotes growth measured by D-band peaks (B, L, G, 2D).
    
    Parameters:
    -----------
    popt : np.ndarray
        Fitted parameters [c0, w0, A0, r0, c1, w1, A1, r1, ...]
    peak_names : List[str]
        Peak identifiers (e.g., ['D1', 'D2a', 'B', 'L', 'G', '2D'])
    centers : List[float]
        Expected peak centers for reference
    
    Returns:
    --------
    Dict[str, float]
        Dictionary with keys:
        - 'loss': Final loss value [0, 4]
        - 'loss_term_1': Area(2D) / Area(G)
        - 'loss_term_2': Area(B) / Area(G)
        - 'area_2D': Fitted area of 2D peak
        - 'area_G': Fitted area of G peak
        - 'area_B': Fitted area of B peak
        - 'area_L': Fitted area of L peak
    """
    # Extract peak indices for B, G, L, 2D
    try:
        idx_B = peak_names.index('B')
        idx_G = peak_names.index('G')
        idx_L = peak_names.index('L')
        idx_2D = peak_names.index('2D')
    except ValueError as e:
        logger.warning(f"Could not find required peaks for loss calculation: {e}")
        return {
            'loss': 2.0,
            'loss_term_1': np.nan,
            'loss_term_2': np.nan,
            'error': f'Missing peak: {e}'
        }
    
    # Extract areas (parameter index 2, 6, 10, ... for each peak)
    area_B = popt[idx_B * 4 + 2]
    area_G = popt[idx_G * 4 + 2]
    area_L = popt[idx_L * 4 + 2]
    area_2D = popt[idx_2D * 4 + 2]
    sum_all_areas = area_B + area_G + area_L + area_2D
    
    # Calculate loss terms
    # Loss term 1: Area(2D) / Area(G) -> should be 0 (no competition with G)
    if area_G > 0:
        loss_term_1_value = area_2D / area_G
    else:
        loss_term_1_value = np.inf if area_2D > 0 else 0.0
    
    # Loss term 2: Area(B) / Area(G) -> should be 1 (equal contribution)
    if area_G > 0:
        loss_term_2_value = area_B / area_G
    else:
        loss_term_2_value = np.inf if area_B > 0 else 0.0
    
    loss_term_1 = loss_term(loss_term_1_value, target=0.0)
    loss_term_2 = loss_term(loss_term_2_value, target=1.0)
    loss = (loss_term_1 + loss_term_2) ** 2
    
    logger.info(f"  ✓ Loss components:")
    logger.info(f"    - Area(2D): {area_2D}")
    logger.info(f"    - Area(G): {area_G}")
    logger.info(f"    - Area(B): {area_B}")
    logger.info(f"    - Area(L): {area_L}")
    logger.info(f"    - Area(2D)/Area(G): {area_2D/area_G if area_G > 0 else 'inf'}")
    logger.info(f"    - Area(B)/Area(G): {area_B/area_G if area_G > 0 else 'inf'}")
    logger.info(f"    - Loss_term_1 (2D/G): {loss_term_1}")
    logger.info(f"    - Loss_term_2 (B/G): {loss_term_2}")
    logger.info(f"    - Loss total: {loss} (range: 0 to 4)")
    
    return {
        'loss': float(loss),
        'loss_term_1': float(loss_term_1),
        'loss_term_2': float(loss_term_2),
        'area_2D': float(area_2D),
        'area_G': float(area_G),
        'area_B': float(area_B),
        'area_L': float(area_L),
        'sum_areas': float(sum_all_areas)
    }



def create_fit_visualization(
    x: np.ndarray,
    y: np.ndarray,
    popt: np.ndarray,
    centers: List[float],
    peak_names: List[str],
    output_path: Path,
    y_std: Optional[np.ndarray] = None
):
    """
    Create visualization of median spectrum with ±1std band and multi-peak fit.
    
    Parameters:
    -----------
    x : np.ndarray
        Raman shift axis
    y : np.ndarray
        Median spectrum
    popt : np.ndarray
        Fitted parameters
    centers : List[float]
        Peak center positions (for reference)
    peak_names : List[str]
        Names of peaks (e.g., ['D1', 'D2', 'G', '2D'])
    output_path : Path
        Path to save figure
    y_std : Optional[np.ndarray]
        Standard deviation spectrum for ±1std band
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 7), constrained_layout=True)
    
    # Calculate fit and components
    y_fit = multi_voigt_free_gamma(x, *popt)
    
    n_peaks = len(centers)
    colors = [f'C{i}' for i in range(n_peaks)]
    
    # Plot 2: Individual components
    ax.plot(x, y, 'ko-', linewidth=2, label='Median', markersize=3, alpha=0.7)
    
    
    # Add ±1std shaded region if std provided
    if y_std is not None:
        ax.fill_between(x, y - y_std, y + y_std, alpha=0.15, color='blue', label='±1 std')
    
    ax.plot(x, y_fit, 'r-', linewidth=3.5, label='Fit')
    
    for i in range(n_peaks):
        params = popt[i*4:(i+1)*4]
        peak_y = voigt_profile(x, params[0], params[1], params[2], params[3])
        baseline = -0.10
        ax.plot(x, peak_y + baseline, '--', linewidth=2.5, color=colors[i], label=f'{peak_names[i]}')
        ax.fill_between(x, peak_y + baseline, baseline, alpha=0.1, color=colors[i])
    
    ax.set_xlabel('Raman Shift (cm$^{-1}$)', fontsize='large')
    ax.set_ylabel('Intensity (a.u.)', fontsize='large')
    ax.legend(fontsize='large', ncol=2)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(300))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.set_ylim(-0.10, 0.25)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved visualization to: {output_path}")
    plt.close()


def create_fit_visualization_no_fit(
    x: np.ndarray,
    y: np.ndarray,
    centers: List[float],
    peak_names: List[str],
    output_path: Path,
    y_std: Optional[np.ndarray] = None
):
    """
    Create visualization of median spectrum when no fitting was performed.
    
    Parameters:
    -----------
    x : np.ndarray
        Raman shift axis
    y : np.ndarray
        Median spectrum
    centers : List[float]
        Peak center positions (for reference lines)
    peak_names : List[str]
        Names of peaks
    output_path : Path
        Path to save figure
    y_std : Optional[np.ndarray]
        Standard deviation spectrum for ±1std band
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 7), constrained_layout=True)
    
    # Plot spectrum
    ax.plot(x, y, 'ko-', linewidth=2, label='Median', markersize=3, alpha=0.7)
    
    # Add ±1std shaded region if std provided
    if y_std is not None:
        ax.fill_between(x, y - y_std, y + y_std, alpha=0.25, color='blue', label='±1 std')
    
    # # Add reference lines for expected peak positions
    # for center, peak_name in zip(centers, peak_names):
    #     ax.axvline(center, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    ax.axhline(FIT_THRESHOLD, color='red', linestyle='--', alpha=0.4, linewidth=1, label='Growth threshold')
    
    ax.set_xlabel('Raman Shift (cm$^{-1}$)', fontsize='large')
    ax.set_ylabel('Intensity (a.u.)', fontsize='large')
    ax.legend(fontsize='large', ncol=2)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(300))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.set_ylim(-0.10, 0.25)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved visualization (no fit) to: {output_path}")
    plt.close()


def save_results_no_fit(
    output_dir: Path,
    centers: List[float],
    peak_names: List[str],
    x: np.ndarray,
    y: np.ndarray,
    predictions: List[Dict[str, Any]],
    configs: Any,
    loss_value: float = 4.0,
    loss_reason: str = "No material growth detected"
):
    """
    Save results when fitting was skipped due to low signal.
    
    Parameters:
    -----------
    output_dir : Path
        Output directory
    centers : List[float]
        Peak centers
    peak_names : List[str]
        Peak names
    x : np.ndarray
        Raman shift axis
    y : np.ndarray
        Median spectrum
    predictions : List[Dict]
        Model predictions
    configs : Any
        Model configuration
    loss_value : float
        Loss value to assign (default 4.0 for no growth)
    loss_reason : str
        Reason why fitting was skipped
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_peaks = len(centers)
    
    # Build results JSON
    results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_peaks": n_peaks,
            "peak_names": peak_names,
            "n_spectra_averaged": len(predictions),
            "fitting_performed": False,
            "fitting_skipped_reason": loss_reason,
            "model_config": {
                "num_heads": getattr(configs, "num_heads", None),
                "num_layers": getattr(configs, "num_layers", None),
                "embedding_dim": getattr(configs, "embedding_dim", None),
            }
        },
        "peaks": [],
        "quality_metrics": {
            "r_squared": np.nan,
            "sum_squared_error": np.nan,
            "mean_absolute_error": np.nan,
            "note": "No fitting performed - skipped due to low signal"
        },
        "loss_function": {
            "loss": float(loss_value),
            "loss_reason": loss_reason,
            "loss_term_1": np.nan,
            "loss_term_2": np.nan,
            "description": {
                "note": "No material growth detected",
                "max_intensity": float(np.max(y)),
                "threshold": FIT_THRESHOLD
            }
        }
    }
    
    # Save JSON
    json_path = output_dir / "fitting_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved JSON results (no fit) to: {json_path}")
    
    # Save CSV placeholder
    csv_path = output_dir / "fitting_parameters.csv"
    with open(csv_path, 'w') as f:
        f.write("note,value\n")
        f.write(f"fitting_performed,false\n")
        f.write(f"reason,{loss_reason}\n")
        f.write(f"max_intensity,{np.max(y):.6e}\n")
        f.write(f"threshold,{FIT_THRESHOLD}\n")
        f.write(f"loss,{loss_value:.4f}\n")
    logger.info(f"Saved CSV results (no fit) to: {csv_path}")

def save_results(
    output_dir: Path,
    popt: np.ndarray,
    pcov: np.ndarray,
    centers: List[float],
    peak_names: List[str],
    x: np.ndarray,
    y: np.ndarray,
    predictions: List[Dict[str, Any]],
    configs: Any,
    loss_dict: Optional[Dict[str, float]] = None
):
    """
    Save fitting results to JSON and CSV files.
    
    Parameters:
    -----------
    output_dir : Path
        Output directory
    popt : np.ndarray
        Fitted parameters
    pcov : np.ndarray
        Covariance matrix
    centers : List[float]
        Peak centers
    peak_names : List[str]
        Peak names
    x : np.ndarray
        Raman shift axis
    y : np.ndarray
        Averaged spectrum
    predictions : List[Dict]
        Model predictions
    configs : Any
        Model configuration
    loss_dict : Optional[Dict[str, float]]
        Loss function calculation results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_peaks = len(centers)
    
    # Build results JSON
    results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_peaks": n_peaks,
            "peak_names": peak_names,
            "n_spectra_averaged": len(predictions),
            "model_config": {
                "num_heads": getattr(configs, "num_heads", None),
                "num_layers": getattr(configs, "num_layers", None),
                "embedding_dim": getattr(configs, "embedding_dim", None),
            }
        },
        "peaks": []
    }
    
    # Add peak results
    uncertainties = np.sqrt(np.diagonal(pcov))
    param_names = ['center', 'width', 'area', 'gamma_ratio']
    
    for i, peak_name in enumerate(peak_names):
        peak_data = {
            "name": peak_name,
            "expected_center_cm_inv": centers[i],
            "parameters": {}
        }
        for j, param_name in enumerate(param_names):
            idx = i * 4 + j
            peak_data["parameters"][param_name] = {
                "value": float(popt[idx]),
                "uncertainty": float(uncertainties[idx])
            }
        results["peaks"].append(peak_data)
    
    # Calculate quality metrics
    y_fit = multi_voigt_free_gamma(x, *popt)
    residuals = y - y_fit
    sse = np.sum(residuals**2)
    sst = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (sse / sst) if sst > 0 else np.nan
    
    results["quality_metrics"] = {
        "r_squared": float(r_squared),
        "sum_squared_error": float(sse),
        "mean_absolute_error": float(np.mean(np.abs(residuals)))
    }
    
    # Add loss function results if provided
    if loss_dict is not None:
        results["loss_function"] = {
            "loss": loss_dict.get('loss', np.nan),
            "loss_term_1": loss_dict.get('loss_term_1', np.nan),
            "loss_term_2": loss_dict.get('loss_term_2', np.nan),
            "area_2D": loss_dict.get('area_2D', np.nan),
            "area_G": loss_dict.get('area_G', np.nan),
            "area_B": loss_dict.get('area_B', np.nan),
            "area_L": loss_dict.get('area_L', np.nan),
            "total_area": loss_dict.get('sum_areas', np.nan),
            "description": {
                "loss_term_1": "Area(2D) / Area(G) - should be 0 for good growth",
                "loss_term_2": "Area(B) / Area(G) - should be 1 for structural"
            }
        }
    
    # Save JSON
    json_path = output_dir / "fitting_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved JSON results to: {json_path}")
    
    # Save CSV with parameter details
    csv_path = output_dir / "fitting_parameters.csv"
    with open(csv_path, 'w') as f:
        f.write("peak_name,parameter,value,uncertainty\n")
        for i, peak_name in enumerate(peak_names):
            for j, param_name in enumerate(param_names):
                idx = i * 4 + j
                f.write(f"{peak_name},{param_name},{popt[idx]:.6e},{uncertainties[idx]:.6e}\n")
    logger.info(f"Saved CSV results to: {csv_path}")


# ============================================================================
# DEFAULT PARAMETERS
# ============================================================================

# Hardcoded peak parameters for graphene Raman spectra
# Adjust these based on your material system (WS2, graphene, etc.)
peak_names = {
    # "D1": 1313, 
    # "D2a": 1363, 
    # "D2b": 1395,
    # "D2c": 1445,
    "D": 1360,
    "B": 1492, 
    "L": 1564, 
    "G": 1607, 
    "2D": 2735
    }
DEFAULT_PEAKS = {
    "centers": list(peak_names.values()),      # Raman shift positions (cm^-1)
    "widths": [40.0] * len(peak_names),       # Initial peak widths (cm^-1)
    "gamma_ratios": [0.01] * len(peak_names),       # Lorentzian fractions [0,1]
    "peak_names": list(peak_names.keys()),    # Peak identifiers
}

# Fitting constraints
DEFAULT_FITTING = {
    "center_windows": [20.0] * len(DEFAULT_PEAKS["peak_names"]),
    "min_widths": [25.0] * len(DEFAULT_PEAKS["peak_names"]),
    "max_widths": [250, 100, 50, 100, 250] ,
    "maxfev": 30000,
}

# Model and paths
DEFAULT_MODEL = "min70_highf"
DEFAULT_CKPTS_DIR = "./saved_models/checkpoints"

PENALTY_LOSS = 100.0  # Loss value to assign when fitting is skipped due to low signal
FIT_THRESHOLD = 0.05  # Maximum intensity threshold to decide if fitting should be performed


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SpectraFormer Multi-Map Raman Unmixing + Voigt Peak Fitting Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python infer_and_fit_raman_map.py --input "data/substrate_measurements/" --output "results/output"

Peak parameters are hardcoded. To modify, edit DEFAULT_PEAKS in the script.
        """
    )
    parser.add_argument(
        '--input', type=str, required=True,
        help='Path to folder containing .txt files with Raman spectral maps'
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Validate input
    input_folder = Path(args.input)
    if not input_folder.is_dir():
        logger.error(f"Input folder not found: {input_folder}")
        return
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpts_path = Path(DEFAULT_CKPTS_DIR)
    checkpoint = DEFAULT_MODEL
    
    # Get peak parameters from defaults
    centers = DEFAULT_PEAKS["centers"]
    widths = DEFAULT_PEAKS["widths"]
    gamma_ratios = DEFAULT_PEAKS["gamma_ratios"]
    peak_names = DEFAULT_PEAKS["peak_names"]
    
    center_windows = DEFAULT_FITTING["center_windows"]
    min_widths = DEFAULT_FITTING["min_widths"]
    max_widths = DEFAULT_FITTING["max_widths"]
    maxfev = DEFAULT_FITTING["maxfev"]
    
    amp_maxs = [None] * len(centers)
    n_peaks = len(centers)
    
    logger.info("="*70)
    logger.info("SpectraFormer Multi-Map Raman Unmixing + Voigt Peak Fitting Pipeline")
    logger.info("="*70)
    logger.info(f"Input folder: {input_folder}")
    logger.info(f"Model checkpoint: {checkpoint}")
    logger.info(f"Peak parameters: {peak_names}")
    logger.info("="*70)
    
    try:
        # ====== STEP 1: Load all .txt files from folder ======
        logger.info(f"Step 1: Loading all .txt files from {input_folder}")
        dataarrays = load_all_txt_files(input_folder)
        if not dataarrays:
            logger.error("No valid .txt files loaded")
            return
        logger.info(f"  ✓ Loaded {len(dataarrays)} Raman maps")
        
        # ====== STEP 2: Preprocess maps ======
        logger.info("Step 2: Preprocessing individual maps with whitaker_hayes_with_outliers")
        from spectraformer.input_pipeline import preprocess_dataset
        
        preprocessed_arrays = []
        for i, da in enumerate(dataarrays):
            logger.info(f"  Preprocessing map {i+1}/{len(dataarrays)}")
            da_prep = preprocess_dataset(da, option="whitaker_hayes_with_outliers")
            preprocessed_arrays.append(da_prep)
        
        # ====== STEP 3: Aggregate all preprocessed maps ======
        logger.info("Step 3: Aggregating all preprocessed maps into single dataset")
        dataset_preprocessed = aggregate_preprocessed_maps(preprocessed_arrays)
        logger.info(f"  ✓ Aggregated preprocessed shape: {dataset_preprocessed.shape}")
        
        # ====== STEP 4: Inference ======
        logger.info("Step 4: Running single SpectraFormer inference on aggregated dataset")
        predictions, configs, mask_windows = load_model_and_predict(
            checkpoint, dataset_preprocessed, ckpts_path
        )
        logger.info(f"  ✓ Inference complete. Generated {len(predictions)} predictions")
        
        # ====== STEP 5: Extract and compute median & std of predicted_difference ======
        logger.info("Step 5: Extracting median and std of predicted_difference")
        predicted_diffs = np.array([p["predicted_difference"] for p in predictions])
        median_diff = np.median(predicted_diffs, axis=0)
        std_diff = np.std(predicted_diffs, axis=0)
        
        logger.info(f"  ✓ Computed from {len(predictions)} spectra")
        logger.info(f"  ✓ Median spectrum shape: {median_diff.shape}")
        logger.info(f"  ✓ Value range: [{np.min(median_diff):.6f}, {np.max(median_diff):.6f}]")
        logger.info(f"  ✓ Std range: [{np.min(std_diff):.6f}, {np.max(std_diff):.6f}]")
        
        # Get wave_number axis
        wave_number = np.asarray(predictions[0]["wave_number"])
        if np.max(np.abs(wave_number)) < 10:
            wave_number = wave_number * 800 + 2000
        
        # ====== STEP 6: Validity check - skip fitting if no signal ======
        logger.info("Step 6: Checking validity of spectrum for fitting")
        max_intensity = np.max(median_diff)
        logger.info(f"  ✓ Maximum intensity: {max_intensity:.6f}")
        
        if max_intensity < FIT_THRESHOLD:
            logger.warning(F"  ✗ Maximum intensity < {FIT_THRESHOLD} - No material growth detected!")
            logger.warning(F"  ✗ Skipping fitting and assigning loss = {PENALTY_LOSS} (no growth penalty)")
            
            # Save results with loss=4 and no fitting
            save_results_no_fit(
                output_dir, centers, peak_names,
                wave_number, median_diff, predictions, configs,
                loss_value=PENALTY_LOSS, loss_reason=f"No material growth (max_intensity < {FIT_THRESHOLD})"
            )
            
            # Create visualization even without fitting
            viz_path = output_dir / "fitting_visualization.png"
            create_fit_visualization_no_fit(
                wave_number, median_diff, centers, peak_names, viz_path, std_diff
            )
            logger.info(f"  ✓ Saved visualization (no fit) to: {viz_path}")
            
            elapsed = time.time() - start_time
            logger.info("="*70)
            logger.info(f"✓ Pipeline completed (no fitting) in {elapsed:.2f} seconds")
            logger.info(f"✓ Results saved to: {output_dir}")
            logger.info(f"✓ Loss = {PENALTY_LOSS} (no growth)")
            logger.info("="*70)
            return
        
        # ====== STEP 7: Fit median spectrum ======
        logger.info("Step 7: Fitting median spectrum with Voigt profiles")
        popt, pcov, fit_status = fit_averaged_spectrum(
            wave_number, median_diff,
            centers, peak_names, widths, amp_maxs, gamma_ratios,
            center_windows, min_widths, max_widths,
            maxfev=maxfev
        )
        
        if popt is None:
            logger.error(f"  ✗ Fitting failed: {fit_status}")
            return
        
        logger.info(f"  ✓ {fit_status}")
        
        # ====== STEP 8: Calculate loss function ======
        logger.info("Step 8: Calculating loss function from peak areas")
        loss_dict = calculate_loss(popt, peak_names, centers)
        
        # ====== STEP 9: Save results and visualizations ======
        logger.info("Step 9: Saving results")
        save_results(
            output_dir, popt, pcov, centers, peak_names,
            wave_number, median_diff, predictions, configs, loss_dict
        )
        
        # Create visualization
        viz_path = output_dir / "fitting_visualization.png"
        create_fit_visualization(
            wave_number, median_diff, popt, centers, peak_names, viz_path, std_diff
        )
        logger.info(f"  ✓ Saved visualization to: {viz_path}")
        
        elapsed = time.time() - start_time
        logger.info("="*70)
        logger.info(f"✓ Pipeline completed successfully in {elapsed:.2f} seconds")
        logger.info(f"✓ Results saved to: {output_dir}")
        logger.info(f"✓ Loss = {loss_dict['loss']:.4f}")
        logger.info("="*70)
        
    except Exception as e:
        logger.exception(f"Pipeline failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
