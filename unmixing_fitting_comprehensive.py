# #!/usr/bin/env python3
# """
# WS₂ Photoluminescence Spectroscopy Analysis Pipeline
# Scientifically rigorous peak fitting with Voigt profiles and adaptive boundary checking

# Usage:
#     python PL_fitting_individual_comprehensive_2.py   --input ~/pl_data/main_20x20_0011_WS2_532_197_1_5_Copy.nc   --output ~/pl_results   --config <(echo '{"processing": {"max_workers": 2, "timeout_seconds": 60}}')"""

# import numpy as np
# import xarray as xr
# from scipy.optimize import curve_fit
# from scipy.special import wofz
# from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
# import multiprocessing
# import logging
# import time
# import gc
# from pathlib import Path
# from typing import List, Tuple, Sequence, Optional, Dict, Any, Union
# import os
# import json
# import argparse
# import sys
# import shutil

# # Configure logging with rotation to prevent excessive file growth
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S"
# )
# logger = logging.getLogger(__name__)

# class SpectroscopyDataError(Exception):
#     """Custom exception for spectroscopy data validation failures"""
#     pass

# def loss_term(xarr, target: float = 0.0):
#     median = xarr.median()
#     var = xarr.var()
#     logger.info(f"Median term loss: {((median - target)**2).values:.2f}, var term loss: {var.values:.2f}")
#     return ((median - target)**2 + var).values

# def validate_dataset_schema(ds: xr.Dataset, var_name: str) -> None:
#     """
#     Validate that the dataset has the required structure for PL analysis.
    
#     Parameters:
#     -----------
#     ds : xr.Dataset
#         Input dataset to validate
#     var_name : str
#         Name of the spectra variable to check
    
#     Raises:
#     -------
#     SpectroscopyDataError
#         If dataset doesn't meet requirements
#     """
#     # Check required dimensions
#     required_dims = ['photon_energy']
#     if not all(dim in ds.dims for dim in required_dims):
#         raise SpectroscopyDataError(
#             f"Dataset missing required dimensions. Found: {list(ds.dims)}, "
#             f"Required: {required_dims}"
#         )
    
#     # Check required variables
#     if var_name not in ds.data_vars:
#         raise SpectroscopyDataError(
#             f"Dataset missing required variable '{var_name}'. "
#             f"Available variables: {list(ds.data_vars)}"
#         )
    
#     # Check coordinate attributes
#     if 'units' not in ds['photon_energy'].attrs:
#         logger.warning("photon_energy coordinate missing 'units' attribute")
    
#     # Check data quality
#     spectra = ds[var_name]
#     if np.all(np.isnan(spectra.values)):
#         raise SpectroscopyDataError("All spectrum values are NaN")
    
#     if spectra.values.size == 0:
#         raise SpectroscopyDataError("Empty dataset - no spectra to process")
    
#     logger.info(f"Dataset validation passed. Dimensions: {dict(ds.dims)}")

# def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
#     """
#     Load configuration parameters from JSON file or use defaults.
    
#     Parameters:
#     -----------
#     config_path : Optional[Path]
#         Path to configuration file. If None, uses built-in defaults.
    
#     Returns:
#     --------
#     Dict[str, Any]
#         Configuration dictionary with all required parameters
#     """
#     default_config = {
#         "peaks": [
#             {"name": "A^0", "center": 2025, "width": 10, "amp_max": 5000, "gamma_ratio": 0.5},
#             {"name": "X^-", "center": 1975, "width": 10, "amp_max": 2500, "gamma_ratio": 0.5},
#             {"name": "X^D", "center": 1925, "width": 10, "amp_max": 1000, "gamma_ratio": 0.5}
#         ],
#         "fitting": {
#             "center_windows": [25, 25, 25],
#             "min_widths": [2, 2, 2],
#             "max_widths": [500, 500, 500],
#             "maxfev": 20000,
#             "tolerances": {
#                 "ftol": 1e-3,
#                 "xtol": 1e-3,
#                 "gtol": 1e-3
#             }
#         },
#         "processing": {
#             "max_workers": None,
#             "chunk_size": 50,
#             "timeout_seconds": 120
#         }
#     }
    
#     if config_path is None or not config_path.exists():
#         logger.info("Using default configuration")
#         return default_config
    
#     try:
#         with open(config_path, 'r') as f:
#             user_config = json.load(f)
        
#         # Merge user config with defaults
#         config = default_config.copy()
#         for key in user_config:
#             if key in config:
#                 if isinstance(config[key], dict) and isinstance(user_config[key], dict):
#                     config[key].update(user_config[key])
#                 else:
#                     config[key] = user_config[key]
#             else:
#                 config[key] = user_config[key]
        
#         logger.info(f"Loaded configuration from {config_path}")
#         return config
#     except Exception as e:
#         logger.warning(f"Error loading config file: {e}. Using defaults.")
#         return default_config

# def true_vectorized_voigt(x: np.ndarray, centers: np.ndarray, sigmas: np.ndarray, 
#                          gammas: np.ndarray, areas: np.ndarray) -> np.ndarray:
#     """
#     Truly vectorized Voigt profile computation using broadcasting.
    
#     Parameters:
#     -----------
#     x : np.ndarray
#         Energy axis (shape: (n_energy,))
#     centers : np.ndarray
#         Peak centers (shape: (n_spectra,))
#     sigmas : np.ndarray
#         Gaussian standard deviations (shape: (n_spectra,))
#     gammas : np.ndarray
#         Lorentzian half-widths (shape: (n_spectra,))
#     areas : np.ndarray
#         Peak areas (shape: (n_spectra,))
    
#     Returns:
#     --------
#     np.ndarray
#         Voigt profiles (shape: (n_spectra, n_energy))
#     """
#     # Reshape for broadcasting: (n_spectra, 1) vs (1, n_energy)
#     x_reshaped = x.reshape(1, -1)
#     centers_reshaped = centers.reshape(-1, 1)
#     sigmas_reshaped = sigmas.reshape(-1, 1)
#     gammas_reshaped = gammas.reshape(-1, 1)
    
#     # Compute z for all spectra and energies at once
#     z = ((x_reshaped - centers_reshaped) + 1j * gammas_reshaped) / (sigmas_reshaped * np.sqrt(2.0))
    
#     # Compute Faddeeva function for all values
#     w_values = wofz(z)
    
#     # Compute profiles
#     sqrt_2pi = np.sqrt(2.0 * np.pi)
#     profiles = np.real(w_values) / (sigmas_reshaped * sqrt_2pi)
    
#     # Scale by areas
#     return profiles * areas.reshape(-1, 1)

# def batch_voigt_profiles(
#     x: np.ndarray,
#     centers: np.ndarray,
#     widths: np.ndarray,
#     areas: np.ndarray,
#     gamma_fracs: np.ndarray,
#     min_width: float = 1e-10
# ) -> np.ndarray:
#     """
#     Vectorized computation of multiple Voigt profiles with shared energy axis.
#     Uses true vectorization via broadcasting for optimal performance.
    
#     Parameters:
#     -----------
#     x : np.ndarray
#         Energy axis (shape: (n_energy,))
#     centers : np.ndarray 
#         Peak centers (shape: (n_spectra,))
#     widths : np.ndarray
#         Voigt FWHM (shape: (n_spectra,))
#     areas : np.ndarray
#         Peak areas (shape: (n_spectra,))
#     gamma_fracs : np.ndarray
#         Lorentzian fractions (shape: (n_spectra,))
#     min_width : float
#         Minimum width for numerical stability
    
#     Returns:
#     --------
#     np.ndarray
#         Array of Voigt profiles (shape: (n_spectra, n_energy))
#     """
#     n_spectra = len(centers)
#     if n_spectra == 0:
#         return np.zeros((0, len(x)))
    
#     # Pre-compute constants
#     sqrt_2ln2 = np.sqrt(2.0 * np.log(2.0))
#     sqrt_2pi = np.sqrt(2.0 * np.pi)
    
#     # Clamp parameters to physical ranges
#     widths = np.maximum(widths, min_width)
#     gamma_fracs = np.clip(gamma_fracs, 0.0, 1.0)
    
#     # Pre-allocate parameters
#     sigma = np.empty(n_spectra, dtype=np.float64)
#     gamma = np.empty(n_spectra, dtype=np.float64)
    
#     # Handle all cases using vectorized operations
#     # Pure Gaussian profiles (gamma_frac ≈ 0)
#     is_gaussian = gamma_fracs < 1e-8
#     if np.any(is_gaussian):
#         sigma[is_gaussian] = widths[is_gaussian] / (2.0 * sqrt_2ln2)
#         gamma[is_gaussian] = 1e-15
    
#     # Pure Lorentzian profiles (gamma_frac ≈ 1)
#     is_lorentz = gamma_fracs > (1.0 - 1e-8)
#     if np.any(is_lorentz):
#         gamma[is_lorentz] = widths[is_lorentz] / 2.0
#         sigma[is_lorentz] = 1e-15
    
#     # Mixed profiles using proper Olivero-Longbothum decomposition
#     is_mixed = ~(is_gaussian | is_lorentz)
#     if np.any(is_mixed):
#         V = widths[is_mixed]
#         f = gamma_fracs[is_mixed]
        
#         # Correct decomposition: L = f*V, solve for G
#         L = f * V
#         term = V - 0.5346 * L
#         G_sq = np.maximum(0.0, term**2 - 0.2166 * L**2)
#         G = np.sqrt(G_sq)
        
#         # Convert to distribution parameters
#         sigma[is_mixed] = G / (2.0 * sqrt_2ln2)
#         gamma[is_mixed] = L / 2.0
    
#     # Final numerical safeguards
#     sigma = np.maximum(sigma, 1e-15)
#     gamma = np.maximum(gamma, 1e-15)
    
#     # True vectorized computation
#     result = true_vectorized_voigt(x, centers, sigma, gamma, areas)
    
#     return result

# def voigt_profile(x: np.ndarray, center: float, voigt_fwhm: float, area: float, 
#                  lorentz_frac: float, min_width: float = 1e-10) -> np.ndarray:
#     """
#     Area-normalized Voigt profile with exact physical parameterization.
#     This is kept for compatibility with curve_fit, but batch_voigt_profiles should be used for reconstruction.
#     """
#     return batch_voigt_profiles(
#         x.reshape(1, -1),
#         np.array([center]),
#         np.array([voigt_fwhm]),
#         np.array([area]),
#         np.array([lorentz_frac]),
#         min_width
#     )[0]

# def multi_voigt_free_gamma(x: np.ndarray, *flat_params: float) -> np.ndarray:
#     """
#     Sum of N Voigt peaks. flat_params should be multiple of 4: (center, width, area, gamma_ratio) per peak.
#     """
#     x = np.asarray(x)
#     y_model = np.zeros_like(x, dtype=float)
#     n_params = len(flat_params)
#     if n_params % 4 != 0:
#         raise ValueError("flat_params length must be a multiple of 4")
#     for i in range(0, n_params, 4):
#         c, w, A, r = flat_params[i:i + 4]
#         y_model += voigt_profile(x, c, w, A, r)
#     return y_model

# def build_initial_guesses_and_bounds(
#     x: np.ndarray, y: np.ndarray, centers: Sequence[float],
#     widths: Sequence[float], amp_maxs: Sequence[Optional[float]],
#     gamma_ratios: Sequence[float], center_windows: Sequence[float],
#     min_widths: Sequence[float], max_widths: Sequence[float],
#     config: Dict[str, Any]
# ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
#     """Build initial parameter guesses and bounds for curve fitting."""
#     p0_list, lb_list, ub_list = [], [], []
#     fitting_config = config.get('fitting', {})
#     tol_config = fitting_config.get('tolerances', {})
    
#     for i, c0 in enumerate(centers):
#         width = widths[i]
#         amp_max = amp_maxs[i]
#         gamma_ratio = gamma_ratios[i]
#         center_window = center_windows[i]
#         min_width = min_widths[i]
#         max_width = max_widths[i]
#         ymax = np.nanmax(y) if amp_max is None else amp_max
        
#         # Ensure numerical stability
#         if not np.isfinite(ymax) or ymax <= 0:
#             ymax = 1.0
        
#         # Center
#         p0_list.append(float(c0))
#         lb_list.append(float(c0 - center_window))
#         ub_list.append(float(c0 + center_window))
        
#         # Width (ensure physical bounds)
#         width = max(min_width, min(width, max_width))
#         p0_list.append(float(width))
#         lb_list.append(float(min_width))
#         ub_list.append(float(max_width))
        
#         # Area with physics-based initialization
#         height_guess = float(max(1e-12, ymax))
        
#         # Correct area factor for Voigt profiles
#         if gamma_ratio < 0.1:  # Near-Gaussian
#             area_factor = 1.064467  # sqrt(pi/(4*ln(2)))
#         elif gamma_ratio > 0.9:  # Near-Lorentzian
#             area_factor = np.pi / 2  # pi/2 for Lorentzian
#         else:  # Mixed profile - non-linear interpolation
#             G_factor = 1.064467
#             L_factor = np.pi / 2
#             # Non-linear weighting based on Voigt properties
#             area_factor = G_factor * (1 - gamma_ratio)**1.5 + L_factor * gamma_ratio**1.5
        
#         A0 = max(1e-12, height_guess * width * area_factor)
#         p0_list.append(A0)
#         lb_list.append(0.0)
#         ub_list.append(1e6 * A0)
        
#         # Gamma ratio (ensure physical bounds)
#         gamma_ratio = np.clip(gamma_ratio, 0.0, 1.0)
#         p0_list.append(float(gamma_ratio))
#         lb_list.append(0.0)
#         ub_list.append(1.0)
        
#     p0 = np.array(p0_list, dtype=float)
#     lb = np.array(lb_list, dtype=float)
#     ub = np.array(ub_list, dtype=float)
#     return p0, (lb, ub)

# def fit_one_spectrum(task_args):
#     """Fit a single spectrum with Voigt profiles."""
#     (x, y, centers, widths, amp_maxs, gamma_ratios, center_windows, min_widths, max_widths, config) = task_args
    
#     # Skip fitting if spectrum is invalid
#     if not np.any(np.isfinite(y)) or np.all(y == 0) or np.all(np.isnan(y)):
#         logger.debug("Skipping invalid spectrum")
#         n_params = len(centers) * 4
#         return np.full(n_params, np.nan), np.full((n_params, n_params), np.nan)
    
#     n_peaks = len(centers)
#     p0, bounds = build_initial_guesses_and_bounds(
#         x, y, centers, widths, amp_maxs, gamma_ratios, center_windows, min_widths, max_widths, config
#     )
    
#     fitting_config = config.get('fitting', {})
#     maxfev = fitting_config.get('maxfev', 20000)
#     tol_config = fitting_config.get('tolerances', {})
    
#     try:
#         # Capture both popt and pcov
#         popt, pcov = curve_fit(
#             multi_voigt_free_gamma, x, y,
#             p0=p0, bounds=bounds, maxfev=maxfev,
#             ftol=tol_config.get('ftol', 1e-5),
#             xtol=tol_config.get('xtol', 1e-5),
#             gtol=tol_config.get('gtol', 1e-5)
#         )
#         return popt, pcov
#     except Exception as exc:
#         logger.debug("curve_fit failed: %s", exc)
#         n_params = len(p0)
#         return np.full_like(p0, np.nan), np.full((n_params, n_params), np.nan)

# def check_fit_at_boundary(
#     fit_results: np.ndarray,
#     centers: Sequence[float],
#     center_windows: Sequence[float],
#     min_widths: Sequence[float],
#     max_widths: Sequence[float],
#     energy_resolution: float,
#     width_scale: float,
#     tol_rel: float = 0.01
# ) -> dict:
#     """
#     Check if fitted parameters are at the boundary constraints with adaptive tolerances.
#     """
#     n_peaks = len(centers)
#     param_names = ['center', 'width', 'area', 'gamma_ratio']
#     boundary_flags = {}
#     boundary_details = []
#     boundary_params = []
#     at_boundary = False
    
#     # Adaptive tolerances based on experimental conditions
#     center_tol_base = max(0.5, 2 * energy_resolution)  # 2x instrument resolution
#     width_tol_base = max(0.2, 0.1 * width_scale)       # 10% of typical peak width
    
#     for i in range(n_peaks):
#         idx_base = i * 4
#         center_val = fit_results[idx_base]
#         width_val = fit_results[idx_base + 1]
#         area_val = fit_results[idx_base + 2]
#         gamma_val = fit_results[idx_base + 3]
        
#         # Skip if parameters are invalid
#         if not (np.isfinite(center_val) and np.isfinite(width_val) and 
#                 np.isfinite(area_val) and np.isfinite(gamma_val)):
#             boundary_details.append(f"Peak {i}: Invalid parameters (NaN/Inf)")
#             boundary_params.append(f"invalid_{i}")
#             boundary_flags[f"invalid_{i}"] = 'invalid'
#             at_boundary = True
#             continue
        
#         # Define constraint boundaries
#         center_lower = centers[i] - center_windows[i]
#         center_upper = centers[i] + center_windows[i]
#         width_lower = min_widths[i]
#         width_upper = max_widths[i]
        
#         # Adaptive tolerances
#         center_tol = max(center_tol_base, tol_rel * center_windows[i])
#         width_tol = max(width_tol_base, tol_rel * (width_upper - width_lower))
#         gamma_tol = max(1e-3, tol_rel)  # Minimum tolerance for fractions
        
#         # Check center
#         if center_val < center_lower + center_tol:
#             boundary_details.append(f"Peak {i}: center ({center_val:.2f}) near lower boundary ({center_lower:.2f})")
#             boundary_params.append(f"center_{i}")
#             boundary_flags[f"center_{i}"] = 'lower'
#             at_boundary = True
#         elif center_val > center_upper - center_tol:
#             boundary_details.append(f"Peak {i}: center ({center_val:.2f}) near upper boundary ({center_upper:.2f})")
#             boundary_params.append(f"center_{i}")
#             boundary_flags[f"center_{i}"] = 'upper'
#             at_boundary = True
        
#         # Check width
#         if width_val < width_lower + width_tol:
#             boundary_details.append(f"Peak {i}: width ({width_val:.2f}) near min boundary ({width_lower:.2f})")
#             boundary_params.append(f"width_{i}")
#             boundary_flags[f"width_{i}"] = 'lower'
#             at_boundary = True
#         elif width_val > width_upper - width_tol:
#             boundary_details.append(f"Peak {i}: width ({width_val:.2f}) near max boundary ({width_upper:.2f})")
#             boundary_params.append(f"width_{i}")
#             boundary_flags[f"width_{i}"] = 'upper'
#             at_boundary = True
        
#         # Check area (only lower boundary meaningful)
#         if area_val < 1e-6:
#             boundary_details.append(f"Peak {i}: area ({area_val:.2e}) near zero boundary")
#             boundary_params.append(f"area_{i}")
#             boundary_flags[f"area_{i}"] = 'lower'
#             at_boundary = True
        
#         # Check gamma_ratio
#         if gamma_val < gamma_tol:
#             boundary_details.append(f"Peak {i}: gamma_ratio ({gamma_val:.4f}) near zero boundary")
#             boundary_params.append(f"gamma_ratio_{i}")
#             boundary_flags[f"gamma_ratio_{i}"] = 'lower'
#             at_boundary = True
#         elif gamma_val > 1.0 - gamma_tol:
#             boundary_details.append(f"Peak {i}: gamma_ratio ({gamma_val:.4f}) near unity boundary")
#             boundary_params.append(f"gamma_ratio_{i}")
#             boundary_flags[f"gamma_ratio_{i}"] = 'upper'
#             at_boundary = True
    
#     return {
#         'at_boundary': at_boundary,
#         'boundary_details': boundary_details,
#         'boundary_params': boundary_params,
#         'boundary_flags': boundary_flags
#     }

# def fit_one_spectrum_with_index(task):
#     """Wrapper that preserves index information for parallel execution"""
#     idx, x, y, centers, widths, amp_maxs, gamma_ratios, center_windows, min_widths, max_widths, config = task
#     popt, pcov = fit_one_spectrum((x, y, centers, widths, amp_maxs, gamma_ratios, center_windows, min_widths, max_widths, config))
#     return idx, popt, pcov

# def create_comprehensive_fit_dataset(fit_results, covariance_results, original_ds, peak_names, param_names, spatial_dims, spatial_shape, boundary_array):
#     """
#     Create a comprehensive dataset with fitting results, reconstructed spectra, and quality metrics.
#     """
#     # Create the main dataset
#     fit_ds = xr.Dataset()
    
#     # Get coordinates
#     x_coords = {dim: original_ds[dim] for dim in spatial_dims}
#     energy_coord = original_ds['photon_energy']
    
#     # Store fitting parameters
#     fit_ds['fitting_results'] = xr.DataArray(
#         fit_results,
#         dims=spatial_dims + ["peak", "param"],
#         coords={
#             **x_coords,
#             "peak": peak_names,
#             "param": param_names
#         }
#     )
    
#     # Store full covariance matrices
#     fit_ds['covariance_matrix'] = xr.DataArray(
#         covariance_results,
#         dims=spatial_dims + ["cov_param1", "cov_param2"],
#         coords={
#             **x_coords,
#             "cov_param1": [f"{peak}_{p}" for peak in peak_names for p in param_names],
#             "cov_param2": [f"{peak}_{p}" for peak in peak_names for p in param_names]
#         }
#     )
    
#     # Also store parameter uncertainties (diagonal of covariance)
#     uncertainties = np.sqrt(np.diagonal(covariance_results, axis1=-2, axis2=-1))
#     fit_ds['parameter_uncertainties'] = xr.DataArray(
#         uncertainties,
#         dims=spatial_dims + ["uncert_param"],
#         coords={
#             **x_coords,
#             "uncert_param": [f"{peak}_{p}" for peak in peak_names for p in param_names]
#         }
#     )
    
#     # Store boundary check results
#     fit_ds['fit_at_boundary'] = xr.DataArray(
#         boundary_array,
#         dims=spatial_dims,
#         coords=x_coords
#     )
    
#     # Extract parameters for easier access
#     peak_params = {}
#     for i, peak in enumerate(peak_names):
#         peak_params[peak] = {
#             'center': fit_ds['fitting_results'].sel(peak=peak, param='center'),
#             'width': fit_ds['fitting_results'].sel(peak=peak, param='width'),
#             'area': fit_ds['fitting_results'].sel(peak=peak, param='area'),
#             'gamma_ratio': fit_ds['fitting_results'].sel(peak=peak, param='gamma_ratio')
#         }
    
#     # Reconstruct fitted spectra
#     x_values = energy_coord.values * 1000  # Convert to meV for fitting
#     fitted_spectra = np.zeros((*spatial_shape, len(x_values)))
    
#     # Calculate individual peak components and total fit
#     peak_components = {}
#     for peak in peak_names:
#         peak_components[peak] = np.zeros((*spatial_shape, len(x_values)))
    
#     # Iterate over all spatial positions
#     spatial_indices = np.ndindex(spatial_shape)
#     for idx in spatial_indices:
#         # Get parameters for all peaks at this position
#         all_params = []
#         for peak in peak_names:
#             params = [
#                 float(peak_params[peak]['center'].values[idx]),
#                 float(peak_params[peak]['width'].values[idx]),
#                 float(peak_params[peak]['area'].values[idx]),
#                 float(peak_params[peak]['gamma_ratio'].values[idx])
#             ]
#             all_params.extend(params)
        
#         # Calculate total fit
#         fitted_spectra[idx] = multi_voigt_free_gamma(x_values, *all_params)
        
#         # Calculate individual components
#         for i, peak in enumerate(peak_names):
#             peak_params_single = all_params[i*4:(i+1)*4]
#             peak_components[peak][idx] = voigt_profile(x_values, *peak_params_single)
    
#     # Add reconstructed spectra to dataset
#     fit_ds['fitted_spectrum'] = xr.DataArray(
#         fitted_spectra,
#         dims=spatial_dims + ['photon_energy'],
#         coords={**x_coords, 'photon_energy': energy_coord}
#     )
    
#     # Add raw data
#     fit_ds['raw_spectrum'] = original_ds['spectra']
    
#     # Add individual peak components
#     for peak in peak_names:
#         fit_ds[f'{peak}_component'] = xr.DataArray(
#             peak_components[peak],
#             dims=spatial_dims + ['photon_energy'],
#             coords={**x_coords, 'photon_energy': energy_coord}
#         )
    
#     # Calculate residuals and quality metrics
#     residuals = fit_ds['raw_spectrum'] - fit_ds['fitted_spectrum']
#     fit_ds['residuals'] = residuals
    
#     # Goodness-of-fit metrics
#     sse = np.sum(residuals**2, axis=-1)  # Sum of squared errors
#     sst = np.sum((fit_ds['raw_spectrum'] - np.mean(fit_ds['raw_spectrum'], axis=-1))**2, axis=-1)  # Total sum of squares
    
#     fit_ds['sse'] = xr.DataArray(sse, dims=spatial_dims, coords=x_coords)
#     fit_ds['sst'] = xr.DataArray(sst, dims=spatial_dims, coords=x_coords)
#     fit_ds['r_squared'] = 1 - (sse / sst)
    
#     n = len(x_values)
#     p = len(peak_names) * len(param_names)
#     fit_ds['adj_r_squared'] = 1 - (1 - fit_ds['r_squared']) * ((n - 1) / (n - p - 1))
    
#     # Add parameter maps for easy access
#     for peak in peak_names:
#         for param in param_names:
#             param_data = fit_ds['fitting_results'].sel(peak=peak, param=param)
#             fit_ds[f'{peak}_{param}'] = param_data
    
#     # Calculate some useful ratios
#     if 'A^0' in peak_names and 'X^-' in peak_names:
#         xd_area = fit_ds['X^D_area']
#         a0_area = fit_ds['A^0_area']
#         xminus_area = fit_ds['X^-_area']
#         fit_ds['Xminus_to_total_ratio'] = xminus_area / ( xd_area + a0_area + xminus_area )
#     if 'A^0' in peak_names and 'X^-' in peak_names and 'X^D' in peak_names:
#         xd_area = fit_ds['X^D_area']
#         a0_area = fit_ds['A^0_area']
#         xminus_area = fit_ds['X^-_area']
#         fit_ds['XD_to_total_ratio'] = xd_area / ( xd_area + a0_area + xminus_area )
    
#     logger.info(f"Created comprehensive dataset with dimensions: {dict(fit_ds.dims)}")
#     logger.info(f"Variables: {list(fit_ds.data_vars)}")
    
#     return fit_ds

# def get_wsl_safe_path(base_path: Path) -> Path:
#     """Get a WSL-safe path for temporary files"""
#     if 'WSL_DISTRO_NAME' in os.environ:
#         # Use Linux filesystem for temporary files
#         return Path('/tmp') / base_path.name
#     return base_path

# def fit_all_spectra_parallel(
#     ds: xr.Dataset, var: str,
#     centers, widths, amp_maxs, gamma_ratios, center_windows, min_widths, max_widths,
#     config: Dict[str, Any]
# ):
#     """Parallel fitting of spectra with efficient processing and adaptive parameters."""
#     processing_config = config.get('processing', {})
#     max_workers = processing_config.get('max_workers')
#     timeout_seconds = processing_config.get('timeout_seconds', 120)
    
#     # Validate and handle energy units
#     energy_units = ds['photon_energy'].attrs.get('units', 'eV').lower()
#     logger.info(f"Energy units: {energy_units}")
    
#     if energy_units in ['ev', 'electronvolt', 'electronvolts']:
#         x = ds['photon_energy'].values * 1000  # eV to meV
#         logger.info("Converted energy from eV to meV")
#     elif energy_units in ['mev', 'millielectronvolt', 'millielectronvolts']:
#         x = ds['photon_energy'].values
#         logger.info("Energy already in meV")
#     else:
#         logger.warning(f"Unknown energy units: '{energy_units}'. Assuming eV and converting to meV.")
#         x = ds['photon_energy'].values * 1000
    
#     # Estimate instrument resolution from energy axis
#     energy_resolution = np.diff(x).mean() if len(x) > 1 else 1.0
#     typical_width = np.mean(widths) if widths else 10.0
#     logger.info(f"Instrument resolution: {energy_resolution:.3f} meV, Typical peak width: {typical_width:.1f} meV")
    
#     var_dims = list(ds[var].dims)
#     spatial_dims = [d for d in var_dims if d != 'photon_energy']
#     spatial_shape = tuple(ds.sizes[d] for d in spatial_dims)
#     y_data = ds[var].values.reshape(-1, x.size)
    
#     # Determine optimal workers with safety limits
#     if max_workers is None:
#         physical_cores = multiprocessing.cpu_count()
#         max_workers = max(1, min(physical_cores, 8))  # Cap at 8 for stability
#         logger.info(f"Auto-detected {physical_cores} cores. Using {max_workers} workers.")
    
#     # Prepare tasks with explicit indexing
#     tasks = [
#         (i, x, y_data[i], centers, widths, amp_maxs, gamma_ratios, 
#          center_windows, min_widths, max_widths, config)
#         for i in range(y_data.shape[0])
#     ]
    
#     # Initialize results arrays for both popt and pcov
#     # Annotate types so static checkers (e.g. pylance) know these lists may hold ndarrays/dicts later
#     popt_results: List[Optional[np.ndarray]] = [None] * len(tasks)
#     pcov_results: List[Optional[np.ndarray]] = [None] * len(tasks)
#     boundary_checks: List[Optional[dict]] = [None] * len(tasks)
    
#     logger.info(f"Starting parallel fitting of {len(tasks)} spectra with {max_workers} workers")
    
#     # Single executor for all tasks
#     try:
#         with ProcessPoolExecutor(max_workers=max_workers) as executor:
#             # Submit all tasks
#             future_to_idx = {
#                 executor.submit(fit_one_spectrum_with_index, task): task[0] 
#                 for task in tasks
#             }
            
#             # Process completed tasks as they finish
#             completed = 0
#             start_time = time.time()
#             for future in as_completed(future_to_idx):
#                 idx = future_to_idx[future]
#                 try:
#                     orig_idx, popt, pcov = future.result(timeout=timeout_seconds)
#                     popt_results[orig_idx] = popt
#                     pcov_results[orig_idx] = pcov
                    
#                     # Check boundaries with adaptive tolerances (using popt only)
#                     boundary_check = check_fit_at_boundary(
#                         popt, centers, center_windows, min_widths, max_widths,
#                         energy_resolution, typical_width
#                     )
#                     boundary_checks[orig_idx] = boundary_check
                    
#                     completed += 1
#                     elapsed = time.time() - start_time
                    
#                     # Progress reporting
#                     if True:
#                         remaining = (elapsed / completed) * (len(tasks) - completed)
#                         logger.info(
#                             f"Completed {completed}/{len(tasks)} spectra "
#                             f"({completed/len(tasks)*100:.1f}%) - "
#                             f"Est. remaining: {remaining/60:.3f} min ({remaining:.2f} sec)"
#                         )
                    
#                     # Memory cleanup periodically
#                     if completed % 50 == 0:
#                         gc.collect()
                        
#                 except TimeoutError:
#                     logger.error(f"Spectrum {idx} timed out after {timeout_seconds} seconds")
#                     n_params = len(centers) * 4
#                     popt_results[idx] = np.full(n_params, np.nan)
#                     pcov_results[idx] = np.full((n_params, n_params), np.nan)
#                     boundary_checks[idx] = {
#                         'at_boundary': True,
#                         'boundary_details': ["Processing timed out"],
#                         'boundary_params': ['timeout'],
#                         'boundary_flags': {'timeout': 'timeout'}
#                     }
#                 except Exception as exc:
#                     logger.error(f"Spectrum {idx} failed with exception: {exc}")
#                     n_params = len(centers) * 4
#                     popt_results[idx] = np.full(n_params, np.nan)
#                     pcov_results[idx] = np.full((n_params, n_params), np.nan)
#                     boundary_checks[idx] = {
#                         'at_boundary': True,
#                         'boundary_details': [f"Exception: {exc}"],
#                         'boundary_params': ['exception'],
#                         'boundary_flags': {'exception': 'exception'}
#                     }
#                 finally:
#                     # Force garbage collection after each task to prevent memory leaks
#                     gc.collect()
    
#     except Exception as e:
#         logger.critical(f"Parallel execution failed: {e}")
#         raise
    
#     # Handle any unprocessed tasks (shouldn't happen with proper error handling)
#     for i, (popt, pcov) in enumerate(zip(popt_results, pcov_results)):
#         if popt is None or pcov is None:
#             logger.warning(f"Task {i} not processed - falling back to serial")
#             task = tasks[i]
#             _, x, y, *params = task
#             popt, pcov = fit_one_spectrum((x, y, *params, config))
#             boundary_check = check_fit_at_boundary(
#                 popt, centers, center_windows, min_widths, max_widths,
#                 energy_resolution, typical_width
#             )
#             popt_results[i] = popt
#             pcov_results[i] = pcov
#             boundary_checks[i] = boundary_check
#             gc.collect()
    
#     # Convert to numpy arrays and reshape
#     n_peaks = len(centers)
#     n_params = n_peaks * 4
    
#     # Reshape popt results
#     nan_popt = np.full(n_params, np.nan)
#     popt_arr = np.asarray([r if r is not None else nan_popt for r in popt_results], dtype=float)
#     popt_arr = popt_arr.reshape(*spatial_shape, n_peaks, 4)
    
#     # Reshape pcov results  
#     nan_pcov = np.full((n_params, n_params), np.nan)
#     pcov_arr = np.asarray([r if r is not None else nan_pcov for r in pcov_results], dtype=float)
#     pcov_arr = pcov_arr.reshape(*spatial_shape, n_params, n_params)
    
#     # Build boundary array (unchanged)
#     boundary_array = np.array([
#         bool(check['at_boundary']) if check is not None else True
#         for check in boundary_checks
#     ], dtype=bool).reshape(spatial_shape)
    
#     # Store only minimal boundary information to save memory
#     minimal_boundary_info = [
#         {'at_boundary': check['at_boundary']} if check is not None else {'at_boundary': True}
#         for check in boundary_checks
#     ]
    
#     return popt_arr, pcov_arr, spatial_dims, spatial_shape, boundary_array, minimal_boundary_info

# def main():
#     """Main entry point with command-line argument parsing"""
#     parser = argparse.ArgumentParser(description='WS₂ Photoluminescence Analysis')
#     parser.add_argument('--input', type=str, required=True, help='Input NetCDF file path')
#     parser.add_argument('--config', type=str, help='Configuration JSON file path')
#     parser.add_argument('--output', type=str, required=True, help='Output directory path')
#     parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    
#     args = parser.parse_args()
    
#     # Set up multiprocessing safely
#     try:
#         # Try to use fork for best performance
#         multiprocessing.set_start_method('fork', force=True)
#     except (RuntimeError, ValueError):
#         # Fall back to spawn if fork is unavailable
#         try:
#             multiprocessing.set_start_method('spawn', force=True)
#             logger.info("Using spawn start method for multiprocessing")
#         except RuntimeError:
#             # If all else fails, continue with whatever is available
#             logger.warning("Could not set multiprocessing start method - using default")
    
#     start_time = time.time()
    
#     try:
#         # Validate and load input file
#         input_path = Path(args.input)
#         if not input_path.exists():
#             raise FileNotFoundError(f"Input file not found: {input_path}")
        
#         output_dir = Path(args.output)
#         output_dir.mkdir(parents=True, exist_ok=True)
        
#         # Load configuration
#         config_path = Path(args.config) if args.config else None
#         config = load_config(config_path)
        
#         # Extract peak parameters from config
#         peaks = config['peaks']
#         peak_names = [peak['name'] for peak in peaks]
#         centers = [peak['center'] for peak in peaks]
#         widths = [peak['width'] for peak in peaks]
#         amp_maxs = [peak['amp_max'] for peak in peaks]
#         gamma_ratios = [peak['gamma_ratio'] for peak in peaks]
        
#         # Get fitting parameters
#         fitting_config = config.get('fitting', {})
#         center_windows = fitting_config.get('center_windows', [50] * len(peaks))
#         min_widths = fitting_config.get('min_widths', [2] * len(peaks))
#         max_widths = fitting_config.get('max_widths', [500] * len(peaks))
        
#         # Load data with validation
#         logger.info(f"Loading data from {input_path}")
#         da_map = xr.load_dataarray(input_path)
#         ds_map = da_map.to_dataset(name='spectra')
        
#         # Add units metadata if missing
#         if 'units' not in ds_map['photon_energy'].attrs:
#             ds_map['photon_energy'].attrs['units'] = 'eV'
#             logger.warning("Added missing 'units' attribute to photon_energy coordinate (assumed eV)")
        
#         # Validate dataset schema
#         validate_dataset_schema(ds_map, 'spectra')
        
#         # Perform fitting
#         results, covariances, spatial_dims, spatial_shape, boundary_array, boundary_info = fit_all_spectra_parallel(
#             ds_map, 'spectra',
#             centers, widths, amp_maxs, gamma_ratios, center_windows, min_widths, max_widths,
#             config
#         )
        
#         fit_time = time.time() - start_time
#         logger.info(f"Fitting completed in {fit_time:.2f} seconds ({fit_time/60:.2f} minutes)")
        
#         # Create comprehensive dataset
#         param_names = ['center', 'width', 'area', 'gamma_ratio']
#         comprehensive_ds = create_comprehensive_fit_dataset(
#             results, covariances, ds_map, peak_names, param_names, spatial_dims, spatial_shape, boundary_array
#         )
        
#         comprehensive_ds.attrs['loss'] = (
#             loss_term(comprehensive_ds['Xminus_to_total_ratio'], target=0) + \
#             loss_term(comprehensive_ds['XD_to_total_ratio'], target=0)
#         )**2
#         logger.info(f"Calculated loss: {comprehensive_ds.attrs['loss']}")
        
#         # Save results with WSL optimization
#         timestamp = time.strftime("%Y%m%d_%H%M%S")
#         base_name = input_path.stem
#         out_name = output_dir / f"{base_name}_comprehensive_fit_results_parallel_{timestamp}.nc"
        
#         # Use WSL-safe path for temporary storage
#         temp_path = get_wsl_safe_path(out_name)
        
#         logger.info(f"Saving results to {out_name}")
#         try:
#             comprehensive_ds.to_netcdf(temp_path)
#             if temp_path != out_name:
#                 # Move from temp location to final destination
#                 shutil.move(str(temp_path), str(out_name))
#             logger.info(f"Results saved successfully to {out_name}")
#         except Exception as e:
#             logger.error(f"Error saving file: {e}")
#             # Final fallback
#             comprehensive_ds.to_netcdf(out_name)
#             logger.info(f"Results saved to {out_name} using fallback method")
        
#         # Print summary statistics
#         total_pixels = boundary_array.size
#         boundary_pixels = np.sum(boundary_array)
#         logger.info(f"Boundary fits: {boundary_pixels}/{total_pixels} ({boundary_pixels/total_pixels*100:.1f}%)")
        
#         # Sample boundary details for diagnostics (only if boundary_info contains details)
#         if boundary_pixels > 0:
#             # Get detailed boundary checks for a sample of boundary pixels
#             boundary_indices = np.where(boundary_array)
#             sample_size = min(5, len(boundary_indices[0]))
#             logger.info(f"Sample boundary issues (first {sample_size} pixels):")
#             for i in range(sample_size):
#                 idx_flat = boundary_indices[0][i] * spatial_shape[1] + boundary_indices[1][i] if len(spatial_shape) > 1 else boundary_indices[0][i]
#                 if idx_flat < len(boundary_info) and 'boundary_details' in boundary_info[idx_flat]:
#                     logger.info(f"  Pixel {tuple(dim[i] for dim in boundary_indices)}: {boundary_info[idx_flat]['boundary_details']}")
#                 else:
#                     logger.info(f"  Pixel {tuple(dim[i] for dim in boundary_indices)}: Boundary flag set (details not stored to save memory)")
        
#         if 'Xminus_to_total_ratio' in comprehensive_ds:
#             ratio = comprehensive_ds['Xminus_to_total_ratio']
#             median_ratio = np.nanmedian(ratio.values)
#             std_ratio = np.nanstd(ratio.values)
#             logger.info(f"X-/total ratio - Median: {median_ratio:.3f}, Std: {std_ratio:.3f}")
            
#             # Physical interpretation
#             if not np.isnan(median_ratio):
#                 if median_ratio > 0.7:
#                     logger.warning("High trion fraction (>70%) - may indicate doping or defect issues")
#                 elif median_ratio < 0.3:
#                     logger.info("Low trion fraction (<30%) - indicates high sample quality")
        
#         # Performance summary
#         total_time = time.time() - start_time
#         spectra_per_sec = results.size / total_time
#         logger.info("=== PERFORMANCE SUMMARY ===")
#         logger.info(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
#         logger.info(f"Throughput: {spectra_per_sec:.1f} spectra/second")
#         logger.info(f"Average adj_r_squared: {np.nanmean(comprehensive_ds['adj_r_squared'].values):.3f}")
#         logger.info("Processing completed successfully")
    
#     except Exception as e:
#         logger.exception(f"Critical error during processing: {e}")
#         sys.exit(1)
    
#     finally:
#         # Final memory cleanup
#         gc.collect()
#         logger.info("Memory cleaned up")

# if __name__ == "__main__":
#     main()