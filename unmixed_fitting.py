
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import wofz
import xarray as xr
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def voigt_profile(x, center, width, area, gamma_ratio):

    sigma = width / (2 * np.sqrt(2 * np.log(2)))
    gamma = width * gamma_ratio / 2.0
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
    voigt = np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
    norm = np.trapz(voigt, x)
    if norm <= 0 or not np.isfinite(norm):
        sigma = max(sigma, 1e-6)
        voigt = np.exp(-0.5 * ((x - center) / sigma) ** 2)
        norm = np.trapz(voigt, x)
    voigt /= norm
    return area * voigt


def multi_voigt_free_gamma(x, *flat_params):
    y_model = np.zeros_like(x, dtype=float)
    for i in range(0, len(flat_params), 4):
        c, w, A, r = flat_params[i:i+4]
        y_model += voigt_profile(x, c, w, A, r)
    return y_model


def build_initial_guesses_and_bounds(x, y, centers, width, amp_max, gamma_ratio, center_window, min_width=1.0, max_width=400.0):
    """
    min_width, max_width: constraints for the width (FWHM) parameter for all peaks
    """
    GAUSS_AREA_FACTOR = 1.0644670194312262
    ymax = np.nanmax(y) if (hasattr(y, "size") and y.size) else (amp_max if amp_max is not None else 1.0)
    p0, lb, ub = [], [], []
    for c0 in centers:
        p0 += [c0];                 lb += [c0 - center_window]; ub += [c0 + center_window]
        p0 += [width];              lb += [min_width];          ub += [max_width]
        height_guess = (amp_max if amp_max is not None else ymax)
        A0 = max(1e-8, height_guess * width * GAUSS_AREA_FACTOR)
        p0 += [A0];                 lb += [0.0];                ub += [1e3 * A0]
        p0 += [gamma_ratio];        lb += [0.0];                ub += [1.0]
    return np.array(p0, float), (np.array(lb, float), np.array(ub, float))


def unpack_params(flat_params):
    return [flat_params[i:i+4] for i in range(0, len(flat_params), 4)]


# --- Fitting parameters ---
rough_centers = [1320, 1370, 1485, 1556, 1606, 2735]
rough_width   = 40.0
min_width = 0.0
max_width = 60.0
rough_amp_max = 0.09
rough_gamma_ratio = 0.5
center_window = 10.0                    # +/- to the rough center
param_names = ['center', 'width', 'area', 'gamma_ratio']
peak_names = ['D1', 'D2', 'B', 'L', 'G', '2D']

def fit_one_spectrum(args):
    x, y, rough_centers, rough_width, rough_amp_max, rough_gamma_ratio, center_window, min_width, max_width = args
    p0, bounds = build_initial_guesses_and_bounds(
        x, y,
        centers=rough_centers,
        width=rough_width,
        amp_max=rough_amp_max,
        gamma_ratio=rough_gamma_ratio,
        center_window=center_window,
        min_width=min_width,
        max_width=max_width
    )
    try:
        popt, _ = curve_fit(
            multi_voigt_free_gamma, x, y,
            p0=p0, bounds=bounds, maxfev=10000
        )
        return popt
    except Exception:
        return np.full_like(p0, np.nan)

def fit_all_spectra(ds, var='predicted_difference', n_peaks=5, min_width=1.0, max_width=400.0):
    x = ds['wave_number'].values
    spatial_dims = [dim for dim in ds[var].dims if dim != 'wave_number']
    spatial_shape = [ds.coords[dim].size for dim in spatial_dims]
    y_data = ds[var].values.reshape(-1, len(x))
    args_list = [
        (x, y, rough_centers[:n_peaks], rough_width, rough_amp_max, rough_gamma_ratio, center_window, min_width, max_width)
        for y in y_data
    ]
    # Use all CPU cores for parallel fitting
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(fit_one_spectrum, args_list), total=len(args_list), desc='Fitting spectra'))
    results = np.array(results).reshape(*spatial_shape, n_peaks*4)
    return results, spatial_dims, spatial_shape

if __name__ == "__main__":
    unmixed_datapath = 'data/unmixed_spatial/min67_highf/buffer+graphene/RUN3REC2_Buffer_20241011_1/5_5_1/unmixed_by_min67_highf_RUN3REC2_Buffer_20241011_1_11x11_pt1.nc'
    ds_unmixed = xr.load_dataset(unmixed_datapath)
    fit_results, spatial_dims, spatial_shape = fit_all_spectra(
        ds_unmixed, var='predicted_difference', n_peaks=len(peak_names), min_width=min_width, max_width=max_width)
    # Reshape to (spatial_dims..., peak, param)
    fit_results = fit_results.reshape(*spatial_shape, len(peak_names), 4)
    fit_da = xr.DataArray(
        fit_results,
        dims=spatial_dims + ['peak', 'param'],
        coords={**{dim: ds_unmixed.coords[dim] for dim in spatial_dims},
                'peak': peak_names,
                'param': param_names}
    )
    
    # Save to a new NetCDF file
    output_path = unmixed_datapath.replace('.nc', '_fitting_results.nc')
    fit_da.to_netcdf(output_path)
    print(f"Fitting results saved to {output_path}")