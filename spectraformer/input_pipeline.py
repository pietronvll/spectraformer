from typing import Iterator, Optional, TypedDict

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
import pybaselines
import copy

from scipy.signal import savgol_filter

def preprocess_dataset(
    dataset: xr.DataArray,
    bg_removal_window: tuple = (2200, 2500),
    sup_norm_threshold: float = 0.15,
    verbose: bool = True,
    is_filter: bool = False,
    option: str = 'proper_bg_proper_norm'
) -> xr.DataArray:
    """Preprocess xarray datasets by subtracting the background, normalizing to the max and removing outliers, i.e. spectra with cosmic rays or other artifacts.

    Args:
        dataset (xr.DataArray): xarray dataset with the spectra, as created by the preprocessing pipeline.
        bg_removal_window (tuple, optional): Wavelength window to use as reference to remove the background. Defaults to (2200, 2500) cm^-1.
        sup_norm_threshold (float, optional): Threshold to discard outliers; a spectra is considered an outlier whence the sup-norm distance with respect to the median is greater than the threshold. Defaults to 0.15.
        verbose (bool, optional): Defaults to False.
        option (str, optional): How to preprocess dataset. ['bg_maxnorm', 'maxnorm', 'bg', 'proper_bg_maxnorm']

    Returns:
        xr.DataArray: Processed dataset
    """
    # Background removal
    def bg_removal_fn(
        dataset, 
        bg_removal_window: tuple = (2200, 2500)
        ):
        bg_removal_window = dataset.sel(wave_number=slice(*bg_removal_window))
        bg_value = bg_removal_window.median()
        dataset_bg = dataset - bg_value
        return dataset_bg
    
    def proper_bg_removal_fn(
        dataset, 
        bg_removal_window: tuple = (2200, 2500)
        ):
        bg_removal_window = dataset.sel(wave_number=slice(*bg_removal_window))
        bg_values = bg_removal_window.median(dim="wave_number")
        full_ds_bg_removed_per_spectra = dataset - bg_values
        return full_ds_bg_removed_per_spectra
    
    # Negative value removal - by shifting everything towards 0
    def neg_val_removal_fn(dataset):
        dataset_positive = dataset - dataset.min(dim="wave_number")
        return dataset_positive
    
    # Normalization to the max
    def maxnorm_fn(dataset):
        dataset_norm = dataset / dataset.max(dim="wave_number")
        return dataset_norm
    
    # Shifting from zero by an arbitrary number
    def shifting_fn(
        dataset, 
        shift: float = 0
        ):
        return dataset + shift
    
    # Normalizing dataset into [0,1] range
    def proper_norm_fn(dataset):
        dataset_norm = maxnorm_fn( neg_val_removal_fn(dataset) )
        return dataset_norm

    # Spatial dimensions stacking
    def stack_spatial_dims(dataset):
        spatial_dims = dataset.dims[:-1]
        return dataset.stack(spectra=spatial_dims).dropna(dim="spectra")
    
    # Outlier removal
    def outlier_removal_fn(
        dataset,
        sup_norm_threshold: float = 0.15
        ):
        spatial_dims = dataset.dims[:-1]
        num_spectra = np.prod([len(dataset[dim]) for dim in spatial_dims]).item()
        median_counts = dataset.median(dim=spatial_dims)
        sup_norm_deviations = (abs(dataset - median_counts)).max(dim="wave_number")
        filtered_dataset = dataset.where(
            sup_norm_deviations < sup_norm_threshold, drop=True
        )
        filtered_dataset = filtered_dataset.stack(spectra=spatial_dims).dropna(
            dim="spectra"
        )
        if verbose:
            print(f"Dropped {num_spectra - len(filtered_dataset.spectra)} spectra")
        return filtered_dataset
    
    def mimic_outlier_removal_fn(
        dataset,
        sup_norm_threshold: float = 0.15
        ):
        num_spectra = len(dataset['spectra'])
        median_counts = dataset.median(dim='spectra')
        sup_norm_deviations = (abs(dataset - median_counts)).max(dim="wave_number")
        filtered_dataset = dataset.where(
            sup_norm_deviations < sup_norm_threshold, drop=True
        )
        if verbose:
            print("##### MIMIC OUTLIER REMOVAL #####")
            print(f"Dataset contain In total {num_spectra} spectra, in which Dropped {num_spectra - len(filtered_dataset.spectra)} spectra")
            print("##### MIMIC OUTLIER REMOVAL #####")
        return filtered_dataset
    
    def subtract_whittaker_background(da, lam=1e7):
        """Subtract Whittaker baseline from xarray DataArray using pybaselines."""
        
        # Process each spectrum
        baselines = []
        for i in range(len(da.spectra)):
            spectrum = da.isel(spectra=i).values
            baseline, _ = pybaselines.whittaker.aspls(spectrum, lam=lam)
            baselines.append(baseline)
        
        # Create baseline DataArray
        baseline_da = xr.DataArray(
            np.array(baselines).T,
            dims=da.dims,
            coords=da.coords
        )
        
        # Subtract baseline
        return da - baseline_da #, baseline_da
    
    def whitaker_hayes(intensity_data, kernel_size: int = 3, threshold: int = 8):
        return xr.DataArray(
            np.apply_along_axis(whitaker_hayes_spectrum, axis=-1, arr=intensity_data, kernel_size=kernel_size, threshold=threshold),
            dims=intensity_data.dims,
            coords=intensity_data.coords
            )


    def whitaker_hayes_spectrum(intensity_values_array, kernel_size, threshold):
        spectrum_array = copy.deepcopy(intensity_values_array)

        spikes = whitaker_hayes_modified_z_score(spectrum_array) > threshold

        while any(spike for spike in spikes if spike):
            changes = False

            for i in range(len(spikes)):
                if spikes[i]:
                    neighbours = np.arange(max(0, i - kernel_size),
                                        min(len(spectrum_array) - 1, i + 1 + kernel_size))
                    fixed_value = np.median(spectrum_array[neighbours[spikes[neighbours] == 0]]) # Median or mean?

                    if np.isnan(fixed_value):
                        continue

                    spectrum_array[i] = fixed_value
                    spikes[i] = 0
                    changes = True

            if not changes:
                break

        return spectrum_array


    def modified_z_score(spectrum):
        """Calculates the modified z-scores of a given spectrum."""
        mad_term = np.median([np.abs(spectrum - np.median(spectrum))])
        modified_z_scores = np.array(0.6745 * (spectrum - np.median(spectrum)) / mad_term)

        return modified_z_scores


    def whitaker_hayes_modified_z_score(spectrum):
        """Calculates the Whitaker-Hayes modified z-scores of a given spectrum."""
        return np.abs(modified_z_score(np.diff(spectrum)))
        # return np.abs(modified_z_score(spectrum))
    
    if is_filter:
        # Apply Savitzky-Golay filter to smooth the dataset
        dataset = xr.apply_ufunc(
            lambda x: savgol_filter(x, window_length=7, polyorder=2, axis=-1),
            dataset,
            input_core_dims=[["wave_number"]],
            output_core_dims=[["wave_number"]],
            output_dtypes=[dataset.dtype],
            keep_attrs=True,
        )
    
    match option:
        case 'bg_maxnorm':
            preprocessed_dataset = maxnorm_fn(
                bg_removal_fn(
                    dataset
                    ) 
                )
        
        case 'maxnorm':
            preprocessed_dataset = maxnorm_fn(
                dataset 
                )
        
        case 'bg':
            preprocessed_dataset = bg_removal_fn(
                dataset
                )

        case 'proper_bg_maxnorm':
            preprocessed_dataset = maxnorm_fn(
                proper_bg_removal_fn(
                    dataset
                    ) 
                )
        case 'proper_bg_proper_norm':
            # (data-data.min)/data.max
            preprocessed_dataset = outlier_removal_fn(
                shifting_fn(
                    proper_norm_fn(
                        dataset
                    ), shift=0.1
                )
            )
        case 'proper_bg_proper_norm_with_outliers':
            # (data-data.min)/data.max
            preprocessed_dataset = shifting_fn(
                proper_norm_fn(
                    dataset
                    ), shift=0.1
            )
        case 'whitaker_hayes':
            # 0. Spatial dimensions stacking
            preprocessed_dataset = stack_spatial_dims(dataset)
            # 1. Background removal
            preprocessed_dataset = subtract_whittaker_background(preprocessed_dataset)
            # 2. Whitaker-Hayes Outlier removal
            preprocessed_dataset = whitaker_hayes(
                preprocessed_dataset
            )
            # 3. Normalization to the max
            preprocessed_dataset = proper_norm_fn(
                preprocessed_dataset
            )
            # 4. Shifting - to avoid large negative log values in loss calculation
            preprocessed_dataset = shifting_fn(
                preprocessed_dataset, shift=0.1
            )
            # 5. mimic
            preprocessed_dataset = mimic_outlier_removal_fn(
                preprocessed_dataset
            )
    return preprocessed_dataset

def preprocess_dataset_raw(
    dataset: xr.DataArray,
    bg_removal_window: tuple = (2200, 2500),
    sup_norm_threshold: float = 0.15,
    verbose: bool = False,
    raw: bool = False,
) -> xr.DataArray:
    """Preprocess xarray datasets by subtracting the background, normalizing to the max and removing outliers, i.e. spectra with cosmic rays or other artifacts.

    Args:
        dataset (xr.DataArray): xarray dataset with the spectra, as created by the preprocessing pipeline.
        bg_removal_window (tuple, optional): Wavelength window to use as reference to remove the background. Defaults to (2200, 2500) cm^-1.
        sup_norm_threshold (float, optional): Threshold to discard outliers; a spectra is considered an outlier whence the sup-norm distance with respect to the median is greater than the threshold. Defaults to 0.15.
        verbose (bool, optional): Defaults to False.

    Returns:
        xr.DataArray: Processed dataset
    """
    # Background removal
    if not raw:
        bg_removal_window = dataset.sel(wave_number=slice(*bg_removal_window))
        bg_value = bg_removal_window.median()
        # dataset = dataset - bg_value
        dataset = dataset - dataset.min(dim='wave_number')
    
    # Normalization to the max
    if not raw:
        dataset = dataset / dataset.max(dim="wave_number")
        
        
        # Outlier removal
        spatial_dims = dataset.dims[:-1]
        num_spectra = np.prod([len(dataset[dim]) for dim in spatial_dims]).item()
        median_counts = dataset.median(dim=spatial_dims)
        sup_norm_deviations = (abs(dataset - median_counts)).max(dim="wave_number")
        dataset = dataset.where(sup_norm_deviations < sup_norm_threshold, drop=True)

        if verbose:
            print(f"Dropped {num_spectra - dataset.count().item()} spectra")

    # Always stack spatial dimensions into a single "spectra" dimension
    spatial_dims = dataset.dims[:-1]  # Everything except wave_number
    dataset = dataset.stack(spectra=spatial_dims)
    
    return dataset


def mask_dataset(
    dataset: xr.DataArray,
    mask_windows: list,
    default_value=-1,
) -> xr.DataArray:
    """Mask a dataset by setting the values of a given window to -1."""
    masks = []
    for mask_window in mask_windows:
        masks.append(
            (dataset.wave_number > mask_window[0])
            & (dataset.wave_number < mask_window[1])  # True INSIDE the window
        )
    if len(masks) == 0:
        return dataset, np.ones_like(dataset.wave_number.values)
    elif len(masks) == 1:
        mask = ~(masks[0])
        return dataset.where(mask, default_value), mask.values
    else:
        mask = masks[0]
        for i in range(len(masks) - 1):
            mask = mask | masks[i + 1]
        mask = ~(mask)
        return dataset.where(mask, default_value), mask.values


# Batch object implemented as a TypedDict
class Batch(TypedDict):
    spectra: jnp.ndarray
    masked_spectra: jnp.ndarray
    wave_number: jnp.ndarray


def batch_sampler(
    filtered_dataset: xr.DataArray,
    mask_windows: list,
    batch_size: Optional[int] = None,
    shuffle: bool = True,
    norm_wv: bool = True,
    rng_seed=0,
    drop_last=True,
    default_mask_value=-1,
) -> Iterator[Batch]:
    """Batch sampler for the dataset

    Args:
        filtered_dataset (xr.DataArray): Filtered dataset
        masked_dataset (xr.DataArray): Masked dataset
        batch_size (Optional[int], optional): Batch size. Defaults to None.
        shuffle (bool, optional): Shuffle the dataset. Defaults to True.
        rng_seed ([type], optional): Random seed. Defaults to None.

    Yields:
        Iterator[Batch]: Iterator over the dataset
    """

    # Reorder dimensions of datasets
    masked_dataset, mask = mask_dataset(
        filtered_dataset, mask_windows=mask_windows, default_value=default_mask_value
    )
    filtered_dataset = filtered_dataset.transpose("spectra", "wave_number")
    masked_dataset = masked_dataset.transpose("spectra", "wave_number")

    # Get the number of samples
    n_samples = len(filtered_dataset.spectra)

    if batch_size is None:
        batch_size = n_samples
    if shuffle:
        # Create a random number generator
        rng = jax.random.PRNGKey(rng_seed)
        # Create a permutation of the indices
        perm = jax.random.permutation(rng, n_samples)
    else:
        perm = jnp.arange(n_samples)

    # Drop last batch if it is not full
    if drop_last:
        n_samples = n_samples - (n_samples % batch_size)
        perm = perm[:n_samples]

    full_spectra = jnp.expand_dims(jnp.asarray(filtered_dataset.values), axis=-1)
    full_masked_spectra = jnp.expand_dims(jnp.asarray(masked_dataset.values), axis=-1)
    wave_number = jnp.expand_dims(
        jnp.asarray(filtered_dataset.wave_number.values), axis=-1
    )
    mask = jnp.asarray(mask)
    if norm_wv:
        wave_number = (wave_number - 2000) / 800  # Pretty arbitrary, but works.

    # Iterate over the dataset
    for i in range(0, n_samples, batch_size):
        # Get the indices for the batch
        indices = perm[i : i + batch_size]
        # Get the spectra
        spectra = full_spectra[indices]
        # Get the masked spectra
        masked_spectra = full_masked_spectra[indices]
        # Yield the batch
        yield Batch(
            spectra=spectra,
            masked_spectra=masked_spectra,
            wave_number=wave_number,
            mask=mask,
        )

def dataset_loader(
    datadir,
    file_location_with_name: str,
    shuffle_rng_seed,
    split_fraction: float = 0.8,
    is_filter: bool = False,
    option: str = 'proper_bg_proper_norm'  # Option for the preprocess_dataset function
):
    """Load the dataset and return the train and validation datasets.

    Args:
        datadir: Usually maindir / "data"
        file_location_with_name (str): everything after folder "data" including the file name file.nc
        shuffle_rng_seed: pass here configs.root_rng_seed
        split_fraction (float, optional): Split fraction. Defaults to 0.8.

    Returns:
        train_ds, val_ds
    """
    ####################################################################################################
    # Dataset loading and separation into train/val section
    #################################################################################################### 
    # Load the full dataset
    print(f"\n----- Loading dataset {file_location_with_name}. -----\n")
    full_ds = preprocess_dataset(
        xr.load_dataarray(datadir / file_location_with_name), is_filter=is_filter, option=option
    )
    print("Original dataset dimensions:", full_ds.dims)  # Should show (wave_number, spectra)
    # Get number of spectra samples
    n_spectra = full_ds.sizes['spectra']

    # Shuffle spectra indices
    np.random.seed(shuffle_rng_seed)
    spectra_indices = np.arange(n_spectra)
    np.random.shuffle(spectra_indices)

    # Split indices
    split_index = int(n_spectra * split_fraction)
    train_spectra_indices = spectra_indices[:split_index]
    val_spectra_indices = spectra_indices[split_index:]

    # Split dataset along spectra dimension
    train_ds = full_ds.isel(spectra=train_spectra_indices)
    val_ds = full_ds.isel(spectra=val_spectra_indices)
    print("\nSplit verification:")
    print(f"Training spectra samples: {train_ds.sizes['spectra']}")
    print(f"Validation spectra samples: {val_ds.sizes['spectra']}")
    print(f"Total spectra: {n_spectra} = {train_ds.sizes['spectra'] + val_ds.sizes['spectra']}")
    print("\nShape verification (wave_number should match):")
    print(f"Original wave_number count: {full_ds.sizes['wave_number']}")
    print(f"Train dataset shape: {train_ds.shape}")
    print(f"Val dataset shape: {val_ds.shape}")
    
    print(f"\n----- Dataset {file_location_with_name} is loaded. -----\n")
    return train_ds, val_ds
