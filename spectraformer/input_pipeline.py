from typing import Iterator, Optional, TypedDict

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr


def preprocess_dataset(
    dataset: xr.DataArray,
    bg_removal_window: tuple = (2200, 2500),
    sup_norm_threshold: float = 0.15,
    verbose: bool = False,
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
    bg_removal_window = dataset.sel(wave_number=slice(*bg_removal_window))
    bg_value = bg_removal_window.median()
    dataset = dataset - bg_value
    # Normalization to the max
    dataset = dataset / dataset.max(dim="wave_number")

    # Outlier removal
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
    mask_windows: list = [(1525, 1650), (2500, 2900)],
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
