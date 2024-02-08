from typing import Iterator, Optional, TypedDict

import jax
import jax.numpy as jnp
import xarray as xr


def preprocess_dataset(
    dataset: xr.DataArray,
    bg_removal_window: tuple = (2200, 2500),
    sup_norm_threshold: float = 0.15,
    verbose: bool = False,
) -> xr.DataArray:
    # Background removal
    bg_removal_window = dataset.sel(wave_number=slice(*bg_removal_window))
    bg_value = bg_removal_window.median()
    dataset = dataset - bg_value
    # Normalization to the max
    dataset = dataset / dataset.max(dim="wave_number")

    # Outlier removal
    median_counts = dataset.median(dim=["X", "Y"])
    sup_norm_deviations = (abs(dataset - median_counts)).max(dim="wave_number")
    filtered_dataset = dataset.where(
        sup_norm_deviations < sup_norm_threshold, drop=True
    )
    filtered_dataset = filtered_dataset.stack(spectra=("X", "Y")).dropna(dim="spectra")
    if verbose:
        print(
            f"Dropped {len(dataset.X)*len(dataset.Y) - len(filtered_dataset.spectra)} spectra"
        )
    return filtered_dataset


def get_masked_dataset(
    dataset: xr.DataArray, mask_window: tuple = (1525, 1650)
) -> xr.DataArray:
    mask = ~(
        (dataset.wave_number < mask_window[1]) & (dataset.wave_number > mask_window[0])
    )
    return dataset.where(mask, -1)


# Batch object implemented as a TypedDict
class Batch(TypedDict):
    spectra: jnp.ndarray
    masked_spectra: jnp.ndarray
    wave_number: jnp.ndarray


def batch_sampler(
    filtered_dataset: xr.DataArray,
    masked_dataset: xr.DataArray,
    batch_size: Optional[int] = None,
    shuffle: bool = True,
    rng_seed=0,
    drop_last=True,
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
            spectra=spectra, masked_spectra=masked_spectra, wave_number=wave_number
        )
