import jax.numpy as jnp


def median(variables, spectra, wave_number):
    params = variables["params"]  # Conform to Flax
    batch_size = spectra.shape[0]
    # Check if wave_number has been normalized
    if jnp.max(jnp.abs(wave_number)) < 10:
        wave_number = wave_number * 800 + 2000
    x = jnp.interp(wave_number, params["wave_number"], params["median_counts"])
    return jnp.stack([x] * batch_size)
