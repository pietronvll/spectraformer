def median_baseline(params, spectra, wave_number):
    from jax.numpy import interp

    x = interp(wave_number, params["wave_number"], params["median_counts"])
    return x
