import matplotlib.pyplot as plt
import numpy as np

from spectraformer.input_pipeline import Batch


def predict(apply_fn, variables, batch: Batch, *apply_fn_args):
    pred = apply_fn(
        variables, batch["masked_spectra"], batch["wave_number"], *apply_fn_args
    )
    res = {k: np.squeeze(v) for k, v in batch.items()}
    res["predicted_spectra"] = np.squeeze(pred)
    res["predicted_difference"] = res["spectra"] - res["predicted_spectra"]
    return res


def plot_results(predictions):
    fig, ax = plt.subplots()
    wave_number = predictions["wave_number"]
    # Check if wave_number has been normalized
    if np.max(np.abs(wave_number)) < 10:
        wave_number = wave_number * 800 + 2000

    mask_boundaries = np.argwhere(
        np.diff(predictions["mask"], prepend=np.array([True]))
    )
    for bdr in mask_boundaries:
        ax.axvline(x=wave_number[bdr[0]], color="k", alpha=0.5, linestyle=":")
    for data_str in ["spectra", "predicted_spectra", "predicted_difference"]:
        data = predictions[data_str]
        if data.ndim > 1:
            data = data.T
        label = data_str.replace("_", " ").capitalize()
        ax.plot(wave_number, data, label=label)
    if data.ndim > 1:
        print(
            f"Warning: found {data.shape[1]} predicted spectra in the provided dictionary. The plot might be crowded."
        )
    ax.legend(frameon=False)
    return fig, ax
