import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from spectraformer.input_pipeline import Batch


def predict(apply_fn, variables, batch: Batch, *apply_fn_args):
    pred = apply_fn(
        variables,
        batch["masked_spectra"],
        batch["wave_number"],
        *apply_fn_args,
        training=False,
    )
    res = {k: np.squeeze(v) for k, v in batch.items()}
    res["predicted_spectra"] = np.squeeze(pred)
    res["predicted_difference"] = res["spectra"] - res["predicted_spectra"]
    return res

def plot_results_train(predictions, step, epoch, current_model_tag):
    fig, ax = plt.subplots(figsize=(10, 6))
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
        ax.plot(wave_number, data, '-o', markersize=1.3, lw=1, label=label)
    if data.ndim > 1:
        print(
            f"Warning: found {data.shape[1]} predicted spectra in the provided dictionary. The plot might be crowded."
        )
    ax.legend(frameon=False)
    ax.margins(x=0)
    ax.grid(visible=True, which='both', axis='both')
    ax.set_title(f'{current_model_tag}\nStep {step} -- Epoch {epoch}', fontsize='x-large')
    
    ax.set_xlabel("Raman shift, cm$^{-1}$", fontsize='x-large')
    ax.set_ylabel("Intensity, a.u.", fontsize='x-large')
    ax.tick_params(axis='both', which='major', labelsize='x-large')
    ax.set_ylim(-0.3, 1.5)
    return fig, ax

def plot_loss(dummy_wave_number, loss, step, epoch, current_model_tag):
    fig, ax = plt.subplots(figsize=(10, 6))
    if np.max(np.abs(dummy_wave_number)) < 10:
        dummy_wave_number = dummy_wave_number * 800 + 2000
    
    ax.plot(dummy_wave_number, loss, label='Loss')
    
    loss_arithm_mean = np.mean(loss)
    ax.axhline(loss_arithm_mean, label="Arithmetic mean", color="r", alpha=1, linestyle=":")
    
    ax.set_xlabel("Raman shift, cm$^{-1}$", fontsize='x-large')
    ax.set_ylabel("Loss, a.u.", fontsize='x-large')
    ax.set_title(f'Loss for {current_model_tag}\nStep {step} -- Epoch {epoch}', fontsize='x-large')
    ax.legend()
    ax.grid(visible=True, which='both', axis='both')
    ax.tick_params(axis='both', which='major', labelsize='x-large')
    
    ax.set_yscale('log')
    ax.set_ylim(1e-14, 1e+1)
    return fig, ax
