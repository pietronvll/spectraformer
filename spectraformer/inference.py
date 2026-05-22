import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path
from spectraformer.input_pipeline import Batch


plt.rcParams['font.size'] = 24


def _restore_wave_number(wave_number):
    wave_number = np.asarray(wave_number)
    if np.max(np.abs(wave_number)) < 10:
        wave_number = wave_number * 800 + 2000
    return wave_number


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
    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    wave_number = _restore_wave_number(predictions["wave_number"])

    mask_boundaries = np.argwhere(
        np.diff(predictions["mask"], prepend=np.array([True]))
    )
    for bdr in mask_boundaries:
        ax.axvline(x=wave_number[bdr[0]], color="gray", alpha=0.5, linestyle=":")

    for data_str in ["spectra", "predicted_spectra", "predicted_difference"]:
        data = predictions[data_str]
        if data.ndim > 1:
            data = data.T
        label_map = {
            "spectra": "Data",
            "predicted_spectra": "Model output",
            "predicted_difference": "Spectral subtraction",
        }
        color = {
            "spectra": "C0",
            "predicted_spectra": "C1",
            "predicted_difference": "C2",
        }[data_str]
        ax.plot(wave_number, data, '-o', markersize=1.3, lw=1.2, label=label_map[data_str], color=color)
    if data.ndim > 1:
        print(
            f"Warning: found {data.shape[1]} predicted spectra in the provided dictionary. The plot might be crowded."
        )

    ax.legend(frameon=True, fontsize='small')
    ax.margins(x=0)
    ax.grid(visible=True, which='both', axis='both', alpha=0.25)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(300))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))
    ax.set_title(f'{current_model_tag}\nStep {step} -- Epoch {epoch}')
    
    ax.set_xlabel("Raman shift, cm$^{-1}$")
    ax.set_ylabel("Intensity, a.u.")
    ax.tick_params(axis='both', which='major')
    ax.set_ylim(-0.3, 1.5)
    return fig, ax

def plot_loss(dummy_wave_number, loss, step, epoch, current_model_tag, mask=None):
    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    dummy_wave_number = _restore_wave_number(dummy_wave_number)

    loss = np.asarray(loss)
    if mask is not None:
        visible_mask = np.asarray(mask).astype(bool)
        hidden_mask = ~visible_mask
        loss_to_plot = np.where(hidden_mask, loss, np.nan)
        arithmetic_mean = np.nanmean(loss_to_plot)
    else:
        loss_to_plot = loss
        arithmetic_mean = np.mean(loss_to_plot)
    
    ax.plot(dummy_wave_number, loss_to_plot, label='Masked-region loss', color='C2', lw=1.2)
    
    ax.axhline(arithmetic_mean, label="Arithmetic mean", color="r", alpha=1, linestyle=":")
    
    ax.set_xlabel("Raman shift, cm$^{-1}$")
    ax.set_ylabel("Loss, a.u.")
    ax.set_title(f'Loss for {current_model_tag}\nStep {step} -- Epoch {epoch}')
    ax.legend(frameon=True, fontsize='small')
    ax.grid(visible=True, which='both', axis='both', alpha=0.25)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(300))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))
    
    ax.set_yscale('log')
    ax.set_ylim(1e-14, 1e+1)
    return fig, ax
