import gpustat
import jax
import jax.numpy as jnp
from flax.training.common_utils import stack_forest
from flax.training.train_state import TrainState

from spectraformer.input_pipeline import Batch, batch_sampler


def log_gpu_usage(gpustat_entry, step, writer):
    name = f"[{gpustat_entry['name']}/{gpustat_entry['index']}"

    writer.add_scalar(f"{name}/usage", gpustat_entry["utilization.gpu"], step)
    writer.add_scalar(
        f"{name}/memory",
        100 * gpustat_entry["memory.used"] / gpustat_entry["memory.total"],
        step,
    )


@jax.jit
def train_step(state: TrainState, batch: Batch, dropout_key):
    dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)

    def loss_fn(params):
        pred_spectra = state.apply_fn(
            {"params": params},
            batch["masked_spectra"],
            batch["wave_number"],
            batch["mask"],
            training=True,
            rngs={"dropout": dropout_train_key},
        )
        # Poisson Loss
        loss = (pred_spectra - batch["spectra"] * jnp.log(pred_spectra)).mean()
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    train_metrics = {"loss": loss}
    return state, train_metrics


def train_epoch(
    state, epoch: int, train_ds, configs, rng_streams, metric_writer, ckpt_manager
):
    mask_windows = list(
        zip(configs.masked_interval_starts, configs.masked_interval_ends)
    )
    data_loader = batch_sampler(
        train_ds,
        mask_windows,
        batch_size=configs.batch_size,
        rng_seed=epoch,
        shuffle=True,
    )
    metrics = []
    for batch in data_loader:
        state, batch_metrics = train_step(state, batch, rng_streams["dropout"])
        metrics.append(batch_metrics)

    metrics = stack_forest(metrics)
    avg_metrics = jax.tree_map(jnp.mean, metrics)  # Log the average error of the epoch

    print(f"Epoch {epoch + 1} -- Loss {avg_metrics['loss'].item():.3e}")
    if epoch % configs.log_every_epochs == 0:
        metric_writer.add_scalar("train/loss", avg_metrics["loss"].item(), state.step)
        for gpu_stats in gpustat.new_query():
            log_gpu_usage(gpu_stats.entry, state.step, metric_writer)
        ckpt_manager.save(state.step, state)
    return state, metrics
