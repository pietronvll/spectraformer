import gpustat
import jax
import jax.numpy as jnp
from jax import lax
from flax.training.common_utils import stack_forest
from flax.training.train_state import TrainState
import optax

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
        
        # NaN or Inf check using lax
        nan_check_pred_spectra = jnp.any(jnp.isnan(pred_spectra))
        inf_check_pred_spectra = jnp.any(jnp.isinf(pred_spectra))

        # Use lax.cond to act on the condition
        lax.cond(nan_check_pred_spectra, lambda _: print("NaN detected in pred_spectra for training step"), lambda _: None, operand=None)
        lax.cond(inf_check_pred_spectra, lambda _: print("Inf detected in pred_spectra for training step"), lambda _: None, operand=None)
        
        
        # pred_spectra = jnp.clip(pred_spectra, min=1e-8) # To be sure that no neg value will be fed into log function. NaN value formation prevention

        # Poisson Loss
        loss = (pred_spectra - batch["spectra"] * jnp.log(pred_spectra)).mean()
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    
    
    
    

        

    
    # Flatten the PyTree of gradients
    flat_grads, _ = jax.tree_util.tree_flatten(grads)
    # Concatenate all gradients into a single array for statistics
    all_grads = jnp.concatenate([jnp.ravel(g) for g in flat_grads])
    
    # NaN or Inf check using lax
    nan_check_grads = jnp.any(jnp.isnan(all_grads))
    inf_check_grads = jnp.any(jnp.isinf(all_grads))

    # Use lax.cond to act on the condition
    lax.cond(nan_check_grads, lambda _: print("NaN detected in grads"), lambda _: None, operand=None)
    lax.cond(inf_check_grads, lambda _: print("Inf detected in grads"), lambda _: None, operand=None)
    
    # Compute statistics
    grad_min = jnp.min(all_grads)
    grad_mean = jnp.mean(all_grads)
    grad_median = jnp.median(all_grads)
    grad_max = jnp.max(all_grads)
    
    state = state.apply_gradients(grads=grads)
    train_metrics = {
        "train_loss": loss,
        "grad_min": grad_min,
        "grad_mean": grad_mean,
        "grad_median": grad_median,
        "grad_max": grad_max
        }
    return state, train_metrics

@jax.jit
def validation_step(state: TrainState, batch: Batch, dropout_key):
    dropout_val_key = jax.random.fold_in(key=dropout_key, data=state.step)
    
    pred_spectra = state.apply_fn(
                {"params": state.params},
                batch["masked_spectra"],
                batch["wave_number"],
                batch["mask"],
                training=False,
                rngs={"dropout": dropout_val_key},
            )
    # NaN or Inf check using lax
    nan_check_pred_spectra = jnp.any(jnp.isnan(pred_spectra))
    inf_check_pred_spectra = jnp.any(jnp.isinf(pred_spectra))

    # Use lax.cond to act on the condition
    lax.cond(nan_check_pred_spectra, lambda _: print("NaN detected in pred_spectra for validation step"), lambda _: None, operand=None)
    lax.cond(inf_check_pred_spectra, lambda _: print("Inf detected in pred_spectra for validation step"), lambda _: None, operand=None)

    # pred_spectra = jnp.clip(pred_spectra, min=1e-8) # To be sure that no neg value will be fed into log function. NaN value formation prevention

    
    def loss_fn(params):
        # Poisson Loss
        loss = (pred_spectra - batch["spectra"] * jnp.log(pred_spectra)).mean()
        return loss

    loss = loss_fn(state.params)                                                        # Poisson loss - suitable for our particular task
    cos_sim = optax.losses.cosine_similarity(pred_spectra, batch["spectra"]).mean()     # Cosine similarity - measure of how close vectors are in terms of a direction (1 - same direction, 0 - orthogonal, -1 - opposite)
    mse = optax.losses.squared_error(pred_spectra, batch["spectra"]).mean()             # Mean square error - normalized L2 loss - scalar value that evaluates the overall prediction accuracy of a model across the dataset
    
    val_metrics = {
        "val_loss": loss, 
        "cos_sim": cos_sim, 
        "MSE": mse
        }
    return state, val_metrics

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

    print(f"Epoch {epoch + 1} -- Loss {avg_metrics['train_loss'].item():.3e}")
    if epoch % configs.log_every_epochs == 0:
        metric_writer.add_scalar("train/loss", avg_metrics["train_loss"].item(), state.step)
        metric_writer.add_scalar("train/grad_min", avg_metrics["grad_min"].item(), state.step)
        metric_writer.add_scalar("train/grad_mean", avg_metrics["grad_mean"].item(), state.step)
        metric_writer.add_scalar("train/grad_median", avg_metrics["grad_median"].item(), state.step)
        metric_writer.add_scalar("train/grad_max", avg_metrics["grad_max"].item(), state.step)
        for gpu_stats in gpustat.new_query():
            log_gpu_usage(gpu_stats.entry, state.step, metric_writer)
        ckpt_manager.save(state.step, state)
    return state, metrics

def validation_epoch(
    state, epoch: int, val_ds, configs, rng_streams, metric_writer, ckpt_manager
):
    mask_windows = list(
        zip(configs.masked_interval_starts, configs.masked_interval_ends)
    )
    data_loader = batch_sampler(
        val_ds,
        mask_windows,
        batch_size=configs.batch_size,
        rng_seed=epoch,
        shuffle=True,
    )
    metrics = []
    for batch in data_loader:
        state, batch_metrics = validation_step(state, batch, rng_streams["dropout"])
        metrics.append(batch_metrics)

    metrics = stack_forest(metrics)
    avg_metrics = jax.tree_map(jnp.mean, metrics)  # Log the average error of the epoch

    print(f"Validation -- Epoch {epoch + 1} -- Loss {avg_metrics['val_loss'].item():.3e} -- Cos_sim {avg_metrics['cos_sim'].item():.3e} -- MSE {avg_metrics['MSE'].item():.3e}")
    if epoch % configs.log_every_epochs == 0:
        metric_writer.add_scalar("val/loss", avg_metrics["val_loss"].item(), state.step)
        metric_writer.add_scalar("val/cos_sim", avg_metrics["cos_sim"].item(), state.step)
        metric_writer.add_scalar("val/MSE", avg_metrics["MSE"].item(), state.step)
        for gpu_stats in gpustat.new_query():
            log_gpu_usage(gpu_stats.entry, state.step, metric_writer)
        ckpt_manager.save(state.step, state)
    return state, metrics