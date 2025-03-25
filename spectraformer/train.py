import gpustat
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from flax.training.common_utils import stack_forest
from flax.training.train_state import TrainState
import optax
from functools import partial

from spectraformer.input_pipeline import Batch, batch_sampler


def log_gpu_usage(gpustat_entry, step, writer):
    name = f"[{gpustat_entry['name']}/{gpustat_entry['index']}"

    writer.add_scalar(f"{name}/usage", gpustat_entry["utilization.gpu"], step)
    writer.add_scalar(
        f"{name}/memory",
        100 * gpustat_entry["memory.used"] / gpustat_entry["memory.total"],
        step,
    )

def shard_batch(batch, num_devices):
    total_size = batch["masked_spectra"].shape[0]
    per_device = total_size // num_devices  # Ensure divisibility
    truncated_size = per_device * num_devices  # Compute new valid size
    
    batch_sharded = {}
    for k, v in batch.items():
        v = v[:truncated_size]  # Truncate excess samples
        new_shape = (num_devices, per_device) + v.shape[1:]
        batch_sharded[k] = v.reshape(new_shape)
    
    return batch_sharded


@partial(jax.jit, static_argnames=("configs_mean",))
def train_step(
    state: TrainState, 
    batch: Batch, 
    dropout_key,
    configs_mean
):
    dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)

    def poisson_loss_fn(params):
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
        lax.cond(nan_check_pred_spectra, lambda _: jax.debug.print(f"{jnp.isnan(pred_spectra).sum()} NaN detected in pred_spectra for training step"), lambda _: None, operand=None)
        lax.cond(inf_check_pred_spectra, lambda _: jax.debug.print(f"{jnp.isinf(pred_spectra).sum()} Inf detected in pred_spectra for training step"), lambda _: None, operand=None)
        
        loss = (pred_spectra - batch["spectra"] * jnp.log(pred_spectra)).mean()
        return loss

    def gamma_loss_fn(params):
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
        lax.cond(nan_check_pred_spectra, lambda _: jax.debug.print(f"{jnp.isnan(pred_spectra).sum()} NaN detected in pred_spectra for training step"), lambda _: None, operand=None)
        lax.cond(inf_check_pred_spectra, lambda _: jax.debug.print(f"{jnp.isinf(pred_spectra).sum()} Inf detected in pred_spectra for training step"), lambda _: None, operand=None)
        
        loss = (jnp.log(pred_spectra) + batch["spectra"] / pred_spectra).mean()
        return loss

    def mse_loss_fn(params):
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
        lax.cond(nan_check_pred_spectra, lambda _: jax.debug.print(f"{jnp.isnan(pred_spectra).sum()} NaN detected in pred_spectra for training step"), lambda _: None, operand=None)
        lax.cond(inf_check_pred_spectra, lambda _: jax.debug.print(f"{jnp.isinf(pred_spectra).sum()} Inf detected in pred_spectra for training step"), lambda _: None, operand=None)
        
        loss = optax.losses.squared_error(pred_spectra, batch["spectra"]).mean()
        return loss
    
    def corrected_poisson_loss_fn(params):
        pred_spectra = state.apply_fn(
            {"params": params},
            batch["masked_spectra"],
            batch["wave_number"],
            batch["mask"],
            training=True,
            rngs={"dropout": dropout_train_key},
        )
        # jax.debug.print('Went inside loss fn')
        
        # print(f'Shape of predicted spectra: {jnp.shape(pred_spectra)}')
        
        # NaN or Inf check using lax
        nan_check_pred_spectra = jnp.any(jnp.isnan(pred_spectra))
        inf_check_pred_spectra = jnp.any(jnp.isinf(pred_spectra))
        # Use lax.cond to act on the condition
        lax.cond(nan_check_pred_spectra, lambda _: jax.debug.print(f"NaN detected in pred_spectra for training step"), lambda _: None, operand=None)
        lax.cond(inf_check_pred_spectra, lambda _: jax.debug.print(f"Inf detected in pred_spectra for training step"), lambda _: None, operand=None)
        # jax.debug.print('AFTER CONDITION')
        # if nan_check_pred_spectra or inf_check_pred_spectra:
        #     print(f'Training: Replacing NaN -> 1e-2, posinf -> 1, neginf -> 1e-2')
        #     pred_spectra = jnp.nan_to_num(pred_spectra, nan=1e-2, posinf=1, neginf=1e-2)
        
        # print(jnp.any(jnp.isneginf(jnp.log(batch["spectra"] / pred_spectra))))
        
        loss = ((pred_spectra - batch["spectra"]) + batch["spectra"] * jnp.log(batch["spectra"] / pred_spectra)).mean()
        # jax.debug.print('About to exit loss fn')
        return loss
    
    def corrected_gamma_loss_fn(params):
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
        lax.cond(nan_check_pred_spectra, lambda _: jax.debug.print(f"NaN detected in pred_spectra for training step"), lambda _: None, operand=None)
        lax.cond(inf_check_pred_spectra, lambda _: jax.debug.print(f"Inf detected in pred_spectra for training step"), lambda _: None, operand=None)
        
        
        loss = (( batch["spectra"]/pred_spectra - 1) - jnp.log( batch["spectra"]/pred_spectra ))
        
        def my_geometric_mean(loss, eps=1e-8):
            """
            Geometric mean calculating using a formula:
            GM(x)=exp( 1/N * sum( log(x_i) ) )
            """
            
            # Making sure to have no negative values in the loss with all information keeping
            non_negative = abs(loss)
            # Making sure having strictly positive values
            clipped = jnp.clip(non_negative, eps)
            # Calculating the log
            log_values = jnp.log(clipped)
            # Log averaging
            mean_log = jnp.mean(log_values)
            # Going back from log to normal value by exponentiation
            return jnp.exp(mean_log)
        
        match configs_mean:
            case 'Arithmetic':
                loss = loss.mean()
            case 'Geometric':
                loss = my_geometric_mean(loss)
            case _:
                raise Exception(f"You didn't specify a mean to be used!")
        
        return loss
    
    grad_fn = jax.value_and_grad(corrected_gamma_loss_fn)
    loss, grads = grad_fn(state.params)
    
    # Flatten the PyTree of gradients
    flat_grads, _ = jax.tree_util.tree_flatten(grads)
    # Concatenate all gradients into a single array for statistics
    all_grads = jnp.concatenate([jnp.ravel(g) for g in flat_grads])
    
    # NaN or Inf check using lax
    nan_check_grads = jnp.any(jnp.isnan(all_grads))
    inf_check_grads = jnp.any(jnp.isinf(all_grads))

    # Use lax.cond to act on the condition
    lax.cond(nan_check_grads, lambda _: jax.debug.print("NaN detected in grads"), lambda _: None, operand=None)
    lax.cond(inf_check_grads, lambda _: jax.debug.print("Inf detected in grads"), lambda _: None, operand=None)
    
    # Compute gradient parameters for logging
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

@partial(jax.jit, static_argnames=("configs_mean",))
def validation_step(
    state: TrainState, 
    batch: Batch, 
    dropout_key,
    configs_mean
):
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
    lax.cond(nan_check_pred_spectra, lambda _: jax.debug.print(f"{jnp.isnan(pred_spectra).sum()} NaN detected in pred_spectra for validation step"), lambda _: None, operand=None)
    lax.cond(inf_check_pred_spectra, lambda _: jax.debug.print(f"{jnp.isinf(pred_spectra).sum()} Inf detected in pred_spectra for validation step"), lambda _: None, operand=None)
    
    def val_poisson_loss_fn(params):
        # Poisson Loss
        loss = (pred_spectra - batch["spectra"] * jnp.log(pred_spectra)).mean()
        return loss
    
    def val_gamma_loss_fn(params):
        # Gamma loss
        loss = (jnp.log(pred_spectra) + batch["spectra"] / pred_spectra).mean()
        return loss

    def val_mse_loss_fn(params):
        # MSE Loss
        loss = (pred_spectra - batch["spectra"] * jnp.log(pred_spectra)).mean()
        return loss
    
    def val_corrected_poisson_loss_fn(params):
        # (pred-true)+true*log(true/pred)
        # if nan_check_pred_spectra or inf_check_pred_spectra:
        #     print(f'Validation: Replacing NaN -> 1e-2, posinf -> 1, neginf -> 1e-2')
        #     pred_spectra = jnp.nan_to_num(pred_spectra, nan=1e-2, posinf=1, neginf=1e-2)
        loss = ((pred_spectra - batch["spectra"]) + batch["spectra"] * jnp.log(batch["spectra"] / pred_spectra)).mean()
        return loss
    
    def val_corrected_gamma_fn(params):
        loss = (( batch["spectra"]/pred_spectra - 1) - jnp.log( batch["spectra"]/pred_spectra ))
        
        def my_geometric_mean(loss, eps=1e-8):
            """
            Geometric mean calculating using a formula:
            GM(x)=exp( 1/N * sum( log(x_i) ) )
            """
            
            # Making sure to have no negative values in the loss with all information keeping
            non_negative = abs(loss)
            # Making sure having strictly positive values
            clipped = jnp.clip(non_negative, eps)
            # Calculating the log
            log_values = jnp.log(clipped)
            # Log averaging
            mean_log = jnp.mean(log_values)
            # Going back from log to normal value by exponentiation
            return jnp.exp(mean_log)
        
        match configs_mean:
            case 'Arithmetic':
                loss = loss.mean()
            case 'Geometric':
                loss = my_geometric_mean(loss)
            case _:
                raise Exception(f"You didn't specify a mean to be used!")
        
        
        return loss

    corrected_gamma_loss = val_corrected_gamma_fn(state.params)
    # corrected_poisson_loss = val_corrected_poisson_loss_fn(state.params)
    # poisson_loss = val_poisson_loss_fn(state.params)
    # gamma_loss = val_gamma_loss_fn(state.params)                                              # Gamma loss
    # cos_sim = optax.losses.cosine_similarity(pred_spectra, batch["spectra"]).mean()     # Cosine similarity - measure of how close vectors are in terms of a direction (1 - same direction, 0 - orthogonal, -1 - opposite)
    mse = optax.losses.squared_error(pred_spectra, batch["spectra"]).mean()             # Mean square error - normalized L2 loss - scalar value that evaluates the overall prediction accuracy of a model across the dataset
    
    val_metrics = {
        "val_corrected_gamma_loss": corrected_gamma_loss,
        # "val_corrected_poisson_loss": corrected_poisson_loss,
        # "val_poisson_loss": poisson_loss,
        # "val_gamma_loss": gamma_loss, 
        # "cos_sim": cos_sim, 
        "MSE": mse
        }
    return state, val_metrics

def train_epoch(
    state, epoch: int, train_ds, configs, rng_streams, metric_writer, ckpt_manager, window_RNG_key, mean_streams
):
    masked_interval_starts_config = configs.masked_interval_starts
    masked_interval_ends_config = configs.masked_interval_ends
    
    ######################################################################################
    if configs.random_mask:
        random_uniform_key_1 = jax.random.uniform(window_RNG_key, minval=0, maxval=1).item()
        random_uniform_key_2 = jax.random.uniform(window_RNG_key, minval=0.10, maxval=1.00).item() # from 10 to 100 percent of a half of a spectra lenght
        spectra_lenght = train_ds["wave_number"][-1].item() - train_ds["wave_number"][0].item()
        spectra_start = train_ds["wave_number"][0].item()
        window_start = spectra_start + random_uniform_key_1 * spectra_lenght / 2
        window_size = random_uniform_key_2 * spectra_lenght / 2
        window_end = window_start + window_size
        masked_interval_starts_config[1] = window_end
        masked_interval_ends_config[0] = window_start
    ######################################################################################
    
    mask_windows = list(
        zip(masked_interval_starts_config, masked_interval_ends_config)
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
        state, batch_metrics = train_step(state, batch, rng_streams["dropout"], mean_streams["mean"])
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
    state, epoch: int, val_ds, configs, rng_streams, metric_writer, ckpt_manager, mean_streams
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
        state, batch_metrics = validation_step(state, batch, rng_streams["dropout"], mean_streams["mean"])
        metrics.append(batch_metrics)

    metrics = stack_forest(metrics)
    avg_metrics = jax.tree_map(jnp.mean, metrics)  # Log the average error of the epoch

    print(f"Validation -- Epoch {epoch + 1} -- CorrGamma Loss {avg_metrics['val_corrected_gamma_loss'].item():.3e}")
    if epoch % configs.log_every_epochs == 0:
        # metric_writer.add_scalar("val/val_poisson_loss", avg_metrics["val_poisson_loss"].item(), state.step)
        # metric_writer.add_scalar("val/val_corrected_poisson_loss", avg_metrics["val_corrected_poisson_loss"].item(), state.step)
        metric_writer.add_scalar("val/val_corrected_gamma_loss", avg_metrics["val_corrected_gamma_loss"].item(), state.step)
        # metric_writer.add_scalar("val/val_gamma_loss", avg_metrics["val_gamma_loss"].item(), state.step)
        # metric_writer.add_scalar("val/cos_sim", avg_metrics["cos_sim"].item(), state.step)
        metric_writer.add_scalar("val/MSE", avg_metrics["MSE"].item(), state.step)
        for gpu_stats in gpustat.new_query():
            log_gpu_usage(gpu_stats.entry, state.step, metric_writer)
        ckpt_manager.save(state.step, state)
    return state, metrics

# ==========================
#     MULTI-DEVICE TRAIN STEP
# ==========================
@partial(
    jax.pmap,
    axis_name="devices", 
    static_broadcasted_argnums=(3,)  # 'configs_mean' is static
)
def train_step_pmap(
    state: TrainState, 
    batch, 
    dropout_key, 
    configs_mean,
    num_devices
):

    # Because 'state.step' is replicated, use state.step[0] for fold_in
    dropout_train_key = jax.random.fold_in(dropout_key, state.step[0])
    
    def my_geometric_mean(loss, eps=1e-8):
            non_negative = abs(loss)
            clipped = jnp.clip(non_negative, eps)
            mean_log = jnp.mean(jnp.log(clipped))
            return jnp.exp(mean_log)
    
    # Local loss per device function
    def corrected_gamma_loss_fn(params):
        pred_spectra = state.apply_fn(
            {"params": params},
            batch["masked_spectra"],
            batch["wave_number"],
            batch["mask"],
            training=True,
            rngs={"dropout": dropout_train_key},
        )
        
        nan_check_pred_spectra = jnp.any(jnp.isnan(pred_spectra))
        inf_check_pred_spectra = jnp.any(jnp.isinf(pred_spectra))
        lax.cond(nan_check_pred_spectra, lambda _: jax.debug.print("NaN in pred_spectra"), lambda _: None, operand=None)
        lax.cond(inf_check_pred_spectra, lambda _: jax.debug.print("Inf in pred_spectra"), lambda _: None, operand=None)
        
        loss_array = (( batch["spectra"]/pred_spectra - 1) - jnp.log( batch["spectra"]/pred_spectra ))
        
        match configs_mean:
            case 'Arithmetic':
                local_loss = loss_array.mean()
            case 'Geometric':
                local_loss = my_geometric_mean(loss_array)
            case _:
                raise Exception(f"You didn't specify a mean to be used!")
        
        return local_loss # scalar per device
    
    
    local_loss, local_grads = jax.value_and_grad(corrected_gamma_loss_fn)(state.params)
    
    log_local = jnp.log(jnp.clip(local_loss, 1e-8))
    sum_log_local = lax.psum(log_local, axis_name="devices")
    mean_log_local = sum_log_local / num_devices
    global_loss = jnp.exp(mean_log_local)
    
    weights = (global_loss / local_loss) / num_devices  # shape: scalar per device
    
    weighted_grads = jax.tree_map(lambda g: g * weights, grads)
    final_grads = jax.tree_map(lambda wg: lax.psum(wg, axis_name="devices"), weighted_grads)
    
    # Check final_grads for NaNs, Infs
    flat_grads, _ = jax.tree_util.tree_flatten(final_grads)
    all_grads = jnp.concatenate([jnp.ravel(g) for g in flat_grads])
    nan_check = jnp.any(jnp.isnan(all_grads))
    inf_check = jnp.any(jnp.isinf(all_grads))
    lax.cond(nan_check, lambda _: jax.debug.print("NaN in final grads"), lambda _: None, operand=None)
    lax.cond(inf_check, lambda _: jax.debug.print("Inf in final grads"), lambda _: None, operand=None)
    
    # Compute gradient parameters for logging
    grad_min = jnp.min(all_grads)
    grad_mean = jnp.mean(all_grads)
    grad_median = jnp.median(all_grads)
    grad_max = jnp.max(all_grads)
    
    # single parameter update
    new_state = state.apply_gradients(grads=final_grads)
    
    train_metrics = {
        "train_loss": global_loss,
        "grad_min": grad_min,
        "grad_mean": grad_mean,
        "grad_median": grad_median,
        "grad_max": grad_max
        }
    return new_state, train_metrics


# ==========================
#     MULTI-DEVICE VALIDATION STEP
# ==========================
@partial(
    jax.pmap,
    axis_name="devices",
    static_broadcasted_argnums=(3,)
)
def validation_step_pmap(
    state: TrainState, 
    batch, dropout_key, 
    configs_mean, 
    num_devices
):

    # Because 'state.step' is replicated, use state.step[0] for fold_in
    dropout_train_key = jax.random.fold_in(dropout_key, state.step[0])
    
    def my_geometric_mean(loss, eps=1e-8):
            non_negative = abs(loss)
            clipped = jnp.clip(non_negative, eps)
            mean_log = jnp.mean(jnp.log(clipped))
            return jnp.exp(mean_log)
    
    # Local loss per device function
    def corrected_gamma_loss_fn(params):
        pred_spectra = state.apply_fn(
            {"params": params},
            batch["masked_spectra"],
            batch["wave_number"],
            batch["mask"],
            training=False,
            rngs={"dropout": dropout_train_key},
        )
        
        nan_check_pred_spectra = jnp.any(jnp.isnan(pred_spectra))
        inf_check_pred_spectra = jnp.any(jnp.isinf(pred_spectra))
        lax.cond(nan_check_pred_spectra, lambda _: jax.debug.print("NaN in pred_spectra"), lambda _: None, operand=None)
        lax.cond(inf_check_pred_spectra, lambda _: jax.debug.print("Inf in pred_spectra"), lambda _: None, operand=None)
        
        loss_array = (( batch["spectra"]/pred_spectra - 1) - jnp.log( batch["spectra"]/pred_spectra ))
        
        match configs_mean:
            case 'Arithmetic':
                local_loss = loss_array.mean()
            case 'Geometric':
                local_loss = my_geometric_mean(loss_array)
            case _:
                raise Exception(f"You didn't specify a mean to be used!")
        
        return local_loss # scalar per device
    
    
    local_loss = corrected_gamma_loss_fn(state.params)
    
    log_local = jnp.log(jnp.clip(local_loss, 1e-8))
    sum_log_local = lax.psum(log_local, axis_name="devices")
    mean_log_local = sum_log_local / num_devices
    global_loss = jnp.exp(mean_log_local)
    
    val_metrics = {
        "val_corrected_gamma_loss": global_loss
        }
    return state, val_metrics

def train_epoch_pmap(
    state, 
    epoch: int, 
    train_ds,
    num_devices,
    configs,
    rng_streams, 
    metric_writer, 
    ckpt_manager, 
    window_RNG_key, 
    mean_streams
):
    if configs.random_mask:
        random_uniform_key_1 = jax.random.uniform(window_RNG_key, minval=0, maxval=1).item()
        random_uniform_key_2 = jax.random.uniform(window_RNG_key, minval=0.10, maxval=1.00).item()
        spectra_lenght = train_ds["wave_number"][-1].item() - train_ds["wave_number"][0].item()
        spectra_start = train_ds["wave_number"][0].item()
        window_start = spectra_start + random_uniform_key_1 * spectra_lenght / 2
        window_size = random_uniform_key_2 * spectra_lenght / 2
        window_end = window_start + window_size
        configs.masked_interval_starts[1] = window_end
        configs.masked_interval_ends[0] = window_start
    
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

    metrics_list = []

    for batch in data_loader:
        # 1) Shard the batch so each device gets a sub-batch
        batch_sharded = shard_batch(batch, num_devices)

        # 2) Create a dropout key for each device
        dropout_sharded = jax.random.split(rng_streams["dropout"], num_devices)

        # 3) Run pmapped train step
        state, batch_metrics = train_step_pmap(
            state, 
            batch_sharded, 
            dropout_sharded, 
            mean_streams["mean"],
            num_devices
            )
        # batch_metrics is a PyTree with shape [num_devices, ...] for each metric

        metrics_list.append(batch_metrics)

    # Convert metrics_list to an array of shape [num_steps, num_devices, ...]
    # and aggregate them. For example:
    stacked_metrics = stack_forest(metrics_list)
    # shape: (num_steps, num_devices, ...) or something similar

    # 4) Average metrics across epoch
    avg_metrics = jax.tree_map(jnp.mean, stacked_metrics)

    print(f"Epoch {epoch+1} -- Loss {avg_metrics['train_loss']:.3e}")

    # 5) Logging & Checkpoints
    if epoch % configs.log_every_epochs == 0:
        metric_writer.add_scalar("train/loss", avg_metrics["train_loss"], state.step[0])
        # if you have other metrics, log them similarly
        for gpu_stats in gpustat.new_query():
            log_gpu_usage(gpu_stats.entry, state.step[0], metric_writer)
        ckpt_manager.save(state.step[0], state)  # use state.step[0] for the integer

    return state, stacked_metrics

def validation_epoch_pmap(
    state, 
    epoch: int, 
    val_ds,
    num_devices,
    configs, 
    rng_streams, 
    metric_writer, 
    ckpt_manager, 
    window_RNG_key, 
    mean_streams
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

    metrics_list = []

    for batch in data_loader:
        # 1) Shard the batch so each device gets a sub-batch
        batch_sharded = shard_batch(batch, num_devices)

        # 2) Create a dropout key for each device
        dropout_sharded = jax.random.split(rng_streams["dropout"], num_devices)

        # 3) Run pmapped train step
        state, batch_metrics = validation_step_pmap(
            state, 
            batch_sharded, 
            dropout_sharded, 
            mean_streams["mean"],
            num_devices
            )
        # batch_metrics is a PyTree with shape [num_devices, ...] for each metric

        metrics_list.append(batch_metrics)

    # Convert metrics_list to an array of shape [num_steps, num_devices, ...]
    # and aggregate them. For example:
    stacked_metrics = stack_forest(metrics_list)
    # shape: (num_steps, num_devices, ...) or something similar

    # 4) Average metrics across epoch
    avg_metrics = jax.tree_map(jnp.mean, stacked_metrics)

    print(f"Validation -- Epoch {epoch + 1} -- ValCorrGamma Loss {avg_metrics['val_corrected_gamma_loss'].item():.3e}")
    
    # 5) Logging & Checkpoints
    if epoch % configs.log_every_epochs == 0:
        metric_writer.add_scalar("val/val_corrected_gamma_loss", avg_metrics["val_corrected_gamma_loss"].item(), state.step[0])
        # if you have other metrics, log them similarly
        for gpu_stats in gpustat.new_query():
            log_gpu_usage(gpu_stats.entry, state.step[0], metric_writer)
        ckpt_manager.save(state.step[0], state)  # use state.step[0] for the integer

    return state, stacked_metrics