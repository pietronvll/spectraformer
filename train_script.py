import gpustat

from pathlib import Path
from etils import epath

# import numpy as np
# import pandas as pd

import jax
import jax.numpy as jnp
print("JAX devices: ", jax.devices())
num_devices = len(jax.devices())
print("Number of devices: ", num_devices)

import ml_confs
import optax
import orbax.checkpoint as ocp
# import xarray as xr
# from flax.training.common_utils import stack_forest
from flax.training.train_state import TrainState
from flax.training.early_stopping import EarlyStopping

class CustomTrainState(TrainState):
    epoch: jax.Array

@jax.pmap
def update_epoch(state):
    return state.replace(epoch=state.epoch + 1)

from tensorboardX import SummaryWriter
# from dataclasses import replace

from spectraformer.input_pipeline import batch_sampler, dataset_loader #, preprocess_dataset
from spectraformer.model import SpectraFormer
from spectraformer.train import train_epoch, validation_epoch, train_epoch_pmap, validation_epoch_pmap, log_gpu_usage
from spectraformer.inference import plot_results_train, predict, plot_loss, plot_dataset_pairs

jax.config.update("jax_debug_nans", True)

maindir = Path(__file__).parent.resolve()

logdir = maindir / "logs"
ckptdir = maindir / "checkpoints"
# Check if logdir and ckptdir exist, if not create them
logdir.mkdir(parents=True, exist_ok=True)
ckptdir.mkdir(parents=True, exist_ok=True)

datadir = maindir / "data"

model_tag = "min65_ArithmLoss_multidata_highf_LRschedule"  # CHOOSE ONE (.yaml file should exist)
=======
model_tag = "min66_highf"  # CHOOSE ONE (.yaml file should exist)
>>>>>>> 306c757cf1727f5269ca4ac4e5303f86d6f357f0
                    # tag also can be found for already trained models in checkpoints folder



training_regime = "All devices" # one from ["One device", "All devices"]

configsdir = maindir / "configs"
configsdir.mkdir(parents=True, exist_ok=True)

config_file_name = f"configs_{model_tag}.yaml"
config_file_path = configsdir / config_file_name

material_name = "SiC-high-f"  # The directory name in parsed_data to load from
parsed_datadir = datadir / "parsed_data"  # Change this to point to parsed_data

# Find all .nc files in the material directory and its subdirectories
material_dir = parsed_datadir / material_name
nc_files = list(material_dir.rglob("*.nc"))
if not nc_files:
    raise ValueError(f"No .nc files found in {material_dir}")

if __name__ == "__main__":
    
    
    configs = ml_confs.from_file(config_file_path)
    configs.tabulate()
    is_early_stop = True if not hasattr(configs, 'is_early_stop') else configs.is_early_stop # turning on early stopping process
    min_delta = 1e-4 if not hasattr(configs, 'early_stop_min_delta') else configs.early_stop_min_delta
    patience = 5 if not hasattr(configs, 'early_stop_patience') else configs.early_stop_patience
    
    if training_regime=="All devices" and configs.batch_size % num_devices !=0:
        raise Exception(f"Sharding requires batch size divisibility by the number of devices. Change it accordingly (preferably to 24).")
    
    # This is an implementation of learning rate schedule - multiple cosine decay cycles from init_value to init_value*alpha, then repeating from init_value.  
    cosine_kwargs = []
    
    init_value = 0.1*configs.learning_rate if not hasattr(configs, 'warmup_coeff') else configs.warmup_coeff*configs.learning_rate
    peak_value = configs.learning_rate
    warmup_steps = 1000 if not hasattr(configs, 'warmup_steps') else configs.warmup_steps
    decay_steps = 2000 if not hasattr(configs, 'decay_steps') else configs.decay_steps
    decline_coeff = 1 if not hasattr(configs, 'decline_coeff') else configs.decline_coeff
    
    for i in range(20 if not hasattr(configs, 'num_cycles') else configs.num_cycles):
        end_value = decline_coeff * init_value
        # 100 cycles - because i don't want to think much about making a cycle per N epochs. Schedule is built for steps. UPD: 20 cycles is more than enough
        cycle_dict = {
            "init_value": init_value, 
            "peak_value": peak_value, 
            "warmup_steps": warmup_steps,
            "decay_steps": decay_steps,            
            "end_value": end_value
        }
        cosine_kwargs.append(cycle_dict)
        init_value = end_value
        peak_value *= decline_coeff
    
    #                           LR schedule graph
    #
    # - - - - - - - - - - - - - - - ___* ___________ - - - - - - - - - - - - - - - - - - - - - - - - > configs.learning_rate (without a schedule it is constant)
    #|                     _______*/   |            \*___                |                         ^
    #|            _______*/            |                 \*___           |                         |
    #|  _______*/                      |                      \*         |                         v
    #|*/                               |                        \________* - - - - - - - - - - - - - > 
    #|                                 |                                 |
    #|           warmup_steps          |                                 |
    #|<------------------------------->|                                 |--------------------------->
    #|       Linear warm-up from       |                                 |
    #|          init_value to          |                                 | Repeat the cycle 100 times
    #|            peak_value           |    decay_steps - warmup_steps   |
    #|                                 |<------------------------------->|
    #|                                 |        Cosine decay from        |
    #|                                 |          peak_value to          |
    #|                                 |            end_value            |
    #|                           decay_steps                             |
    #|<----------------------------------------------------------------->|
    
    learning_rate_decay = getattr(configs, 'learning_rate_decay', 'Constant')
    match learning_rate_decay:
        case "Multiple cosine decay cycles":
            learning_rate_fn = optax.schedules.sgdr_schedule(cosine_kwargs=cosine_kwargs)
            tx = optax.adam(learning_rate=learning_rate_fn)
        case "Constant":
            tx = optax.adam(learning_rate=configs.learning_rate)
        case _:
            raise Exception(f"You didn't specify a learning rate schedule!")
    
    # New automatic dataset loading
    datasets = []
    dataset_names = []
    # No filtering load
    for nc_file in nc_files:
        # Get relative path from datadir (e.g., "SiC/subdir/filename.nc")
        relative_path = nc_file.relative_to(parsed_datadir)
        
        train_ds, val_ds = dataset_loader(
            datadir=parsed_datadir,
            file_location_with_name=str(relative_path),
            shuffle_rng_seed=configs.root_rng_seed,
            is_filter=False,
            option='whitaker_hayes'
        )
        # Load only those, who is large enough to be treated in parallel
        if train_ds.sizes['spectra'] >= configs.batch_size and val_ds.sizes['spectra']>=configs.batch_size:
            datasets.append((train_ds, val_ds))
            dataset_names.append(nc_file.name)
    
    #  Filtering load - only if filtering is set (to not double-load same data)
    if hasattr(configs, 'is_filter') and configs.is_filter:
        for nc_file in nc_files:
            # Get relative path from datadir (e.g., "SiC/subdir/filename.nc")
            relative_path = nc_file.relative_to(parsed_datadir)
            
            train_ds, val_ds = dataset_loader(
                datadir=parsed_datadir,
                file_location_with_name=str(relative_path),
                shuffle_rng_seed=configs.root_rng_seed,
                is_filter=configs.is_filter if hasattr(configs, 'is_filter') else False,
                option='whitaker_hayes'
            )
            # Load only those, who is large enough to be treated in parallel
            if train_ds.sizes['spectra'] >= configs.batch_size and val_ds.sizes['spectra']>=configs.batch_size:
                datasets.append((train_ds, val_ds))

    print(f"\n===== Loaded {len(datasets)}/{len(nc_files)} datasets from {material_dir} =====\n")
    # plot_dataset_pairs(datasets, save_dir='temp/datasets_plots/no_dropping/2', nc_files=dataset_names)
    
    mask_windows = list(
        zip(configs.masked_interval_starts, configs.masked_interval_ends)
    )
    
    dummy_example = next(batch_sampler(datasets[0][0], mask_windows, batch_size=1))
    dummy_wave_number = jnp.squeeze(dummy_example["wave_number"])
    
    print(dummy_wave_number)
    print(f"\nDummy example -- Train dataset of length {len(datasets[0][0].spectra)} with leaves of shape:")
    for k, v in dummy_example.items():
        print(f"  {k} -> {v.shape}")
    
    model = SpectraFormer(
        num_heads=configs.num_heads,
        num_layers=configs.num_layers,
        embedding_dim=configs.embedding_dim,
        dropout_rate=configs.dropout_rate
    )
    
    # RNG Keys
    root_key = jax.random.PRNGKey(seed=configs.root_rng_seed)
    main_key, params_key, dropout_key = jax.random.split(key=root_key, num=3)
    window_RNG_key = jax.random.split(main_key, num=1)[0]
    
    # Model Initialization
    variables = model.init(
        params_key,
        dummy_example["masked_spectra"][0],
        dummy_example["wave_number"],
        dummy_example["mask"],
        training=True,
    )
    
    state = CustomTrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        epoch=jnp.array(0, dtype=jnp.int32),
        # step=jnp.array(0, dtype=jnp.int32)
    )
    # After state creation
    print("\nInitial step type:", type(state.step))  # Should be DeviceArray
    
    # Checkpointing: load from checkpoint and resume training if available
    ckpt_options = ocp.CheckpointManagerOptions(
        #----------------------------------------------------------------------------------------------------#
        max_to_keep=patience+1 # this is for having best model in case of training process only worsens the loss
        #----------------------------------------------------------------------------------------------------#
        )
    
    if not epath.Path(ckptdir / configs.tag).exists():
        epath.Path(ckptdir / configs.tag).mkdir()
        epath.Path(ckptdir / configs.tag / ".tmp").touch()
    
    ckpt_manager = ocp.CheckpointManager(
        ckptdir / configs.tag,
        options=ckpt_options,
        item_handlers=ocp.StandardCheckpointHandler(),
        metadata=configs.to_dict(),
    )
    
    # After initialization remove the dummy file
    if epath.Path(ckptdir / configs.tag / ".tmp").exists():
        epath.Path(ckptdir / configs.tag / ".tmp").rmtree()
    
    if len(ckpt_manager.all_steps()) > 0:
        restored = ckpt_manager.restore(
            ckpt_manager.latest_step(),
            args=ocp.args.StandardRestore({"state": state})
        )
        state = restored["state"]
        print(f"Resuming from epoch {state.epoch}, step {state.step}")
    else:
        print(f"No checkpoint found with tag {configs.tag}, training from scratch.")
    
    restored_epoch = state.epoch
    print(f'Restored epoch: {restored_epoch}')
    
    metric_writer = SummaryWriter(logdir / configs.tag)
    rng_streams = {"dropout": dropout_key}
    mean_streams = {"mean": "Not specified" if not hasattr(configs, 'mean') else configs.mean}
    
    # Early stopping initialization
    early_stop = EarlyStopping(min_delta=min_delta, patience=patience)
    
    train_metrics = []
    val_metrics = []
    
    # This is for drawing on TensorBoard both train and validation losses on a single graph
    layout = {
        "my_layout": {
            "loss_step": ["Multiline", ["train/train_loss_step", "val/val_loss_step"]],
            "loss_epoch": ["Multiline", ["train/train_loss_epoch", "val/val_loss_epoch"]],
            },
        }
    metric_writer.add_custom_scalars(layout)
    
    # metric_writer.add_graph(model=model)
    
    ####################################################################################################
    # Training & metrics calculation section
    ####################################################################################################
    state = jax.device_put_replicated(state, jax.devices())
    print(f"\n===DEBUGGING===          Replicated step shape before main loop: {jnp.shape(state.step)}\n")  # (num_devices,)
    print(f"\n===DEBUGGING===          Replicated epoch shape before main loop: {jnp.shape(state.epoch)}\n")  # (num_devices,)

    # Main training loop
    for epoch in range(restored_epoch + 1, restored_epoch + configs.num_epochs + 1):
        
        print(f'\n==== Epoch {epoch} -- Begin ====\n')
        
        # Restoring early stop ???
        
        # Key updating
        window_RNG_key = jax.random.split(window_RNG_key, num=1)[0]
        
        # Reset metrics for the epoch
        epoch_train_metrics = []
        epoch_val_metrics = []
        
        # Shuffle dataset order each epoch
        dataset_order = jax.random.permutation(window_RNG_key, len(datasets))
        print('Dataset order: ',dataset_order)
        
        for ds_idx in dataset_order:
            train_ds, val_ds = datasets[ds_idx]
            print(f'Dataset number: {ds_idx}, Train dataset lenght: {train_ds.shape[1]}, Val dataset lenght: {val_ds.shape[1]}')
            
            match training_regime:
                case "One device":
                    
                    # Training
                    state, train_metrics_ds = train_epoch(
                        state, epoch, train_ds, configs, rng_streams,
                        metric_writer, ckpt_manager, window_RNG_key, mean_streams
                    )
                    # Validation
                    state, val_metrics_ds = validation_epoch(
                        state, epoch, val_ds, configs, rng_streams, 
                        metric_writer, ckpt_manager, mean_streams
                    )
                case "All devices":
                    
                    state, train_metrics_ds = train_epoch_pmap(
                        state=state, epoch=epoch, train_ds=train_ds, configs=configs, 
                        rng_streams=rng_streams, metric_writer=metric_writer, ckpt_manager=ckpt_manager, 
                        window_RNG_key=window_RNG_key, mean_streams=mean_streams
                        )
                    state, val_metrics_ds = validation_epoch_pmap(
                        state=state, epoch=epoch, val_ds=val_ds, configs=configs, 
                        rng_streams=rng_streams, metric_writer=metric_writer, ckpt_manager=ckpt_manager,
                        window_RNG_key=window_RNG_key, mean_streams=mean_streams
                        )
                case _:
                    raise Exception(f"Specify training_regime correctly!")
            
            epoch_train_metrics.append(train_metrics_ds)
            epoch_val_metrics.append(val_metrics_ds)
        
        # print(f"epoch_train_metrics: {epoch_train_metrics['train_loss_step']}")
        # print(f"epoch_val_metrics: {epoch_val_metrics['val_loss_step']}")
        
        
        train_metrics.append(
            jax.tree.map(lambda *xs: jnp.mean(jnp.stack(xs)), *epoch_train_metrics)
        )
        val_metrics.append(
            jax.tree.map(lambda *xs: jnp.mean(jnp.stack(xs)), *epoch_val_metrics)
        )
        
        # Write epoch+1 to the state
        state = update_epoch(state)
        
        # Logging
        if epoch % configs.log_every_epochs == 0:
            
            params0 = jax.tree.map(lambda x: x[0], state.params)
            
            # Making a prediction on a dummy for logging in tensorboard
            dummy_prediction = predict(
                state.apply_fn,
                {"params": params0},
                dummy_example,
                dummy_example["mask"],
            )
            
            # Calculating a loss for plotting
            dummy_spectra = jnp.squeeze(dummy_example["spectra"])
            dummy_pred_spectra = jnp.squeeze(dummy_prediction["predicted_spectra"])
            
            match configs.loss_fn if hasattr(configs, 'loss_fn') else "CorrGamma":
                case "MSE":
                    loss = (dummy_pred_spectra - dummy_spectra) ** 2
                case "CorrGamma":
                    dummy_ratio = dummy_spectra / dummy_pred_spectra
                    loss = (( dummy_ratio - 1) - jnp.log( dummy_ratio ))
                case _:
                    raise Exception(f"Specify loss_fn correctly in config!")
            
            
            
            loss = (( dummy_ratio - 1) - jnp.log( dummy_ratio ))
            
            # Making a plot as in a dashboard
            fig_res, ax_res = plot_results_train(dummy_prediction, state.step[0], state.epoch[0], model_tag)
            metric_writer.add_figure('model_predictions', fig_res, global_step=state.epoch[0])
            
            # print(dummy_wave_number)
            # print(loss)
            # print(dummy_example["spectra"])
            # print(dummy_prediction["predicted_spectra"])
            
            fig_loss, ax_loss = plot_loss(dummy_wave_number, loss, state.step[0], state.epoch[0], model_tag)
            metric_writer.add_figure('model_prediction_losses', fig_loss, global_step=state.epoch[0])
            
            metric_writer.add_scalar("train/train_loss_epoch",          train_metrics[-1]["train_loss_step"],   state.epoch[0])
            metric_writer.add_scalar("val/val_loss_epoch",              val_metrics[-1]["val_loss_step"],       state.epoch[0])
            
            metric_writer.add_scalar("train/train_loss_step",           train_metrics[-1]["train_loss_step"],   state.step[0])
            metric_writer.add_scalar("val/val_loss_step",               val_metrics[-1]["val_loss_step"],       state.step[0])
            
            metric_writer.add_scalar("grad/train/grad_min_step",        train_metrics[-1]["grad_min"],          state.step[0])
            metric_writer.add_scalar("grad/train/grad_mean_step",       train_metrics[-1]["grad_mean"],         state.step[0])
            metric_writer.add_scalar("grad/train/grad_median_step",     train_metrics[-1]["grad_median"],       state.step[0])
            metric_writer.add_scalar("grad/train/grad_max_step",        train_metrics[-1]["grad_max"],          state.step[0])
            
            for gpu_stats in gpustat.new_query():
                log_gpu_usage(gpu_stats.entry, state.step[0], metric_writer)
            
            single_state = jax.tree_util.tree_map(lambda x: x[0], state)
            ckpt_manager.save(
                step=single_state.step,
                items={"state": single_state}
            )
        print(f'\n==== Epoch {epoch} -- End ====\n')
        
        # Early stop
        early_stop = early_stop.update(val_metrics[-1]["val_loss_step"])
        best_metric_value = min(metric["val_loss_step"] for metric in val_metrics)
        metrics_difference = best_metric_value - val_metrics[-1]["val_loss_step"]
        print(f"\n==== Metrics difference w.r.t. the best value: {metrics_difference:.5e} ==== Patience count {early_stop.patience_count}\n")
        if is_early_stop and early_stop.should_stop:
            print(f"Met early stopping criteria, breaking at epoch {epoch}. Last saved epoch is {epoch}.")
            break
        
    ckpt_manager.wait_until_finished()
    metric_writer.close()
    
    print("See you later, Alligator!")
