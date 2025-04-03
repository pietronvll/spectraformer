from pathlib import Path
from etils import epath

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
print("JAX devices: ", jax.devices())
num_devices = len(jax.devices())
print("Number of devices: ", num_devices)

import ml_confs
import optax
import orbax.checkpoint as ocp
import xarray as xr
from flax.training.common_utils import stack_forest
from flax.training.train_state import TrainState
from tensorboardX import SummaryWriter
from dataclasses import replace

from spectraformer.input_pipeline import batch_sampler, preprocess_dataset
from spectraformer.model import SpectraFormer
from spectraformer.train import train_epoch, validation_epoch, train_epoch_pmap, validation_epoch_pmap

jax.config.update("jax_debug_nans", True)

maindir = Path(__file__).parent.resolve()

logdir = maindir / "logs"
ckptdir = maindir / "checkpoints"
# Check if logdir and ckptdir exist, if not create them
logdir.mkdir(parents=True, exist_ok=True)
ckptdir.mkdir(parents=True, exist_ok=True)

datadir = maindir / "data"

model_tag = "min42_GeomLoss_LRSchedule_4cycles_decline0p8"  # CHOOSE ONE (.yaml file should exist)
                    # tag also can be found for already trained models in checkpoints folder

training_regime = "All devices" # one from ["One device", "All devices"]

configsdir = maindir / "configs"
configsdir.mkdir(parents=True, exist_ok=True)

config_file_name = f"configs_{model_tag}.yaml"
config_file_path = configsdir / config_file_name

# Test with simplest possible pmap function
@jax.pmap
def dummy_pmap(x):
    jax.debug.print("DUMMY PMAP WORKING - input shape: {}", x.shape)
    return x + 1

if __name__ == "__main__":
        

    # Try with known-good input
    dummy_input = jnp.ones((jax.device_count(), 1))  # (4,1) for 4 devices
    print("Dummy pmap result:", dummy_pmap(dummy_input))
    
    
    configs = ml_confs.from_file(config_file_path)
    configs.tabulate()
    
    if training_regime=="All devices" and configs.batch_size % num_devices !=0:
        raise Exception(f"Sharding requires batch size divisibility by the number of devices. Change it accordingly (preferably to 24).")
    
    # This is an implementation of learning rate schedule - multiple cosine decay cycles from init_value to init_value*alpha, then repeating from init_value.  
    cosine_kwargs = []
    
    init_value = 0.1*configs.learning_rate
    peak_value = configs.learning_rate
    warmup_steps = 1000 if not hasattr(configs, 'warmup_steps') else configs.warmup_steps
    decay_steps = 2000 if not hasattr(configs, 'decay_steps') else configs.decay_steps
    decline_coeff = 1 if not hasattr(configs, 'decline_coeff') else configs.decline_coeff
    
    for i in range(100 if not hasattr(configs, 'num_cycles') else configs.num_cycles):
        end_value = decline_coeff * init_value
        # 100 cycles - because i don't want to think much about making a cycle per N epochs. Schedule is built for steps.
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
    
    def create_rank_safe_adam(configs, cosine_kwargs):
        """Creates an Adam optimizer"""
        learning_rate_decay = getattr(configs, 'learning_rate_decay', 'Constant')
        
        match learning_rate_decay:
            case "Multiple cosine decay cycles":
                lr_schedule = optax.schedules.sgdr_schedule(cosine_kwargs=cosine_kwargs)
                base_optimizer = optax.adam(learning_rate=lr_schedule)
            case "Constant":
                base_optimizer = optax.adam(learning_rate=configs.learning_rate)
            case _:
                raise ValueError(f"Unknown schedule: {learning_rate_decay}")

        # Wrapper to ensure rank-safe states
        def init_fn(params):
            opt_state = base_optimizer.init(params)
            # Convert scalar counts to rank-1
            return jax.tree_map(
                lambda x: jnp.expand_dims(x, 0) if hasattr(x, 'ndim') and x.ndim == 0 else x,
                opt_state
            )
        
        def update_fn(grads, state, params):
            updates, new_state = base_optimizer.update(grads, state, params)
            new_params = optax.apply_updates(params, updates)
            # Maintain rank-1 for scalar states
            new_state = jax.tree_map(
                lambda x: jnp.expand_dims(x, 0) if hasattr(x, 'ndim') and x.ndim == 0 else x,
                new_state
            )
            return new_params, new_state
        
        return optax.GradientTransformation(init_fn, update_fn)
    
    tx = create_rank_safe_adam(configs, cosine_kwargs)
    
    
    # learning_rate_decay = getattr(configs, 'learning_rate_decay', 'Constant')
    # match learning_rate_decay:
    #     case "Multiple cosine decay cycles":
    #         learning_rate_fn = optax.schedules.sgdr_schedule(cosine_kwargs=cosine_kwargs)
    #         tx = optax.adam(learning_rate=learning_rate_fn)
    #     case "Constant":
    #         tx = optax.adam(learning_rate=configs.learning_rate)
    #     case _:
    #         raise Exception(f"You didn't specify a learning rate schedule!")
    
    ####################################################################################################
    # Dataset loading and separation into train/val section
    #################################################################################################### 
    # Load the full dataset
    full_ds = preprocess_dataset(
        xr.load_dataarray(datadir / f"{configs.train_dataset}.nc")
    )
    # Define the split fraction and random seed
    split_fraction = 0.8  # 80% for training, 20% for validation
    rng_seed = configs.root_rng_seed  # Use the seed from the config for reproducibility
    # Shuffle indices
    np.random.seed(rng_seed)
    indices = np.arange(len(full_ds))
    np.random.shuffle(indices)
    # Split indices
    split_index = int(len(indices) * split_fraction)
    train_indices = indices[:split_index]
    val_indices = indices[split_index:]
    # Split dataset
    train_ds = full_ds[train_indices]
    val_ds = full_ds[val_indices]
    print(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}, Total: {len(full_ds)}={len(train_ds)+len(val_ds)}")
    print("Filtered train dataset shape:", train_ds.shape)
    # ####################################################################################################
    # END of "Dataset loading and separation into train/val section"
    # ####################################################################################################
    
    mask_windows = list(
        zip(configs.masked_interval_starts, configs.masked_interval_ends)
    )
    
    dummy_example = next(batch_sampler(train_ds, mask_windows, batch_size=1))
    print(f"Train dataset of length {len(train_ds.spectra)} with leaves of shape:")
    for k, v in dummy_example.items():
        print(f"  {k} -> {v.shape}")
    
    model = SpectraFormer(
        num_heads=configs.num_heads,
        num_layers=configs.num_layers,
        embedding_dim=configs.embedding_dim,
        dropout_rate=configs.dropout_rate
    )
    
    # RNG Keys
    root_key = jax.random.key(seed=configs.root_rng_seed)
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
    
    state = TrainState.create(
        apply_fn=jax.jit(
            model.apply, static_argnames=("training", "capture_intermediates")
        ),
        params=variables["params"],
        tx=tx
    )
    
    # Checkpointing: load from checkpoint and resume training if available
    ckpt_options = ocp.CheckpointManagerOptions(
        #----------------------------------------------------------------------------------------------------#
        # max_to_keep=5
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
        state = ckpt_manager.restore(
            ckpt_manager.latest_step(), args=ocp.args.StandardRestore(state)
        )
        print(f"Resuming training from step {state.step}.")
    else:
        print(f"No checkpoint found with tag {configs.tag}, training from scratch.")
    
    metric_writer = SummaryWriter(logdir / configs.tag)
    rng_streams = {"dropout": dropout_key}
    mean_streams = {"mean": "Not specified" if not hasattr(configs, 'mean') else configs.mean}
    # early_stop = EarlyStopping(min_delta=1e-3, patience=2)
    train_metrics = []
    val_metrics = []
    
    # This is for drawing on TensorBoard both train and validation losses on a single graph
    layout = {
        "my_layout": {
            "loss": ["Multiline", ["train/loss", "val/val_corrected_gamma_loss"]],
            },
        }
    metric_writer.add_custom_scalars(layout)
    
    ####################################################################################################
    # Training & metrics calculation section
    ####################################################################################################
    
    for epoch in range(configs.num_epochs):
        # Key updating
        window_RNG_key = jax.random.split(window_RNG_key, num=1)[0]
        
        match training_regime:
            case "One device":
                
                # Training
                state, epoch_train_metrics = train_epoch(
                    state, epoch, train_ds, configs, rng_streams, metric_writer, ckpt_manager, window_RNG_key, mean_streams
                )
                train_metrics.append(epoch_train_metrics)
                
                # Validation
                state, epoch_val_metrics = validation_epoch(
                    state, epoch, val_ds, configs, rng_streams, metric_writer, ckpt_manager, mean_streams
                )
                val_metrics.append(epoch_val_metrics)
                
            case "All devices":
                
                state, epoch_train_metrics = train_epoch_pmap(
                    state=state, 
                    epoch=epoch, 
                    train_ds=train_ds,
                    configs=configs, 
                    rng_streams=rng_streams, 
                    metric_writer=metric_writer, 
                    ckpt_manager=ckpt_manager, 
                    window_RNG_key=window_RNG_key, 
                    mean_streams=mean_streams
                    )
                train_metrics.append(epoch_train_metrics)
                
                state, epoch_val_metrics = validation_epoch_pmap(
                    state=state, 
                    epoch=epoch, 
                    val_ds=val_ds,
                    configs=configs, 
                    rng_streams=rng_streams, 
                    metric_writer=metric_writer, 
                    ckpt_manager=ckpt_manager, 
                    window_RNG_key=window_RNG_key, 
                    mean_streams=mean_streams
                    )
                val_metrics.append(epoch_val_metrics)
                
            case _:
                raise Exception(f"Specify training_regime correctly!")
        
        # # Early stop (?)
        # early_stop = early_stop.update(metrics["loss"])
        # if early_stop.should_stop:
        #     print(f"Met early stopping criteria, breaking at epoch {epoch}")
        #     break
    
    # Need to save metrics to the writer
    train_metrics = stack_forest(train_metrics)
    val_metrics = stack_forest(val_metrics)
    
    ckpt_manager.wait_until_finished()
    metric_writer.close()
