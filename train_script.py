from pathlib import Path

import numpy as np
import jax
print("JAX devices: ", jax.devices())
import ml_confs
import optax
import orbax.checkpoint as ocp
import xarray as xr
from etils import epath
from flax.training.common_utils import stack_forest
from flax.training.train_state import TrainState
from tensorboardX import SummaryWriter

from spectraformer.input_pipeline import batch_sampler, preprocess_dataset
from spectraformer.model import SpectraFormer
from spectraformer.train import train_epoch, validation_epoch

jax.config.update("jax_debug_nans", True)

maindir = Path(__file__).parent.resolve()

logdir = maindir / "logs"
ckptdir = maindir / "checkpoints"
# Check if logdir and ckptdir exist, if not create them
logdir.mkdir(parents=True, exist_ok=True)
ckptdir.mkdir(parents=True, exist_ok=True)

datadir = maindir / "data"

model_tag = "min25_CorrGamma"  # CHOOSE ONE (.yaml file should exist)
                    # tag also can be found for already trained models in checkpoints folder

configsdir = maindir / "configs"
configsdir.mkdir(parents=True, exist_ok=True)

config_file_name = f"configs_{model_tag}.yaml"
config_file_path = configsdir / config_file_name


if __name__ == "__main__":
    
    configs = ml_confs.from_file(config_file_path)
    configs.tabulate()



    ####################################################################################################
    # Dataset loading and separation into train/val section
    ####################################################################################################

    # # Full dataset for training
    # train_ds = preprocess_dataset(
    #     xr.load_dataarray(datadir / f"{configs.train_dataset}.nc")
    # )



    # # Part of dataset for training, part for evaluation
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
    # 
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
        training=False,
    )

    state = TrainState.create(
        apply_fn=jax.jit(
            model.apply, static_argnames=("training", "capture_intermediates")
        ),
        params=variables["params"],
        tx=optax.adam(configs.learning_rate),
    )

    # # Checkpointing: load from checkpoint and resume training if available
    ckpt_options = ocp.CheckpointManagerOptions(max_to_keep=5)
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
    # early_stop = EarlyStopping(min_delta=1e-3, patience=2)
    train_metrics = []
    val_metrics = []
    
    ####################################################################################################
    # Training & metrics calculation section
    ####################################################################################################
    for epoch in range(configs.num_epochs):
        # Key updating
        window_RNG_key = jax.random.split(window_RNG_key, num=1)[0]
        # Training
        state, epoch_train_metrics = train_epoch(
            state, epoch, train_ds, configs, rng_streams, metric_writer, ckpt_manager, window_RNG_key
        )
        train_metrics.append(epoch_train_metrics)
        
        # Validation
        state, epoch_val_metrics = validation_epoch(
            state, epoch, val_ds, configs, rng_streams, metric_writer, ckpt_manager
        )
        val_metrics.append(epoch_val_metrics)
        
        
        
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
