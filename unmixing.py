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

model_tag = "min23_CorrGamma"  # CHOOSE ONE (.yaml file should exist)
                    # tag also can be found for already trained models in checkpoints folder



configsdir = maindir / "configs"
configsdir.mkdir(parents=True, exist_ok=True)

config_file_name = f"configs_{model_tag}.yaml"
config_file_path = configsdir / config_file_name

def load_model(
    model_tag: str,
    dataset
    ):
    ckpt_manager = ocp.CheckpointManager(
        ckptdir + 'spectraformer:' + model_tag,
        item_handlers=ocp.StandardCheckpointHandler(),
    )
    configs = ml_confs.from_dict(ckpt_manager.metadata())
    mask_windows = list(
        zip(configs.masked_interval_starts, configs.masked_interval_ends)
    )
    dummy_example = next(batch_sampler(dataset, mask_windows, batch_size=1))
    # Re-initialize model based on loaded configs
    model = SpectraFormer(
        num_heads=configs.num_heads,
        num_layers=configs.num_layers,
        embedding_dim=configs.embedding_dim,
    )

    # RNG Keys
    root_key = jax.random.key(seed=configs.root_rng_seed)
    main_key, params_key, dropout_key = jax.random.split(key=root_key, num=3)

    # Model Initialization
    variables = model.init(
        params_key,
        dummy_example["masked_spectra"][0],
        dummy_example["wave_number"],
        dummy_example["mask"],
        training=False,
    )

    state = TrainState.create(
        apply_fn=jax.jit(model.apply, static_argnames=("training",)),
        params=variables["params"],
        tx=optax.adam(configs.learning_rate),
    )
    # Restore checkpoint
    state = ckpt_manager.restore(
        ckpt_manager.latest_step(), args=ocp.args.StandardRestore(state)
    )

    # Loading Databases
    test_data = list(batch_sampler(test_ds, mask_windows, shuffle=False, batch_size=1))

    spectraformer_predictions = [
        predict(
            state.apply_fn,
            {"params": state.params},
            datapoint,
            datapoint["mask"],
        )
        for datapoint in test_data
    ]
    return state, spectraformer_predictions


if __name__ == "__main__":
    
    # Config file reading
    configs = ml_confs.from_file(config_file_path)
    configs.tabulate()
    
    
    
    
    
    print('Unmixing done.')