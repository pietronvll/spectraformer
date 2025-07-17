from pathlib import Path

import numpy as np
# import pandas as pd
from scipy.signal import savgol_filter
import jax
import jax.numpy as jnp
print("JAX devices: ", jax.devices())
import ml_confs
import optax
import orbax.checkpoint as ocp
import xarray as xr
from etils import epath
# from flax.training.common_utils import stack_forest
from flax.training.train_state import TrainState
# from tensorboardX import SummaryWriter

from spectraformer.input_pipeline import batch_sampler, preprocess_dataset
from spectraformer.model import SpectraFormer
# from spectraformer.train import train_epoch, validation_epoch
from spectraformer.inference import predict #, plot_results

jax.config.update("jax_debug_nans", True)

maindir = Path(__file__).parent.resolve()

logdir = maindir / "logs"
ckptdir = maindir / "saved_models" / "checkpoints"
# Check if logdir and ckptdir exist, if not create them
logdir.mkdir(parents=True, exist_ok=True)
ckptdir.mkdir(parents=True, exist_ok=True)

datadir = maindir / "data"

# ####################################################################################################
# Section of Parameters choise for unmixing
# ####################################################################################################

model_tag = "min62_ArithmLoss_multidata_highf_LRschedule"  # CHOOSE ONE (.yaml file should exist)
                    # tag also can be found for already trained models in checkpoints folder
material = 'buffer+graphene' #Change this accordingly to the folder name where your mixtures are

# Savgol filter parameters
window_length = 100
polyorder = 3

# ####################################################################################################
# END of Section of Parameters choise for unmixing
# ####################################################################################################

configsdir = maindir / "configs"
configsdir.mkdir(parents=True, exist_ok=True)

config_file_name = f"configs_{model_tag}.yaml"
config_file_path = configsdir / config_file_name

mixdir = datadir / "parsed_data" / material
mixdir.mkdir(parents=True, exist_ok=True)

unmixdir = datadir / "unmixed"
unmixdir.mkdir(parents=True, exist_ok=True)

unmixdir_model = unmixdir / model_tag
unmixdir_model.mkdir(parents=True, exist_ok=True)

unmixdir_model_material = unmixdir_model / material
unmixdir_model_material.mkdir(parents=True, exist_ok=True)

class CustomTrainState(TrainState):
    epoch: jax.Array

def load_model(
    configs,
    dataset
    ):
    # For unmixing this schedule is important to keep this as a model parameter. It is necessary for checkpoint matching
    
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
    
    
    learning_rate_fn = optax.schedules.sgdr_schedule(cosine_kwargs=cosine_kwargs)
    
    learning_rate_decay = getattr(configs, 'learning_rate_decay', 'Constant')
    match learning_rate_decay:
        case "Multiple cosine decay cycles":
            tx = optax.adam(learning_rate=learning_rate_fn)
        case "Constant":
            tx = optax.adam(learning_rate=configs.learning_rate)
        case _:
            raise Exception(f"You didn't specify a learning rate schedule!")
    
    mask_windows = list(
        zip(configs.masked_interval_starts, configs.masked_interval_ends)
    )
    
    dummy_example = next(batch_sampler(dataset, mask_windows, batch_size=1))
    
    model = SpectraFormer(
        num_heads=configs.num_heads,
        num_layers=configs.num_layers,
        embedding_dim=configs.embedding_dim,
    )
    
    root_key = jax.random.key(seed=configs.root_rng_seed)
    main_key, params_key, dropout_key = jax.random.split(key=root_key, num=3)
    
    variables = model.init(
        params_key,
        dummy_example["masked_spectra"][0],
        dummy_example["wave_number"],
        dummy_example["mask"],
        training=False,
    )

    state = CustomTrainState.create(
        apply_fn=jax.jit(model.apply, static_argnames=("training",)),
        params=variables["params"],
        tx=tx,
        epoch=jnp.array(0, dtype=jnp.int32),
        # step=jnp.array(0, dtype=jnp.int32)
    )
    
    ckpt_options = ocp.CheckpointManagerOptions(max_to_keep=5, read_only=True)
    
    ckpt_manager = ocp.CheckpointManager(
        ckptdir / configs.tag,
        options=ckpt_options,
        item_handlers=ocp.StandardCheckpointHandler(),
        metadata=configs.to_dict(),
    )
    
    # After initialization remove the dummy file
    if epath.Path(ckptdir / configs.tag / ".tmp").exists():
        epath.Path(ckptdir / configs.tag / ".tmp").rmtree()
    
    # Restore checkpoint
    try:
        restored = ckpt_manager.restore(
            # ckpt_manager.latest_step(),
            90945,
            args=ocp.args.StandardRestore({"state": state})
        )
        state = restored["state"]
    except ValueError:
        old_state = TrainState.create(
                apply_fn=model.apply,
                params=variables["params"],
                tx=tx
                )
        state = old_state
        state = ckpt_manager.restore(
                ckpt_manager.latest_step(), 
                args=ocp.args.StandardRestore(state)
                )
    print(f"Checkpoint restored from step {state.step}.")
    
    return state

def prediction_fn(
    configs,
    dataset,
    state
    ):
    
    mask_windows = list(
        zip(configs.masked_interval_starts, configs.masked_interval_ends)
    )
    
    # Loading Databases
    test_data = list(batch_sampler(dataset, mask_windows, shuffle=False, batch_size=1))
    
    spectraformer_predictions = [
        predict(
            state.apply_fn,
            {"params": state.params},
            datapoint,
            datapoint["mask"],
        )
        for datapoint in test_data
    ]
    return spectraformer_predictions

if __name__ == "__main__":
    
    # Config reading
    configs = ml_confs.from_file(config_file_path)
    configs.tabulate()
    
    # Loading a train dataset to initialize model
    train_ds = preprocess_dataset(xr.load_dataarray(datadir / f"{configs.train_dataset}.nc"), option='whitaker_hayes_with_outliers')
    
    # Initializing a model
    state = load_model(configs, dataset=train_ds)
    base_path = Path(mixdir)
    output_base = Path(unmixdir_model_material)
    
    # Loop over a folder with mixed spectra we want to process
    for elem in base_path.rglob('*'):
        if elem.is_file() and elem.suffix.lower() == '.nc':
            # Get relative path from base directory
            relative_path = elem.relative_to(base_path)
            # Create new filename with original directory structure
            output_path = output_base / relative_path.with_name(
                f"unmixed_by_{model_tag}_{relative_path.name}"
            )
            # Create parent directories if they don't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Loading dataset
            dataarray_elem = xr.load_dataarray(elem)
            # Ensure the DataArray has at least two dimensions for preprocessing
            if len(dataarray_elem.dims) == 1:
                # Assume the dimension is "wave_number" or similar, add a "sample" dimension
                dataarray_elem = dataarray_elem.expand_dims("sample")
            dataset_elem = preprocess_dataset(
                dataarray_elem, 
                # is_filter=True, 
                option='whitaker_hayes_with_outliers'
                )
            # Making predictions
            predictions = prediction_fn(configs, dataset_elem, state)
            # 1) Figure out how many samples and how long each array is:
            N = len(predictions)  # Number of dictionaries
            M = len(predictions[0]["wave_number"])  # Assuming all wave_number arrays have same length

            # 2) Allocate NumPy arrays for each key:
            arr_spectra = np.zeros((N, M), dtype=np.float32)
            arr_masked_spectra = np.zeros((N, M), dtype=np.float32)
            arr_mask = np.zeros((N, M), dtype=bool)
            arr_predicted_spectra = np.zeros((N, M), dtype=np.float32)
            arr_predicted_difference = np.zeros((N, M), dtype=np.float32)

            # We'll store wave_number just once, assuming it's the same for all dictionaries:
            arr_wave_number = np.zeros(M, dtype=np.float32)

            # 3) Populate these arrays by looping over the list of dictionaries:
            for i, d in enumerate(predictions):
                # Convert JAX -> NumPy if needed using jax.device_get or np.asarray
                arr_spectra[i, :] = np.asarray(jax.device_get(d["spectra"]))
                arr_masked_spectra[i, :] = np.asarray(jax.device_get(d["masked_spectra"]))
                arr_mask[i, :] = np.asarray(jax.device_get(d["mask"]))
                arr_predicted_spectra[i, :] = np.asarray(jax.device_get(d["predicted_spectra"]))
                arr_predicted_difference[i, :] = np.asarray(jax.device_get(d["predicted_difference"]))

            # 3.5) Filter the arrays before building the xarray.Dataset
            filtered_spectra = savgol_filter(arr_spectra, window_length=window_length, polyorder=polyorder, axis=1)
            filtered_predicted_difference = savgol_filter(arr_predicted_difference, window_length=window_length, polyorder=polyorder, axis=1)

            # For wave_number, we just take from the first dictionary (assuming identical for all)
            # Also un-normalizing wave_number
            arr_wave_number[:] = np.asarray(jax.device_get(predictions[0]["wave_number"])) * 800 + 2000

            # 4) Build an xarray.Dataset with dimensions ("sample", "wave_number")
            ds = xr.Dataset(
                {
                    "spectra": (("sample", "wave_number"), arr_spectra),
                    "masked_spectra": (("sample", "wave_number"), arr_masked_spectra),
                    "mask": (("sample", "wave_number"), arr_mask),
                    "predicted_spectra": (("sample", "wave_number"), arr_predicted_spectra),
                    "predicted_difference": (("sample", "wave_number"), arr_predicted_difference),
                    
                    "filtered_spectra": (("sample", "wave_number"), filtered_spectra),
                    "filtered_predicted_difference": (("sample", "wave_number"), filtered_predicted_difference),
                },
                coords={
                    "sample": np.arange(N),
                    "wave_number": arr_wave_number,
                },
            )

            # 5) Save the dataset to NetCDF
            # Saving predictions
            ds.to_netcdf(output_path, engine="netcdf4")
            print(f"Saved unmixed_by_{model_tag}_{elem.name}")
    
    print('Unmixing done.')