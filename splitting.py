# This script is to check if the splitting works

import numpy as np
import xarray as xr

from spectraformer.input_pipeline import preprocess_dataset
from pathlib import Path
import ml_confs

maindir = Path(__file__).parent.resolve()

logdir = maindir / "logs"
ckptdir = maindir / "checkpoints"
# Check if logdir and ckptdir exist, if not create them
logdir.mkdir(parents=True, exist_ok=True)
ckptdir.mkdir(parents=True, exist_ok=True)

datadir = maindir / "data"

model_tag = "min"  # CHOOSE ONE (.yaml file should exist)
                   # tag also can be found for already trained models in checkpoints folder

configsdir = maindir / "configs"
configsdir.mkdir(parents=True, exist_ok=True)
config_file_name = f"configs_{model_tag}.yaml"
config_file_path = configsdir / config_file_name

if __name__ == "__main__":
    
    configs = ml_confs.from_file(config_file_path)
    configs.tabulate()

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

    print(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
