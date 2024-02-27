from pathlib import Path

import jax
import ml_confs
import optax
import orbax.checkpoint as ocp
import xarray as xr
from absl import logging
from flax.training.train_state import TrainState
from tensorboardX import SummaryWriter

from spectraformer.input_pipeline import batch_sampler, preprocess_dataset
from spectraformer.model import SpectraFormer
from spectraformer.train import train_step

logging.set_verbosity(logging.INFO)

maindir = Path(__file__).parent.resolve()
logdir = "gs://spectraformer/logs/"
ckptdir = "gs://spectraformer/checkpoints/"
datadir = maindir / "data"

if __name__ == "__main__":
    configs = ml_confs.from_file(maindir / "configs.yaml")
    configs.tabulate()

    # Data Loading
    train_ds = preprocess_dataset(
        xr.load_dataarray(datadir / f"{configs.train_dataset}.nc")
    )

    dummy_example = next(batch_sampler(train_ds, batch_size=1))
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
    ckpt_manager = ocp.CheckpointManager(
        ckptdir + configs.tag,
        options=ckpt_options,
        item_handlers=ocp.StandardCheckpointHandler(),
        metadata=configs.to_dict(),
    )

    if len(ckpt_manager.all_steps()) > 0:
        state = ckpt_manager.restore(
            ckpt_manager.latest_step(), args=ocp.args.StandardRestore(state)
        )
        print(f"Resuming training from step {state.step}.")
    else:
        print(f"No checkpoint found with tag {configs.tag}, training from scracth.")

    writer = SummaryWriter(logdir + configs.tag)

    for epoch in range(configs.num_epochs):
        data_loader = batch_sampler(
            train_ds, batch_size=configs.batch_size, rng_seed=epoch, shuffle=True
        )
        for batch in data_loader:
            state, loss = train_step(state, batch, dropout_key)
        if epoch % configs.log_every_epochs == 0:
            writer.add_scalar("train/loss", loss.item(), state.step)
            ckpt_manager.save(state.step, state)
    ckpt_manager.wait_until_finished()
