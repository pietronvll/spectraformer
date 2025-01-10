import jax

jax.config.update("jax_platform_name", "cpu")
import flax.linen as nn
import ml_confs
import optax
import orbax.checkpoint as ocp
import streamlit as st
import xarray as xr
from etils import epath
from flax.training.train_state import TrainState

from spectraformer.inference import plot_results, predict
from spectraformer.input_pipeline import batch_sampler, preprocess_dataset
from spectraformer.model import SpectraFormer

train_data_file = "SiC_19x10x3.nc"
test_data_file = "SiC+Graphene_50x20.nc"

train_ds = preprocess_dataset(xr.load_dataarray(f"data/{train_data_file}"))
test_ds = preprocess_dataset(xr.load_dataarray(f"data/{test_data_file}"))

ckpts_path = "/home/dpoteryayev/SpectraFormer/checkpoints/"
available_models = []

for elem in epath.Path(ckpts_path).iterdir():
    tagname = str(elem).split("/")[-1]
    if tagname != "checkpoints":
        available_models.append(tagname)


@st.cache_resource
def load_model(model_tag: str):
    st.write("Loading Checkpoint")
    ckpt_manager = ocp.CheckpointManager(
        ckpts_path + model_tag,
        item_handlers=ocp.StandardCheckpointHandler(),
    )
    st.write("Parsing Configuration File")
    configs = ml_confs.from_dict(ckpt_manager.metadata())
    mask_windows = list(
        zip(configs.masked_interval_starts, configs.masked_interval_ends)
    )
    dummy_example = next(batch_sampler(train_ds, mask_windows, batch_size=1))
    # Re-initialize model based on loaded configs
    st.write("Initializing Model")
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
    st.write("Restoring Weights")
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


st.title("Spectraformer dashboard")
current_model_tag = st.selectbox("Select model:", available_models, index=None)

if current_model_tag is not None:
    with st.status(f"Loading {current_model_tag}"):
        state, spectraformer_predictions = load_model(current_model_tag)
    st.write(f"Explore Predictions on `{test_data_file}`")
    data_idx = st.slider("", 1, len(spectraformer_predictions), value=1)

    fig, ax = plot_results(spectraformer_predictions[data_idx])
    st.pyplot(fig)
