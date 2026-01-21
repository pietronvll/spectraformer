import jax
import jax.numpy as jnp
jax.config.update("jax_platform_name", "cpu")
import flax.linen as nn
import ml_confs
import optax
import orbax.checkpoint as ocp
import streamlit as st
import xarray as xr
from pathlib import Path
from flax.training.train_state import TrainState

from spectraformer.inference import plot_results, predict
from spectraformer.input_pipeline import batch_sampler, preprocess_dataset
from spectraformer.model import SpectraFormer, CustomTrainState

# Inject custom CSS - for the summary table, mainly for font adjustment
st.markdown(
    """
    <style>
    /* Make code text smaller */
    div[data-testid="stCode"] pre code {
        font-size: 9px;  /* or 10px, etc. */
        line-height: 1.2; /* optional, tighten spacing */
    }
    /* Also make the container wide and scrollable if needed */
    div[data-testid="stCode"] {
        width: 100% !important;
        max-width: 100% !important;
        overflow-x: auto !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

maindir = Path(__file__).parent.resolve()
configsdir = maindir / "configs"

train_data_file = "SiC_19x10x3.nc"
test_data_file = "SiC_19x10x3.nc"

train_ds = preprocess_dataset(xr.load_dataarray(f"data/{train_data_file}"))

ckpts_path = maindir / "saved_models/checkpoints/"
available_models = []

for elem in Path(ckpts_path).iterdir():
    tagname = str(elem).split("/")[-1]
    if tagname != "checkpoints":
        # Remove 'spectraformer:' prefix if present
        tagname = tagname.replace("spectraformer:", "")
        available_models.append(tagname)

datasets_path = maindir / "data/mixtures/"
available_datasets = []

for elem in Path(datasets_path).iterdir():
    tagname = str(elem).split("/")[-1]
    if tagname != "mixtures":
        available_datasets.append(tagname)

@st.cache_resource
def load_model(
    model_tag: str, 
    dataset_tag: str,
    desired_step: int,
    step_choise_tag: str = 'Latest'
    ):
    
    st.write("Loading Checkpoint")
    # Add 'spectraformer:' prefix for checkpoint path if not already present
    checkpoint_tag = model_tag if model_tag.startswith("spectraformer:") else f"spectraformer:{model_tag}"
    ckpt_manager = ocp.CheckpointManager(
        ckpts_path / checkpoint_tag,
        item_handlers=ocp.StandardCheckpointHandler(),
    )
    st.write("Parsing Configuration File")
    
    config_file_name = f"configs_{model_tag}.yaml"
    config_file_path = configsdir / config_file_name
    configs = ml_confs.from_file(config_file_path)
    
    test_ds = preprocess_dataset(xr.load_dataarray(f"data/mixtures/{dataset_tag}"), option='whitaker_hayes')
    
    
    # This is an implementation of learning rate schedule - multiple cosine decay cycles from init_value to init_value*alpha, then repeating from init_value.  
    cosine_kwargs = []
    for i in range(100):    # 100 cycles - arbitrary large number to ensure enough cycles
        cycle_dict = {
            "init_value": 0.1*configs.learning_rate, 
            "peak_value": configs.learning_rate, 
            "warmup_steps": 1000 if not hasattr(configs, 'warmup_steps') else configs.warmup_steps,
            "decay_steps": 2000 if not hasattr(configs, 'decay_steps') else configs.decay_steps,            
            "end_value": 0.1*configs.learning_rate
        }
        cosine_kwargs.append(cycle_dict)
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

    state = CustomTrainState.create(
        apply_fn=jax.jit(model.apply, static_argnames=("training",)),
        params=variables["params"],
        tx=tx,
        epoch=jnp.array(0, dtype=jnp.int32),
    )
    
    st.write("Restoring Weights")
    match step_choise_tag:
        case 'Latest':
            try:
                restored = ckpt_manager.restore(
                    ckpt_manager.latest_step(),
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
        case 'Desired':
            restored = ckpt_manager.restore(
            desired_step,
            args=ocp.args.StandardRestore({"state": state})
            )
            state = restored["state"]
        case _:
            st.write("SOMETHING WENT WRONG")
    
    # Loading Databases
    test_data = list(batch_sampler(test_ds, mask_windows, shuffle=False, batch_size=1))
    # Predict
    spectraformer_predictions = [
        predict(
            state.apply_fn,
            {"params": state.params},
            datapoint,
            datapoint["mask"],
        )
        for datapoint in test_data
    ]
    # Summary table
    tabulate_fn = nn.tabulate(model,jax.random.key(0),depth = 1,console_kwargs={"force_terminal": False,"color_system": None})
    return state, spectraformer_predictions, tabulate_fn, mask_windows


st.title("Spectraformer dashboard")

current_dataset_tag = st.selectbox("Select dataset (from mixtures folder):", available_datasets, index=2)

current_model_tag = st.selectbox("Select model (from a checkpoint):", available_models, index=None)

step_choise_tags = ['Latest', 'Desired']
current_step_choise_tag = st.selectbox("Choose a step to be:", step_choise_tags, index=0)

if current_step_choise_tag == 'Desired':
    current_desired_step = st.number_input("Insert a step number", value=17880)
    st.write("The chosen number is ", current_desired_step)
else:
    current_desired_step = 0


if current_model_tag is not None:
    # Model loading
    with st.status(f"Loading {current_model_tag}"):
        state, spectraformer_predictions, tabulate_fn, mask_windows = load_model(
            model_tag=current_model_tag,
            dataset_tag=current_dataset_tag,
            desired_step=current_desired_step,
            step_choise_tag=current_step_choise_tag
        )
    
    # Predictions exploration on a graph
    st.write(f"Explore Predictions on `{current_dataset_tag}`")
    data_idx = st.slider("", 1, len(spectraformer_predictions), value=1)
    fig, ax = plot_results(spectraformer_predictions[data_idx])
    if hasattr(state, 'epoch'):
        ax.set_title(f'{current_model_tag}\nStep {state.step} -- Epoch {state.epoch}\nDataset {current_dataset_tag}', fontsize='x-large')
    else: ax.set_title(f'{current_model_tag}\nStep {state.step} -- Epoch [NOT RECORDED]\nDataset {current_dataset_tag}', fontsize='x-large')
    
    ax.set_xlabel("Raman shift, $cm^{-1}$", fontsize='x-large')
    ax.set_ylabel("Intensity, a.u.", fontsize='x-large')
    ax.tick_params(axis='both', which='major', labelsize='x-large')
    st.pyplot(fig)
    
    # Summary table
    dummy_example = next(batch_sampler(train_ds, mask_windows, batch_size=1))
    st.code(
        f"{tabulate_fn(dummy_example['masked_spectra'][0], dummy_example['wave_number'], dummy_example['mask'], training = False)}",
        language=None
        )
