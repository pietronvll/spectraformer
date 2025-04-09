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

train_data_file = "SiC_19x10x3.nc"
test_data_file = "SiC_19x10x3.nc"

train_ds = preprocess_dataset(xr.load_dataarray(f"data/{train_data_file}"))

ckpts_path = "/home/dpoteryayev/SpectraFormer/checkpoints/"
available_models = []

for elem in epath.Path(ckpts_path).iterdir():
    tagname = str(elem).split("/")[-1]
    if tagname != "checkpoints":
        available_models.append(tagname)

datasets_path = "/home/dpoteryayev/SpectraFormer/data/mixtures/"
available_datasets = []

for elem in epath.Path(datasets_path).iterdir():
    tagname = str(elem).split("/")[-1]
    if tagname != "mixtures":
        available_datasets.append(tagname)

@st.cache_resource
def load_model(
    model_tag: str, 
    dataset_tag: str,
    desired_step: int,
    mask_start_tag: int = 1700,    
    mask_end_tag: int = 2500,
    step_choise_tag: str = 'Latest'
    ):
    
    st.write("Loading Checkpoint")
    ckpt_manager = ocp.CheckpointManager(
        ckpts_path + model_tag,
        item_handlers=ocp.StandardCheckpointHandler(),
    )
    st.write("Parsing Configuration File")
    configs = ml_confs.from_dict(ckpt_manager.metadata())
    st.write("Checkpoint Metadata:", ckpt_manager.metadata())
    test_ds = preprocess_dataset(xr.load_dataarray(f"data/mixtures/{dataset_tag}"))
    
    
    # This is an implementation of learning rate schedule - multiple cosine decay cycles from init_value to init_value*alpha, then repeating from init_value.  
    cosine_kwargs = []
    for i in range(100):    # 100 cycles - because i don't want to think much about making a cycle per N epochs. Schedule is built for steps.
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
        zip([1000, mask_end_tag], [mask_start_tag, 2900])
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
        tx=tx,
    )
    
    st.write("Restoring Weights")
    match step_choise_tag:
        case 'Latest':
            state = ckpt_manager.restore(
            ckpt_manager.latest_step(), 
            args=ocp.args.StandardRestore(state)
            )
        case 'Desired':
            state = ckpt_manager.restore(
            desired_step, 
            args=ocp.args.StandardRestore(state)
            )
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

current_dataset_tag = st.selectbox("Select dataset (from mixtures folder):", available_datasets, index=None)

available_mask_start = [1400,1500,1600,1660,1700,1800,1900,2000]
current_mask_start_tag = st.selectbox("Select left window boundary:", available_mask_start, index=None)
available_mask_end = [2000,2100,2200,2300,2400,2500,2600,2700,2800]
current_mask_end_tag = st.selectbox("Select right window boundary:", available_mask_end, index=None)

current_model_tag = st.selectbox("Select model (from a checkpoint):", available_models, index=None)


step_choise_tags = ['Latest', 'Desired']
current_step_choise_tag = st.selectbox("Choose a step to be:", step_choise_tags, index=None)

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
            mask_start_tag=current_mask_start_tag,
            mask_end_tag=current_mask_end_tag,
            step_choise_tag=current_step_choise_tag
        )
    
    # Predictions exploration on a graph
    st.write(f"Explore Predictions on `{current_dataset_tag}`")
    data_idx = st.slider("", 1, len(spectraformer_predictions), value=1)
    fig, ax = plot_results(spectraformer_predictions[data_idx])
    ax.set_title(current_model_tag)
    st.pyplot(fig)
    
    # Summary table
    dummy_example = next(batch_sampler(train_ds, mask_windows, batch_size=1))
    st.code(
        f"{tabulate_fn(dummy_example['masked_spectra'][0], dummy_example['wave_number'], dummy_example['mask'], training = False)}",
        language=None
        )
