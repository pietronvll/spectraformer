from pathlib import Path

import orbax.checkpoint as ocp
import streamlit as st
import xarray as xa

from spectraformer.input_pipeline import preprocess_dataset
from spectraformer.models.median_baseline import median_baseline

# Load the model
ckpts = Path("checkpoints").resolve()
checkpointer = ocp.StandardCheckpointer()
variables = checkpointer.restore(ckpts / "median/100x10")

test_data = xa.load_dataarray(Path("data").resolve() / "SiC+Graphene_8x8.nc")
filtered_test_data = preprocess_dataset(test_data)


def predict(apply_fn, params, spectra, wave_number):
    import pandas as pd

    pred_SiC = apply_fn(params, spectra, wave_number)
    pred_Graphene = spectra - pred_SiC
    return pd.DataFrame(
        {
            "SiC": pred_SiC,
            "Graphene": pred_Graphene,
            "Total": spectra,
            "Wave number": wave_number,
        }
    )


st.markdown(
    """
# Spectraformer dashboard

This is an interactive dashboard to explore the SpectraFormer models.
"""
)


spectra_idxs = [i.values for i in filtered_test_data.spectra]
data_idx = st.slider("Spectra index", 1, len(spectra_idxs), value=1)


pred = predict(
    median_baseline,
    variables["params"],
    filtered_test_data.sel(spectra=spectra_idxs[data_idx - 1]),
    filtered_test_data.wave_number.values,
)

st.line_chart(
    pred, x="Wave number", y=["Total", "SiC", "Graphene"], use_container_width=True
)
