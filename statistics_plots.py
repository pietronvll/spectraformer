import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
from etils import epath

maindir = Path('/home/dpoteryayev/SpectraFormer/notebook.ipynb').parent.resolve()

datadir = maindir / "data"

mixdir = datadir / "mixtures"
mixdir.mkdir(parents=True, exist_ok=True)

unmixdir = mixdir / "unmixed"
unmixdir.mkdir(parents=True, exist_ok=True)

def my_statistics(
    unmixed_ds,
    dim: str = 'sample'
    ):
    unmixed_ds['wave_number'] = unmixed_ds['wave_number'] * 800 + 2000
    unmixed_ds = unmixed_ds.drop_vars(['mask','masked_spectra','predicted_spectra', 'spectra']).to_dataarray(dim='predicted_difference').drop_vars(names='predicted_difference')
    exp_val = unmixed_ds.mean(dim=[dim])
    variance = unmixed_ds.var(dim=[dim])
    std = unmixed_ds.std(dim=[dim])
    
    z1 = (unmixed_ds - exp_val)/std
    
    skewness = (z1**3).mean(dim=[dim])
    kurtosis = (z1**4).mean(dim=[dim])
    
    return exp_val, variance, std, skewness, kurtosis

def plot_statistics(
    dataset
):
    return 0