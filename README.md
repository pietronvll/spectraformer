SpectraFormer project is a transformer-based Machine Learning project aimed for Raman spectra unmixing for graphene buffer layer on SiC substrate.

# Tips

## Tensorboard

To use tensorboard, type in terminal 


```console
tensorboard --logdir=logs --samples_per_plugin images=1000
```

## Streamlit dashboard app

To use dashboard, type in terminal 

```console
streamlit run dashboard.py
```

## GPU usage in terminal

Useful command to watch gpu load in real time: 

```console
watch -n 1 nvidia-smi
```

where 1 is update time in seconds.
