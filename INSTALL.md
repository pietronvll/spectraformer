# Installation instructions
Spectraformer has been developed and tested on an environment with `python 3.11`.

## Prerequisites:
1. A working installation of Anaconda or Mamba

## Step 0: Initialize Conda Environment.
Create a new environment with Python 3.11

```bash
conda create --name spectraformer python==3.11
```

Activate the environment
```bash
conda activate spectraformer
```

## Step 1: Install JAX

The first step to run Spectraformer is to install JAX.

### CPU-Only
```bash
pip install --upgrade "jax[cpu]==0.4.21"
```

### GPU
Quoting from [JAX's installation guide](https://jax.readthedocs.io/en/latest/installation.html)
> JAX supports NVIDIA GPUs that have SM version 5.2 (Maxwell) or newer. Note that Kepler-series GPUs are no longer supported by JAX since NVIDIA has dropped support for Kepler GPUs in its software.
> 
> You must first install the NVIDIA driver. We recommend installing the newest driver available from NVIDIA, but the driver must be version >= 525.60.13 for CUDA 12 and >= 450.80.02 for CUDA 11 on Linux. If you need to use a newer CUDA toolkit with an older driver, for example on a cluster where you cannot update the NVIDIA driver easily, you may be able to use the CUDA forward compatibility packages that NVIDIA provides for this purpose.

Once drivers are installed run 

```bash
pip install --upgrade "jax[cuda12_pip]==0.4.21" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
## Step 2: Install the additional requirements
```bash
pip install -r requirements.txt
```