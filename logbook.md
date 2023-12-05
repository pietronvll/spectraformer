#### Dec 5, 2023

**Phase 1 - specifications.**

_Problem:_ Given a mixed RAMAN spectra of a Graphene - SiC (silicon carbide) sample, perform spectral unmixing.

_Data preprocessing:_ A single data point is a SiC RAMAN spectra.
1. Shift a datapoint by removing the background, identified as the average counts in the spectral region between $2200$ and $2500 \, {\rm cm}^{-1}$.
2. Normalize a datapoint by rescaling everything by the maximum value.
3. Filter spectra containing cosmic rays peaks:
   1. Get the _median_ spectrum
   2. Compute the $\sup$-norm between a data point and the median spectrum
   3. Filter out anything with $\sup$-norm above 0.2 (that is, filter out spectra which have maxima 20% or more higher then the median maxima.)

_Training procedure:_ 
1. Mask each datapoint in the window between $1525$ and $1650 \, {\rm cm}^{-1}$ (for the moment).
2. Compute frequency _and_ counts embeddings and sum them $x \gets e_{f} + e_{c}$
3. Feed $x$ to a transformer and train it to recover the _unmasked_ spectra with the MSE error.

Open questions:
- How to enforce the physical requirement of _positive_ spectrum counts: that is, how to enforce that $y_{\rm mix} - y_{SiC} \geq 0$ ? 

_Testing & Validation:_ 

1. On the SiC just monitor the MSE error.
2. On the mixed spectra monitor the correlation between the $2D$ and $G$ peaks (how?)
3. On the mixed spectra monitor additional independent measures such as Field-emission microscopy (TBD)

**Phase 2 - Implementation.**

Use `JAX+Flax` to perform the implementation. 

**Phase 3 - Deployment.**

TBD as of today.


#### Nov 10, 2023
1. Moved the folder to a git repository.  
2. Normalization of the data is a non-trivial issue when transferring to the mixed data (substrate + graphene). For the moment averaging the signal at $[1800, 1900]$ and dividing by that. (To be discussed)
3. I should just reconstruct what is inside the mask and not outside? (To be discussed)


#### Nov 7, 2023
Components of the code:
1. Filter cosmic rays spikes
2. Exposition-independent data normalization
3. Data splitting and loading
4. Model definition
    1. Loss function
    2. Architecture `embedding -> transformer`
    3. Loss
5. Training loop
6. Testing utils
7. Serialization and inference