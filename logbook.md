#### Feb 11, 2025
Successfully implemented corrected Poissonian loss function.

Poisson loss have this formula:

$$
L_{Poisson}(true, pred)=pred-true*log(pred)
$$

From this we know that prediction can not be negative, ant therefore true data.

Moreower, for every true data point value there is a separate loss function with its own minimum. In contrast, for MSE loss regardless of data - loss minima will be zero.

I wanted to implement that also for Poisson.

For this, I've found a function of each minimum at a given true value:

$$
min = true - true*log(true)
$$

Substracting one from the other gives the corrected formula:

$$
L_{CorrPoiss}=(pred-true) + true*log(true/pred).
$$

But it is not solving the issue of log calculation. After implementing that I've faced an old problem of NaN issue. To solve this, I've looked at true data and figured out - there is no NaN or PosInf or NegInf. At least after proper preprocessing. It appears that my NaN/Inf catching itself is wrongly executed: it seems that lax.cond makes both actions regardless of the condition. Whatever.

And I am more than sure that the model being initialised from scratch is not predicting NaN/Inf values, which I've also checked.

But still the question how to preprocess data is present.

Raw data doesn't show Poissonian statistics. Mean is proportional to the STD and not variance. While after normalization it resolves Poissonian statistics, but still in not a pure way. It should be $$var = \mu$$, but i found $$var = (8.36)e-4 * \mu + 1e-5$$. So, first - there is an offset, second - there is a coefficient.

Relation between mean and STD:
$$
\sigma = 0.02*\mu+0.009
$$

What is the statistical distribution that could have such property?

Anyway, for using every Poissonian or Gamma or whatever function containing logarithm - we need to be sure of correct calculations. For this reason, I've shifted all data by arbitrary number 0.1 after normalization. For now, data is in range $$[ 0.1 ; 1.1 ]$$. It is to be sure there will be no NegInf after log operation. I think every range inside $$(0,e]$$ is ok since old Poisson loss function had non-negative minima in this range. I will check it by comparing the performance of the models taught on different offsets. But I think they will be the same.

#### Feb 05, 2025
Trying to implement Gamma loss. Without any changes in input pipeline it is negative both train and val.
With changes of a pipeline 

```python
    # Negative value removal
    dataset = dataset - dataset.min(dim="wave_number")
    # Normalization to the max - squeezing into range [ 0 ; 1 ]
    dataset = dataset / dataset.max(dim="wave_number")
    # Shifting from zero = range [ 1e-8 ; 1 + 1e-8 ]
    dataset = dataset + 1e-8
```

it is still negative. I have a hint - because of $$
log(0<x<1)<<0
$$.
I've tried to change sign in the formula, but it is still negative - but only the train (which was the one i changed...).


#### Jan 23, 2025
I've tried to implement data shifting from negative values to positive half-plane by the following code in input_pipeline.py in preprocess_dataset function:
```python
    # Shifting dataset to the positive half-plane
    dataset = dataset - (1 + epsilon) * dataset.min(dim="wave_number")
```
This shift worked well to remove NaN issue since the log function now is well defined but the loss itself stucks at certain value no matter the epsilon (I've tried in range from 1e-1 to 1e-10).

After the discussion with Antonio he suggested to shift the data after normalization from (0,1) to (1,2). 
This is to set the logarithm to be positive or very slightly negative ( -0.3+1=0.7 -> log(0.7)=-0.15... ).
Let's see how does it affects the loss function.
UPD: it didn't work (adding +1 to the dataset).

The only alternative ways I see are using MSE instead of Poisson or Fenchel-Young loss (which I have no idea how it works).
Let's try to train with MSE.

#### Jan 21, 2025
Added gradient metrics to be written during training. Let's see what happens with a usual conditions without trics to make positive values before log calculation (min6).

Also added some checks for NaN/Inf values.
While training as usual without tricks with "positivization" it appears that as expected before first epoch prediction is NaN/Inf, gradients are NaN/Inf (since they are just initialized). While for all training it is basically ok. 
After reaching the condition for meeting NaN in training process the program shuts down as intended.
But if I try to re-run the train_script it gives me the sequence of:
This behavior occures when there is no decorator.
```python
    No checkpoint found with tag spectraformer:min8, training from scratch.
    NaN detected in pred_spectra for training step
    Inf detected in pred_spectra for training step
    NaN detected in grads
    Inf detected in grads
    NaN detected in pred_spectra for training step
    Inf detected in pred_spectra for training step
    NaN detected in grads
    Inf detected in grads
    ...
```
So, like it is not exiting the training step? And it is not logging anything else as intended like #epoch, loss etc. 

NaN input -> log(NaN)=NaN, grad=NaN -> ...

With decorator:
```python
    No checkpoint found with tag spectraformer:min8, training from scratch.
    NaN detected in pred_spectra for training step
    Inf detected in pred_spectra for training step
    NaN detected in grads
    Inf detected in grads
    Epoch 1 -- Loss 5.690e-01
    NaN detected in pred_spectra for validation step
    Inf detected in pred_spectra for validation step
    Validation -- Epoch 1 -- Loss 2.814e-01 -- Cos_sim 3.477e-01 -- MSE 2.408e-02
    Epoch 2 -- Loss 2.488e-01
    Validation -- Epoch 2 -- Loss 2.548e-01 -- Cos_sim 3.477e-01 -- MSE 1.898e-02
    ...
```

Maybe the solution could be to **CHANGE LOSS FUNCTION**?
Could be: 
1. log( abs(predicted_spectra) ) 
2. or if we want to keep the sign: log( abs(predicted_spectra) )*sign(predicted_spectra)?
3. Or Fenchel Young loss? https://arxiv.org/pdf/1901.02324
4. optax.losses.softmax_cross_entropy

#### Jan 20, 2025
Working on validation metrics and NaN issue.

NaN issue is due to the decorator "@jax.jit". Without it the model trains as usual but much slower. Probably it's because of log function calculation. Trying to solve the issue to still use the decorator optimization.

Metrics so far:
1. Poisson loss - it is convenient to use it since our data is Poisson-noised.
2. Mean square error.
3. Cosine similarity.

For later: what is early stop and why does it commented...

Update. Successfully implemented validation metrics. But the model at step 6630 got stuck in train loss and val loss
Also fixed an issue of NaN appearing. Indeed, it is solved after I put a positive threshold before log calculation.

Now problem is that the loss saturates (min3 and min4). Let's try to increase lr by 1 order of magnitude.

Also cosine similarity seems to be useless metric since it is constant. At least for the same model configuration.

Ok maybe this "stucking" is not due to lr. But rather due to my clipping of values before log function.

#### Jan 14, 2025
Code for dataset splitting is integrated into train script. Now I need to validate training.

Think about merging all data I have into one large file?

#### Jan 10, 2025
I need to separate data into train and validation datasets to visualize overfitting. Just train/loss is not enough to understand the model behaviour.

#### Jan 9, 2025
Last time I've added new raw data and tried to train some model.
Now config file doesn't require to be edited for playing with different models. Every model config file is stored in "configs" folder.

Question: how to choose model parameters.

Think about:
1. Evaluating models using TensorBoard (re-train models with modified code to write metrics?). NOTE: to use tensorboard, type in terminal "tensorboard --logdir=logs"
2. Show the amount of parameters on the dashboard maybe? (tabulate)

#### Mar 5, 2024
Back to the project. Last time I have solved the technical issues on trainin on Franklin. Now, focus on code: 
1. Batchsize finder and HP opt.
2. Early Stopping.

**Architecture**
1. Find a way to make it scale invariant
2. Perform denoising.

#### Feb 27, 2024
With a MVP in place, move to the training of more refined models.

**General.**
1. Read the paper [Contribution of the buffer layer to the Raman
spectrum of epitaxial graphene on SiC(0001)](https://iopscience.iop.org/article/10.1088/1367-2630/15/4/043031/pdf)
3. Solve the issue with logging on Franklin.
4. Containerize the package for fast deployment.

**Training & Code**
1. Batch size finder.
2. Early stopping - [Flax Docs](https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#early-stopping).
3. Add random mask upon training.
4. Streamlit dashboard.

**Next generation of models should be**
1. Scale independent, or learnable scale params.
2. Should perform denoising



#### Feb 21, 2024
Worked quite a bit on the spectraformer model. Having in place a MVP. Some obvious things to do:
1. Ad an asymmetric attention matrix masking any _inputs_ from the masked region (but not outputs)
2. Consolidating the code.

#### Feb 16, 2024
As working on this project is fragmented, I need to set up a solid schedule. 
1. Data input-pipeline & trivial benchmark.

#### Feb 9, 2024
1. Working on GPU is _much_ faster.
2. I am trying to make it work in a rush. Let's stop, I need loads of time, and a model deployment plan.
3. Mostly, I also need solid testing and benchmarking utilities.


#### Feb 8, 2024
Back to the project. 

1. Working Locally on Mac is impossibly slow.
2. The basic training pipeline is up and running.
3. I need to implement a _testing_ pipeline, with a proper data splitting. 
4. I need to check the data scaling. For now I have added a `layer norm` at the very beginning of the network, but don't think it is sufficient. 
5. Need to implement cosmetics: logging and checkpointing.


#### Dec 18, 2023

Updates & next steps:
1. Architecture almost completely defined. Last thing to check is the position of Layer Norms ~~and Dropouts~~.
2. Script to measure throughput and choose the batch size. Work locally first, and then use Franklin's `A100`.
3. Script to select the `lr`, check if using `AdamW` instead of `Adam`
4. Transformer initialization?
5. Think about the proper loss function for the kind of noise at hand (poissonian), and the best strategy for masked learning.

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