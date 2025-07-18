#### Jul 18, 2025

Trying **min64** - twin of **min63** to check God of randon's decision. 
Loss pattern repeats itself perfectly. Abort.

#### Jul 17, 2025

Based on the expected spectral features knowledge and the output of the model for buffer layer graphene, I have added more revealed regions for model to see in the region of interest. Trained **min63**. 

AND FOR SOME REASON it converged into a crosing point... While last **min62** attempt didn't cross at all. I have completely no idea why it is happening. God of random is playing with me?

#### Jul 15, 2025

90945 - this is the best step (epoch 387) for **min62**: a model trained without filtered dataset trick. Is it that I am stuck with lack of data? Or data is of bad quality and the predictor jumps around 1 STD away from the mean spectrum point? Let's enlarge the dataset with idk 15s acquisition or 20s...

#### Jul 14, 2025

Trying to remove as many garbage from git as I can to decrease the weight on the cluster.

#### Jul 07, 2025

Franklin is back! Let's integrate baseline + outlier removal used in my statistics investigations. In particular:

```python
# background subtraction
import pybaselines

def subtract_whittaker_background(da, lam=1e7):
    """Subtract Whittaker baseline from xarray DataArray using pybaselines."""
    
    # Process each spectrum
    baselines = []
    for i in range(len(da.spectra)):
        spectrum = da.isel(spectra=i).values
        baseline, _ = pybaselines.whittaker.aspls(spectrum, lam=lam)
        baselines.append(baseline)
    
    # Create baseline DataArray
    baseline_da = xr.DataArray(
        np.array(baselines).T,
        dims=da.dims,
        coords=da.coords
    )
    
    # Subtract baseline
    return da - baseline_da, baseline_da

# Outlier finding and substituting by mean value within 2*kernel_size window
import copy

def whitaker_hayes(intensity_data, kernel_size: int = 3, threshold: int = 8):
    return xr.DataArray(
        np.apply_along_axis(whitaker_hayes_spectrum, axis=-1, arr=intensity_data, kernel_size=kernel_size, threshold=threshold),
        dims=intensity_data.dims,
        coords=intensity_data.coords
        )


def whitaker_hayes_spectrum(intensity_values_array, kernel_size, threshold):
    spectrum_array = copy.deepcopy(intensity_values_array)

    spikes = whitaker_hayes_modified_z_score(spectrum_array) > threshold

    while any(spike for spike in spikes if spike):
        changes = False

        for i in range(len(spikes)):
            if spikes[i]:
                neighbours = np.arange(max(0, i - kernel_size),
                                       min(len(spectrum_array) - 1, i + 1 + kernel_size))
                fixed_value = np.median(spectrum_array[neighbours[spikes[neighbours] == 0]]) # Median or mean?

                if np.isnan(fixed_value):
                    continue

                spectrum_array[i] = fixed_value
                spikes[i] = 0
                changes = True

        if not changes:
            break

    return spectrum_array


def modified_z_score(spectrum):
    """Calculates the modified z-scores of a given spectrum."""
    mad_term = np.median([np.abs(spectrum - np.median(spectrum))])
    modified_z_scores = np.array(0.6745 * (spectrum - np.median(spectrum)) / mad_term)

    return modified_z_scores


def whitaker_hayes_modified_z_score(spectrum):
    """Calculates the Whitaker-Hayes modified z-scores of a given spectrum."""
    return np.abs(modified_z_score(np.diff(spectrum)))
    # return np.abs(modified_z_score(spectrum))
```

And try it on my local machine with **micro61**.

#### Jun 09, 2025

Image logging is working. Deleting **min60** and **micro59**. And committing **min59** to be saved. Also I guess I will commit full folder of logs since I may require loss curves of any model later.

#### Jun 05, 2025

Today 4 GPUs.

Fictional doubling of the data amount worked quite fine. The model during training has reached lower loss value and by-eye has captured closer main broad SiC features. But there are signs of over-training: with more epochs difference between train_loss and val_loss is decreasing, tending more towards crossing point. It also can be seen by eye of loosing some features.

For experiment let's overtrain **min59** to confirm loosing captured features.

And maybe let's train even smaller model? For debugging I made **micro59** model config. Let's try training it with quasi-doubled data on Franklin.

I wonder if I need to make plots per epoch and store them in tensorboard, would it be useful? Like the one on the dashboard and log_loss-wavenumber. Implementing it with **min60**.

#### Jun 04, 2025

Played around with savgol filter. Included it in the training process. Trained **min58**. It looks like filtering doesn't help to resolve overfitting issue. 
One possible solution could be acquiring more data. Can I do both unfiltered and filtered? Let's try on **min59**.


#### May 30, 2025

I guess the main reason why the loss can go up during the training (and, therefore, triggering the patience mechanism) - is the LR schedule. In particular, its "warm-up" part. Let's try to put warm-up steps to zero? No. I'm introducing configs.warmup_coeff, where its value is a multiplier of peak value. Also added back compatibility with default value 0.1. 

Let's try it with **min57** with 3 GPUs of today. And let's keep maybe 5 last checkpoints not to make a lot of them for space issues. 

Graph for LRdecay now (sketchy):
_______   -> LR
        \
         \
          \ _______   -> 0.9 * LR
                    \
                     \
                      \ _______   -> 0.9 * 0.9 * LR , etc.

Also added in config all early stop settings.

Also replaced jax.tree_map -> jax.tree.map, because of a warning.


#### May 29, 2025

Ok, the switch to arithmetic mean worked. But now there is an issue with overfitting. Implementing early stop mechanism. Deleting overfitted models **min53** and **min54**.
Also **min51** and **min52** are useless since they are overfitted in the mask-free region.
Also **min27**, **min29**, **min30**.
Also **base50** - anyway it performs not good.


Ok I did the early stopping. I have to adjust its parameters. Playing with **min56** for that.


#### May 28, 2025

After yesterday's meeting today I've implemented also arithmetic loss logic, and cleaned the code a bit from some debugging printing and unnecessary checks. Run **min53** with default min-model settings. Today Franklin gave me not 4 but 2 GPUs in interactive environment. And it seems that arithmetic average is so faster that 2 GPU Arithmetic is faster than 4 GPUs Geometric. Wow. Not only that, the model is actually capturing general features within masked region! And the loss is comparable with one of the **min23** - my previous best result (printed on a poster).


To-do:
1. Finish the training of **min53** that is 600 epochs.
2. Train another base-model with arithmetic loss for 600 epochs as well to see the difference in feature capturing and/or convergence speed.
3. Maybe ask for more Raman data? Different substrate?
4. Maybe introduce stopping mechanism of over-learning train data? Because I can see now from epoch ~140 train loss goes lower while validation loss is staying still or going slightly up. There is a cross section of train loss and validation loss.


UPD: yes, as i was afraid, the model is over-fitting. I've aborted the process. I can clearly see that its performanse becomes worse. Anyway, I consider this as a huge success for today. Still, by comparing **min23** and **min53** I see that **min23** is predicting better with lower subtraction noise. But why is that? Has it reached a different local minimum? Maybe training a different model will give me an answer.


Wait, **min23** should've had constant LR, but **min53** - LR schedule with 15 cycles (1.00e-03 --> 3.52e-05). Maybe that's the issue - I reduce LR too rapidly. Let's decrease with decline_coeff=0.9 not 0.8 ? For 15 cycles it will go (1.00e-03 --> 2.06e-04). Ok, let's see it on **min54** for 150 epochs.


Ok, the performance is kinda ok. Let's train for another 50 epochs (200 in total). Despite the fact that train loss has overcame val loss.


Let me continue let's say with another 100 epochs just to see what will be the behavior of prediction. But so, a general conclusion could be that this crossing point is a bad indicator of model finishing learning? Or the loss function should be reconsidered?! Maybe it is fine, anyway it is not adjusting itself to minimise the pure difference but more complicated loss function.

#### May 27, 2025

Trained **min51** for another 600 epochs (1800 in total) didn't provide me any significant improvement in prediction by eye. Before leaving yesterday I run **min52** in which I reduced number of cycles for LR decay from 20 to 15, and increased embedding dimension x2 (from 64 to 128). But it seems that it was trained only for 50 epochs, why is that - a mystery. I asked for 600. So, fine, let's continue its training.

#### May 26, 2025

Franklin is back, and so do I.
**min51** is trained for 1200 epochs. Let's run another 600 and see what's the difference. The model seems to diverge into some loss minima with barely recognizible negative slope. Maybe 20 cycles is too much for LR reduction? And now to fine-tune it requires much more steps?

Anyway after that I would like to train a min-configuration model but with enlarged embedding dimension. Because the experiment of rapid increase of all parameters (min -> base) showed that for the same training larger model could not tune itself enough to perform at comparable level. So, maybe just embedding dimension enlargement will show an effect sooner.

And I think I really need to figure out how to stage the job on Franklin properly via command prompt. I want to ask a better gpu-equipped node (anode) than this interactive environment. Here I have 4 x NVIDIA Tesla V100 16Gb (really Tesla V100-SXM2-32GB -- why does their own wiki wrong??) but I would like to try 8 x NVIDIA Ampere A100 80Gb.

#### May 19, 2025

**min51** is trained for 1078 epochs, but it should be for 1200. So I started another training for 122 epochs.

Also Franklin is going to be shut down for 20-24 May. Committing now.


#### May 13, 2025
I have trained **min51** for 600 epochs with multiple cosine decay. Now I've tried to change N of cycles to be not 100 but 20. With decay coefficient 0.8 my LR will be 0.011529215 times smaller (1.00E-03 => 1.15E-05), so 2 o.o.m. smaler.

This worked to start another training session for 600 epochs. Let's see. 

But in principle -- LR Schedule is better than Constant LR. But up to some point I guess, otherwise LR will be too small to significantly adjust model weights.

**Base50** is trained for 572 epochs. Performance is absolutely the worst I saw. No meaningful predictions at all.



#### May 09, 2025
**Base50** didn't finish its job - last epoch is 65. Ok, let's continue with another 600-65=535 epochs.

#### May 08, 2025
I left the training overnight. It seems that **base48** didn't finish training... There are only 131 epochs on tensorboard and folders names are not the same with **min47**. So, 300-131=169. 


I've noticed that the amount of epochs is not saved in checkpoint. Have to do it.


Also maybe the performance is bad because of region overlapping for low-f and high-f datasets? Could it be better to do something about it? Let's try separate models for separate frequencies. I'll need to rearrange datasets for SiC into SiC-low-f and SiC-high-f.

Ok the performance is veeery bad. Let's try another time with separated lowf and highf **min49** and **base50**. Deleting **min47** and **base48**.

Performance is bad but still compatible with my first attempts for geometric mean. 


*NOTE: it seems that bigger amount of cores for server is faster for calculations. So, those cores are not CPUs but GPU cores? But Volatile GPU-Util seems to be the same as before. Power consumption is bigger.*


Trained **min49** for 100 epochs. Figured out that epochs are not stored in checkpoint by having an error in streamlit. Fixing that. Deleted the model, started again 10 epochs for checking.

Trained **min49** for 600 epochs in total. Left training of **base50** overnight.


#### May 07, 2025
I have overreached 50 GB of space on franklin... By urgent cleaning I managed to start a server. Committing right now. Have to clean more models. And always keep not all but only 5 last checkpoints.

Started again **min47** training. For 300 epochs and 24 batch_size. In the meantime deleting all checkpoint steps except last 5 for every folder. After training it looks badly. Started training of **base48** again for 300 epochs 24 batch_size.

#### May 6, 2025
Looked at **min47**: after 100 epochs the unmixing performace is bad, probably will require more time to learn. But on loss graph it kinda reached saturation. Maybe LR schedule will help? Maybe change LR after N epochs instead of steps? Is it possible?


For hyperparameters comparison I've started same training but with **base48** configuration. During training it looks like "uglieness" is much more shrinked on a graph, but let's see it after 100 epochs. UPD: it's the same...


But it's independent of epochs. Somewhat around 5 epochs? Yes, it is logging every N % log_every_epoch == 0. So, for log_every_epoch=5 it is 5, 10, ..., 100 (or 0, 5, ..., 95 ?). But it looks like the cluster run out of time before finishing? Because there are 18 points clusters, and should be 20.

There is points cluster for each metrics writing of 20 points each. They are all datasets = 20. Also in-between distance is different, suggesting different dataset sizes. I don't want a total mess on loss-step curves that would be present if I change log_every_epoch parameter to 1. So, let's keep metrics writing every 5 epochs for steps and log every epoch averaged (should be num_epochs points, somehow no mess I hope).


To-do's:
1. (+) Add loss-epoch metric for train and val
2. (+) Change name of metric loss to loss-step for train and val
3. (?) LR schedule dependent on epochs? But if I run same model several times how will script know how many epochs were before?
4. Invent more accurate Data preprocessing workflow: somehow removing outliers and maybe bg removal mechanism?
5. (+) Maybe discard small datasets from the folder? To allow training to be on at least batch_size.
6. Low-f datasets: what's the mask, preprocessing workflow etc? 


#### May 5, 2025
I've really needed to log more often.
So, I have implemented computations parallelization. I ask all GPUs - I use all GPUs. And have made it to be trained for all datasets I have. And arranged datasets in data folder with some logic.
For several datasets there are a lot of spectra dropped. Because of weird bump in the middle of spectra. And several problem with low frequency datasets: there is a huge optical phonon state, and everything else seems to be miserable, especially after normalization. I have to think about that in details.


Also, for supposed batch size 24 (divisible by 1,2,3,and 4 GPUs) for some datasets there are not enough spectra to fill a single batch because of many being rejected. But on my local machine everything worked for batch size 12. So, changed it to 12 untill I'll somehow clean data / substract background / change preprocess flow.


Also I need to know what mask should I apply for low-f range data. Because we know that the model tends to directly copy-paste unmasked regions.


Last model **min47** is trained for 100 epochs with batchsize 12 constant LR.
Antonio asked also to change embedding dimension and other hyperparameters.


Also I need to think about LR schedule...


Should I go back to arithmetic loss? Because the best model so far is ***min23** trained for arithmetic loss. But there were 2 datasets, let's see the performance of **min47** to confirm or reject the influence of multiple datasets on performance (which I expect to be present A LOT).


And I need to track loss vs epoch not vs step. Looks ugly...


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