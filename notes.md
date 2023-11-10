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