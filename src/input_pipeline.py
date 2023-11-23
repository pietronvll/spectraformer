import copy
from os import PathLike

import numpy as np
import pandas as pd
import torch


class RamanSpectraDataset(torch.utils.data.Dataset):
    def __init__(self, configs, train_ratio: float = 0.8, rng_seed=None):
        self.configs = configs
        raw_data = load_data(self.configs.data_path)
        # Normalizing counts
        raw_data = normalize_counts(raw_data)
        outliers, sup_norm_deviations = find_outliers(raw_data["counts"])
        self._sup_norm_deviations = sup_norm_deviations
        # Filtering outliers
        self.raman_shifts = raw_data["raman_shift"][~outliers, ...]
        self.counts = raw_data["counts"][~outliers, ...]

        # Splitting dataset
        rng = torch.Generator()
        if rng_seed is not None:
            rng.manual_seed(rng_seed)  # Setting seed for reproducibility

        val_ratio = 1 - train_ratio
        self.train_subset, self.val_subset = torch.utils.data.random_split(
            self.counts, [train_ratio, val_ratio], generator=rng
        )
        self._ds = None

    def __getitem__(self, index):
        return {k: v[index] for k, v in self._ds.items()}

    def __len__(self):
        return self._ds["counts"].shape[0]

    def get_split(self, split: str = "train"):
        split = split.lower()
        if split == "train":
            ds = self.train_subset
        elif split == "val":
            ds = self.val_subset
        else:
            raise ValueError(
                f"Unknown split {split}. Allowed choices are 'train' or 'val'."
            )
        counts = self.counts[ds.indices, ...]
        raman_shifts = self.raman_shifts[ds.indices, ...]
        masked = mask_raman_shift(
            raman_shifts,
            min_shift=self.configs.min_shift,
            max_shift=self.configs.max_shift,
        )
        masked_counts = torch.where(masked, -1, counts)
        self._ds = {
            "counts": counts,
            "masked_counts": masked_counts,
            "raman_shift": raman_shifts,
            "masked": masked,
        }
        return copy.deepcopy(self)


def normalize_counts(dataset):
    raman_shift, counts = dataset["raman_shift"], dataset["counts"]
    background_idxs = ((raman_shift[0] > 2100) & (raman_shift[0] < 2600)).nonzero()
    background = counts[:, background_idxs[:, 0]].mean(1)
    counts = counts - background[:, None]
    counts = counts / torch.max(counts, dim=1, keepdim=True)[0]
    return {"raman_shift": raman_shift, "counts": 1.5 * counts}


def mask_raman_shift(raman_shift: torch.tensor, min_shift: float, max_shift: int):
    masked = torch.logical_and(raman_shift > min_shift, raman_shift < max_shift)
    return masked


def find_outliers(normalized_counts: torch.tensor, threshold: float = 0.2):
    median_spectra = torch.median(normalized_counts, dim=0, keepdim=True).values[0]
    sup_norm_deviations = torch.tensor(
        [torch.max(torch.abs(sp - median_spectra)) for sp in normalized_counts]
    )
    outliers = sup_norm_deviations > threshold
    print(f"Found {outliers.sum()} outliers out of {outliers.shape[0]} spectra")
    return outliers, sup_norm_deviations


def load_data(path: PathLike):
    df = pd.read_csv(path, sep="\t", names=["X", "Y", "Wave", "Intensity"], comment="#")
    repetitions = df["Wave"].value_counts()[df["Wave"].iloc[0]]
    raman_shift = torch.tensor(np.asarray(np.split(df["Wave"].values, repetitions)))
    raman_shift, row_perm = torch.sort(raman_shift, 1)
    counts = torch.tensor(np.asarray(np.split(df["Intensity"].values, repetitions)))
    sorted_counts = []
    for row_idx, row in enumerate(counts):
        sorted_counts.append(row[row_perm[row_idx]])
    counts = torch.stack(sorted_counts).float()
    raman_shift = raman_shift.float()
    return {"raman_shift": raman_shift, "counts": counts}
