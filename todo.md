# Worklog

- [x] Remove useless stuff out of dashboard
- [x] Use interactive charts in dashboard
- [x] Retrain model & host it on HF
- [x] Docs with vitepress
- [x] Replace required `--output` argument in spectraformer-unmix cli with a reasonable default
- [ ] Search the data for dead code, and prune it
- [ ] In the dashboard, the inputs should be simple .csv files, to be parsed with code currently in the `data_parser_script.py` (and maybe `spectraformer.input_pipeline`). The default dataset should be called `Example data`, and should consist of _a single_ Raman Specturm, not in a whole datasets of it.

---

## Cleanup List

### 1. Dead Code - Functions Never Called

| File | Function | Lines | Notes |
| ---- | -------- | ----- | ----- |
| `spectraformer/input_pipeline.py` | `preprocess_dataset_raw()` | 270-314 | Legacy function, replaced by `preprocess_dataset()` |
| `spectraformer/inference.py` | `plot_results()` | 21-47 | Only `plot_results_train()` is used |
| `spectraformer/inference.py` | `plot_dataset_pairs()` | 104-150 | Never called anywhere |

**Unused loss functions in `spectraformer/train.py`:**

- [ ] `poisson_loss_fn()` (line 66)
- [ ] `gamma_loss_fn()` (line 86)
- [ ] `mse_loss_fn()` (line 106)
- [ ] `corrected_poisson_loss_fn()` (line 126)
- [ ] `val_poisson_loss_fn()`, `val_gamma_loss_fn()`, `val_mse_loss_fn()`, `val_corrected_poisson_loss_fn()` (lines 260-312)

Only `corrected_gamma_loss_fn` and `val_corrected_gamma_fn` are actually used.

### 2. Duplicate Code Patterns

**A. `my_geometric_mean()` defined 4 times identically in `train.py`:**

- Lines 176, 286, 631, 724 - Same 5-line function repeated

**B. NaN/Inf checking pattern repeated ~8 times in `train.py`:**

```python
nan_check = jnp.any(jnp.isnan(pred_spectra))
inf_check = jnp.any(jnp.isinf(pred_spectra))
lax.cond(nan_check, lambda _: jax.debug.print(...), ...)
```

**C. Functions defined twice in `manuscript_figures/fig4/utils.py`:**

- [ ] `build_base_xy()` - lines 650 AND 1211 (identical)
- [ ] `tile_xy()` - lines 677 AND 1238 (identical)
- [ ] `read_pdb_xyz_cell()` - lines 692 AND 767 (identical, 75 lines each)

~150 lines of pure duplicate code in fig4/utils.py alone.

### 3. Commented-Out Code (Can Be Removed)

| File | Lines | Description |
| ---- | ----- | ----------- |
| `spectraformer/input_pipeline.py` | 100-117 | Commented background removal |
| `spectraformer/input_pipeline.py` | 159-177 | Commented Whittaker baseline |
| `spectraformer/input_pipeline.py` | 191-213 | Commented preprocessing cases |
| `spectraformer/train.py` | 315-318 | Commented validation losses |
| `spectraformer/train.py` | 323-326 | Commented validation logging |
| `spectraformer/train.py` | 405-409 | Commented epoch logging |
| `manuscript_figures/fig4/utils.py` | 153-217 | Entire commented function |

### 4. Self-Contained Cleanup Actions

**Quick wins (safe to delete):**

- [ ] Delete `preprocess_dataset_raw()` from input_pipeline.py (lines 270-314)
- [ ] Delete `plot_results()` and `plot_dataset_pairs()` from inference.py (lines 21-47, 104-150)
- [ ] Delete unused loss functions from train.py (lines 66-126, 260-282, 295-312)
- [ ] Remove commented code blocks listed in section 3
- [ ] Deduplicate fig4/utils.py - remove second definitions (lines 1211-1312)

**Minor refactors (optional):**

- [ ] Extract `my_geometric_mean()` once at module level in train.py, remove the 3 duplicates
- [ ] Create a shared `check_nan_inf()` helper to replace the 8 repeated patterns
