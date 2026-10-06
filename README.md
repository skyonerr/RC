# Bearing Remaining Useful Life Prediction

Bearing remaining useful life (RUL) prediction using a temperature-modulated reservoir and an MLP readout, with support for cross-bearing training, repeated-run statistics, and separate prediction plots.

## Dataset

This project uses the **FEMTO-ST PRONOSTIA (IEEE PHM 2012)** accelerated bearing degradation dataset, which contains vibration and temperature recordings. See the [NASA FEMTO Bearing dataset listing](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) for the data source and the [PRONOSTIA paper by Nectoux et al.](https://publiweb.femto-st.fr/tntnet/entries/1528/documents/author/data) for a description of the experimental platform.

Seven processed datasets from operating condition 1 (1800 rpm, 4000 N) and condition 2 (1650 rpm, 4200 N) are provided in [`data/processed/`](data/processed/). Each task uses two bearings for training and a different bearing for testing, all under the same operating condition. The project uses the following custom splits:

| Test bearing | Training bearings |
| --- | --- |
| Bearing1_1 | Bearing1_4, Bearing1_6 |
| Bearing1_4 | Bearing1_1, Bearing1_6 |
| Bearing2_1 | Bearing2_5, Bearing2_6 |
| Bearing2_4 | Bearing2_1, Bearing2_5 |

## Repository Structure

```text
models/     Reservoir and MLP models
src/        Data loading, training, evaluation, and plotting
scripts/    Result summaries and prediction plot export
configs/    Experiment configurations
data/       Processed data and checksum manifest
results/    Reference results and new experiment outputs
tests/      Data and model tests
run.py      Experiment entry point
```

## Installation and Usage

Use Python 3.10 and run the following commands from the repository root (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run.py --verify
python -m unittest discover -s tests -v
python run.py --smoke
python run.py
```

`--smoke` runs only two epochs to check the pipeline. The default experiment covers four target bearings, three model configurations, and seeds `[0, 1, 2, 3]`, with 30,000 epochs per run and 48 runs in total. Outputs are saved in a separate directory under `results/runs/`.

To run a single combination:

```powershell
python run.py --targets Bearing1_1 --arms multimodal --seeds 0
```

See `python run.py --help` and [`configs/experiment.json`](configs/experiment.json) for all options. For CUDA 12.8, install the dependencies listed in [`requirements-cuda.txt`](requirements-cuda.txt). Conda users can run `conda env create -f environment.yml`, followed by `conda activate bearing-rul`.

## Models

| Configuration | Description |
| --- | --- |
| `multimodal` | Vibration input; environmental temperature modulates the reservoir leak rate and state bias |
| `vibration_only` | The same vibration input; leak rate fixed at 0.1, with temperature modulation disabled |
| `temperature_only` | Temperature as the only input; leak rate fixed at 0.1, with temperature modulation disabled |

In `multimodal`, temperature acts as an environmental variable and is not concatenated with the vibration features. Model definitions are provided in [`reservoir.py`](models/reservoir.py) and [`MLP.py`](models/MLP.py). Input construction and evaluation metrics are described below.

## Experimental Settings and Metrics

- **Data fields:** The `train` dictionary in each PKL contains `X_vib` (N × 21), `T_K` (K), and `y_10s` (labels currently on an approximately 0–100 scale; the field name does not imply seconds). `train` is only a dictionary key; the configuration determines the actual training and test roles. File sources and SHA256 checksums are listed in [`data/manifest.json`](data/manifest.json). Reproduction starts from these PKL files; reconstructing all files from raw signals has not been fully verified.
- **Input construction:** Both vibration-based models append the coordinate `u_i = i/(N-1)`, where N is the complete record length of each bearing, giving 22 input dimensions. This coordinate and some upstream standardization steps use the complete record. The temperature-input baseline has a one-dimensional input without this auxiliary coordinate, so auxiliary inputs are not fully matched across the three configurations.
- **Standardization and windowing:** Shared vibration statistics are computed by pooling the first 50% of each training source, using at least 100 points per source or all available points if fewer than 100 are available. The window length is 20, and labels are taken at the window endpoint. Equal numbers of windows are sampled uniformly across each of the two training sequences, and training uses unweighted MSE. The reservoir mapping is shared within each run. Each window starts from a zero state, and the MLP reads the final state.
- **Temperature processing:** Temperatures are clipped to 300–450 K. The temperature-input baseline is standardized using the mean and population standard deviation of the complete training temperature sequences. The temperature-modulated model uses the window-end temperature to compute the states within that window, with a reference temperature of 375.15 K and an activation energy parameter of 0.7 eV. The state bias is not fed back into the recurrence.
- **Training and partitioning:** The first MLP layer uses bounded positive conductance weights. Training uses a hidden width of 128, Adam with a learning rate of 0.001 and weight decay of 0.00001, and full batches. Let L be the maximum label after balancing the training samples. The model fits `L-y`, and RUL is recovered as `clip(L-prediction, 0, L)`. Each task is trained independently, without random window-level splitting or a validation set. The training/validation/test proportions by bearing count are 66.67%/0%/33.33%. Bearings may be reused across different tasks, so these tasks do not form a single independent test set for one model. Configurations and the seed range were selected during prior exploration, in which test performance informed selection; the current experiments reproduce those fixed configurations.

| Metric field | Definition |
| --- | --- |
| `rmse_raw` | `sqrt(mean((y_pred - y_true)**2))` |
| `rmse_normalized` | `rmse_raw / max(y_true)` |
| `nrmse_range` | `rmse_raw / (max(y_true) - min(y_true))` |

MAE and R² are also reported. Normalization denominators use the ground-truth labels of the evaluated windows and are not assumed to equal 100. The label scale is not directly converted to seconds. The mean and sample standard deviation (`ddof=1`) across the four seeds are computed for each bearing–model combination, describing run-to-run variation under a fixed split.

## Results and Visualization

- [`results/current/per_seed.csv`](results/current/per_seed.csv): 48 reference metric records, with summaries in [`statistics.csv`](results/current/statistics.csv). Individual prediction files are available for 24 records; the remaining records contain metrics only. See [`prediction_availability.csv`](results/current/prediction_availability.csv) for availability.
- [`results/current/figures/`](results/current/figures/): Separate seed 0 prediction plots for four bearings and three model configurations, totaling 12 figures.
- [`results/validation/seed0/`](results/validation/seed0/): 12 archived seed 0 runs, including configurations, environments, weights, and predictions. Comparisons are provided in [`comparison.csv`](results/validation/seed0/comparison.csv). This archive does not establish that all 48 combinations have been revalidated using the current source code.
- `results/runs/`: Outputs from new experiments, including predictions, losses, weights, preprocessing statistics, configurations, environments, code hashes, and metric summaries.

```powershell
python scripts/report.py results/current/per_seed.csv --output results/new_report
python scripts/plot_fits.py
```

The reporting command requires an output directory that does not already exist. Plots use saved predictions without smoothing or refitting. The reference metric table contains raw RMSE and RMSE normalized by the maximum label; new runs additionally save range-normalized RMSE, MAE, and R².

## Environment and Archived Runs

The archived environment uses Python 3.10.9, PyTorch 2.11.0+cu128, and CUDA 12.8. Dependency installation has not yet been verified in a clean environment, and bitwise agreement across hardware or library versions is not guaranteed. The archived model source hashes differ from those of the current files, so the pointwise agreement reported in the comparisons applies only to the archived runs. Replaying configurations with noise also requires the same implementation, seed, and sequence of random-number calls; loading the weights and input mapping alone is insufficient to guarantee identical results.
