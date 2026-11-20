# Defending ML Intrusion Detectors Against Data Poisoning Attacks
CSCY413 Special Topics in Cybersecurity, The British University in Dubai

## Summary
We evaluated random and targeted (DDoS) label-flipping attacks on Random Forest and Logistic Regression classifiers trained on CICIoT2023 (binary: Benign vs Attack). A two-stage sanitisation defence combining k-NN label consistency checks and outlier filtering was evaluated without trusted clean data.

## Team
Haneen Almohammed, Neam Arnaoud, Mariam Mohammed


## Structure
- `data/` dataset notes only; data not committed. [Dataset source](https://www.kaggle.com/datasets/madhavmalhotra/unb-cic-iot-dataset)
- `src/` preprocessing, poisoning, defence, models
- `experiments/` experiment notebooks and supporting files
- `results/` metrics tables and figures
- `docs/` reports (Deliverable 1, 2)

## Tools
Python, pandas, NumPy, scikit-learn, Matplotlib

## Status
Main experiments, ablation runs and automated tests have been completed.

## Setup
From the repository root, install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Automated tests

Run the project tests:

```bash
python -m pytest tests -v
```

Some checks require local processed data and may be skipped when it is unavailable.

## Synthetic demonstration

Run a small demonstration without downloading CICIoT2023:

```bash
python src/run_experiments.py --demo --seeds 42 --rates .1 --equal-flips 20 --output-dir demo_results
```

These synthetic results are separate from the reported research findings.

## Main experiments

Place the following files in `data/processed/`:

- `X_train.npy`, `X_test.npy`
- `y_train.npy`, `y_test.npy`
- `cat_train.npy`, `cat_test.npy`
- `neighbours.npy`

The neighbour cache must match the training features and row order.

Run the experiments:

```bash
python src/run_experiments.py --data-dir data/processed --output-dir results_reproduced
```

Repeat the same command to resume an interrupted run using the same data, code and settings.

## Experiment settings

Defaults are defined in `src/run_experiments.py`:

- Seeds: 42–46.
- Poisoning rates: 5%, 10% and 20%.
- Equal-flip counts: 520, 1040 and 2080.

Settings can be overridden through command-line options. The runner records them in `run_manifest.json`; it does not read a separate configuration file.

The dataset split remains fixed. Results are summarised as mean ± standard deviation across the five seeds.

## Repository rules

Raw and processed data, neighbour caches, fitted objects, local secrets and temporary files are excluded using `.gitignore`.

Use task branches and pull requests to review changes before merging into `main`.
