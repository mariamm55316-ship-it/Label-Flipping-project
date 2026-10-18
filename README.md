# Defending ML Intrusion Detectors Against Data Poisoning Attacks
CSCY413 Special Topics in Cybersecurity, The British University in Dubai

## Summary
We study random and targeted (DDoS) label-flipping attacks on Random Forest and
Logistic Regression classifiers trained on CICIoT2023 (binary: Benign vs Attack),
and evaluate a two-stage sanitisation defence (k-NN label check + outlier filtering)
that needs no trusted clean data.

## Team
Haneen Almohammed, Neam Arnaoud, Mariam Mohammed


## Structure
- `data/` dataset notes only (data not committed; see Kaggle link in report)
- `src/` preprocessing, poisoning, defence, models
- `experiments/` experiment scripts and configs (seeds 42-46)
- `results/` metrics tables and figures
- `docs/` reports (Deliverable 1, 2)

## Tools
Python, pandas, NumPy, scikit-learn, Matplotlib

## Status
Deliverable 1 done. Weeks 4-8 tasks tracked on the project board.
