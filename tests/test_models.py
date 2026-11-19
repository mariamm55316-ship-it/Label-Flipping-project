import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baselines import make_models, evaluate


def test_model_settings():
    # RF and LR are built with the settings stated in the report
    m = make_models(42)
    assert m["RF"].n_estimators == 100
    assert m["RF"].criterion == "gini"
    assert m["LR"].C == 1.0
    assert m["LR"].max_iter == 1000


def test_evaluate():
    # Tiny example: 4 rows, 1 mistake (a Benign row predicted as Attack)
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    cat = np.array(["Benign", "Benign", "DDoS", "DDoS"])
    r = evaluate(y_true, y_pred, cat)
    assert r["accuracy"] == 0.75
    assert r["fpr"] == 0.5
    assert r["recall_DDoS"] == 1.0
