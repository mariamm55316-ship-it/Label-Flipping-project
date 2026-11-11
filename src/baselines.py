"""Clean baselines: Random Forest and Logistic Regression (owner: Haneen)."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

DATA = "data/processed/"


def make_models(seed=42):
    """Same hyperparameters in every condition (clean, poisoned, sanitised)."""
    rf = RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=None,
                                random_state=seed, n_jobs=-1)
    lr = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                            max_iter=1000, random_state=seed)
    return {"RF": rf, "LR": lr}


def evaluate(y_true, y_pred, cat):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    row = {"accuracy": accuracy_score(y_true, y_pred),
           "precision": precision_score(y_true, y_pred),
           "recall": recall_score(y_true, y_pred),
           "f1": f1_score(y_true, y_pred),
           "fpr": fp / (fp + tn)}
    for c in sorted(set(cat)):
        if c != "Benign":
            row["recall_" + c] = (y_pred[cat == c] == 1).mean()
    return row


if __name__ == "__main__":
    X_train = np.load(DATA + "X_train.npy")
    X_test = np.load(DATA + "X_test.npy")
    y_train = np.load(DATA + "y_train.npy")
    y_test = np.load(DATA + "y_test.npy")
    cat_test = np.load(DATA + "cat_test.npy")

    rows = []
    for name, model in make_models(42).items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({"model": name, "condition": "clean", **evaluate(y_test, pred, cat_test)})

    df = pd.DataFrame(rows)
    df.to_csv("results/baseline_results.csv", index=False)
    print(df.round(4).to_string(index=False))
