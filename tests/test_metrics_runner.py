"""Small known-answer checks and checkpoint/integration checks; no private data."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metrics import classification_metrics, defence_metrics
from poisoning import poison_labels
from run_experiments import demo_data, run, validate_data, validate_tables, read_records
from baselines import evaluate


class MetricsTests(unittest.TestCase):
    def test_classification_known_answer_and_team_agreement(self):
        actual, pred = np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])
        cat = np.array(["Benign", "Benign", "DDoS", "DDoS"])
        scores = classification_metrics(actual, pred, cat)
        self.assertEqual(scores, evaluate(actual, pred, cat))
        self.assertTrue(all(value == .5 for value in scores.values()))

    def test_no_benign_test_rows(self):
        scores = classification_metrics([1, 1], [1, 0], ["DDoS", "DDoS"])
        self.assertEqual(scores["recall"], .5)
        self.assertTrue(np.isnan(scores["fpr"]))

    def test_no_predicted_attacks(self):
        scores = classification_metrics([0, 1], [0, 0], ["Benign", "DDoS"])
        self.assertEqual(scores["precision"], 0.)
        self.assertEqual(scores["recall"], 0.)
        self.assertEqual(scores["f1"], 0.)

    def test_misaligned_categories_rejected(self):
        with self.assertRaises(ValueError):
            classification_metrics([0, 1], [0, 1], ["DDoS", "Benign"])

    def test_defence_known_answer(self):
        scores = defence_metrics(10, [1, 4], [0, 2, 5, 6, 7, 8, 9])
        self.assertEqual((scores["n_flipped"], scores["n_removed"], scores["n_caught"]), (2, 3, 2))
        self.assertEqual(scores["poison_detection_rate"], 1.)
        self.assertAlmostEqual(scores["removal_precision"], 2/3)
        self.assertEqual(scores["clean_removal_rate"], .125)

    def test_zero_denominators(self):
        scores = defence_metrics(10, [1, 4], np.arange(10))
        self.assertTrue(np.isnan(scores["removal_precision"]))
        self.assertEqual(scores["poison_detection_rate"], 0.)
        scores = defence_metrics(10, [], [0, 1, 2, 4, 5, 6, 7, 8, 9])
        self.assertTrue(np.isnan(scores["poison_detection_rate"]))
        self.assertEqual(scores["clean_removal_rate"], .1)
        scores = defence_metrics(0, [], [])
        self.assertTrue(all(np.isnan(scores[k]) for k in ["poison_detection_rate", "removal_precision", "clean_removal_rate"]))

    def test_bad_indices_rejected(self):
        for flipped, kept in [([1, 1], [0]), ([10], [0]), ([1.5], [0]), ([1], [-1]), ([1], [0, 0])]:
            with self.subTest(flipped=flipped, kept=kept), self.assertRaises(ValueError):
                defence_metrics(10, flipped, kept)


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = demo_data()
        cls.config = dict(seeds=[42], rates=[.1], equal_flips=[20])

    def test_poisoning_copy_scope_and_repeatability(self):
        y, cat = self.data["y_train"], self.data["cat_train"]
        original = y.copy()
        poisoned, idx = poison_labels(y, cat, "targeted", .1, 42)
        again, idx2 = poison_labels(y, cat, "targeted", .1, 42)
        np.testing.assert_array_equal(y, original)
        np.testing.assert_array_equal(idx, idx2)
        np.testing.assert_array_equal(poisoned, again)
        np.testing.assert_array_equal(np.flatnonzero(poisoned != y), idx)
        self.assertTrue((cat[idx] == "DDoS").all())
        self.assertTrue((poisoned[idx] == 0).all())
        self.assertFalse(np.shares_memory(poisoned, y))

    def test_invalid_neighbour_cache(self):
        bad = dict(self.data)
        bad["neighbours"] = bad["neighbours"].copy()
        bad["neighbours"][0, 0] = 0
        with self.assertRaises(ValueError):
            validate_data(bad)

    def test_checkpoint_resume_and_different_data_rejection(self):
        fits = []
        labels = self.data["y_test"]
        class SmallModel:
            def fit(self, X, y):
                fits.append(len(y))
                return self
            def predict(self, X):
                return labels.copy()
        def factory(seed):
            return {name: SmallModel() for name in ["RF", "LR"]}
        before = {k: v.copy() for k, v in self.data.items()}
        with tempfile.TemporaryDirectory() as temp:
            with patch("builtins.print"):
                report = run(self.data, temp, self.config, "synthetic_demo", factory)
            self.assertEqual(report, dict(model_rows=14, defence_rows=3, complete=True))
            self.assertEqual(len(fits), 14)
            # A complete resume must not fit another model.
            with patch("builtins.print"):
                run(self.data, temp, self.config, "synthetic_demo", factory)
            self.assertEqual(len(fits), 14)
            # A partial resume should fit exactly the missing result.
            path = Path(temp) / "model_results_all_runs.csv"
            pd.read_csv(path).iloc[:-1].to_csv(path, index=False)
            with patch("builtins.print"):
                run(self.data, temp, self.config, "synthetic_demo", factory)
            self.assertEqual(len(fits), 15)
            changed = dict(self.data)
            changed["X_train"] = changed["X_train"].copy()
            changed["X_train"][0, 0] += .01
            with self.assertRaises(ValueError):
                run(changed, temp, self.config, "synthetic_demo", factory)
            manifest = json.loads((Path(temp) / "run_manifest.json").read_text())
            models = read_records(path)
            with self.assertRaises(ValueError):
                validate_tables(models + [models[0]], read_records(Path(temp) / "defence_results.csv"), self.config, manifest["counts"])
        for name in before:
            np.testing.assert_array_equal(self.data[name], before[name])


if __name__ == "__main__":
    unittest.main()
