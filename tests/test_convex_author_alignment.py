"""Deterministic internal measurements. Author parity requires Colab exports."""

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import expit

from core.convex_population import gaussian_population_loss_gradient, gaussian_population_oracle
from core.utils import load_config
from experiments.convex_author_aligned import (
    ALIGNED, ABLATION, Case, bce, evaluate, loss_gradient, matrix, phases_with_data,
    reference_phases, train,
)
from experiments.convex_author_analysis import verify_pairs
from experiments.convex_author_evidence import load_export, sha256
from scripts.export_convex_author_colab import SNAPSHOT_SHA256, URL
from scripts.run_convex_author_full import validate_history, validated

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "configs/convex_author_v1.yaml")
SMALL = {**CFG, "steps": 30, "batch_size": 12}
CASE = Case(ALIGNED, 8, 1)


class AuthorAlignmentTests(unittest.TestCase):
    def test_matrix(self):
        plan = matrix(CFG)
        self.assertEqual(sum(c.protocol == ALIGNED for c in plan), 45)
        self.assertEqual(sum(c.protocol == ABLATION for c in plan), 36)
        self.assertEqual(len({c.key for c in plan}), 81)

    def test_reference_and_switch_index(self):
        phases = reference_phases(CFG, 8)
        self.assertEqual(phases.shape, (200, 8))
        np.testing.assert_array_equal(phases[0], [1, 0] * 4)
        np.testing.assert_allclose(phases[-1], [-1, 0] * 4, atol=5e-15)
        phase = 17
        expected = (1 - 2 * phase / 199)**3 * np.sin(3 * 8 * np.pi * phase / 199)
        self.assertAlmostEqual(phases[phase, 5], expected)

    def test_native_draw_order(self):
        phases = reference_phases(SMALL, 8)
        x, y = next(phases_with_data(SMALL, CASE, phases, .1, native_rng=True))
        rng = np.random.RandomState(0)
        expected_x = rng.standard_normal(x.shape).astype(np.float32)
        z = expected_x @ phases[0] + .1 * rng.standard_normal(len(x))
        np.testing.assert_array_equal(x, expected_x)
        np.testing.assert_allclose(y, (1 / (1 + np.exp(-z))).astype(np.float32))

    def test_bce_and_gradient(self):
        logits = np.array([-1000, 0, 1000], dtype=np.float32)
        y = np.array([.2, .5, .8], dtype=np.float32)
        self.assertAlmostEqual(float(bce(logits, y)), (400 + np.log(2)) / 3, places=4)
        rng = np.random.default_rng(5)
        x = rng.normal(size=(17, 6)).astype(np.float32)
        theta = rng.normal(size=6).astype(np.float32)
        y = rng.uniform(size=17).astype(np.float32)
        loss, gradient = loss_gradient(theta, x, y)
        eps = 1e-3
        perturb = (eps * np.eye(6)).astype(np.float32)
        finite = [(bce(x @ (theta + delta), y) - bce(x @ (theta - delta), y)) / (2 * eps)
                  for delta in perturb]
        np.testing.assert_allclose(gradient, finite, rtol=0, atol=8e-5)
        np.testing.assert_allclose(loss, np.mean(np.logaddexp(0, x @ theta) - y * (x @ theta)),
                                   atol=3e-7)

    def test_initialization_threshold_and_update_order(self):
        history, _ = train(SMALL, CASE)
        np.testing.assert_array_equal(history["theta"][:10], 0)
        np.testing.assert_array_equal(history["lr"][:9], 0)
        self.assertGreater(history["lr"][9], 0)
        self.assertEqual(history["gamma"][-1], 0)
        gamma = history["gamma"]
        b1 = gamma**2 + 16 * gamma
        for column, b in enumerate((b1, (gamma + 8)**2)):
            v_over_b = 2 / SMALL["batch_size"]
            expected = (np.sqrt((.25 * b)**2 + 4 * v_over_b * b) - .25 * b) / (2 * v_over_b)
            np.testing.assert_allclose(history["thresholds"][:, column], expected, atol=1e-13)
        np.testing.assert_array_equal(history["lr"], np.clip(np.r_[0., history["lr"][:-1]],
                                                             *history["thresholds"].T))
        np.testing.assert_array_equal(history["post_theta"],
                                      history["theta"] - history["lr"][:, None].astype(np.float32) * history["gradient"])
        np.testing.assert_array_equal(history["post_theta"][:-1], history["theta"][1:])
        self.assertFalse(history["projection_active"].any())
        np.testing.assert_array_equal(history["M0"], np.cumsum(history["loss"] - history["reference_loss"]))

    def test_matched_inputs_and_ema_zero_lr(self):
        base, expected = train(SMALL, CASE)
        for method in ("constant", "inverse_sqrt"):
            history, actual = train(SMALL, replace(CASE, method=method))
            self.assertEqual(expected, actual)
            np.testing.assert_array_equal(history["initial"], base["initial"])
            lr = .1 if method == "constant" else .1 / np.sqrt(np.arange(1, 31))
            np.testing.assert_allclose(history["lr"], lr)
        ema, actual = train(SMALL, replace(CASE, intervention="ema"))
        self.assertEqual(expected, actual)
        for key in ("lr", "gamma", "theta"):
            np.testing.assert_array_equal(ema[key], 0)
        init, _ = train(SMALL, replace(CASE, intervention="initial_lr"))
        self.assertEqual(init["lr"][0], 1.)
        _, actual = train(SMALL, replace(CASE, intervention="noise"))
        self.assertEqual(expected["X_sha256"], actual["X_sha256"])
        self.assertNotEqual(expected["y_sha256"], actual["y_sha256"])

    def test_projection_activation(self):
        history, _ = train({**SMALL, "projection_radius_per_dimension": .005},
                           replace(CASE, intervention="projection"))
        self.assertTrue(history["projection_active"].any())
        self.assertLessEqual(np.linalg.norm(history["post_theta"], axis=1).max(), .04000001)

    def test_population_identity_and_independent_oracle_quadrature(self):
        history, _ = train(SMALL, CASE)
        values = evaluate(SMALL, CASE, history, {})
        np.testing.assert_allclose(values["M2"] - values["M1"], np.cumsum(values["correction"]), atol=3e-15)
        ref = np.array([1.3])
        noise = .31622776601683794
        oracle = gaussian_population_oracle(ref, noise)
        w = oracle["theta"][0]
        gaussian = lambda z: np.exp(-z*z/2) / np.sqrt(2*np.pi)

        def target(x):
            return quad(lambda e: expit(1.3*x + noise*e)*gaussian(e), -10, 10)[0]

        expected = quad(lambda x: (np.logaddexp(0, w*x) - w*x*target(x))*gaussian(x), -10, 10)[0]
        derivative = quad(lambda x: x*(expit(w*x)-target(x))*gaussian(x), -10, 10)[0]
        loss, _, _ = gaussian_population_loss_gradient(oracle["theta"], ref, noise)
        self.assertAlmostEqual(loss, expected, places=10)
        self.assertAlmostEqual(derivative, 0, places=10)
        self.assertLess(w, ref[0])
        np.testing.assert_array_equal(gaussian_population_oracle(ref, 0.)["theta"], ref)

    def test_ema_cold_start_has_nonzero_gradients_but_zero_updates(self):
        case = Case(ABLATION, 512, 0, intervention="ema")
        history, _ = train({**SMALL, "steps": 100}, case)
        self.assertTrue(np.all(np.linalg.norm(history["gradient"], axis=1) > 0))
        for key in ("lr", "gamma", "theta", "post_theta"):
            np.testing.assert_array_equal(history[key], 0)

    def test_resume_preserves_unsealed_payload(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "runs" / CASE.key
            folder.mkdir(parents=True)
            self.assertIsNone(validated(root, CASE, {}))
            path = folder / "history.npz"
            path.write_bytes(b"unsealed")
            with self.assertRaisesRegex(ValueError, "unsealed history"):
                validated(root, CASE, {})
            self.assertEqual(path.read_bytes(), b"unsealed")

    def test_history_checks_metrics_and_dtype(self):
        history, _ = train(SMALL, CASE)
        history.update(evaluate(SMALL, CASE, history, {}))
        metrics = {name: float(history[name][-1]) for name in ("M0", "M1", "M2")}
        validate_history(history, SMALL["steps"], CASE.dimension, metrics)
        with self.assertRaisesRegex(ValueError, "cumulative M1"):
            validate_history(history, SMALL["steps"], CASE.dimension, {**metrics, "M1": -1.})
        with self.assertRaisesRegex(ValueError, "invalid array M0"):
            validate_history({**history, "M0": history["M0"].astype(float)},
                             SMALL["steps"], CASE.dimension, metrics)

    def test_paired_aggregation_rejects_unmatched_streams(self):
        base = {"data": {"X_sha256": "common_x", "y_sha256": "common_y"}}
        variant = {"data": {"X_sha256": "common_x", "y_sha256": "noisier_y"}}
        noise = Case(ABLATION, CASE.dimension, CASE.seed, intervention="noise")
        constant = replace(CASE, method="constant")
        verify_pairs({CASE: base, constant: base, noise: variant})
        with self.assertRaisesRegex(ValueError, "unmatched method"):
            verify_pairs({CASE: base, constant: variant})
        with self.assertRaisesRegex(ValueError, "unmatched ablation"):
            verify_pairs({CASE: base, noise: base})
        with self.assertRaisesRegex(ValueError, "missing ablation control"):
            verify_pairs({noise: variant})

    def test_export_provenance_and_checksum_checks(self):
        # Only validate sidecar mechanics. No author experiment runs in this test.
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "native-d2.npz"
            sidecar = path.with_suffix(".json")
            self.assertIsNone(load_export(root, "native", 2, CFG))
            path.write_bytes(b"test payload")
            with self.assertRaisesRegex(ValueError, "incomplete Colab export"):
                load_export(root, "native", 2, CFG)
            meta = {"author_commit": CFG["author_commit"], "author_blob": CFG["author_blob"],
                    "author_url": URL, "notebook_sha256": SNAPSHOT_SHA256, "kind": "native",
                    "dimension": 2, "executed_at_utc": "2026-11-20T09:00:00+00:00",
                    "environment": {"runtime": "google-colab", "python": "test",
                                    "numpy": "test", "tensorflow": "test"},
                    "sha256": sha256(path)}
            sidecar.write_text(json.dumps(meta), encoding="utf-8")
            self.assertEqual(load_export(root, "native", 2, CFG), (meta, path))
            for invalid in ({"executed_at_utc": "local test"}, {"notebook_sha256": "wrong"},
                            {"environment": {**meta["environment"], "runtime": "local"}}):
                sidecar.write_text(json.dumps(meta | invalid), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "invalid Colab evidence metadata"):
                    load_export(root, "native", 2, CFG)
            sidecar.write_text(json.dumps(meta), encoding="utf-8")
            path.write_bytes(b"corruption")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_export(root, "native", 2, CFG)


if __name__ == "__main__":
    unittest.main()
