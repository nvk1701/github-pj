"""Deterministic internal measurements. Author parity requires Colab exports."""

from dataclasses import replace
from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()
