"""Independent NumPy implementation of the audited Figure 5-bottom contract."""

from dataclasses import dataclass
import hashlib

import numpy as np
from scipy.special import expit

from core.convex_population import PopulationEvaluator
from core.drift import EMADriftEstimator
from core.utils import project_to_ball

ALIGNED = "convex-author-aligned-v1"
ABLATION = "convex-author-ablation-v1"
PILOT = "convex-author-pilot-v1"
DIAGNOSTIC = "convex-author-rng-diagnostic-v1"


@dataclass(frozen=True)
class Case:
    protocol: str
    dimension: int
    seed: int
    method: str = "convex"
    intervention: str = "none"

    @property
    def key(self):
        return f"{self.protocol}-d{self.dimension}-s{self.seed}-{self.method}-{self.intervention}"


def matrix(cfg):
    aligned = [Case(ALIGNED, d, s, m) for d in cfg["dimensions"]
               for s in cfg["seeds"] for m in cfg["methods"]]
    ablations = [Case(ABLATION, d, s, intervention=i) for d in cfg["ablation_dimensions"]
                 for s in cfg["seeds"] for i in cfg["interventions"]]
    return aligned + ablations


def reference_phases(cfg, dimension):
    phases = cfg["steps"] // cfg["phase_length"]
    angles = np.linspace(0, 2 * np.pi * cfg["revolutions"], phases)
    radii = np.linspace(1, -1, phases)
    pairs = angles[:, None] * np.arange(1, dimension // 2 + 1)
    path = np.empty((phases, dimension), dtype=np.float64)
    path[:, 0::2] = np.cos(pairs)
    path[:, 1::2] = np.sin(pairs)
    return path * (radii * radii * radii)[:, None]


def settings(cfg, case):
    intervention = case.intervention
    return {
        "noise_std": cfg["ablation_noise_std"] if intervention == "noise" else cfg["label_noise_std"],
        "ema_beta": cfg["ema_beta"] if intervention == "ema" else None,
        "initial_lr": cfg["ablation_initial_lr"] if intervention == "initial_lr" else None,
        "radius": cfg["projection_radius_per_dimension"] * case.dimension if intervention == "projection" else None,
    }


def phases_with_data(cfg, case, phases, noise_std, *, native_rng=False):
    """Allocate at most one phase of X (plus the float64 conversion during draws)."""
    phase_size = cfg["phase_length"] * cfg["batch_size"]
    if not native_rng:
        x_rng, noise_rng = [np.random.default_rng(child) for child in
                            np.random.SeedSequence([case.seed, case.dimension]).spawn(2)]
    for phase, ref in enumerate(phases):
        if native_rng:
            x_rng = noise_rng = np.random.RandomState(phase)
        x = x_rng.normal(size=(phase_size, case.dimension)).astype(np.float32)
        noise = noise_rng.normal(scale=noise_std, size=phase_size)
        y = expit(x @ ref + noise).astype(np.float32)
        yield x, y


def bce(logits, y):
    return (np.maximum(logits, 0) - logits * y + np.log1p(np.exp(-np.abs(logits)))).mean(dtype=np.float32)


def loss_gradient(theta, x, y):
    logits = x @ theta
    # This order mirrors the stable BCE expression in float32. Compute the
    # positive-logit residual with sigmoid(-z) to avoid sigmoid rounding to 1.
    residual = np.where(logits >= 0, (1 - y) - expit(-logits), expit(logits) - y)
    gradient = x.T @ (residual / np.float32(len(y)))
    return bce(logits, y), gradient


def author_thresholds(gamma, batch_size, smoothness, diameter, variance):
    """Evaluate Eq. 18–19 with the scalar precision observed on Colab."""
    gamma = np.float32(gamma)
    diameter = np.float32(diameter)
    smoothness = np.float32(smoothness)
    bounds = np.array([
        np.square(gamma) + (np.float32(2) * diameter) * gamma,
        np.square(gamma + diameter),
    ], dtype=np.float32)
    # The notebook's weak scalar promotion keeps these intermediates in float32.
    # The explicit NumPy float64 variance factor then promotes the radical.
    quadratic = np.square(bounds) * np.square(smoothness)
    linear = bounds * smoothness
    variance_ratio = np.square(np.sqrt(np.float64(variance))) / batch_size
    return (np.sqrt(quadratic.astype(np.float64) + (4 * variance_ratio) * bounds.astype(np.float64))
            - linear) * (np.float64(0.5) / variance_ratio)


def train(cfg, case, *, native_rng=False, supplied=None):
    """Train on bounded-memory phases, or replay a diagnostic export."""
    d, steps, batch = case.dimension, cfg["steps"], cfg["batch_size"]
    phase_length = cfg["phase_length"]
    knobs = settings(cfg, case)
    phases = reference_phases(cfg, d)
    refs = np.repeat(phases.astype(np.float32), phase_length, axis=0)
    if supplied is not None:
        refs = supplied["reference"]
        phase_rows = np.arange(0, steps, phase_length)
        data = ((supplied["X"][i:i + phase_length].reshape(-1, d),
                 supplied["y"][i:i + phase_length].reshape(-1)) for i in phase_rows)
    else:
        data = phases_with_data(cfg, case, phases, knobs["noise_std"], native_rng=native_rng)

    history = {name: np.empty(steps, dtype=np.float64) for name in ("gamma", "lr")}
    for name in ("loss", "reference_loss"):
        history[name] = np.empty(steps, dtype=np.float32)
    for name in ("theta", "post_theta", "gradient"):
        history[name] = np.empty((steps, d), dtype=np.float32)
    history["thresholds"] = np.zeros((steps, 2), dtype=np.float64)
    history["projection_active"] = np.zeros(steps, dtype=bool)
    history["reference"] = refs
    history["initial"] = theta = np.zeros(d, dtype=np.float32)
    diameter = d * cfg["diameter_per_dimension"]
    variance = d * cfg["variance_per_dimension"]
    ema = EMADriftEstimator(knobs["ema_beta"]) if knobs["ema_beta"] is not None else None
    x_hash, y_hash = hashlib.sha256(), hashlib.sha256()
    previous_lr = knobs["initial_lr"]

    for phase_index, (phase_x, phase_y) in enumerate(data):
        x_hash.update(phase_x.tobytes())
        y_hash.update(phase_y.tobytes())
        for offset in range(phase_length):
            t = phase_index * phase_length + offset
            sl = slice(offset * batch, (offset + 1) * batch)
            x, y = phase_x[sl], phase_y[sl]
            gamma = (
                ema.update(theta) if ema is not None
                else float(np.linalg.norm(refs[t] - refs[min(t + 1, steps - 1)]))
            )
            if case.method == "convex":
                lo, hi = author_thresholds(gamma, batch, cfg["L"], diameter, variance)
                lr = lo if previous_lr is None or previous_lr <= lo else min(hi, previous_lr)
                history["thresholds"][t] = lo, hi
            elif case.method == "constant":
                lr = cfg["baseline_lr0"]
            else:
                lr = cfg["baseline_lr0"] / np.sqrt(t + 1)
            previous_lr = lr
            loss, grad = loss_gradient(theta, x, y)
            history["gamma"][t], history["lr"][t] = gamma, lr
            history["loss"][t] = loss
            history["reference_loss"][t] = bce(x @ refs[t], y)
            history["theta"][t], history["gradient"][t] = theta, grad
            theta = theta - np.float32(lr) * grad
            radius = knobs["radius"]
            if radius is not None and np.linalg.norm(theta.astype(np.float64)) > radius:
                theta = project_to_ball(theta.astype(np.float64), radius).astype(np.float32)
                history["projection_active"][t] = True
            history["post_theta"][t] = theta

    history["M0_step"] = history["loss"] - history["reference_loss"]
    history["M0"] = np.cumsum(history["M0_step"], dtype=np.float32)
    if not all(np.isfinite(x).all() for x in history.values()):
        raise ValueError(f"nonfinite history for {case.key}")
    return history, {"X_sha256": x_hash.hexdigest(), "y_sha256": y_hash.hexdigest()}


def evaluate(cfg, case, history, evaluators):
    knobs = settings(cfg, case)
    ref = np.repeat(reference_phases(cfg, case.dimension), cfg["phase_length"], axis=0)
    key = case.dimension, knobs["noise_std"]
    if key not in evaluators:
        evaluators[key] = PopulationEvaluator(
            ref, knobs["noise_std"], tolerance=cfg["population_tolerance"]
        )
    evaluator = evaluators[key]
    radius = knobs["radius"]
    if radius is not None and np.linalg.norm(evaluator.oracle_path, axis=1).max() > radius:
        raise ValueError("population oracle is outside the declared projection domain")
    values = evaluator.evaluate(history["theta"])
    m1 = values["population_A"]
    m2 = values["population_B"]
    correction = values["population_correction"]
    if not np.allclose(m2 - m1, correction, atol=3e-15, rtol=0):
        raise ValueError("population comparator identity failed")
    return {
        "M1_step": m1, "M2_step": m2, "M1": np.cumsum(m1), "M2": np.cumsum(m2),
        "correction": correction, "oracle": evaluator.oracle_path,
        "m1_quadrature_budget": values["population_error_A"],
        "m2_quadrature_budget": values["population_error_B"],
        "oracle_parameter_error": values["oracle_parameter_error_estimate"],
        "oracle_suboptimality": values["oracle_risk_suboptimality_estimate"],
    }
