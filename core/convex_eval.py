"""BCE va danh gia signed excess loss so voi reference, khong goi la true regret."""

import numpy as np
from scipy.special import expit


def logistic_loss_gradient(theta, X, y):
    """Tra mean BCE va gradient theo theta cho mot batch soft targets.

    Inputs huu han: theta(d,), X(B,d), y(B,) trong [0,1]. Khong clip loss gap.
    """
    theta = np.asarray(theta, dtype=float)
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    if (theta.ndim != 1 or X.ndim != 2 or X.shape[0] < 1
            or X.shape[1] != theta.size or y.shape != (X.shape[0],)):
        raise ValueError("incompatible theta/X/y shapes")
    if not all(np.isfinite(a).all() for a in (theta, X, y)) or np.any((y < 0) | (y > 1)):
        raise ValueError("finite inputs and targets in [0,1] required")
    logits = X @ theta
    if not np.isfinite(logits).all():
        raise ValueError("logits overflow")
    # Dang theo dau logits tranh tru hai so lon khi soft target gan 1.
    losses = np.maximum(logits, 0) * (1 - y) + np.maximum(-logits, 0) * y
    losses += np.log1p(np.exp(-np.abs(logits)))
    gradient = X.T @ (expit(logits) - y) / len(y)
    if not np.isfinite(losses).all() or not np.isfinite(gradient).all():
        raise ValueError("loss or gradient overflow")
    return float(np.mean(losses)), gradient


def sample_logistic_batch(reference, batch_size, label_noise_std, rng):
    """Sinh X Gaussian va y=sigmoid(X reference+epsilon) giong shared trainer.

    label_noise_std la do lech chuan epsilon, khong phai sigma cua schedule.
    """
    reference = np.asarray(reference, dtype=float)
    if reference.ndim != 1 or not np.isfinite(reference).all() or reference.size == 0:
        raise ValueError("reference must be a finite nonempty vector")
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if (isinstance(label_noise_std, bool) or not np.isfinite(label_noise_std)
            or label_noise_std < 0):
        raise ValueError("label_noise_std must be finite and nonnegative")
    X = rng.normal(loc=0.0, scale=1.0, size=(batch_size, len(reference)))
    eps = rng.normal(loc=0.0, scale=label_noise_std, size=(batch_size,))
    return X, expit(X @ reference + eps)


def evaluate_history(history, *, batch_size, label_noise_std, eval_batch_size, eval_seed):
    """Danh gia theta_t truoc update: replay train batch va MC doc lap.

    Cung eval_seed cho cac methods tao common evaluation samples. MC gap signed
    la uoc luong excess loss toi generating reference, chua la dynamic regret.
    """
    theta, reference = history.theta_path, history.theta_star_path
    if theta.shape != reference.shape or theta.ndim != 2 or len(theta) < 1:
        raise ValueError("nonempty matching trajectories required")
    if not np.isfinite(theta).all() or not np.isfinite(reference).all():
        raise ValueError("trajectories must be finite")
    if eval_batch_size < 2:
        raise ValueError("eval_batch_size must be >= 2 for standard error")
    train_rng = np.random.default_rng(history.metadata["seed"])
    eval_rng = np.random.default_rng(eval_seed)
    metrics = {key: np.empty(len(theta)) for key in (
        "train_batch_loss", "train_reference_loss", "train_empirical_excess",
        "eval_mc_loss", "eval_mc_reference_loss", "eval_mc_excess", "eval_mc_excess_se")}
    for i, (w, ref) in enumerate(zip(theta, reference)):
        X, y = sample_logistic_batch(ref, batch_size, label_noise_std, train_rng)
        train_loss, _ = logistic_loss_gradient(w, X, y)
        ref_loss, _ = logistic_loss_gradient(ref, X, y)
        if not np.isclose(train_loss, history.batch_loss[i], rtol=1e-11, atol=1e-12):
            raise ValueError("training replay differs from shared trainer; RNG contract changed")
        metrics["train_batch_loss"][i] = train_loss
        metrics["train_reference_loss"][i] = ref_loss
        metrics["train_empirical_excess"][i] = train_loss - ref_loss
        X, y = sample_logistic_batch(ref, eval_batch_size, label_noise_std, eval_rng)
        loss, _ = logistic_loss_gradient(w, X, y)
        ref_loss, _ = logistic_loss_gradient(ref, X, y)
        z, zr = X @ w, X @ ref
        gap = np.logaddexp(0, z) - np.logaddexp(0, zr) - y * (z - zr)
        metrics["eval_mc_loss"][i] = loss
        metrics["eval_mc_reference_loss"][i] = ref_loss
        metrics["eval_mc_excess"][i] = np.mean(gap)
        metrics["eval_mc_excess_se"][i] = np.std(gap, ddof=1) / np.sqrt(eval_batch_size)
    metrics["train_cum_empirical_excess"] = np.cumsum(metrics["train_empirical_excess"])
    metrics["eval_cum_mc_excess"] = np.cumsum(metrics["eval_mc_excess"])
    # Conditional tren learned path, cac eval batches doc lap qua thoi gian.
    metrics["eval_cum_mc_excess_se"] = np.sqrt(np.cumsum(metrics["eval_mc_excess_se"] ** 2))
    if not all(np.isfinite(value).all() for value in metrics.values()):
        raise ValueError("nonfinite evaluation metrics")
    return metrics
