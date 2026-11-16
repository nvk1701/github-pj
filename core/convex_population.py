"""Gaussian soft-label population risk and a numerical radial oracle.

These reductions require independent X~N(0,I) and epsilon~N(0,s^2).
Quadrature errors are numerical estimates, not certified interval bounds.
See docs/convex_protocol.md for the derivation and protocol labels.
"""

import math
import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import brentq
from scipy.special import expit


def _vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or not result.size or not np.isfinite(result).all():
        raise ValueError(f'{name} must be a finite nonempty vector')
    if not math.isfinite(float(np.linalg.norm(result))):
        raise ValueError(f'{name} norm must be representable')
    return result


def _nonnegative(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f'{name} must be a finite nonnegative scalar')
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f'{name} must be a finite nonnegative scalar')
    return result


def _integral(function, tolerance):
    # Symmetric one-dimensional Gaussian expectations use only z >= 0.
    with warnings.catch_warnings():
        warnings.simplefilter('error', IntegrationWarning)
        value, error = quad(lambda z: function(z)*math.exp(-z*z/2)/math.sqrt(2*math.pi),
                            0, np.inf, epsabs=tolerance, epsrel=tolerance, limit=200)
    if not math.isfinite(value) or error > 10*tolerance:
        raise ValueError('population quadrature did not meet its error budget')
    return value, error


def _moment(radius, tolerance):
    # h(u) = E[Z sigmoid(u Z)]. Centering removes the constant odd term.
    return _integral(lambda z: z*math.tanh(radius*z/2), tolerance)


def _tolerance(value):
    value = _nonnegative(value, 'tolerance')
    if not 1e-13 <= value <= 1e-6:
        raise ValueError('tolerance must be in [1e-13, 1e-6]')
    return value


def gaussian_population_loss_gradient(theta, reference, label_noise_std, *, tolerance=1e-11):
    """Return (risk, gradient, numerical error estimates) under Gaussian labels.

    Risk = E[softplus(X theta) - sigmoid(X reference + epsilon) X theta].
    Errors separate loss and gradient estimates and exclude sampling error.
    This function uses deterministic one-dimensional quadrature, not MC.
    """
    theta, reference = _vector(theta, 'theta'), _vector(reference, 'reference')
    if theta.shape != reference.shape:
        raise ValueError('theta and reference must have matching shapes')
    noise = _nonnegative(label_noise_std, 'label_noise_std')
    tolerance = _tolerance(tolerance)
    u, r = float(np.linalg.norm(theta)), float(np.linalg.norm(reference))
    q = math.hypot(r, noise)
    hu, eu = _moment(u, tolerance)
    hq, eq = _moment(q, tolerance)
    target_scale = hq/q if q else .25
    loss, loss_error = _integral(
        lambda z: float(np.logaddexp(0, u*z) + np.logaddexp(0, -u*z)), tolerance)
    dot = float(theta @ reference)
    loss -= dot*target_scale
    gradient = theta*(hu/u if u else .25) - reference*target_scale
    errors = dict(loss=loss_error + (abs(dot)/q*eq if q else 0),
                  gradient=eu + (r/q*eq if q else 0))
    if not math.isfinite(loss) or not np.isfinite(gradient).all():
        raise ValueError('population risk or gradient overflow')
    return float(loss), gradient, errors


def gaussian_population_oracle(reference, label_noise_std, *, tolerance=1e-11):
    """Return a radial approximate population minimizer and error diagnostics.

    Solve h(u) = r/sqrt(r^2+s^2) h(sqrt(r^2+s^2)) on [0,r].
    The exact minimizer is unique and shrinks the generating reference for
    r>0 and s>0. Zero noise or zero reference returns the known exact vector.
    Parameter/risk error estimates use quadrature residuals and curvature.
    They are not formal interval certificates. No projection is applied.
    """
    reference = _vector(reference, 'reference')
    noise = _nonnegative(label_noise_std, 'label_noise_std')
    tolerance = _tolerance(tolerance)
    r = float(np.linalg.norm(reference))
    if not r or not noise:
        return dict(theta=reference.copy(), radial_norm=r, gradient_residual=0.,
                    gradient_error_estimate=0., parameter_error_estimate=0.,
                    risk_suboptimality_estimate=0., quadrature_tolerance=tolerance)
    q = math.hypot(r, noise)
    hq, eq = _moment(q, tolerance)
    target = (r/q)*hq
    hr, er = _moment(r, tolerance)
    if target > hr:
        if target-hr > eq+er:
            raise ValueError('radial oracle bracket failed')
        u = r
    else:
        u = brentq(lambda value: _moment(value, tolerance)[0]-target,
                   0, r, xtol=tolerance, rtol=tolerance)
    hu, eu = _moment(u, tolerance)
    curvature, ec = _integral(lambda z: 2*z*z*expit(r*z)*expit(-r*z), tolerance)
    if curvature <= ec:
        raise ValueError('oracle curvature cannot be resolved at this tolerance')
    residual = abs(hu-target)
    gradient_error = eu+(r/q)*eq
    parameter_error = (residual+gradient_error)/(curvature-ec)
    return dict(theta=reference*(u/r), radial_norm=float(u), gradient_residual=residual,
                gradient_error_estimate=gradient_error, parameter_error_estimate=parameter_error,
                risk_suboptimality_estimate=.125*parameter_error**2,
                quadrature_tolerance=tolerance)


class PopulationEvaluator:
    """Cache a distribution path's scalar quadrature across methods and seeds.

    Exact floating-point reference norms are keys, without rounding or fitting
    to evaluation samples. Only the learner softplus expectation varies per
    evaluation. Errors retain the CP02 estimated, non-certified meaning.
    """

    def __init__(self, reference_path, label_noise_std, *, tolerance=1e-11):
        reference = np.asarray(reference_path, dtype=float)
        if reference.ndim != 2 or min(reference.shape) < 1 or not np.isfinite(reference).all():
            raise ValueError('a finite nonempty reference path is required')
        self.reference = reference.copy()
        self.noise = _nonnegative(label_noise_std, 'label_noise_std')
        self.tolerance = _tolerance(tolerance)
        self.cache = {}
        self.entries = []
        oracle_path = []
        for ref in self.reference:
            r = float(np.linalg.norm(ref))
            if r not in self.cache:
                scalar = np.array([r])
                oracle = gaussian_population_oracle(scalar, self.noise, tolerance=self.tolerance)
                ref_loss, _, ref_errors = gaussian_population_loss_gradient(scalar, scalar, self.noise, tolerance=self.tolerance)
                oracle_loss, _, oracle_errors = gaussian_population_loss_gradient(oracle['theta'], scalar, self.noise, tolerance=self.tolerance)
                q = math.hypot(r, self.noise)
                moment, error = _moment(q, self.tolerance)
                self.cache[r] = dict(scale=moment/q if q else .25, scale_error=error/q if q else 0.,
                    ref_loss=ref_loss, ref_error=ref_errors['loss'], oracle_loss=oracle_loss,
                    oracle_error=oracle_errors['loss'], coefficient=oracle['radial_norm']/r if r else 0.,
                    parameter_error=oracle['parameter_error_estimate'], risk_error=oracle['risk_suboptimality_estimate'])
            entry = self.cache[r]
            self.entries.append(entry)
            oracle_path.append(ref*entry['coefficient'])
        self.oracle_path = np.array(oracle_path)

    def evaluate(self, theta_path):
        """Return pre-update population A/B, correction and error arrays (T,)."""
        theta = np.asarray(theta_path, dtype=float)
        if theta.shape != self.reference.shape or not np.isfinite(theta).all():
            raise ValueError('learner path must match the finite reference path')
        rows = []
        for w, ref, entry in zip(theta, self.reference, self.entries):
            u = float(np.linalg.norm(w))
            if not math.isfinite(u):
                raise ValueError('learner norm overflow')
            loss, error = _integral(lambda z: float(np.logaddexp(0,u*z)+np.logaddexp(0,-u*z)), self.tolerance)
            dot = float(w@ref)
            loss -= dot*entry['scale']
            error += abs(dot)*entry['scale_error']
            rows.append((loss-entry['ref_loss'], loss-entry['oracle_loss'],
                entry['ref_loss']-entry['oracle_loss'], error+entry['ref_error'],
                error+entry['oracle_error'],entry['parameter_error'],entry['risk_error']))
        keys = ('population_A','population_B','population_correction','population_error_A',
                'population_error_B','oracle_parameter_error_estimate','oracle_risk_suboptimality_estimate')
        result = dict(zip(keys,np.array(rows).T))
        result['population_cum_A'] = np.cumsum(result['population_A'])
        result['population_cum_B'] = np.cumsum(result['population_B'])
        if not all(np.isfinite(v).all() for v in result.values()):
            raise ValueError('nonfinite population evaluation')
        return result
