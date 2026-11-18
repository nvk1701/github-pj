"""Independent cubature, optimizer and MC checks for the CP02 protocol."""

import unittest
from types import SimpleNamespace
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from core.convex_eval import evaluate_history
from core.convex_population import gaussian_population_loss_gradient, gaussian_population_oracle
from core.schedules import ConstantSchedule, ConvexSchedule
from core.trainer import OnlineSGDTrainer
from core.utils import load_config
from experiments.convex import ROOT
from experiments.convex_diagnostics import _DiameterBoundSchedule, run_diagnostics, validate_protocol_config


def cubature(reference, noise, order=40):
    """Independent full two-feature plus label-noise Gauss-Hermite grid."""
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    z = np.sqrt(2)*nodes
    x1, x2, epsilon = np.meshgrid(z, z, z, indexing='ij')
    X = np.column_stack((x1.ravel(), x2.ravel()))
    y = expit(X@reference + noise*epsilon.ravel())
    probability = (weights[:,None,None]*weights[None,:,None]*weights[None,None,:]/np.pi**1.5).ravel()
    return X, y, probability


class PopulationTests(unittest.TestCase):
    """Check the Gaussian reduction without reusing its one-dimensional code."""

    def test_population_risk_gradient_against_full_cubature(self):
        reference = np.array([.7, -.4])
        theta = np.array([-.2, .9])
        X, y, probability = cubature(reference, .6)
        z = X@theta
        expected_loss = probability@(np.logaddexp(0,z)-y*z)
        expected_gradient = X.T@(probability*(expit(z)-y))
        loss, gradient, errors = gaussian_population_loss_gradient(theta, reference, .6)
        self.assertAlmostEqual(loss, expected_loss, places=10)
        np.testing.assert_allclose(gradient, expected_gradient, atol=1e-10)
        self.assertLess(errors['loss'], 1e-9)
        self.assertLess(errors['gradient'], 1e-9)

    def test_gradient_finite_difference_and_smoothness(self):
        theta, reference = np.array([.3,-.5]), np.array([1.,0.])
        delta = 1e-5
        numerical, hessian = [], []
        for axis in np.eye(2):
            a, ga, _ = gaussian_population_loss_gradient(theta+delta*axis,reference,.8)
            b, gb, _ = gaussian_population_loss_gradient(theta-delta*axis,reference,.8)
            numerical.append((a-b)/(2*delta))
            hessian.append((ga-gb)/(2*delta))
        _, gradient, _ = gaussian_population_loss_gradient(theta,reference,.8)
        np.testing.assert_allclose(gradient,numerical,atol=1e-10)
        eigenvalues = np.linalg.eigvalsh(np.array(hessian).T)
        self.assertGreater(eigenvalues.min(),0)
        self.assertLessEqual(eigenvalues.max(),.25+1e-10)

    def test_radial_oracle_matches_unrestricted_2d_optimizer(self):
        reference = np.array([1.,-.5])
        X, y, probability = cubature(reference,1.,order=50)
        def objective(theta):
            z = X@theta
            return (float(probability@(np.logaddexp(0,z)-y*z)),
                    X.T@(probability*(expit(z)-y)))
        result = minimize(objective, np.zeros(2), jac=True, method='BFGS',tol=1e-10)
        oracle = gaussian_population_oracle(reference,1.)
        np.testing.assert_allclose(oracle['theta'],result.x,atol=1e-7)
        self.assertLess(np.linalg.norm(objective(oracle['theta'])[1]),1e-8)
        self.assertLess(objective(oracle['theta'])[0],objective(reference)[0])
        self.assertLess(oracle['parameter_error_estimate'],1e-8)

    def test_noise_zero_origin_symmetry_and_shrinkage(self):
        reference = np.array([1.,-.5])
        np.testing.assert_array_equal(gaussian_population_oracle(reference,0)['theta'],reference)
        np.testing.assert_array_equal(gaussian_population_oracle([0.,0.],2)['theta'],[0,0])
        norms = [gaussian_population_oracle(reference,s)['radial_norm'] for s in (0,.1,.5,1,2)]
        self.assertTrue(np.all(np.diff(norms)<0))
        positive = gaussian_population_oracle(reference,1)['theta']
        negative = gaussian_population_oracle(-reference,1)['theta']
        np.testing.assert_array_equal(positive,-negative)
        _, reference_gradient, _ = gaussian_population_loss_gradient(reference,reference,1)
        self.assertGreater(float(reference_gradient@reference),.01)

    def test_oracle_error_convergence_and_domain_dimensions(self):
        for d in (2,8,32,128,512):
            reference = np.zeros(d)
            reference[::2] = 1
            coarse = gaussian_population_oracle(reference,np.sqrt(.1),tolerance=1e-8)
            fine = gaussian_population_oracle(reference,np.sqrt(.1),tolerance=1e-12)
            difference = np.linalg.norm(coarse['theta']-fine['theta'])
            self.assertLessEqual(difference,coarse['parameter_error_estimate']+1e-8)
            self.assertLessEqual(np.linalg.norm(fine['theta']),d/2)
            _, gradient, errors = gaussian_population_loss_gradient(fine['theta'],reference,np.sqrt(.1))
            self.assertLess(np.linalg.norm(gradient)+errors['gradient'],1e-9)

    def test_paper_variance_profile_is_not_uniform_bound(self):
        # Integrate along the reference axis and use E||X_perp||^2=d-1.
        d = 8
        nodes, weights = np.polynomial.hermite.hermgauss(100)
        z, epsilon = np.sqrt(2)*nodes[:,None], np.sqrt(2)*nodes[None,:]
        probability = weights[:,None]*weights[None,:]/np.pi
        residual = expit(-4*z)-expit(2*z+np.sqrt(.1)*epsilon)
        second_moment = np.sum(probability*(z*z+d-1)*residual**2)
        mean = np.sum(probability*z*residual)
        variance = second_moment-mean**2
        self.assertGreater(variance,d/4)
        self.assertLess(variance,d)

    def test_population_input_guards(self):
        for ref, noise in (([],1),([np.nan],1),([1],-1),([1],np.inf),([1],True)):
            with self.subTest(ref=ref,noise=noise),self.assertRaises((ValueError,TypeError)):
                gaussian_population_oracle(ref,noise)
        with self.assertRaises(ValueError):
            gaussian_population_loss_gradient([0,0],[1],.1)
        with self.assertRaises(ValueError):
            gaussian_population_oracle([1],.1,tolerance=0)

    def test_iid_batch_variance_scales_as_inverse_batch_size(self):
        rng = np.random.default_rng(77)
        X = rng.normal(size=(65536,2))
        y = expit(X[:,0]+rng.normal(size=len(X)))
        gradient = X*(expit(.2*X[:,0]-.1*X[:,1])-y)[:,None]
        single = np.var(gradient,axis=0,ddof=1).sum()
        for batch in (4,16):
            variance = np.var(gradient.reshape(-1,batch,2).mean(axis=1),axis=0,ddof=1).sum()
            self.assertAlmostEqual(variance*batch/single,1,delta=.1)


class ComparatorEvaluationTests(unittest.TestCase):
    """Keep reference metrics stable while adding separately named oracle metrics."""

    def test_optional_oracle_keeps_all_reference_metrics_and_rng(self):
        reference = np.tile([1.,0.],(8,1))
        history = OnlineSGDTrainer(2,ConstantSchedule(.1),loss_type='logistic',batch_size=16,
                                  noise_std=1,max_norm=1,use_wandb=False).run(reference,seed=3)
        path = np.tile(gaussian_population_oracle([1.,0.],1)['theta'],(8,1))
        kwargs = dict(batch_size=16,label_noise_std=1,eval_batch_size=128,eval_seed=71)
        before = evaluate_history(history,**kwargs)
        after = evaluate_history(history,approx_oracle_path=path,**kwargs)
        for key in before:
            np.testing.assert_array_equal(before[key],after[key])
        self.assertIn('eval_cum_mc_approx_oracle_excess',after)
        self.assertTrue(all(np.isfinite(v).all() for v in after.values()))
        np.testing.assert_array_equal(after['eval_cum_mc_approx_oracle_excess'],
                                      np.cumsum(after['eval_mc_approx_oracle_excess']))
        with self.assertRaises(ValueError):
            evaluate_history(history,approx_oracle_path=np.zeros((7,2)),**kwargs)

    def test_independent_mc_agrees_with_population_gap(self):
        reference = np.tile([1.,0.],(4,1))
        theta = np.tile([.2,.1],(4,1))
        from core.convex_eval import logistic_loss_gradient, sample_logistic_batch
        rng = np.random.default_rng(3)
        losses = [logistic_loss_gradient(w,*sample_logistic_batch(ref,8,1,rng))[0]
                  for w,ref in zip(theta,reference)]
        history = SimpleNamespace(theta_path=theta,theta_star_path=reference,
                                  batch_loss=np.array(losses),metadata={'seed':3})
        oracle = gaussian_population_oracle(reference[0],1)['theta']
        metrics = evaluate_history(history,batch_size=8,label_noise_std=1,eval_batch_size=8192,
                                   eval_seed=91,approx_oracle_path=np.tile(oracle,(4,1)))
        expected = (gaussian_population_loss_gradient(theta[0],reference[0],1)[0]
                    - gaussian_population_loss_gradient(oracle,reference[0],1)[0])
        error = abs(metrics['eval_mc_approx_oracle_excess'].mean()-expected)
        se = np.sqrt(np.sum(metrics['eval_mc_approx_oracle_excess_se']**2))/4
        self.assertLess(error,5*se+1e-10)

    def test_diameter_signal_is_future_free_and_distinct_from_trainer_ema(self):
        a,b = _DiameterBoundSchedule(.25,2,2),_DiameterBoundSchedule(.25,2,2)
        first = [a.step(t,gamma_prev=0,batch_size=32) for t in range(1,5)]
        second = [b.step(t,gamma_prev=100*t,batch_size=32) for t in range(1,5)]
        np.testing.assert_array_equal(first,second)
        expected = ConvexSchedule(.25,2,2)
        self.assertEqual(first[0],expected.step(1,gamma=2,batch_size=32))

    def test_protocol_guards(self):
        cfg = load_config(ROOT/'configs/convex.yaml')
        cfg.update(steps=8,dimensions=[2])
        with self.assertRaises(ValueError):
            validate_protocol_config(cfg,gamma_mode='diameter_bound',variance_profile='paper')
        safe = validate_protocol_config(cfg,gamma_mode='diameter_bound',variance_profile='safe')
        self.assertEqual(safe['gradient_variance_per_dimension'],1)
        self.assertEqual(cfg['gradient_variance_per_dimension'],.25)
        for override in ({'steps':2000},{'dimensions':[128]},{'mode':'online'},{'L':.1}):
            with self.assertRaises(ValueError):
                validate_protocol_config(dict(cfg,**override))

    def test_diagnostic_exports_and_repeatability(self):
        cfg = load_config(ROOT/'configs/convex.yaml')
        with tempfile.TemporaryDirectory() as temp:
            cfg.update(steps=4,dimensions=[2],batch_size=8,eval_batch_size=32,
                       seeds=[0],methods=['convex'],mode='disabled',output_dir=str(Path(temp)/'first'))
            with patch.dict('os.environ',{'WANDB_MODE':'disabled'}):
                first = run_diagnostics(cfg)
                with self.assertRaises(FileExistsError):
                    run_diagnostics(cfg)
                cfg['output_dir'] = str(Path(temp)/'second')
                second = run_diagnostics(cfg)
            self.assertEqual(first[0]['final_reference_mc'],second[0]['final_reference_mc'])
            with np.load(Path(temp)/'first/d2/convex/seed0/history.npz') as data:
                self.assertTrue(all(np.isfinite(data[k]).all() for k in data.files))
                self.assertIn('approx_oracle',data.files)
