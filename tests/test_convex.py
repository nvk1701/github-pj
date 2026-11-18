"""Kiem cong thuc, y nghia metric va integration khong can W&B online."""

import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit

from core.convex_eval import evaluate_history, logistic_loss_gradient, sample_logistic_batch
from core.drift import EMADriftEstimator, generate_spiral_trajectory
from core.schedules import ConstantSchedule, ConvexSchedule, InverseSqrtSchedule
from core.trainer import OnlineSGDTrainer
from core.utils import load_config
from experiments.convex import ROOT, run_experiments, validate_config


class ScheduleTests(unittest.TestCase):
    """Doi chieu equations doc lap, bien so hoc va state cua schedule."""

    def test_original_threshold_formula(self):
        for L, D, v, B, g in [(0.25, 2, 0.5, 256, 0.01), (2, 3, 5, 7, 2), (1, .1, .5, 1, 0)]:
            with self.subTest(L=L, D=D, v=v, B=B, g=g):
                s = ConvexSchedule(L, D, v, min(.1, 1/L))
                b = np.array([g*g + 2*D*g, (g+D)**2])
                expected = B/(2*v) * (np.sqrt(b*b*L*L + 4*v*b/B) - b*L)
                np.testing.assert_allclose(s.thresholds(g, B), expected, rtol=2e-11, atol=1e-14)

    def test_three_cases(self):
        reference = ConvexSchedule(1, 1, 10, .1)
        lo, hi = reference.thresholds(.02, 1)
        for previous, expected in [(lo/2, lo), ((lo+hi)/2, (lo+hi)/2), ((hi+1)/2, hi)]:
            s = ConvexSchedule(1, 1, 10, previous)
            self.assertAlmostEqual(s.step(1, gamma=.02, batch_size=1), expected)

    def test_minimizes_equation17_independently(self):
        for g in (0, .001, .2, 2):
            for previous in (.01, .3, .9):
                s = ConvexSchedule(1, 1, 4, previous)
                eta = s.step(1, gamma=g, batch_size=3)
                def objective(e):
                    a = 2*e-e*e
                    return max(1/a-1/(2*previous-previous**2), 0) + (4*e*e/3+g*g+2*g)/a
                # Kink tai previous: toi uu tung doan rieng va kiem ca kink.
                candidates = [previous, 1.0]
                for low, high in [(1e-12, previous), (previous, 1.0)]:
                    if high > low:
                        candidates.append(minimize_scalar(objective, bounds=(low,high), method="bounded").x)
                self.assertLessEqual(objective(eta), min(map(objective,candidates))+1e-7)

    def test_shift_and_batch_monotonicity(self):
        s = ConvexSchedule(.25, 8, 2, .1)
        thresholds = np.array([s.thresholds(g, 256) for g in (0, 1e-12, .001, .1, 1, 1e6)])
        self.assertTrue(np.all(np.diff(thresholds, axis=0) >= -1e-14))
        values = [s.thresholds(.1, B) for B in (1, 8, 256)]
        self.assertTrue(np.all(np.diff(values, axis=0) >= 0))
        rates = [ConvexSchedule(.25,8,2,.1).step(1,gamma=g,batch_size=256) for g in (0,.1,1)]
        self.assertTrue(np.all(np.diff(rates) >= 0))

    def test_dimension_profiles(self):
        for d in (2,8,32,128,512):
            s = ConvexSchedule(.25, d, d/4, 1)
            self.assertTrue(0 < s.step(1, gamma=.1, batch_size=256) <= 4)

    def test_numerical_cancellation_small_gamma(self):
        s = ConvexSchedule(.25, 2, .5, 1)
        for g in (1e-300, 1e-100, 1e-12, 1e100, 1e300):
            lo, hi = s.thresholds(g, 256)
            self.assertTrue(math.isfinite(lo) and math.isfinite(hi))
            self.assertTrue(0 <= lo <= hi <= 4)
        self.assertGreater(s.thresholds(1e-300,256)[0], 0)

    def test_extreme_threshold_against_decimal(self):
        from decimal import Decimal, localcontext
        with localcontext() as context:
            context.prec = 100
            L, D, v, g, B = map(Decimal, ['0.25','2','0.5','1e-50','256'])
            b = g*g+2*D*g
            expected = 2*b / ((b*b*L*L+4*v*b/B).sqrt()+b*L)
            actual = ConvexSchedule(float(L),float(D),float(v)).thresholds(float(g),int(B))[0]
            self.assertAlmostEqual(actual/float(expected), 1, places=12)

    def test_zero_variance_limits(self):
        s = ConvexSchedule(.25, 2, 0, .5)
        self.assertEqual(s.thresholds(.1,256), (4,4))
        self.assertEqual(s.step(1, gamma=.1, batch_size=256), 4)
        s.reset()
        self.assertEqual(s.thresholds(0,256), (0,4))
        self.assertEqual(s.step(1,gamma=0,batch_size=256), .5)

    def test_reset_and_time_validation(self):
        s = ConvexSchedule(.25,2,.5)
        first = s.step(1,gamma=.1,batch_size=16)
        s.step(2,gamma=1,batch_size=8)
        s.reset()
        self.assertEqual(s.eta_prev,1)
        self.assertEqual(first,s.step(1,gamma=.1,batch_size=16))
        for t in (1,3,0,-1,True,1.5):
            with self.assertRaises(ValueError): s.step(t,gamma=0,batch_size=16)

    def test_invalid_constructor(self):
        base = dict(L=.25,D_max=2,gradient_variance=.5,eta_init=1)
        for key, values in {"L":[0,-1,float('nan'),float('inf'),True],
                            "D_max":[0,-1,float('nan')],
                            "gradient_variance":[-1,float('inf'),True],
                            "eta_init":[0,-1,5,float('nan'),True]}.items():
            for value in values:
                with self.subTest(key=key,value=value), self.assertRaises((ValueError,TypeError)):
                    ConvexSchedule(**dict(base,**{key:value}))

    def test_invalid_step_preserves_state(self):
        for kwargs in ({}, {'gamma':-1,'batch_size':1}, {'gamma':float('nan'),'batch_size':1},
                       {'gamma':0,'batch_size':0}, {'gamma':0,'batch_size':True},
                       {'gamma':0,'batch_size':1.5}, {'gamma':0,'gamma_prev':0,'batch_size':1}):
            s = ConvexSchedule(.25,2,.5)
            with self.assertRaises((ValueError,TypeError)): s.step(1,**kwargs)
            self.assertEqual(s.last_t,0)
            self.assertEqual(s.eta_prev,1)

    def test_current_loss_and_label_noise_are_not_scheduler_inputs(self):
        a,b=ConvexSchedule(.25,2,.5),ConvexSchedule(.25,2,.5)
        self.assertEqual(a.step(1,gamma_prev=.1,batch_size=16,noise_std=.1,loss_val=.1),
                         b.step(1,gamma_prev=.1,batch_size=16,noise_std=100,loss_val=100))


class LogisticTests(unittest.TestCase):
    """Kiem BCE, sampler va phan biet generating reference voi minimizer."""

    def test_gradient_finite_difference_and_shared_formula(self):
        rng=np.random.default_rng(12)
        w=np.array([.1,-.2,.3]); X=rng.normal(size=(10,3)); y=rng.uniform(size=10)
        loss,grad=logistic_loss_gradient(w,X,y)
        numerical=np.empty(3)
        for j in range(3):
            delta=np.eye(3)[j]*1e-6
            numerical[j]=(logistic_loss_gradient(w+delta,X,y)[0]-logistic_loss_gradient(w-delta,X,y)[0])/2e-6
        np.testing.assert_allclose(grad,numerical,atol=1e-9)
        np.testing.assert_allclose(grad,X.T@(expit(X@w)-y)/10)
        self.assertEqual(grad.shape,w.shape)
        self.assertTrue(math.isfinite(loss))

    def test_bce_extreme_logits(self):
        loss,grad=logistic_loss_gradient([1],[[1000],[-1000]],[1,0])
        self.assertEqual(loss,0)
        self.assertTrue(np.isfinite(grad).all())
        loss,_=logistic_loss_gradient([1],[[1000],[-1000]],[0,1])
        self.assertEqual(loss,1000)

    def test_invalid_logistic_inputs(self):
        for args in (([0],[[1]],[2]),([0],[[float('nan')]],[0]),([0],[[1,2]],[0]),([0],np.empty((0,1)),[])):
            with self.assertRaises(ValueError): logistic_loss_gradient(*args)
        for std in (-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError): sample_logistic_batch([1],4,std,np.random.default_rng(0))

    def test_noiseless_reference_is_population_minimizer(self):
        X,y=sample_logistic_batch(np.array([1.,-.5]),100,0,np.random.default_rng(0))
        _,grad=logistic_loss_gradient([1,-.5],X,y)
        np.testing.assert_allclose(grad,0,atol=1e-16)

    def test_noisy_reference_is_not_population_minimizer(self):
        # Gauss-Hermite doc lap, khong can training hoac toi uu lon.
        nodes,weights=np.polynomial.hermite.hermgauss(50)
        x=np.sqrt(2)*nodes[:,None]; eps=np.sqrt(2)*nodes[None,:]
        prob=weights[:,None]*weights[None,:]/np.pi
        y=expit(x+eps)
        grad=np.sum(prob*x*(expit(x)-y))
        self.assertGreater(grad,.02)
        def risk(w): return np.sum(prob*(np.logaddexp(0,w*x)-y*w*x))
        self.assertLess(risk(.95),risk(1))

    def test_signed_gap_and_clipping_bias(self):
        gaps=[]
        for y in (.2,.9):
            learner,_=logistic_loss_gradient([0],[[1]],[y])
            reference,_=logistic_loss_gradient([1],[[1]],[y])
            gaps.append(learner-reference)
        self.assertLess(gaps[0],0)
        self.assertGreater(np.mean(np.maximum(gaps,0)),np.mean(gaps))

    def test_generator_matches_shared_trainer(self):
        w=np.array([.2,-.1]); ref=np.array([1.,0.])
        X,y=sample_logistic_batch(ref,16,.3,np.random.default_rng(4))
        loss,grad=logistic_loss_gradient(w,X,y)
        tr=OnlineSGDTrainer(2,ConstantSchedule(.1),loss_type="logistic",batch_size=16,noise_std=.3)
        shared_grad,shared_loss,_=tr._sample_batch_and_compute_grad(w,ref,np.random.default_rng(4))
        np.testing.assert_allclose(grad,shared_grad,atol=1e-14)
        self.assertAlmostEqual(loss,shared_loss)


class IntegrationTests(unittest.TestCase):
    """Kiem shared trainer, tinh nhan qua, evaluation va export evidence."""

    def history(self,schedule=None,seed=3,oracle=False):
        ref,gamma=generate_spiral_trajectory(16,2,ell=4)
        trainer=OnlineSGDTrainer(2,schedule or ConvexSchedule(.25,2,.5),loss_type="logistic",
            batch_size=8,noise_std=.3,max_norm=1,use_oracle_gamma=oracle,use_wandb=False)
        return trainer.run(ref,gamma,seed=seed),gamma

    def test_reproducibility_reset_and_shapes(self):
        schedule=ConvexSchedule(.25,2,.5)
        a,_=self.history(schedule)
        self.history(schedule,seed=99)
        b,_=self.history(schedule)
        np.testing.assert_array_equal(a.theta_path,b.theta_path)
        np.testing.assert_array_equal(a.lr_history,b.lr_history)
        self.assertEqual(a.theta_path.shape,(16,2))
        self.assertTrue(np.isfinite(a.theta_path).all())
        self.assertTrue(np.all(np.linalg.norm(a.theta_path,axis=1)<=1+1e-12))
        self.assertEqual(a.theta_path[0].tolist(),[0,0])

    def test_ema_causality(self):
        h,_=self.history()
        expected=[]; ema=EMADriftEstimator()
        for w in h.theta_path: expected.append(ema.update(w))
        np.testing.assert_array_equal(h.gamma_used,expected)

    def test_lagged_reference_indexing_not_gamma_t(self):
        h,gamma=self.history(oracle=True)
        np.testing.assert_array_equal(h.gamma_used,np.r_[0,gamma[:-1]])
        self.assertFalse(np.array_equal(h.gamma_used,gamma))

    def test_future_reference_does_not_change_prefix(self):
        ref,_=generate_spiral_trajectory(20,2,ell=1)
        altered=ref.copy(); altered[10:]*=-1
        def run(path):
            return OnlineSGDTrainer(2,ConvexSchedule(.25,2,.5),loss_type="logistic",batch_size=8,
                use_oracle_gamma=False,max_norm=1).run(path,seed=0)
        a,b=run(ref),run(altered)
        np.testing.assert_array_equal(a.lr_history[:10],b.lr_history[:10])
        np.testing.assert_array_equal(a.theta_path[:11],b.theta_path[:11])

    def test_same_seed_common_training_and_evaluation(self):
        common=[]
        def capture(self_tr,w,ref,rng):
            X,y=sample_logistic_batch(ref,self_tr.batch_size,self_tr.noise_std,rng)
            common[-1].append((X.copy(),y.copy()))
            loss,grad=logistic_loss_gradient(w,X,y)
            return grad,loss,0
        for schedule in (ConstantSchedule(.1),InverseSqrtSchedule(.1),ConvexSchedule(.25,2,.5)):
            common.append([])
            with patch.object(OnlineSGDTrainer,"_sample_batch_and_compute_grad",capture): self.history(schedule)
        for stream in common[1:]:
            for a,b in zip(common[0],stream):
                np.testing.assert_array_equal(a[0],b[0]); np.testing.assert_array_equal(a[1],b[1])
        h,_=self.history()
        a=evaluate_history(h,batch_size=8,label_noise_std=.3,eval_batch_size=32,eval_seed=71)
        b=evaluate_history(h,batch_size=8,label_noise_std=.3,eval_batch_size=32,eval_seed=71)
        for key in a: np.testing.assert_array_equal(a[key],b[key])
        np.testing.assert_allclose(a['eval_cum_mc_excess'],np.cumsum(a['eval_mc_excess']))

    def test_evaluator_independence_and_signed_exports(self):
        h,_=self.history()
        before=h.theta_path.copy()
        a=evaluate_history(h,batch_size=8,label_noise_std=.3,eval_batch_size=32,eval_seed=7)
        b=evaluate_history(h,batch_size=8,label_noise_std=.3,eval_batch_size=32,eval_seed=8)
        np.testing.assert_array_equal(h.theta_path,before)
        np.testing.assert_array_equal(a['train_empirical_excess'],b['train_empirical_excess'])
        self.assertFalse(np.array_equal(a['eval_mc_excess'],b['eval_mc_excess']))
        with self.assertRaises(ValueError):
            evaluate_history(h,batch_size=9,label_noise_std=.3,eval_batch_size=32,eval_seed=7)

    def test_linear_analytic_metric_and_baselines_unchanged(self):
        ref,gamma=generate_spiral_trajectory(8,2)
        for schedule in (ConstantSchedule(.1),InverseSqrtSchedule(.1)):
            h=OnlineSGDTrainer(2,schedule,batch_size=8).run(ref,gamma,seed=0)
            np.testing.assert_allclose(h.inst_regret,.5*np.sum((h.theta_path-ref)**2,axis=1))
            np.testing.assert_allclose(h.cum_regret,np.cumsum(h.inst_regret))

    def test_runner_schema_and_projection_guards(self):
        cfg=load_config(ROOT/'configs/convex.yaml')
        validate_config(cfg)
        for override in ({'mode':'online'},{'seeds':[True]}, {'dimensions':[1]}, {'batch_size':0},
                         {'projection_radius_per_dimension':1}, {'label_noise_std':-1},
                         {'methods':['typo']}, {'seeds':[0,0]}, {'eta_init':0}, {'ema_beta':1}):
            with self.subTest(override=override),self.assertRaises((ValueError,TypeError)):
                validate_config(dict(cfg,**override))

    def test_runner_small_multiseed_output(self):
        cfg=load_config(ROOT/'configs/convex.yaml')
        with tempfile.TemporaryDirectory() as temp:
            cfg.update(steps=8,dimensions=[2],batch_size=8,eval_batch_size=16,seeds=[0,1],output_dir=temp)
            with patch.dict('os.environ',{'MPLCONFIGDIR':str(Path(temp)/'mpl'),'WANDB_MODE':'disabled'}):
                runs=run_experiments(cfg)
            self.assertEqual(len(runs),6)
            for run in runs:
                p=Path(run['output_dir'])
                for name in ('config.json','metrics.csv','history.npz','summary.json','figure.png'):
                    self.assertTrue((p/name).is_file())
                resolved=json.loads((p/'config.json').read_text(encoding='utf-8'))
                self.assertIsInstance(resolved['source_uncommitted'],bool)
                self.assertEqual(len(resolved['head']),40)
                self.assertIn('experiments/convex.py',resolved['source_sha256'])
                with np.load(p/'history.npz') as data:
                    self.assertEqual(data['theta'].shape,(8,2))
                    self.assertTrue(all(np.isfinite(data[key]).all() for key in data.files))
                    self.assertNotIn('inst_regret',data.files)
            aggregate=json.loads((Path(temp)/'summary.json').read_text())['aggregate']
            self.assertTrue(all(row['n_seeds']==2 for row in aggregate))

    def test_runner_qa_preserves_logging_error_when_cleanup_fails(self):
        cfg = load_config(ROOT/'configs/convex.yaml')
        run = Mock(tags=['lab1'])
        run.log.side_effect = RuntimeError('original logging failure')
        run.finish.side_effect = OSError('cleanup failure')
        with tempfile.TemporaryDirectory() as temp:
            cfg.update(steps=1, dimensions=[2], batch_size=8, eval_batch_size=8,
                       seeds=[0], methods=['constant'], output_dir=temp)
            with patch('experiments.convex.init_run', return_value=run), patch('experiments.convex._plot'):
                with self.assertRaisesRegex(RuntimeError, 'original logging failure'):
                    run_experiments(cfg)
            run.finish.assert_called_once_with(exit_code=1)

    def test_runner_qa_refuses_existing_summary_before_training(self):
        cfg = load_config(ROOT/'configs/convex.yaml')
        with tempfile.TemporaryDirectory() as temp:
            summary = Path(temp)/'summary.json'
            original = b'{"previous_experiment": true}\n'
            summary.write_bytes(original)
            cfg.update(steps=1, dimensions=[2], batch_size=8, eval_batch_size=8,
                       seeds=[99], methods=['constant'], output_dir=temp)
            with patch.object(OnlineSGDTrainer, 'run', side_effect=AssertionError('training started before collision check')) as train:
                with self.assertRaises(FileExistsError):
                    run_experiments(cfg)
                train.assert_not_called()
            self.assertEqual(summary.read_bytes(), original)
            self.assertFalse((Path(temp)/'d2').exists())


if __name__ == '__main__':
    unittest.main()
