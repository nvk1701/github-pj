"""CP03 manifest, cached numerical equivalence and release-path regressions."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch

import numpy as np

from core.convex_population import PopulationEvaluator,gaussian_population_loss_gradient
from core.utils import load_config
from experiments.convex_official import ROOT,build_plan,main,run_official,validate_official


class OfficialTests(unittest.TestCase):
    """Keep the locked main profile separate from small, labelled pilot runs."""

    def config(self):
        return load_config(ROOT/'configs/convex_cp03.yaml')

    def test_manifest_45_unique_and_dry_run_has_no_side_effects(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'not-created'
            cfg=dict(self.config(),output_dir=str(output))
            plan=build_plan(cfg)
            self.assertEqual(plan['run_count'],45)
            self.assertEqual(len({r['key'] for r in plan['runs']}),45)
            self.assertEqual(len({r['path'] for r in plan['runs']}),45)
            with patch('experiments.convex_official.init_run') as tracking,patch('experiments.convex_official.OnlineSGDTrainer.run') as train:
                with contextlib.redirect_stdout(io.StringIO()) as capture:
                    main(['--dry-run','--output-dir',str(output)])
                self.assertEqual(json.loads(capture.getvalue())['run_count'],45)
                tracking.assert_not_called()
                train.assert_not_called()
            self.assertFalse(output.exists())

    def test_frozen_config_rejects_missing_conflicting_or_unknown_keys(self):
        cfg=self.config()
        for key in ('protocol_version','steps','label_noise_std','log_interval'):
            changed=dict(cfg)
            del changed[key]
            with self.subTest(key=key),self.assertRaises((ValueError,KeyError)):
                validate_official(changed)
        for override in ({'steps':8},{'gradient_variance_per_dimension':1},{'shift_mode':'lagged_reference'},
                         {'constant_lr':.3},{'seeds':[100]},{'typo':1},{'variance_profile':'safe'},
                         {'mode':'typo'},{'log_interval':False},{'oracle_tolerance':float('nan')}):
            with self.subTest(override=override),self.assertRaises((ValueError,TypeError)):
                validate_official(dict(cfg,**override))

    def test_pilot_and_sensitivity_deviations_are_explicit(self):
        cfg=self.config()
        validate_official(dict(cfg,run_category='pilot',steps=8,batch_size=8,eval_batch_size=16))
        validate_official(dict(cfg,run_category='sensitivity',ell=1,label_noise_std=1))
        with self.assertRaises(ValueError):
            validate_official(dict(cfg,run_category='sensitivity',constant_lr=.3))

    def test_cache_matches_uncached_population_and_correction(self):
        rng=np.random.default_rng(4)
        for d in (2,128,512):
            ref=np.tile(rng.normal(size=d)/np.sqrt(d),(6,1))
            evaluator=PopulationEvaluator(ref,np.sqrt(.1))
            self.assertEqual(len(evaluator.cache),1)
            corrections=[]
            for amplitude in (.1,.5):
                theta=amplitude*rng.normal(size=ref.shape)
                values=evaluator.evaluate(theta)
                for i in range(len(ref)):
                    loss,_,a=gaussian_population_loss_gradient(theta[i],ref[i],np.sqrt(.1))
                    ref_loss,_,b=gaussian_population_loss_gradient(ref[i],ref[i],np.sqrt(.1))
                    oracle_loss,_,c=gaussian_population_loss_gradient(evaluator.oracle_path[i],ref[i],np.sqrt(.1))
                    self.assertAlmostEqual(values['population_A'][i],loss-ref_loss,delta=1e-9+a['loss']+b['loss'])
                    self.assertAlmostEqual(values['population_B'][i],loss-oracle_loss,delta=1e-9+a['loss']+c['loss'])
                np.testing.assert_allclose(values['population_B']-values['population_A'],values['population_correction'],atol=1e-12)
                np.testing.assert_array_equal(values['population_cum_B'],np.cumsum(values['population_B']))
                self.assertTrue(np.isfinite(values['population_cum_B']).all())
                corrections.append(values['population_correction'])
            np.testing.assert_array_equal(*corrections)

    def test_cached_noiseless_oracle_and_shape_guard(self):
        ref=np.tile([1.,0.],(4,1))
        evaluator=PopulationEvaluator(ref,0)
        np.testing.assert_array_equal(evaluator.oracle_path,ref)
        values=evaluator.evaluate(np.zeros_like(ref))
        np.testing.assert_array_equal(values['population_A'],values['population_B'])
        with self.assertRaises(ValueError):
            evaluator.evaluate(np.zeros((3,2)))

    def test_runner_failure_is_failed_and_preserves_original_exception(self):
        cfg=self.config()
        fake=Mock(id='failed-test-run',tags=['lab1'],settings=Mock(base_url='https://api.forge.coreweave.com'))
        fake.log.side_effect=RuntimeError('logging failed')
        fake.finish.side_effect=OSError('cleanup failed')
        fake.summary={}
        with tempfile.TemporaryDirectory() as temp:
            cfg.update(run_category='pilot',steps=2,dimensions=[2],seeds=[0],methods=['constant'],
                       batch_size=8,eval_batch_size=8,mode='disabled',output_dir=str(Path(temp)/'failed'))
            with patch('experiments.convex_official.init_run',return_value=fake):
                with self.assertRaisesRegex(RuntimeError,'logging failed'):
                    run_official(cfg)
            status=json.loads((Path(cfg['output_dir'])/'d2/constant/seed0/status.json').read_text())
            self.assertEqual(status['local_status'],'failed')
            self.assertFalse(fake.summary['completed'])
            self.assertFalse((Path(cfg['output_dir'])/'summary.json').exists())

    def test_exported_curves_mapping_and_reproducibility(self):
        cfg=self.config()
        with tempfile.TemporaryDirectory() as temp:
            cfg.update(run_category='pilot',steps=4,dimensions=[128],seeds=[0],methods=['convex'],
                       batch_size=8,eval_batch_size=16,mode='disabled',output_dir=str(Path(temp)/'first'))
            with patch.dict('os.environ',{'WANDB_DATA_DIR':str(Path(temp)/'sdk_data')}):
                a=run_official(cfg)
                with self.assertRaises(FileExistsError):
                    run_official(cfg)
                cfg['output_dir']=str(Path(temp)/'second')
                b=run_official(cfg)
            self.assertEqual(a[0]['final_population_B'],b[0]['final_population_B'])
            root=Path(temp)/'first'
            for name in ('figure5_left.png','figure5_right.png','baseline_comparison.png','population_curves.csv','trajectory_d128.csv','sheet4_mapping.csv'):
                self.assertTrue((root/name).is_file(),name)
            curve=np.genfromtxt(root/'population_curves.csv',delimiter=',',names=True,dtype=None,encoding='utf-8')
            with np.load(root/'d128/convex/seed0/history.npz') as data:
                np.testing.assert_array_equal(curve['mean'],data['population_cum_B'])
            self.assertFalse(a[0]['remote_verified'])

    def test_online_summary_api_requires_a_mapping(self):
        class SDKSummary(dict):
            def update(self,payload):
                super().update(payload)
        fake=Mock(id='test-run',entity='gr_ml_29_37_37',project='ml-distribution-shift',
                  tags=['lab1'],settings=Mock(base_url='https://api.forge.coreweave.com'))
        fake.summary=SDKSummary()
        cfg=self.config()
        with tempfile.TemporaryDirectory() as temp:
            cfg.update(run_category='pilot',steps=2,dimensions=[2],seeds=[0],methods=['constant'],
                       batch_size=8,eval_batch_size=8,mode='disabled',output_dir=str(Path(temp)/'mapping'))
            with patch('experiments.convex_official.init_run',return_value=fake):
                result=run_official(cfg)
            self.assertEqual(result[0]['local_status'],'complete')
            self.assertTrue(fake.summary['completed'])
