"""Fault injection and synthetic multi-root recovery, never official training."""

import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock,patch

import numpy as np

from core.convex_recovery import atomic_json,canonical_hash,science_hash,seal_payload,select_attempts,validate_attempt
from core.utils import load_config
from experiments.convex_official import ROOT,build_plan,run_official
from experiments.convex_reconcile import aggregate_roots,reconcile


class RecoveryTests(unittest.TestCase):
    """Inject SDK/storage faults and enforce one valid attempt per logical cell."""

    def config(self,output):
        cfg=load_config(ROOT/'configs/convex_cp03.yaml')
        cfg.update(run_category='pilot',steps=2,dimensions=[2],seeds=[0],methods=['constant'],
                   batch_size=8,eval_batch_size=8,mode='disabled',output_dir=str(output))
        return cfg

    def fake_run(self):
        run=Mock(id='test-id',entity='gr_ml_29_37_37',project='ml-distribution-shift',tags=['lab1'],
                 settings=Mock(base_url='https://api.forge.coreweave.com',mode='online'))
        run.url='https://example.invalid/test-id'
        run.summary={}
        return run

    def test_failure_phases_preserve_local_evidence_and_no_false_complete(self):
        scenarios=[('init',ConnectionError('network unavailable')),('logging',RuntimeError('log')),
                   ('finish',OSError('finish')),('summary',TypeError('summary')),
                   ('logging',TimeoutError('timeout')),('logging',KeyboardInterrupt())]
        with tempfile.TemporaryDirectory() as temp:
            for i,(phase,error) in enumerate(scenarios):
                with self.subTest(phase=phase,error=type(error).__name__):
                    output=Path(temp)/str(i)
                    cfg=self.config(output)
                    run=self.fake_run()
                    if phase=='logging': run.log.side_effect=error
                    if phase=='finish': run.finish.side_effect=error
                    if phase=='summary': run.summary=Mock(update=Mock(side_effect=error))
                    with patch('experiments.convex_official.init_run',side_effect=error if phase=='init' else None,return_value=run):
                        with self.assertRaises(type(error)):
                            run_official(cfg)
                    status=json.loads((output/'d2/constant/seed0/status.json').read_text())
                    self.assertEqual(status['local_status'],'failed')
                    self.assertEqual(status['error_stage'],phase)
                    self.assertFalse(status['sdk_finished'])
                    self.assertEqual(status['local_completed'],phase!='init')
                    self.assertTrue(list((output/'d2/constant/seed0').glob('failure-*.json')))
                    events=[json.loads(line)['event'] for line in (output/'attempts.jsonl').read_text().splitlines()]
                    self.assertEqual(events[0],'started')
                    self.assertEqual(events[-1],'failed')
                    if phase!='init': self.assertTrue((output/'d2/constant/seed0/checksums.json').exists())

    def test_upload_failure_preserves_completed_local_compute(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg=self.config(Path(temp)/'upload')
            cfg['mode']='online'
            run=self.fake_run()
            run.log_artifact.side_effect=ConnectionError('upload')
            with patch('experiments.convex_official.init_run',return_value=run),patch('wandb.Artifact'):
                with self.assertRaises(ConnectionError): run_official(cfg)
            status=json.loads((Path(cfg['output_dir'])/'d2/constant/seed0/status.json').read_text())
            self.assertEqual(status['error_stage'],'upload')
            self.assertTrue(status['local_completed'])
            self.assertFalse(status['sdk_finished'])
            run.log_artifact.assert_called_once()

    def test_unwritable_sdk_directory_fails_before_init(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch('experiments.convex_official.check_sdk_directories',side_effect=PermissionError()),patch('experiments.convex_official.init_run') as init:
                with self.assertRaises(PermissionError): run_official(self.config(Path(temp)/'denied'))
                init.assert_not_called()

    def test_atomic_serialization_preserves_previous_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'status.json'
            atomic_json(path,{'state':'running'})
            before=path.read_bytes()
            for value in ({'bad':float('nan')},{'bad':object()}):
                with self.assertRaises((ValueError,TypeError)): atomic_json(path,value)
                self.assertEqual(path.read_bytes(),before)

    def test_middle_batch_failure_and_fresh_retry_numeric_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg=self.config(Path(temp)/'first')
            cfg['methods']=['constant','inverse_sqrt']
            with patch('experiments.convex_official.init_run',side_effect=[self.fake_run(),ConnectionError('second cell')]):
                with self.assertRaises(ConnectionError): run_official(cfg)
            first=Path(cfg['output_dir'])/'d2/constant/seed0'
            original=first.joinpath('history.npz').read_bytes()
            cfg.update(methods=['inverse_sqrt'],output_dir=str(Path(temp)/'retry'))
            with patch('experiments.convex_official.init_run',return_value=self.fake_run()):
                retry=run_official(cfg)
            self.assertEqual(first.joinpath('history.npz').read_bytes(),original)
            cfg.update(methods=['constant','inverse_sqrt'],output_dir=str(Path(temp)/'baseline'))
            with patch('experiments.convex_official.init_run',side_effect=[self.fake_run(),self.fake_run()]):
                baseline=run_official(cfg)
            self.assertEqual(retry[0]['final_population_B'],baseline[1]['final_population_B'])
            plan=build_plan(cfg)
            selected,excluded=select_attempts([Path(temp)/'first',Path(temp)/'retry'],plan,require_remote=False)
            self.assertEqual(len(selected),2)
            self.assertEqual(len(excluded),1)
            with self.assertRaises(ValueError):
                select_attempts([Path(temp)/'first',Path(temp)/'retry',Path(temp)/'baseline'],plan,require_remote=False)

    def make_fixture(self,root,cfg,source,cells):
        root.mkdir()
        cfg=dict(cfg,output_dir=str(root),dimensions=sorted({c['dimension'] for c in cells}))
        manifest=dict(config=cfg,config_sha256=canonical_hash(cfg),source={'head':source},runs=[])
        for index,cell in enumerate(cells):
            directory=root/f"d{cell['dimension']}"/cell['method']/f"seed{cell['seed']}"
            directory.mkdir(parents=True)
            manifest['runs'].append(dict(cell,path=str(directory)))
            attempt=f'synthetic-{root.name}-{index}'
            metadata=dict(cfg,**dict(cell,path=str(directory)),head=source,source_uncommitted=False,
                          source_sha256={},config_sha256=manifest['config_sha256'],science_sha256=science_hash(cfg),attempt_id=attempt)
            atomic_json(directory/'config.json',metadata)
            curve=np.arange(1,cfg['steps']+1,dtype=float)*(cell['seed']+1)
            metrics=dict(step=np.arange(1,cfg['steps']+1),population_cum_B=curve)
            np.savez_compressed(directory/'history.npz',theta=np.zeros((cfg['steps'],cell['dimension'])),
                                reference=np.zeros((cfg['steps'],cell['dimension'])),approx_oracle=np.zeros((cfg['steps'],cell['dimension'])),**metrics)
            with (directory/'metrics.csv').open('w',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(metrics);writer.writerows(zip(*metrics.values()))
            directory.joinpath('figure.png').write_bytes(b'unit-test fixture, not experiment evidence')
            checksums=seal_payload(directory)
            status=dict(cell,path=str(directory),attempt_id=attempt,local_status='complete',local_completed=True,
                        sdk_finished=True,run_id=attempt,wandb_url=None)
            atomic_json(directory/'status.json',status)
            atomic_json(directory/'remote_verification.json',dict(key=cell['key'],attempt_id=attempt,head=source,
                config_sha256=manifest['config_sha256'],science_sha256=science_hash(cfg),run_id=attempt,
                server_finished=True,artifact_uploaded=True,artifact_verified=True,checksums=checksums))
        atomic_json(root/'run_manifest.json',manifest)

    def test_45_cell_union_cross_root_equals_single_root_and_detects_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg=load_config(ROOT/'configs/convex_cp03.yaml')
            plan=build_plan(cfg)
            # Synthetic fixtures validate coverage/aggregation without 45 trainings.
            source=plan['source']['head']
            roots=[]
            for d in cfg['dimensions']:
                root=Path(temp)/f'batch{d}'
                self.make_fixture(root,cfg,source,[c for c in plan['runs'] if c['dimension']==d])
                roots.append(root)
            selected,_=select_attempts(roots,plan)
            self.assertEqual(len(selected),45)
            single=Path(temp)/'single'
            self.make_fixture(single,cfg,source,plan['runs'])
            aggregate_roots(roots,plan,Path(temp)/'merged')
            aggregate_roots([single],plan,Path(temp)/'monolithic')
            self.assertEqual((Path(temp)/'merged/population_curves.csv').read_bytes(),(Path(temp)/'monolithic/population_curves.csv').read_bytes())
            with self.assertRaises(ValueError): select_attempts(roots[:-1],plan)
            with self.assertRaises(ValueError): select_attempts(roots+[single],plan)
            cell=Path(selected[0]['path'])
            (cell/'metrics.csv').write_text('corrupt')
            with self.assertRaises(ValueError): validate_attempt(cell)

    def test_config_sha_source_and_missing_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg=self.config(Path(temp)/'unused')
            plan=build_plan(cfg)
            for kind in ('config','source','csv','npz'):
                root=Path(temp)/kind
                self.make_fixture(root,cfg,plan['source']['head'],plan['runs'])
                cell=root/'d2/constant/seed0'
                if kind in ('csv','npz'):
                    (cell/('metrics.csv' if kind=='csv' else 'history.npz')).rename(cell/'preserved-missing-file')
                else:
                    metadata=json.loads((cell/'config.json').read_text())
                    metadata['config_sha256' if kind=='config' else 'head']='wrong'
                    atomic_json(cell/'config.json',metadata)
                with self.assertRaises((ValueError,FileNotFoundError)):
                    validate_attempt(cell,source_head=plan['source']['head'])

    def test_server_failures_download_corruption_and_uncertain_finish_reconciliation(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg=self.config(Path(temp)/'unused')
            plan=build_plan(cfg)
            root=Path(temp)/'api'
            self.make_fixture(root,cfg,plan['source']['head'],plan['runs'])
            cell=root/'d2/constant/seed0'
            metadata=json.loads((cell/'config.json').read_text())
            artifact=Mock(state='COMMITTED',type='convex-evidence',metadata={k:metadata[k] for k in ('attempt_id','config_sha256','science_sha256')})
            artifact.name=metadata['key']+'-'+metadata['attempt_id']+':v0'
            artifact.files.return_value=[Mock(name='unused') for _ in range(5)]
            for item,name in zip(artifact.files.return_value,('config.json','metrics.csv','history.npz','figure.png','checksums.json')):
                item.name=name
            def download(**kwargs):
                self.assertTrue(kwargs['skip_cache'])
                for name in ('config.json','metrics.csv','history.npz','figure.png','checksums.json'):
                    shutil.copyfile(cell/name,Path(kwargs['root'])/name)
            artifact.download.side_effect=download
            remote=Mock(id=metadata['attempt_id'],state='finished',url='https://example.invalid/run',
                        config={'resolved':metadata},tags=['lab1'],summary={'completed':True})
            remote.logged_artifacts.return_value=[artifact]
            api=Mock()
            api.run.return_value=remote
            for state in ('running','failed'):
                remote.state=state
                proof=reconcile(cell,api=api)
                self.assertFalse(proof['remote_verified'])
                self.assertFalse(proof['server_finished'])
            remote.state='finished'
            api.run.side_effect=TimeoutError()
            self.assertEqual(reconcile(cell,api=api)['error_type'],'TimeoutError')
            api.run.side_effect=None
            artifact.download.side_effect=ConnectionError()
            proof=reconcile(cell,api=api)
            self.assertTrue(proof['artifact_uploaded'])
            self.assertFalse(proof['artifact_verified'])
            def corrupt(**kwargs):
                download(**kwargs)
                (Path(kwargs['root'])/'metrics.csv').write_text('corrupt download')
            artifact.download.side_effect=corrupt
            self.assertFalse(reconcile(cell,api=api)['artifact_verified'])
            artifact.download.side_effect=download
            status=json.loads((cell/'status.json').read_text())
            status.update(local_status='failed',sdk_finished=False)
            atomic_json(cell/'status.json',status)
            proof=reconcile(cell,api=api)
            self.assertTrue(proof['artifact_verified'])
            self.assertFalse(proof['sdk_finished'])
            self.assertTrue(validate_attempt(cell,require_remote=True)['reconciled'])
            remote.logged_artifacts.return_value=[artifact,artifact]
            self.assertFalse(reconcile(cell,api=api)['remote_verified'])

    def test_ambiguous_init_is_discovered_by_attempt_id_without_new_upload(self):
        with tempfile.TemporaryDirectory() as temp:
            for count in (1,2):
                cfg=self.config(Path(temp)/str(count))
                with patch('experiments.convex_official.init_run',side_effect=ConnectionError()):
                    with self.assertRaises(ConnectionError): run_official(cfg)
                cell=Path(cfg['output_dir'])/'d2/constant/seed0'
                metadata=json.loads((cell/'config.json').read_text())
                remote=Mock(id='discovered-id',state='failed',url='https://example.invalid/failed',
                            config={'resolved':metadata},tags=['lab1'],summary={'completed':False})
                api=Mock()
                api.runs.return_value=[remote]*count
                api.run.return_value=remote
                proof=reconcile(cell,api=api)
                self.assertFalse(proof['remote_verified'])
                if count==1: self.assertEqual(proof['run_id'],'discovered-id')
                else: api.run.assert_not_called()
                remote.logged_artifacts.assert_not_called()

    def test_selected_seed_order_is_the_declared_plan_order(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg=self.config(Path(temp)/'unused')
            cfg.update(seeds=[0,1,2],methods=['convex','constant','inverse_sqrt'])
            plan=build_plan(cfg)
            root=Path(temp)/'ordered'
            self.make_fixture(root,cfg,plan['source']['head'],plan['runs'])
            selected,_=select_attempts([root],plan)
            self.assertEqual([row['key'] for row in selected],[cell['key'] for cell in plan['runs']])
