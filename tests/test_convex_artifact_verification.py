"""Scientific artifact identity and backend-history regressions; no online runs."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from experiments.convex_official import build_plan
from experiments.convex_reconcile import reconcile,scientific_artifact
import test_convex_recovery as fixtures


class ArtifactVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.root=self.base/'input'
        helper=fixtures.RecoveryTests();cfg=helper.config(self.root)
        plan=build_plan(cfg);helper.make_fixture(self.root,cfg,plan['source']['head'],plan['runs'])
        self.cell=self.root/'d2/constant/seed0'
        self.config=json.loads((self.cell/'config.json').read_text())
        self.artifact=Mock(type='convex-evidence',state='COMMITTED',
            metadata={k:self.config[k] for k in ('attempt_id','config_sha256','science_sha256')})
        self.artifact.name=self.config['key']+'-'+self.config['attempt_id']+':v0'
        self.artifact.files.return_value=[SimpleNamespace(name=n) for n in ('config.json','metrics.csv','history.npz','figure.png','checksums.json')]
        self.artifact.download.side_effect=self.download
        self.run=Mock(id=self.config['attempt_id'],state='finished',config={'resolved':copy.deepcopy(self.config)},
            tags=['lab1'],summary={'completed':True},url='https://example.invalid/run')
        self.run.logged_artifacts.return_value=[self.artifact]
        self.api=Mock();self.api.run.return_value=self.run

    def download(self,**kwargs):
        self.assertTrue(kwargs['skip_cache'])
        for item in self.artifact.files():shutil.copyfile(self.cell/item.name,Path(kwargs['root'])/item.name)

    def verify(self):
        return reconcile(self.cell,api=self.api,read_only=True,download_root=self.base/'downloads')

    def history(self,version=0):
        return SimpleNamespace(type='wandb-history',name='run-'+self.run.id+f'-history:v{version}',state='COMMITTED')

    def snapshot(self):
        return {str(p.relative_to(self.root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.rglob('*') if p.is_file()}

    def test_one_scientific_without_history(self):
        before=self.snapshot();proof=self.verify()
        self.assertTrue(proof['remote_verified']);self.assertEqual(proof['backend_history_artifacts'],[])
        self.assertEqual(proof['head'],self.config['head']);self.assertEqual(proof['training_source_sha'],self.config['head'])
        self.assertEqual(len(proof['verification_tool_file_sha256']),64)
        self.assertEqual(self.snapshot(),before)

    def test_one_or_multiple_backend_history_do_not_replace_scientific_checks(self):
        for n in (1,3):
            with self.subTest(n=n):
                artifacts=[self.history(i) for i in range(n)]+[self.artifact]
                selected,backend=scientific_artifact(artifacts,self.config,self.run.id)
                self.assertIs(selected,self.artifact);self.assertEqual(len(backend),n)
        self.run.logged_artifacts.return_value=artifacts
        self.assertTrue(self.verify()['remote_verified'])

    def test_duplicate_scientific_rejected_even_with_history(self):
        self.run.logged_artifacts.return_value=[self.artifact,self.history(),self.artifact]
        proof=self.verify();self.assertFalse(proof['remote_verified']);self.artifact.download.assert_not_called()

    def test_wrong_scientific_name_and_version(self):
        for name in ('unrelated:v0',self.artifact.name.replace(':v0',':v1')):
            with self.subTest(name=name):
                self.artifact.name=name
                with self.assertRaisesRegex(ValueError,'name/version'):scientific_artifact([self.artifact],self.config,self.run.id)

    def test_wrong_type_and_unknown_extra_rejected(self):
        for kind in ('dataset',None):
            with self.subTest(kind=kind):
                self.artifact.type=kind
                with self.assertRaisesRegex(ValueError,'unexpected artifact'):scientific_artifact([self.artifact],self.config,self.run.id)
        self.artifact.type='convex-evidence'
        with self.assertRaises(ValueError):scientific_artifact([self.artifact,SimpleNamespace(type='dataset')],self.config,self.run.id)

    def test_missing_scientific_history_only(self):
        for artifacts in ([],[self.history()]):
            with self.assertRaisesRegex(ValueError,'exactly one'):scientific_artifact(artifacts,self.config,self.run.id)

    def test_backend_identity_must_belong_to_the_same_run(self):
        bad=self.history();bad.name='run-another-history:v0'
        with self.assertRaisesRegex(ValueError,'unexpected artifact'):scientific_artifact([self.artifact,bad],self.config,self.run.id)

    def test_scientific_pending_and_wrong_attempt_metadata(self):
        self.artifact.state='PENDING'
        with self.assertRaises(ValueError):scientific_artifact([self.artifact],self.config,self.run.id)
        self.artifact.state='COMMITTED'
        for key in ('attempt_id','config_sha256','science_sha256'):
            old=self.artifact.metadata[key];self.artifact.metadata[key]='wrong'
            with self.assertRaisesRegex(ValueError,'metadata mismatch'):scientific_artifact([self.artifact],self.config,self.run.id)
            self.artifact.metadata[key]=old

    def test_remote_source_config_science_protocol_and_logical_identity(self):
        for key in ('head','config_sha256','science_sha256','attempt_id','key','source_sha256','protocol_version','dimension','seed','method'):
            with self.subTest(key=key):
                self.run.config={'resolved':dict(self.config,**{key:'wrong'})}
                proof=self.verify();self.assertFalse(proof['remote_verified']);self.assertEqual(proof['error_stage'],'remote_provenance')
        self.artifact.download.assert_not_called()

    def test_failed_incomplete_remote_and_missing_lab_tag(self):
        for state,completed,tags in [('failed',True,['lab1']),('running',False,['lab1']),('finished',False,['lab1']),('finished',True,[])]:
            with self.subTest(state=state,completed=completed,tags=tags):
                self.run.state=state;self.run.summary={'completed':completed};self.run.tags=tags
                proof=self.verify();self.assertFalse(proof['remote_verified']);self.assertEqual(proof['error_stage'],'server_state')
        self.artifact.download.assert_not_called()

    def test_corrupt_download_hash_stays_rejected_with_history(self):
        self.run.logged_artifacts.return_value=[self.artifact,self.history()]
        def corrupt(**kwargs):
            self.download(**kwargs);(Path(kwargs['root'])/'metrics.csv').write_text('corrupt')
        self.artifact.download.side_effect=corrupt
        proof=self.verify();self.assertFalse(proof['remote_verified']);self.assertEqual(proof['error_stage'],'artifact_checksum')

    def test_wrong_file_manifest(self):
        self.artifact.files.return_value=self.artifact.files.return_value[:-1]
        self.assertFalse(self.verify()['remote_verified']);self.artifact.download.assert_not_called()

    def test_read_only_failure_preserves_proofs_status_and_ledger(self):
        before=self.snapshot();self.api.run.side_effect=TimeoutError()
        self.assertFalse(self.verify()['remote_verified']);self.assertEqual(self.snapshot(),before)

    def test_read_only_requires_external_new_download_root(self):
        for target in (None,self.cell/'downloads',self.root/'downloads'):
            with self.assertRaisesRegex(ValueError,'outside the input batch'):
                reconcile(self.cell,api=self.api,read_only=True,download_root=target)
        target=self.base/'exists';target.mkdir()
        proof=reconcile(self.cell,api=self.api,read_only=True,download_root=target)
        self.assertFalse(proof['remote_verified']);self.assertEqual(proof['error_type'],'FileExistsError')


if __name__=='__main__':unittest.main()
