"""Read-only W&B reconciliation and strict multi-root Convex aggregation."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from core.convex_recovery import PAYLOAD_FILES,atomic_json,append_event,select_attempts,validate_attempt
from core.tracking import DEFAULT_ENTITY,DEFAULT_PROJECT
from core.utils import load_config
from experiments.convex_official import build_plan


def reconcile(directory,*,api=None,download_root=None):
    """Verify server state and freshly downloaded SHA256 payload without uploads.

    Failed or unfinished observations remain explicit. An uncertain finish can
    be recovered only with complete local bytes and independently verified
    remote success. SDK-finished stays false if that call never returned.
    """
    directory=Path(directory)
    status=validate_attempt(directory,allow_pending=True,metadata_only=True)
    config=json.loads((directory/'config.json').read_text())
    proof=dict(key=config['key'],attempt_id=config['attempt_id'],head=config['head'],config_sha256=config['config_sha256'],
        science_sha256=config['science_sha256'],run_id=status.get('run_id'),local_completed=status.get('local_completed',False),
        sdk_finished=status.get('sdk_finished',False),server_finished=False,artifact_uploaded=False,
        artifact_verified=False,publicly_accessible=None,remote_verified=False)
    stage='api_run'
    try:
        if proof['local_completed']:
            validate_attempt(directory,allow_pending=True)
        if api is None:
            import wandb
            api=wandb.Api(timeout=20)
        if not proof['run_id']:
            candidates=list(api.runs(f'{DEFAULT_ENTITY}/{DEFAULT_PROJECT}',filters={'config.resolved.attempt_id':config['attempt_id']}))
            if len(candidates)!=1:
                raise ValueError('ambiguous init, find exactly one remote attempt before any retry')
            proof['run_id']=candidates[0].id
        run=api.run(f"{DEFAULT_ENTITY}/{DEFAULT_PROJECT}/{proof['run_id']}")
        proof.update(server_state=run.state,wandb_url=run.url)
        resolved=run.config['resolved']
        stage='remote_provenance'
        for key in ('head','config_sha256','science_sha256','attempt_id','key','source_sha256'):
            if resolved[key]!=config[key]:
                raise ValueError('remote provenance mismatch: '+key)
        stage='server_state'
        if 'lab1' not in run.tags or run.state!='finished' or run.summary.get('completed') is not True:
            raise ValueError('remote run is not a completed successful attempt')
        proof['server_finished']=True
        artifacts=list(run.logged_artifacts())
        stage='artifact_state'
        if len(artifacts)!=1 or artifacts[0].state!='COMMITTED':
            raise ValueError('expected exactly one committed artifact, no implicit duplicate uploads')
        artifact=artifacts[0]
        if artifact.metadata.get('attempt_id')!=config['attempt_id']:
            raise ValueError('artifact belongs to a different attempt')
        names={item.name for item in artifact.files()}
        if names!=set(PAYLOAD_FILES)|{'checksums.json'}:
            raise ValueError('remote artifact file set mismatch')
        proof.update(artifact_uploaded=True,artifact_name=artifact.name)
        target=Path(download_root) if download_root else directory/'remote_downloads'/uuid.uuid4().hex
        target.mkdir(parents=True,exist_ok=False)
        stage='artifact_download'
        artifact.download(root=str(target),skip_cache=True)
        artifact.verify(str(target))
        stage='artifact_checksum'
        local=json.loads((directory/'checksums.json').read_text())
        remote=json.loads((target/'checksums.json').read_text())
        if local!=remote:
            raise ValueError('downloaded checksum manifest mismatch')
        for name,expected in local.items():
            if hashlib.sha256((target/name).read_bytes()).hexdigest()!=expected:
                raise ValueError('downloaded SHA256 mismatch: '+name)
        proof.update(artifact_verified=True,remote_verified=True,checksums=local,download_path=str(target))
        if not proof['local_completed']:
            raise ValueError('remote success cannot repair missing local computation')
    except Exception as error:
        proof.update(remote_verified=False,error_type=type(error).__name__,error_stage=stage,
                     recovery_action='inspect proof; reconcile before retrying into a fresh root')
    atomic_json(directory/('verification-'+uuid.uuid4().hex+'.json'),proof)
    atomic_json(directory/'remote_verification.json',proof)
    append_event(directory,'remote_observation',**proof)
    updated=dict(status,remote_verified=proof['remote_verified'],server_finished=proof['server_finished'],
                 artifact_uploaded=proof['artifact_uploaded'],artifact_verified=proof['artifact_verified'])
    if proof['run_id']:
        updated.update(run_id=proof['run_id'],wandb_url=proof.get('wandb_url',status.get('wandb_url')))
    if proof['remote_verified']:
        updated.update(local_status='complete',reconciled=True,run_id=proof['run_id'],wandb_url=proof['wandb_url'])
    atomic_json(directory/'status.json',updated)
    return proof


def aggregate_roots(roots,plan,output,*,require_remote=True,selections=None):
    """Merge exact coverage from fresh attempts, never retraining or uploading."""
    from experiments.convex_figures import export_figures
    records,excluded=select_attempts(roots,plan,require_remote=require_remote,selections=selections)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'selection_manifest.json',dict(source_head=plan['source']['head'],
        expected_keys=[c['key'] for c in plan['runs']],selected=records,excluded=excluded,
        explicit_selections=selections,require_remote=require_remote))
    export_figures(output,records,plan['config'])
    atomic_json(output/'summary.json',dict(protocol_version=plan['protocol_version'],runs=records))
    return records


def main():
    """Reconcile one cell or merge explicitly supplied batch roots."""
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    verify=sub.add_parser('verify')
    verify.add_argument('cell',type=Path)
    verify.add_argument('--download-root',type=Path)
    merge=sub.add_parser('merge')
    merge.add_argument('--roots',nargs='+',type=Path,required=True)
    merge.add_argument('--config',type=Path,default=ROOT/'configs/convex_cp03.yaml')
    merge.add_argument('--plan',type=Path)
    merge.add_argument('--output',type=Path,required=True)
    merge.add_argument('--local-only',action='store_true')
    merge.add_argument('--selections',type=Path)
    args=parser.parse_args()
    if args.command=='verify':
        proof=reconcile(args.cell,download_root=args.download_root)
        print(json.dumps(proof,indent=2))
        if not proof['remote_verified']:
            raise SystemExit(1)
    else:
        plan=json.loads(args.plan.read_text()) if args.plan else build_plan(load_config(args.config))
        if args.local_only and plan['config']['run_category']=='reproduction':
            raise ValueError('official aggregate requires remote proof')
        selections=json.loads(args.selections.read_text()) if args.selections else None
        print(json.dumps(aggregate_roots(args.roots,plan,args.output,require_remote=not args.local_only,selections=selections),indent=2))


if __name__=='__main__':
    main()
