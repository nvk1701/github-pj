"""Durable Convex attempt evidence and strict cross-root selection.

Retries use fresh output roots. This module never trains, creates a W&B run
or uploads an artifact. It does not silently pick between duplicate successes.
"""

import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import uuid

import numpy as np

PAYLOAD_FILES=('config.json','metrics.csv','history.npz','figure.png')


def canonical_hash(value):
    """Hash canonical finite JSON without reading any credential settings."""
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def science_hash(config):
    """Ignore selection/location/logging only, retaining every scientific value."""
    operational={'dimensions','methods','seeds','output_dir','mode','log_interval'}
    return canonical_hash({k:v for k,v in config.items() if k not in operational})


def atomic_json(path,value):
    """Serialize before touching disk, fsync a new file then atomically replace."""
    path=Path(path)
    payload=json.dumps(value,indent=2,allow_nan=False)
    temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('x',encoding='utf-8') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary,path)


def append_event(directory,event,**values):
    """Append and fsync one immutable event to the batch's attempt ledger."""
    directory=Path(directory)
    row=dict(event=event,time_ns=time.time_ns(),path=str(directory),**values)
    payload=json.dumps(row,allow_nan=False)+'\n'
    with (directory.parents[2]/'attempts.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def check_sdk_directories():
    """Probe explicitly configured SDK storage before creating any remote run."""
    for name in ('WANDB_DIR','WANDB_DATA_DIR','WANDB_CACHE_DIR'):
        value=os.environ.get(name)
        if value:
            directory=Path(value)
            directory.mkdir(parents=True,exist_ok=True)
            with tempfile.TemporaryFile(dir=directory) as stream:
                stream.write(b'probe')
                stream.flush()


def seal_payload(directory):
    """Write SHA256 checksums only after every local scientific file exists."""
    directory=Path(directory)
    checksums={name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in PAYLOAD_FILES}
    atomic_json(directory/'checksums.json',checksums)
    return checksums


def validate_attempt(directory,*,source_head=None,expected_science=None,require_remote=False,allow_pending=False,metadata_only=False):
    """Reject missing/corrupt/mismatched attempts, not merely missing directories."""
    directory=Path(directory)
    config=json.loads((directory/'config.json').read_text(encoding='utf-8'))
    status=json.loads((directory/'status.json').read_text(encoding='utf-8'))
    manifest=json.loads((directory.parents[2]/'run_manifest.json').read_text(encoding='utf-8'))
    cfg=manifest['config']
    matching=[c for c in manifest['runs'] if Path(c['path']).resolve()==directory.resolve()]
    if len(matching)!=1 or any(config[k]!=matching[0][k] for k in ('key','dimension','seed','method')):
        raise ValueError('logical cell does not match the batch manifest')
    if canonical_hash(cfg)!=config['config_sha256'] or manifest['config_sha256']!=config['config_sha256']:
        raise ValueError('config hash mismatch')
    if science_hash(cfg)!=config['science_sha256'] or expected_science not in (None,config['science_sha256']):
        raise ValueError('scientific configuration mismatch')
    if config['head']!=manifest['source']['head'] or source_head not in (None,config['head']):
        raise ValueError('source Git SHA mismatch')
    if cfg['run_category']=='reproduction' and config['source_uncommitted']:
        raise ValueError('official source was dirty')
    recovered=status.get('reconciled') is True
    if not allow_pending and (status['local_status']!='complete' or not status.get('local_completed') or not (status.get('sdk_finished') or recovered)):
        raise ValueError('attempt is incomplete, reconcile or retry into a fresh root')
    if status['attempt_id']!=config['attempt_id'] or status['key']!=config['key']:
        raise ValueError('attempt identity mismatch')
    if metadata_only:
        return dict(status,path=str(directory),dimension=config['dimension'],seed=config['seed'],method=config['method'])
    checksums=json.loads((directory/'checksums.json').read_text())
    if set(checksums)!=set(PAYLOAD_FILES):
        raise ValueError('payload file set mismatch')
    for name,expected in checksums.items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'checksum mismatch: {name}')
    with np.load(directory/'history.npz',allow_pickle=False) as data:
        if data['theta'].shape!=(cfg['steps'],config['dimension']):
            raise ValueError('trajectory shape does not match config')
        if not all(np.isfinite(data[k]).all() for k in data.files):
            raise ValueError('nonfinite scientific payload')
        if not np.array_equal(data['step'],np.arange(1,cfg['steps']+1)):
            raise ValueError('step alignment mismatch')
        with (directory/'metrics.csv').open(newline='',encoding='utf-8') as stream:
            reader=csv.DictReader(stream)
            rows=list(reader)
        if len(rows)!=cfg['steps']:
            raise ValueError('CSV row count mismatch')
        for key in reader.fieldnames:
            if key not in data.files or not np.array_equal(np.array([float(r[key]) for r in rows]),data[key]):
                raise ValueError(f'CSV/NPZ metric mismatch: {key}')
    if require_remote or (recovered and not allow_pending):
        proof=json.loads((directory/'remote_verification.json').read_text())
        if not all(proof.get(k) is True for k in ('server_finished','artifact_uploaded','artifact_verified')):
            raise ValueError('remote evidence is not verified')
        if any(proof.get(k)!=config[k] for k in ('key','attempt_id','head','config_sha256','science_sha256')):
            raise ValueError('remote proof provenance mismatch')
        if proof.get('checksums')!=checksums or proof.get('run_id')!=status.get('run_id'):
            raise ValueError('remote artifact identity mismatch')
    return dict(status,path=str(directory),dimension=config['dimension'],seed=config['seed'],method=config['method'])


def select_attempts(roots,plan,*,require_remote=True,selections=None):
    """Require exact coverage and one explicitly selected valid success per key."""
    keys=[cell['key'] for cell in plan['runs']]
    expected=set(keys)
    if len(expected)!=len(keys):
        raise ValueError('duplicate logical keys in plan')
    candidates={key:[] for key in keys}
    excluded=[]
    seen=set()
    for root in roots:
        root=Path(root).resolve()
        if root in seen:
            raise ValueError('duplicate root')
        seen.add(root)
        manifest=json.loads((root/'run_manifest.json').read_text())
        if science_hash(manifest['config'])!=science_hash(plan['config']) or manifest['source']['head']!=plan['source']['head']:
            raise ValueError('root source/config differs from the expected plan')
        for cell in manifest['runs']:
            directory=Path(cell['path'])
            if directory.resolve().parents[2]!=root:
                raise ValueError('manifest path escapes its batch root')
            if cell['key'] not in expected:
                raise ValueError('unexpected logical cell')
            status_path=directory/'status.json'
            status=json.loads(status_path.read_text()) if status_path.exists() else {'local_status':'not_started'}
            if status['local_status']!='complete':
                excluded.append(dict(key=cell['key'],path=str(directory),reason=status['local_status']))
                continue
            valid=validate_attempt(directory,source_head=plan['source']['head'],expected_science=science_hash(plan['config']),require_remote=require_remote)
            candidates[cell['key']].append(valid)
    selected=[]
    for key,values in candidates.items():
        if selections and key in selections:
            values=[v for v in values if v['attempt_id']==selections[key]]
        if len(values)!=1:
            raise ValueError(f'expected exactly one verified success for {key}, found {len(values)}')
        selected.append(values[0])
    return selected,excluded
