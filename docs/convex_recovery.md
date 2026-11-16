# Convex recovery and reproduction guide

CP03.5 adds operational evidence only. Frozen scientific protocol remains
`convex-cp03-v1` in configs/convex_cp03.yaml. No trainer, RNG, comparator,
population equation or main matrix value changes.

## Fresh checkout on Windows

The repository currently contains a literal empty `configs/*.yaml` path that
Windows cannot materialize. Do not delete it or change global Git settings.
After the release branch is published, clone with `--no-checkout` first.
Before publication, teammates cannot fetch the new commit from GitHub.

```powershell
git clone --no-checkout https://github.com/nvk1701/github-pj.git convex-reproduce
Set-Location convex-reproduce
# Replace RELEASE_SHA with the exact approved commit from the release report.
git symbolic-ref HEAD refs/heads/reproduce
git update-ref refs/heads/reproduce RELEASE_SHA
git -c core.protectNTFS=false read-tree RELEASE_SHA
git -c core.protectNTFS=false update-index --skip-worktree -- 'configs/*.yaml'
git -c core.protectNTFS=false update-index --no-skip-worktree -- configs/convex.yaml configs/convex_cp03.yaml configs/convex_cp03_sensitivity.yaml
git checkout-index -a
```

The new local reproduce branch makes HEAD agree with the materialized index.
Do not run these bootstrap commands in an existing dirty repo.
The per-command NTFS exception only permits loading the index. The invalid
file stays skipped. No command attempts to write it to disk. A proposed rename
patch is kept outside the repository for team review, not applied automatically.

Create a separate Python environment and install the existing requirements.
The development interpreter is Project/.venv-t2-clean/Scripts/python.exe.
On another machine use that machine's environment, not the development absolute
path. Core imports resolve relative to this checkout. The tests and `--dry-run`
work before any online authentication. Configure writable WANDB_DIR,
WANDB_DATA_DIR and WANDB_CACHE_DIR and the actual Forge endpoint. Let the SDK
load existing credentials or use its supported login flow. Never put a key in
the YAML, CLI, metadata or report.

```powershell
python -B -m unittest discover -s tests -v
python -B experiments/convex_official.py --config configs/convex_cp03.yaml --dry-run
```

## Batch layout and durable attempts

Prefer five roots, one per dimension, nine cells each. The union is still45.
Every explicit run creates fresh UUID attempt identities under the deterministic
logical cell keys. Root directories must be new. No resume, silent skip,
automatic retrain or upload retry occurs. The same source SHA and normalized
scientific hash must be used across all roots. A full config hash is retained
per root. Scientific hash ignores only root/selection/mode/logging options.

Status/config JSON is fsynced and atomically replaced. `attempts.jsonl` is an
append-only fsynced ledger. Immutable `failure-*.json` records preserve each
failure. `checksums.json` seals config/CSV/NPZ/PNG before logging/upload.
Local computation, SDK finish, server finish, artifact upload/verification and
public access are separate states. A directory's existence is never success.
Checksums, source/config identity, shape/finite checks and CSV/NPZ equality are
required before aggregation.

## Failure and recovery walkthrough

1. Stop the failed batch. Keep its root and ledger. Earlier completed cells
   stay valid. Unstarted cells are explicit missing coverage.
2. If init/finish/upload outcome is uncertain, reconcile before retry. Init
   discovery queries the existing remote attempt ID in config, without init or
   upload. Zero/multiple matches require inspection, not an automatic new run.
3. A failed/running server run is not verified success. Wait/reconcile if running.
   If failure is established, retry only failed/unstarted selections into a NEW
   root with the same source/scientific config. Do not overwrite old evidence.
4. For uncertain finish, a complete local payload plus independently finished
   server and matching downloaded artifact can reconcile success. A false SDK
   finish flag is preserved rather than rewritten as if the call returned.
5. Aggregate all roots only when every logical key has one verified successful
   attempt. Duplicate successes reject by default. An explicit selection JSON
   may choose one attempt per key after review. Failures are listed as excluded,
   not hidden. A corrupt supposedly complete attempt fails closed.

```powershell
python -B experiments/convex_reconcile.py verify PATH_TO_CELL --download-root NEW_DOWNLOAD_PATH
python -B experiments/convex_official.py --config configs/convex_cp03.yaml --dimensions 128 --output-dir outputs/cp04-d128
# Example retry selection, only after inspecting the original failure:
python -B experiments/convex_official.py --config configs/convex_cp03.yaml --dimensions 128 --seeds 1 --methods constant --output-dir outputs/cp04-d128-retry1
python -B experiments/convex_reconcile.py merge --roots outputs/cp04-d2 outputs/cp04-d8 outputs/cp04-d32 outputs/cp04-d128 outputs/cp04-d512 outputs/cp04-d128-retry1 --output outputs/cp04-merged
```

Reconciliation creates no run and uploads nothing. It requires exactly one
COMMITTED artifact for that attempt, downloads with skip_cache=True, calls SDK
artifact.verify and independently checks SHA256 for every payload file. Source,
config, logical key and attempt identity must match the server config.
Remote proof history is preserved in verification-*.json and the ledger.
Main merges require remote proof. `--local-only` is for pilot/testing only.
Merged plots read original selected files without copying or retraining them.

## Limits and release

This is single-writer per output root, not a multi-host database. Hard process
kill may leave a running/computed status. It is rejected until reconciled or
retried. Source changes between batches require a new coordinated matrix, not
silently combining SHA versions. Scientific formulas and all legacy outputs
are unchanged. Raw SHA256 may differ across platform newline conventions,
while Git commit/blob identity and numerical checks establish source content.
Public UI/member permissions are verified separately from authenticated API
and artifact access. Do not infer teammate write permission from public viewing.
No Git publication or official45-cell matrix is authorized by this guide.
