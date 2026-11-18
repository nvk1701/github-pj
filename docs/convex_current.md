# Convex: current source, results and review boundaries

Audit dated 18 November 2026. Earlier checkpoint documents retain their historical
scope. This index supplies current interpretation without relabelling old runs.

## Source and entry points

All 60 CP04/CP05 scientific runs used frozen training source
`7e0499c450a3e503ab09d4c6c3d272e6576b4432`.
The [published results](results/cp04-cp05/RESULTS.md),
[per-run URLs/hashes](results/cp04-cp05/data/run_provenance.csv) and
[publication manifest](results/cp04-cp05/publication_manifest.json) were committed
at `4d6bce54e68e559d77cb64bd5dab5eeb1cf7fac6`.
CP05.7 maintenance commits `37e2492` and `1316573` repair verification and add
analysis documents; they did not generate those results. Later maintenance must
continue recording the training source separately from the verification tool.

From the repository root, with the tested CPython 3.12 environment:

```powershell
python -m pip check
python -B -m scripts.check_environment --require-isolated
python -B -m scripts.smoke_tracking --mode offline --member Khoi
python -B -m unittest discover -s tests -v
python -B experiments/convex_official.py --config configs/convex_cp03.yaml --dry-run
python -B experiments/convex_reconcile.py verify --help
```

Offline smoke writes only local SDK files; choose external writable WANDB_DIR,
WANDB_DATA_DIR, WANDB_CACHE_DIR and WANDB_CONFIG_DIR. It does not establish online
upload or another person's permissions. `.venv/Scripts` contains environment
executables; repository `scripts/` contains the Python modules above. Use `-m`
for these modules. A member label is not an authenticated uploader identity.

`experiments/convex.py` is the older diagnostic runner in this branch.
`convex_official.py` supplies the guarded official protocol; `--dry-run` plans
45 cells without training. This page authorizes no experiment or online smoke.
The [recovery guide](convex_recovery.md) describes optional read-only verification.

## Scientific interpretation

Features are Gaussian N(0,I), with soft logistic responses and noise standard
deviation sqrt(.1) in CP04. Primary B uses deterministic Gaussian population risk
against a numerical approximate oracle; secondary A uses the generating reference.
Their cumulative values are sums. Signed Monte Carlo excess is preserved.
Sample SD across three seeds, conditional Monte Carlo SE and quadrature/oracle
error estimates describe different uncertainties; numerical errors are not formal
certificates. Shared trainer `inst_regret`/`cum_regret` are legacy clipped fields
and must not be substituted for official population or signed MC arrays.

Projection radius is d/2 and its ball has diameter d. The learner uses causal
learner-motion EMA (.9), not privileged true reference shifts. L=.25, variance
profile d/4 and initial LR1 define this project's declared schedule. These choices
do not certify every uniform assumption of a regret theorem.

[Paper](https://proceedings.mlr.press/v202/fahrbach23a/fahrbach23a.pdf) §6.1 uses
Gaussian features. The isolated factual correction in [R3 notes](04_convex.md)
does not change the user's separate local copy. Compared with the
[immutable author notebook](https://github.com/fahrbach/learning-rate-schedules/blob/1b3c0d456d70c9236190ac512bbf5168527fc837/src/regression/regression_experiments.ipynb),
project noise, shift signal, comparator, first-step behavior and numeric/path
implementation differ. Results are Figure-5-style project evidence; directly
comparable paper scalars and relative deviations remain N/A.

## Completed scope and historical plans

CP04 covers all 45 method/dimension/seed cells. Convex wins mean B at d2/8/32/128;
constant wins at d512 (504.8952 versus Convex 1385.0089). Preserve that outcome.
Baselines use declared untuned LR0=.1. Three seeds do not establish universal
dominance or a causal explanation for the d512 result.

CP05 executed 15 Convex-only d128 runs: five OFAT conditions × three seeds.
The [five-condition table](results/cp04-cp05/data/cp05_condition_summary.csv) and
[per-seed results](results/cp04-cp05/data/cp05_per_seed.csv) supersede interpreting
the historical T17 plan as global nonexecution. The old sensitivity YAML planned
a broader method matrix; it is not the actual resolved CP05 run configuration.
Changing ell changes sample-and-hold transition counts, jump sizes and variation
together. Noise also changes the label distribution and population comparator.
Neither sweep demonstrates general robustness or a baseline sensitivity win.

Current aggregate export marks T17 as outside that export and links CP05; it does
not regenerate or overwrite any published checkpoint figure/table/proof.

## Live main and team gates

Remote main observed during this audit is `8310f5b1b7cd5b3c93b9775f082c4972f8bf7fb7`.
It is a separate descendant of `ff3579a`, with copied Convex content, rewritten
diagnostic/tests and Linear work; it does not contain the frozen/results/CP05.7
commits as ancestors. A simple branch-ready conclusion requires renewed review.
Do not replace its shared schedule or teammate implementation automatically.

Main already renamed the invalid literal `configs/*.yaml` to `configs/test.yaml`
at `5026dff`. Historical frozen/CP05.7 and this maintenance branch still require
the documented no-checkout bootstrap on Windows. Do not apply a second repair
or reintroduce the invalid filename when integrating. Main's three CP05 PNGs
differ from the frozen publication manifest, which main currently omits. Review
that provenance discrepancy and its sensitivity exporter before integration.

[Team audit](results/cp04-cp05/WANDB_TEAM_READINESS.md) records observations on
17 November, not a live count or present-day membership certificate. Main now
declares Khoa's online runs and limited independent checks. Read-only SDK metadata
on 18 November confirms 85 runs: the original 65 under dangkhoi plus 20 under
nkhoa8508 (18 Linear reproduction runs and two smoke-test jobs). All original 60
scientific runs retain their source/config/scientific hashes and metadata. This
supports actual second-account logging; independently identify the account owner
and obtain executor/command/source/coverage evidence before closing T2/T19. Hòa's
membership, personal upload and runnable data/NN contract still require evidence.
Public rendered access does not imply write permission. Stable proposed groups
are linear/convex/nonconvex; job_type describes purpose. Existing runs stay intact.
The wrapper accepts flat metadata, including group/job_type and list tags;
tuple tags normalize to lists. Do not pass credentials or invented CLI flags.

Khoa reviews shared modules, README mapping, current-main integration and T19
acceptance; Khôi owns Convex science/provenance. Hòa owns the data/NN contract.
[Contributions](02_contributions.md), [complexity](06_complexity.md) and
[references](references.md) require whole-team review before issue acceptance.
The older Full Scope declaration exists in motivation_example.md; it does not
by itself attest implementation completion or a human acceptance meeting.
