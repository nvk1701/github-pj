# Convex CP03 locked experimental protocol

Version: **convex-cp03-v1**. Main configuration is `configs/convex_cp03.yaml`.
This locks the project's experiment, not unknown author implementation details.
Full scope stays selected. CP04 has 45 cells: five dimensions, three seeds and three methods.
No official matrix is run during CP03.

## Scientific contract

Gaussian identity covariates and independent Gaussian label noise generate soft labels
sigmoid(X v + epsilon), with the spiral from shared `core.drift`. No Bernoulli labels.
The loss and projected SGD are the unchanged shared trainer. Snapshots are pre-update.
The primary schedule uses only EMA of completed learner updates with beta=.9.
It ignores current-batch loss and never uses evaluation data or future oracle drift.
`gamma_history` and `gamma_schedule` coincide in this primary mode.

Primary B = F(theta)-F(numerical approximate population oracle).
Secondary A = F(theta)-F(generating reference). Both use the same learner trajectory.
Deterministic one-dimensional Gaussian quadrature is the primary evaluator.
Paired independent MC estimates/SE remain separate validation columns.
Signed estimates are not clipped. Cumulative quantities are sums without division by T.
B-A equals the distribution-only correction F(reference)-F(oracle), up to numerical error.
Do not impose exact population inequalities on finite-sample MC estimates.

The CP02 derivation and independent checks remain in `docs/convex_protocol.md`.
The true Gaussian minimizer shrinks the reference, so R=d/2 contains it when it contains v.
The ball diameter is d, not its radius. L=.25 bounds the population Hessian.
Paper profile variance=d/4 is retained with its known lack of a uniform bound on this ball.
Safe variance=d and diameter-bound gamma remain separately labelled CP02 diagnostics.
This profile does not claim all theoretical assumptions are met by EMA+d/4.

## Locked values and sources

| Classification | Decisions |
|---|---|
| PAPER-SPECIFIED | Gaussian/sigmoid setup, spiral, schedule equations and logistic profile L=.25,variance=d/4,Dmax=d |
| MATHEMATICALLY-DERIVED | Radial population oracle, population smoothness, safe variance bound, oracle containment |
| EXPERIMENTALLY-VERIFIED | CP02 independent cubature/optimization, CP03 cached-vs-uncached agreement and MC checks |
| PROJECT-ASSUMPTION | Label std=sqrt(.1), eta_init1, EMA.9, radius=d/2, evaluation seeds/samples and untuned LR0=.1 |
| UNKNOWN | Exact author noise units, evaluator, initialization and gamma implementation |

Main T2000,dimensions2/8/32/128/512,B256,E512,seeds0/1/2,methodsConvex/constant/inverse_sqrt,
a1,b-1,k4,ell8. Constant/inverse_sqrt LR0=.1 is **predeclared untuned**, not selected from evaluation seeds.
Optional later tuning uses separate validation seeds100/101/102, not official seeds.
The original `configs/convex.yaml` and legacy runner defaults are unchanged.
Critical missing keys/conflicts fail. Reproduction requires clean committed source.
Only pilot/smoke may change T/B/E. Only sensitivity may change ell/noise.
Selection of frozen dimensions/seeds/methods is allowed for splitting execution.

## Oracle policy and uncertainty

PopulationEvaluator caches exact floating-point reference norms without rounding.
Reference/oracle loss and label moments are shared across methods/seeds for a dimension.
Each learner snapshot needs only its scalar softplus quadrature and a dot product.
No comparator is fitted to training or evaluation samples.
Tolerance1e-11 is the default. Quadrature errors, oracle parameter error and resulting
risk-suboptimality estimates are exported. They are estimated, not rigorous interval certificates.
Cached and CP02 uncached values must agree within1e-9 plus reported integration estimates.

MC uses SeedSequence([training_seed,1,dimension]) independently of shared trainer default_rng(seed).
Common random numbers are shared between methods. Conditional cumulative MC SE is sqrt(sum SE_t^2).
Between-seed sample SD uses ddof=1 and remains distinct from MC and integration errors.
Paper extraction uncertainty is unknown until a comparable sourced value exists.

## Operation and artifacts

`python -B experiments/convex_official.py --config configs/convex_cp03.yaml --dry-run`
prints the exact 45-cell manifest without directories, training or W&B initialization.
Run keys and paths are deterministic. Existing output roots are rejected. No resume/skipping framework.
Partial failures write failed status and propagate. Complete arrays are retained even if logging is sparse.
Main command after CP04 authorization:

```powershell
python -B experiments/convex_official.py --config configs/convex_cp03.yaml --mode online --output-dir outputs/convex-cp04-v1
```

Use the absolute Project `.venv-t2-clean/Scripts/python.exe` from the CP03 worktree.
Entity/project are explicitly the existing gr_ml_29_37_37/ml-distribution-shift through shared init_run.
The configured API endpoint is recorded. Forge is distinct from public api.wandb.ai.
Tags contain lab1, protocol, category, gamma and variance profile.
Each run records SHA,dirty state,source hashes,config hash,method/seed/dimension,comparators and packages.
Online smoke is explicitly named cp03-online-smoke and has category smoke, not reproduction.
Local complete, SDK finished, uploaded and API-verified are distinct states.
The runner never asserts uploaded=true without an independent remote check.

Per cell: config.json,status.json,metrics.csv,history.npz,figure.png and W&B evidence artifact.
NPZ/CSV retain full resolution. W&B logging stride20 only reduces dashboard logging overhead.
Global outputs include manifest,summary,Figure5-style dimension curves,d128 coordinate panel,
T15 baseline panel,source CSVs and Sheet4 mapping. Trajectory panel uses the lowest seed,
not an averaged trajectory. Legends distinguish generating reference and approximate oracle.
All plots use actual outputs and are labelled by category. No author screenshot is copied.

Sheet4 mapping leaves paper_value and relative_deviation N/A without comparable provenance.
A trajectory has no invented scalar comparator. Slide26 maps the plots and Slide28 the planned T17 study.
Sensitivity config is a one-factor plan after main runs. It cannot retune the frozen main protocol.

## Acceptance gates and limits

Require all tests, legacy byte/array comparisons, independent oracle/correction/MC tests,
finite d128/d512 pilots, source/config checks,no-overwrite/failure checks and a 45-key dry plan.
Online smoke must be API-verified or have an explicit isolated auth/network blocker.
Run official CP04 only with accepted compute budget, correct endpoint/auth and this locked version.
If source or protocol changes, regenerate hashes/manifest and review again.
T15 is not DONE before official runs, W&B evidence and appropriate Sheet4 comparison.
No CP02/CP03 push, PR or merge is authorized by this checkpoint.
