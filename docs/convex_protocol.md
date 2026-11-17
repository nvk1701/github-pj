# Convex protocol — convex-cp02-v1-provisional

Status: **provisional**. Mathematical comparator reconciliation is established for the Gaussian model below. The paper's hidden evaluation/noise/init details and team acceptance remain unresolved. Neither protocol is declared an exact reproduction of Figure 5.

## Purpose and source classification

The target is Section 6.1 and Figure 5-bottom of [Fahrbach et al., PMLR202](https://proceedings.mlr.press/v202/fahrbach23a/fahrbach23a.pdf). Read Section2, Section4/Assumption4.1/Theorem4.2/Proposition4.3/Eq15–20/Remarks4.4–4.5 and AppendixB.1. No author implementation was used. Original R3 stays unchanged. `docs/convex_sprint1.md` remains the historical CP01 contract. This document adds CP02 decisions.

| Setting or claim | Classification | Decision/source |
|---|---|---|
| Expected-loss minimizer defines the dynamic-regret comparator | PAPER-SPECIFIED | Section2,Eq1–2 |
| Gaussian covariates, sigmoid soft labels, spiral construction | PAPER-SPECIFIED | Section6.1,Eq26. No Bernoulli label resampling |
| Logistic schedule L=1/4, sigma²=d/4,Dmax=d | PAPER-SPECIFIED | Section6.1. This states an experimental profile, not a proof of uniform validity |
| Gaussian regression epsilon notation N(0,0.1) | PAPER-SPECIFIED | Section6.1 Linear paragraph |
| Logistic inherits that noise, second parameter is variance0.1 | PROJECT-ASSUMPTION | Use label_noise_std=sqrt(.1). Logistic paragraph does not restate all noise details |
| Population oracle is a scalar shrinkage of the generating reference | MATHEMATICALLY-DERIVED | Proof below, independently checked by multidimensional cubature and optimization |
| L=1/4 is globally valid for this expected Gaussian logistic risk | MATHEMATICALLY-DERIVED | Expected Hessian bound below, not a bound on each Gaussian sample |
| v=d is a safe gradient variance bound, v/B for batch mean | MATHEMATICALLY-DERIVED | Proof below. v=d/4 is not a uniform bound on the chosen ball |
| Numerical oracle, deterministic risk and paired-MC agreement | EXPERIMENTALLY-VERIFIED | `tests/test_convex_population.py` and CP02 evidence. Error estimates are not interval certificates |
| RadiusR=d/2,zero init,EMA beta.9,eta_init1,three seeds,eval estimator | PROJECT-ASSUMPTION | Explicit protocol choices, not all specified by paper |
| Exact authors' comparator implementation/noise units/initial eta and tuning budget | UNKNOWN | Not inferred from the visual shape of a curve |

## Population risk and oracle derivation

At a fixed time let v be the generating reference, X~N(0,I_d), epsilon~N(0,s²), independent, and Y=sigmoid(X·v+epsilon). The loss for learner w is

\[
 F_v(w)=E[\log(1+e^{X\cdot w})-Y(X\cdot w)].
\]

Let r=||v||,q=sqrt(r²+s²),Z~N(0,1),a(u)=E[sigmoid'(uZ)] and h(u)=u a(u)=E[Z sigmoid(uZ)]. Integration by parts against the Gaussian density gives E[X f(X·w)]=w E[f'(X·w)]. For the joint Gaussian scalar X·v+epsilon, the same calculation gives E[XY]=v a(q). Boundary terms vanish because sigmoid is bounded and its derivative is bounded. Consequently

\[
 \nabla F_v(w)=w a(||w||)-v a(q),\qquad
 F_v(w)=E[softplus(||w||Z)]-(w\cdot v)a(q).
\]

For any finite w, the Hessian E[sigmoid'(X·w)XXᵀ] is positive definite: for nonzero b, its quadratic form is E[sigmoid'(X·w)(b·X)²]>0. Thus the risk is strictly convex. Every stationary solution is parallel to v with a positive coefficient. Its radius u solves

\[
 h(u)=r a(q)=(r/q)h(q),\qquad w_*=(u/r)v.
\]

For u>=0, h'(u)=E[Z² sigmoid'(uZ)]>0. Also a(u)>0 and strictly decreases for u>0, since sigmoid' is even and strictly decreases with absolute argument. If r>0 and s>0, q>r implies 0<r a(q)<r a(r)=h(r). There is exactly one root u in (0,r). This stationary point is the unique global minimizer. If s=0 the minimizer is v. If v=0 it is zero, even with noisy labels. These are mathematical statements for the specified independent Gaussian model, not conclusions for all noise distributions.

For non-Gaussian independent symmetric noise, rotational symmetry still suggests a radial reduction but the Gaussian q collapse is unavailable. Neither the implementation nor its error analysis claims support for arbitrary dependent/asymmetric noise or non-isotropic covariates.

The numerical solver in `core/convex_population.py` integrates h on z>=0, using h(u)=integral z*tanh(uz/2)*phi(z) dz, then brackets on [0,r] with Brent's method. No optimizer is run on training/evaluation data. It returns a **numerical approximate oracle**, not an exact floating-point oracle. Adaptive quadrature tolerance is1e-11 by default, supported1e-13..1e-6, with warnings/failure rejected.

### Error accounting

For u in [0,r], h'(u)>=h'(r)>0. A moment residual plus estimated quadrature errors, divided by a lower curvature estimate h'(r)-error, estimates the parameter error. L-smoothness then gives a risk suboptimality estimate L/2 times squared parameter error. The code uses L=1/4 for this purpose. Loss quadrature separately propagates estimated integration errors through the linear term. Error estimates are checked against tighter tolerances and independent full cubature/optimization.

QUADPACK errors are estimates, not rigorous enclosures. Very ill-conditioned/extreme scales outside tested profiles need additional checks. MC SE excludes systematic approximate-oracle and quadrature error. Do not label it a complete confidence interval for true regret.

## Two comparison protocols

### A — paper-profile generating-reference comparison

Keep the data generator and reference path. Preserve CP01 outputs `eval_mc_excess`, `eval_cum_mc_excess` as **signed excess to generating reference**, never silently rename them regret. These may be negative. Use v_schedule=d/4 as the explicit paper profile. EMA is the practical history-only shift choice. This is a paper-profile reference study, not an assertion that every hidden author setting has been recovered.

### B — numerical population-oracle comparison

Keep the same distribution and learner snapshots. Construct w_* numerically using the population model, only in the evaluator. Export `eval_mc_approx_oracle_excess` and `eval_cum_mc_approx_oracle_excess` with their paired-MC SE. Store comparator weights and parameter-error estimates. Deterministic quadrature checks use `population_approx_oracle_excess_quad`. The comparator correction changes the scientific metric, not the labels/data generator.

Both metrics may be recorded on the same trajectory to isolate the comparator effect. They are separate columns and separate summary fields, never mixed within one comparison table. Expected loss to the exact population minimizer is nonnegative. Finite MC estimates and approximate-oracle estimates are signed and must not be clipped to enforce that property.

A more conservative B schedule diagnostic uses v_schedule=d and gamma_bound=Dmax. This changes the schedule profile as well as the metric and must be reported as a distinct diagnostic, not attributed to comparator correction alone. It is not the original Figure 5 protocol.

## Noise and smoothness

Use `label_noise_std=s` to generate epsilon. Its variance is s². The schedule's `gradient_variance` is v_schedule=sigma_grad², distinct from label noise. Thresholds divide v_schedule by batch size, so do not pass a pre-divided batch variance and divide twice.

For g=X(sigmoid(X·w)-Y), |sigmoid(X·w)-Y|<=1. Therefore

\[
 E||g-Eg||²\le E||g||²\le E||X||²=d.
\]

Independent batches give Var(mean g)=Var(g)/B. This safe bound applies uniformly in w,v,s for the stated label range and Gaussian identity X. It is conservative, not a claim of tightness. A deterministic Gaussian counterexample within the selected domain (d8,reference2e1,w=-4e1,s=sqrt(.1)) disproves v=d/4 as a uniform bound. It does not show that every actual learned trajectory violates d/4. Paper-profile diagnostics retain d/4 with this limitation disclosed. Theory-compatible conservative diagnostics require d.

For any b, bᵀ Hessian F(w)b=E[sigmoid'(X·w)(b·X)²]<=E[(b·X)²]/4=||b||²/4, so L=1/4 globally for population risk. Individual Gaussian samples have unbounded ||X|| and do not have a uniform per-example1/4 smoothness bound. The lower-bound theorem's extra strong-convexity assumptions are not automatically satisfied by each rank-one logistic sample.

## Gamma, filtration and projection

At step t the learner holds w_t before seeing the training batch. It chooses eta_t from completed updates, then computes loss/gradient on batch t, projects w_t-eta_t*g_t and proceeds. Evaluation occurs afterward using saved pre-update snapshots and an independent stream. No evaluation quantity enters the schedule. Shared trainer computes the batch before calling schedule.step, but ConvexSchedule ignores loss_val/noise_std and only consumes prior-state drift, so the information used remains history-only.

| Mode | Actual schedule input | Information/role | Labels |
|---|---|---|---|
| estimated | EMA of completed learner updates | Online implementability, practical reproduction proxy | estimated, paper/safe profile |
| lagged_reference | ||v_(t-1)-v_t||,first0 | Privileged simulator diagnostic, not true-oracle gamma_t | lagged_reference, diagnostic |
| diameter_bound | Dmax every step | Known future-free upper bound on true-oracle shift inside Theta | diameter_bound,safe,diagnostic |

True gamma_t=||w_*t-w_*t+1|| under ProtocolB, not necessarily reference shift. Neither lagged_reference nor EMA is proved an upper bound. No direct future-reference mode is implemented. Exact authors' Figure5 shift mode remains UNKNOWN, so estimated is the project's operational choice, not falsely called author-exact. `gamma_history` records trainer-supplied EMA/lagged signal. `gamma_schedule` records the actual diameter constant when the diagnostic adapter overrides that signal. For baseline methods these fields are informational only.

Theta is the Euclidean ball centered0 with radiusR=d/2, diameterDmax=d. Spiral norm is bounded by sqrt(ceil(d/2))*max(|a|,|b|)^3. Config validation requires this bound<=R, d>=2 and2R<=Dmax. The true oracle shrinks v, so it belongs to the same ball. A too-small domain would require a constrained comparator rather than silently clipping and calling it the unconstrained minimizer. No global projection utility is changed.

The conservative combination v=d,gamma_bound=Dmax,L>=1/4,oracle in Theta satisfies the relevant upper-bound assumptions for this fixed distribution path. This does not certify the numerical implementation against every floating-point error or claim to reproduce the paper's chosen rates. Existing EMA+d/4 should not inherit the theorem guarantee without these assumptions.

## Locked experiment choices and seeds

Full candidate grid for CP03/04: d=[2,8,32,128,512],T2000,B256,ell8,a1,b-1,k4,zero init. These grid/T/path choices must be described by their source versus project choice, not all claimed author-specified. eta_init1,constantLR.1,inverse_sqrtLR0.1,EMA beta.9 are current project choices. No baseline tuning has been completed.

CP02 entrypoint explicitly limits T<=128,d<=32 to avoid accidental full reproduction. Its default diagnostics are T32,d2,B32,E1024,seeds0/1/2,three methods,noise_std=sqrt(.1),offline. Full profile remains in `configs/convex.yaml` unchanged.

Evaluation uses paired X/Y for learner/reference/approximate oracle. Methods with the same d/seed share training streams (local default_rng(seed)) and evaluation streams (SeedSequence([seed,eval_seed_stream=1,d])). Each timestep gets a new evaluation batch independent of training. Cumulative SE is sqrt(sum SE_t²) conditional on the learned path. Aggregate seed std is a different quantity. Increasing E reduces MC noise, not comparator approximation error.

Proposed CP03 baseline tuning: validation seeds100/101/102 only, LR0 grid[.01,.03,.1,.3,1] under the same comparator/gamma/domain/budget, choose by mean validation population/MC metric fixed in advance. Freeze choices before evaluation seeds0/1/2. Do not choose settings based on which looks closest to Figure5 or on held-out evaluation seeds. This grid is a project proposal, not already executed tuning.

## Run metadata, artifacts and operation

Diagnostic version string is `convex-cp02-v1-provisional`. Naming: `cp02-{gamma_mode}-{variance_profile}-d{d}-{method}-seed{seed}`. Every diagnostic run has tags lab1,diagnostic,CP02,version,gamma_mode,variance_profile. Metadata records both comparator definitions, source SHA/dirty state/source hashes, known remote-tracking tip and whether source HEAD is absent from that history. A local commit not yet pushed is explicitly identified. Remote-tracking provenance is an observation at fetch time, not a live network check per run.

Use existing project `gr_ml_29_37_37/ml-distribution-shift` and Forge endpoint. CP02 uses offline/disabled runs with uploaded=false and URLs=null, no official-reproduction tags. Online is authorized but not required for these numerical questions. No automatic sync or fallback from online is attempted.

Each run exports config.json,metrics.csv,history.npz (theta/reference/approx_oracle plus all metric arrays) and summary.json. Root summary lists all runs. Output root must not exist, so old evidence is never overwritten. Original CP01 runner still exports its PNG and reference-only metrics, bitwise baseline checks continue. No NPZ/PDF/run logs/reports/venvs enter Git. W&B offline directory and local reports stay outside the repo.

```powershell
# From the Convex worktree, with the Project clean interpreter
python -B experiments/convex_diagnostics.py --mode offline --output-dir D:/HOCBAI/HK7/ML/Project/_reports/cp02_evidence/diagnostics
python -B experiments/convex_diagnostics.py --mode offline --gamma-mode diameter_bound --variance-profile safe --seeds 0 --methods convex --output-dir D:/HOCBAI/HK7/ML/Project/_reports/cp02_evidence/diameter_diagnostic
```

## Acceptance and remaining blockers

VERIFIED: Gaussian radial oracle derivation, numerical validation, smoothness/domain argument, safe variance bound, meaningful counterexample to uniform d/4, causal estimator separation and non-overwriting diagnostics.

PARTIAL: Figure5-paper fidelity and uncertainty budget at full dimension/time, cross-platform runtime and baseline tuning. The new B metric is an approximate-population comparison, not Figure5-original by definition.

UNKNOWN: exact authors' epsilon interpretation/evaluator/eta_init/shift mode and digitization error of paper curves. BLOCKED for final scientific sign-off: team acceptance of primary comparator/protocol and approved official reproduction plan. ProtocolA can be reproduced operationally, but cannot be renamed true regret when s>0.

Khoa review: shared trainer clipping/naming, gamma_prev contract and radius guidance, plus acceptance of independent Convex evaluator. No shared implementation is patched. CP03 may lock metadata/tuning/online config after review. CP04 needs accepted protocol, baseline freeze, resource pilot/uncertainty budget and full evidence. T9/T15 are not closed by these diagnostics.
