# T10 — Computational complexity and implementation costs

Review-ready for [Issue13](https://github.com/nvk1701/github-pj/issues/13). All operation counts below are derived here from the cited update/source, not quoted runtime theorems. Bibliography numbers are provisional in [references.md](references.md). Regret bounds and computational complexity are different quantities.

## Model and assumptions

T is horizon, d parameter dimension, B training batch and E evaluation batch. Dense floating-point arithmetic has constant scalar cost; a gradient oracle costs C_g and a generic projection C_Pi(d). Project ball projection costs Theta(d). Complexity does not include tuning, discovering bounds or estimating a general distribution oracle unless explicitly stated. K=ceil(1/kappa) counts Euler substeps in Algorithm1; keep K variable when comparing discretizations.

## Comparable per-step work

| Method / component | Rate computation | Update/oracle work | State beyond scalar controls | Evidence / assumptions |
|---|---|---|---|---|
| Constant and inverse-sqrt baselines | Theta(1) | C_g + Theta(d) + C_Pi(d) | Theta(d) model | [4] core/schedules.py ConstantSchedule/InverseSqrtSchedule; projected SGD |
| Convex two thresholds | Theta(1) scalar exp/log/sqrt operations | Same gradient/projection; shift estimation separate | Theta(1) scheduler; model Theta(d) | [1] Proposition4.3; [4] ConvexSchedule.thresholds/step |
| Learner-displacement EMA | Not a separate learning rule | Theta(d) difference/norm, scalar EMA | Theta(d) previous learner | [4] core/drift.py EMADriftEstimator; vector input assumed available |
| Linear envelope Algorithm1 | Theta(K) scalar Euler work | Gradient/gamma cost separate | Theta(1) envelope plus model/estimator | [1] Algorithm1; not the cost of solving a general HJB PDE |
| Non-convex closed-form scalar rate | Theta(1) if scalar loss/shift/variance supplied | Computing those inputs and DNN gradient excluded | Theta(1) rate state | [1] Eq25; not a full neural-network training bound |
| Yang noisy-gradient projected update | Theta(1) scalar rate | C_g + Theta(d) + C_Pi(d) | Theta(d) model | [2] §3.2 Eq13; only this update, not bandit/META variants |
| Besbes restarted OGD | Theta(1) rate/counter | C_g + Theta(d) + C_Pi(d); reset Theta(d) | Theta(d) OGD state | [3] restart principle §3 and OGD instantiation §4 |

The schedule-only Convex overhead is constant under the fixed-precision scalar model. The practical estimator has vector cost, and general online oracle/projection work may dominate. For restarted OGD with epochs of Delta steps, at most ceil(T/Delta) vector resets add O(d ceil(T/Delta)), not an extra multiplier Delta on every gradient call. This is a comparison of computational steps, not a claim that the methods share identical regret assumptions, variation measures or empirical speed.

## Dense logistic training

In [4], core/convex_eval.py logistic_loss_gradient computes X theta and X-transpose residual: Theta(Bd), plus Theta(B) scalar loss operations. Sampling a Gaussian batch also costs Theta(Bd). Learner update, ball projection and EMA add Theta(d). Therefore one T-step dense training is Theta(TBd), with fixed numbers of passes and no d-by-d Hessian. Constant scalar schedule cost does not make dense SGD O(T).

An online-only implementation can keep model/previous-vector state Theta(d) and active batch Theta(Bd). The actual frozen runner stores full theta/reference/oracle paths, so its trajectory storage is Theta(Td), not just Theta(d). It does not store every training batch; no Theta(TBd) data-storage claim follows.

## Evaluation and numerical oracle are separate work

Sources [4]: core/convex_eval.py evaluate_history; core/convex_population.py gaussian_population_oracle, GaussianPopulationEvaluator, _integral, _moment. Evaluation replays the training stream and draws an independent evaluation stream. With a constant number of losses/comparators, replay/MC work is Theta(T(B+E)d); it does not update a new learner. Dashboard stride20 does not reduce full-resolution arrays or evaluation to T/20.

Let U<=T be the number of exact reference norms cached, I_o the root-solver iteration count, Q_o/Q_r quadrature integrand evaluation counts for root/reference calculations, and Q_theta,t the learner-risk integral work at t. These depend on tolerances and geometry; adaptive quadrature has no fixed tolerance-independent cost asserted here.

| Evaluation component | Work under this model | Memory/storage |
|---|---|---|
| Reference path / radial embedding | O(Td) | Theta(Td) |
| Oracle/cache preparation | O(Td + U(I_o Q_o + Q_r)) | O(Td+U) |
| Learner population risk | O(Td + sum_t Q_theta,t) | O(T) scalar metric arrays beyond paths |
| Training replay and independent MC | Theta(T(B+E)d) | O(max(B,E)d) active batches |
| Fixed-count scalar metrics / CSV | Theta(T) values | Theta(T) |
| Stored trajectories | Theta(Td) values | Theta(Td); NPZ compression ratio is data dependent |

Gaussian rotational symmetry reduces integrals to scalar quadrature, but vector norms/dot products still cost O(d). The evaluator is not dimension-free O(1). A useful algorithm-level memory bound for this runner is O(Td + max(B,E)d + T + U), excluding Python/SDK/plot allocator overhead and temporary library buffers. Measured RSS is not an exact count of these arrays.

## Matrix totals and measured time

For CP04, nine runs per dimension give a dense training feature-visit proxy 9*T*B*sum(d)=3,142,656,000, sum(d)=682. CP05's15 d128 runs give983,040,000. Replay/MC proxies replacing B by B+E are9,427,968,000 and2,949,120,000. These proxies are not FLOP counts, number of oracle calls or time predictions; the integrals, multiple passes, nonlinear functions and I/O remain separate.

The frozen result package [4], [RESULTS](results/cp04-cp05/RESULTS.md), reports actual scientific outputs. Local execution records measured CP04 runner24.03min, verifier14.12min, end-to-end gate/results44.51min, sampled process-tree peak501.6MiB; CP05 runner7.86min and verifier4.36min, peak428.9MiB. Timing provenance is the local CP04_EXECUTION_REPORT.md and CP05_NIGHT_EXECUTION_REPORT.md under the workspace's external _reports folder, not an author-paper runtime claim; reviewers need those local records to audit the measurements. They depend on this CPU/software/network and are not a benchmark against [2]/[3], a cost theorem or a GPU speedup claim. Backend artifacts/upload/download/plotting may affect wall time independently of SGD complexity.

## Defense and acceptance boundaries

“O(1) schedule” means scalar rate selection after inputs exist. “EMA O(d)” includes computing learner displacement. “Theta(TBd) training” assumes dense logistic data; a DNN needs its own forward/backward cost model. No universal runtime speedup over predecessors is established. Team review must consolidate IEEE numbering and align T4's comparison language; no Khoa/Hòa file or original workbook was edited.
