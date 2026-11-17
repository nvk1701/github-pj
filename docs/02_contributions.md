# T8 — Contributions and novelty

Review-ready for [Issue11](https://github.com/nvk1701/github-pj/issues/11). Numbering [1]–[5] follows the provisional shared [bibliography](references.md); whole-team numbering still needs consolidation. This document summarizes author contributions, not the group's engineering achievements.

## Contribution table

| Paper contribution | Classification | Previous limitation / relevant comparison | Novelty in this paper | Exact support | Boundary |
|---|---|---|---|---|---|
| SGD dynamics with a moving linear-regression target | Theoretical / technical | A fixed landscape does not describe a drifting target | A shift-dependent SDE and moment dynamics | [1] §3, Proposition3.1, Theorem3.2 | Approximation under stated model/scaling assumptions |
| Rate control for a scalar envelope | Algorithmic / theoretical | Describing dynamics does not select a schedule | Optimal control of the specified envelope, implemented with Euler substeps | [1] Theorem3.3, Algorithm1, Eq12–13 | Envelope optimality does not imply universal finite-SGD optimality |
| General convex shift-aware schedule | Algorithmic / theoretical | A global variation budget alone does not specify each local rate | Sequential bound minimization gives two thresholds and projection of previous LR | [1] Theorem4.2, Proposition4.3, Eq15–20 | Requires smoothness, variance/distance bounds and admissible information |
| Complementary regret lower bound | Theoretical | An upper bound alone does not assess tightness | A lower expression under additional curvature assumptions | [1] Assumption4.6, Theorem4.7, Eq21 | Strong convexity is additional; not every convex logistic setting satisfies it |
| Non-convex schedule criterion | Theoretical / algorithmic | Global-minimum comparison is generally unsuitable | Gradient-norm-based regret bound and a rate minimizing its stated bound | [1] §5, Theorem5.1, Eq22–25 | Stationarity criterion is not global-optimum recovery |
| Empirical illustrations under shift | Empirical | Theory needs examples of resulting behavior | Regression trajectories and neural-network illustrations | [1] §6.1, Figure4–5; AppendixE, Figure6 | Examples, not proof of superiority on every dataset |

## Relation to Khoa's T4

The existing [context document](01_context.md), §3–4, compares static decay, dynamic online convex optimization and continuous-time analyses. It is useful context; it must not be read as claiming all prior work assumes a stationary distribution. Yang et al. [2], §3.2, already study changing minimizers with noisy gradients. Besbes et al. [3], §3–4, already use variation budgets and restarted optimization. Thus neither dynamic regret nor adaptation to nonstationarity was invented by [1].

The local-rate thresholds and shift-aware control analysis are the relevant additions to compare, under their own assumptions. The predecessor papers use different feedback/variation models; no unconditional dominance or comparable numerical regret-rate ranking is asserted here. Khoa should review the strong wording in T4's predecessor/general-gap table before oral presentation; this checkpoint does not edit his document or attribute an unreviewed statement to him.

## What the project contributes separately

The group's float64 Gaussian population evaluator, numerical oracle, A/B comparator diagnostics, immutable attempts/checksums, verifier maintenance, CP04 matrix and CP05 sensitivity are project implementation/evidence work [4]. They are not author-paper novelty. Conversely, matching threshold algebra does not make the project's EMA/noise/comparator choices an exact reproduction of the author notebook [5].

For oral defense: explain the optimization target before saying “optimal”; name the required assumptions; distinguish regret bounds from observed curves; retain d512's baseline win. The T8 contribution/novelty table is ready for review, while human acceptance and T4 alignment remain open. No slide positions or scientific claims are inferred from the existence of this file alone.
