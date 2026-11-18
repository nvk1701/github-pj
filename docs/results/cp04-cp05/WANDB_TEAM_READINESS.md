# CP05.6 W&B team readiness audit

Read-only observation: 2026-11-17T08:21:27.499710+00:00; Forge SDK 0.31.0, API endpoint `https://api.forge.coreweave.com`. Project: [gr_ml_29_37_37/ml-distribution-shift](https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift). No runs, tags, groups, summaries, permissions or artifacts were edited and no experiments were started. The account queried by its own SDK login was `dangkhoi`; no teammate credentials were requested or used.

## Complete 65-run inventory

[All 65 runs and actual URLs](data/forge_run_inventory.csv), [count table](data/inventory_counts.csv), [all logged artifact metadata](data/forge_artifact_inventory.csv). Each count partition below totals 65. `unset` is an observed missing field, not an inferred experiment/category. Tag counts overlap.

| Field | Value | Runs |
|---|---|---:|
| experiment | <unset> | 2 |
| experiment | convex | 63 |
| category | <unset> | 2 |
| category | reproduction | 45 |
| category | sensitivity | 15 |
| category | smoke | 3 |
| group | <unset> | 65 |
| job_type | <unset> | 64 |
| job_type | smoke-test | 1 |
| uploader_username | dangkhoi | 65 |
| state | failed | 1 |
| state | finished | 64 |
| local_evidence_checkpoint | <unset> | 5 |
| local_evidence_checkpoint | CP04 | 45 |
| local_evidence_checkpoint | CP05 | 15 |

| Tag | Memberships |
|---|---:|
| colab | 1 |
| convex-cp03-v1 | 63 |
| estimated | 63 |
| lab1 | 65 |
| paper | 63 |
| reproduction | 45 |
| sensitivity | 15 |
| smoke | 3 |
| smoke-test | 2 |

Exactly 65 = 45 CP04 reproduction + 15 CP05 sensitivity + 2 legacy T2 integration tests with no category/experiment + 3 explicitly categorized Convex smoke runs. All 60 CP04/CP05 IDs match preserved local proof/config/scientific-hash/source metadata and are finished with completed=true. The two unclassified legacy tests carry `smoke-test` tags; they must not be reported as category=`smoke`. There are no scientific runs explicitly identified as Linear or Nonconvex. All 65 uploader identities are `dangkhoi`; labels, run names and Colab tags do not demonstrate another person's authentication.

| Extra run | Actual name | Experiment | Category | State | Tags |
|---|---|---|---|---|---|
| [kc5iilt2](https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift/runs/kc5iilt2) | smoke-khoi-seed42 | unset | unset | finished | lab1, smoke-test |
| [1a7wn7fi](https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift/runs/1a7wn7fi) | smoke-colab-khoi-seed42 | unset | unset | finished | colab, lab1, smoke-test |
| [h2tcymgy](https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift/runs/h2tcymgy) | cp03-online-smoke-convex-cp03-v1-d2-convex-seed0 | convex | smoke | failed | convex-cp03-v1, estimated, lab1, paper, smoke |
| [fg6ezqxp](https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift/runs/fg6ezqxp) | cp03-online-smoke-convex-cp03-v1-d2-convex-seed0 | convex | smoke | finished | convex-cp03-v1, estimated, lab1, paper, smoke |
| [o0ln0580](https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift/runs/o0ln0580) | cp035-wandb-e2e-smoke-1f0bfb77b5 | convex | smoke | finished | convex-cp03-v1, estimated, lab1, paper, smoke |

- `kc5iilt2`: legacy T2 test, method `smoke-test`, seed42, job_type=`smoke-test`; no Convex scientific protocol/hash. Its only logged artifact is backend history.
- `1a7wn7fi`: legacy Colab-tagged T2 test, method `smoke-test`, seed42, job_type unset; same uploader `dangkhoi`. A `colab` tag supports the declared context but is not independent runtime attestation. Its only logged artifact is backend history.
- `h2tcymgy`: CP03 online smoke, failed and completed=false, source `e802cd89...`; a committed four-file scientific artifact exists, but that does not turn the failed run into verified completion.
- `fg6ezqxp`: successful CP03 online smoke, completed=true, source `3e1f0b59...`, four-file scientific artifact. It is outside the official 45-cell matrix.
- `o0ln0580`: successful CP03.5 end-to-end smoke, completed=true, frozen source `7e0499c450a3e503ab09d4c6c3d272e6576b4432`, five-file scientific artifact. It is also outside the 60 results runs.

Do not delete/relabel the failed run or count these five extras as additional scientific repetitions.

## Artifact inventory and anonymous access

| Run set | Artifact type | Artifacts |
|---|---|---:|
| CP04 | convex-evidence | 45 |
| CP04 | wandb-history | 16 |
| CP05 | convex-evidence | 15 |
| CP05 | wandb-history | 10 |
| other | convex-evidence | 3 |
| other | wandb-history | 4 |

The complete live listing contains 63 `convex-evidence` artifacts and 30 backend `wandb-history` artifacts (93 artifacts; still 65 runs). All 60 verified attempts still have exactly one COMMITTED scientific artifact with the same expected name. Backend history is not an extra scientific run or a second scientific attempt.

Operational caveat: the frozen `convex_reconcile.py` checks the total number of logged artifacts, not just scientific type. That check passed at original CP04/CP05 verification time. Later backend history additions mean a fresh invocation may now reject an otherwise unchanged run at its total-count guard. CP05.6 does not modify that verifier, weaken checks, rerun experiments, or silently claim a fresh 60/60 verifier pass. A separately reviewed future maintenance change should distinguish artifact types while preserving exactly-one scientific artifact/name/state/hash checks.

Public rendered access: PASS for the actual project and CP04 run [br6md2c7](https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift/runs/br6md2c7), using a newly created empty Chrome profile plus incognito context, no login and no user browser profile. The project rendered 65 listed runs and charts; the run rendered Finished, reproduction/lab1 tags and 27 charts. Login/Sign up remained visible. Browser request audit observed Basic headers generated with empty/undefined placeholder passwords on trace/analytics requests, not usable account credentials; raw header values were not exported. This is an anonymous rendered-content observation, not proof that every run page or artifact UI tab was individually exercised.

Anonymous API access: raw HTTP session with trust_env=false, no authentication/cookies/netrc supplied; project access=USER_READ and viewer=null. Anonymous manifest query, manifest download and all five artifact payload downloads passed for CP04 `br6md2c7` and CP05 [7i4l64n4](https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift/runs/7i4l64n4). Both sets of config.json/metrics.csv/history.npz/figure.png/checksums.json SHA256 values match preserved local files (10/10). Storage capability URLs remained in memory and are excluded from publication. This is a two-artifact anonymous payload test, not an all-60 anonymous-download claim. An initial local inspection harness incorrectly marked manifest success false after an unrelated unsupported schema-introspection lookup; the corrected stage-separated follow-up obtained the actual 10 successful payload comparisons without changing server data.

Raw browser screenshots, safe audit JSON and hash checks remain in the local `_reports/cp056_evidence` review folder; they are excluded from this Git publication.

## T2 #6: evidence and remaining acceptance

[Issue #6](https://github.com/nvk1701/github-pj/issues/6) requires a public shared project, invitations for Khoa/Hoa, shared init_run(config) with lab1 tags, a Colab test with personal Secrets, and a test run from each of three people. Group/job_type are an organization proposal here, not literal requirements in that issue. The issue was open when inspected and was not edited.

| Item | Evidence/status |
|---|---|
| Common project | PASS: all 65 in the same entity/project; no alternate project created |
| lab1 tagging | PASS: 65/65 carry lab1 |
| Public rendered content | PASS for project and one actual verified run, as scoped above |
| Anonymous scientific artifacts | PASS for two existing artifacts, 10/10 payload hash matches |
| Existing group/job_type | All 65 groups unset; job_type unset64, smoke-test1; not retroactively changed |
| Membership Khoa | `nkhoa8508` appears with admin=true; team-to-person mapping should be confirmed by Khoa himself |
| Membership Hoa | No Hoa account identified in the two-member listing; invitation/acceptance not demonstrated |
| Independent Khoa logging | NOT VERIFIED: no run attributed to a second uploader; membership does not prove successful self-login/write |
| Independent Hoa logging | NOT VERIFIED: neither membership nor self-logged run demonstrated |
| Colab | Existing legacy run is colab-tagged under dangkhoi; teammates still need their own Secret and runtime verification |
| Pinned environment | Existing requirements/source preserved; CP05.6 is not a new environment rebuild or certification |
| Three-person Definition of Done | NOT MET by current evidence; do not close T2 |

Only `dangkhoi` and `nkhoa8508` were returned as current members, both with admin=true. This is a team listing, not a test of every project-scoped write permission. No teammate account was impersonated and no write-permission probe was run.

## Proposed convention for future authorized runs

One entity/project, three stable groups: `linear`, `convex`, `nonconvex`; group describes experiment family, not uploader, seed or checkpoint. Use job_type=`reproduction`, `sensitivity` or `integration-smoke`; tags include `lab1`, the protocol/version, category, checkpoint and optionally a human `member:` label. The label is metadata, never evidence of authenticated ownership. Current 65 runs retain their original metadata and existing scientific protocols/hashes.

The existing `core/tracking.py` accepts flat config keys including group/job_type; explicit config overrides `WANDB_GROUP`/`WANDB_JOB_TYPE`. Endpoint/login are SDK concerns outside config. Do not pass nested wandb/config/settings, base_url, resume, id or credentials. Use lists for tags. The official Convex CLI currently has no group/job_type flags: do not invent flags or edit frozen YAML. A future authorized invocation may use the supported environment fallback where its caller leaves these keys absent, after checking that invocation's resolved metadata. It must not change an existing artifact or frozen run.

The following are documentation examples compatible with the current wrapper; no call was made to create a run. Method names here are illustrative metadata for a future integration test, not scientific result labels.

```python
from core.tracking import init_run

configs = [{'entity': 'gr_ml_29_37_37',
  'project': 'ml-distribution-shift',
  'mode': 'online',
  'lab': 'lab1',
  'experiment': 'linear',
  'group': 'linear',
  'method': 'sgd',
  'seed': 42,
  'job_type': 'integration-smoke',
  'tags': ['lab1', 'integration-smoke', 'cp056-followup', 'member:<your-account>']},
 {'entity': 'gr_ml_29_37_37',
  'project': 'ml-distribution-shift',
  'mode': 'online',
  'lab': 'lab1',
  'experiment': 'convex',
  'group': 'convex',
  'method': 'convex',
  'seed': 42,
  'job_type': 'integration-smoke',
  'tags': ['lab1', 'integration-smoke', 'cp056-followup', 'member:<your-account>']},
 {'entity': 'gr_ml_29_37_37',
  'project': 'ml-distribution-shift',
  'mode': 'online',
  'lab': 'lab1',
  'experiment': 'nonconvex',
  'group': 'nonconvex',
  'method': 'sgd',
  'seed': 42,
  'job_type': 'integration-smoke',
  'tags': ['lab1', 'integration-smoke', 'cp056-followup', 'member:<your-account>']}]

# Example only: select one config in a future authorized teammate test.
# run = init_run(configs[0])
# run.log({"integration_probe": 1}, step=0)
# run.finish()
```

Set `WANDB_BASE_URL=https://api.forge.coreweave.com` outside config. Each teammate authenticates personally using SDK login or their own Colab Secret; never put a key into config, source, output table or a shared notebook. A later authorized test must finish successfully, show that teammate's real uploader, correct entity/project/group/job_type/lab1, and preserve the resulting URL as evidence. Merely copying the example or changing a member tag does not satisfy T2.

Khoa: confirm his account/member identity, independently authenticate and log one authorized test, then perform the separate T19 scientific cross-check under the reviewed protocol. Hoa: identify his own account, have an authorized team admin invite it, accept personally, then authenticate/log his own authorized test. Admin: verify least-needed project write access after acceptance. No new experiment or test run is launched as part of CP05.6.

Documentation publication readiness and team acceptance are separate decisions: these frozen-results documents can be ready for review/push while T2 and T19 remain open. The checkpoint does not claim team readiness is complete.
