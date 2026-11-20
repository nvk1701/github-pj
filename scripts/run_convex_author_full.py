"""Orchestrate the post-freeze Convex study with explicit parity gates."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from time import perf_counter
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Limit default threading before NumPy imports. Allow an explicit override.
for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np

from core.convex_recovery import atomic_json, canonical_hash
from core.utils import load_config
from experiments.convex_author_aligned import (
    ABLATION, ALIGNED, DIAGNOSTIC, PILOT, Case, evaluate, matrix, settings, train,
)
from experiments.convex_author_evidence import inspect_evidence, require_array, sha256


BASE = "8686740adca5f6879775e52a500b28cb203123d0"
FROZEN = "docs/results/cp04-cp05"
CONFIG = "configs/convex_author_v1.yaml"
AUDIT = "docs/convex_author_v1/SOURCE_AUDIT.md"
SOURCES = (CONFIG, AUDIT, "requirements.txt", "scripts/run_convex_author_full.py",
           "experiments/convex_author_aligned.py", "experiments/convex_author_evidence.py",
           "experiments/convex_author_analysis.py", "tests/test_convex_author_alignment.py",
           "scripts/export_convex_author_colab.py",
           "core/convex_population.py", "core/drift.py", "core/schedules.py",
           "core/utils.py", "core/tracking.py", "core/convex_recovery.py")


def git(*args):
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", *args], cwd=ROOT
    )


def frozen_hashes():
    expected = git("ls-tree", "-r", "--name-only", BASE, FROZEN).decode().splitlines()
    if not expected:
        raise ValueError("frozen publication inventory is empty")
    actual = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / FROZEN).rglob("*") if p.is_file())
    if sorted(expected) != actual:
        raise ValueError("frozen publication inventory differs from base")
    hashes = {}
    for name in expected:
        blob = git("rev-parse", f"{BASE}:{name}").strip()
        if git("hash-object", "--path", name, name).strip() != blob:
            raise ValueError(f"frozen publication differs from base: {name}")
        hashes[name] = sha256(ROOT / name)
    return hashes


def config():
    # Lock the published v1 contract against a silent config replacement.
    cfg = load_config(ROOT / CONFIG)
    if canonical_hash(cfg) != CONFIG_HASH:
        raise ValueError("configuration differs from the registered v1 contract")
    return cfg


CONFIG_HASH = "bbc68aa1d54d14d4f9f1904863c91b74b84b93db0ee95d612b6cbdc8cd2fcea3"


def provenance(cfg):
    return {
        "git_commit": git("rev-parse", "HEAD").decode().strip(),
        "config_sha256": sha256(ROOT / CONFIG),
        "science_sha256": canonical_hash(cfg),
        "source_sha256": {name: sha256(ROOT / name) for name in SOURCES},
        "environment": {
            "python": sys.version, "platform": platform.platform(),
            "packages": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "matplotlib", "PyYAML", "wandb")},
            "threads": {k: os.environ[k] for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")},
        },
        "author_commit": cfg["author_commit"], "author_blob": cfg["author_blob"],
    }


def validate_history(stored, steps, dimension, metrics):
    vectors = {"initial": (dimension,)}
    vectors.update({name: (steps, dimension) for name in ("theta", "post_theta", "gradient", "reference")})
    for name, shape in vectors.items():
        require_array(stored[name], shape, ("float32",), name)
    require_array(stored["oracle"], (steps, dimension), ("float64",), "oracle")
    require_array(stored["thresholds"], (steps, 2), ("float64",), "thresholds")
    require_array(stored["projection_active"], (steps,), ("bool",), "projection_active")
    for name in ("loss", "reference_loss", "M0_step", "M0"):
        require_array(stored[name], (steps,), ("float32",), name)
    for name in ("M1", "M2", "M1_step", "M2_step", "correction", "gamma", "lr",
                 "m1_quadrature_budget", "m2_quadrature_budget", "oracle_parameter_error",
                 "oracle_suboptimality"):
        require_array(stored[name], (steps,), ("float64",), name)
    if (not np.array_equal(stored["initial"], np.zeros(dimension))
            or not np.array_equal(stored["theta"][0], stored["initial"])
            or not np.array_equal(stored["theta"][1:], stored["post_theta"][:-1])
            or not np.array_equal(stored["M0_step"], stored["loss"] - stored["reference_loss"])
            or not np.allclose(stored["M2_step"] - stored["M1_step"], stored["correction"],
                               atol=3e-15, rtol=0)):
        raise ValueError("history failed trajectory or comparator identity")
    for name, dtype in (("M0", np.float32), ("M1", np.float64), ("M2", np.float64)):
        series = stored[name]
        if (not np.array_equal(series, np.cumsum(stored[name + "_step"], dtype=dtype))
                or float(series[-1]) != metrics[name]):
            raise ValueError(f"history failed cumulative {name} identity")


def validated(root, case, proof):
    folder = root / "runs" / case.key
    path = folder / "record.json"
    if not path.is_file():
        if (folder / "history.npz").exists():
            raise ValueError(f"unsealed history retained; use a new output root: {case.key}")
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    keys = ("science_sha256", "config_sha256", "source_sha256", "environment")
    if (record["case"] != asdict(case) or record["status"] != "VERIFIED"
            or any(record["provenance"][key] != proof[key] for key in keys)
            or record["steps"] != proof["steps"] or record["batch"] != proof["batch_size"]
            or record["history_sha256"] != sha256(folder / "history.npz")):
        raise ValueError(f"incompatible or corrupt completed record: {case.key}")
    with np.load(folder / "history.npz", allow_pickle=False) as stored:
        validate_history(stored, proof["steps"], case.dimension, record["metrics"])
    return record


def execute(root, cfg, case, proof, evaluators):
    folder = root / "runs" / case.key
    folder.mkdir(parents=True, exist_ok=True)
    # Exclusive marker prevents concurrent writers in the same output root.
    lock = folder / ".writer"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        existing = validated(root, case, proof)
        if existing is not None:
            return existing
        start = perf_counter()
        history, data_hashes = train(cfg, case, native_rng=(case.protocol == DIAGNOSTIC))
        history.update(evaluate(cfg, case, history, evaluators))
        metrics = {name: float(history[name][-1]) for name in ("M0", "M1", "M2")}
        validate_history(history, cfg["steps"], case.dimension, metrics)
        path = folder / "history.npz"
        with tempfile.NamedTemporaryFile(dir=folder, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            np.savez_compressed(temporary, **history)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        record = {
            "case": asdict(case), "status": "VERIFIED", "provenance": proof,
            "date_utc": datetime.now(timezone.utc).isoformat(),
            "settings": settings(cfg, case), "steps": cfg["steps"], "batch": cfg["batch_size"],
            "training_dtype": "float32", "population_dtype": "float64", "initial_weights": "zero",
            "rng": "MT19937 seeded by phase, X then noise" if case.protocol == DIAGNOSTIC
            else "PCG64 independent X/noise SeedSequence([seed, dimension]).spawn(2)",
            "data": data_hashes, "history_sha256": sha256(path),
            "metrics": metrics,
            "diagnostics": {
                "lr_mean": float(history["lr"].mean()),
                "fraction_lr_ge_3_96": float((history["lr"] >= 3.96).mean()),
                "max_post_norm": float(np.linalg.norm(history["post_theta"], axis=1).max()),
                "projection_count": int(history["projection_active"].sum()),
                "correction": float(history["correction"].sum()),
                "M1_quadrature_budget": float(history["m1_quadrature_budget"].sum()),
                "M2_quadrature_budget": float(history["m2_quadrature_budget"].sum()),
                "max_oracle_parameter_error": float(history["oracle_parameter_error"].max()),
                "summed_oracle_suboptimality": float(history["oracle_suboptimality"].sum()),
            },
            "elapsed_seconds": perf_counter() - start,
        }
        atomic_json(folder / "record.json", record)
        (folder / "failure.json").unlink(missing_ok=True)
        return record
    except Exception as exc:
        atomic_json(folder / "failure.json", {"case": asdict(case), "error": type(exc).__name__})
        raise
    finally:
        lock.unlink()


def log_tracking(root, case, record, mode):
    folder = root / "runs" / case.key
    ledger = folder / f"wandb-{mode}.json"
    if mode == "disabled" or ledger.is_file():
        return
    from core.tracking import init_run
    import wandb

    endpoint = urlsplit(os.getenv("WANDB_BASE_URL", "https://api.wandb.ai"))
    if endpoint.scheme != "https" or not endpoint.hostname or endpoint.username or endpoint.password:
        raise ValueError("W&B endpoint must be an HTTPS URL without credentials")
    state = {"mode": mode, "state": "SDK_STARTED", "endpoint": endpoint.hostname}
    # 'x' reserves the upload before init. In-doubt events require reconciliation.
    with ledger.open("x", encoding="utf-8") as stream:
        json.dump(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        run = init_run({
            "mode": mode, "group": case.protocol, "experiment": "convex-author",
            "job_type": case.protocol, "name": case.key, **asdict(case),
            "tags": [case.protocol, case.method, f"d{case.dimension}", f"seed{case.seed}", case.intervention],
            "science_sha256": record["provenance"]["science_sha256"],
            "git_commit": record["provenance"]["git_commit"],
            "history_sha256": record["history_sha256"],
        })
        state["id"] = run.id
        atomic_json(ledger, state)
        with np.load(folder / "history.npz", allow_pickle=False) as history:
            for t in range(19, len(history["lr"]), 20):
                run.log({key: float(history[key][t]) for key in ("M0", "M1", "M2", "lr")}, step=t+1)
        run.summary["validated_history_sha256"] = record["history_sha256"]
        run.finish()
        state["state"] = "SDK_FINISHED"
        atomic_json(ledger, state)
        if mode == "online":
            remote = wandb.Api().run(f"{run.entity}/{run.project}/{run.id}")
            if (remote.state == "finished" and remote.id == run.id
                    and remote.summary.get("validated_history_sha256") == record["history_sha256"]):
                state.update(state="REMOTE_API_VERIFIED", url=remote.url)
                atomic_json(ledger, state)
    except Exception as exc:
        state["error"] = type(exc).__name__
        atomic_json(ledger, state)
        print(f"W&B {case.key}: {state['state']} ({type(exc).__name__})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    operations = parser.add_mutually_exclusive_group(required=True)
    for name in ("plan", "smoke", "full", "resume", "report-only"):
        operations.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--wandb-mode", choices=("disabled", "offline", "online"), default="disabled")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/convex-author-v1")
    parser.add_argument("--evidence-root", type=Path)
    args = parser.parse_args()
    root = args.output_root.resolve()
    if root == ROOT or ((root.is_relative_to(ROOT)) and not root.is_relative_to(ROOT / "outputs")):
        parser.error("an in-repository output root must be under outputs/")
    if root.is_relative_to(ROOT / FROZEN) or (ROOT / FROZEN).is_relative_to(root):
        parser.error("output would overlap the protected publication")
    before = frozen_hashes()
    cfg = config()
    proof = provenance(cfg)
    proof.update(steps=cfg["steps"], batch_size=cfg["batch_size"])
    cases = matrix(cfg)
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "plan.json", {"provenance": proof, "frozen_sha256": before,
                                     "cases": [asdict(c) for c in cases]})
    if args.plan:
        print("Planned: I=45, A=36, R=5 external")
        return 0
    parity, native, matched = inspect_evidence((args.evidence_root or root / "colab").resolve(), cfg)
    parity_verified = all(row["status"] == "VERIFIED" for row in parity.values())
    failed_parity = any(row["status"] == "FAILED_VALIDATION" for row in parity.values())
    author_gate = "VERIFIED" if parity_verified else "FAILED_VALIDATION" if failed_parity else "UNVERIFIED"
    gates = {"F0_source": "VERIFIED", "F1_unit": "UNVERIFIED", "F2_fixture": author_gate,
             "F3_full_replay": author_gate, "F4_high_dimension": "UNVERIFIED",
             "F5_regression": "UNVERIFIED"}
    gate_path = root / "gate_evidence.json"
    have_pilot_failure = False
    any_failure = False

    def run_cases(batch, evaluators):
        failed = False
        for case in batch:
            try:
                record = execute(root, cfg, case, proof, evaluators)
            except (RuntimeError, ValueError, OSError) as exc:
                failed = True
                print(f"{case.key}: {type(exc).__name__}: {exc}", flush=True)
                continue
            print(f"{case.key}: {record['metrics']}", flush=True)
            if case.protocol != DIAGNOSTIC:
                log_tracking(root, case, record, args.wandb_mode)
        return failed

    if args.report_only and gate_path.is_file():
        saved = json.loads(gate_path.read_text(encoding="utf-8"))
        if saved["source_sha256"] == proof["source_sha256"]:
            for key in ("F1_unit", "F4_high_dimension", "F5_regression"):
                gates[key] = saved["gates"][key]
    if not args.report_only:
        scope = ["-s", "tests"]
        if args.smoke:
            scope += ["-p", "test_convex_author_alignment.py"]
        subprocess.run([sys.executable, "-m", "unittest", "discover", *scope],
                       cwd=ROOT, check=True)
        gates["F1_unit"] = "VERIFIED"
        if not args.smoke:
            gates["F5_regression"] = "VERIFIED"
        seed = cfg["seeds"][0]
        pilots = [Case(PILOT, d, seed) for d in cfg["dimensions"]]
        pilots += [Case(PILOT, 512, seed, m) for m in cfg["methods"] if m != "convex"]
        pilots += [Case(DIAGNOSTIC, d, seed) for d in cfg["parity_dimensions"]]
        evaluators = {}
        have_pilot_failure = run_cases(pilots, evaluators)
        gates["F4_high_dimension"] = "FAILED_VALIDATION" if have_pilot_failure else "VERIFIED"
        if parity_verified and not args.smoke and not have_pilot_failure:
            any_failure = run_cases([c for c in cases if c.protocol == ALIGNED], evaluators)
            if not any_failure:
                any_failure = run_cases([c for c in cases if c.protocol == ABLATION], evaluators)
        atomic_json(gate_path, {"source_sha256": proof["source_sha256"], "gates": gates})
    if frozen_hashes() != before:
        raise ValueError("frozen publication byte hashes changed during execution")
    from experiments.convex_author_analysis import report
    report(root, ROOT / FROZEN, cfg, proof, before, cases, parity, native, matched, gates, validated)
    if any_failure or have_pilot_failure or failed_parity:
        return 1
    if (args.full or args.resume) and not parity_verified:
        print("Full matrices blocked: complete d=2,8,32 Colab diagnostic parity is required.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
