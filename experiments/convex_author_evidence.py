"""Validate external Colab exports without loading notebook source or pickle data."""

from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.convex_author_aligned import (
    Case, DIAGNOSTIC, phases_with_data, reference_phases, train,
)
from scripts.export_convex_author_colab import SNAPSHOT_SHA256, URL


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_export(root, kind, dimension, cfg):
    sidecar = root / f"{kind}-d{dimension}.json"
    path = sidecar.with_suffix(".npz")
    if not sidecar.is_file() and not path.is_file():
        return None
    if not sidecar.is_file() or not path.is_file():
        raise ValueError(f"incomplete Colab export: {sidecar.stem}")
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        env = meta["environment"]
        executed = datetime.fromisoformat(meta["executed_at_utc"])
        valid = (
            meta["author_commit"] == cfg["author_commit"]
            and meta["author_blob"] == cfg["author_blob"]
            and meta["notebook_sha256"] == SNAPSHOT_SHA256
            and meta["author_url"] == URL
            and meta["kind"] == kind and type(meta["dimension"]) is int
            and meta["dimension"] == dimension
            and executed.utcoffset() == timedelta(0)
            and env["runtime"] == "google-colab"
            and all(isinstance(env.get(k), str) and env[k].strip()
                    for k in ("python", "numpy", "tensorflow"))
        )
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"invalid Colab evidence metadata: {sidecar.name}") from exc
    if not valid:
        raise ValueError(f"invalid Colab evidence metadata: {sidecar.name}")
    if sha256(path) != meta.get("sha256"):
        raise ValueError(f"export checksum mismatch: {path.name}")
    return meta, path


def require_array(array, shape, dtypes, name):
    if array.shape != shape or array.dtype.name not in dtypes or not np.isfinite(array).all():
        raise ValueError(f"invalid array {name}: {array.shape}, {array.dtype}")


def discrepancies(actual, expected, tolerance):
    abs_tol, rel_tol = tolerance
    actual, expected = actual.astype(np.float64), expected.astype(np.float64)
    error = np.abs(actual - expected)
    allowed = abs_tol + rel_tol * np.abs(expected)
    return {"max": float(error.max()), "median": float(np.median(error)),
            "p99": float(np.quantile(error, .99)), "out_of_tolerance": int((error > allowed).sum())}


def inspect_evidence(root, cfg):
    parity, native, matched = {}, {}, {}
    steps, batch, phase_length = cfg["steps"], cfg["batch_size"], cfg["phase_length"]
    for d in cfg["dimensions"]:
        export = load_export(root, "native", d, cfg)
        if export:
            meta, path = export
            with np.load(path, allow_pickle=False) as record:
                learner, reference, curve = (record[k] for k in ("loss", "reference_loss", "M0"))
                for array in (learner, reference, curve):
                    require_array(array, (steps,), ("float32",), "native loss/M0")
                np.testing.assert_array_equal(curve, np.cumsum(learner - reference, dtype=np.float32))
                native[d] = {"metadata": meta, "M0": curve.copy()}
        if d not in cfg["parity_dimensions"]:
            continue
        export = load_export(root, "diagnostic", d, cfg)
        if export is None:
            parity[d] = {"status": "UNVERIFIED"}
            continue
        meta, path = export
        checks = {}
        case = Case(DIAGNOSTIC, d, 0)
        with np.load(path, allow_pickle=False) as stored:
            fixture = {name: stored[name] for name in cfg["parity_tolerances"]}
            shapes = {"X": (steps, batch, d), "y": (steps, batch), "initial": (d,),
                      "reference": (steps, d), "theta": (steps, d), "post_theta": (steps, d),
                      "gradient": (steps, d), "thresholds": (steps, 2)}
            for name, array in fixture.items():
                types = ("float32", "float64") if name in {"thresholds", "lr", "gamma"} else ("float32",)
                require_array(array, shapes.get(name, (steps,)), types, name)
            phases = reference_phases(cfg, d)
            computed_ref = np.repeat(phases.astype(np.float32), phase_length, axis=0)
            checks["reference"] = discrepancies(computed_ref, fixture["reference"],
                                                  cfg["parity_tolerances"]["reference"])
            for name in ("X", "y"):
                checks[name] = {"max": 0., "out_of_tolerance": 0, "per_phase": []}
            for i, (x, y) in enumerate(phases_with_data(cfg, case, phases, cfg["label_noise_std"],
                                                        native_rng=True)):
                sl = slice(i * phase_length, (i + 1) * phase_length)
                for name, samples in (("X", x), ("y", y)):
                    expected = fixture[name][sl]
                    stats = discrepancies(samples.reshape(expected.shape), expected,
                                          cfg["parity_tolerances"][name])
                    checks[name]["per_phase"].append(stats)
                    checks[name]["max"] = max(checks[name]["max"], stats["max"])
                    checks[name]["out_of_tolerance"] += stats["out_of_tolerance"]
            history, _ = train(cfg, case, supplied=fixture)
            for name, tolerance in cfg["parity_tolerances"].items():
                if name not in checks:
                    checks[name] = discrepancies(history[name], fixture[name], tolerance)
            parity[d] = {
                "status": "FAILED_VALIDATION" if any(v["out_of_tolerance"] for v in checks.values()) else "VERIFIED",
                "metadata": meta, "errors": checks,
                "replay_M0": float(history["M0"][-1]),
                "diagnostic_M0": float(fixture["M0"][-1]),
                "reference_M0_max_error": checks["M0"]["max"],
            }
            if (d in native and parity[d]["status"] == "VERIFIED"
                    and meta.get("native_sha256") == native[d]["metadata"]["sha256"]
                    and meta["environment"] == native[d]["metadata"]["environment"]
                    and not discrepancies(fixture["M0"], native[d]["M0"],
                                          cfg["parity_tolerances"]["M0"])["out_of_tolerance"]):
                matched[d] = history["M0"].copy()
    return parity, native, matched
