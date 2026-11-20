"""Colab-only wrapper to execute and export the immutable author notebook.

This file is project-owned instrumentation. The notebook is downloaded into
the Colab runtime and executed there. See docs/convex_author_v1.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from urllib.request import urlopen

COMMIT = "32ae9da816fc26e6a728b27fcd20255536b2fc41"
BLOB = "0932fa3227867cd835cf67728bef3e96b0230cfd"
SNAPSHOT_SHA256 = "c1493733e42d3226146458625431491585f0babffb8db109defdb235f13ac874"
URL = (f"https://raw.githubusercontent.com/fahrbach/learning-rate-schedules/{COMMIT}"
       "/src/regression/regression_experiments.ipynb")
DIMENSIONS = (2, 8, 32, 128, 512)
CELLS = {0: "dnZz5YnDr_z3", 2: "bNK7yUS2swoc", 3: "0yiEN799Q7JN",
         6: "oUiIV7TPssCN", 8: "bZENdZi25_0I", 23: "TaJuBJa1OMW4"}


def export(root, kind, dimension, snapshot_hash, env, arrays, native_sha=None):
    import numpy as np

    stem = root / f"{kind}-d{dimension}"
    path = stem.with_suffix(".npz")
    np.savez_compressed(path, **arrays)
    with path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    meta = {"author_commit": COMMIT, "author_blob": BLOB, "notebook_sha256": snapshot_hash,
            "kind": kind, "dimension": dimension, "author_url": URL,
            "executed_at_utc": datetime.now(timezone.utc).isoformat(), "environment": env,
            "sha256": checksum, "native_sha256": native_sha}
    stem.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def instrument(scope, source, root, snapshot_hash, env):
    import numpy as np

    original = scope["run_online_learning_algorithm"]
    lines = source.splitlines()
    apply = next(i for i, line in enumerate(lines, 1) if "vars.assign_sub(step_size * grad)" in line)
    after = next(i for i, line in enumerate(lines, 1) if i > apply and line.lstrip().startswith("if False:"))

    def wrapped(X, y, thetas, batch_size, model_type, learning_rate_schedule):
        deltas = {k: [] for k in ("gamma", "thresholds", "gradient", "post_theta")}

        def line_trace(frame, event, arg):
            if event == "line" and frame.f_lineno == apply:
                local = frame.f_locals
                deltas["gamma"].append(local["gamma"])
                deltas["thresholds"].append([local["tau_1t"], local["tau_2t"]])
                deltas["gradient"].append(local["grad"].numpy().reshape(-1))
            elif event == "line" and frame.f_lineno == after:
                deltas["post_theta"].append(frame.f_locals["vars"].numpy().reshape(-1))
            return line_trace

        def trace(frame, event, arg):
            return line_trace if event == "call" and frame.f_code is original.__code__ else None

        previous_trace = sys.gettrace()
        sys.settrace(trace)
        try:
            result = original(X, y, thetas, batch_size, model_type, learning_rate_schedule)
        finally:
            sys.settrace(previous_trace)
        loss, ref_loss, theta, ref, lr = result
        d = X.shape[1]
        native = root / f"native-d{d}.json"
        native_sha = json.loads(native.read_text(encoding="utf-8"))["sha256"] if native.is_file() else None
        arrays = {k: np.asarray(v) for k, v in deltas.items()}
        arrays.update(X=X.reshape(-1, batch_size, d), y=y.reshape(-1, batch_size),
                      initial=theta[0], theta=theta, reference=ref, lr=lr, loss=loss,
                      reference_loss=ref_loss, M0=np.cumsum(loss - ref_loss, dtype=np.float32))
        export(root, "diagnostic", d, snapshot_hash, env, arrays, native_sha)
        return result

    scope["run_online_learning_algorithm"] = wrapped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=("native", "diagnostic"))
    parser.add_argument("--output", type=Path, default=Path("/content/convex-author-evidence"))
    args = parser.parse_args()

    import google.colab  # This entry point is deliberately scoped to Colab.
    import matplotlib
    import numpy as np
    import psutil
    import seaborn
    import tensorflow as tf

    # Native d512 contains two 1000-MiB persistent covariate/reference matrices
    # before NumPy concatenation temporaries and TensorFlow copies.
    minimum = 10 * 2**30 if args.kind == "native" else 2 * 2**30
    if psutil.virtual_memory().available < minimum:
        raise MemoryError(f"{args.kind} requires at least {minimum / 2**30:.0f} GiB available before starting")
    env = {"runtime": "google-colab", "python": sys.version, "numpy": np.__version__,
           "tensorflow": tf.__version__, "matplotlib": matplotlib.__version__,
           "seaborn": seaborn.__version__, "platform": platform.platform(),
           "devices": [d.name for d in tf.config.list_physical_devices()]}
    with urlopen(URL, timeout=30) as response:
        snapshot = response.read(10 * 2**20)
    git_blob = hashlib.sha1(f"blob {len(snapshot)}\0".encode() + snapshot).hexdigest()
    if git_blob != BLOB:
        raise ValueError(f"unexpected upstream blob: {git_blob}")
    content_hash = hashlib.sha256(snapshot).hexdigest()
    if content_hash != SNAPSHOT_SHA256:
        raise ValueError("snapshot SHA-256 mismatch")
    notebook = json.loads(snapshot)
    sources = {}
    for index, id_ in CELLS.items():
        cell = notebook["cells"][index]
        if cell["metadata"]["id"] != id_:
            raise ValueError(f"unexpected snapshot cell {index}")
        sources[index] = "".join(cell["source"])

    args.output.mkdir(parents=True, exist_ok=True)
    scope = {"__name__": "__main__"}
    for index in (0, 2, 3, 6, 8):
        exec(compile(sources[index], f"<author-cell-{index}>", "exec"), scope)
    source = sources[23]
    if args.kind == "diagnostic":
        instrument(scope, sources[8], args.output, content_hash, env)
        assignment = "DIMENSION_LIST = [2, 8, 32, 128, 512]"
        if source.count(assignment) != 1:
            raise ValueError("unexpected diagnostic loop")
        source = source.replace(assignment, "DIMENSION_LIST = [2, 8, 32]")
    try:
        exec(compile(source, "<author-cell-23>", "exec"), scope)
    finally:
        if args.kind == "native":
            # Lists persist when a Python-level exception interrupts a late dimension.
            runs = zip(DIMENSIONS, scope.get("losses_hat_list", []),
                       scope.get("losses_opt_list", []), scope.get("regret_list", []))
            for d, loss, reference_loss, m0 in runs:
                export(args.output, "native", d, content_hash, env,
                       {"loss": loss, "reference_loss": reference_loss, "M0": m0})


if __name__ == "__main__":
    main()
