"""Build compact tables and figures exclusively from validated trajectories."""

from collections import defaultdict
import csv
from dataclasses import asdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.convex_recovery import atomic_json
from experiments.convex_author_aligned import ABLATION, ALIGNED, DIAGNOSTIC, PILOT, Case
from experiments.convex_author_evidence import sha256

METRICS = ("M0", "M1", "M2")
ARCHIVED = {2: .98582214, 8: 3.7352686, 32: 48.065575}


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def history(root, case, *keys):
    with np.load(root / "runs" / case.key / "history.npz", allow_pickle=False) as stored:
        return [stored[k] for k in keys]


def summary(group, metric):
    return {
        "n": len(group), "mean": float(np.mean([r[metric] for r in group])),
        "sample_sd": float(np.std([r[metric] for r in group], ddof=1)) if len(group) > 1 else "",
    }


def tracking(root, case):
    return {mode: json.loads(path.read_text(encoding="utf-8"))
            for mode in ("offline", "online")
            if (path := root / "runs" / case.key / f"wandb-{mode}.json").is_file()}


def verify_pairs(records):
    streams = {}
    for case, record in records.items():
        if case.protocol in (ALIGNED, PILOT):
            key = (case.protocol, case.dimension, case.seed)
            expected = streams.setdefault(key, record["data"])
            if record["data"] != expected:
                raise ValueError(f"unmatched method inputs: {case.key}")
        elif case.protocol == ABLATION:
            control = Case(ALIGNED, case.dimension, case.seed)
            if control not in records:
                raise ValueError(f"missing ablation control: {case.key}")
            baseline = records[control]["data"]
            noise_changed = case.intervention == "noise"
            if (record["data"]["X_sha256"] != baseline["X_sha256"]
                    or (record["data"]["y_sha256"] != baseline["y_sha256"]) != noise_changed):
                raise ValueError(f"unmatched ablation inputs: {case.key}")


def save(fig, figures, name):
    fig.tight_layout()
    fig.savefig(figures / f"{name}.png", dpi=160)
    plt.close(fig)


def supplemental_diagnostics(root, data, figures, records, seed):
    protocol = ALIGNED if sum(c.protocol == ALIGNED for c in records) == 45 else PILOT
    cases = [c for c in records if c.protocol == protocol and c.seed == seed]
    if not cases:
        return
    fig, ax = plt.subplots(figsize=(8, 4.4))
    correction_rows = []
    for case in sorted((c for c in cases if c.method == "convex"), key=lambda c: c.dimension):
        [correction] = history(root, case, "correction")
        cumsum = np.cumsum(correction)
        ax.plot(np.arange(1, len(cumsum) + 1), cumsum, label=f"d={case.dimension}")
        correction_rows.append({**asdict(case), "final_correction": float(cumsum[-1])})
    ax.set(title=f"{protocol}, seed {seed}, label-noise std=0.1",
           xlabel="Batch step", ylabel="Cumulative comparator correction (M2 - M1)")
    ax.legend(ncol=3)
    save(fig, figures, "comparator_correction")
    write_csv(data / "comparator_correction.csv",
              ["protocol", "dimension", "seed", "method", "intervention", "final_correction"],
              correction_rows)

    high = [c for c in cases if c.dimension == 512]
    if not high:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharex=True)
    high_rows = []
    for case in high:
        grad, theta, post, gamma, lr, m2, projected = history(
            root, case, "gradient", "theta", "post_theta", "gamma", "lr", "M2_step", "projection_active")
        grad_norm = np.linalg.norm(grad, axis=1)
        update_norm = np.linalg.norm(post - theta, axis=1)
        series = (grad_norm, update_norm, gamma, m2)
        step = np.arange(1, len(lr) + 1)
        for ax, values, label in zip(axes.flat, series, ("Gradient norm", "Update norm",
                                                        "Forward reference shift", "Per-step M2")):
            ax.plot(step, values, linewidth=.7, alpha=.8, label=case.method)
            ax.set_ylabel(label)
        for start, stop in ((0, 20), (900, 1100), (0, len(step))):
            window = slice(start, stop)
            high_rows.append({
                **asdict(case), "step_start": start + 1, "step_end": stop,
                "mean_grad_norm": float(grad_norm[window].mean()),
                "mean_update_norm": float(update_norm[window].mean()),
                "rms_update_norm": float(np.sqrt(np.mean(update_norm[window].astype(float)**2))),
                "mean_gamma": float(gamma[window].mean()),
                "mean_lr": float(lr[window].mean()), "sum_M2": float(m2[window].sum()),
                "projected_updates": int(projected[window].sum()),
            })
    axes[0, 0].legend(fontsize=8)
    fig.supxlabel("Batch step")
    fig.suptitle(f"{protocol}, d=512, seed {seed}, label-noise std=0.1")
    save(fig, figures, "d512_gradient_drift")
    write_csv(data / "d512_windows.csv", list(high_rows[0]), high_rows)


def figures_from_runs(root, target, cfg, records, native, matched, frozen_csv, paired):
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .25, "font.size": 10})
    official = [c for c in records if c.protocol == ALIGNED]
    seed = cfg["seeds"][0]
    sizes = cfg["dimensions"]
    pilot = [c for c in records if c.protocol == PILOT]
    shown = official if len(official) == 45 else pilot
    adaptive = [c for c in shown if c.method == "convex"]
    if adaptive:
        fig, ax = plt.subplots(figsize=(8, 4.4))
        for d in sizes:
            group = sorted((c for c in adaptive if c.dimension == d), key=lambda c: c.seed)
            if not group:
                continue
            curves = np.array([history(root, c, "M0")[0] for c in group])
            ax.plot(np.arange(1, curves.shape[1]+1), curves.mean(axis=0), label=f"d={d}, n={len(group)}")
            if len(group) > 1:
                sd = curves.std(axis=0, ddof=1)
                ax.fill_between(np.arange(1, len(sd)+1), curves.mean(axis=0)-sd, curves.mean(axis=0)+sd, alpha=.13)
        ax.set(title=f"{shown[0].protocol}: adaptive Convex", xlabel="Batch step",
               ylabel="Cumulative M0 (batch BCE difference)", yscale="symlog")
        ax.legend(ncol=2)
        save(fig, target, "adaptive_m0")

    if native:
        fig, ax = plt.subplots(figsize=(8, 4.4))
        for d, entry in sorted(native.items()):
            line, = ax.plot(np.arange(1, len(entry["M0"])+1), entry["M0"], label=f"Colab R d={d}")
            if d in matched:
                curve = matched[d]
                ax.plot(np.arange(1, len(curve)+1), curve, "--", color=line.get_color(),
                        label=f"Independent matched replay d={d}")
        ax.set(xlabel="Batch step", ylabel="Cumulative M0 (batch BCE difference)")
        ax.legend()
        save(fig, target, "colab_m0")

    if shown:
        fig, axes = plt.subplots(2, 3, figsize=(11, 6.3), sharex=True)
        for d, ax in zip(sizes, axes.flat):
            for method in cfg["methods"]:
                group = sorted((c for c in shown if c.dimension == d and c.method == method), key=lambda c: c.seed)
                if group:
                    ys = np.array([history(root, c, "M2")[0] for c in group])
                    x = np.arange(1, ys.shape[1]+1)
                    ax.plot(x, ys.mean(axis=0), label=f"{method} (n={len(group)})")
                    if len(group) > 1:
                        sd = ys.std(axis=0, ddof=1)
                        ax.fill_between(x, ys.mean(axis=0)-sd, ys.mean(axis=0)+sd, alpha=.12)
            ax.set_title(f"d={d}")
        axes.flat[-1].axis("off")
        axes.flat[4].legend(fontsize=8)
        fig.supxlabel("Batch step")
        fig.supylabel("Cumulative M2 (population BCE difference)")
        fig.suptitle(shown[0].protocol + (f" · seed {seed}" if shown == pilot else " · seeds 0, 1, 2"))
        save(fig, target, "matched_methods")

    d128 = Case(shown[0].protocol if shown else PILOT, 128, seed)
    if d128 in records:
        theta, ref, oracle = history(root, d128, "theta", "reference", "oracle")
        fig, ax = plt.subplots(figsize=(7, 4.8))
        ax.plot(ref[:, 0], ref[:, 1], color="0.2", lw=1.7, label="Generating reference")
        ax.plot(theta[:, 0], theta[:, 1], lw=1., label="Pre-update learner")
        ax.plot(oracle[:, 0], oracle[:, 1], "--", lw=1., label="Approximate population oracle")
        ax.set(title=f"{d128.protocol} · d=128, seed {seed}", xlabel="Coordinate 1", ylabel="Coordinate 2")
        ax.axis("equal")
        ax.legend()
        save(fig, target, "trajectory_d128")

    high = [c for c in shown if c.dimension == 512 and c.seed == seed]
    if high:
        fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharex=True)
        for case in high:
            lr, theta, m0, m2 = history(root, case, "lr", "post_theta", "M0", "M2")
            x = np.arange(1, len(lr)+1)
            for ax, y, label in zip(axes.flat, (lr, np.linalg.norm(theta, axis=1), m0, m2),
                                    ("Learning rate", "Post-update norm", "Cumulative M0", "Cumulative M2")):
                ax.plot(x, y, label=case.method)
                ax.set_ylabel(label)
        axes[0, 0].legend()
        fig.supxlabel("Batch step")
        fig.suptitle(f"{high[0].protocol} · d=512, seed {seed}")
        save(fig, target, "d512_diagnostics")

    if adaptive:
        with frozen_csv.open(encoding="utf-8", newline="") as stream:
            old = [r for r in csv.DictReader(stream) if r["method"] == "convex"]
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
        axes[0].errorbar([str(r["dimension"]) for r in old],
                         [float(r["mean_final_population_B"]) for r in old],
                         yerr=[float(r["sample_SD_B"]) for r in old], fmt="o", capsize=3)
        axes[0].set_title("P: std=√0.1, 3 seeds")
        for d in sizes:
            results = [records[c]["metrics"]["M2"] for c in adaptive if c.dimension == d]
            if results:
                axes[1].errorbar(str(d), np.mean(results), yerr=np.std(results, ddof=1) if len(results)>1 else None,
                                 fmt="o", capsize=3)
        axes[1].set_title(f"{adaptive[0].protocol}\nstd=0.1, distinct reference path", fontsize=9)
        axes[0].set_ylabel("Final cumulative M2")
        axes[0].set_yscale("log")
        fig.supxlabel("Dimension")
        save(fig, target, "project_context_m2")
        fig, ax = plt.subplots(figsize=(8, 4))
        for case in adaptive:
            if case.seed == seed:
                [lr] = history(root, case, "lr")
                ax.plot(np.arange(1, len(lr)+1), lr, label=f"d={case.dimension}")
        ax.set(title=f"{adaptive[0].protocol} · seed {seed}", xlabel="Batch step", ylabel="Learning rate")
        ax.legend(ncol=3)
        save(fig, target, "learning_rates")

    for intervention in cfg["interventions"]:
        rows = [r for r in paired if r["intervention"] == intervention]
        if not rows:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharex=True)
        for metric, ax in zip(METRICS, axes):
            for d in cfg["ablation_dimensions"]:
                subset = [r["delta"] for r in rows if r["metric"] == metric and r["dimension"] == d]
                if subset:
                    ax.scatter([str(d)]*len(subset), subset, alpha=.6)
                    ax.scatter(str(d), np.mean(subset), marker="_", s=150, color="k")
            ax.axhline(0, color="0.3", lw=.7)
            ax.set_title(metric)
        fig.supxlabel("Dimension · individual seeds and mean")
        fig.supylabel("Intervention minus matched control")
        fig.suptitle(f"{ABLATION} · {intervention}")
        save(fig, target, f"ablation_{intervention}")


def report(root, frozen, cfg, proof, frozen_hashes, plan, parity, native, matched, gates, validate):
    target = root / "publication"
    data, figures = target / "data", target / "figures"
    for path in (data, figures):
        path.mkdir(parents=True, exist_ok=True)
    seed = cfg["seeds"][0]
    pilots = [Case(PILOT, d, seed) for d in cfg["dimensions"]]
    pilots += [Case(PILOT, 512, seed, m) for m in cfg["methods"] if m != "convex"]
    rng_cases = [Case(DIAGNOSTIC, d, seed) for d in cfg["parity_dimensions"]]
    all_cases = [*plan, *pilots, *rng_cases]
    records = {c: record for c in all_cases if (record := validate(root, c, proof)) is not None}
    verify_pairs(records)
    errors = {c.key for c in all_cases if (root / "runs" / c.key / "failure.json").is_file() and c not in records}
    logs = {c.key: tracking(root, c) for c in records}

    rows = []
    groups = defaultdict(list)
    pairs = []
    for case, record in records.items():
        row = {**asdict(case), **record["metrics"], **record["diagnostics"],
               "elapsed_seconds": record["elapsed_seconds"], "history_sha256": record["history_sha256"],
               **record["data"]}
        rows.append(row)
        if case.protocol in (PILOT, ALIGNED):
            groups[case.protocol, case.dimension, case.method].append(row)
        if case.protocol == ABLATION:
            control = Case(ALIGNED, case.dimension, case.seed)
            if control in records:
                for metric in METRICS:
                    pairs.append({"intervention": case.intervention, "control": control.key,
                                  "dimension": case.dimension, "seed": case.seed, "metric": metric,
                                  "delta": row[metric] - records[control]["metrics"][metric],
                                  "evaluation": "own distributions (std changes)" if case.intervention == "noise"
                                  else "same distribution"})
    write_csv(data / "per_run.csv", list(rows[0]) if rows else [*asdict(plan[0]), *METRICS], rows)
    write_csv(data / "method_summary.csv",
              ["protocol", "dimension", "method", "distribution", "baseline_policy", "metric", "n", "mean", "sample_sd"],
              [{"protocol": p, "dimension": d, "method": m, "distribution": "Gaussian soft labels, std=.1",
                "baseline_policy": ".1" if m == "constant" else ".1/sqrt(t)" if m == "inverse_sqrt" else "Prop. 4.3",
                "metric": metric, **summary(group, metric)}
               for (p, d, m), group in sorted(groups.items()) for metric in METRICS])
    pair_groups = defaultdict(list)
    for row in pairs:
        pair_groups[row["intervention"], row["dimension"], row["metric"]].append(row)
    write_csv(data / "ablation_pairs.csv", ["intervention", "control", "dimension", "seed", "metric", "delta", "evaluation"], pairs)
    write_csv(data / "ablation_summary.csv", ["intervention", "dimension", "metric", "n", "mean", "sample_sd"],
              [{"intervention": i, "dimension": d, "metric": m, **summary(group, "delta")}
               for (i, d, m), group in sorted(pair_groups.items())])

    fidelity = []
    for d in cfg["dimensions"]:
        check = parity.get(d, {})
        record = records.get(Case(DIAGNOSTIC, d, seed))
        native_value = float(native[d]["M0"][-1]) if d in native else None
        replay = check.get("replay_M0")
        # Replay deviation uses matched diagnostic inputs and does not compare
        # an unrelated evaluation seed or a stored historical screenshot.
        delta = abs(replay - check["diagnostic_M0"]) if replay is not None else None
        fidelity.append({"dimension": d, "archived_notebook_M0": ARCHIVED.get(d),
                         "colab_native_M0": native_value,
                         "independent_MT19937_M0": record["metrics"]["M0"] if record else None,
                         "matched_native_replay_M0": float(matched[d][-1]) if d in matched else None,
                         "matched_native_curve_max_error": float(np.max(np.abs(matched[d].astype(float) - native[d]["M0"])))
                         if d in matched else None,
                         "matched_diagnostic_M0": check.get("diagnostic_M0"), "matched_replay_M0": replay,
                         "matched_absolute_final_deviation": delta,
                         "matched_relative_final_deviation": delta/abs(check["diagnostic_M0"])
                         if delta is not None and abs(check["diagnostic_M0"]) >= 1e-3 else None,
                         "matched_curve_max_error": check.get("reference_M0_max_error"),
                         "fidelity_status": check.get("status", "UNVERIFIED"),
                         "evidence": "Colab diagnostic export" if replay is not None else "immutable notebook cell 23"})
    write_csv(data / "fidelity.csv", list(fidelity[0]), fidelity)

    families = [("P", 60, 60, 0, 0, "VERIFIED", "historical freeze"),
                ("R", 5, len(native), 0, 5-len(native),
                 "VERIFIED" if len(native) == 5 else "BLOCKED_EXTERNAL_EXECUTION", "Colab exports")]
    verified_parity = all(row["status"] == "VERIFIED" for row in parity.values())
    for name, protocol, expected in (("I", ALIGNED, 45), ("A", ABLATION, 36), ("pilot", PILOT, 7), ("rng diagnostic", DIAGNOSTIC, 3)):
        count = sum(c.protocol == protocol for c in records)
        failed = sum(c.protocol == protocol and c.key in errors for c in all_cases)
        blocked = expected-count-failed if name in {"I", "A"} and not verified_parity else 0
        state = ("VERIFIED" if count == expected else "FAILED_VALIDATION" if failed
                 else "BLOCKED_EXTERNAL_EXECUTION" if blocked else "NOT_STARTED")
        families.append((name, expected, count, failed, blocked, state, "SHA-256 sealed local histories"))
    inventory = [{"family": n, "expected": e, "validated": v, "failed": f, "blocked": b,
                  "status": s, "evidence": details,
                  "wandb_remote_verified": 60 if n == "P" else sum(
                      log.get("online", {}).get("state") == "REMOTE_API_VERIFIED" for key,log in logs.items()
                      if key.startswith({"I": ALIGNED, "A": ABLATION, "pilot": PILOT}.get(n, "$"))
                  )}
                 for n, e, v, f, b, s, details in families]
    write_csv(data / "inventory.csv", list(inventory[0]), inventory)
    atomic_json(target / "provenance.json", {**proof, "frozen_sha256": frozen_hashes})
    if native:
        write_csv(data / "native_curves.csv", ["dimension", "step", "M0"],
                  [{"dimension": d, "step": i+1, "M0": float(value)}
                   for d, entry in sorted(native.items()) for i, value in enumerate(entry["M0"])])
    figures_from_runs(root, figures, cfg, records, native, matched, frozen / "data/cp04_method_summary.csv", pairs)
    supplemental_diagnostics(root, data, figures, records, seed)
    atomic_json(target / "manifest.json", {
        "status": "VERIFIED" if all(row["status"] == "VERIFIED" for row in inventory)
        else "COMPLETED_WITH_LIMITATIONS",
        "families": inventory, "gates": gates, "parity": parity, "native": {str(d): row["metadata"] for d,row in native.items()},
        "failed_cases": sorted(errors), "wandb": logs,
        "files": {p.relative_to(target).as_posix(): sha256(p) for p in sorted(target.rglob("*"))
                  if p.is_file() and p.name != "manifest.json"},
    })
