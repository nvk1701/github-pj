"""Chay logistic drift bang shared trainer, danh gia reference gap rieng."""

import argparse
import copy
import csv
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.convex_eval import evaluate_history
from core.drift import generate_spiral_trajectory
from core.schedules import ConstantSchedule, ConvexSchedule, InverseSqrtSchedule
from core.tracking import init_run
from core.trainer import OnlineSGDTrainer
from core.utils import load_config, set_seed


def validate_config(config):
    """Sao chep va kiem schema, khong chap nhan typo hoac mode online."""
    if not isinstance(config, dict):
        raise ValueError("config must be a mapping")
    cfg = copy.deepcopy(config)
    allowed = {"steps", "dimensions", "batch_size", "eval_batch_size", "seeds", "methods",
               "label_noise_std", "label_noise_assumption", "L", "gradient_variance_per_dimension",
               "D_max_per_dimension", "projection_radius_per_dimension", "eta_init", "constant_lr",
               "inverse_sqrt_lr0", "shift_mode", "ema_beta", "eval_seed_stream", "a", "b", "k",
               "ell", "mode", "output_dir"}
    if set(cfg) != allowed:
        raise ValueError(f"config keys missing={sorted(allowed-set(cfg))}, unknown={sorted(set(cfg)-allowed)}")
    for key in ("steps", "batch_size", "eval_batch_size", "ell", "eval_seed_stream"):
        ConvexSchedule._integer(cfg[key], key)
    if cfg["eval_batch_size"] < 2:
        raise ValueError("eval_batch_size must be >=2")
    for key in ("dimensions", "seeds", "methods"):
        values = cfg[key]
        if not isinstance(values, list) or not values:
            raise ValueError(f"{key} must be a nonempty list")
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate {key}")
    for d in cfg["dimensions"]:
        ConvexSchedule._integer(d, "dimension")
        if d < 2:
            raise ValueError("runner requires d>=2 for the selected ball and trajectory plot")
    for seed in cfg["seeds"]:
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
    if not set(cfg["methods"]) <= {"convex", "constant", "inverse_sqrt"}:
        raise ValueError("unknown method")
    for key in ("label_noise_std", "L", "gradient_variance_per_dimension", "D_max_per_dimension",
                "projection_radius_per_dimension", "eta_init", "constant_lr", "inverse_sqrt_lr0",
                "ema_beta", "k"):
        cfg[key] = ConvexSchedule._number(cfg[key], key, positive=key not in ("label_noise_std", "gradient_variance_per_dimension", "ema_beta", "k"))
    for key in ("a", "b"):
        if type(cfg[key]) not in (int, float) or not math.isfinite(cfg[key]):
            raise ValueError(f"{key} must be finite")
    if not 0 <= cfg["ema_beta"] < 1:
        raise ValueError("ema_beta must be in [0,1)")
    if cfg["shift_mode"] not in ("estimated", "lagged_reference"):
        raise ValueError("shift_mode must be estimated or lagged_reference")
    if cfg["mode"] not in ("disabled", "offline"):
        raise ValueError("Sprint 1 permits disabled/offline logging only")
    if not isinstance(cfg["output_dir"], str) or not cfg["output_dir"].strip():
        raise ValueError("output_dir must be a nonempty path")
    if not isinstance(cfg["label_noise_assumption"], str) or not cfg["label_noise_assumption"].strip():
        raise ValueError("label noise assumption must be documented")
    if max(cfg["constant_lr"], cfg["inverse_sqrt_lr0"], cfg["eta_init"]) > 1/cfg["L"]:
        raise ValueError("all initial learning rates must be <=1/L")
    for d in cfg["dimensions"]:
        ConvexSchedule(cfg["L"], d*cfg["D_max_per_dimension"],
                       d*cfg["gradient_variance_per_dimension"], cfg["eta_init"])
        R = d * cfg["projection_radius_per_dimension"]
        if not math.isfinite(R) or 2*R > d*cfg["D_max_per_dimension"]:
            raise ValueError("D_max must bound the diameter 2R")
        reference_bound = math.sqrt(math.ceil(d/2)) * max(abs(cfg["a"]), abs(cfg["b"]))**3
        if R + 1e-12 < reference_bound:
            raise ValueError("projection ball must contain the entire reference path")
    return cfg


def _save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _plot(path, history, metrics):
    """Ve tu metrics that, khong dung hinh hay so lieu cua paper."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    axes[0].plot(metrics["step"], metrics["eval_cum_mc_excess"])
    axes[0].set(title="MC cumulative excess to reference", xlabel="Step")
    axes[1].plot(metrics["step"], history.lr_history)
    axes[1].set(title="Learning rate", xlabel="Step")
    axes[2].plot(history.theta_star_path[:, 0], history.theta_star_path[:, 1], label="reference")
    axes[2].plot(history.theta_path[:, 0], history.theta_path[:, 1], label="learner, pre-update")
    axes[2].set(title="First two coordinates")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_experiments(config):
    """Chay cac methods/seeds, export CSV/NPZ/JSON/PNG va summary mean/std.

    Train seed duoc shared trainer su dung truc tiep. Evaluation co SeedSequence
    rieng. Cung seed/d cho methods dung chung streams, khong cap state qua run.
    Tu choi output da co summary de bao toan bang chung cua experiment truoc.
    """
    cfg = validate_config(config)
    output = Path(cfg["output_dir"])
    if not output.is_absolute():
        output = ROOT / output
    if (output / "summary.json").exists():
        raise FileExistsError("output already contains summary.json; choose a new output_dir")
    output.mkdir(parents=True, exist_ok=True)
    git_result = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True)
    head = git_result.stdout.strip()
    git_status = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                                capture_output=True, text=True, check=True)
    source_uncommitted = bool(git_status.stdout.strip())
    source_hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [ROOT/"core/schedules.py", ROOT/"core/convex_eval.py",
                               ROOT/"core/trainer.py", Path(__file__), ROOT/"configs/convex.yaml"]}
    versions = {p: version(p) for p in ("numpy", "scipy", "matplotlib", "wandb")}
    summaries = []
    for d in cfg["dimensions"]:
        reference, reference_gamma = generate_spiral_trajectory(
            cfg["steps"], d, cfg["a"], cfg["b"], cfg["k"], cfg["ell"])
        for seed in cfg["seeds"]:
            eval_seed = int(np.random.SeedSequence([seed, cfg["eval_seed_stream"], d]).generate_state(1)[0])
            for method in cfg["methods"]:
                run_dir = output / f"d{d}" / method / f"seed{seed}"
                run_dir.mkdir(parents=True, exist_ok=False)
                started = time.perf_counter()
                set_seed(seed)
                D_max = d * cfg["D_max_per_dimension"]
                gradient_variance = d * cfg["gradient_variance_per_dimension"]
                radius = d * cfg["projection_radius_per_dimension"]
                if method == "convex":
                    schedule = ConvexSchedule(cfg["L"], D_max, gradient_variance, cfg["eta_init"])
                elif method == "constant":
                    schedule = ConstantSchedule(cfg["constant_lr"])
                else:
                    schedule = InverseSqrtSchedule(cfg["inverse_sqrt_lr0"])
                trainer = OnlineSGDTrainer(d=d, schedule=schedule, loss_type="logistic",
                    batch_size=cfg["batch_size"], noise_std=cfg["label_noise_std"], max_norm=radius,
                    use_oracle_gamma=cfg["shift_mode"] == "lagged_reference",
                    ema_beta=cfg["ema_beta"], use_wandb=False)
                history = trainer.run(reference, reference_gamma, seed=seed)
                train_seconds = time.perf_counter() - started
                metrics = evaluate_history(history, batch_size=cfg["batch_size"],
                    label_noise_std=cfg["label_noise_std"], eval_batch_size=cfg["eval_batch_size"],
                    eval_seed=eval_seed)
                metrics.update(step=np.arange(1, cfg["steps"]+1), lr=history.lr_history,
                               gamma_used=history.gamma_used, reference_gamma=reference_gamma)
                if isinstance(schedule, ConvexSchedule):
                    thresholds = np.array([schedule.thresholds(g, cfg["batch_size"]) for g in history.gamma_used])
                    metrics.update(tau1=thresholds[:, 0], tau2=thresholds[:, 1])
                if not all(np.isfinite(v).all() for v in metrics.values()):
                    raise ValueError("nonfinite export refused")
                resolved = dict(cfg, dimension=d, seed=seed, eval_seed=eval_seed,
                    method=method, schedule_name=schedule.name, D_max=D_max,
                    gradient_variance=gradient_variance, projection_radius=radius,
                    head=head, source_uncommitted=source_uncommitted, source_sha256=source_hashes,
                    python=sys.version, packages=versions,
                    metric_kind="mc_excess_to_generating_reference_not_true_dynamic_regret",
                    shift_semantics=("causal EMA of learner updates" if cfg["shift_mode"] == "estimated"
                                     else "privileged lagged reference shift gamma_(t-1), not gamma_t"),
                    initial_weights="zero", terminal_reference_gamma="zero extension")
                _save_json(run_dir / "config.json", resolved)
                with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(metrics)
                    writer.writerows(zip(*metrics.values()))
                np.savez_compressed(run_dir / "history.npz", theta=history.theta_path,
                                    reference=reference, **metrics)
                _plot(run_dir / "figure.png", history, metrics)
                # Shared trainer da tat logging de tranh xuat metric clipped cua no.
                run = init_run(dict(experiment="convex", method=method, seed=seed, mode=cfg["mode"],
                    tags=["convex", "sprint1"], group=f"convex-d{d}", resolved=resolved))
                try:
                    if "lab1" not in list(run.tags):
                        raise RuntimeError("lab1 tag missing")
                    for i in range(cfg["steps"]):
                        run.log({key: float(v[i]) for key, v in metrics.items() if key != "step"},
                                step=i+1, commit=True)
                    run.summary["metric_kind"] = resolved["metric_kind"]
                    run.summary["completed"] = True
                    run.summary["final_mc_cum_excess"] = float(metrics["eval_cum_mc_excess"][-1])
                    tracking = dict(mode=cfg["mode"], run_id=run.id, tags=list(run.tags),
                                    uploaded=False, run_dir=run.dir)
                except BaseException:
                    try:
                        run.finish(exit_code=1)
                    except BaseException:
                        # Loi cleanup khong duoc che loi logging ban dau.
                        pass
                    raise
                else:
                    run.finish()
                summary = dict(dimension=d, seed=seed, method=method,
                    final_mc_cum_excess=float(metrics["eval_cum_mc_excess"][-1]),
                    final_mc_cum_excess_se=float(metrics["eval_cum_mc_excess_se"][-1]),
                    final_train_cum_excess=float(metrics["train_cum_empirical_excess"][-1]),
                    final_lr=float(history.lr_history[-1]), train_seconds=train_seconds,
                    total_seconds=time.perf_counter()-started, output_dir=str(run_dir), tracking=tracking)
                _save_json(run_dir / "summary.json", summary)
                summaries.append(summary)
    aggregate = []
    for d in cfg["dimensions"]:
        for method in cfg["methods"]:
            values = [s["final_mc_cum_excess"] for s in summaries
                      if s["dimension"] == d and s["method"] == method]
            aggregate.append(dict(dimension=d, method=method, n_seeds=len(values),
                mean_final_mc_cum_excess=float(np.mean(values)),
                sample_std_final_mc_cum_excess=float(np.std(values, ddof=1)) if len(values)>1 else None))
    _save_json(output / "summary.json", {"runs": summaries, "aggregate": aggregate})
    return summaries


def main(argv=None):
    """Doc YAML va CLI overrides. Online logging khong thuoc Sprint 1."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/convex.yaml")
    for flag in ("steps", "batch-size", "eval-batch-size", "ell"):
        parser.add_argument("--"+flag, type=int)
    parser.add_argument("--dimensions", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--methods", nargs="+", choices=("convex", "constant", "inverse_sqrt"))
    parser.add_argument("--mode", choices=("disabled", "offline"))
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if not isinstance(config, dict):
        raise ValueError("YAML root must be a mapping")
    for key, value in vars(args).items():
        if key != "config" and value is not None:
            config[key] = value
    results = run_experiments(config)
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
