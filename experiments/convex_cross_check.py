r"""
================================================================================
FILE: experiments/convex.py
MỤC ĐÍCH:
    Script thực thi và tái hiện Thực nghiệm 2 — Hồi quy Logistic với Hàm mất mát
    Lồi Tổng quát (Task T15 - TV2: Khôi & Kiểm tra chéo Task T19 - TV1: Khoa):
    - Tái hiện Figure 5 (bottom) trong Mục 6.1 (trang 9) của bài báo Fahrbach et al.
      (ICML 2023) sử dụng `ConvexSchedule` (Proposition 4.3, Công thức 18-20) với
      L = 1/4, \sigma^2 = d/4, D_max = d và bán kính chiếu R = d/2.
    - Hỗ trợ chạy kiểm chứng nhanh hoặc tái tạo đầy đủ các biểu đồ và bảng số liệu
      từ kết quả CP04 đã xác minh trong `docs/results/cp04-cp05/`.
================================================================================
"""

import argparse
import csv
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.convex_population import PopulationEvaluator
from core.drift import generate_spiral_trajectory
from core.schedules import ConstantSchedule, ConvexSchedule, InverseSqrtSchedule
from core.trainer import OnlineSGDTrainer
from core.utils import load_config, set_seed


def run_convex_verification_and_export(mode: str = "disabled", quick_verify: bool = True) -> None:
    """Chạy kiểm chứng thực nghiệm Convex (Figure 5-bottom) và xuất hình/CSV chuẩn vào `figures/` & `results/logistic/`."""
    cfg = load_config(PROJECT_ROOT / "configs" / "convex.yaml")
    figures_dir = PROJECT_ROOT / "figures"
    results_dir = PROJECT_ROOT / "results" / "logistic"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    cp04_dir = PROJECT_ROOT / "docs" / "results" / "cp04-cp05"
    cp04_summary_src = cp04_dir / "data" / "cp04_method_summary.csv"
    cp04_per_seed_src = cp04_dir / "data" / "cp04_per_seed.csv"

    if cp04_summary_src.exists():
        shutil.copyfile(cp04_summary_src, results_dir / "fig5_bottom_summary.csv")
    if cp04_per_seed_src.exists():
        shutil.copyfile(cp04_per_seed_src, results_dir / "fig5_bottom_per_seed.csv")

    # Sao chép các biểu đồ CP04 chuẩn sang `figures/fig5_bottom_*.png` theo đúng quy ước Sheet 2
    fig_map = {
        "cp04_figure5_left.png": "fig5_bottom_logistic_regret.png",
        "cp04_trajectory_d128.png": "fig5_bottom_trajectory_d128.png",
        "cp04_all_dimension_baselines.png": "fig5_bottom_baselines_comparison.png",
    }
    for src_name, dst_name in fig_map.items():
        src_path = cp04_dir / "figures" / src_name
        if src_path.exists():
            shutil.copyfile(src_path, figures_dir / dst_name)

    # Chạy kiểm chứng trực tiếp (live replay verification) và kiểm tra khớp tuyệt đối với bảng đóng băng
    dims_to_run = [2, 8] if quick_verify else [int(d) for d in cfg["dimensions"]]
    seeds_to_run = [0] if quick_verify else [int(s) for s in cfg["seeds"]]
    T = int(cfg["steps"])
    B = int(cfg["batch_size"])
    noise_std = float(cfg["label_noise_std"])

    with open(cp04_per_seed_src, "r", encoding="utf-8") as f:
        frozen_rows = {r["key"]: r for r in csv.DictReader(f)}

    for d in dims_to_run:
        theta_star, gamma_seq = generate_spiral_trajectory(
            T=T, d=d, a=float(cfg["a"]), b=float(cfg["b"]), k=float(cfg["k"]), ell=int(cfg["ell"])
        )
        pop_eval = PopulationEvaluator(theta_star, label_noise_std=noise_std, tolerance=1e-11)
        for seed in seeds_to_run:
            set_seed(seed)
            sched = ConvexSchedule(
                L=float(cfg["L"]),
                D_max=float(cfg["D_max_per_dimension"]) * d,
                gradient_variance=float(cfg["gradient_variance_per_dimension"]) * d,
                eta_init=float(cfg["eta_init"]),
            )
            trainer = OnlineSGDTrainer(
                d=d,
                schedule=sched,
                loss_type="logistic",
                batch_size=B,
                noise_std=noise_std,
                max_norm=float(cfg["projection_radius_per_dimension"]) * d,
                use_oracle_gamma=False,
                ema_beta=float(cfg["ema_beta"]),
                use_wandb=False,
            )
            hist = trainer.run(theta_star_path=theta_star, gamma_seq=gamma_seq, seed=seed)
            pop_metrics = pop_eval.evaluate(hist.theta_path)
            final_B = float(pop_metrics["population_cum_B"][-1])
            key = f"convex-cp03-v1-d{d}-convex-seed{seed}"
            expected_B = float(frozen_rows[key]["final_population_B"])
            diff = abs(final_B - expected_B)
            print(
                f"[T15/T19 Verify] d={d:3d}, seed={seed} | Live B = {final_B:.8f} | "
                f"Frozen B = {expected_B:.8f} | abs_diff = {diff:.2e}"
            )
            if diff > 1e-10:
                raise RuntimeError(f"Sai lệch tái lập tại {key}: {diff}")

    print(f"[OK] Convex experiment verification & artifacts ready in {results_dir} and {figures_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run or verify Convex Logistic Regression experiments (T15/T19).")
    parser.add_argument("--mode", choices=("online", "offline", "disabled"), default="disabled")
    parser.add_argument("--full", action="store_true", help="Run all dimensions and seeds instead of quick verify.")
    args = parser.parse_args()
    run_convex_verification_and_export(mode=args.mode, quick_verify=not args.full)
