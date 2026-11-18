r"""
================================================================================
FILE: experiments/linear_regression.py
MỤC ĐÍCH:
    Thực thi trọn vẹn Thực nghiệm 1 — Hồi quy Tuyến tính dưới Dịch chuyển Phân phối
    (Task T14 - TV1: Khoa), tái hiện đầy đủ các biểu đồ và bảng số liệu của Mục 3
    và Mục 6.1 trong bài báo Fahrbach et al. (ICML 2023):

    1. [Figure 4 - Mục 6.1, trang 9]:
       - Quỹ đạo xoắn ốc 2D (d = 2, T = 2000, B_t = 256, k = 4, a = 1, b = -1)
         và sự dao động của tốc độ học tối ưu \eta_t (Algorithm 1, \eta_max = 0.5)
         khi rời rạc hóa quỹ đạo theo \ell \in {1, 4, 16}.
       - Xuất hình: `figures/fig4_linear_trajectory_and_lr.png`
       - Xuất CSV:  `results/linear/fig4_summary.csv`

    2. [Figure 5 (top) - Mục 6.1, trang 9]:
       - (Top-Left): Cumulative Dynamic Regret của Algorithm 1 (\eta_max = 1 / sqrt(d),
         \ell = 8, B_t = 256, T = 2000) theo số chiều d \in {2, 8, 32, 128, 512}
         trên 3 random seeds {0, 1, 2} (mean \pm std).
       - (Top-Right): Quỹ đạo 2 tọa độ đầu (\theta_t(1), \theta_t(2)) của SGD bám đuổi
         quỹ đạo xoắn ốc \hat{\theta}^*_t ở không gian cao chiều d = 128.
       - Đối chiếu thêm với bản ước lượng trực tuyến EMA (\hat{\gamma}_t) và 3 lịch trình
         baseline (`Constant`, `1/t`, `1/sqrt(t)`).
       - Xuất hình: `figures/fig5_top_linear_regret_and_path.png`,
                    `figures/fig5_top_baselines_comparison.png`
       - Xuất CSV:  `results/linear/fig5_top_per_seed.csv`,
                    `results/linear/fig5_top_summary.csv`,
                    `results/linear/linear_baselines_comparison.csv`

    3. [Hình bổ trợ Lý thuyết Mục 3 cho Slide 3, 14, 15]:
       - `figures/fig1_constant_lr_trajectories.png` (Tái hiện Figure 1 trang 2)
       - `figures/fig2_alg1_schedules.png` (Tái hiện Figure 2 trang 6: Bursty & Smooth shifts)
       - `figures/fig3_no_shift_asymptotics.png` (Tái hiện Figure 3 trang 6: Lemma 3.6)

CÁCH CHẠY:
    - Chạy mặc định với W&B offline:
      `& './.venv-t2-clean/Scripts/python.exe' experiments/linear_regression.py --mode offline`
    - Chạy đẩy kết quả lên W&B Public (`gr_ml_29_37_37/ml-distribution-shift`):
      `& './.venv-t2-clean/Scripts/python.exe' experiments/linear_regression.py --mode online`
    - Chạy nhanh không bật W&B:
      `& './.venv-t2-clean/Scripts/python.exe' experiments/linear_regression.py --mode disabled`
================================================================================
"""

import argparse
import csv
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Đảm bảo thư mục cấu hình tạm của Matplotlib luôn ghi được trên Windows
_MPL_DIR = PROJECT_ROOT / ".mplconfig"
_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.drift import (
    generate_bursty_shifts,
    generate_smooth_shifts,
    generate_spiral_trajectory,
)
from core.schedules import (
    ConstantSchedule,
    InverseSqrtSchedule,
    InverseTimeSchedule,
    LinearOptimalSchedule,
)
from core.tracking import init_run
from core.trainer import OnlineSGDTrainer, TrainingHistory
from core.utils import load_config, set_seed


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    """Ghi danh sách dictionary ra file CSV với định dạng chuẩn UTF-8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_with_tracking(
    trainer: OnlineSGDTrainer,
    theta_star: np.ndarray,
    gamma_seq: np.ndarray,
    seed: int,
    mode: str,
    run_meta: Dict[str, Any],
) -> tuple[TrainingHistory, str, str]:
    """Chạy `trainer.run(...)` bọc trong `init_run` của `core/tracking.py` nếu mode != 'disabled'."""
    if mode == "disabled":
        trainer.use_wandb = False
        hist = trainer.run(theta_star_path=theta_star, gamma_seq=gamma_seq, seed=seed)
        return hist, "disabled", "disabled"

    cfg = dict(run_meta)
    cfg["mode"] = mode
    cfg["seed"] = int(seed)
    run = init_run(cfg)
    run_id = str(getattr(run, "id", "offline"))
    run_url = str(
        getattr(run, "url", None)
        or f"https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift/runs/{run_id}"
    )
    try:
        trainer.use_wandb = True
        hist = trainer.run(theta_star_path=theta_star, gamma_seq=gamma_seq, seed=seed)
        run.summary["eval/final_cum_regret"] = float(hist.cum_regret[-1])
        run.summary["eval/mean_inst_regret"] = float(np.mean(hist.inst_regret))
        run.summary["train/mean_lr"] = float(np.mean(hist.lr_history))
        run.summary["train/max_lr"] = float(np.max(hist.lr_history))
        run.summary["train/min_lr"] = float(np.min(hist.lr_history))
        run.summary["completed"] = True
    except BaseException:
        try:
            run.finish(exit_code=1)
        except Exception:
            pass
        raise
    else:
        run.finish()
    return hist, run_id, run_url


def generate_theory_figures(figures_dir: Path) -> None:
    """Tái hiện Figure 1, Figure 2 và Figure 3 (Mục 1 & Mục 3) phục vụ Slide lý thuyết của TV1."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tái hiện Figure 1 (trang 2): Quỹ đạo SGD với 4 tốc độ học hằng số
    T_fig1 = 1700
    d_fig1 = 2
    ell_fig1 = 100
    lrs_fig1 = [0.003, 0.01, 0.03, 0.1]
    theta_star_f1, gamma_f1 = generate_spiral_trajectory(
        T=T_fig1, d=d_fig1, a=1.0, b=-1.55, k=1.2, ell=ell_fig1
    )

    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.6), dpi=180)
    for ax, lr in zip(axes, lrs_fig1):
        set_seed(0)
        trainer = OnlineSGDTrainer(
            d=d_fig1,
            schedule=ConstantSchedule(lr=lr),
            loss_type="linear",
            batch_size=1,
            noise_std=math.sqrt(0.1),
            use_wandb=False,
        )
        hist = trainer.run(theta_star_path=theta_star_f1, gamma_seq=gamma_f1, seed=0)
        idx_markers = np.arange(0, T_fig1, ell_fig1)
        ax.plot(
            theta_star_f1[idx_markers, 0],
            theta_star_f1[idx_markers, 1],
            "o-",
            color="#1f77b4",
            lw=1.3,
            ms=4.0,
            label=r"Oracle $\theta^*_t$",
        )
        ax.plot(
            hist.theta_path[:, 0],
            hist.theta_path[:, 1],
            "-",
            color="#ff7f0e",
            lw=0.9,
            alpha=0.85,
        )
        ax.plot(
            hist.theta_path[idx_markers, 0],
            hist.theta_path[idx_markers, 1],
            "s",
            color="#ff7f0e",
            ms=4.0,
            label=r"SGD $\theta_t$",
        )
        ax.set_xlim(-4.0, 4.0)
        ax.set_ylim(-4.0, 4.0)
        ax.set_title(f"$\\eta_t = {lr}$ (Regret = {hist.cum_regret[-1]:.0f})", fontsize=10)
        ax.grid(True, alpha=0.25)
        if lr == lrs_fig1[0]:
            ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        "Figure 1 Reproduction: Online Linear Regression Trajectories under Constant Learning Rates ($B_t = 1$)",
        fontsize=11.5,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "fig1_constant_lr_trajectories.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Tái hiện Figure 2 (trang 6): Lịch trình Algorithm 1 dưới Bursty & Smooth Shifts
    T_fig2 = 200
    d_fig2 = 100
    B_fig2 = 100
    eps_fig2 = 0.1
    sigma_fig2 = 2.0
    t_axis = np.arange(1, T_fig2 + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=180)
    jump_styles = [
        (0.0, "-", "#1f77b4", "$s = 0$"),
        (0.1, "--", "#d62728", "$s = 0.1$"),
        (0.5, "-.", "#2ca02c", "$s = 0.5$"),
        (2.0, ":", "#9467bd", "$s = 2$"),
    ]
    for s_val, ls, col, lbl in jump_styles:
        gamma_burst = generate_bursty_shifts(T=T_fig2, episode_length=40, jump_size=s_val)
        sched = LinearOptimalSchedule(max_lr=eps_fig2, kappa=0.01, v_init=1.0)
        lrs = [
            sched.step(
                t=int(t),
                gamma_prev=float(gamma_burst[t - 1]),
                batch_size=B_fig2,
                d=d_fig2,
                noise_std=sigma_fig2,
            )
            for t in t_axis
        ]
        axes[0].plot(t_axis, lrs, linestyle=ls, color=col, lw=1.8, label=lbl)

    axes[0].set_title("Bursty Shifts (Episode = 40 steps)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Iteration $t$")
    axes[0].set_ylabel(r"Optimal Learning Rate $\eta^*_t$")
    axes[0].set_ylim(0.0, 0.15)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=9)

    alpha_styles = [
        (float("inf"), "-", "#1f77b4", r"$\alpha = \infty$"),
        (2.0, "--", "#d62728", r"$\alpha = 2$"),
        (1.5, "-.", "#2ca02c", r"$\alpha = 1.5$"),
        (1.0, ":", "#9467bd", r"$\alpha = 1$"),
        (0.5, "--", "#e377c2", r"$\alpha = 0.5$"),
    ]
    for alpha_val, ls, col, lbl in alpha_styles:
        gamma_smooth = generate_smooth_shifts(T=T_fig2, alpha=alpha_val)
        sched = LinearOptimalSchedule(max_lr=eps_fig2, kappa=0.01, v_init=1.0)
        lrs = [
            sched.step(
                t=int(t),
                gamma_prev=float(gamma_smooth[t - 1]),
                batch_size=B_fig2,
                d=d_fig2,
                noise_std=sigma_fig2,
            )
            for t in t_axis
        ]
        axes[1].plot(t_axis, lrs, linestyle=ls, color=col, lw=1.8, label=lbl)

    axes[1].set_title(r"Smooth Shifts ($\gamma_t = 1 / t^\alpha$)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Iteration $t$")
    axes[1].set_ylabel(r"Optimal Learning Rate $\eta^*_t$")
    axes[1].set_ylim(0.0, 0.105)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=9)

    fig.suptitle(
        r"Figure 2 Reproduction: Algorithm 1 Schedules $\eta^*_t$ ($B_t=100, d=100, \varepsilon=0.1, \sigma=2$)",
        fontsize=11.5,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "fig2_alg1_schedules.png", bbox_inches="tight")
    plt.close(fig)

    # 3. Tái hiện Figure 3 (trang 6): Lemma 3.6 khi không có shift (\gamma_t = 0)
    eps_f3 = 0.1
    a_f3 = 0.1
    b_f3 = 0.3
    v0_f3 = 1.0
    tau_star = (1.0 / (2.0 - a_f3)) * math.log((1.0 - a_f3) * (v0_f3 * (2.0 - a_f3) / b_f3 - 1.0))
    c_const = a_f3 * math.log((1.0 - a_f3) / b_f3) + 1.0 - a_f3 - tau_star

    T_f3 = 50
    sched_f3 = LinearOptimalSchedule(max_lr=eps_f3, kappa=0.005, v_init=v0_f3)
    d_f3 = 99
    B_f3 = 100
    sigma_f3 = math.sqrt(300.0 / float(d_f3))

    t_steps_f3 = np.arange(1, T_f3 + 1)
    eta_f3 = []
    v_f3 = [v0_f3]
    for t in t_steps_f3:
        lr_val = sched_f3.step(t=int(t), gamma_prev=0.0, batch_size=B_f3, d=d_f3, noise_std=sigma_f3)
        eta_f3.append(lr_val)
        v_f3.append(sched_f3.v)

    tau_grid = np.linspace(0.0, 5.0, T_f3 + 1)
    asymp_v = b_f3 / (tau_grid + c_const)
    asymp_eta = eps_f3 / (a_f3 + c_const + eps_f3 * t_steps_f3)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), dpi=180)
    axes[0].plot(tau_grid, v_f3, "-", color="#1f77b4", lw=2.0, label=r"$\tilde{v}_\tau$")
    axes[0].plot(tau_grid, asymp_v, "--", color="#9467bd", lw=1.8, label=r"$\frac{\mathsf{b}}{\tau + C}$")
    axes[0].axvline(tau_star, color="#d62728", linestyle=":", lw=1.2, label=f"$\\tau_* \\approx {tau_star:.2f}$")
    axes[0].set_xlim(0.0, 5.0)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_xlabel(r"Continuous time $\tau$")
    axes[0].set_ylabel(r"Upper envelope $\tilde{v}_\tau$")
    axes[0].set_title(r"Evolution of $\tilde{v}_\tau$ under No Shift (Lemma 3.6)", fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)

    axes[1].plot(t_steps_f3, eta_f3, "-", color="#1f77b4", lw=2.0, label=r"$\eta^*_t$ (Alg. 1)")
    axes[1].plot(
        t_steps_f3,
        asymp_eta,
        "--",
        color="#9467bd",
        lw=1.8,
        label=r"$\frac{\varepsilon}{\mathsf{a} + C + \varepsilon t} \asymp \mathcal{O}(1/t)$",
    )
    axes[1].axvline(math.ceil(tau_star / eps_f3), color="#d62728", linestyle=":", lw=1.2, label=r"$t_* = \lceil \tau_*/\varepsilon \rceil$")
    axes[1].set_xlim(0, 50)
    axes[1].set_ylim(0.0, 0.12)
    axes[1].set_xlabel(r"Discrete step $t$")
    axes[1].set_ylabel(r"Optimal Learning Rate $\eta^*_t$")
    axes[1].set_title(r"Asymptotic $\mathcal{O}(1/t)$ Decay when $\gamma_t = 0$", fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)

    fig.suptitle(
        r"Figure 3 Reproduction: Case Study with No Distribution Shift ($\varepsilon=0.1, \mathsf{a}=0.1, \mathsf{b}=0.3, \tilde{v}_0=1$)",
        fontsize=11.5,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "fig3_no_shift_asymptotics.png", bbox_inches="tight")
    plt.close(fig)


def run_figure4_experiment(
    cfg: Dict[str, Any],
    seeds: Sequence[int],
    mode: str,
    figures_dir: Path,
    results_dir: Path,
) -> List[Dict[str, Any]]:
    """Tái hiện Figure 4 (Mục 6.1, trang 9): Quỹ đạo 2D và dao động LR theo ell in {1, 4, 16}."""
    T = int(cfg["horizon_T"])
    B = int(cfg["batch_size"])
    noise_std = math.sqrt(float(cfg["noise_variance"]))
    kappa = float(cfg["kappa"])

    f4_cfg = cfg["figure4"]
    d = int(f4_cfg["d"])
    max_lr = float(f4_cfg["max_lr"])
    ell_list = [int(x) for x in f4_cfg["ell_list"]]

    sp = cfg["spiral"]
    a, b, k = float(sp["a"]), float(sp["b"]), float(sp["k"])

    summary_rows: List[Dict[str, Any]] = []
    seed0_histories: Dict[int, TrainingHistory] = {}

    for ell in ell_list:
        theta_star, gamma_seq = generate_spiral_trajectory(T=T, d=d, a=a, b=b, k=k, ell=ell)
        regrets_per_seed = []
        run_urls = []

        for seed in seeds:
            set_seed(seed)
            sched = LinearOptimalSchedule(
                max_lr=max_lr,
                kappa=kappa,
                v_init=float(np.sum(theta_star[0] ** 2)),
                name=f"alg1-ell{ell}",
            )
            trainer = OnlineSGDTrainer(
                d=d,
                schedule=sched,
                loss_type="linear",
                batch_size=B,
                noise_std=noise_std,
                use_oracle_gamma=True,
                wandb_log_interval=20,
            )
            hist, run_id, run_url = _run_with_tracking(
                trainer=trainer,
                theta_star=theta_star,
                gamma_seq=gamma_seq,
                seed=seed,
                mode=mode if seed == seeds[0] else "disabled",
                run_meta={
                    "experiment": "linear",
                    "method": sched.name,
                    "group": "linear",
                    "job_type": "reproduction",
                    "tags": ["lab1", "linear", "figure4", f"ell-{ell}", "member:Khoa"],
                    "T": T,
                    "d": d,
                    "batch_size": B,
                    "ell": ell,
                    "max_lr": max_lr,
                },
            )
            regrets_per_seed.append(float(hist.cum_regret[-1]))
            run_urls.append(run_url)
            if seed == seeds[0]:
                seed0_histories[ell] = hist

        reg_arr = np.asarray(regrets_per_seed, dtype=np.float64)
        h0 = seed0_histories[ell]
        osc_peaks = int(np.sum((h0.lr_history[1:] >= 0.95 * max_lr) & (h0.lr_history[:-1] < 0.90 * max_lr)))
        row = {
            "ell": ell,
            "d": d,
            "T": T,
            "batch_size": B,
            "max_lr": max_lr,
            "num_shift_events": int(np.sum(gamma_seq > 0.0)),
            "lr_min": float(np.min(h0.lr_history)),
            "lr_max": float(np.max(h0.lr_history)),
            "lr_mean": float(np.mean(h0.lr_history)),
            "lr_std": float(np.std(h0.lr_history)),
            "num_lr_restarts_to_max": osc_peaks,
            "final_regret_mean": float(np.mean(reg_arr)),
            "final_regret_std": float(np.std(reg_arr, ddof=1)) if len(reg_arr) > 1 else 0.0,
            "wandb_runs": ";".join(run_urls),
        }
        summary_rows.append(row)
        print(
            f"[T14 - Fig 4] ell={ell:2d} | shifts={row['num_shift_events']:4d} | "
            f"LR range=[{row['lr_min']:.4f}, {row['lr_max']:.4f}] | "
            f"Final Regret={row['final_regret_mean']:.3f} ± {row['final_regret_std']:.3f} | "
            f"W&B={run_urls[0]}",
            flush=True,
        )

    _write_csv(
        results_dir / "fig4_summary.csv",
        fieldnames=list(summary_rows[0].keys()),
        rows=summary_rows,
    )

    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.2), dpi=200)
    steps = np.arange(1, T + 1)

    for col_idx, ell in enumerate(ell_list):
        h = seed0_histories[ell]
        ax_top = axes[0, col_idx]
        ax_bot = axes[1, col_idx]

        ax_top.plot(
            h.theta_star_path[:, 0],
            h.theta_star_path[:, 1],
            "o-",
            color="#1f77b4",
            lw=1.1,
            ms=2.4,
            label=r"$\hat{\theta}^*_t$ (Oracle)",
        )
        ax_top.plot(
            h.theta_path[:, 0],
            h.theta_path[:, 1],
            "o-",
            color="#ff7f0e",
            lw=0.9,
            ms=1.8,
            alpha=0.85,
            label=r"$\theta_t$ (SGD)",
        )
        ax_top.set_title(f"2D Trajectory ($\\ell = {ell}$)", fontsize=11, fontweight="bold")
        ax_top.set_xlabel(r"$\theta(1)$")
        ax_top.set_ylabel(r"$\theta(2)$")
        ax_top.set_xlim(-1.05, 1.05)
        ax_top.set_ylim(-1.05, 1.05)
        ax_top.grid(True, alpha=0.3)
        ax_top.legend(loc="upper right", fontsize=8.5)

        ax_bot.plot(steps, h.lr_history, "-", color="#1f77b4", lw=0.9, label=r"$\eta_t$ (Alg. 1)")
        ax_bot.set_title(
            f"Learning Rate $\\eta_t$ ($\\ell = {ell}$, $\\eta_{{\\max}} = {max_lr}$)",
            fontsize=11,
            fontweight="bold",
        )
        ax_bot.set_xlabel("Iteration $t$")
        ax_bot.set_ylabel(r"Learning rate $\eta_t$")
        ax_bot.set_xlim(0, T)
        ax_bot.set_ylim(0.0, max_lr * 1.06)
        ax_bot.grid(True, alpha=0.3)
        ax_bot.legend(loc="lower left", fontsize=8.5)

    fig.suptitle(
        "Figure 4 Reproduction: SGD Trajectories (top) & Oscillating Learning Rates of Algorithm 1 (bottom)",
        fontsize=12.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "fig4_linear_trajectory_and_lr.png", bbox_inches="tight")
    plt.close(fig)
    return summary_rows


def run_figure5_top_experiment(
    cfg: Dict[str, Any],
    seeds: Sequence[int],
    mode: str,
    figures_dir: Path,
    results_dir: Path,
    skip_cached_baselines: bool = False,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Tái hiện Figure 5 (top) (Mục 6.1, trang 9) và đối chiếu với các lịch trình Baseline."""
    T = int(cfg["horizon_T"])
    B = int(cfg["batch_size"])
    noise_std = math.sqrt(float(cfg["noise_variance"]))
    kappa = float(cfg["kappa"])

    f5_cfg = cfg["figure5_top"]
    ell = int(f5_cfg["ell"])
    dimensions = [int(d) for d in f5_cfg["dimensions"]]
    traj_d = int(f5_cfg["trajectory_d"])
    ema_beta = float(f5_cfg["ema_beta"])

    sp = cfg["spiral"]
    a, b, k = float(sp["a"]), float(sp["b"]), float(sp["k"])

    # Nếu đã có sẵn kết quả các baseline phụ và đang chạy online, có thể tái sử dụng để tiết kiệm thời gian
    cached_baselines: Dict[tuple[int, str], Dict[str, Any]] = {}
    baseline_csv_path = results_dir / "linear_baselines_comparison.csv"
    if skip_cached_baselines and baseline_csv_path.exists():
        with open(baseline_csv_path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                cached_baselines[(int(r["d"]), r["method"])] = {
                    "d": int(r["d"]),
                    "method": r["method"],
                    "max_lr_cap": float(r["max_lr_cap"]),
                    "final_regret_mean": float(r["final_regret_mean"]),
                    "final_regret_std": float(r["final_regret_std"]),
                    "mean_lr": float(r["mean_lr"]),
                    "wandb_runs": r["wandb_runs"],
                }

    per_seed_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    baseline_rows: List[Dict[str, Any]] = []

    alg1_curves_by_d: Dict[int, np.ndarray] = {}
    traj_d128_hist: TrainingHistory | None = None

    for d in dimensions:
        theta_star, gamma_seq = generate_spiral_trajectory(T=T, d=d, a=a, b=b, k=k, ell=ell)
        max_lr_d = 1.0 / math.sqrt(float(d))
        v_init_d = float(np.sum(theta_star[0] ** 2))

        method_specs = [
            (
                "alg1-oracle",
                lambda: LinearOptimalSchedule(max_lr=max_lr_d, kappa=kappa, v_init=v_init_d, name=f"alg1-d{d}"),
                True,
            ),
            (
                "alg1-ema",
                lambda: LinearOptimalSchedule(max_lr=max_lr_d, kappa=kappa, v_init=v_init_d, name=f"alg1-ema-d{d}"),
                False,
            ),
            (
                "constant",
                lambda: ConstantSchedule(lr=round(0.25 * max_lr_d, 6)),
                True,
            ),
            (
                "inv-sqrt",
                lambda: InverseSqrtSchedule(lr0=round(max_lr_d, 6)),
                True,
            ),
            (
                "inv-time",
                lambda: InverseTimeSchedule(lr0=round(max_lr_d, 6), decay=0.05),
                True,
            ),
        ]

        for method_key, sched_factory, use_oracle in method_specs:
            if method_key != "alg1-oracle" and (d, method_key) in cached_baselines:
                baseline_rows.append(cached_baselines[(d, method_key)])
                continue

            cum_curves = []
            final_regrets = []
            mean_lrs = []
            run_urls = []

            for seed in seeds:
                set_seed(seed)
                sched = sched_factory()
                trainer = OnlineSGDTrainer(
                    d=d,
                    schedule=sched,
                    loss_type="linear",
                    batch_size=B,
                    noise_std=noise_std,
                    use_oracle_gamma=use_oracle,
                    ema_beta=ema_beta,
                    wandb_log_interval=20,
                )
                hist, run_id, run_url = _run_with_tracking(
                    trainer=trainer,
                    theta_star=theta_star,
                    gamma_seq=gamma_seq,
                    seed=seed,
                    mode=mode if method_key == "alg1-oracle" else "disabled",
                    run_meta={
                        "experiment": "linear",
                        "method": sched.name,
                        "group": "linear",
                        "job_type": "reproduction",
                        "tags": ["lab1", "linear", "figure5-top", f"d-{d}", method_key, "member:Khoa"],
                        "T": T,
                        "d": d,
                        "batch_size": B,
                        "ell": ell,
                        "max_lr": max_lr_d,
                        "use_oracle_gamma": use_oracle,
                    },
                )
                cum_curves.append(hist.cum_regret)
                final_regrets.append(float(hist.cum_regret[-1]))
                mean_lrs.append(float(np.mean(hist.lr_history)))
                run_urls.append(run_url)

                if method_key == "alg1-oracle":
                    per_seed_rows.append(
                        {
                            "d": d,
                            "seed": seed,
                            "method": sched.name,
                            "max_lr": round(max_lr_d, 6),
                            "final_cum_regret": float(hist.cum_regret[-1]),
                            "regret_at_t500": float(hist.cum_regret[499]),
                            "regret_at_t1000": float(hist.cum_regret[999]),
                            "regret_at_t1500": float(hist.cum_regret[1499]),
                            "mean_lr": float(np.mean(hist.lr_history)),
                            "min_lr": float(np.min(hist.lr_history)),
                            "max_lr_observed": float(np.max(hist.lr_history)),
                            "wandb_run_id": run_id,
                            "wandb_url": run_url,
                        }
                    )
                    if d == traj_d and seed == seeds[0]:
                        traj_d128_hist = hist

            reg_arr = np.asarray(final_regrets, dtype=np.float64)
            lr_arr = np.asarray(mean_lrs, dtype=np.float64)
            baseline_rows.append(
                {
                    "d": d,
                    "method": method_key,
                    "max_lr_cap": round(max_lr_d, 6),
                    "final_regret_mean": float(np.mean(reg_arr)),
                    "final_regret_std": float(np.std(reg_arr, ddof=1)) if len(reg_arr) > 1 else 0.0,
                    "mean_lr": float(np.mean(lr_arr)),
                    "wandb_runs": ";".join(run_urls),
                }
            )

            if method_key == "alg1-oracle":
                alg1_curves_by_d[d] = np.stack(cum_curves, axis=0)
                c_mat = alg1_curves_by_d[d]
                s_row = {
                    "d": d,
                    "max_lr": round(max_lr_d, 6),
                    "regret_t500_mean": float(np.mean(c_mat[:, 499])),
                    "regret_t1000_mean": float(np.mean(c_mat[:, 999])),
                    "regret_t1500_mean": float(np.mean(c_mat[:, 1499])),
                    "final_regret_mean": float(np.mean(reg_arr)),
                    "final_regret_std": float(np.std(reg_arr, ddof=1)) if len(reg_arr) > 1 else 0.0,
                    "mean_lr": float(np.mean(lr_arr)),
                    "wandb_runs": ";".join(run_urls),
                }
                summary_rows.append(s_row)
                print(
                    f"[T14 - Fig 5-top] d={d:3d} | eta_max={max_lr_d:.4f} | "
                    f"Alg.1 Final Regret = {s_row['final_regret_mean']:9.2f} ± {s_row['final_regret_std']:6.2f} | "
                    f"W&B={run_urls[0]}",
                    flush=True,
                )

    _write_csv(
        results_dir / "fig5_top_per_seed.csv",
        fieldnames=list(per_seed_rows[0].keys()),
        rows=per_seed_rows,
    )
    _write_csv(
        results_dir / "fig5_top_summary.csv",
        fieldnames=list(summary_rows[0].keys()),
        rows=summary_rows,
    )
    _write_csv(
        results_dir / "linear_baselines_comparison.csv",
        fieldnames=list(baseline_rows[0].keys()),
        rows=baseline_rows,
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), dpi=200)
    steps = np.arange(1, T + 1)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    ax_left = axes[0]
    for d, col in zip(dimensions, colors):
        c_mat = alg1_curves_by_d[d]
        mean_curve = np.mean(c_mat, axis=0)
        std_curve = np.std(c_mat, axis=0, ddof=1) if c_mat.shape[0] > 1 else np.zeros_like(mean_curve)
        ax_left.plot(steps, mean_curve, lw=2.0, color=col, label=f"$d = {d}$")
        ax_left.fill_between(
            steps,
            np.maximum(1e-3, mean_curve - std_curve),
            mean_curve + std_curve,
            color=col,
            alpha=0.18,
        )

    ax_left.set_yscale("log")
    ax_left.set_xlim(0, T)
    ax_left.set_ylim(1e-1, 1e5)
    ax_left.set_xlabel("Time $t$", fontsize=11)
    ax_left.set_ylabel("Cumulative Regret $\\mathrm{Reg}(t)$ (log scale)", fontsize=11)
    ax_left.set_title(
        r"Figure 5 (top-left): Algorithm 1 Regret ($\eta_{\max}=1/\sqrt{d}, \ell=8$)",
        fontsize=11.5,
        fontweight="bold",
    )
    ax_left.grid(True, which="both", alpha=0.3)
    ax_left.legend(loc="lower right", ncol=3, fontsize=9)

    ax_right = axes[1]
    assert traj_d128_hist is not None
    ax_right.plot(
        traj_d128_hist.theta_star_path[:, 0],
        traj_d128_hist.theta_star_path[:, 1],
        "o-",
        color="#1f77b4",
        lw=1.2,
        ms=3.0,
        label=r"$\hat{\theta}^*_t$ (Oracle)",
    )
    ax_right.plot(
        traj_d128_hist.theta_path[:, 0],
        traj_d128_hist.theta_path[:, 1],
        "o-",
        color="#ff7f0e",
        lw=0.9,
        ms=1.8,
        alpha=0.85,
        label=r"$\theta_t$ (SGD)",
    )
    ax_right.set_xlim(-1.05, 1.05)
    ax_right.set_ylim(-1.05, 1.05)
    ax_right.set_xlabel(r"First coordinate $\theta(1)$", fontsize=11)
    ax_right.set_ylabel(r"Second coordinate $\theta(2)$", fontsize=11)
    ax_right.set_title(
        f"Figure 5 (top-right): 2D Projection of SGD Trajectory ($d={traj_d}, B_t={B}$)",
        fontsize=11.5,
        fontweight="bold",
    )
    ax_right.grid(True, alpha=0.3)
    ax_right.legend(loc="upper right", fontsize=9.5)

    fig.tight_layout()
    fig.savefig(figures_dir / "fig5_top_linear_regret_and_path.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=200)
    method_labels = [
        ("alg1-oracle", "Algorithm 1 (Oracle $\\gamma_{t-1}$)", "#1f77b4"),
        ("alg1-ema", "Algorithm 1 (EMA $\\hat{\\gamma}_t$)", "#17becf"),
        ("constant", "Constant ($0.25 / \\sqrt{d}$)", "#ff7f0e"),
        ("inv-sqrt", "Inverse-Sqrt ($1 / \\sqrt{d \\cdot t}$)", "#2ca02c"),
        ("inv-time", "Inverse-Time ($\\mathcal{O}(1/t)$)", "#d62728"),
    ]
    x_pos = np.arange(len(dimensions))
    bar_w = 0.15

    for m_idx, (m_key, m_label, m_col) in enumerate(method_labels):
        vals = [
            next(r["final_regret_mean"] for r in baseline_rows if r["d"] == d and r["method"] == m_key)
            for d in dimensions
        ]
        stds = [
            next(r["final_regret_std"] for r in baseline_rows if r["d"] == d and r["method"] == m_key)
            for d in dimensions
        ]
        offset = (m_idx - 2.0) * bar_w
        ax.bar(
            x_pos + offset,
            vals,
            width=bar_w,
            yerr=stds,
            capsize=3,
            color=m_col,
            alpha=0.9,
            label=m_label,
        )

    ax.set_yscale("log")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"$d = {d}$" for d in dimensions], fontsize=10.5)
    ax.set_ylabel("Final Cumulative Regret $\\mathrm{Reg}(T)$ (log scale)", fontsize=10.5)
    ax.set_title(
        "Linear Regression under Distribution Shift: Algorithm 1 vs. Classical Baselines ($T=2000, B_t=256$)",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig5_top_baselines_comparison.png", bbox_inches="tight")
    plt.close(fig)

    return summary_rows, baseline_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Linear Regression experiments (Figure 4 & Figure 5-top) for Task T14."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "linear.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--mode",
        choices=("online", "offline", "disabled"),
        default="offline",
        help="W&B tracking mode (default: offline).",
    )
    parser.add_argument(
        "--skip-cached-baselines",
        action="store_true",
        help="Reuse already-computed baseline rows when logging online.",
    )
    args = parser.parse_args()

    t_start = time.time()
    cfg = load_config(args.config)
    seeds = [int(s) for s in cfg.get("seeds", [0, 1, 2])]

    figures_dir = PROJECT_ROOT / "figures"
    results_dir = PROJECT_ROOT / "results" / "linear"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print(f"[T14] Starting Linear Regression Full Reproduction (mode={args.mode}, seeds={seeds})", flush=True)
    print("=" * 80, flush=True)

    if not (figures_dir / "fig1_constant_lr_trajectories.png").exists():
        generate_theory_figures(figures_dir=figures_dir)
        print("[T14] Generated theory figures: fig1_*, fig2_*, fig3_* in figures/", flush=True)

    run_figure4_experiment(
        cfg=cfg,
        seeds=seeds,
        mode=args.mode,
        figures_dir=figures_dir,
        results_dir=results_dir,
    )

    run_figure5_top_experiment(
        cfg=cfg,
        seeds=seeds,
        mode=args.mode,
        figures_dir=figures_dir,
        results_dir=results_dir,
        skip_cached_baselines=args.skip_cached_baselines or (args.mode == "online"),
    )

    elapsed = time.time() - t_start
    print("=" * 80, flush=True)
    print(f"[OK] Task T14 completed in {elapsed:.2f}s. Figures saved to {figures_dir}, CSVs to {results_dir}.", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
