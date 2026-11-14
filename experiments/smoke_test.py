"""
================================================================================
FILE: experiments/smoke_test.py
MỤC ĐÍCH:
    Script kiểm thử nhanh (< 1 phút) toàn bộ khung code dùng chung `core/` (Task T6)
    và tự động vẽ sơ đồ Pipeline tổng thể 2 giai đoạn (Task T5):
    1. Vẽ và xuất sơ đồ `figures/pipeline.png` minh họa rõ cả 2 quá trình:
       - Stage A: Online Inference & Evaluation Pipeline (Test-First).
       - Stage B: Online Training & Adaptive Learning Rate Pipeline.
    2. Kiểm thử hàm sinh quỹ đạo xoắn ốc `generate_spiral_trajectory` (Công thức 26).
    3. Kiểm thử `OnlineSGDTrainer` kết hợp với 3 lịch trình baseline (`ConstantSchedule`,
       `InverseTimeSchedule`, `InverseSqrtSchedule`) và module `core/tracking.py`
       (`init_run`) của Khôi (Task T2).
    4. Xuất đồ thị kiểm chứng `figures/smoke_test.png`.

CÁCH CHẠY:
    - Chạy kiểm thử nhanh không cần mạng (offline W&B):
      `& './.venv-t2-clean/Scripts/python.exe' experiments/smoke_test.py --mode offline`
    - Chạy đẩy thẳng run lên W&B Public Project của nhóm (`gr_ml_29_37_37/ml-distribution-shift`):
      `& './.venv-t2-clean/Scripts/python.exe' experiments/smoke_test.py --mode online`
    - Chạy tắt W&B (chỉ kiểm tra toán học & xuất hình):
      `& './.venv-t2-clean/Scripts/python.exe' experiments/smoke_test.py --mode disabled`
================================================================================
"""

import argparse
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Đảm bảo thư mục gốc của repo nằm trong sys.path khi chạy trực tiếp file này
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.drift import generate_spiral_trajectory
from core.schedules import ConstantSchedule, InverseSqrtSchedule, InverseTimeSchedule
from core.tracking import init_run
from core.trainer import OnlineSGDTrainer
from core.utils import set_seed


def draw_pipeline_diagram(output_path: Path) -> None:
    """Vẽ sơ đồ Pipeline tổng thể gồm cả Quá trình Inference và Quá trình Training (Task T5).

    Theo yêu cầu của giảng viên đối với chủ đề Never-Ending / Online Learning,
    sơ đồ tách biệt rõ 2 tầng:
    - Tầng trên (Stage A - Màu xanh dương): Quá trình Suy luận & Đánh giá Trực tuyến
      (dự đoán bằng trọng số \\theta_t đóng băng trước khi nhìn thấy nhãn, tính Dynamic Regret).
    - Tầng dưới (Stage B - Màu xanh lá): Quá trình Huấn luyện & Điều khiển Learning Rate
      Thích ứng (tính gradient, ước lượng độ trôi \\gamma, cập nhật \\eta^*_t và \\theta_{t+1}).

    Parameters
    ----------
    output_path : Path
        Đường dẫn lưu file ảnh PNG (`figures/pipeline.png`).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14.0, 7.6), dpi=200)
    ax.set_xlim(0, 14.0)
    ax.set_ylim(0, 7.6)
    ax.axis("off")

    # Khung nền cho Tầng trên: STAGE A — ONLINE INFERENCE / TESTING PIPELINE
    infer_bg = mpatches.FancyBboxPatch(
        (0.25, 4.15),
        13.5,
        2.75,
        boxstyle="round,pad=0.08,rounding_size=0.18",
        facecolor="#F4F8FE",
        edgecolor="#1A73E8",
        linewidth=2.0,
        linestyle="-",
    )
    ax.add_patch(infer_bg)
    ax.text(
        0.5,
        6.62,
        "STAGE A — ONLINE INFERENCE & EVALUATION PIPELINE (Test-First / Prequential Serving at Step t)",
        ha="left",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="#174EA6",
    )

    # Khung nền cho Tầng dưới: STAGE B — ONLINE TRAINING & ADAPTATION PIPELINE
    train_bg = mpatches.FancyBboxPatch(
        (0.25, 0.55),
        13.5,
        2.95,
        boxstyle="round,pad=0.08,rounding_size=0.18",
        facecolor="#F6FBF7",
        edgecolor="#1E8E3E",
        linewidth=2.0,
        linestyle="-",
    )
    ax.add_patch(train_bg)
    ax.text(
        0.5,
        3.22,
        "STAGE B — ONLINE TRAINING & ADAPTIVE LR PIPELINE (Post-Feedback Model Update for Step t + 1)",
        ha="left",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="#0D652D",
    )

    # Các khối chức năng ở Tầng trên (Inference: đi từ trái sang phải 1 -> 2 -> 3)
    top_boxes = [
        {
            "x": 0.5,
            "y": 4.38,
            "w": 3.8,
            "h": 1.95,
            "color": "#FFFFFF",
            "edge": "#1A73E8",
            "title": "1. Shifting Data Stream $P_t$",
            "body": "Incoming unlabeled batch $\\{x_{t,k}\\}_{k=1}^{B_t} \\sim P_t$\n"
                    "Hidden oracle $\\theta^*_t = \\arg\\min_\\theta \\bar{\\ell}_t(\\theta)$\n"
                    "Distribution shift: $\\gamma_t = \\|\\theta^*_t - \\theta^*_{t+1}\\|_2$",
        },
        {
            "x": 4.85,
            "y": 4.38,
            "w": 3.8,
            "h": 1.95,
            "color": "#FFFFFF",
            "edge": "#1A73E8",
            "title": "2. Real-Time Inference (Frozen $\\theta_t$)",
            "body": "Predict before seeing true labels $y_t$:\n"
                    "$\\hat{y}_{t,k} = f(x_{t,k};\\, \\theta_t)$\n"
                    "(e.g., Linear $\\langle x, \\theta_t\\rangle$ or Student MLP)",
        },
        {
            "x": 9.2,
            "y": 4.38,
            "w": 4.25,
            "h": 1.95,
            "color": "#FEF7E0",
            "edge": "#F29900",
            "title": "3. Label Feedback & Dynamic Regret",
            "body": "Observe ground-truth / Teacher labels $y_{t,k}$\n"
                    "Incur instantaneous & cumulative regret:\n"
                    "$\\mathrm{reg}_t = \\bar{\\ell}_t(\\theta_t) - \\bar{\\ell}_t(\\theta^*_t),\\;\\; "
                    "\\mathrm{Reg}(T) = \\sum_{s=1}^t \\mathrm{reg}_s$",
        },
    ]

    # Các khối chức năng ở Tầng dưới (Training: đi từ phải sang trái 4 -> 5 -> 6 tạo vòng khép kín)
    bottom_boxes = [
        {
            "x": 9.2,
            "y": 0.8,
            "w": 4.25,
            "h": 2.1,
            "color": "#FCE8E6",
            "edge": "#D93025",
            "title": "4. Batch Gradient & Shift Tracking",
            "body": "Form labeled batch $z_t = \\{(x_{t,k}, y_{t,k})\\}_{k=1}^{B_t}$\n"
                    "Stochastic grad: $\\nabla \\ell^B_t(\\theta_t) = \\frac{1}{B_t}\\sum_k \\nabla \\ell(\\theta_t, z_{t,k})$\n"
                    "Shift signal: Oracle $\\gamma_{t-1}$ or EMA $\\hat{\\gamma}_t = \\beta\\hat{\\gamma}_{t-1}+(1-\\beta)\\|\\Delta\\theta_t\\|$",
        },
        {
            "x": 4.85,
            "y": 0.8,
            "w": 3.8,
            "h": 2.1,
            "color": "#E6F4EA",
            "edge": "#1E8E3E",
            "title": "5. Adaptive LR Schedule $\\eta^*_t$",
            "body": "• Linear (Sec. 3): SDE/HJB Euler ODE (Alg. 1)\n"
                    "• Convex (Sec. 4): Clip $[\\tau_{1,t}, \\tau_{2,t}]$ (Prop. 4.3)\n"
                    "• Non-Convex (Sec. 5): Eq. (25) / Adam Restart\n"
                    "(Increases $\\eta^*_t$ when shift $\\gamma$ spikes!)",
        },
        {
            "x": 0.5,
            "y": 0.8,
            "w": 3.8,
            "h": 2.1,
            "color": "#F3E8FD",
            "edge": "#9334E6",
            "title": "6. Projected Mini-Batch SGD Update",
            "body": "Update weights for next round:\n"
                    "$\\theta_{t+1} = \\Pi_\\Theta\\!\\left(\\theta_t - \\eta^*_t \\nabla \\ell^B_t(\\theta_t)\\right)$\n"
                    "Log $(\\eta^*_t, \\mathrm{reg}_t, \\mathrm{Reg}(t))$ to Weights & Biases",
        },
    ]

    for b in top_boxes + bottom_boxes:
        rect = mpatches.FancyBboxPatch(
            (b["x"], b["y"]),
            b["w"],
            b["h"],
            boxstyle="round,pad=0.06,rounding_size=0.14",
            facecolor=b["color"],
            edgecolor=b["edge"],
            linewidth=2.0,
        )
        ax.add_patch(rect)
        cx = b["x"] + b["w"] / 2.0
        ax.text(
            cx,
            b["y"] + b["h"] - 0.28,
            b["title"],
            ha="center",
            va="center",
            fontsize=10.0,
            fontweight="bold",
            color="#202124",
        )
        ax.text(
            cx,
            b["y"] + (b["h"] - 0.42) / 2.0,
            b["body"],
            ha="center",
            va="center",
            fontsize=8.6,
            color="#3C4043",
        )

    # Mũi tên nối các khối ở Tầng trên (1 -> 2 -> 3)
    y_top = 5.35
    ax.annotate(
        "",
        xy=(top_boxes[1]["x"] - 0.04, y_top),
        xytext=(top_boxes[0]["x"] + top_boxes[0]["w"] + 0.04, y_top),
        arrowprops=dict(arrowstyle="->,head_width=0.32,head_length=0.32", lw=2.2, color="#1A73E8"),
    )
    ax.annotate(
        "",
        xy=(top_boxes[2]["x"] - 0.04, y_top),
        xytext=(top_boxes[1]["x"] + top_boxes[1]["w"] + 0.04, y_top),
        arrowprops=dict(arrowstyle="->,head_width=0.32,head_length=0.32", lw=2.2, color="#1A73E8"),
    )

    # Mũi tên chuyển từ Stage A (Khối 3) xuống Stage B (Khối 4)
    x_right = top_boxes[2]["x"] + top_boxes[2]["w"] / 2.0
    ax.annotate(
        "",
        xy=(x_right, bottom_boxes[0]["y"] + bottom_boxes[0]["h"] + 0.05),
        xytext=(x_right, top_boxes[2]["y"] - 0.05),
        arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.35", lw=2.4, color="#D93025"),
    )
    ax.text(
        x_right + 0.15,
        3.82,
        "Pass labeled batch $z_t = (x_t, y_t)$\nfrom Inference to Training",
        ha="left",
        va="center",
        fontsize=8.8,
        fontweight="bold",
        color="#B31412",
    )

    # Mũi tên nối các khối ở Tầng dưới (4 -> 5 -> 6, đi từ phải sang trái)
    y_bot = 1.85
    ax.annotate(
        "",
        xy=(bottom_boxes[1]["x"] + bottom_boxes[1]["w"] + 0.04, y_bot),
        xytext=(bottom_boxes[0]["x"] - 0.04, y_bot),
        arrowprops=dict(arrowstyle="->,head_width=0.32,head_length=0.32", lw=2.2, color="#1E8E3E"),
    )
    ax.annotate(
        "",
        xy=(bottom_boxes[2]["x"] + bottom_boxes[2]["w"] + 0.04, y_bot),
        xytext=(bottom_boxes[1]["x"] - 0.04, y_bot),
        arrowprops=dict(arrowstyle="->,head_width=0.32,head_length=0.32", lw=2.2, color="#1E8E3E"),
    )

    # Mũi tên vòng lặp triển khai trọng số mới \theta_{t+1} từ Khối 6 lên Khối 2 cho bước t+1
    ax.annotate(
        "",
        xy=(4.85, 4.65),
        xytext=(4.30, 2.55),
        arrowprops=dict(
            arrowstyle="->,head_width=0.35,head_length=0.35",
            lw=2.4,
            color="#9334E6",
            connectionstyle="arc3,rad=-0.12",
        ),
    )
    ax.text(
        4.85,
        3.82,
        "Deploy updated model $\\theta_{t+1}$ for step $t \\leftarrow t + 1$ (Never-Ending Loop)",
        ha="left",
        va="center",
        fontsize=8.8,
        fontweight="bold",
        color="#681DA8",
    )

    ax.set_title(
        "Never-Ending Online Learning Pipeline: Inference (Test-First) & Adaptive SGD Training",
        fontsize=13.0,
        fontweight="bold",
        pad=10,
        color="#202124",
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run_smoke_test(mode: str = "offline", seed: int = 42) -> None:
    """Chạy kiểm thử tích hợp toàn bộ `core/` và xuất đồ thị `figures/smoke_test.png`.

    Parameters
    ----------
    mode : str, optional
        Chế độ W&B truyền vào `core.tracking.init_run`: `"offline"` (mặc định),
        `"online"`, hoặc `"disabled"`.
    seed : int, optional
        Hạt giống ngẫu nhiên cố định (mặc định: 42).
    """
    t0 = time.time()
    set_seed(seed)

    figures_dir = PROJECT_ROOT / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Sinh sơ đồ Pipeline 2 tầng (Task T5)
    pipeline_fig_path = figures_dir / "pipeline.png"
    draw_pipeline_diagram(pipeline_fig_path)
    print(f"[T5] Saved 2-stage pipeline diagram to: {pipeline_fig_path}")

    # 2. Sinh quỹ đạo xoắn ốc 2 chiều (Công thức 26)
    T = 500
    d = 2
    batch_size = 256
    ell = 8
    theta_star, gamma_seq = generate_spiral_trajectory(T=T, d=d, a=1.0, b=-1.0, k=4.0, ell=ell)
    assert theta_star.shape == (T, d), f"Expected shape {(T, d)}, got {theta_star.shape}"
    assert gamma_seq.shape == (T,), f"Expected shape {(T,)}, got {gamma_seq.shape}"
    print(
        f"[T6] Generated spiral trajectory: shape={theta_star.shape}, "
        f"max shift gamma={gamma_seq.max():.4f}, non-zero jumps={(gamma_seq > 0).sum()}"
    )

    # 3. Chạy thử 3 lịch trình baseline trên bài toán Linear Regression + log W&B qua `init_run`
    schedules = [
        ConstantSchedule(lr=0.05),
        InverseTimeSchedule(lr0=0.2, decay=0.02),
        InverseSqrtSchedule(lr0=0.2),
    ]

    histories = []
    for sched in schedules:
        # Khởi tạo W&B run theo đúng quy ước `core/tracking.py` của Khôi (Task T2)
        run = init_run(
            {
                "experiment": "linear",
                "method": sched.name,
                "seed": seed,
                "mode": mode,
                "group": "T6-smoke-test",
                "job_type": "smoke-test",
                "tags": ["smoke-test", "linear"],
                "T": T,
                "d": d,
                "batch_size": batch_size,
                "ell": ell,
            }
        )
        try:
            trainer = OnlineSGDTrainer(
                d=d,
                schedule=sched,
                loss_type="linear",
                batch_size=batch_size,
                noise_std=0.1,
                use_oracle_gamma=False,  # Kiểm thử luôn cả bộ ước lượng EMADriftEstimator
                use_wandb=True,
                wandb_log_interval=25,
            )
            hist = trainer.run(theta_star_path=theta_star, gamma_seq=gamma_seq, seed=seed)
            run.summary["eval/final_cum_regret"] = float(hist.cum_regret[-1])
            run.summary["smoke/completed"] = True
        except BaseException:
            try:
                run.finish(exit_code=1)
            except Exception:
                pass
            raise
        else:
            run.finish()

        histories.append(hist)
        print(
            f"[T6] Linear SGD ({sched.name:<15}) -> Final Cum. Regret = {hist.cum_regret[-1]:.2f}, "
            f"Final LR = {hist.lr_history[-1]:.5f} (W&B mode={mode}, run_id={run.id})"
        )

    # 4. Kiểm thử nhanh chế độ Logistic Regression (Binary Cross-Entropy)
    logistic_trainer = OnlineSGDTrainer(
        d=d,
        schedule=ConstantSchedule(lr=0.1),
        loss_type="logistic",
        batch_size=batch_size,
        noise_std=0.1,
        max_norm=float(d),
        use_wandb=False,
    )
    log_hist = logistic_trainer.run(theta_star_path=theta_star, gamma_seq=gamma_seq, seed=seed)
    print(f"[T6] Logistic SGD (constant-0.1   ) -> Final Cum. Regret = {log_hist.cum_regret[-1]:.2f}")

    # 5. Vẽ và lưu đồ thị tổng hợp kiểm thử `figures/smoke_test.png`
    smoke_fig_path = figures_dir / "smoke_test.png"
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), dpi=160)

    axes[0].plot(theta_star[:, 0], theta_star[:, 1], "o-", lw=1.2, ms=2.5, label=r"Oracle $\hat{\theta}^*_t$")
    axes[0].plot(
        histories[0].theta_path[:, 0],
        histories[0].theta_path[:, 1],
        "-",
        lw=1.2,
        alpha=0.85,
        label=f"SGD ({histories[0].schedule_name})",
    )
    axes[0].set_title("2D Spiral Trajectory ($\\ell=8$)")
    axes[0].set_xlabel(r"$\theta(1)$")
    axes[0].set_ylabel(r"$\theta(2)$")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    for h in histories:
        axes[1].plot(np.arange(1, T + 1), h.lr_history, lw=1.8, label=h.schedule_name)
    axes[1].set_title("Baseline Learning Rate Schedules $\\eta_t$")
    axes[1].set_xlabel("Step $t$")
    axes[1].set_ylabel("Learning rate $\\eta_t$")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    for h in histories:
        axes[2].plot(np.arange(1, T + 1), h.cum_regret, lw=1.8, label=h.schedule_name)
    axes[2].set_title("Cumulative Dynamic Regret $\\mathrm{Reg}(t)$")
    axes[2].set_xlabel("Step $t$")
    axes[2].set_ylabel("Cumulative Regret")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(smoke_fig_path, bbox_inches="tight")
    plt.close(fig)

    elapsed = time.time() - t0
    print(f"[T6] Saved smoke test figure to: {smoke_fig_path}")
    print(f"[OK] Smoke test completed successfully in {elapsed:.2f}s (< 60s requirement).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run smoke test for core/ online SGD framework.")
    parser.add_argument(
        "--mode",
        choices=("online", "offline", "disabled"),
        default="offline",
        help="W&B mode passed to core.tracking.init_run (default: offline).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    args = parser.parse_args()
    run_smoke_test(mode=args.mode, seed=args.seed)
