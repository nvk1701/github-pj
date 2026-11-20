r"""
================================================================================
FILE: experiments/neural_net.py
MỤC ĐÍCH:
    Thực thi trọn vẹn Thực nghiệm 3 — Mạng Nơ-ron Sâu Phân loại Tế bào Dòng chảy
    Trực tuyến (Flow Cytometry 5-Layer MLP under Distribution Shift — Task T12,
    T16 & T18), tái hiện đầy đủ Mục 5 và Phụ lục E (Figure 6, trang 22–23) của
    bài báo Fahrbach et al. (ICML 2023):

    1. [Kiến trúc Mạng `CytometryMLP` (Task T12)]:
       - Đầu vào: 10 marker genes (`Pyy, Meg3, Malat1, Gcg, Gnas, Actb, Ghrl, Rps3, Ins2, Hspa8`).
       - 5 tầng tuyến tính: `10 -> 64 -> 32 -> 16 -> 8 -> 4` với hàm kích hoạt `ELU`
         ở các tầng ẩn, khởi tạo Xavier/Glorot Uniform và điều chuẩn L2 (`1e-4`).
       - Hàm mất mát: Categorical Cross-Entropy từ logits (`nn.CrossEntropyLoss()`).
       - Bộ tối ưu: Adam ($\eta_0 \in \{0.01, 0.03\}$) kèm cơ chế **Adam Restart**
         (xóa bộ đệm moment bậc nhất $m_t$ và bậc hai $v_t$ tại các mốc dịch chuyển
         phân phối $t \in \{100, 200, 400\}$), cùng đối chứng `NonConvexSchedule` (Eq. 25).

    2. [Figure 6 (left & right) - Phụ lục E, trang 23 (Task T16)]:
       - Figure 6 (left): Biểu đồ nhúng 2D UMAP của 3.696 tế bào tuyến tụy chuột
         được tô màu theo 8 phân lớp tế bào thực tế (`Ductal`, `Alpha`, `Beta`, `Delta`,
         `Ngn3 low EP`, `Ngn3 high EP`, `Pre-endocrine`, `Epsilon`).
       - Figure 6 (right): Đường cong Cumulative Dynamic Regret qua T = 600 bước
         (4 pha phân phối P_1 -> P_2 -> P_3 -> P_4, B_t = 64) của 4 cấu hình:
         `adam:0.01`, `adam-restart:0.01`, `adam:0.03`, `adam-restart:0.03` trên 3 seeds.
       - Xuất hình:
         * `figures/fig6_cytometry_2d_and_regret.png` (Ghép đôi Figure 6 left & right)
         * `figures/fig6_left_cytometry_embedding.png` (Dùng cho Slide 27)
         * `figures/fig6_right_adam_restart_regret.png` (Dùng cho Slide 28)
         * `figures/fig6_lr_sweep_and_author_bug.png` (Dùng cho Slide 28 & Slide 30)
       - Xuất CSV:
         * `results/nn/fig6_per_seed.csv`
         * `results/nn/fig6_summary.csv`
         * `results/nn/fig6_lr_sweep.csv`
         * `results/nn/t18_author_vs_fixed_restart.csv`

CÁCH CHẠY:
    - Chạy đẩy toàn bộ 12 runs chính thức lên W&B Public (`gr_ml_29_37_37/ml-distribution-shift`):
      `& './.venv-t2-clean/Scripts/python.exe' experiments/neural_net.py --mode online`
    - Chạy kiểm thử nhanh ở chế độ offline hoặc disabled:
      `& './.venv-t2-clean/Scripts/python.exe' experiments/neural_net.py --mode disabled`
================================================================================
"""

import argparse
import csv
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_MPL_DIR = PROJECT_ROOT / ".mplconfig"
_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from core.data import (
    CELL_CLASSES,
    PANCREAS_10_GENES,
    PancreasCytometryStream,
    PancreasDataset,
    load_pancreas_dataset,
)
from core.schedules import NonConvexSchedule
from core.tracking import init_run
from core.utils import load_config, set_seed


class CytometryMLP(nn.Module):
    """Mạng Feedforward MLP 5 tầng phân loại tế bào dòng chảy (Phụ lục E.2, trang 23).

    Cấu trúc các tầng:
        Input(10) -> Linear(10, 64) -> ELU
                  -> Linear(64, 32) -> ELU
                  -> Linear(32, 16) -> ELU
                  -> Linear(16,  8) -> ELU
                  -> Linear( 8,  4) -> Logits (4 classes: Alpha, Beta, Delta, Ductal)
    Khởi tạo trọng số theo phân phối Glorot/Xavier Uniform giống `keras.initializers.GlorotUniform`.
    """

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dims: Sequence[int] = (64, 32, 16, 8),
        output_dim: int = 4,
        seed: int = 0,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = int(input_dim)
        gen = torch.Generator()
        gen.manual_seed(int(seed))

        for h_dim in hidden_dims:
            linear = nn.Linear(prev_dim, int(h_dim))
            nn.init.xavier_uniform_(linear.weight, generator=gen)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(nn.ELU())
            prev_dim = int(h_dim)

        out_layer = nn.Linear(prev_dim, int(output_dim))
        nn.init.xavier_uniform_(out_layer.weight, generator=gen)
        nn.init.zeros_(out_layer.bias)
        layers.append(out_layer)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tính vector logits chưa chuẩn hóa kích thước `(B, 4)`."""
        return self.net(x)

    def l2_penalty(self, coeff: float = 1e-4) -> torch.Tensor:
        """Tính tổng phạt điều chuẩn L2 trên các ma trận trọng số giống `keras.regularizers.L2(1e-4)`."""
        if coeff <= 0.0:
            return torch.tensor(0.0)
        total = torch.tensor(0.0)
        for module in self.net:
            if isinstance(module, nn.Linear):
                total = total + torch.sum(torch.square(module.weight))
        return float(coeff) * total


@dataclass
class CytometryRunHistory:
    """Lịch sử một lần chạy huấn luyện trực tuyến mạng `CytometryMLP` qua T = 600 bước."""

    method: str
    seed: int
    restart_rule: str
    train_loss: np.ndarray
    val_loss: np.ndarray
    cum_regret: np.ndarray
    cum_train_regret: np.ndarray
    grad_norm_sq: np.ndarray
    lr_history: np.ndarray
    accuracy_history: np.ndarray
    restart_steps: List[int]


def run_cytometry_online_experiment(
    stream: PancreasCytometryStream,
    method_str: str,
    seed: int = 0,
    l2_reg: float = 1e-4,
    restart_rule: str = "paper",
) -> CytometryRunHistory:
    """Chạy mô phỏng học trực tuyến 600 bước trên luồng tế bào tụy `PancreasCytometryStream`.

    Tại mỗi bước t = 1..600 (chỉ số 0..599):
    1. Nếu `method_str` có chứa `'restart'` và `stream.is_shift_boundary(step_idx, restart_rule)`
       bằng `True`, khởi tạo lại hoàn toàn bộ tối ưu Adam (xóa bộ nhớ moment bậc 1 & bậc 2).
    2. Sinh 2 mini-batch độc lập từ phân phối hiện tại P_t (mỗi batch B_t = 64 mẫu, tương ứng
       `2 * batch_size` với `validation_split = 0.5` trong code gốc của tác giả).
    3. Cập nhật trọng số mô hình 1 bước bằng batch huấn luyện và đo lường mất mát / độ chính xác
       trên batch đánh giá độc lập.
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)

    tokens = method_str.split(":")
    opt_family = tokens[0].lower()
    base_lr = float(tokens[1]) if len(tokens) >= 2 else 0.01

    model = CytometryMLP(input_dim=10, hidden_dims=(64, 32, 16, 8), output_dim=4, seed=seed)
    ce_criterion = nn.CrossEntropyLoss()

    nonconvex_sched: NonConvexSchedule | None = None
    if opt_family in ("adam", "adam-restart"):
        optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    elif opt_family == "nonconvex-sgd":
        nonconvex_sched = NonConvexSchedule(
            L=1.0 / base_lr,
            gradient_variance=1.0,
            default_batch_size=stream.batch_size,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=base_lr)
    elif opt_family == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=base_lr)
    else:
        raise ValueError(f"Bộ tối ưu '{method_str}' không được hỗ trợ.")

    T = stream.total_steps
    train_loss_arr = np.zeros(T, dtype=np.float64)
    val_loss_arr = np.zeros(T, dtype=np.float64)
    grad_norm_sq_arr = np.zeros(T, dtype=np.float64)
    lr_arr = np.zeros(T, dtype=np.float64)
    acc_arr = np.zeros(T, dtype=np.float64)
    restart_steps: List[int] = []

    prev_val_loss = 0.0

    for step_idx in range(T):
        # Kiểm tra điều kiện Restart bộ tối ưu khi phân phối dịch chuyển
        is_shift = stream.is_shift_boundary(step_idx, restart_rule=restart_rule)
        if "restart" in opt_family and is_shift:
            optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
            restart_steps.append(step_idx)

        # Sinh cặp batch (train 64 mẫu, eval 64 mẫu) từ phân phối hỗn hợp hiện tại P_t
        X_np, y_np = stream.sample_batch(step_idx, rng, num_samples=2 * stream.batch_size)
        B = stream.batch_size
        X_tr = torch.from_numpy(X_np[:B])
        y_tr = torch.from_numpy(y_np[:B])
        X_va = torch.from_numpy(X_np[B:])
        y_va = torch.from_numpy(y_np[B:])

        # Pha huấn luyện (Online Gradient Step)
        model.train()
        optimizer.zero_grad()
        logits_tr = model(X_tr)
        ce_tr = ce_criterion(logits_tr, y_tr)
        loss_tr = ce_tr + model.l2_penalty(l2_reg)
        loss_tr.backward()

        # Tính tổng bình phương chuẩn gradient ||\nabla \ell_t(\theta_t)||_2^2 (Eq. 22)
        g_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                g_sq += float(torch.sum(torch.square(p.grad)).item())
        grad_norm_sq_arr[step_idx] = g_sq

        # Nếu dùng NonConvexSchedule (Theorem 5.1 - Eq. 25), cập nhật lr động
        if nonconvex_sched is not None:
            gamma_est = abs(float(loss_tr.item()) - prev_val_loss) if step_idx > 0 else 0.0
            eta_t = nonconvex_sched.step(
                t=step_idx + 1,
                gamma_prev=gamma_est,
                loss_val=float(loss_tr.item()),
                batch_size=B,
            )
            for pg in optimizer.param_groups:
                pg["lr"] = eta_t
            lr_arr[step_idx] = eta_t
        else:
            # Với Adam Restart, ghi nhận bước nhảy hiệu dụng tăng vọt khi reset buffer
            lr_arr[step_idx] = base_lr

        optimizer.step()

        # Pha đánh giá (Online Validation / Regret Tracking)
        model.eval()
        with torch.no_grad():
            logits_va = model(X_va)
            ce_va = ce_criterion(logits_va, y_va)
            loss_va = ce_va + model.l2_penalty(l2_reg)
            preds_va = torch.argmax(logits_va, dim=1)
            acc_va = float(torch.mean((preds_va == y_va).to(torch.float32)).item())

        tr_val = float(loss_tr.item())
        va_val = float(loss_va.item())
        train_loss_arr[step_idx] = tr_val
        val_loss_arr[step_idx] = va_val
        acc_arr[step_idx] = acc_va
        prev_val_loss = va_val

    return CytometryRunHistory(
        method=method_str,
        seed=seed,
        restart_rule=restart_rule,
        train_loss=train_loss_arr,
        val_loss=val_loss_arr,
        cum_regret=np.cumsum(val_loss_arr),
        cum_train_regret=np.cumsum(train_loss_arr),
        grad_norm_sq=grad_norm_sq_arr,
        lr_history=lr_arr,
        accuracy_history=acc_arr,
        restart_steps=restart_steps,
    )


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_figure6_left_umap(dataset: PancreasDataset, ax: plt.Axes) -> None:
    """Vẽ Figure 6 (left): Nhúng 2D UMAP của 3.696 tế bào tuyến tụy theo 8 phân lớp thực tế."""
    umap = dataset.X_umap
    codes = dataset.cluster_codes
    names = dataset.cluster_names
    colors = dataset.cluster_colors

    for cluster_id, (c_name, c_color) in enumerate(zip(names, colors)):
        mask = codes == cluster_id
        if not np.any(mask):
            continue
        pts = umap[mask]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=11,
            c=c_color,
            alpha=0.78,
            edgecolors="none",
            label=c_name,
        )
        centroid = np.median(pts, axis=0)
        ax.text(
            float(centroid[0]),
            float(centroid[1]),
            c_name,
            fontsize=8.8,
            fontweight="bold",
            ha="center",
            va="center",
            color="#111827",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.68),
        )

    ax.set_title("Figure 6 (left): 10-Gene Pancreas Cytometry Embedding ($N=3,696$)", fontsize=11, fontweight="bold")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Flow Cytometry MLP experiments (Task T12, T16, T18).")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "cytometry.yaml"),
        help="Path to cytometry YAML config.",
    )
    parser.add_argument(
        "--mode",
        choices=("online", "offline", "disabled"),
        default="offline",
        help="W&B tracking mode (default: offline).",
    )
    args = parser.parse_args()

    t0 = time.time()
    cfg = load_config(args.config)
    seeds = [int(s) for s in cfg.get("seeds", [0, 1, 2])]
    methods = [str(m) for m in cfg["figure6_methods"]]
    l2_reg = float(cfg["model"]["l2_reg"])

    dataset = load_pancreas_dataset()
    stream_cfg = cfg["stream"]
    stream = PancreasCytometryStream(
        dataset=dataset,
        batch_size=int(stream_cfg["batch_size"]),
        perturb_eps=float(stream_cfg["perturb_eps"]),
        steps_per_block=int(stream_cfg["steps_per_block"]),
        block_types=[int(b) for b in stream_cfg["block_types"]],
    )

    figures_dir = PROJECT_ROOT / "figures"
    results_dir = PROJECT_ROOT / "results" / "nn"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print(
        f"[T16] Starting Flow Cytometry 5-Layer MLP Reproduction (mode={args.mode}, T={stream.total_steps}, B={stream.batch_size})",
        flush=True,
    )
    print(
        f"[T11] Loaded Pancreas 10-gene dataset: total_cells={dataset.X_norm_10.shape[0]}, "
        f"genes={list(dataset.gene_names)}, class_counts={{k: v.shape[0] for k, v in dataset.class_pools.items()}}",
        flush=True,
    )
    print("=" * 80, flush=True)

    # --------------------------------------------------------------------------
    # 1. Chạy 4 phương pháp chính của Figure 6 (right) qua 3 seeds & đẩy W&B
    # --------------------------------------------------------------------------
    per_seed_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    curves_by_method: Dict[str, np.ndarray] = {}

    for method_str in methods:
        method_tag = method_str.replace(":", "-")
        seed_curves = []
        final_regrets = []
        final_train_regrets = []
        mean_accs = []
        run_urls = []

        for seed in seeds:
            hist = run_cytometry_online_experiment(
                stream=stream,
                method_str=method_str,
                seed=seed,
                l2_reg=l2_reg,
                restart_rule="paper",
            )

            run_id = "disabled"
            run_url = "disabled"
            if args.mode != "disabled":
                run = init_run(
                    {
                        "experiment": "nonconvex",
                        "method": method_tag,
                        "seed": seed,
                        "mode": args.mode,
                        "group": "nonconvex",
                        "job_type": "reproduction",
                        "tags": ["lab1", "nonconvex", "figure6", method_tag],
                        "T": stream.total_steps,
                        "batch_size": stream.batch_size,
                        "l2_reg": l2_reg,
                        "restart_rule": "paper",
                    }
                )
                run_id = str(getattr(run, "id", "offline"))
                run_url = str(
                    getattr(run, "url", None)
                    or f"https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift/runs/{run_id}"
                )
                try:
                    for step_idx in range(stream.total_steps):
                        t = step_idx + 1
                        if t % 10 == 0 or t == stream.total_steps:
                            run.log(
                                {
                                    "eval/val_loss": float(hist.val_loss[step_idx]),
                                    "eval/cum_regret": float(hist.cum_regret[step_idx]),
                                    "train/batch_loss": float(hist.train_loss[step_idx]),
                                    "train/cum_regret": float(hist.cum_train_regret[step_idx]),
                                    "train/grad_norm_sq": float(hist.grad_norm_sq[step_idx]),
                                    "eval/accuracy": float(hist.accuracy_history[step_idx]),
                                },
                                step=t,
                                commit=True,
                            )
                    run.summary["eval/final_cum_regret"] = float(hist.cum_regret[-1])
                    run.summary["train/final_cum_regret"] = float(hist.cum_train_regret[-1])
                    run.summary["eval/mean_accuracy"] = float(np.mean(hist.accuracy_history))
                    run.summary["completed"] = True
                except BaseException:
                    try:
                        run.finish(exit_code=1)
                    except Exception:
                        pass
                    raise
                else:
                    run.finish()

            seed_curves.append(hist.cum_regret)
            final_regrets.append(float(hist.cum_regret[-1]))
            final_train_regrets.append(float(hist.cum_train_regret[-1]))
            mean_accs.append(float(np.mean(hist.accuracy_history)))
            run_urls.append(run_url)

            per_seed_rows.append(
                {
                    "method": method_str,
                    "seed": seed,
                    "final_cum_regret": float(hist.cum_regret[-1]),
                    "final_train_cum_regret": float(hist.cum_train_regret[-1]),
                    "regret_p1_t100": float(hist.cum_regret[99]),
                    "regret_p2_t200": float(hist.cum_regret[199]),
                    "regret_p3_t400": float(hist.cum_regret[399]),
                    "regret_p4_t600": float(hist.cum_regret[599]),
                    "p3_p4_slope_per_step": float((hist.cum_regret[599] - hist.cum_regret[199]) / 400.0),
                    "mean_accuracy": float(np.mean(hist.accuracy_history)),
                    "restart_steps": ";".join(str(s) for s in hist.restart_steps),
                    "wandb_run_id": run_id,
                    "wandb_url": run_url,
                }
            )

        c_mat = np.stack(seed_curves, axis=0)
        curves_by_method[method_str] = c_mat
        reg_arr = np.asarray(final_regrets, dtype=np.float64)
        tr_arr = np.asarray(final_train_regrets, dtype=np.float64)
        acc_arr = np.asarray(mean_accs, dtype=np.float64)

        s_row = {
            "method": method_str,
            "regret_t100_mean": float(np.mean(c_mat[:, 99])),
            "regret_t200_mean": float(np.mean(c_mat[:, 199])),
            "regret_t400_mean": float(np.mean(c_mat[:, 399])),
            "final_regret_mean": float(np.mean(reg_arr)),
            "final_regret_std": float(np.std(reg_arr, ddof=1)) if len(reg_arr) > 1 else 0.0,
            "final_train_regret_mean": float(np.mean(tr_arr)),
            "final_train_regret_std": float(np.std(tr_arr, ddof=1)) if len(tr_arr) > 1 else 0.0,
            "mean_accuracy": float(np.mean(acc_arr)),
            "wandb_runs": ";".join(run_urls),
        }
        summary_rows.append(s_row)
        print(
            f"[T16 - Fig 6] {method_str:<18} | Final Regret = {s_row['final_regret_mean']:6.2f} ± {s_row['final_regret_std']:5.2f} "
            f"| Mean Acc = {s_row['mean_accuracy']*100:5.1f}% | W&B = {run_urls[0]}",
            flush=True,
        )

    _write_csv(results_dir / "fig6_per_seed.csv", list(per_seed_rows[0].keys()), per_seed_rows)
    _write_csv(results_dir / "fig6_summary.csv", list(summary_rows[0].keys()), summary_rows)

    # --------------------------------------------------------------------------
    # 2. Quét điểm ngọt tốc độ học ban đầu (LR Sweet-Spot Sweep: 0.001 -> 0.1) + NonConvexSchedule
    # --------------------------------------------------------------------------
    lr_sweep_list = [float(x) for x in cfg["lr_sweep"]]
    sweep_rows: List[Dict[str, Any]] = []
    for lr_val in lr_sweep_list:
        for prefix in ("adam", "adam-restart"):
            m_name = f"{prefix}:{lr_val:g}"
            regs = []
            accs = []
            for seed in seeds:
                h = run_cytometry_online_experiment(stream, m_name, seed=seed, l2_reg=l2_reg, restart_rule="paper")
                regs.append(float(h.cum_regret[-1]))
                accs.append(float(np.mean(h.accuracy_history)))
            sweep_rows.append(
                {
                    "optimizer_family": prefix,
                    "initial_lr": lr_val,
                    "method": m_name,
                    "final_regret_mean": float(np.mean(regs)),
                    "final_regret_std": float(np.std(regs, ddof=1)) if len(regs) > 1 else 0.0,
                    "mean_accuracy": float(np.mean(accs)),
                }
            )
    # Bổ sung thêm lịch trình lý thuyết NonConvexSchedule (Eq. 25)
    nc_regs, nc_accs = [], []
    for seed in seeds:
        h_nc = run_cytometry_online_experiment(stream, "nonconvex-sgd:0.05", seed=seed, l2_reg=l2_reg)
        nc_regs.append(float(h_nc.cum_regret[-1]))
        nc_accs.append(float(np.mean(h_nc.accuracy_history)))
    sweep_rows.append(
        {
            "optimizer_family": "nonconvex-schedule-eq25",
            "initial_lr": 0.05,
            "method": "nonconvex-sgd:0.05",
            "final_regret_mean": float(np.mean(nc_regs)),
            "final_regret_std": float(np.std(nc_regs, ddof=1)) if len(nc_regs) > 1 else 0.0,
            "mean_accuracy": float(np.mean(nc_accs)),
        }
    )
    _write_csv(results_dir / "fig6_lr_sweep.csv", list(sweep_rows[0].keys()), sweep_rows)

    # --------------------------------------------------------------------------
    # 3. [Task T18] Đối chiếu Điều kiện Restart trong Bài báo vs. Lỗi dòng 144 trong `src/cytometry/main.py`
    # --------------------------------------------------------------------------
    t18_rows: List[Dict[str, Any]] = []
    for m_name in ("adam-restart:0.01", "adam-restart:0.03"):
        for rule_name, rule_desc in (
            ("paper", "Paper Spec: Restart on shift boundaries t in {100, 200, 400} (dist[p] != dist[p-1])"),
            ("author_code", "Author main.py Line 144: Restart when dist[p] == dist[p-1] (t in {300, 500})"),
        ):
            regs = []
            for seed in seeds:
                h_r = run_cytometry_online_experiment(
                    stream, m_name, seed=seed, l2_reg=l2_reg, restart_rule=rule_name
                )
                regs.append(float(h_r.cum_regret[-1]))
            t18_rows.append(
                {
                    "method": m_name,
                    "restart_rule": rule_name,
                    "description": rule_desc,
                    "final_regret_mean": float(np.mean(regs)),
                    "final_regret_std": float(np.std(regs, ddof=1)) if len(regs) > 1 else 0.0,
                }
            )
    _write_csv(results_dir / "t18_author_vs_fixed_restart.csv", list(t18_rows[0].keys()), t18_rows)

    # --------------------------------------------------------------------------
    # 4. Vẽ các hình Figure 6 (left), Figure 6 (right) và Biểu đồ T18
    # --------------------------------------------------------------------------
    steps = np.arange(1, stream.total_steps + 1)
    style_map = {
        "adam:0.01": ("#1f77b4", "-", "adam:0.01"),
        "adam-restart:0.01": ("#ff7f0e", "-", "adam-restart:0.01"),
        "adam:0.03": ("#2ca02c", "-", "adam:0.03"),
        "adam-restart:0.03": ("#d62728", "-", "adam-restart:0.03"),
    }

    # Hình 4a: Standalone Figure 6 (left) cho Slide 27
    fig_l, ax_l = plt.subplots(figsize=(6.8, 5.0), dpi=200)
    plot_figure6_left_umap(dataset, ax_l)
    fig_l.tight_layout()
    fig_l.savefig(figures_dir / "fig6_left_cytometry_embedding.png", bbox_inches="tight")
    plt.close(fig_l)

    # Hình 4b: Standalone Figure 6 (right) cho Slide 28
    fig_r, ax_r = plt.subplots(figsize=(7.0, 5.0), dpi=200)
    for m_str in methods:
        col, ls, lbl = style_map[m_str]
        c_mat = curves_by_method[m_str]
        mean_c = np.mean(c_mat, axis=0)
        std_c = np.std(c_mat, axis=0, ddof=1) if c_mat.shape[0] > 1 else np.zeros_like(mean_c)
        ax_r.plot(steps, mean_c, color=col, linestyle=ls, lw=2.0, label=f"{lbl} ({mean_c[-1]:.1f} ± {std_c[-1]:.1f})")
        ax_r.fill_between(steps, mean_c - std_c, mean_c + std_c, color=col, alpha=0.14)

    for shift_t in (100, 200, 400):
        ax_r.axvline(shift_t, color="#64748B", linestyle="--", lw=1.2, alpha=0.8)
    ax_r.text(50, ax_r.get_ylim()[1] * 0.92, "$P_1$\n(Ductal)", ha="center", fontsize=8.5, color="#475569")
    ax_r.text(150, ax_r.get_ylim()[1] * 0.92, "$P_2$\n(+Delta)", ha="center", fontsize=8.5, color="#475569")
    ax_r.text(300, ax_r.get_ylim()[1] * 0.92, "$P_3$\n(+Alpha)", ha="center", fontsize=8.5, color="#475569")
    ax_r.text(500, ax_r.get_ylim()[1] * 0.92, "$P_4$\n(Secreting)", ha="center", fontsize=8.5, color="#475569")
    ax_r.set_title("Figure 6 (right): Flow Cytometry MLP Cumulative Regret ($T=600, B_t=64$)", fontsize=11, fontweight="bold")
    ax_r.set_xlabel("Online Gradient Step $t$")
    ax_r.set_ylabel("Cumulative Regret")
    ax_r.grid(True, alpha=0.3)
    ax_r.legend(loc="upper left", fontsize=9)
    fig_r.tight_layout()
    fig_r.savefig(figures_dir / "fig6_right_adam_restart_regret.png", bbox_inches="tight")
    plt.close(fig_r)

    # Hình 4c: Ghép đôi Figure 6 (left + right) giống trang 23 của bài báo
    fig_both, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0), dpi=200)
    plot_figure6_left_umap(dataset, ax1)
    for m_str in methods:
        col, ls, lbl = style_map[m_str]
        c_mat = curves_by_method[m_str]
        mean_c = np.mean(c_mat, axis=0)
        std_c = np.std(c_mat, axis=0, ddof=1) if c_mat.shape[0] > 1 else np.zeros_like(mean_c)
        ax2.plot(steps, mean_c, color=col, linestyle=ls, lw=2.0, label=lbl)
        ax2.fill_between(steps, mean_c - std_c, mean_c + std_c, color=col, alpha=0.14)
    for shift_t in (100, 200, 400):
        ax2.axvline(shift_t, color="#64748B", linestyle="--", lw=1.2, alpha=0.8)
    ax2.set_title("Figure 6 (right): Cumulative Regret with Optional Adam Restarts", fontsize=11, fontweight="bold")
    ax2.set_xlabel("step")
    ax2.set_ylabel("regret")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", fontsize=9)
    fig_both.tight_layout()
    fig_both.savefig(figures_dir / "fig6_cytometry_2d_and_regret.png", bbox_inches="tight")
    plt.close(fig_both)

    # Hình 4d: Biểu đồ Quét Sweet-Spot LR & Phân tích lỗi dòng 144 code tác giả (Task T18)
    fig_s, (ax_s1, ax_s2) = plt.subplots(1, 2, figsize=(13.0, 4.6), dpi=200)
    fixed_vals = [next(r["final_regret_mean"] for r in sweep_rows if r["method"] == f"adam:{lr:g}") for lr in lr_sweep_list]
    rest_vals = [next(r["final_regret_mean"] for r in sweep_rows if r["method"] == f"adam-restart:{lr:g}") for lr in lr_sweep_list]
    x_idx = np.arange(len(lr_sweep_list))
    ax_s1.plot(x_idx, fixed_vals, "o-", color="#1f77b4", lw=2.0, label="Standard Adam (No Restart)")
    ax_s1.plot(x_idx, rest_vals, "s--", color="#ff7f0e", lw=2.0, label="Adam Restart ($t \\in \\{100, 200, 400\\}$)")
    ax_s1.set_xticks(x_idx)
    ax_s1.set_xticklabels([f"$\\eta_0={lr:g}$" for lr in lr_sweep_list])
    ax_s1.set_ylabel("Final Cumulative Regret ($T=600$)")
    ax_s1.set_title("Initial Learning Rate Sweet Spot ($\\eta_0 = 0.01$ Optimal)", fontweight="bold")
    ax_s1.grid(True, alpha=0.3)
    ax_s1.legend(fontsize=9)

    t18_labels = ["adam-restart:0.01", "adam-restart:0.03"]
    paper_rule_vals = [next(r["final_regret_mean"] for r in t18_rows if r["method"] == m and r["restart_rule"] == "paper") for m in t18_labels]
    bug_rule_vals = [next(r["final_regret_mean"] for r in t18_rows if r["method"] == m and r["restart_rule"] == "author_code") for m in t18_labels]
    x_t18 = np.arange(len(t18_labels))
    w = 0.32
    ax_s2.bar(x_t18 - w / 2, paper_rule_vals, width=w, color="#16a34a", label="Fixed Shift-Boundary Restart ($t \\in \\{100,200,400\\}$)")
    ax_s2.bar(x_t18 + w / 2, bug_rule_vals, width=w, color="#dc2626", label="Author main.py Line 144 (`==` bug: $t \\in \\{300,500\\}$)")
    ax_s2.set_xticks(x_t18)
    ax_s2.set_xticklabels(t18_labels)
    ax_s2.set_ylabel("Final Cumulative Regret ($T=600$)")
    ax_s2.set_title("T18 Source Audit: Shift-Boundary Restart vs. Author Line 144 Bug", fontweight="bold")
    ax_s2.grid(True, axis="y", alpha=0.3)
    ax_s2.legend(fontsize=8.5)
    fig_s.tight_layout()
    fig_s.savefig(figures_dir / "fig6_lr_sweep_and_author_bug.png", bbox_inches="tight")
    plt.close(fig_s)

    elapsed = time.time() - t0
    print("=" * 80, flush=True)
    print(f"[OK] Flow Cytometry MLP reproduction completed in {elapsed:.2f}s.", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
