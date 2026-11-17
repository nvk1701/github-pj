r"""
================================================================================
FILE: experiments/sensitivity.py
MỤC ĐÍCH:
    Script phân tích độ nhạy mở rộng (Bonus Sensitivity Analysis - Task T17 của TV2
    & Kiểm tra chéo Task T19 của TV1):
    - Khảo sát độ bền của lịch trình tốc độ học khi thay đổi:
      1. Bước rời rạc hóa / tốc độ trôi phân phối: \ell \in {1, 8, 16} (với d = 128).
      2. Độ lệch chuẩn nhiễu nhãn: \sigma \in {0.0, \sqrt{0.1}, 1.0} (với d = 128).
    - Xuất biểu đồ tổng hợp `figures/sensitivity_drift_and_noise.png` và các biểu đồ
      chi tiết trong `docs/results/cp04-cp05/figures/cp05_*.png`, cùng bảng CSV
      trong `results/sensitivity/`.
================================================================================
"""

import csv
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def generate_sensitivity_figures_and_tables() -> None:
    """Đọc dữ liệu đã xác minh từ `docs/results/cp04-cp05/data/cp05_*.csv` và vẽ biểu đồ độ nhạy."""
    cp05_data_dir = PROJECT_ROOT / "docs" / "results" / "cp04-cp05" / "data"
    cp05_fig_dir = PROJECT_ROOT / "docs" / "results" / "cp04-cp05" / "figures"
    figures_dir = PROJECT_ROOT / "figures"
    results_dir = PROJECT_ROOT / "results" / "sensitivity"

    cp05_fig_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    cond_csv = cp05_data_dir / "cp05_condition_summary.csv"
    per_seed_csv = cp05_data_dir / "cp05_per_seed.csv"
    d512_csv = cp05_data_dir / "d512_diagnostic_summary.csv"

    for src in (cond_csv, per_seed_csv, d512_csv):
        if src.exists():
            shutil.copyfile(src, results_dir / src.name)

    with open(cond_csv, "r", encoding="utf-8") as f:
        cond_rows = {r["condition"]: r for r in csv.DictReader(f)}

    # 1. Vẽ biểu đồ độ nhạy theo tốc độ trôi (\ell \in {1, 8, 16})
    drift_order = [
        ("ell16", "$\\ell = 16$\n(Infrequent shifts\n$\\sum \\gamma_t = 339.3$)"),
        ("baseline", "$\\ell = 8$ (Baseline)\n(Moderate shifts\n$\\sum \\gamma_t = 688.8$)"),
        ("ell1", "$\\ell = 1$\n(Continuous shifts\n$\\sum \\gamma_t = 1848.5$)"),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=200)
    x_drift = np.arange(len(drift_order))
    b_drift = [float(cond_rows[k]["mean_B"]) for k, _ in drift_order]
    sd_drift = [float(cond_rows[k]["sample_SD_B"]) for k, _ in drift_order]
    bars = ax.bar(x_drift, b_drift, yerr=sd_drift, capsize=5, width=0.52, color=["#2ca02c", "#1f77b4", "#d62728"], alpha=0.88)
    for bar, val, sd in zip(bars, b_drift, sd_drift):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val + sd + 6.0, f"{val:.1f} ± {sd:.2f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_xticks(x_drift)
    ax.set_xticklabels([lbl for _, lbl in drift_order], fontsize=9.5)
    ax.set_ylim(0, max(b_drift) * 1.18)
    ax.set_ylabel("Final Cumulative Population Excess Risk $B$ ($d=128$)")
    ax.set_title("CP05 Sensitivity to Distribution Shift Frequency ($d=128, T=2000$)", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(cp05_fig_dir / "cp05_drift_sensitivity.png", bbox_inches="tight")
    fig.savefig(figures_dir / "sensitivity_drift.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Vẽ biểu đồ độ nhạy theo phương sai nhiễu nhãn (\sigma \in {0, \sqrt{0.1}, 1.0})
    noise_order = [
        ("noise0", "$\\sigma = 0.0$\n(Noiseless)"),
        ("baseline", "$\\sigma = \\sqrt{0.1} \\approx 0.316$\n(Baseline)"),
        ("noise1", "$\\sigma = 1.0$\n(High Noise)"),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=200)
    x_noise = np.arange(len(noise_order))
    b_noise = [float(cond_rows[k]["mean_B"]) for k, _ in noise_order]
    sd_noise = [float(cond_rows[k]["sample_SD_B"]) for k, _ in noise_order]
    bars = ax.bar(x_noise, b_noise, yerr=sd_noise, capsize=5, width=0.52, color=["#17becf", "#1f77b4", "#ff7f0e"], alpha=0.88)
    for bar, val, sd in zip(bars, b_noise, sd_noise):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val + sd + 5.0, f"{val:.1f} ± {sd:.2f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_xticks(x_noise)
    ax.set_xticklabels([lbl for _, lbl in noise_order], fontsize=9.5)
    ax.set_ylim(0, max(b_noise) * 1.18)
    ax.set_ylabel("Final Cumulative Population Excess Risk $B$ ($d=128$)")
    ax.set_title("CP05 Sensitivity to Label Noise Standard Deviation ($d=128, T=2000$)", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(cp05_fig_dir / "cp05_noise_sensitivity.png", bbox_inches="tight")
    fig.savefig(figures_dir / "sensitivity_noise.png", bbox_inches="tight")
    plt.close(fig)

    # 3. Vẽ biểu đồ chẩn đoán d=512 từ `d512_diagnostic_summary.csv`
    if d512_csv.exists():
        with open(d512_csv, "r", encoding="utf-8") as f:
            d512_rows = list(csv.DictReader(f))
        fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=200)
        labels = [r.get("method", r.get("condition", f"row-{i}")) for i, r in enumerate(d512_rows)]
        vals = [
            float(r.get("mean_final_population_B", r.get("final_population_B", r.get("mean_B", 0.0))))
            for r in d512_rows
        ]
        ax.bar(np.arange(len(labels)), vals, color="#9467bd", alpha=0.85)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8.5)
        ax.set_ylabel("Cumulative Excess Risk $B$")
        ax.set_title("d=512 High-Dimensional Regime Diagnostic Summary", fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(cp05_fig_dir / "cp05_d512_loss_dynamics.png", bbox_inches="tight")
        plt.close(fig)

    print(f"[OK] Sensitivity figures and CSVs saved to {figures_dir} and {results_dir}.")


if __name__ == "__main__":
    generate_sensitivity_figures_and_tables()
