r"""
================================================================================
FILE: tests/test_convex.py
MỤC ĐÍCH:
    Bộ kiểm thử đơn vị và kiểm chứng chéo độc lập (Task T9 & Task T19) cho:
    1. `ConvexSchedule` (Proposition 4.3, Công thức 18-20) trong `core/schedules.py`.
    2. `core/convex_eval.py` (hàm mất mát BCE ổn định số học & đánh giá Monte Carlo).
    3. `core/convex_population.py` (tích phân cầu phương Gaussian 1D & nghiệm Oracle).
    4. Đối chiếu tái lập số học 100% (Task T19 - TV1 kiểm tra chéo TV2) giữa vòng lặp
       `OnlineSGDTrainer(loss_type="logistic")` trực tiếp trên máy của TV1 và bảng
       kết quả đóng băng `docs/results/cp04-cp05/data/cp04_per_seed.csv` của TV2.
================================================================================
"""

import csv
import math
from pathlib import Path
import unittest

import numpy as np

from core.convex_eval import evaluate_history, logistic_loss_gradient, sample_logistic_batch
from core.convex_population import (
    PopulationEvaluator,
    gaussian_population_loss_gradient,
    gaussian_population_oracle,
)
from core.drift import generate_spiral_trajectory
from core.schedules import ConstantSchedule, ConvexSchedule, InverseSqrtSchedule
from core.trainer import OnlineSGDTrainer
from core.utils import set_seed

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestConvexScheduleAndCrossCheck(unittest.TestCase):
    """Kiểm thử `ConvexSchedule` (Prop. 4.3) và kiểm tra chéo T19 với số liệu CP04 của TV2."""

    def test_convex_schedule_thresholds_and_clipping(self) -> None:
        """Kiểm tra 2 ngưỡng tau_1 <= tau_2 <= 1/L và tính chất tăng theo độ trôi gamma."""
        sched = ConvexSchedule(L=0.25, D_max=8.0, gradient_variance=2.0, eta_init=1.0)
        t1_zero, t2_zero = sched.thresholds(gamma=0.0, batch_size=256)
        self.assertEqual(t1_zero, 0.0)
        self.assertGreater(t2_zero, 0.0)
        self.assertLessEqual(t2_zero, 4.0)

        t1_low, t2_low = sched.thresholds(gamma=0.1, batch_size=256)
        t1_high, t2_high = sched.thresholds(gamma=1.0, batch_size=256)
        self.assertLess(t1_low, t1_high)
        self.assertLess(t2_low, t2_high)
        self.assertLessEqual(t1_high, t2_high)
        self.assertLessEqual(t2_high, 4.0)

        # Kiểm tra bước cập nhật liên tiếp và reset
        lr1 = sched.step(t=1, gamma_prev=0.0, batch_size=256)
        self.assertEqual(lr1, 1.0)  # Vì tau_1 = 0 <= 1.0 <= tau_2 nên giữ nguyên eta_init = 1.0
        sched.reset()
        lr1_again = sched.step(t=1, gamma_prev=0.0, batch_size=256)
        self.assertEqual(lr1, lr1_again)

    def test_gaussian_population_oracle_shrinkage(self) -> None:
        """Khi có nhiễu nhãn s > 0, nghiệm tối ưu quần thể co nhẹ về gốc tọa độ so với reference."""
        ref = np.array([1.0, -0.5])
        oracle = gaussian_population_oracle(ref, label_noise_std=math.sqrt(0.1))
        self.assertLess(oracle["radial_norm"], float(np.linalg.norm(ref)))
        self.assertGreater(oracle["radial_norm"], 0.0)
        self.assertLess(oracle["gradient_residual"], 1e-10)

        _, grad_at_oracle, _ = gaussian_population_loss_gradient(
            oracle["theta"], ref, label_noise_std=math.sqrt(0.1)
        )
        self.assertLess(float(np.linalg.norm(grad_at_oracle)), 1e-9)

    def test_t19_cross_check_reproduces_tv2_cp04_d2_seed0_exactly(self) -> None:
        """[Task T19] Chạy lại độc lập d=2, seed=0 cho cả 3 phương pháp và đối chiếu khớp tuyệt đối với CSV của TV2."""
        cp04_csv = PROJECT_ROOT / "docs" / "results" / "cp04-cp05" / "data" / "cp04_per_seed.csv"
        with open(cp04_csv, "r", encoding="utf-8") as f:
            rows = {r["key"]: r for r in csv.DictReader(f)}

        T = 2000
        d = 2
        B = 256
        noise_std = math.sqrt(0.1)
        theta_star, gamma_seq = generate_spiral_trajectory(T=T, d=d, a=1.0, b=-1.0, k=4.0, ell=8)
        pop_eval = PopulationEvaluator(theta_star, label_noise_std=noise_std, tolerance=1e-11)

        method_configs = [
            (
                "convex",
                ConvexSchedule(L=0.25, D_max=float(d), gradient_variance=0.25 * d, eta_init=1.0),
            ),
            ("constant", ConstantSchedule(lr=0.1)),
            ("inverse_sqrt", InverseSqrtSchedule(lr0=0.1)),
        ]

        for m_name, sched in method_configs:
            set_seed(0)
            trainer = OnlineSGDTrainer(
                d=d,
                schedule=sched,
                loss_type="logistic",
                batch_size=B,
                noise_std=noise_std,
                max_norm=0.5 * d,
                use_oracle_gamma=False,
                ema_beta=0.9,
                use_wandb=False,
            )
            hist = trainer.run(theta_star_path=theta_star, gamma_seq=gamma_seq, seed=0)
            pop_res = pop_eval.evaluate(hist.theta_path)
            mc_res = evaluate_history(
                hist,
                batch_size=B,
                label_noise_std=noise_std,
                eval_batch_size=512,
                eval_seed=1000003,
                approx_oracle_path=pop_eval.oracle_path,
            )

            expected = rows[f"convex-cp03-v1-d2-{m_name}-seed0"]
            self.assertAlmostEqual(
                float(pop_res["population_cum_A"][-1]),
                float(expected["final_population_A"]),
                places=11,
            )
            self.assertAlmostEqual(
                float(pop_res["population_cum_B"][-1]),
                float(expected["final_population_B"]),
                places=11,
            )
            self.assertAlmostEqual(
                float( np.max(np.linalg.norm(hist.theta_path, axis=1)) ),
                float(expected["theta_norm_max"]),
                places=12,
            )
            self.assertTrue(np.isfinite(mc_res["eval_cum_mc_excess"][-1]))


if __name__ == "__main__":
    unittest.main()
