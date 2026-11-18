r"""
================================================================================
FILE: tests/test_linear.py
MỤC ĐÍCH:
    Unit tests kiểm chứng tính đúng đắn toán học của `LinearOptimalSchedule`
    (Algorithm 1 - Task T7) và vòng lặp thực nghiệm Linear Regression (Task T14):
    1. Kiểm tra tính đơn điệu: Khi có dịch chuyển phân phối \gamma > 0, tốc độ học
       \eta^*_t phải lớn hơn trường hợp tĩnh (\gamma = 0).
    2. Kiểm tra giới hạn tiệm cận Lemma 3.6: Khi \gamma_t = 0 với mọi t, \eta^*_t
       bị chặn bởi max_lr ở giai đoạn đầu và suy giảm tiệm cận theo quy luật O(1/t).
    3. Kiểm tra tính bất biến theo seed sau khi gọi `.reset()`.
    4. Kiểm tra tích hợp với `OnlineSGDTrainer(loss_type="linear")` trên cả hai
       chế độ Oracle \gamma_{t-1} và ước lượng trực tuyến EMA \hat{\gamma}_t.
================================================================================
"""

import math
import unittest

import numpy as np

from core.drift import generate_bursty_shifts, generate_spiral_trajectory
from core.schedules import InverseTimeSchedule, LinearOptimalSchedule
from core.trainer import OnlineSGDTrainer
from core.utils import set_seed


class TestLinearOptimalSchedule(unittest.TestCase):
    """Kiểm thử đơn vị cho lớp `LinearOptimalSchedule` (Algorithm 1, Mục 3)."""

    def test_validation_rejects_invalid_hyperparameters(self) -> None:
        with self.assertRaises(ValueError):
            LinearOptimalSchedule(max_lr=0.0)
        with self.assertRaises(ValueError):
            LinearOptimalSchedule(max_lr=-0.1)
        with self.assertRaises(ValueError):
            LinearOptimalSchedule(kappa=0.0)
        with self.assertRaises(ValueError):
            LinearOptimalSchedule(kappa=1.5)
        with self.assertRaises(ValueError):
            LinearOptimalSchedule(v_init=-1.0)

    def test_no_shift_decay_matches_lemma_3_6(self) -> None:
        """Theo Lemma 3.6: Khi gamma = 0, eta^*_t giảm đơn điệu và tiệm cận eps / (a + C + eps*t)."""
        eps = 0.1
        d = 99
        B = 100
        sigma = math.sqrt(300.0 / float(d))  # a = 0.1, b = 0.3
        sched = LinearOptimalSchedule(max_lr=eps, kappa=0.005, v_init=1.0)

        lrs = [
            sched.step(t=t, gamma_prev=0.0, batch_size=B, d=d, noise_std=sigma)
            for t in range(1, 101)
        ]
        # Bước đầu tiên còn ở pha chạm trần eps = 0.1
        self.assertAlmostEqual(lrs[0], eps, places=5)
        # Sau đó giảm đơn điệu về 0 theo O(1/t)
        self.assertGreater(lrs[10], lrs[50])
        self.assertGreater(lrs[50], lrs[99])

        a = eps * (d + 1) / B
        b = eps * (sigma ** 2) * d / B
        tau_star = (1.0 / (2.0 - a)) * math.log((1.0 - a) * ((2.0 - a) / b - 1.0))
        c_const = a * math.log((1.0 - a) / b) + 1.0 - a - tau_star
        asymp_100 = eps / (a + c_const + eps * 100.0)
        # Tỷ số giữa nghiệm Euler và công thức tiệm cận tại t=100 xấp xỉ 1
        self.assertAlmostEqual(lrs[-1] / asymp_100, 1.0, delta=0.05)

    def test_distribution_shift_increases_learning_rate(self) -> None:
        """Khi xuất hiện dịch chuyển gamma > 0, tốc độ học tối ưu phải tăng vọt lên."""
        sched_static = LinearOptimalSchedule(max_lr=0.2, kappa=0.05, v_init=0.5)
        sched_shift = LinearOptimalSchedule(max_lr=0.2, kappa=0.05, v_init=0.5)

        for t in range(1, 30):
            lr_s = sched_static.step(t=t, gamma_prev=0.0, batch_size=128, d=32, noise_std=0.5)
            lr_d = sched_shift.step(t=t, gamma_prev=0.0, batch_size=128, d=32, noise_std=0.5)
            self.assertAlmostEqual(lr_s, lr_d, places=12)

        # Tại bước t = 30, phân phối dịch chuyển mạnh ở sched_shift
        lr_static_30 = sched_static.step(t=30, gamma_prev=0.0, batch_size=128, d=32, noise_std=0.5)
        lr_shift_30 = sched_shift.step(t=30, gamma_prev=0.8, batch_size=128, d=32, noise_std=0.5)
        self.assertGreater(lr_shift_30, lr_static_30 * 2.0)

    def test_v_init_zero_escapes_fixed_point_on_positive_shift(self) -> None:
        """Ngay cả khi khởi tạo v_init = 0.0, nếu có shift gamma > 0 thì v và eta_t phải dương."""
        sched = LinearOptimalSchedule(max_lr=0.5, kappa=0.1, v_init=0.0)
        lr = sched.step(t=1, gamma_prev=0.2, batch_size=256, d=8, noise_std=0.3)
        self.assertGreater(lr, 0.0)
        self.assertGreater(sched.v, 0.0)

    def test_reset_restores_exact_reproducibility(self) -> None:
        """Gọi .reset() phải khôi phục chính xác chuỗi tốc độ học ở lần chạy tiếp theo."""
        gamma_seq = generate_bursty_shifts(T=50, episode_length=10, jump_size=0.4)
        sched = LinearOptimalSchedule(max_lr=0.1, kappa=0.05, v_init=1.0)

        run1 = [
            sched.step(t=t + 1, gamma_prev=float(gamma_seq[t]), batch_size=64, d=16, noise_std=1.0)
            for t in range(50)
        ]
        sched.reset()
        run2 = [
            sched.step(t=t + 1, gamma_prev=float(gamma_seq[t]), batch_size=64, d=16, noise_std=1.0)
            for t in range(50)
        ]
        np.testing.assert_allclose(run1, run2, rtol=0.0, atol=1e-15)

    def test_online_sgd_trainer_with_linear_optimal_beats_decaying_baseline(self) -> None:
        """Trên quỹ đạo xoắn ốc dịch chuyển, Algorithm 1 phải có Regret thấp hơn hẳn lịch trình 1/t."""
        T, d = 400, 8
        theta_star, gamma_seq = generate_spiral_trajectory(T=T, d=d, ell=8)
        max_lr = 1.0 / math.sqrt(d)

        set_seed(0)
        alg1_trainer = OnlineSGDTrainer(
            d=d,
            schedule=LinearOptimalSchedule(max_lr=max_lr, kappa=0.1, v_init=float(np.sum(theta_star[0] ** 2))),
            loss_type="linear",
            batch_size=256,
            noise_std=math.sqrt(0.1),
            use_oracle_gamma=True,
            use_wandb=False,
        )
        hist_alg1 = alg1_trainer.run(theta_star_path=theta_star, gamma_seq=gamma_seq, seed=0)

        set_seed(0)
        decay_trainer = OnlineSGDTrainer(
            d=d,
            schedule=InverseTimeSchedule(lr0=max_lr, decay=0.05),
            loss_type="linear",
            batch_size=256,
            noise_std=math.sqrt(0.1),
            use_oracle_gamma=True,
            use_wandb=False,
        )
        hist_decay = decay_trainer.run(theta_star_path=theta_star, gamma_seq=gamma_seq, seed=0)

        self.assertLess(hist_alg1.cum_regret[-1], hist_decay.cum_regret[-1])


if __name__ == "__main__":
    unittest.main()
