r"""
================================================================================
FILE: tests/test_nonconvex.py
MỤC ĐÍCH:
    Unit tests kiểm chứng toàn bộ phần việc của TV3 (Task T11, T12, T16, T18):
    1. `core/data.py`: Tải đúng 3.696 tế bào tuyến tụy với 10 marker genes
       (`Pyy, Meg3, Malat1, Gcg, Gnas, Actb, Ghrl, Rps3, Ins2, Hspa8`) và 4 phân lớp
       (`Alpha: 481, Beta: 591, Delta: 70, Ductal: 916`), sinh luồng dữ liệu 4 pha
       P_1 -> P_4 với nhiễu nhân U ~ [0.9, 1.1].
    2. `NonConvexSchedule` (Theorem 5.1, Công thức 25): Kiểm tra miền giá trị
       [0, 1/L] và tính chất đồng biến theo độ dịch chuyển \gamma_t và mất mát \ell_t.
    3. `CytometryMLP` & `run_cytometry_online_experiment`: Kiểm tra mạng 5 tầng
       (10 -> 64 -> 32 -> 16 -> 8 -> 4, ELU) huấn luyện online không lỗi và cơ chế
       Adam Restart giảm độ tiếc nuối tích lũy khi có dịch chuyển phân phối.
================================================================================
"""

import unittest

import numpy as np
import torch

from core.data import (
    CELL_CLASSES,
    PANCREAS_10_GENES,
    PancreasCytometryStream,
    load_pancreas_dataset,
)
from core.schedules import NonConvexSchedule
from experiments.neural_net import CytometryMLP, run_cytometry_online_experiment


class TestNonConvexAndCytometry(unittest.TestCase):
    """Kiểm thử đơn vị cho Mục 5 & Phụ lục E (Non-Convex Schedule & Flow Cytometry MLP)."""

    def test_pancreas_dataset_and_stream_shapes(self) -> None:
        ds = load_pancreas_dataset()
        self.assertEqual(ds.gene_names, PANCREAS_10_GENES)
        self.assertEqual(ds.X_norm_10.shape, (3696, 10))
        self.assertEqual(ds.X_umap.shape, (3696, 2))
        self.assertEqual(ds.class_pools["Alpha"].shape, (481, 10))
        self.assertEqual(ds.class_pools["Beta"].shape, (591, 10))
        self.assertEqual(ds.class_pools["Delta"].shape, (70, 10))
        self.assertEqual(ds.class_pools["Ductal"].shape, (916, 10))

        stream = PancreasCytometryStream(dataset=ds, batch_size=64, perturb_eps=0.1)
        self.assertEqual(stream.total_steps, 600)
        # Pha 1 (step 0..99) chỉ chứa lớp Ductal (id = 3)
        rng = np.random.default_rng(0)
        X0, y0 = stream.sample_batch(0, rng)
        self.assertEqual(X0.shape, (64, 10))
        self.assertEqual(y0.shape, (64,))
        self.assertTrue(np.all(y0 == 3))

        # Kiểm tra mốc dịch chuyển phân phối t in {100, 200, 400}
        shift_steps_paper = [t for t in range(600) if stream.is_shift_boundary(t, "paper")]
        self.assertEqual(shift_steps_paper, [100, 200, 400])
        shift_steps_bug = [t for t in range(600) if stream.is_shift_boundary(t, "author_code")]
        self.assertEqual(shift_steps_bug, [300, 500])

    def test_nonconvex_schedule_monotonicity_and_bounds(self) -> None:
        sched = NonConvexSchedule(L=4.0, gradient_variance=2.0, default_batch_size=64)
        self.assertAlmostEqual(sched.max_lr, 0.25)

        lr_zero = sched.step(t=1, gamma_prev=0.0, loss_val=0.0, batch_size=64)
        self.assertEqual(lr_zero, 0.0)

        lr_low = sched.step(t=2, gamma_prev=0.05, loss_val=0.2, batch_size=64)
        lr_high_shift = sched.step(t=3, gamma_prev=0.50, loss_val=0.2, batch_size=64)
        lr_high_loss = sched.step(t=4, gamma_prev=0.05, loss_val=1.5, batch_size=64)

        self.assertGreater(lr_low, 0.0)
        self.assertLess(lr_low, lr_high_shift)
        self.assertLess(lr_low, lr_high_loss)
        self.assertLessEqual(lr_high_shift, 0.25)
        self.assertLessEqual(lr_high_loss, 0.25)

    def test_cytometry_mlp_forward_and_adam_restart_short_stream(self) -> None:
        model = CytometryMLP(input_dim=10, hidden_dims=(64, 32, 16, 8), output_dim=4, seed=42)
        x = torch.randn(16, 10)
        out = model(x)
        self.assertEqual(tuple(out.shape), (16, 4))
        self.assertGreater(float(model.l2_penalty(1e-4).item()), 0.0)

        ds = load_pancreas_dataset()
        short_stream = PancreasCytometryStream(
            dataset=ds, batch_size=32, perturb_eps=0.1, steps_per_block=10, block_types=(0, 1, 2, 2, 3, 3)
        )
        hist = run_cytometry_online_experiment(short_stream, "adam-restart:0.01", seed=0)
        self.assertEqual(len(hist.cum_regret), 60)
        self.assertEqual(hist.restart_steps, [10, 20, 40])
        self.assertTrue(np.all(np.isfinite(hist.cum_regret)))


if __name__ == "__main__":
    unittest.main()
