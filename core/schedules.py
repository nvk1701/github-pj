"""
================================================================================
FILE: core/schedules.py
MỤC ĐÍCH:
    Định nghĩa giao diện chuẩn (`BaseSchedule`) và các lịch trình tốc độ học cơ sở
    (Baselines) cho thuật toán Online Mini-batch SGD (Task T6):
    1. `BaseSchedule`: Lớp cơ sở trừu tượng quy định phương thức `.step(t, **kwargs)`
       và `.reset()`. Mọi lịch trình mới của nhóm đều kế thừa từ lớp này.
    2. `ConstantSchedule`: Giữ tốc độ học cố định \\eta_t = \\eta_0 (dùng để tái hiện
       Figure 1 và làm baseline đối chứng).
    3. `InverseTimeSchedule`: Lịch trình giảm dần cổ điển Robbins-Monro O(1/t):
       \\eta_t = \\eta_0 / (1 + decay * (t - 1)).
    4. `InverseSqrtSchedule`: Lịch trình giảm dần chuẩn OCO O(1/sqrt(t)):
       \\eta_t = \\eta_0 / sqrt(t).

CÁCH MỞ RỘNG CHO CÁC TASK TIẾP THEO (T7, T9, T12):
    - TV1 (Khoa - Task T7): Cài đặt thêm `LinearOptimalSchedule(BaseSchedule)`
      theo Algorithm 1 (trang 6) vào file này.
    - TV2 (Khôi - Task T9): Cài đặt thêm `ConvexSchedule(BaseSchedule)` theo
      Proposition 4.3 (Công thức 18-20, trang 7) vào file này.
    - TV3 (Hòa - Task T12): Cài đặt thêm `NonConvexSchedule(BaseSchedule)` theo
      Theorem 5.1 (Công thức 25, trang 8) vào file này.

LƯU Ý TƯƠNG THÍCH VỚI `core/tracking.py` (T2 của Khôi):
    Thuộc tính `schedule.name` được dùng làm trường `"method"` khi gọi `init_run(config)`
    để tự sinh tên run W&B (`{experiment}-{method}-seed{seed}`). Vì vậy, `name`
    được đặt bằng dấu gạch ngang (ví dụ: `"constant-0.05"`, `"inv-time-0.2"`)
    không chứa khoảng trắng hay ký tự đặc biệt `/\\#?%:`.
================================================================================
"""

from abc import ABC, abstractmethod
import math
from typing import Any


class BaseSchedule(ABC):
    """Lớp cơ sở trừu tượng (Abstract Base Class) cho mọi lịch trình tốc độ học (Learning Rate Schedule)."""

    def __init__(self, name: str = "base-schedule") -> None:
        """Khởi tạo lịch trình với tên định danh `name` tương thích với W&B.

        Parameters
        ----------
        name : str, optional
            Tên định danh của lịch trình (ví dụ: `"constant-0.05"`, `"linear-optimal"`).
        """
        self.name = name

    @abstractmethod
    def step(self, t: int, **kwargs: Any) -> float:
        """Tính và trả về tốc độ học \\eta_t tại bước thời gian rời rạc t >= 1.

        Parameters
        ----------
        t : int
            Chỉ số bước thời gian hiện tại (đánh số từ 1: t = 1, 2, ..., T).
        **kwargs : Any
            Các tín hiệu trạng thái truyền từ `OnlineSGDTrainer` tại bước t, bao gồm:
            - `gamma_prev` (float): Độ dịch chuyển phân phối \\gamma_{t-1} (hoặc EMA \\hat{\\gamma}_t).
            - `batch_size` (int): Kích thước mini-batch B_t.
            - `d` (int): Số chiều không gian tham số.
            - `noise_std` (float): Độ lệch chuẩn nhiễu \\sigma.
            - `loss_val` (float): Giá trị hàm mất mát trung bình trên mini-batch hiện tại \\ell^B_t(\\theta_t).

        Returns
        -------
        float
            Giá trị bước nhảy \\eta_t >= 0 dùng để cập nhật SGD tại bước t.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Đặt lại các biến trạng thái nội tại của lịch trình trước khi chạy một seed mới.

        Ví dụ:
        - Với `LinearOptimalSchedule` (Alg. 1): đặt lại biến trạng thái ODE `v = 0`.
        - Với `ConvexSchedule` (Prop. 4.3): đặt lại `eta_prev` về giá trị ban đầu.
        """
        pass


class ConstantSchedule(BaseSchedule):
    """Lịch trình tốc độ học hằng số: \\eta_t = \\eta_0 với mọi t >= 1."""

    def __init__(self, lr: float = 0.01) -> None:
        """Khởi tạo lịch trình tốc độ học cố định.

        Parameters
        ----------
        lr : float, optional
            Giá trị tốc độ học cố định \\eta_0 >= 0 (mặc định: 0.01).
        """
        super().__init__(name=f"constant-{lr:g}")
        if lr < 0.0:
            raise ValueError(f"Learning rate phải không âm, nhận được {lr}.")
        self.lr = float(lr)

    def step(self, t: int, **kwargs: Any) -> float:
        """Trả về tốc độ học không đổi \\eta_0."""
        return self.lr


class InverseTimeSchedule(BaseSchedule):
    """Lịch trình suy giảm Robbins-Monro bậc O(1/t): \\eta_t = \\eta_0 / (1 + decay * (t - 1)).

    Theo Lemma 3.6 (Mục 3.1, trang 6): Khi không có dịch chuyển phân phối (\\gamma_t = 0),
    lịch trình tối ưu của Algorithm 1 cũng suy giảm tiệm cận theo quy luật O(1/t) này.
    Tuy nhiên, khi có dịch chuyển phân phối ở giai đoạn muộn, lịch trình 1/t thất bại
    vì \\eta_t đã tiến về 0 khiến mô hình không thể bám đuổi \\theta^*_t mới.
    """

    def __init__(self, lr0: float = 0.1, decay: float = 0.05) -> None:
        """Khởi tạo lịch trình giảm dần 1/t.

        Parameters
        ----------
        lr0 : float, optional
            Tốc độ học ban đầu \\eta_0 tại bước t = 1 (mặc định: 0.1).
        decay : float, optional
            Hệ số suy giảm ở mẫu số (mặc định: 0.05).
        """
        super().__init__(name=f"inv-time-{lr0:g}")
        if lr0 < 0.0 or decay < 0.0:
            raise ValueError("lr0 và decay phải là số không âm.")
        self.lr0 = float(lr0)
        self.decay = float(decay)

    def step(self, t: int, **kwargs: Any) -> float:
        """Tính \\eta_t = \\eta_0 / (1 + decay * (t - 1)) tại bước t >= 1."""
        t_eff = max(1, int(t))
        return self.lr0 / (1.0 + self.decay * (t_eff - 1))


class InverseSqrtSchedule(BaseSchedule):
    """Lịch trình suy giảm bậc O(1/sqrt(t)) trong Tối ưu Lồi Trực tuyến: \\eta_t = \\eta_0 / sqrt(t)."""

    def __init__(self, lr0: float = 0.1) -> None:
        """Khởi tạo lịch trình giảm dần 1/sqrt(t).

        Parameters
        ----------
        lr0 : float, optional
            Tốc độ học ban đầu \\eta_0 tại bước t = 1 (mặc định: 0.1).
        """
        super().__init__(name=f"inv-sqrt-{lr0:g}")
        if lr0 < 0.0:
            raise ValueError("lr0 phải là số không âm.")
        self.lr0 = float(lr0)

    def step(self, t: int, **kwargs: Any) -> float:
        """Tính \\eta_t = \\eta_0 / sqrt(t) tại bước t >= 1."""
        t_eff = max(1, int(t))
        return self.lr0 / math.sqrt(float(t_eff))
