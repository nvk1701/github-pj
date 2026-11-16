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


class ConvexSchedule(BaseSchedule):
    """Tinh LR theo Prop. 4.3, Eq. (18)-(20). Variance la sigma binh phuong.

    L, D_max va gradient_variance la tham so cua bound, tach khoi nhieu nhan.
    eta_init nam trong (0, 1/L]. reset khoi phuc LR nay cho seed moi.
    """

    def __init__(self, L: float, D_max: float, gradient_variance: float,
                 eta_init: float = 1.0) -> None:
        super().__init__(name="convex")
        self.L = self._number(L, "L", positive=True)
        self.D_max = self._number(D_max, "D_max", positive=True)
        self.gradient_variance = self._number(gradient_variance, "gradient_variance")
        self.max_lr = 1.0 / self.L
        if not math.isfinite(self.max_lr):
            raise ValueError("1/L must be representable")
        self.eta_init = self._number(eta_init, "eta_init", positive=True)
        if self.eta_init > self.max_lr:
            raise ValueError("eta_init must be <= 1/L")
        self.reset()

    @staticmethod
    def _number(value, name, *, positive=False):
        """Doc scalar huu han, tu choi bool va gia tri am."""
        from numbers import Real
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real scalar")
        result = float(value)
        if not math.isfinite(result) or result < 0 or (positive and result == 0):
            raise ValueError(f"{name} is outside its finite domain")
        return result

    @staticmethod
    def _integer(value, name):
        """Chi chap nhan so nguyen duong cho step va batch size."""
        from numbers import Integral
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        return int(value)

    @staticmethod
    def _logadd(a, b):
        """Tinh log(exp(a)+exp(b)) ma khong lam tran so."""
        high, low = max(a, b), min(a, b)
        return high + math.log1p(math.exp(low - high))

    def _tau(self, log_b, batch_size):
        """Rationalize Eq.18: tau=2/[L*(1+sqrt(1+4v/(B*b*L^2)))]."""
        if log_b == -math.inf:
            return 0.0
        if self.gradient_variance == 0:
            return self.max_lr
        log_q = (math.log(4.0) + math.log(self.gradient_variance)
                 - math.log(batch_size) - log_b - 2.0 * math.log(self.L))
        log_sqrt = 0.5 * self._logadd(0.0, log_q)
        log_tau = math.log(2.0) - math.log(self.L) - self._logadd(0.0, log_sqrt)
        return min(self.max_lr, math.exp(log_tau))

    def thresholds(self, gamma: float, batch_size: int) -> tuple[float, float]:
        """Tra tau1,tau2 tu shift signal va B, khong thay doi state."""
        g = self._number(gamma, "gamma")
        batch_size = self._integer(batch_size, "batch_size")
        log_d = math.log(self.D_max)
        log_g = math.log(g) if g else -math.inf
        log_b1 = (log_g + self._logadd(log_g, math.log(2.0) + log_d)
                  if g else -math.inf)
        log_b2 = 2.0 * self._logadd(log_g, log_d)
        lower, upper = self._tau(log_b1, batch_size), self._tau(log_b2, batch_size)
        if not (0.0 <= lower <= upper <= self.max_lr) or upper == 0.0:
            raise ValueError("thresholds cannot be represented in the valid LR domain")
        return lower, upper

    def step(self, t: int, **kwargs: Any) -> float:
        """Tra eta_t, nhan gamma hoac adapter gamma_prev cua shared trainer.

        gamma_prev chi la signal dau vao: EMA hoac lagged reference do caller chon.
        Khong coi gamma_(t-1) la true gamma_t. noise_std/loss_val khong duoc dung.
        """
        t = self._integer(t, "t")
        if t != self.last_t + 1:
            raise ValueError("steps must be consecutive; reset before a new run")
        keys = [key for key in ("gamma", "gamma_prev") if key in kwargs]
        if len(keys) != 1 or "batch_size" not in kwargs:
            raise ValueError("supply exactly one shift signal and batch_size")
        lower, upper = self.thresholds(kwargs[keys[0]], kwargs["batch_size"])
        eta = min(upper, max(lower, self.eta_prev))
        if not math.isfinite(eta) or eta <= 0:
            raise ValueError("learning rate must remain positive and finite")
        self.last_thresholds = (lower, upper)
        self.eta_prev, self.last_t = eta, t
        return float(eta)

    def reset(self) -> None:
        """Bo state cua run cu de LR seed truoc khong anh huong seed sau."""
        self.eta_prev = self.eta_init
        self.last_t = 0
        self.last_thresholds = None
