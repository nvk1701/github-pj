"""
================================================================================
FILE: core/schedules.py
MỤC ĐÍCH:
    Định nghĩa giao diện chuẩn (`BaseSchedule`), các lịch trình tốc độ học cơ sở
    (Baselines) và các lịch trình thích ứng đề xuất trong bài báo cho thuật toán
    Online Mini-batch SGD (Task T6, T7, T9):
    1. `BaseSchedule`: Lớp cơ sở trừu tượng quy định phương thức `.step(t, **kwargs)`
       và `.reset()`. Mọi lịch trình mới của nhóm đều kế thừa từ lớp này.
    2. `ConstantSchedule`: Giữ tốc độ học cố định \\eta_t = \\eta_0 (dùng để tái hiện
       Figure 1 và làm baseline đối chứng).
    3. `InverseTimeSchedule`: Lịch trình giảm dần cổ điển Robbins-Monro O(1/t):
       \\eta_t = \\eta_0 / (1 + decay * (t - 1)).
    4. `InverseSqrtSchedule`: Lịch trình giảm dần chuẩn OCO O(1/sqrt(t)):
       \\eta_t = \\eta_0 / sqrt(t).
    5. `LinearOptimalSchedule`: (TV1 - Khoa - Task T7) Lịch trình tốc độ học tối ưu
       dẫn xuất từ xấp xỉ SDE và điều khiển tối ưu HJB theo Algorithm 1 (Mục 3, trang 6).
    6. `ConvexSchedule`: (TV2 - Khôi - Task T9) Lịch trình tốc độ học thích ứng 2 ngưỡng
       [\\tau_{1,t}, \\tau_{2,t}] cho hàm lồi tổng quát theo Proposition 4.3 (Mục 4, trang 7).

LƯU Ý TƯƠNG THÍCH VỚI `core/tracking.py` (T2 của Khôi):
    Thuộc tính `schedule.name` được dùng làm trường `"method"` khi gọi `init_run(config)`
    để tự sinh tên run W&B (`{experiment}-{method}-seed{seed}`). Vì vậy, `name`
    được đặt bằng dấu gạch ngang (ví dụ: `"constant-0.05"`, `"linear-optimal"`)
    không chứa khoảng trắng hay ký tự đặc biệt `/\\#?%:`.
================================================================================
"""

from abc import ABC, abstractmethod
import math
from numbers import Integral, Real
from typing import Any, List, Optional


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
        - Với `LinearOptimalSchedule` (Alg. 1): đặt lại biến trạng thái ODE `v = v_init`.
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


class LinearOptimalSchedule(BaseSchedule):
    """Lịch trình tốc độ học tối ưu cho Hồi quy Tuyến tính dưới Dịch chuyển Phân phối (Algorithm 1, trang 6).

    Dẫn xuất từ xấp xỉ phương trình vi phân ngẫu nhiên (SDE - Eq. 8), hệ phương trình
    moment ODE (Theorem 3.2 - Eq. 10-11) và điều khiển tối ưu Hamilton-Jacobi-Bellman
    trên quá trình bao trên \\tilde{v}_\\tau (Theorem 3.3 - Eq. 12-13).

    Tại mỗi bước thời gian rời rạc t >= 1, thuật toán thực hiện `ceil(1 / kappa)` bước
    lặp con Euler thuận (forward Euler sub-steps) để cập nhật trạng thái phương sai
    sai số kỳ vọng `v` theo độ dịch chuyển phân phối `gamma_{t-1}`:
        r <- min( (v * B_t) / ((d + 1) * v + sigma^2 * d), max_lr )
        v <- v + kappa * (((d + 1) / B_t) * r^2 - 2 * r) * v
               + kappa * ((sigma^2 * d) / B_t) * r^2
               + 2 * kappa * gamma_{t-1} * sqrt(v)
    Sau đó xuất ra tốc độ học tối ưu cho bước SGD thứ t:
        eta^*_t <- min( (v * B_t) / ((d + 1) * v + sigma^2 * d), max_lr )
    """

    def __init__(
        self,
        max_lr: float = 0.1,
        kappa: float = 0.01,
        v_init: float = 1.0,
        d: Optional[int] = None,
        batch_size: Optional[int] = None,
        noise_std: Optional[float] = None,
        name: Optional[str] = None,
    ) -> None:
        """Khởi tạo `LinearOptimalSchedule` theo Algorithm 1 (Task T7 - Khoa).

        Parameters
        ----------
        max_lr : float, optional
            Tốc độ học tối đa cho phép \\varepsilon = \\eta_{\\max} > 0 (Figure 2 & 3 dùng 0.1,
            Figure 4 dùng 0.5, Figure 5-top dùng 1 / sqrt(d)). Mặc định: 0.1.
        kappa : float, optional
            Độ chia nhỏ bước thời gian \\kappa \\in (0, 1] cho phương pháp Euler thuận
            (mặc định: 0.01, tương ứng 100 bước con Euler mỗi vòng SGD).
        v_init : float, optional
            Giá trị khởi tạo \\tilde{v}_0 >= 0 của quá trình bao trên moment bậc hai
            (mặc định: 1.0 theo Figure 3, hoặc truyền ||\\theta_0 - \\theta^*_1||_2^2).
        d : Optional[int], optional
            Số chiều tham số mặc định nếu caller không truyền `d` vào `.step(...)`.
        batch_size : Optional[int], optional
            Kích thước mini-batch B_t mặc định nếu caller không truyền vào `.step(...)`.
        noise_std : Optional[float], optional
            Độ lệch chuẩn nhiễu nhãn \\sigma mặc định nếu caller không truyền vào `.step(...)`.
        name : Optional[str], optional
            Tên định danh lịch trình cho W&B. Nếu `None`, đặt mặc định là `"linear-optimal"`.
        """
        if isinstance(max_lr, bool) or not isinstance(max_lr, Real) or not math.isfinite(float(max_lr)) or float(max_lr) <= 0.0:
            raise ValueError(f"max_lr (epsilon) phải là số thực dương hữu hạn, nhận được {max_lr}.")
        if isinstance(kappa, bool) or not isinstance(kappa, Real) or not math.isfinite(float(kappa)) or not (0.0 < float(kappa) <= 1.0):
            raise ValueError(f"kappa phải nằm trong khoảng (0, 1], nhận được {kappa}.")
        if isinstance(v_init, bool) or not isinstance(v_init, Real) or not math.isfinite(float(v_init)) or float(v_init) < 0.0:
            raise ValueError(f"v_init phải là số thực không âm hữu hạn, nhận được {v_init}.")

        sched_name = name if name is not None else "linear-optimal"
        super().__init__(name=sched_name)

        self.max_lr = float(max_lr)
        self.kappa = float(kappa)
        self.n_substeps = int(math.ceil(1.0 / self.kappa))
        self.v_init = float(v_init)
        self.default_d = int(d) if d is not None else None
        self.default_batch_size = int(batch_size) if batch_size is not None else None
        self.default_noise_std = float(noise_std) if noise_std is not None else None

        self.v: float = self.v_init
        self.last_lr: Optional[float] = None
        self.v_history: List[float] = []

    def _compute_candidate_lr(self, v_val: float, d_val: float, b_val: float, sigma2: float) -> float:
        """Tính bước nhảy ứng viên min(v * B_t / ((d + 1) * v + sigma^2 * d), max_lr) theo Eq. (13)."""
        denom = (d_val + 1.0) * v_val + sigma2 * d_val
        if denom <= 0.0:
            return 0.0
        return min((v_val * b_val) / denom, self.max_lr)

    def step(self, t: int, **kwargs: Any) -> float:
        """Thực hiện vòng lặp Euler của Algorithm 1 tại bước t và trả về \\eta^*_t.

        Parameters
        ----------
        t : int
            Chỉ số bước thời gian hiện tại (t >= 1).
        **kwargs : Any
            - `gamma_prev` hoặc `gamma` (float): Độ dịch chuyển phân phối \\gamma_{t-1} >= 0.
            - `batch_size` (int): Kích thước mini-batch B_t.
            - `d` (int): Số chiều không gian tham số d.
            - `noise_std` (float): Độ lệch chuẩn nhiễu \\sigma (sao cho phương sai là \\sigma^2).

        Returns
        -------
        float
            Tốc độ học tối ưu \\eta^*_t \\in [0, max_lr].
        """
        if isinstance(t, bool) or not isinstance(t, Integral) or int(t) < 1:
            raise ValueError(f"Bước thời gian t phải là số nguyên dương >= 1, nhận được {t}.")

        gamma_raw = kwargs.get("gamma_prev", kwargs.get("gamma", 0.0))
        gamma_val = float(gamma_raw)
        if not math.isfinite(gamma_val) or gamma_val < 0.0:
            raise ValueError(f"Độ dịch chuyển gamma phải không âm và hữu hạn, nhận được {gamma_raw}.")

        d_raw = kwargs.get("d", self.default_d)
        if d_raw is None or int(d_raw) <= 0:
            raise ValueError("Cần cung cấp số chiều d > 0 cho LinearOptimalSchedule.")
        d_val = float(int(d_raw))

        b_raw = kwargs.get("batch_size", self.default_batch_size)
        if b_raw is None or int(b_raw) <= 0:
            raise ValueError("Cần cung cấp batch_size B_t > 0 cho LinearOptimalSchedule.")
        b_val = float(int(b_raw))

        noise_raw = kwargs.get("noise_std", self.default_noise_std)
        if noise_raw is None or float(noise_raw) < 0.0:
            raise ValueError("Cần cung cấp noise_std >= 0 cho LinearOptimalSchedule.")
        sigma2 = float(noise_raw) ** 2

        # Vòng lặp con Forward Euler: j = 1, 2, ..., ceil(1 / kappa)
        for _ in range(self.n_substeps):
            r = self._compute_candidate_lr(self.v, d_val, b_val, sigma2)
            if self.v == 0.0 and gamma_val > 0.0:
                # Nghiệm giải tích chính xác của dv/dtau = 2 * gamma * sqrt(v) khi xuất phát từ v = 0
                drift_inc = (self.kappa * gamma_val) ** 2
            else:
                drift_inc = 2.0 * self.kappa * gamma_val * math.sqrt(max(self.v, 0.0))

            dv = (
                self.kappa * (((d_val + 1.0) / b_val) * (r * r) - 2.0 * r) * self.v
                + self.kappa * ((sigma2 * d_val) / b_val) * (r * r)
                + drift_inc
            )
            self.v = max(0.0, self.v + dv)

        eta_t = self._compute_candidate_lr(self.v, d_val, b_val, sigma2)
        self.last_lr = float(eta_t)
        self.v_history.append(self.v)
        return float(eta_t)

    def reset(self) -> None:
        """Đặt lại biến trạng thái ODE `v` về `v_init` trước khi chạy seed mới."""
        self.v = self.v_init
        self.last_lr = None
        self.v_history = []


class ConvexSchedule(BaseSchedule):
    """Tinh LR theo Prop. 4.3, Eq. (18)-(20). Variance la sigma binh phuong.

    L, D_max va gradient_variance la tham so cua bound, tach khoi nhieu nhan.
    eta_init nam trong (0, 1/L]. reset khoi phuc LR nay cho seed moi.
    """

    def __init__(
        self,
        L: float,
        D_max: float,
        gradient_variance: float,
        eta_init: float = 1.0,
    ) -> None:
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
    def _number(value: Any, name: str, *, positive: bool = False) -> float:
        """Doc scalar huu han, tu choi bool va gia tri am."""
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real scalar")
        result = float(value)
        if not math.isfinite(result) or result < 0 or (positive and result == 0):
            raise ValueError(f"{name} is outside its finite domain")
        return result

    @staticmethod
    def _integer(value: Any, name: str) -> int:
        """Chi chap nhan so nguyen duong cho step va batch size."""
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        return int(value)

    @staticmethod
    def _logadd(a: float, b: float) -> float:
        """Tinh log(exp(a)+exp(b)) ma khong lam tran so."""
        high, low = max(a, b), min(a, b)
        return high + math.log1p(math.exp(low - high))

    def _tau(self, log_b: float, batch_size: int) -> float:
        """Rationalize Eq.18: tau=2/[L*(1+sqrt(1+4v/(B*b*L^2)))]."""
        if log_b == -math.inf:
            return 0.0
        if self.gradient_variance == 0:
            return self.max_lr
        log_q = (
            math.log(4.0)
            + math.log(self.gradient_variance)
            - math.log(batch_size)
            - log_b
            - 2.0 * math.log(self.L)
        )
        log_sqrt = 0.5 * self._logadd(0.0, log_q)
        log_tau = math.log(2.0) - math.log(self.L) - self._logadd(0.0, log_sqrt)
        return min(self.max_lr, math.exp(log_tau))

    def thresholds(self, gamma: float, batch_size: int) -> tuple[float, float]:
        """Tra tau1,tau2 tu shift signal va B, khong thay doi state."""
        g = self._number(gamma, "gamma")
        batch_size = self._integer(batch_size, "batch_size")
        log_d = math.log(self.D_max)
        log_g = math.log(g) if g else -math.inf
        log_b1 = log_g + self._logadd(log_g, math.log(2.0) + log_d) if g else -math.inf
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


class NonConvexSchedule(BaseSchedule):
    """Lịch trình tốc độ học tối ưu cho Hàm Không lồi theo Theorem 5.1 (Công thức 25, trang 8).

    Cực tiểu hóa cận trên kỳ vọng của tổng bình phương chuẩn gradient (Eq. 24) theo
    từng bước thời gian t >= 1 với nghiệm đóng:
        b_t = L * (gamma_t + loss_val)
        eta^*_t = (B_t / (L * sigma^2)) * (sqrt(b_t^2 + (2 * sigma^2 / B_t) * b_t) - b_t)
                = 2 * b_t / (L * (b_t * B_t * L / B_t ...))
                = 2 / (L * (1 + sqrt(1 + 2 * sigma^2 / (B_t * b_t))))  khi b_t > 0.
    Giá trị eta^*_t luôn nằm trong miền ổn định [0, 1 / L] và đồng biến nghiêm ngặt
    theo cả độ dịch chuyển phân phối `gamma_t` lẫn giá trị hàm mất mát `loss_val`.
    """

    def __init__(
        self,
        L: float = 1.0,
        gradient_variance: float = 1.0,
        default_batch_size: int = 64,
        name: str = "nonconvex-optimal",
    ) -> None:
        """Khởi tạo `NonConvexSchedule` (Task T12).

        Parameters
        ----------
        L : float, optional
            Hằng số trơn Lipschitz của gradient L > 0 (mặc định: 1.0).
        gradient_variance : float, optional
            Cận trên phương sai gradient \\sigma^2 >= 0 (mặc định: 1.0).
        default_batch_size : int, optional
            Kích thước mini-batch B_t mặc định nếu không truyền vào `.step(...)` (mặc định: 64).
        name : str, optional
            Tên định danh lịch trình cho W&B (mặc định: `"nonconvex-optimal"`).
        """
        super().__init__(name=name)
        if isinstance(L, bool) or not isinstance(L, Real) or not math.isfinite(float(L)) or float(L) <= 0.0:
            raise ValueError(f"Hằng số trơn L phải là số thực dương hữu hạn, nhận được {L}.")
        if (
            isinstance(gradient_variance, bool)
            or not isinstance(gradient_variance, Real)
            or not math.isfinite(float(gradient_variance))
            or float(gradient_variance) < 0.0
        ):
            raise ValueError(f"gradient_variance (sigma^2) phải >= 0 và hữu hạn, nhận được {gradient_variance}.")
        if isinstance(default_batch_size, bool) or not isinstance(default_batch_size, Integral) or int(default_batch_size) < 1:
            raise ValueError(f"default_batch_size phải là số nguyên dương, nhận được {default_batch_size}.")

        self.L = float(L)
        self.gradient_variance = float(gradient_variance)
        self.max_lr = 1.0 / self.L
        self.default_batch_size = int(default_batch_size)
        self.last_lr: Optional[float] = None

    def step(self, t: int, **kwargs: Any) -> float:
        """Tính tốc độ học tối ưu \\eta^*_t tại bước t >= 1 theo Công thức (25).

        Parameters
        ----------
        t : int
            Chỉ số bước thời gian hiện tại (t >= 1).
        **kwargs : Any
            - `gamma_prev` hoặc `gamma` (float): Độ dịch chuyển phân phối \\gamma_t >= 0 (mặc định: 0.0).
            - `loss_val` (float): Giá trị hàm mất mát trung bình trên mini-batch hiện tại \\ell^B_t(\\theta_t) >= 0.
            - `batch_size` (int, optional): Kích thước mini-batch B_t.

        Returns
        -------
        float
            Bước nhảy tối ưu \\eta^*_t \\in [0, 1 / L].
        """
        if isinstance(t, bool) or not isinstance(t, Integral) or int(t) < 1:
            raise ValueError(f"Bước thời gian t phải là số nguyên dương >= 1, nhận được {t}.")

        gamma_val = float(kwargs.get("gamma_prev", kwargs.get("gamma", 0.0)))
        loss_val = float(kwargs.get("loss_val", 0.0))
        batch_size = int(kwargs.get("batch_size", self.default_batch_size))

        if not math.isfinite(gamma_val) or gamma_val < 0.0:
            raise ValueError(f"gamma phải không âm và hữu hạn, nhận được {gamma_val}.")
        if not math.isfinite(loss_val) or loss_val < 0.0:
            raise ValueError(f"loss_val phải không âm và hữu hạn, nhận được {loss_val}.")
        if batch_size < 1:
            raise ValueError(f"batch_size phải >= 1, nhận được {batch_size}.")

        b_t = self.L * (gamma_val + loss_val)
        if b_t <= 0.0:
            eta_t = 0.0
        elif self.gradient_variance == 0.0:
            eta_t = self.max_lr
        else:
            # Dạng hữu tỷ hóa tương đương Eq. (25) giúp tránh mất chữ số có nghĩa khi b_t lớn
            ratio = (2.0 * self.gradient_variance) / (float(batch_size) * b_t)
            eta_t = min(self.max_lr, 2.0 / (self.L * (1.0 + math.sqrt(1.0 + ratio))))

        self.last_lr = float(eta_t)
        return float(eta_t)

    def reset(self) -> None:
        """Đặt lại trạng thái lịch trình trước mỗi seed mới."""
        self.last_lr = None

