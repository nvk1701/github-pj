"""
================================================================================
FILE: core/trainer.py
MỤC ĐÍCH:
    Cài đặt bộ huấn luyện Học Trực tuyến Không ngừng (Never-Ending Online Mini-batch
    SGD Trainer) dùng chung cho các bài toán Linear Regression (Mục 3) và Logistic
    Regression (Mục 4), tuân thủ nghiêm ngặt chu trình 2 pha (Test-then-Train)
    trong Mục 2 của bài báo (Task T6):

    Tại mỗi bước thời gian rời rạc t = 1, 2, ..., T:
    - [STAGE A — ONLINE INFERENCE / EVALUATION PIPELINE]:
      1. Nhận mini-batch mẫu mới z_t = {(x_{t,k}, y_{t,k})}_{k=1}^{B_t} ~ P_t.
      2. Dùng trọng số hiện tại \\theta_t (đóng băng, chưa nhìn thấy nhãn bước t)
         để dự đoán và tính độ tiếc nuối tức thời (Instantaneous Dynamic Regret):
             reg_t = \\bar{\\ell}_t(\\theta_t) - \\bar{\\ell}_t(\\theta^*_t)
         và cộng dồn vào tổng Dynamic Regret: Reg(t) = \\sum_{s=1}^t reg_s.

    - [STAGE B — ONLINE TRAINING & ADAPTIVE LR PIPELINE]:
      3. Tính đạo hàm trung bình trên mini-batch: \\nabla \\ell^B_t(\\theta_t).
      4. Lấy tín hiệu độ trôi phân phối \\gamma_{t-1} (từ Oracle hoặc từ bộ ước lượng
         trực tuyến `EMADriftEstimator`).
      5. Gọi `schedule.step(t, ...)` để nhận tốc độ học thích ứng \\eta_t.
      6. Cập nhật Projected Mini-batch SGD (Công thức 3):
             \\theta_{t+1} = \\Pi_\\Theta(\\theta_t - \\eta_t \\nabla \\ell^B_t(\\theta_t))
      7. Nếu bật W&B (`use_wandb=True`), ghi log trực tiếp vào run đang active
         (được khởi tạo bởi `core.tracking.init_run`) với `commit=True`.

AI SỬ DỤNG FILE NÀY?
    - TV1 (Khoa): Dùng `OnlineSGDTrainer(..., loss_type="linear")` trong
      `experiments/linear_regression.py` để tái hiện Figure 4 và Figure 5 (top).
    - TV2 (Khôi): Dùng `OnlineSGDTrainer(..., loss_type="logistic", max_norm=d)`
      trong `experiments/convex.py` và `experiments/sensitivity.py` để tái hiện
      Figure 5 (bottom) và phân tích độ nhạy.
================================================================================
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np

from core.drift import EMADriftEstimator
from core.schedules import BaseSchedule
from core.utils import project_to_ball

try:
    import wandb
except ImportError:
    wandb = None


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Hàm logistic sigmoid ổn định số học (chống tràn số exp khi |z| lớn)."""
    z_clipped = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))


@dataclass
class TrainingHistory:
    """Cấu trúc dữ liệu lưu toàn bộ lịch sử quỹ đạo và metric của 1 lần chạy Online SGD.

    Attributes
    ----------
    schedule_name : str
        Tên lịch trình tốc độ học đã sử dụng.
    theta_path : np.ndarray
        Quỹ đạo trọng số học được \\theta_t qua T bước, kích thước `(T, d)`.
        Dùng để vẽ quỹ đạo 2 tọa độ đầu trong Figure 4 và Figure 5 (right).
    theta_star_path : np.ndarray
        Quỹ đạo trọng số tối ưu Oracle \\theta^*_t qua T bước, kích thước `(T, d)`.
    lr_history : np.ndarray
        Lịch sử tốc độ học \\eta_t qua T bước, kích thước `(T,)`.
        Dùng để vẽ đồ thị dao động learning rate ở Figure 4 (bottom).
    inst_regret : np.ndarray
        Độ tiếc nuối tức thời `reg_t` tại từng bước, kích thước `(T,)`.
    cum_regret : np.ndarray
        Độ tiếc nuối tích lũy `Reg(t)` từ bước 1 đến bước t, kích thước `(T,)`.
        Dùng để vẽ đường cong Cumulative Regret ở Figure 5 (left).
    batch_loss : np.ndarray
        Hàm mất mát trung bình trên từng mini-batch, kích thước `(T,)`.
    gamma_used : np.ndarray
        Tín hiệu độ dịch chuyển \\gamma đã truyền cho schedule tại mỗi bước, kích thước `(T,)`.
    metadata : Dict[str, Any]
        Thông tin cấu hình của lần chạy (T, d, loss_type, batch_size, noise_std, seed).
    """

    schedule_name: str
    theta_path: np.ndarray
    theta_star_path: np.ndarray
    lr_history: np.ndarray
    inst_regret: np.ndarray
    cum_regret: np.ndarray
    batch_loss: np.ndarray
    gamma_used: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


class OnlineSGDTrainer:
    """Bộ thực thi vòng lặp Online Mini-batch SGD cho phân phối dữ liệu biến thiên theo thời gian."""

    def __init__(
        self,
        d: int,
        schedule: BaseSchedule,
        loss_type: str = "linear",
        batch_size: int = 256,
        noise_std: float = 0.1,
        max_norm: Optional[float] = None,
        use_oracle_gamma: bool = True,
        ema_beta: float = 0.9,
        use_wandb: bool = False,
        wandb_log_interval: int = 10,
    ) -> None:
        """Khởi tạo bộ huấn luyện `OnlineSGDTrainer`.

        Parameters
        ----------
        d : int
            Số chiều của vector trọng số mô hình \\theta_t \\in R^d.
        schedule : BaseSchedule
            Đối tượng lịch trình tốc độ học (ví dụ: `ConstantSchedule`, `LinearOptimalSchedule`,
            `ConvexSchedule`).
        loss_type : str, optional
            Loại hàm mất mát:
            - `"linear"`: Hồi quy tuyến tính bình phương tối thiểu (Mục 3 & 6.1).
            - `"logistic"`: Hồi quy logistic với Binary Cross-Entropy loss (Mục 4 & 6.1).
        batch_size : int, optional
            Kích thước mini-batch B_t tại mỗi bước (mặc định: 256 theo Mục 6.1).
        noise_std : float, optional
            Độ lệch chuẩn nhiễu \\sigma của biến phản hồi \\varepsilon_{t,k} ~ N(0, \\sigma^2)
            (mặc định: 0.1 theo Mục 6.1).
        max_norm : Optional[float], optional
            Bán kính miền chấp nhận được \\Theta cho phép chiếu \\Pi_\\Theta (mặc định: `None`).
        use_oracle_gamma : bool, optional
            - Nếu `True`: Truyền độ dịch chuyển thực \\gamma_{t-1} = ||\\theta^*_{t-1} - \\theta^*_t||_2
              vào `schedule.step(...)`.
            - Nếu `False`: Tự động ước lượng độ dịch chuyển bằng `EMADriftEstimator` (Remark 3.5).
        ema_beta : float, optional
            Hệ số làm mượt \\beta cho bộ ước lượng EMA khi `use_oracle_gamma=False` (mặc định: 0.9).
        use_wandb : bool, optional
            Nếu `True` và đang có một W&B run active (được tạo bởi `core.tracking.init_run`),
            trainer sẽ tự động gọi `wandb.log(metrics, step=t, commit=True)`.
        wandb_log_interval : int, optional
            Chu kỳ bước thời gian gửi log lên W&B (mặc định: mỗi 10 bước và bước cuối T).
        """
        if loss_type not in ("linear", "logistic"):
            raise ValueError(f"loss_type '{loss_type}' không hợp lệ. Chỉ hỗ trợ 'linear' hoặc 'logistic'.")
        self.d = int(d)
        self.schedule = schedule
        self.loss_type = loss_type
        self.batch_size = int(batch_size)
        self.noise_std = float(noise_std)
        self.max_norm = max_norm
        self.use_oracle_gamma = use_oracle_gamma
        self.drift_estimator = EMADriftEstimator(beta=ema_beta)
        self.use_wandb = use_wandb
        self.wandb_log_interval = max(1, int(wandb_log_interval))

    def _sample_batch_and_compute_grad(
        self,
        theta_t: np.ndarray,
        theta_star_t: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, float, float]:
        """Sinh mini-batch từ phân phối P_t, đánh giá Inference (Stage A) và tính Gradient (Stage B).

        Thiết lập dữ liệu tuân thủ đúng Mục 6.1 (trang 9) của bài báo:
        - Đặc trưng đầu vào: X \\in R^{B_t x d} với mỗi phần tử x_{i,j} ~ N(0, 1).
        - Nhiễu: \\varepsilon \\in R^{B_t} với \\varepsilon_i ~ N(0, noise_std^2).

        Parameters
        ----------
        theta_t : np.ndarray
            Trọng số hiện tại của bộ học trước khi cập nhật ở bước t, kích thước `(d,)`.
        theta_star_t : np.ndarray
            Trọng số tối ưu Oracle tại bước t, kích thước `(d,)`.
        rng : np.random.Generator
            Bộ sinh số ngẫu nhiên độc lập của NumPy.

        Returns
        -------
        tuple[np.ndarray, float, float]
            - `grad`: Vector đạo hàm trung bình mẫu \\nabla \\ell^B_t(\\theta_t), kích thước `(d,)`.
            - `batch_loss_val`: Giá trị hàm mất mát trung bình trên mini-batch hiện tại.
            - `inst_regret_val`: Độ tiếc nuối tức thời `reg_t = \\bar{\\ell}_t(\\theta_t) - \\bar{\\ell}_t(\\theta^*_t)`.
        """
        # Sinh ma trận đặc trưng X ~ N(0, I_d) và vector nhiễu eps ~ N(0, \sigma^2)
        X = rng.normal(loc=0.0, scale=1.0, size=(self.batch_size, self.d))
        eps = rng.normal(loc=0.0, scale=self.noise_std, size=(self.batch_size,))

        if self.loss_type == "linear":
            # ------------------------------------------------------------------
            # BÀI TOÁN 1: HỒI QUY TUYẾN TÍNH (Mục 3 & Mục 6.1)
            # Mô hình sinh nhãn: y = X @ \theta^*_t + \varepsilon
            # Hàm mất mát: \ell(\theta, z) = 1/2 * (y - <x, \theta>)^2
            # ------------------------------------------------------------------
            # [Stage A - Online Inference]: Dự đoán bằng trọng số hiện tại \theta_t
            preds = X @ theta_t
            y = X @ theta_star_t + eps
            residuals = preds - y  # Sai số dự đoán: <x, \theta_t> - y
            batch_loss_val = float(0.5 * np.mean(np.square(residuals)))

            # Kỳ vọng giải tích của Dynamic Regret khi x ~ N(0, I) (trang 5 bài báo):
            # E[reg_t] = E[\bar{\ell}_t(\theta_t) - \bar{\ell}_t(\theta^*_t)] = 1/2 * ||\theta_t - \theta^*_t||_2^2
            diff = theta_t - theta_star_t
            inst_regret_val = float(0.5 * np.dot(diff, diff))

            # [Stage B - Online Training]: Đạo hàm trung bình mẫu \nabla \ell^B_t(\theta_t)
            grad = (X.T @ residuals) / float(self.batch_size)
        else:
            # ------------------------------------------------------------------
            # BÀI TOÁN 2: HỒI QUY LOGISTIC (Mục 4 & Mục 6.1)
            # Mô hình sinh nhãn mềm: y = sigmoid(X @ \theta^*_t + \varepsilon)
            # Hàm mất mát: Binary Cross-Entropy (Log Loss)
            # ------------------------------------------------------------------
            # [Stage A - Online Inference]: Dự đoán xác suất với \theta_t hiện tại
            logits_pred = X @ theta_t
            probs_pred = _sigmoid(logits_pred)
            y = _sigmoid(X @ theta_star_t + eps)

            # Tính Binary Cross-Entropy ổn định số học qua hàm logaddexp:
            # -[y * log(sigmoid(z)) + (1 - y) * log(1 - sigmoid(z))] = log(1 + exp(z)) - y * z
            loss_vec = np.logaddexp(0.0, logits_pred) - y * logits_pred
            batch_loss_val = float(np.mean(loss_vec))

            logits_oracle = X @ theta_star_t
            loss_oracle_vec = np.logaddexp(0.0, logits_oracle) - y * logits_oracle
            oracle_loss_val = float(np.mean(loss_oracle_vec))
            inst_regret_val = max(0.0, batch_loss_val - oracle_loss_val)

            # [Stage B - Online Training]: Đạo hàm của Binary Cross-Entropy là (sigmoid(X @ \theta_t) - y) * X
            grad = (X.T @ (probs_pred - y)) / float(self.batch_size)

        return grad, batch_loss_val, inst_regret_val

    def run(
        self,
        theta_star_path: np.ndarray,
        gamma_seq: Optional[np.ndarray] = None,
        theta_init: Optional[np.ndarray] = None,
        seed: int = 42,
        step_callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
    ) -> TrainingHistory:
        """Chạy toàn bộ vòng lặp học trực tuyến qua T bước thời gian.

        Parameters
        ----------
        theta_star_path : np.ndarray
            Mảng quỹ đạo nghiệm tối ưu Oracle \\theta^*_t, kích thước `(T, d)`.
        gamma_seq : Optional[np.ndarray], optional
            Chuỗi độ dịch chuyển \\gamma_t đã tính sẵn, kích thước `(T,)`. Nếu `None`,
            hàm sẽ tự động tính từ `theta_star_path`.
        theta_init : Optional[np.ndarray], optional
            Trọng số khởi tạo \\theta_1 tại đầu bước 1. Mặc định là vector 0
            (`np.zeros(d)`), khớp với mô tả "Setting \\theta_0 = 0" ở Mục 6.1.
        seed : int, optional
            Hạt giống ngẫu nhiên để sinh luồng mini-batch (mặc định: 42).
        step_callback : Optional[Callable[[int, Dict[str, float]], None]], optional
            Hàm callback tùy chọn gọi ở cuối mỗi bước t với tham số `(t, metrics)`.

        Returns
        -------
        TrainingHistory
            Đối tượng chứa toàn bộ lịch sử quỹ đạo, tốc độ học và regret của lần chạy.
        """
        T, d_path = theta_star_path.shape
        if d_path != self.d:
            raise ValueError(f"Số chiều không khớp: trainer d={self.d}, nhưng quỹ đạo có d={d_path}.")

        if gamma_seq is None:
            from core.drift import compute_oracle_shifts
            gamma_seq = compute_oracle_shifts(theta_star_path)

        # Khởi tạo generator ngẫu nhiên riêng biệt theo seed và reset trạng thái schedule
        rng = np.random.default_rng(seed)
        self.schedule.reset()
        self.drift_estimator.reset()

        # Khởi tạo trọng số mô hình \theta_1 = 0
        theta_t = (
            np.zeros(self.d, dtype=np.float64)
            if theta_init is None
            else np.asarray(theta_init, dtype=np.float64).copy()
        )

        # Cấp phát sẵn bộ nhớ lưu lịch sử qua T bước
        theta_path = np.zeros((T, self.d), dtype=np.float64)
        lr_history = np.zeros(T, dtype=np.float64)
        inst_regret = np.zeros(T, dtype=np.float64)
        cum_regret = np.zeros(T, dtype=np.float64)
        batch_loss = np.zeros(T, dtype=np.float64)
        gamma_used = np.zeros(T, dtype=np.float64)

        running_regret = 0.0

        for idx in range(T):
            t = idx + 1  # Chỉ số bước thời gian đánh số từ 1: t in {1, 2, ..., T}
            theta_star_t = theta_star_path[idx]
            theta_path[idx] = theta_t

            # ------------------------------------------------------------------
            # BƯỚC 1 (STAGE A - INFERENCE): Đánh giá mô hình \theta_t trên batch mới
            # và tính đạo hàm mẫu cho Stage B.
            # ------------------------------------------------------------------
            grad, b_loss, reg_t = self._sample_batch_and_compute_grad(theta_t, theta_star_t, rng)
            running_regret += reg_t

            # ------------------------------------------------------------------
            # BƯỚC 2 (STAGE B - SHIFT TRACKING): Lấy độ dịch chuyển \gamma_{t-1}
            # ------------------------------------------------------------------
            ema_gamma = self.drift_estimator.update(theta_t)
            if self.use_oracle_gamma:
                gamma_prev = float(gamma_seq[idx - 1]) if idx >= 1 else 0.0
            else:
                gamma_prev = ema_gamma
            gamma_used[idx] = gamma_prev

            # ------------------------------------------------------------------
            # BƯỚC 3 (STAGE B - ADAPTIVE LR): Hỏi schedule tốc độ học \eta_t
            # ------------------------------------------------------------------
            eta_t = float(
                self.schedule.step(
                    t=t,
                    gamma_prev=gamma_prev,
                    batch_size=self.batch_size,
                    d=self.d,
                    noise_std=self.noise_std,
                    loss_val=b_loss,
                )
            )

            lr_history[idx] = eta_t
            inst_regret[idx] = reg_t
            cum_regret[idx] = running_regret
            batch_loss[idx] = b_loss

            # ------------------------------------------------------------------
            # BƯỚC 4 (STAGE B - SGD UPDATE): Cập nhật \theta_{t+1} và chiếu lên \Theta
            # ------------------------------------------------------------------
            theta_next = theta_t - eta_t * grad
            theta_t = project_to_ball(theta_next, self.max_norm)

            # Đóng gói các metric tại bước t
            metrics = {
                "train/lr": eta_t,
                "eval/inst_regret": reg_t,
                "eval/cum_regret": running_regret,
                "eval/batch_loss": b_loss,
                "shift/gamma_prev": gamma_prev,
            }
            if step_callback is not None:
                step_callback(t, metrics)

            # Ghi log lên Weights & Biases (tương thích 100% với `core/tracking.py` của Khôi)
            if (
                self.use_wandb
                and wandb is not None
                and getattr(wandb, "run", None) is not None
                and (t % self.wandb_log_interval == 0 or t == T)
            ):
                wandb.log(metrics, step=t, commit=True)

        return TrainingHistory(
            schedule_name=self.schedule.name,
            theta_path=theta_path,
            theta_star_path=theta_star_path.copy(),
            lr_history=lr_history,
            inst_regret=inst_regret,
            cum_regret=cum_regret,
            batch_loss=batch_loss,
            gamma_used=gamma_used,
            metadata={
                "T": T,
                "d": self.d,
                "loss_type": self.loss_type,
                "batch_size": self.batch_size,
                "noise_std": self.noise_std,
                "seed": seed,
            },
        )
