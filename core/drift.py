"""
================================================================================
FILE: core/drift.py
MỤC ĐÍCH:
    Mô phỏng sự dịch chuyển phân phối dữ liệu (Distribution Shift) theo thời gian
    và cung cấp bộ ước lượng độ trôi trực tuyến (Task T6):
    1. `generate_spiral_trajectory(...)`: Sinh quỹ đạo nghiệm tối ưu (Oracle)
       xoắn ốc \\theta^*_t \\in R^d và bản rời rạc hóa \\hat{\\theta}^*_t theo
       đúng Công thức (26), Mục 6.1 (trang 9 bài báo).
    2. `compute_oracle_shifts(...)`: Tính độ dịch chuyển thực tế giữa 2 bước liên tiếp
       \\gamma_t = ||\\theta^*_t - \\theta^*_{t+1}||_2 theo Definition 2.1 (Công thức 6).
    3. `EMADriftEstimator`: Bộ ước lượng độ trôi bằng trung bình trượt hàm mũ (EMA)
       \\hat{\\gamma}_t = \\beta \\hat{\\gamma}_{t-1} + (1 - \\beta)||\\theta_t - \\theta_{t-1}||_2
       khi bộ học không được biết trước \\gamma_t (Remark 3.5 & Remark 4.4).
    4. `generate_bursty_shifts(...)` & `generate_smooth_shifts(...)`: Sinh chuỗi
       độ trôi tổng hợp dạng nhảy đột ngột (bursty) hoặc trôi mượt (smooth) để
       minh họa hành vi của Algorithm 1 giống Figure 2 (Mục 3, trang 6).

AI SỬ DỤNG FILE NÀY?
    - TV1 (Khoa): Dùng `generate_spiral_trajectory` cho Thực nghiệm 1 (Linear
      Regression - tái hiện Figure 4 với l in {1, 4, 16} và Figure 5-top với l=8,
      d in {2, 8, 32, 128, 512}).
    - TV2 (Khôi): Dùng `generate_spiral_trajectory` cho Thực nghiệm 2 (Logistic
      Regression - tái hiện Figure 5-bottom với l=8, d in {2, 8, 32, 128, 512})
      và dùng cho bài phân tích độ nhạy (Sensitivity Analysis - Task T17).
================================================================================
"""

import math
from typing import Optional, Tuple

import numpy as np


def generate_spiral_trajectory(
    T: int,
    d: int,
    a: float = 1.0,
    b: float = -1.0,
    k: float = 4.0,
    ell: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sinh quỹ đạo mô hình tối ưu (Oracle) xoắn ốc `d` chiều và chuỗi độ dịch chuyển `gamma_t`.

    Cài đặt chính xác Công thức (26) trong Mục 6.1 (trang 9) của bài báo:
        Với mỗi chiều i \\in {1, 2, ..., d} (đánh số từ 1) và bước t \\in {1, ..., T}:
        - Nếu i lẻ:  \\theta^*_t(i) = r_{a,b}(t)^3 * cos( ceil(i/2) * 2 * k * pi * alpha(t) )
        - Nếu i chẵn: \\theta^*_t(i) = r_{a,b}(t)^3 * sin( ceil(i/2) * 2 * k * pi * alpha(t) )
    Trong đó:
        - r_{a,b}(t) = linspace(a, b, T): điều khiển bán kính xoắn ốc. Với a=1, b=-1,
          bán kính lập phương r^3 giảm từ 1 về 0 (xoắn vào gốc tọa độ) rồi từ 0
          ra -1 (xoắn ngược ra ngoài), tạo tốc độ di chuyển không đơn điệu!
        - alpha(t) = linspace(0, 1, T): biến thời gian chuẩn hóa trên đoạn [0, 1].
        - k: tần số góc cơ sở (bài báo dùng k = 4).

    Để khảo sát sự khác biệt giữa dịch chuyển liên tục (continuous shift) và
    dịch chuyển rời rạc nhảy bậc (discrete/bursty shift), bài báo lấy mẫu thưa
    (downsample) quỹ đạo theo bước nhảy `ell` (ký hiệu l \\in {1, 4, 8, 16}):
        \\hat{\\theta}^*_t = \\theta^*_{ ceil(t / ell) * ell }

    Parameters
    ----------
    T : int
        Tổng số bước thời gian của luồng dữ liệu (ví dụ: T = 2000 trong Mục 6.1).
    d : int
        Số chiều của vector trọng số \\theta^*_t (ví dụ: d \\in {2, 8, 32, 128, 512}).
    a : float, optional
        Giá trị đầu của bán kính r_{a,b}(t) (mặc định: 1.0).
    b : float, optional
        Giá trị cuối của bán kính r_{a,b}(t) (mặc định: -1.0).
    k : float, optional
        Tần số cơ sở của hàm lượng giác (mặc định: 4.0 theo Mục 6.1).
    ell : int, optional
        Bước rời rạc hóa (downsampling stride) l >= 1 (mặc định: 1 là trôi liên tục;
        Figure 4 dùng l in {1, 4, 16}; Figure 5 dùng l = 8).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - `theta_star_discretized`: Mảng NumPy kích thước `(T, d)` chứa quỹ đạo \\hat{\\theta}^*_t.
        - `gamma_seq`: Mảng NumPy kích thước `(T,)` chứa độ dịch chuyển thực sự
          \\gamma_t = ||\\hat{\\theta}^*_t - \\hat{\\theta}^*_{t+1}||_2 tại mỗi bước.
    """
    if T <= 0 or d <= 0:
        raise ValueError(f"T và d phải là số nguyên dương, nhận được T={T}, d={d}.")
    if ell <= 0:
        raise ValueError(f"Bước rời rạc hóa ell phải >= 1, nhận được ell={ell}.")

    # Tạo lưới bán kính r_{a,b}(t) và thời gian chuẩn hóa alpha(t)
    r = np.linspace(a, b, T, dtype=np.float64)
    alpha = np.linspace(0.0, 1.0, T, dtype=np.float64)
    radius_cubed = np.power(r, 3)  # r_{a,b}(t)^3, kích thước (T,)

    # Tính quỹ đạo xoắn ốc liên tục \theta^*_t cho từng tọa độ i = 1..d
    theta_continuous = np.zeros((T, d), dtype=np.float64)
    for i in range(1, d + 1):
        # Hệ số nhân tần số ceil(i/2) giúp các cặp tọa độ (1,2), (3,4), ... độc lập tuyến tính
        freq_multiplier = math.ceil(i / 2.0)
        angle = freq_multiplier * 2.0 * k * math.pi * alpha
        if i % 2 == 1:
            # Tọa độ lẻ (i = 1, 3, 5, ...) -> chỉ số mảng Python là i - 1 (0, 2, 4, ...)
            theta_continuous[:, i - 1] = radius_cubed * np.cos(angle)
        else:
            # Tọa độ chẵn (i = 2, 4, 6, ...) -> chỉ số mảng Python là i - 1 (1, 3, 5, ...)
            theta_continuous[:, i - 1] = radius_cubed * np.sin(angle)

    # Rời rạc hóa quỹ đạo theo bước nhảy ell: \hat{\theta}^*_t = \theta^*_{ceil(t / ell) * ell}
    if ell == 1:
        theta_discretized = theta_continuous.copy()
    else:
        theta_discretized = np.zeros_like(theta_continuous)
        for t_one_based in range(1, T + 1):
            # Kẹp chỉ số tối đa bằng T để không vượt quá biên mảng khi T không chia hết cho ell
            sampled_t = min(T, int(math.ceil(t_one_based / float(ell)) * ell))
            theta_discretized[t_one_based - 1] = theta_continuous[sampled_t - 1]

    # Tính chuỗi độ dịch chuyển \gamma_t = ||\hat{\theta}^*_t - \hat{\theta}^*_{t+1}||_2
    gamma_seq = compute_oracle_shifts(theta_discretized)
    return theta_discretized, gamma_seq


def compute_oracle_shifts(theta_star_path: np.ndarray) -> np.ndarray:
    """Tính chuỗi độ dịch chuyển phân phối \\gamma_t = ||\\theta^*_t - \\theta^*_{t+1}||_2.

    Theo Definition 2.1 (Công thức 6, trang 4):
        \\gamma_t đo khoảng cách Euclid mà nghiệm tối ưu Oracle dịch chuyển sau bước t.
    Tại bước cuối cùng t = T (chỉ số T - 1), đặt \\gamma_T = 0.0.

    Parameters
    ----------
    theta_star_path : np.ndarray
        Mảng quỹ đạo Oracle kích thước `(T, d)`.

    Returns
    -------
    np.ndarray
        Mảng 1 chiều kích thước `(T,)` trong đó phần tử `gamma[t-1]` ứng với \\gamma_t.
    """
    T = theta_star_path.shape[0]
    gamma = np.zeros(T, dtype=np.float64)
    if T > 1:
        # Hiệu giữa bước t và bước t+1
        diffs = theta_star_path[:-1] - theta_star_path[1:]
        gamma[:-1] = np.linalg.norm(diffs, ord=2, axis=1)
    return gamma


def generate_bursty_shifts(T: int, episode_length: int = 40, jump_size: float = 0.5) -> np.ndarray:
    """Sinh chuỗi độ trôi dạng nhảy đột ngột theo chu kỳ (Bursty Shifts - Figure 2 trái, Mục 3).

    Ở đầu mỗi episode (mỗi `episode_length` bước), \\gamma_t nhảy vọt lên giá trị `jump_size` (s),
    sau đó bằng 0 trong suốt phần còn lại của episode.

    Parameters
    ----------
    T : int
        Tổng số bước thời gian (Figure 2 dùng T = 200).
    episode_length : int, optional
        Độ dài mỗi episode tĩnh (Figure 2 dùng 40 bước).
    jump_size : float, optional
        Biên độ bước nhảy s tại đầu mỗi episode (Figure 2 xét s in {0, 0.1, 0.5, 2}).

    Returns
    -------
    np.ndarray
        Mảng kích thước `(T,)` chứa chuỗi \\gamma_t.
    """
    gamma = np.zeros(T, dtype=np.float64)
    for t in range(T):
        if t % episode_length == 0:
            gamma[t] = jump_size
    return gamma


def generate_smooth_shifts(T: int, alpha: float = 1.0) -> np.ndarray:
    """Sinh chuỗi độ trôi biến thiên liên tục \\gamma_t = 1 / t^alpha (Smooth Shifts - Figure 2 phải, Mục 3).

    Parameters
    ----------
    T : int
        Tổng số bước thời gian (Figure 2 dùng T = 200).
    alpha : float, optional
        Số mũ suy giảm \\alpha (Figure 2 xét \\alpha in {0.5, 1, 1.5, 2, inf}).
        Nếu `alpha = float('inf')`, trả về mảng toàn 0 (không có shift).

    Returns
    -------
    np.ndarray
        Mảng kích thước `(T,)` chứa \\gamma_t = 1 / t^alpha cho t = 1, ..., T.
    """
    if math.isinf(alpha):
        return np.zeros(T, dtype=np.float64)
    t_steps = np.arange(1, T + 1, dtype=np.float64)
    return np.power(t_steps, -alpha)


class EMADriftEstimator:
    """Bộ ước lượng trực tuyến độ dịch chuyển \\gamma_t bằng Trung bình Trượt Hàm mũ (EMA).

    Theo Remark 3.5 (trang 6) và Remark 4.4 (trang 7):
    Trong thực tế, bộ học (Learner) không biết trước nghiệm tối ưu \\theta^*_t của Tự nhiên,
    nên không có \\gamma_t chính xác. Bài báo đề xuất xấp xỉ \\gamma_t bằng trung bình trượt
    khoảng cách giữa hai trọng số liên tiếp mà SGD vừa cập nhật:
        \\hat{\\gamma}_t = \\beta * \\hat{\\gamma}_{t-1} + (1 - \\beta) * ||\\theta_t - \\theta_{t-1}||_2
    với hệ số làm mượt \\beta \\in (0, 1).
    """

    def __init__(self, beta: float = 0.9, init_gamma: float = 0.0) -> None:
        """Khởi tạo bộ ước lượng EMA.

        Parameters
        ----------
        beta : float, optional
            Hệ số nhớ \\beta \\in [0, 1) (mặc định là 0.9).
        init_gamma : float, optional
            Giá trị khởi tạo ban đầu \\hat{\\gamma}_0 (mặc định là 0.0).
        """
        if not (0.0 <= beta < 1.0):
            raise ValueError(f"Hệ số EMA beta phải nằm trong [0, 1), nhận được {beta}.")
        self.beta = float(beta)
        self.init_gamma = float(init_gamma)
        self.gamma_hat = float(init_gamma)
        self._prev_theta: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Đặt lại trạng thái bộ ước lượng về ban đầu trước mỗi lần chạy (run) mới."""
        self.gamma_hat = self.init_gamma
        self._prev_theta = None

    def update(self, theta_t: np.ndarray) -> float:
        """Cập nhật và trả về ước lượng \\hat{\\gamma}_t khi biết trọng số hiện tại `theta_t`.

        Parameters
        ----------
        theta_t : np.ndarray
            Vector trọng số mô hình tại bước t, kích thước `(d,)`.

        Returns
        -------
        float
            Giá trị ước lượng độ trôi \\hat{\\gamma}_t hiện tại.
        """
        theta_arr = np.asarray(theta_t, dtype=np.float64)
        if self._prev_theta is not None:
            # Tính độ dịch chuyển của chính mô hình học được: ||\theta_t - \theta_{t-1}||_2
            step_drift = float(np.linalg.norm(theta_arr - self._prev_theta, ord=2))
            self.gamma_hat = self.beta * self.gamma_hat + (1.0 - self.beta) * step_drift
        self._prev_theta = theta_arr.copy()
        return self.gamma_hat
