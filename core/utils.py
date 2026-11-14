"""
================================================================================
FILE: core/utils.py
MỤC ĐÍCH:
    Cung cấp các hàm tiện ích nền tảng dùng chung cho toàn bộ dự án (Task T6):
    1. `set_seed(seed)`: Cố định toàn bộ bộ sinh số ngẫu nhiên (Python `random`,
       `numpy`, `torch` CPU/GPU) để đảm bảo tính tái lập (Reproducibility) 100%
       theo yêu cầu của Lab 1.
    2. `load_config(path)` & `save_config(cfg, path)`: Đọc và lưu file cấu hình
       thí nghiệm định dạng YAML (trong thư mục `configs/`).
    3. `project_to_ball(theta, max_norm)`: Thực hiện phép chiếu Euclid \\Pi_\\Theta
       lên hình cầu đóng bán kính `max_norm` (theo Công thức 3, Mục 2 bài báo).

AI SỬ DỤNG FILE NÀY?
    - Cả 3 thành viên (TV1: Khoa, TV2: Khôi, TV3: Hòa) đều gọi `set_seed(seed)`
      ở đầu mỗi script thực nghiệm trong `experiments/` trước khi tạo dữ liệu
      hoặc khởi tạo mô hình.
    - Lưu ý tương thích với `core/tracking.py` (T2 của Khôi): Hàm `init_run(config)`
      trong `core/tracking.py` chỉ ghi nhận `seed` vào metadata W&B chứ không tự
      seed RNG, nên các script thực nghiệm sẽ gọi `set_seed(seed)` tại đây.
================================================================================
"""

import os
from pathlib import Path
import random
from typing import Any, Dict, Optional, Union

import numpy as np

# Import tùy chọn PyTorch và PyYAML để không gây lỗi nếu chạy ở môi trường tối giản
try:
    import torch
except ImportError:
    torch = None

try:
    import yaml
except ImportError:
    yaml = None


def set_seed(seed: int = 42) -> None:
    """Cố định random seed cho Python, NumPy và PyTorch nhằm đảm bảo tái lập kết quả.

    Mọi thí nghiệm trong Lab 1 cần chạy trên tối thiểu 3 seed cố định (ví dụ: 0, 1, 2
    hoặc 42, 43, 44) để tính trung bình và độ lệch chuẩn (mean ± std).

    Parameters
    ----------
    seed : int, optional
        Số nguyên không âm dùng làm hạt giống ngẫu nhiên (mặc định là 42).

    Returns
    -------
    None
    """
    # 1. Cố định thư viện random chuẩn của Python và biến môi trường hash
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2. Cố định bộ sinh số ngẫu nhiên toàn cục của NumPy
    np.random.seed(seed)

    # 3. Cố định bộ sinh số ngẫu nhiên của PyTorch (cho bài toán MLP Flow Cytometry của TV3)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Bật chế độ tính toán tất định (deterministic) trên cuDNN
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Đọc các siêu tham số (hyperparameters) của thí nghiệm từ file YAML.

    Parameters
    ----------
    config_path : Union[str, Path]
        Đường dẫn tới file cấu hình `.yaml` (ví dụ: `configs/linear.yaml`).

    Returns
    -------
    Dict[str, Any]
        Dictionary chứa toàn bộ cấu hình đã đọc. Nếu file rỗng trả về `{}`.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình tại: {path}")
    if yaml is None:
        raise ImportError("Cần cài đặt thư viện PyYAML (`pip install pyyaml`) để đọc file config.")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def save_config(config: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """Lưu dictionary siêu tham số của thí nghiệm ra file định dạng YAML.

    Parameters
    ----------
    config : Dict[str, Any]
        Dictionary chứa cấu hình cần lưu.
    output_path : Union[str, Path]
        Đường dẫn file `.yaml` đích (tự động tạo thư mục cha nếu chưa có).

    Returns
    -------
    None
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is None:
        raise ImportError("Cần cài đặt thư viện PyYAML (`pip install pyyaml`) để lưu file config.")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


def project_to_ball(theta: np.ndarray, max_norm: Optional[float] = None) -> np.ndarray:
    """Chiếu vector trọng số `theta` lên hình cầu Euclid đóng bán kính `max_norm`.

    Thực hiện toán tử chiếu \\Pi_\\Theta(u) trong Công thức (3) của bài báo:
        \\theta_{t+1} = \\Pi_\\Theta(\\theta_t - \\eta_t \\nabla \\ell^B_t(\\theta_t))
    Trong đó \\Theta \\subseteq R^p là tập lồi bị chặn có đường kính D_max.
    - Với Linear Regression (Mục 3 & 6.1): không cần chiếu (`max_norm=None`).
    - Với Convex / Logistic Regression (Mục 4 & 6.1): bài báo đặt D_max = d,
      nên có thể truyền `max_norm = d` để đảm bảo giả thiết Assumption 4.1(iii).

    Parameters
    ----------
    theta : np.ndarray
        Vector tham số mô hình hiện tại, kích thước `(d,)`.
    max_norm : Optional[float], optional
        Bán kính tối đa cho phép của chuẩn Euclid ||theta||_2. Nếu `None` hoặc <= 0
        thì giữ nguyên không chiếu.

    Returns
    -------
    np.ndarray
        Vector tham số sau khi đã chiếu vào trong hình cầu bán kính `max_norm`.
    """
    if max_norm is None or max_norm <= 0:
        return theta
    norm = float(np.linalg.norm(theta, ord=2))
    # Nếu vector vượt ra ngoài hình cầu bán kính max_norm, co nó về đúng biên
    if norm > max_norm:
        return theta * (max_norm / norm)
    return theta
