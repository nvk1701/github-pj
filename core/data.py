r"""
================================================================================
FILE: core/data.py
MỤC ĐÍCH:
    Tải dữ liệu biểu hiện RNA đơn bào tuyến tụy chuột (Murine Pancreas Single-Cell
    RNA-seq, Bastidas-Ponce et al., 2019 / `scvelo.datasets.pancreas()`) và mô phỏng
    luồng dữ liệu Phân tích Tế bào Dòng chảy Trực tuyến (Online Flow Cytometry Stream)
    qua 4 pha dịch chuyển phân phối P_1 -> P_2 -> P_3 -> P_4 (Task T11 & T16):

    1. 10 Marker Genes (lọc bởi `min_shared_counts=30000` và `normalize_per_cell`):
       `Pyy, Meg3, Malat1, Gcg, Gnas, Actb, Ghrl, Rps3, Ins2, Hspa8`.
    2. 4 Phân lớp tế bào mục tiêu (K = 4):
       - `Alpha`  (481 tế bào)
       - `Beta`   (591 tế bào)
       - `Delta`  ( 70 tế bào)
       - `Ductal` (916 tế bào)
       (Tổng 2.058 tế bào thuộc 4 lớp mục tiêu trên tổng số 3.696 tế bào tuyến tụy).
    3. Nhiễu nhân ngẫu nhiên (Multiplicative Perturbation - Phụ lục E.2, trang 23):
       Mỗi đặc trưng biểu hiện gene x_j được nhân với U_j ~ Uniform[0.9, 1.1] độc lập.
    4. Bốn chế độ phân phối hỗn hợp (Mixture Distributions) qua T = 600 bước (B_t = 64):
       Theo vector xác suất (p_Alpha, p_Beta, p_Delta, p_Ductal):
       - Pha 1 (bước 1..100):   P_1(y) = (0.0, 0.0, 0.0, 1.0) — 100% Ductal (không tiết)
       - Pha 2 (bước 101..200): P_2(y) = (0.0, 0.0, 0.1, 0.9) — 10% Delta, 90% Ductal
       - Pha 3 (bước 201..400): P_3(y) = (0.1, 0.0, 0.2, 0.7) — 10% Alpha, 20% Delta, 70% Ductal
       - Pha 4 (bước 401..600): P_4(y) = (0.3, 0.5, 0.1, 0.1) — 30% Alpha, 50% Beta, 10% Delta, 10% Ductal
================================================================================
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

PANCREAS_10_GENES: Tuple[str, ...] = (
    "Pyy",
    "Meg3",
    "Malat1",
    "Gcg",
    "Gnas",
    "Actb",
    "Ghrl",
    "Rps3",
    "Ins2",
    "Hspa8",
)

CELL_CLASSES: Tuple[str, ...] = ("Alpha", "Beta", "Delta", "Ductal")

# Bốn phân phối hỗn hợp P_1, P_2, P_3, P_4 trên (Alpha, Beta, Delta, Ductal) theo Phụ lục E.2 & main.py
PHASE_MIXTURES: Dict[int, Tuple[float, float, float, float]] = {
    0: (0.0, 0.0, 0.0, 1.0),  # P_1: 100% Ductal
    1: (0.0, 0.0, 0.1, 0.9),  # P_2: 10% Delta, 90% Ductal
    2: (0.1, 0.0, 0.2, 0.7),  # P_3: 10% Alpha, 20% Delta, 70% Ductal
    3: (0.3, 0.5, 0.1, 0.1),  # P_4: 30% Alpha, 50% Beta, 10% Delta, 10% Ductal
}

# Lịch trình 6 khối x 100 bước = 600 bước: [P_1 (100), P_2 (100), P_3 (200), P_4 (200)]
DEFAULT_BLOCK_TYPES: Tuple[int, ...] = (0, 1, 2, 2, 3, 3)


@dataclass
class PancreasDataset:
    """Cấu trúc lưu trữ dữ liệu 10-gene và tọa độ nhúng 2D UMAP của 3.696 tế bào tuyến tụy."""

    X_norm_10: np.ndarray
    X_raw_10: np.ndarray
    gene_names: Tuple[str, ...]
    cluster_codes: np.ndarray
    cluster_names: Tuple[str, ...]
    cluster_colors: Tuple[str, ...]
    X_umap: np.ndarray
    class_pools: Dict[str, np.ndarray]


def load_pancreas_dataset(npz_path: Optional[Path] = None) -> PancreasDataset:
    """Tải bộ dữ liệu tế bào tuyến tụy 10-gene đã chuẩn hóa theo đúng quy trình `scVelo`.

    Ưu tiên đọc từ file nén chuẩn `core/pancreas_10genes.npz` (trích xuất trực tiếp từ
    `endocrinogenesis_day15.h5ad` của `scvelo.datasets.pancreas()` sau khi chạy
    `filter_genes(min_shared_counts=30000)` và `normalize_per_cell()`). Nếu chưa có file
    nén và môi trường có cài `scvelo`, hàm sẽ tự động gọi `scvelo.datasets.pancreas()`.

    Parameters
    ----------
    npz_path : Optional[Path], optional
        Đường dẫn tới file `.npz`. Mặc định là `core/pancreas_10genes.npz`.

    Returns
    -------
    PancreasDataset
        Đối tượng chứa ma trận đặc trưng 10-gene, nhãn cụm, tọa độ UMAP và pool 4 lớp.
    """
    default_path = Path(__file__).resolve().parent / "pancreas_10genes.npz"
    target_path = Path(npz_path) if npz_path is not None else default_path

    if not target_path.exists():
        try:
            import scvelo as scv  # type: ignore[import-not-found]
        except ImportError as exc:
            raise FileNotFoundError(
                f"Không tìm thấy {target_path} và môi trường hiện tại chưa cài scvelo."
            ) from exc

        adata = scv.datasets.pancreas()
        scv.pp.filter_genes(adata, min_shared_counts=30000)
        X_raw = adata.X.toarray().astype(np.float32)
        scv.pp.normalize_per_cell(adata)
        X_norm = adata.X.toarray().astype(np.float32)
        g_names = tuple(str(g) for g in adata.var_names)
        c_series = adata.obs["clusters"].astype("category")
        c_names = tuple(str(c) for c in c_series.cat.categories)
        c_codes = c_series.cat.codes.to_numpy(dtype=np.int8)
        c_colors = tuple(
            str(c)
            for c in adata.uns.get(
                "clusters_colors",
                ["#8fbc8f", "#f4a460", "#fdbf6f", "#ff7f00", "#b2df8a", "#1f78b4", "#6a3d9a", "#cab2d6"],
            )
        )
        umap_coords = np.asarray(adata.obsm["X_umap"], dtype=np.float32)
    else:
        with np.load(target_path, allow_pickle=False) as data:
            X_norm = np.asarray(data["X_norm_10"], dtype=np.float32)
            X_raw = np.asarray(data["X_raw_10"], dtype=np.float32)
            g_names = tuple(str(g) for g in data["gene_names"].tolist())
            c_codes = np.asarray(data["cluster_codes"], dtype=np.int8)
            c_names = tuple(str(c) for c in data["cluster_names"].tolist())
            c_colors = tuple(str(c) for c in data["cluster_colors"].tolist())
            umap_coords = np.asarray(data["X_umap"], dtype=np.float32)

    cluster_labels = np.array([c_names[idx] for idx in c_codes])
    pools: Dict[str, np.ndarray] = {}
    for cls_name in CELL_CLASSES:
        mask = cluster_labels == cls_name
        if not np.any(mask):
            raise ValueError(f"Không tìm thấy tế bào nào thuộc phân lớp '{cls_name}'.")
        pools[cls_name] = X_norm[mask].copy()

    return PancreasDataset(
        X_norm_10=X_norm,
        X_raw_10=X_raw,
        gene_names=g_names,
        cluster_codes=c_codes,
        cluster_names=c_names,
        cluster_colors=c_colors,
        X_umap=umap_coords,
        class_pools=pools,
    )


class PancreasCytometryStream:
    """Bộ sinh luồng dữ liệu Flow Cytometry qua 4 pha phân phối P_1 -> P_2 -> P_3 -> P_4."""

    def __init__(
        self,
        dataset: Optional[PancreasDataset] = None,
        batch_size: int = 64,
        perturb_eps: float = 0.1,
        steps_per_block: int = 100,
        block_types: Sequence[int] = DEFAULT_BLOCK_TYPES,
    ) -> None:
        """Khởi tạo luồng dữ liệu tế bào dòng chảy.

        Parameters
        ----------
        dataset : Optional[PancreasDataset], optional
            Bộ dữ liệu tế bào tụy đã tải. Nếu `None`, tự động gọi `load_pancreas_dataset()`.
        batch_size : int, optional
            Kích thước mini-batch B_t tại mỗi bước (mặc định: 64 theo Phụ lục E.2).
        perturb_eps : float, optional
            Biên độ nhiễu nhân đồng nhất U ~ Uniform[1 - eps, 1 + eps] (mặc định: 0.1).
        steps_per_block : int, optional
            Số bước SGD trong mỗi khối thời gian (mặc định: 100 bước/khối).
        block_types : Sequence[int], optional
            Chuỗi chỉ số phân phối của các khối (mặc định: `(0, 1, 2, 2, 3, 3)` ứng với
            100 bước P_1, 100 bước P_2, 200 bước P_3, 200 bước P_4 => tổng T = 600 bước).
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size phải > 0, nhận được {batch_size}.")
        if not (0.0 <= perturb_eps < 1.0):
            raise ValueError(f"perturb_eps phải nằm trong [0, 1), nhận được {perturb_eps}.")
        if steps_per_block <= 0 or len(block_types) == 0:
            raise ValueError("steps_per_block và block_types phải khác rỗng.")

        self.dataset = dataset if dataset is not None else load_pancreas_dataset()
        self.batch_size = int(batch_size)
        self.perturb_eps = float(perturb_eps)
        self.steps_per_block = int(steps_per_block)
        self.block_types = tuple(int(b) for b in block_types)
        self.total_steps = self.steps_per_block * len(self.block_types)

    def distribution_type_at(self, step_idx: int) -> int:
        """Trả về mã phân phối (0, 1, 2, hoặc 3) tại bước `step_idx` (đánh số từ 0 đến T-1)."""
        if not (0 <= step_idx < self.total_steps):
            raise IndexError(f"step_idx={step_idx} nằm ngoài [0, {self.total_steps - 1}].")
        block_idx = step_idx // self.steps_per_block
        return self.block_types[block_idx]

    def mixture_probs_at(self, step_idx: int) -> Tuple[float, float, float, float]:
        """Trả về vector xác suất pha trộn (p_Alpha, p_Beta, p_Delta, p_Ductal) tại bước `step_idx`."""
        dist_type = self.distribution_type_at(step_idx)
        return PHASE_MIXTURES[dist_type]

    def is_shift_boundary(self, step_idx: int, restart_rule: str = "paper") -> bool:
        """Kiểm tra xem bước `step_idx` (0-based) có phải là mốc khởi động lại (Restart) hay không.

        Parameters
        ----------
        step_idx : int
            Chỉ số bước thời gian 0-based (0 <= step_idx < T).
        restart_rule : str, optional
            - `"paper"`: Khởi động lại đúng tại đầu mỗi đợt dịch chuyển phân phối thực sự
              (`dist[phase] != dist[phase - 1]`), tức là tại `step_idx in {100, 200, 400}`
              theo đúng mô tả ở Phụ lục E.2 và vạch kẻ dọc trong Figure 6 (right).
            - `"author_code"`: Mô phỏng chính xác câu lệnh điều kiện trong `src/cytometry/main.py`
              dòng 144 của tác giả (`phase == 0 or dist[phase] == dist[phase - 1]`), tức là
              restart tại `step_idx in {300, 500}` (dùng để đối chiếu Task T18).

        Returns
        -------
        bool
            `True` nếu cần khởi động lại trạng thái bộ tối ưu tại đầu bước `step_idx`.
        """
        if step_idx <= 0 or (step_idx % self.steps_per_block) != 0:
            return False
        phase = step_idx // self.steps_per_block
        curr_dist = self.block_types[phase]
        prev_dist = self.block_types[phase - 1]

        if restart_rule == "paper":
            return curr_dist != prev_dist
        if restart_rule == "author_code":
            return curr_dist == prev_dist
        raise ValueError(f"restart_rule='{restart_rule}' không hợp lệ (chọn 'paper' hoặc 'author_code').")

    def sample_batch(
        self,
        step_idx: int,
        rng: np.random.Generator,
        num_samples: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sinh một batch mẫu `(X, y)` tại bước `step_idx` với nhiễu nhân U ~ Uniform[1-eps, 1+eps].

        Parameters
        ----------
        step_idx : int
            Chỉ số bước thời gian 0-based (0 <= step_idx < T).
        rng : np.random.Generator
            Bộ sinh số ngẫu nhiên NumPy.
        num_samples : Optional[int], optional
            Số lượng mẫu cần sinh. Mặc định bằng `self.batch_size` (64).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - `X`: Ma trận biểu hiện 10 gene kích thước `(N, 10)` kiểu `float32`.
            - `y`: Vector nhãn lớp nguyên `{0, 1, 2, 3}` kích thước `(N,)` kiểu `int64`
              (tương ứng với `0: Alpha, 1: Beta, 2: Delta, 3: Ductal`).
        """
        n = int(num_samples) if num_samples is not None else self.batch_size
        probs = np.asarray(self.mixture_probs_at(step_idx), dtype=np.float64)

        # Chọn ngẫu nhiên phân lớp cho từng tế bào theo phân phối hỗn hợp P_t
        class_indices = rng.choice(len(CELL_CLASSES), size=n, replace=True, p=probs)
        X = np.empty((n, len(PANCREAS_10_GENES)), dtype=np.float32)

        for cls_id, cls_name in enumerate(CELL_CLASSES):
            positions = np.flatnonzero(class_indices == cls_id)
            if positions.size == 0:
                continue
            pool = self.dataset.class_pools[cls_name]
            chosen_rows = rng.integers(0, pool.shape[0], size=positions.size)
            base_samples = pool[chosen_rows]
            noise_factors = rng.uniform(
                low=1.0 - self.perturb_eps,
                high=1.0 + self.perturb_eps,
                size=base_samples.shape,
            ).astype(np.float32)
            X[positions] = base_samples * noise_factors

        return X, class_indices.astype(np.int64)
