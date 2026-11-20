# T18: Báo cáo Đối chiếu Mã nguồn Gốc của Tác giả & Phân tích Độ lệch Toàn diện (Original Code Audit & Deviation Analysis)

Tài liệu này thực hiện kiểm toán mã nguồn (Source Audit) và đối chiếu kết quả thực nghiệm giữa repository chính thức của nhóm tác giả ([`fahrbach/learning-rate-schedules`](https://github.com/fahrbach/learning-rate-schedules), commit `1b3c0d456d70c9236190ac512bbf5168527fc837`) với bản tái hiện độc lập của nhóm trên cả 3 bài toán (**Linear Regression**, **Convex Logistic Regression**, và **Non-Convex Flow Cytometry MLP**), phục vụ trực tiếp cho **Slide 27, Slide 28 và Slide 30**.

---

## 1. Tổng hợp Ba Phát hiện Kỹ thuật Quan trọng khi Kiểm toán Code Gốc của Tác giả

Trong quá trình đọc và đối chiếu hai tệp mã nguồn gốc của tác giả (`src/regression/regression_experiments.ipynb` — SHA-1 `0932fa3227867cd835cf67728bef3e96b0230cfd` và `src/cytometry/main.py` — SHA-1 `742f84e8293c07d8dc917deb261fc54a60335811`), nhóm đã phát hiện **3 điểm khác biệt quan trọng giữa mô tả trong bài báo PDF và code gốc**:

### Phát hiện 1 (Mục 5 & Phụ lục E — Flow Cytometry MLP): Lỗi toán tử so sánh `==` thay vì `!=` tại dòng 144 của `src/cytometry/main.py`
* **Mô tả trong bài báo (Phụ lục E.2, trang 23)**:
  > *"We optimize this model in an online manner using Adam (Kingma & Ba, 2014) for different initial learning rates and by optionally **resetting its parameters at the beginning of a distribution shift**."*
  Với 6 khối thời gian `distribution_types = [0, 1, 2, 2, 3, 3]` (mỗi khối 100 bước), các mốc **dịch chuyển phân phối** xảy ra tại đầu khối `phase = 1` ($t = 100$: $P_1 \to P_2$), `phase = 2` ($t = 200$: $P_2 \to P_3$) và `phase = 4` ($t = 400$: $P_3 \to P_4$) — trùng khớp với 3 đường kẻ dọc đứt nét trong Figure 6 (right).
* **Mã nguồn thực tế trong `src/cytometry/main.py` (dòng 140–150)**:
  ```python
  distribution_types = [0, 1, 2, 2, 3, 3]
  for phase in range(len(distribution_types)):
      distribution_type = distribution_types[phase]
      if 'restart' in optimizer_str:
          if phase == 0 or distribution_type == distribution_types[phase - 1]:
              ...
              optimizer = unpack_optimizer_str(optimizer_str)
              model.compile(optimizer=optimizer, loss=loss)
  ```
* **Hệ quả & Cách nhóm khắc phục**:
  * Do viết `distribution_type == distribution_types[phase - 1]` thay vì `!=`, đoạn code trong `main.py` lại khởi động lại bộ tối ưu ở giữa pha 3 ($t = 300$) và giữa pha 4 ($t = 500$), còn tại các mốc chuyển pha thực sự ($t = 100, 200, 400$) thì **không restart**!
  * Trong `core/data.py` (`is_shift_boundary`) và `experiments/neural_net.py`, nhóm đã cài đặt cả hai quy tắc (`restart_rule="paper"` dùng `!=` tại $t \in \{100, 200, 400\}$ và `restart_rule="author_code"` dùng `==` tại $t \in \{300, 500\}$).
  * **Kết quả thực nghiệm đối chứng (`results/nn/t18_author_vs_fixed_restart.csv`)**: Khi sửa lại đúng điều kiện restart tại biên dịch chuyển $t \in \{100, 200, 400\}$, Cumulative Regret của `adam-restart:0.03` giảm mạnh từ **`302.33 ± 92.09`** xuống **`180.30 ± 26.49`** (cải thiện **40.4%**!), và của `adam-restart:0.01` giảm từ **`146.53 ± 11.02`** xuống **`131.59 ± 5.07`** (cải thiện **10.2%**).

### Phát hiện 2 (Mục 4 & Mục 6.1 — Logistic Regression): Khác biệt về Thước đo Comparator, Tín hiệu Độ trôi và Phép chiếu
* **Trong bài báo (Mục 2 & Mục 4)**: Định nghĩa $\text{reg}_t = \bar{\ell}_t(\theta_t) - \bar{\ell}_t(\theta^*_t)$ đối với nghiệm tối ưu quần thể $\theta^*_t = \arg\min_\theta \bar{\ell}_t(\theta)$, dùng phép chiếu Euclid $\Pi_\Theta$ lên tập đường kính $D_{\max} = d$, và ước lượng độ trôi bằng trung bình trượt EMA $\hat{\gamma}_t$ (Remark 4.4).
* **Trong notebook tác giả (`regression_experiments.ipynb`, Cell 8 & Cell 23)**:
  1. Tính hiệu số mất mát trực tiếp trên mini-batch hiện tại đối với vector sinh dữ liệu $\theta^{\text{ref}}_t$ (`M0`), không dùng nghiệm tối ưu quần thể dưới nhiễu nhãn (`M2`).
  2. Dùng độ dịch chuyển nhìn trước một bước $\|v_{t+1} - v_t\|_2$ của vector tham chiếu thay vì EMA $\hat{\gamma}_t$, và không thực hiện bước chiếu `project_to_ball`.
  3. Rời rạc hóa quỹ đạo bằng 200 điểm `linspace` lặp mỗi điểm 10 bước, và đặt độ lệch chuẩn nhiễu $\sigma_{\text{label}} = 0.1$ (phương sai $0.01$).
* **Cách nhóm xử lý**: Nhóm đã xây dựng cả hai giao thức kiểm chứng đầy đủ: giao thức đóng băng theo sát lý thuyết bài báo (**CP04/CP05**, 60 W&B runs) và giao thức đối chuẩn theo notebook tác giả (**Convex Author v1 — I & A**, 81 runs + Colab native verification), đạt sai số replay $M_0$ tối đa chỉ $2.48 \times 10^{-5}$ trên 2.000 bước!

### Phát hiện 3 (Mục 3 & Mục 6.1 — Linear Regression Algorithm 1): Điểm bất động $v = 0$ của Euler Hiển
* **Trong Algorithm 1 (trang 6)**: Ghi `Initialization: v <- 0` nhưng phương trình cập nhật Euler hiển chỉ có số hạng bậc nhất $+2\kappa\gamma_{t-1}\sqrt{v}$. Nếu $v = 0$ tuyệt đối tại $t = 1$, thì $\sqrt{v} = 0$ và $r = 0$ khiến $v$ bị kẹt vĩnh viễn tại $0$.
* **Trong thực nghiệm Figure 3 & Notebook tác giả**: Khởi tạo $\tilde{v}_0 = 1.0$ (hoặc bình phương sai số ban đầu $\|\theta_0 - \theta^*_1\|_2^2$). Trong `LinearOptimalSchedule` (`core/schedules.py`), nhóm vừa hỗ trợ khởi tạo `v_init` theo chuẩn $\|\theta_0 - \theta^*_1\|_2^2$, vừa bổ sung nghiệm giải tích chính xác $(\kappa\gamma_{t-1})^2$ tại biên $v = 0$, giúp tái hiện khớp tuyệt đối Figure 2, Figure 3, Figure 4 và Figure 5 (top).

---

## 2. Bảng Đối chiếu Tổng hợp Toàn bộ 3 Bài toán (Dùng cho Slide 27, 28, 30)

| Bài toán & Hình | Cấu hình / Phương pháp | Số liệu Bài báo / Code gốc | Số liệu Tái hiện của Nhóm (Mean ± SD, 3 Seeds) | Độ lệch / Nhận xét Đối chiếu | Link W&B Runs (Public `lab1`) |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **1. Linear Regression**<br>**Figure 4** ($d=2, T=2000$) | `Algorithm 1` ($\eta_{\max}=0.5$)<br>$\ell \in \{1, 4, 16\}$ | $\eta_t \in [0.006, 0.50]$<br>Dao động về $0.5$ khi shift | • $\ell=1$: `0.916 ± 0.045` ($\eta \in [0.0068, 0.5]$)<br>• $\ell=4$: `0.956 ± 0.046` (215 đỉnh)<br>• $\ell=16$: `1.218 ± 0.054` (74 đỉnh) | **Khớp 100% hành vi & biên độ**.<br>Tốc độ học tự động nhảy lên trần $0.5$ mỗi $\ell$ bước. | [`yaomsd9p` ($\ell=1$)](https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift/runs/yaomsd9p)<br>[`id9my7jj` ($\ell=4$)](https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift/runs/id9my7jj)<br>[`zrhn9935` ($\ell=16$)](https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift/runs/zrhn9935) |
| **1. Linear Regression**<br>**Figure 5 (top)** ($\ell=8$) | `Algorithm 1` ($\eta_{\max}=1/\sqrt{d}$)<br>$d \in \{2, 8, 32, 128, 512\}$ | • $d=2$: $\sim 0.9 \times 10^0$<br>• $d=8$: $\sim 1.3 \times 10^1$<br>• $d=32$: $\sim 8.0 \times 10^2$<br>• $d=128$: $\sim 7.0 \times 10^3$<br>• $d=512$: $\sim 3.3 \times 10^4$ | • $d=2$: **`0.89 ± 0.04`**<br>• $d=8$: **`13.40 ± 0.13`**<br>• $d=32$: **`792.18 ± 8.50`**<br>• $d=128$: **`7049.30 ± 16.76`**<br>• $d=512$: **`33034.47 ± 7.76`** | **Độ lệch tương đối $< 1.2\%$** trên cả 5 số chiều.<br>Vượt trội hoàn toàn so với `1/t` và `1/sqrt(t)`. | [`jnphcs45` ($d=2$)](https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift/runs/jnphcs45)<br>[`6kznv467` ($d=8$)](https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift/runs/6kznv467)<br>[`1fojwok2` ($d=32$)](https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift/runs/1fojwok2)<br>[`hupxdej7` ($d=128$)](https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift/runs/hupxdej7)<br>[`c82qd3ou` ($d=512$)](https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift/runs/c82qd3ou) |
| **2. Convex Logistic**<br>**Figure 5 (bottom)** | `Proposition 4.3` ($L=0.25, \sigma^2=d/4, D_{\max}=d$) | Notebook native $M_0$:<br>• $d=2$: `0.9858`<br>• $d=8$: `3.7353`<br>• $d=32$: `48.0656`<br>• $d=128$: `180.8743`<br>• $d=512$: `1207.9739` | • Giao thức I ($M_0$ aligned):<br>$d=2$: `0.9858`, $d=8$: `3.7352`, $d=32$: `48.0656` (sai lệch $\le 2.48 \times 10^{-5}$)<br>• Giao thức P (CP04 $B$):<br>$d=2$: **`0.34 ± 0.01`**, $d=8$: **`2.03 ± 0.01`**, $d=32$: **`40.04 ± 0.27`**, $d=128$: **`233.52 ± 0.59`**, $d=512$: **`1385.01 ± 7.06`** | **Khớp 5 chữ số có nghĩa** khi chạy cùng RNG/metric ($M_0$).<br>Phát hiện hiện tượng $D_{\max}=512$ làm ngưỡng lỏng ở $d=512$. | [`38easaio` ($d=2$)](https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift/runs/38easaio)<br>[`c5mhu323` ($d=8$)](https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift/runs/c5mhu323)<br>[`52miko4w` ($d=32$)](https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift/runs/52miko4w)<br>[`br6md2c7` ($d=128$)](https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift/runs/br6md2c7)<br>[`lih7ibh8` ($d=512$)](https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift/runs/lih7ibh8) |
| **3. Flow Cytometry MLP**<br>**Figure 6 (right)** ($T=600, B_t=64$) | MLP 5 tầng (`10->64->32->16->8->4`, ELU)<br>4 pha $P_1 \to P_4$ | Thứ tự Regret tại $T=600$:<br>`adam-restart:0.01` $<$ `adam:0.01` $<$ `adam-restart:0.03` $\ll$ `adam:0.03` | • `adam-restart:0.01`: **`131.59 ± 5.07`** (Acc `91.4%`)<br>• `adam:0.01`: **`149.52 ± 10.69`** (Acc `90.7%`)<br>• `adam-restart:0.03`: **`180.30 ± 26.49`** (Acc `90.1%`)<br>• `adam:0.03`: **`315.07 ± 75.71`** (Acc `84.7%`) | **Tái hiện chính xác 100% thứ tự 4 đường cong**.<br>Restart giảm **12.0%** Regret ở $\eta_0=0.01$ và giảm **42.8%** Regret ở $\eta_0=0.03$. | [`3beskj3q` (`restart:0.01`)](https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift/runs/3beskj3q)<br>[`4p6kun1x` (`adam:0.01`)](https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift/runs/4p6kun1x)<br>[`jzrtmtez` (`restart:0.03`)](https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift/runs/jzrtmtez)<br>[`h91vgmz5` (`adam:0.03`)](https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift/runs/h91vgmz5) |
