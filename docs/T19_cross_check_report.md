# T19: Báo cáo Kiểm tra Chéo Độc lập (Cross-Verification Report) — TV1 (Khoa) kiểm chứng TV2 (Khôi)

Tài liệu này ghi nhận kết quả kiểm tra chéo độc lập của **TV1 (Khoa)** đối với toàn bộ mã nguồn, bộ kiểm thử và dữ liệu thực nghiệm của **TV2 (Khôi)** theo yêu cầu của **Task T19 (Hạn 18/11/2026)**.

---

## 1. Phạm vi Kiểm tra Chéo

1. **Mã nguồn & Giao diện dùng chung (`core/` & `scripts/`)**:
   - `core/tracking.py`: Quản lý khởi tạo W&B run (`init_run`), kiểm soát JSON-safe metadata, khử trùng lặp tags `lab1`, chặn rò rỉ thông tin nhạy cảm (secret keys).
   - `core/schedules.py`: Lớp `ConvexSchedule` cài đặt chính xác **Proposition 4.3 (Công thức 18–20)** dùng biến đổi hữu tỷ hóa chống tràn số học (`_logadd`, `_tau`).
   - `core/convex_eval.py` & `core/convex_population.py`: Hàm mất mát Binary Cross-Entropy ổn định dấu, bộ đánh giá Monte Carlo có kiểm soát sai số chuẩn (SE), và bộ tích phân cầu phương Gaussian 1 chiều (`scipy.integrate.quad` + `brentq`) với dung sai $10^{-11}$.
2. **Thực nghiệm T15 (Figure 5-bottom — Logistic Regression) & T17 (Bonus Sensitivity Analysis)**:
   - Đối chiếu kết quả chạy lại trực tiếp trên máy của TV1 (Khoa) với bảng số liệu đóng băng `docs/results/cp04-cp05/data/cp04_per_seed.csv` (45 runs chính thức) và `cp05_condition_summary.csv` (15 runs độ nhạy).

---

## 2. Kết quả Chạy Kiểm thử & Đối chiếu Số học

| Hạng mục Kiểm tra | Công cụ / Script thực thi | Kết quả Nghiệm thu | Sai số Tuyệt đối Tối đa ($\max |\Delta|$) |
| :--- | :--- | :--- | :--- |
| **Unit Tests Tracking (T2)** | `python -m unittest discover -s tests -p "test_tracking*.py"` | **PASS (100%)** | $0.0$ |
| **Unit Tests Convex & Prop. 4.3 (T9)** | `python -m unittest tests/test_convex.py -v` | **PASS (100%)** | $< 10^{-11}$ |
| **Unit Tests Linear & Alg. 1 (T7/T14)** | `python -m unittest tests/test_linear.py -v` | **PASS (100%)** | $< 10^{-15}$ |
| **Tái lập trực tiếp CP04 ($d=2, 8$, seed $0$)** | `python experiments/convex.py` | **KHỚP TUYỆT ĐỐI** | $\le 1.11 \times 10^{-16}$ (cấp độ chấm động IEEE 754) |
| **Xuất biểu đồ Độ nhạy CP05 (T17)** | `python experiments/sensitivity.py` | **PASS (100%)** | Khớp 100% với `cp05_condition_summary.csv` |

### Bảng Đối chiếu Chi tiết Tái lập Độc lập tại $d = 2$, Seed $0$ (3 Phương pháp)

| Phương pháp | Chỉ số đối chiếu | Giá trị đóng băng của TV2 (`cp04_per_seed.csv`) | Giá trị TV1 chạy lại độc lập | Độ lệch tuyệt đối | Link W&B Run (Public) |
| :--- | :--- | ---: | ---: | ---: | :--- |
| **`convex` (Prop. 4.3)** | Final Population $B$ | `0.333866998129` | `0.333866998129` | `< 1e-15` | [run 38easaio](https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift/runs/38easaio) |
| **`constant` ($\eta=0.1$)** | Final Population $B$ | `7.248281085797` | `7.248281085797` | `< 1e-15` | [run 76y9bplv](https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift/runs/76y9bplv) |
| **`inverse_sqrt` ($\eta_0=0.1$)** | Final Population $B$ | `29.520155304493` | `29.520155304493` | `< 1e-15` | [run fr6ts7n5](https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift/runs/fr6ts7n5) |

---

## 3. Kết luận Đóng băng Kết quả Thực nghiệm 1 (TV1) và Thực nghiệm 2 + Độ nhạy (TV2)

1. **Thực nghiệm 1 — Linear Regression (TV1 - Khoa)**:
   - Đã hoàn thành toàn bộ mã nguồn `LinearOptimalSchedule` (**T7**) và `experiments/linear_regression.py` (**T14**).
   - Đã xác thực quyền ghi W&B của tài khoản `nkhoa8508` qua smoke run ([run clxjezxq](https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift/runs/clxjezxq)) và đẩy thành công toàn bộ **18 runs chính thức** lên W&B Public:
     - **Figure 4 ($\ell \in \{1, 4, 16\}$)**: [yaomsd9p ($\ell=1$)](https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift/runs/yaomsd9p), [id9my7jj ($\ell=4$)](https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift/runs/id9my7jj), [zrhn9935 ($\ell=16$)](https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift/runs/zrhn9935).
     - **Figure 5-top ($d \in \{2, 8, 32, 128, 512\}$, 3 seeds)**: [jnphcs45 ($d=2$)](https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift/runs/jnphcs45), [6kznv467 ($d=8$)](https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift/runs/6kznv467), [1fojwok2 ($d=32$)](https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift/runs/1fojwok2), [hupxdej7 ($d=128$)](https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift/runs/hupxdej7), [c82qd3ou ($d=512$)](https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift/runs/c82qd3ou) (chi tiết 15 URLs tại `results/linear/fig5_top_per_seed.csv`).
   - Tái hiện thành công **Figure 4** và **Figure 5 (top)**, chứng minh `Algorithm 1` đạt Cumulative Regret thấp vượt trội so với các lịch trình suy giảm cổ điển (`1/t`, `1/sqrt(t)`) và bám sát quỹ đạo xoắn ốc ngay cả ở không gian $d = 128, 512$.
2. **Thực nghiệm 2 — Convex / Logistic Regression & Độ nhạy (TV2 - Khôi)**:
   - Đã được TV1 chạy kiểm tra chéo độc lập (**T19**), xác nhận khả năng tái lập chính xác tuyệt đối tới cấp độ máy ($10^{-16}$) và đầy đủ 60 W&B runs public.
3. **Các mục còn chờ hoàn tất từ các thành viên khác**:
   - **TV2 (Khôi)**: Bổ sung các file tài liệu đọc hiểu chung Milestone 2 (`docs/02_contributions.md`, `docs/06_complexity.md`, `docs/theorems.md`, `docs/references.md`).
   - **TV1 (Khoa — thực hiện thay phần việc TV3 Hòa không hoàn thành)**: Đã hoàn thiện toàn bộ code `core/data.py` (**T11**), MLP 5 tầng + Adam Restart (**T12**), `experiments/neural_net.py` (**T16**) và đối chiếu code gốc (**T18**).
