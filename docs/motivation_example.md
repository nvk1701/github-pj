# T3: Thống nhất Phạm vi Thực nghiệm (Full Scope) & Ví dụ Thực tế (Motivation)

## 1. Xác nhận Phạm vi Thực nghiệm: FULL SCOPE (100%)

Nhóm thống nhất lựa chọn **Full Scope** — tái hiện trọn vẹn 100% cả 3 bài toán thực nghiệm chính trong bài báo *"Learning Rate Schedules in the Presence of Distribution Shift"* (Fahrbach et al., ICML 2023), kèm theo phân tích độ nhạy mở rộng (Bonus Sensitivity Analysis).

### Lý do lựa chọn Full Scope
1. **Chi phí tính toán khả thi**: Cả 3 bài toán (High-dimensional Linear Regression, High-dimensional Logistic Regression, và Deep Learning-based Flow Cytometry MLP 5 tầng) đều là học trực tuyến (online stream) với $T \in [600, 2000]$ bước và kích thước batch $B_t \in \{64, 256\}$. Tổng thời gian chạy toàn bộ thí nghiệm với 3 random seeds chỉ dưới 15 phút trên CPU/Colab.
2. **Dữ liệu mở và tự động hóa 100%**: Dữ liệu hồi quy được sinh tổng hợp theo quỹ đạo xoắn ốc công thức (26), còn dữ liệu tế bào tụy (4.000 tế bào, 10 marker genes) tải trực tiếp bằng một lệnh qua thư viện `scvelo.datasets.pancreas()`.
3. **Độ bao phủ lý thuyết – thực nghiệm trọn vẹn**: Bao quát cả 3 lớp hàm mục tiêu mà bài báo xây dựng:
   - **Hàm bậc hai (Linear Regression)**: Kiểm chứng xấp xỉ phương trình vi phân ngẫu nhiên (SDE) và điều khiển tối ưu HJB (Algorithm 1 — Figure 4, Figure 5-top).
   - **Hàm lồi tổng quát (Logistic Regression)**: Kiểm chứng lịch trình 2 ngưỡng $\tau_{1,t}, \tau_{2,t}$ không phụ thuộc tham số (Proposition 4.3 — Figure 5-bottom).
   - **Hàm không lồi (Flow Cytometry MLP)**: Kiểm chứng hiệu quả tăng bước nhảy / khởi động lại bộ tối ưu khi phân phối dịch chuyển qua 4 pha $P_1 \to P_4$ (Mục 5 & Phụ lục E — Figure 6).

---

## 2. Phân công Nhiệm vụ & Nhánh Code giữa các Thành viên

| Thành viên | Mảng phụ trách chính | Thí nghiệm & Hình tái hiện | Script tương ứng trong Repo |
| :--- | :--- | :--- | :--- |
| **TV1 (Khoa)** | Linear Regression & SDE Theory + Khung `core/` | **Figure 4** (Quỹ đạo xoắn ốc 2D & dao động $\eta_t$) và **Figure 5-top** (Cumulative Regret theo chiều $d \in \{2, 8, 32, 128, 512\}$) | `experiments/linear_regression.py`, `experiments/smoke_test.py` |
| **TV2 (Khôi)** | General Convex Loss & W&B Tracking (`core/tracking.py`) + Sensitivity | **Figure 5-bottom** (Logistic Regression theo chiều $d$) + **Bonus Sensitivity** (Quét tốc độ trôi $\gamma$ và phương sai nhiễu $\sigma$) | `experiments/convex.py`, `experiments/sensitivity.py` |
| **TV3 (Hòa)** | Non-Convex Loss, Flow Cytometry MLP & Đối chiếu Code gốc | **Figure 6** (Flow Cytometry MLP 5 tầng qua 4 pha $P_1 \to P_4$) + Đối chiếu với repo gốc của tác giả | `experiments/neural_net.py`, `core/data.py` |

---

## 3. Ví dụ Thực tế của Nhóm (Real-world Motivation)

> **Bối cảnh thực tế: Hệ thống Dự đoán Tỷ lệ Nhấp chuột (CTR Prediction) và Gợi ý Sản phẩm Thời gian thực trên Sàn Thương mại Điện tử**

### Mô tả bài toán
Trên một sàn thương mại điện tử lớn (như Shopee/TikTok Shop), mô hình học máy trực tuyến (never-ending online learning) liên tục nhận luồng sự kiện hiển thị quảng cáo $z_{t,k} = (x_{t,k}, y_{t,k})$ theo từng giây (với $x_{t,k}$ là đặc trưng người dùng – mặt hàng và $y_{t,k}$ là phản hồi nhấp chuột/mua hàng). Ở mỗi bước $t$, mô hình phải **suy luận (inference)** dự đoán xác suất tương tác trước, sau đó nhận nhãn thực tế và **huấn luyện (training)** cập nhật trọng số $\theta_t \to \theta_{t+1}$ bằng Mini-batch SGD.

### Hai kiểu trôi dạt phân phối (Distribution Shift) trong thực tế
1. **Trôi dạt mượt (Smooth / Gradual Drift)**: Thị hiếu người dùng thay đổi dần từ sáng đến tối hoặc chuyển mùa (ví dụ: buổi sáng tìm đồ ăn sáng/văn phòng phẩm, buổi tối tìm đồ giải trí/thời trang). Khi đó, bộ trọng số tối ưu $\theta^*_t$ di chuyển liên tục trên không gian tham số ($\gamma_t = \|\theta^*_t - \theta^*_{t+1}\|_2 > 0$ ở mọi bước).
2. **Trôi dạt đột ngột theo đợt (Bursty / Abrupt Shift)**: Vào đúng khung giờ **Flash Sale 0h00** hoặc khi một xu hướng livestream bùng nổ, phân phối dữ liệu nhảy vọt sang một chế độ hoàn toàn mới ($\gamma_t$ tăng đột biến tại đầu mỗi phiên rồi giữ ổn định trong suốt phiên đó).

### Tại sao các lịch trình Learning Rate truyền thống thất bại?
- **Nếu dùng lịch trình giảm dần cổ điển ($\eta_t = \eta_0 / t$ hoặc Cosine Decay)**: Sau vài ngày vận hành ($t$ lớn), $\eta_t \to 0$. Khi sự kiện Flash Sale nổ ra, mô hình gần như "đóng băng" tại nghiệm cũ, không thể dịch chuyển $\theta_t$ sang $\theta^*_t$ mới $\implies$ dự đoán sai hàng loạt và gây thiệt hại doanh thu tức thì.
- **Nếu giữ $\eta_t$ cố định ở mức quá cao**: Trong những khung giờ bình thường (phân phối ổn định, $\gamma_t \approx 0$), bước nhảy lớn làm mô hình dao động mạnh quanh cực tiểu do nhiễu mini-batch ($\sigma^2 / B_t$), làm giảm độ chính xác.
- **Giải pháp của bài báo**: Cần một lịch trình **Learning Rate thích ứng (Adaptive Schedule)** có khả năng **tự động tăng $\eta_t$** khi phát hiện phân phối dịch chuyển mạnh (để khám phá và bám đuổi $\theta^*_t$) và **giảm dần $\eta_t$** khi môi trường ổn định (để hội tụ mịn).
