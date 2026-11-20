# T11 & T12: Hàm Mất mát Không lồi (Non-Convex Loss) & Ứng dụng Phân loại Tế bào Dòng chảy Trực tuyến (Flow Cytometry MLP — Mục 5 & Phụ lục E)

Tài liệu này trình bày chi tiết cơ sở lý thuyết của **Mục 5 (Non-convex loss)** và thiết lập thực nghiệm y sinh của **Phụ lục E (Flow Cytometry)** trong bài báo *"Learning Rate Schedules in the Presence of Distribution Shift"* (Fahrbach et al., ICML 2023), phục vụ trực tiếp cho **Slide 18, Slide 19, Slide 27 và Slide 28**.

---

## 1. Tại sao phải Định nghĩa lại Dynamic Regret cho Hàm Không lồi? (Mục 5 — Slide 18)

### 1.1. Ba câu hỏi cốt lõi (Làm gì? Vì sao cần? Hoạt động thế nào?)
* **Làm gì?** Thay thế định nghĩa Dynamic Regret theo giá trị hàm mục tiêu $\bar{\ell}_t(\theta_t) - \bar{\ell}_t(\theta^*_t)$ bằng **tổng bình phương chuẩn gradient kỳ vọng** dọc theo quỹ đạo SGD:
  $$\text{Reg}(T) := \sum_{t=1}^T \|\nabla \bar{\ell}_t(\theta_t)\|_2^2 \tag{22}$$
  đồng thời đo độ dịch chuyển phân phối bằng **khoảng cách supremum toàn cục giữa hai bề mặt mất mát liên tiếp**:
  $$\gamma_t := \sup_{\theta \in \mathbb{R}^p} \left|\bar{\ell}_t(\theta) - \bar{\ell}_{t+1}(\theta)\right| \tag{23}$$
* **Vì sao cần?**
  1. Trong bài toán huấn luyện mạng nơ-ron sâu, bề mặt mất mát $\bar{\ell}_t(\theta)$ không lồi và chứa vô số cực tiểu địa phương (local minima) cũng như điểm yên ngựa (saddle points). Ngay cả khi phân phối tĩnh ($\gamma_t = 0$), các thuật toán bậc nhất như SGD không thể đảm bảo hội tụ về cực tiểu toàn cục $\theta^*_t$ nếu không có thêm giả thiết cấu trúc mạnh.
  2. Vì không chỉ định được một nghiệm toàn cục duy nhất $\theta^*_t$, ta không thể đo khoảng cách tham số $\|\theta^*_t - \theta^*_{t+1}\|_2$ như ở Mục 3 và Mục 4. Thay vào đó, công thức (22) đánh giá khả năng mô hình **duy trì quỹ đạo ở gần các điểm dừng (stationary points / local minima)** của chuỗi hàm mất mát biến thiên theo thời gian.

---

## 2. Chặn trên Regret (Theorem 5.1) & Công thức Bước nhảy Đóng (Công thức 25 — Slide 19)

### 2.1. Phát biểu Theorem 5.1 (Phụ lục C.1, Công thức 24)
Giả sử tại mỗi bước $t = 1, \dots, T$:
1. Gradient mẫu trên mini-batch không chệch và có phương sai bị chặn: $\mathbb{E}_{z \sim P_t}[\|\nabla \ell(\theta_t, z) - \nabla \bar{\ell}_t(\theta_t)\|_2^2] \le \sigma^2$.
2. Hàm mất mát kỳ vọng $\bar{\ell}_t(\theta) \ge 0$ có gradient $L$-smooth: $\|\nabla \bar{\ell}_t(\theta) - \nabla \bar{\ell}_t(\theta')\|_2 \le L\|\theta - \theta'\|_2$.
3. Tốc độ học thích ứng với lịch sử $\boldsymbol{z}_{[t-1]}$ và thỏa mãn $0 < \eta_t \le 1/L$ sao cho $a_t := 2\eta_t - L\eta_t^2 > 0$.

Khi đó, dùng khai triển Taylor bậc hai (Bổ đề Descent Lemma):
$$\bar{\ell}_t(\theta_{t+1}) - \bar{\ell}_t(\theta_t) \le \langle \nabla \bar{\ell}_t(\theta_t), \theta_{t+1} - \theta_t \rangle + \frac{L}{2}\|\theta_{t+1} - \theta_t\|_2^2$$
và chèn độ dịch chuyển bề mặt $\bar{\ell}_t(\theta_{t+1}) = \bar{\ell}_{t+1}(\theta_{t+1}) + (\bar{\ell}_t(\theta_{t+1}) - \bar{\ell}_{t+1}(\theta_{t+1})) \ge \bar{\ell}_{t+1}(\theta_{t+1}) - \gamma_t$, tổng bình phương chuẩn gradient kỳ vọng bị chặn trên bởi (**Theorem 5.1**):

$$\mathbb{E}[\text{Reg}(T)] \le \mathbb{E}\left[\frac{2\bar{\ell}_1(\theta_1)}{a_1} + \sum_{t=2}^T 2\bar{\ell}_t(\theta_t)\left(\frac{1}{a_t} - \frac{1}{a_{t-1}}\right)\right] + \sum_{t=1}^T \mathbb{E}\left[\frac{1}{a_t}\left(\frac{L\sigma^2 \eta_t^2}{B_t} + 2\gamma_t\right)\right] \tag{24}$$

### 2.2. Dẫn xuất Tốc độ học Tối ưu $\eta^*_t$ (Công thức 25)
Để cực tiểu hóa tuần tự cận trên (24) tại bước $t$ khi biết lịch sử $\boldsymbol{z}_{[t-1]}$, ta giải bài toán tối ưu lồi trên miền $\eta \in (0, 1/L]$:

$$\eta^*_t := \arg\min_{0 \le \eta \le \frac{1}{L}} \left\{ \frac{2\bar{\ell}_t(\theta_t) + 2\gamma_t}{2\eta - L\eta^2} + \frac{L\sigma^2}{B_t} \cdot \frac{\eta^2}{2\eta - L\eta^2} \right\} \tag{25}$$

Lấy đạo hàm theo $\eta$ và giải phương trình bậc hai, nghiệm dừng dương duy nhất luôn thỏa mãn điều kiện biên $0 \le \eta^*_t \le 1/L$ và có dạng đóng:

$$\eta^*_t = \frac{B_t}{L\sigma^2}\left(\sqrt{b_t^2 + \frac{2\sigma^2}{B_t}b_t} - b_t\right) = \frac{2}{L\left(1 + \sqrt{1 + \frac{2\sigma^2}{B_t b_t}}\right)}, \quad \text{với } b_t := L\left(\gamma_t + \bar{\ell}_t(\theta_t)\right)$$

* **Trực giác cốt lõi**:
  * Khi xuất hiện **dịch chuyển phân phối ($\gamma_t$ tăng)** hoặc **mô hình rơi vào vùng có sai số cao ($\bar{\ell}_t(\theta_t)$ tăng)** $\implies$ đại lượng $b_t$ tăng lên $\implies$ tỷ số $\frac{2\sigma^2}{B_t b_t}$ giảm xuống $\implies$ **tốc độ học tối ưu $\eta^*_t$ TĂNG LÊN (tiến sát trần $1/L$)**!
  * Việc tăng bước nhảy ngay khi $\gamma_t$ hoặc $\bar{\ell}_t(\theta_t)$ nhảy vọt giúp mạng nơ-ron thoát khỏi hố cực tiểu cục bộ cũ (vốn đã trở nên lỗi thời sau khi phân phối dịch chuyển).

---

## 3. Ứng dụng Y sinh: Phân loại Tế bào Dòng chảy Trực tuyến (Flow Cytometry — Phụ lục E)

### 3.1. Bối cảnh Hệ thống Teacher–Student trong Y sinh (Phụ lục E.1 — Slide 27)
* **Bài toán thực tế**: Trong kỹ thuật phân tích và phân loại tế bào dòng chảy (Flow Cytometry), một dòng tế bào từ mô sinh học đi qua chùm tia laser với tốc độ hàng nghìn tế bào mỗi giây.
* **Kiến trúc 2 tầng (Two-step Teacher–Student Architecture)**:
  1. **Mạng Student (MLP 5 tầng gọn nhẹ)**: Nhận tín hiệu huỳnh quang từ chỉ **10 phân tử RNA đánh dấu (fluorescent RNA markers)** để phân loại tức thời tế bào theo thời gian thực (Stage A — Online Inference).
  2. **Mạng Teacher (Pretrained CNN quy mô lớn)**: Xử lý hình ảnh không cần chất đánh dấu (marker-free) trên cụm máy chủ song song để trả về nhãn chuẩn (ground-truth label) gần thời gian thực, giúp mạng Student cập nhật trọng số trực tuyến (Stage B — Online Training).
* **Nguồn gốc của Distribution Shift trong Sinh học**: Khi quét mẫu mô tuyến tụy từ vùng đầu (anterior) sang vùng đuôi (posterior), tỷ lệ giữa các tế bào không tiết (non-secreting ductal cells) và các tế bào nội tiết (secreting islet cells: alpha, beta, delta) thay đổi mạnh theo không gian và thời gian.

### 3.2. Đặc tả Kỹ thuật của Thực nghiệm Flow Cytometry (Phụ lục E.2 — Slide 27 & 28)

| Thành phần | Thiết lập Chính xác trong Bài báo & Mã nguồn (`core/data.py`, `experiments/neural_net.py`) |
| :--- | :--- |
| **Nguồn dữ liệu** | `scvelo.datasets.pancreas()` (Bastidas-Ponce et al., 2019) — 3.696 tế bào tuyến tụy chuột (`endocrinogenesis_day15.h5ad`). |
| **Tiền xử lý & 10 Marker Genes** | Lọc bởi `min_shared_counts=30000` và chuẩn hóa `normalize_per_cell`, thu được đúng 10 genes: **`Pyy, Meg3, Malat1, Gcg, Gnas, Actb, Ghrl, Rps3, Ins2, Hspa8`**. |
| **4 Phân lớp Mục tiêu ($K = 4$)** | `Alpha` ($481$ tế bào), `Beta` ($591$ tế bào), `Delta` ($70$ tế bào), `Ductal` ($916$ tế bào) — tổng cộng $2.058$ tế bào thuộc 4 phân lớp đích. |
| **Nhiễu nhân dòng chảy** | Mỗi mẫu tế bào được nhân ngẫu nhiên theo từng gene: $x_j \leftarrow x_j \cdot U_j$ với $U_j \stackrel{\text{i.i.d.}}{\sim} \text{Uniform}[0.9, 1.1]$. |
| **4 Pha Dịch chuyển ($T = 600$)** | Vector xác suất lớp $(p_{\text{Alpha}}, p_{\text{Beta}}, p_{\text{Delta}}, p_{\text{Ductal}})$:<br>• **Pha 1 ($t = 1..100$)**: $P_1 = (0.0, 0.0, 0.0, 1.0)$ — 100% Ductal<br>• **Pha 2 ($t = 101..200$)**: $P_2 = (0.0, 0.0, 0.1, 0.9)$ — 10% Delta, 90% Ductal<br>• **Pha 3 ($t = 201..400$)**: $P_3 = (0.1, 0.0, 0.2, 0.7)$ — 10% Alpha, 20% Delta, 70% Ductal<br>• **Pha 4 ($t = 401..600$)**: $P_4 = (0.3, 0.5, 0.1, 0.1)$ — 30% Alpha, 50% Beta, 10% Delta, 10% Ductal |
| **Kiến trúc Mạng (`CytometryMLP`)** | MLP 5 tầng tuyến tính: `10 -> 64 -> 32 -> 16 -> 8 -> 4`, kích hoạt `ELU` ở các tầng ẩn, khởi tạo Glorot/Xavier Uniform, phạt điều chuẩn $L_2 = 10^{-4}$. |
| **Hàm mất mát & Batch size** | Categorical Cross-Entropy từ logits (`from_logits=True`), kích thước mini-batch $B_t = 64$. |
| **Cơ chế Adam Restart** | Khởi tạo lại bộ đệm moment bậc nhất $m_t$ và bậc hai $v_t$ của Adam tại đầu mỗi đợt dịch chuyển phân phối ($t \in \{100, 200, 400\}$). |

### 3.3. Tại sao Adam Restart Tương đương với Tăng Tốc độ học Hiệu dụng?
Trong bộ tối ưu Adam, bước cập nhật tham số tỉ lệ với $\frac{\eta_0}{\sqrt{\hat{v}_t} + \epsilon}$, trong đó $\hat{v}_t$ là trung bình trượt hàm mũ của bình phương gradient quá khứ.
* Sau 100–200 bước hội tụ ở pha cũ, bộ đệm $\hat{v}_t$ và hướng quán tính $\hat{m}_t$ đã thích nghi chặt với cực tiểu cục bộ của phân phối cũ.
* Khi phân phối nhảy sang pha mới ($t \in \{100, 200, 400\}$), việc **Reset bộ tối ưu Adam** xóa bỏ mẫu số $\sqrt{\hat{v}_t}$ cũ và đặt lại bộ đếm bước $t_{\text{adam}} \leftarrow 1$, tạo ra một **xung bước nhảy hiệu dụng lớn** giúp mạng MLP nhanh chóng thoát khỏi điểm cực tiểu cũ để thích nghi với tỷ lệ tế bào mới!
