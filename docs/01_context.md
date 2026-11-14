# T4: Bối cảnh, Động lực Thực tế & Phân tích các Phương pháp Tiền nhiệm (Context, Motivation & Predecessors)

Tài liệu này phục vụ trực tiếp cho phần **Milestone 2 – Context & Motivation (Slide 3 đến Slide 7)**, được diễn đạt hoàn toàn bằng lời của nhóm và tuân thủ quy tắc phân tích 3 lớp cho mỗi phương pháp tiền nhiệm: **Ý tưởng cốt lõi $\to$ Hạn chế còn tồn tại $\to$ Hệ quả thực tế**.

---

## 1. Vấn đề Nghiên cứu (The Core Problem)

Trong học máy truyền thống (offline / batch learning), dữ liệu huấn luyện được giả định lấy độc lập và cùng phân phối (i.i.d.) từ một phân phối tĩnh $P$. Khi đó, mục tiêu của thuật toán Stochastic Gradient Descent (SGD) là hội tụ về một điểm cực tiểu cố định $\theta^* = \arg\min_\theta \mathbb{E}_{z \sim P}[\ell(\theta, z)]$. Để triệt tiêu phương sai nhiễu của mini-batch gradient khi tiến gần $\theta^*$, quy tắc kinh điển là **giảm dần tốc độ học (learning rate decay)** theo thời gian ($\eta_t \to 0$ khi $t \to \infty$).

Tuy nhiên, trong các hệ thống **Học trực tuyến không ngừng (Never-Ending / Online Sequential Learning)**:
1. Dữ liệu đến liên tục theo luồng (stream) từ một phân phối biến thiên theo thời gian $P_t$.
2. Tại mỗi bước $t$, hệ thống phải đưa ra dự đoán bằng mô hình hiện tại $\theta_t$ (**Inference**) trước khi nhận nhãn thực tế và cập nhật trọng số $\theta_t \to \theta_{t+1}$ (**Training**).
3. Do phân phối $P_t$ dịch chuyển (distribution shift), nghiệm tối ưu thực sự $\theta^*_t = \arg\min_\theta \bar{\ell}_t(\theta)$ cũng di chuyển liên tục trên không gian tham số với tốc độ dịch chuyển $\gamma_t = \|\theta^*_t - \theta^*_{t+1}\|_2$.

**Câu hỏi nghiên cứu trung tâm**:
> *Khi phân phối dữ liệu liên tục thay đổi ngay trong quá trình huấn luyện, ta nên đặt tốc độ học $\eta_t$ cao đến mức nào, và đâu là lịch trình điều chỉnh $\{\eta_t\}_{t=1}^T$ tối ưu để cực tiểu hóa tổng độ tiếc nuối động (Dynamic Regret) trong suốt vòng đời của hệ thống?*

---

## 2. Ví dụ Thực tế Minh họa của Nhóm (Real-World Motivation)

Xét **Hệ thống Dự đoán Tỷ lệ Nhấp chuột (CTR) Quảng cáo Trực tuyến** hoặc **Hệ thống Phân loại Tế bào Dòng chảy (Real-time Flow Cytometry)**:
- **Trong thương mại điện tử**: Hành vi người dùng vừa trôi mượt theo khung giờ sáng – tối (*smooth shift*), vừa nhảy vọt đột ngột khi bắt đầu khung giờ Flash Sale 0h00 (*bursty shift*).
- **Trong thiết bị y sinh Flow Cytometry (Phụ lục E)**: Khi quét mẫu mô tụy từ vùng trước ra vùng sau, tỷ lệ các loại tế bào nội tiết (alpha, beta, delta) và ngoại tiết (ductal) thay đổi qua 4 pha phân phối $P_1 \to P_4$. Mạng nơ-ron gọn nhẹ (Student MLP) phải phân loại tế bào theo thời gian thực ngay dưới tia laser.
- **Mâu thuẫn cốt lõi (Exploration vs. Exploitation dưới Distribution Shift)**:
  - Nếu giảm $\eta_t \to 0$ theo thời gian: Khi sang pha phân phối mới ($P_2, P_3, P_4$), mô hình mất khả năng khám phá không gian trọng số, bị kẹt cứng ở cực tiểu cũ và dự đoán sai hàng loạt.
  - Nếu giữ $\eta_t$ cố định ở mức cao: Khi phân phối đang tạm ổn định trong nội bộ một pha, bước nhảy quá lớn khiến trọng số dao động hỗn loạn quanh cực tiểu do nhiễu mẫu mini-batch.

---

## 3. Phân tích Chuyên sâu các Phương pháp Tiền nhiệm (Predecessor Approaches)

Bài báo của Fahrbach et al. (2023) được xây dựng trên nền tảng của 3 nhánh nghiên cứu tiền nhiệm. Dưới đây là phân tích chi tiết từng nhánh:

### Tiền nhiệm 1: Lịch trình Suy giảm Tốc độ học Cổ điển trên Phân phối Tĩnh (Classical Decaying & Cyclical Schedules — Robbins & Monro, 1951; Smith, 2015; Ge et al., 2019; Jain et al., 2019)
* **Ý tưởng cốt lõi**:
  Thiết kế tốc độ học giảm dần theo hàm đa thức ($\eta_t \propto 1/t$ hoặc $1/\sqrt{t}$), giảm theo bậc thang (*step decay* — Ge et al., 2019), hoặc tăng kích thước batch thay vì giảm learning rate (Smith et al., 2018) nhằm giảm phương sai gradient ngẫu nhiên và hội tụ chặt về nghiệm tối ưu tĩnh $\theta^*$.
* **Hạn chế còn tồn tại**:
  Toàn bộ các phân tích hội tụ này đều dựa trên giả thiết cứng rằng phân phối dữ liệu không đổi theo thời gian ($P_t = P, \; \forall t$, tức $\gamma_t \equiv 0$). Chúng hoàn toàn không có cơ chế phát hiện hay phản ứng khi cực tiểu $\theta^*_t$ dịch chuyển ở giai đoạn muộn ($t$ lớn).
* **Hệ quả thực tế**:
  Khi triển khai vào hệ thống streaming dài hạn (never-ending learning), sau giai đoạn đầu, $\eta_t$ đã suy giảm về sát $0$. Ngay khi xảy ra một đợt dịch chuyển phân phối (ví dụ: bước nhảy $\theta^*_t$ mới), bước cập nhật $-\eta_t \nabla \ell^B_t(\theta_t) \approx 0$ làm mô hình **mất hoàn toàn khả năng bám đuổi nghiệm mới** (model freezing), khiến Dynamic Regret tích lũy tăng vọt tuyến tính theo $T$.

---

### Tiền nhiệm 2: Tối ưu Lồi Trực tuyến trong Môi trường Động (Online Convex Optimization with Dynamic Regret — Besbes et al., 2015; Yang et al., 2016)
* **Ý tưởng cốt lõi**:
  Đánh giá bộ học trực tuyến so với chuỗi nghiệm động $\{\theta^*_t\}_{t=1}^T$ dưới ràng buộc tổng ngân sách biến thiên đường đi $\sum_{t=1}^T \|\theta^*_t - \theta^*_{t+1}\|_2 \le V_T$ (Yang et al., 2016) hoặc biến thiên hàm mục tiêu (Besbes et al., 2015). Yang et al. (2016) đề xuất giữ bước nhảy hằng số $\eta \propto \sqrt{V_T / T}$, còn Besbes et al. (2015) đề xuất khởi động lại (restart) thuật toán cố định sau mỗi chu kỳ $\Delta_T = (T / V_T)^{2/3}$ bước với $\eta \propto 1/\sqrt{\Delta_T}$.
* **Hạn chế còn tồn tại**:
  1. **Bước nhảy tiền định (Non-adaptive step size)**: Đòi hỏi phải biết trước tổng ngân sách biến thiên $V_T$ và chân trời $T$ ngay từ đầu để ấn định một $\eta$ hằng số hoặc chu kỳ restart $\Delta_T$ cố định, thay vì điều chỉnh $\eta_t$ thích ứng theo lịch sử dữ liệu tại từng thời điểm $t$.
  2. **Mô hình đối kháng phi thực tế ở cận dưới**: Chuỗi hàm mục tiêu được giả định cố định trước (oblivious) và cận dưới worst-case được xây dựng bằng cách chế tạo chuỗi hàm bệnh lý thay đổi hình dạng tùy ý, thay vì xuất phát từ cùng một hàm mất mát $\ell(\theta, z)$ với phân phối dữ liệu $P_t$ dịch chuyển thích ứng.
* **Hệ quả thực tế**:
  Trong môi trường thực tế nơi phân phối giữ ổn định suốt thời gian dài rồi bất ngờ nhảy vọt (*bursty shift*), việc dùng một $\eta \propto \sqrt{V_T / T}$ cố định dẫn đến hiệu năng kém ở cả hai pha: bước nhảy **quá lớn** trong giai đoạn tĩnh (gây sai số phương sai cao) nhưng lại **quá nhỏ** ngay tại thời điểm có cú nhảy đột ngột (làm mô hình bắt kịp nghiệm mới rất chậm).

---

### Tiền nhiệm 3: Phân tích Động lực học SGD qua Phương trình Vi phân Ngẫu nhiên Liên tục (Continuous-Time SDE Approximation for SGD — Li et al., 2017; Shi et al., 2020)
* **Ý tưởng cốt lõi**:
  Xấp xỉ quỹ đạo cập nhật rời rạc của SGD bằng một quá trình khuếch tán liên tục thông qua Phương trình Vi phân Ngẫu nhiên (SDE) dựa trên lược đồ Euler–Maruyama. Shi et al. (2020) sử dụng SDE và toán tử Schrödinger để chứng minh SGD hội tụ tuyến tính về phân phối dừng (stationary distribution) và làm rõ mối quan hệ giữa tốc độ hội tụ với learning rate.
* **Hạn chế còn tồn tại**:
  Phương trình SDE trong Shi et al. (2020) và Li et al. (2017) chỉ mô hình hóa trường vector gradient trên một cảnh quan mất mát tĩnh ($\theta^*$ cố định). Phương trình vi phân của họ không chứa số hạng vận tốc trôi $Y'(\tau)$ của nghiệm tối ưu theo thời gian và chưa kết hợp với lý thuyết điều khiển tối ưu (Optimal Control / HJB) để giải bài toán tìm hàm điều chỉnh tốc độ học $\zeta^*(\tau)$.
* **Hệ quả thực tế**:
  Các kết quả SDE trước đó chỉ giải thích được hành vi của SGD khi phân phối đứng yên, không cung cấp được thuật toán tính toán trực tuyến bước nhảy tối ưu $\eta^*_t$ khi quỹ đạo $\theta^*_t$ uốn lượn hoặc nhảy bậc theo thời gian.

---

## 4. Bảng Tổng hợp Khoảng trống Nghiên cứu (The Research Gap — Dùng cho Slide 7)

| Tiêu chí so sánh | Suy giảm cổ điển (Ge et al., 2019) | OCO Dynamic Regret (Besbes 2015; Yang 2016) | Xấp xỉ SDE cổ điển (Shi et al., 2020) | **Bài báo này (Fahrbach et al., ICML 2023)** |
| :--- | :--- | :--- | :--- | :--- |
| **Môi trường dữ liệu** | Phân phối tĩnh ($\gamma_t = 0$) | Chuỗi hàm biến thiên bị chặn bởi tổng $V_T$ | Phân phối tĩnh ($\gamma_t = 0$) | **Phân phối $P_t$ dịch chuyển thích ứng theo lịch sử ($\gamma_t \ge 0$)** |
| **Bản chất lịch trình $\eta_t$** | Giảm đơn điệu về $0$ ($\mathcal{O}(1/t)$) | Hằng số $\eta \propto \sqrt{V_T/T}$ hoặc restart chu kỳ cố định $\Delta_T$ | Phân tích $\eta$ cố định hoặc giảm dần | **Thích ứng trực tuyến tại từng bước $t$; tự động TĂNG $\eta^*_t$ khi có shift** |
| **Công cụ lý thuyết** | Chuỗi martingale trên nghiệm tĩnh | Cận trên worst-case theo tổng $V_T$ | SDE trên hàm mất mát tĩnh | **SDE có số hạng trôi $Y'(\tau)$ + Điều khiển tối ưu HJB (Alg. 1) & Cận trên/dưới chặt đến hằng số (Thm. 4.2, 4.7)** |
| **Phạm vi hàm mất mát** | Least Squares / Lồi | Hàm lồi | Hàm trơn tĩnh | **Đầy đủ 3 lớp: Linear Regression, General Convex, và Non-Convex (DNN)** |
