# T5: Bảng Ký hiệu Thống nhất, Bài toán Hình thức & Sơ đồ Pipeline

Tài liệu này quy chuẩn toàn bộ hệ thống ký hiệu toán học dùng chung cho cả 3 thành viên (TV1: Khoa, TV2: Khôi, TV3: Hòa) trong quá trình viết tài liệu, lập trình thực nghiệm và thiết kế Slide Deck (Milestone 2 + Lab 1).

---

## 1. Bảng Ký hiệu Thống nhất (Unified Notation Table)

### 1.1. Thiết lập Học Trực tuyến & Dynamic Regret (Mục 2)

| Ký hiệu | Miền giá trị / Kiểu | Ý nghĩa trong bài toán của nhóm | Công thức / Định nghĩa liên hệ |
| :--- | :--- | :--- | :--- |
| $t, T$ | $t \in \{1, 2, \dots, T\}$ | Chỉ số bước thời gian rời rạc (vòng lặp SGD) và tổng chân trời thời gian (horizon) | — |
| $d, p$ | $\mathbb{N}_+$ | Số chiều của mẫu dữ liệu $z_{t,k} \in \mathbb{R}^d$ và số chiều của vector trọng số mô hình $\theta \in \mathbb{R}^p$ | Trong hồi quy tuyến tính/logistic: $p = d$ |
| $B_t$ | $\mathbb{N}_+$ | Kích thước mini-batch tại bước $t$ | $B_t = 256$ (hồi quy), $B_t = 64$ (cytometry) |
| $P_t$ | Phân phối xác suất trên $\mathbb{R}^d$ | Phân phối dữ liệu tại bước thời gian $t$ (có thể thay đổi theo thời gian) | — |
| $z_t = \{z_{t,k}\}_{k=1}^{B_t}$ | $z_{t,k} \stackrel{\text{i.i.d.}}{\sim} P_t$ | Tập mẫu mini-batch quan sát được tại bước $t$; với bài toán có giám sát, $z_{t,k} = (x_{t,k}, y_{t,k})$ | Lịch sử dữ liệu: $\boldsymbol{z}_{[t-1]} = \{z_1, \dots, z_{t-1}\}$ |
| $\ell(\theta, z)$ | $\mathbb{R}^p \times \mathbb{R}^d \to \mathbb{R}_{\ge 0}$ | Hàm mất mát điểm (pointwise loss) của mô hình $\theta$ trên mẫu dữ liệu $z$ | Least-squares: $\frac{1}{2}(y - \langle x, \theta \rangle)^2$ |
| $\bar{\ell}_t(\theta)$ | $\mathbb{R}^p \to \mathbb{R}_{\ge 0}$ | Hàm mất mát kỳ vọng trên phân phối $P_t$ tại bước $t$ | $\bar{\ell}_t(\theta) := \mathbb{E}_{z \sim P_t}[\ell(\theta, z)]$ |
| $\nabla \ell^B_t(\theta_t)$ | $\mathbb{R}^p$ | Gradient trung bình mẫu trên mini-batch tại bước $t$ (ước lượng không chệch của $\nabla \bar{\ell}_t(\theta_t)$) | $\nabla \ell^B_t(\theta_t) := \frac{1}{B_t}\sum_{k=1}^{B_t} \nabla \ell(\theta_t, z_{t,k})$ |
| $\Theta, \Pi_\Theta(\cdot)$ | $\Theta \subseteq \mathbb{R}^p$ | Tập trọng số chấp nhận được (tập lồi đóng bị chặn đường kính $D_{\max}$) và phép chiếu Euclid lên $\Theta$ | $\theta_{t+1} = \Pi_\Theta(\theta_t - \eta_t \nabla \ell^B_t(\theta_t))$ |
| $\theta_t$ | $\Theta \subseteq \mathbb{R}^p$ | Trọng số mô hình của bộ học (learner) tại đầu bước $t$ | Phụ thuộc vào lịch sử $\boldsymbol{z}_{[t-1]}$ |
| $\theta^*_t$ | $\Theta \subseteq \mathbb{R}^p$ | Trọng số **mô hình Oracle (comparator)** tối ưu tại bước $t$ | $\theta^*_t := \arg\min_{\theta \in \mathbb{R}^p} \bar{\ell}_t(\theta)$ |
| $\eta_t, \eta^*_t$ | $\mathbb{R}_{\ge 0}$ | Tốc độ học (learning rate / step size) tại bước $t$ và tốc độ học tối ưu theo lịch trình đề xuất | Chọn thích ứng dựa trên $\boldsymbol{z}_{[t-1]}$ |
| $\gamma_t$ | $\mathbb{R}_{\ge 0}$ | **Độ dịch chuyển phân phối (Distribution shift)** giữa bước $t$ và $t+1$ | • Lồi (Eq. 6): $\gamma_t := \|\theta^*_t - \theta^*_{t+1}\|_2$<br>• Không lồi (Eq. 23): $\gamma_t := \sup_{\theta} |\bar{\ell}_t(\theta) - \bar{\ell}_{t+1}(\theta)|$ |
| $\hat{\gamma}_t$ | $\mathbb{R}_{\ge 0}$ | Ước lượng thực nghiệm của $\gamma_t$ bằng trung bình trượt hàm mũ (EMA) khi không biết $\theta^*_t$ | $\hat{\gamma}_t = \beta \hat{\gamma}_{t-1} + (1 - \beta)\|\theta_t - \theta_{t-1}\|_2, \; \beta \in (0, 1)$ |
| $\text{reg}_t, \text{Reg}(T)$ | $\mathbb{R}_{\ge 0}$ | **Dynamic Regret** tức thời tại bước $t$ và tổng Dynamic Regret tích lũy đến bước $T$ | • Lồi (Eq. 2): $\text{reg}_t := \bar{\ell}_t(\theta_t) - \bar{\ell}_t(\theta^*_t)$<br>• Không lồi (Eq. 22): $\text{reg}_t := \|\nabla \bar{\ell}_t(\theta_t)\|_2^2$ |

---

### 1.2. Các Biến Xấp xỉ Liên tục SDE & Hàm Lồi / Không lồi (Mục 3, 4, 5)

| Ký hiệu | Miền giá trị | Ý nghĩa kỹ thuật | Vai trò trong thuật toán |
| :--- | :--- | :--- | :--- |
| $\varepsilon$ | $\mathbb{R}_{>0}$ | Bước nhảy tối đa cho phép ($\eta_{\max}$) kiêm hệ số co giãn thời gian rời rạc sang liên tục | $\tau = \varepsilon t$ là biến thời gian liên tục |
| $\zeta(\tau)$ | $[0, 1]$ | Hệ số điều chỉnh tốc độ học liên tục (control policy) | Liên hệ rời rạc: $\eta_t = \varepsilon \zeta(\varepsilon t)$ |
| $\nu(\tau)$ | $\mathbb{R}_{>0}$ | Hàm kích thước batch liên tục | Liên hệ rời rạc: $B_t = \lceil \varepsilon \nu(\varepsilon t) \rceil$ |
| $Y(\tau), Y'(\tau)$ | $\mathbb{R}^d$ | Đường cong nội suy trơn của chuỗi mô hình oracle $\theta^*_t$ và đạo hàm vận tốc trôi | $Y(\varepsilon t) = \theta^*_t$, $\|Y'(\varepsilon t)\| \approx \gamma_t / \varepsilon$ |
| $X(\tau)$ | $\mathbb{R}^d$ | Quá trình ngẫu nhiên liên tục xấp xỉ vector sai số $\theta_t - \theta^*_t$ tại $\tau = \varepsilon t$ | Nghiệm của phương trình SDE (8) |
| $W(\tau)$ | $\mathbb{R}^d$ | Chuyển động Brown chuẩn $d$ chiều (Standard Brownian Motion) | Mô hình hóa nhiễu gradient mini-batch |
| $m_\tau, v_\tau$ | $\mathbb{R}^d, \mathbb{R}_{\ge 0}$ | Moment bậc nhất $m_\tau = \mathbb{E}[X(\tau)]$ và moment bậc hai $v_\tau = \mathbb{E}[\|X(\tau)\|_2^2]$ | Với hồi quy tuyến tính: $\mathbb{E}[\text{reg}_t] \approx \frac{1}{2}v_{\varepsilon t}$ |
| $\tilde{v}_\tau$ | $\mathbb{R}_{\ge 0}$ | Quá trình bao trên (upper envelope) của moment bậc hai $v_\tau$ | Giải bằng phương pháp Euler thuận trong **Algorithm 1** |
| $\kappa$ | $(0, 1]$ | Độ chia nhỏ bước thời gian (discretization scale) khi giải ODE cho $\tilde{v}_\tau$ | Số bước con Euler mỗi vòng lặp: $\lceil 1/\kappa \rceil$ |
| $\sigma^2$ | $\mathbb{R}_{\ge 0}$ | Phương sai nhiễu nhãn (Mục 3: $\varepsilon_{t,k} \sim \mathcal{N}(0, \sigma^2)$) hoặc cận trên phương sai gradient (Mục 4, 5) | $\mathbb{E}_{P_t}[\|\nabla \ell(\theta_t, z_{t,k}) - \nabla \bar{\ell}_t(\theta_t)\|^2] \le \sigma^2$ |
| $L, \mu$ | $\mathbb{R}_{>0}$ | Hằng số trơn Lipschitz của gradient ($L$-smooth) và hằng số lồi mạnh ($\mu$-strongly convex) | Điều kiện ổn định bước nhảy: $0 \le \eta_t \le 1/L$ |
| $D_t, D_{\max}$ | $\mathbb{R}_{\ge 0}$ | Khoảng cách Euclid $D_t := \|\theta^*_t - \theta_t\|_2$ và đường kính miền tham số $D_{\max}$ | Dùng để chặn trên chuỗi đan dấu trong Theorem 4.2 |
| $\tau_{1,t}, \tau_{2,t}$ | $[0, 1/L]$ | Hai ngưỡng dưới và ngưỡng trên cập nhật $\eta^*_t$ cho hàm lồi tổng quát (**Proposition 4.3**) | $\eta^*_t = \text{clip}(\eta^*_{t-1}, \tau_{1,t}, \tau_{2,t})$ |

---

## 2. Phát biểu Bài toán Hình thức (Formal Problem Formulation)

Xét trò chơi học trực tuyến tuần tự qua $T$ vòng giữa **Tự nhiên (Nature)** và **Bộ học (Learner)**:

1. **Tại mỗi bước $t = 1, 2, \dots, T$**:
   - Tự nhiên chọn phân phối dữ liệu $P_t$ (có thể phụ thuộc thích ứng vào lịch sử dữ liệu $\boldsymbol{z}_{[t-1]}$), xác định hàm mất mát kỳ vọng $\bar{\ell}_t(\theta) = \mathbb{E}_{z \sim P_t}[\ell(\theta, z)]$ và nghiệm tối ưu tức thời (oracle) $\theta^*_t = \arg\min_{\theta \in \mathbb{R}^p} \bar{\ell}_t(\theta)$.
   - Giữa hai bước liên tiếp, mô hình tối ưu dịch chuyển một lượng:
     $$\gamma_t := \|\theta^*_t - \theta^*_{t+1}\|_2$$
   - Bộ học sở hữu mô hình $\theta_t \in \Theta$, nhận một mini-batch gồm $B_t$ mẫu độc lập $z_t = \{z_{t,k}\}_{k=1}^{B_t} \stackrel{\text{i.i.d.}}{\sim} P_t$, chịu độ tiếc nuối tức thời (instantaneous dynamic regret):
     $$\text{reg}_t := \bar{\ell}_t(\theta_t) - \bar{\ell}_t(\theta^*_t)$$
2. **Quy tắc cập nhật trọng số**:
   - Bộ học chọn bước nhảy $\eta_t \ge 0$ dựa trên lịch sử $\boldsymbol{z}_{[t-1]}$ và cập nhật mô hình theo Projected Mini-batch SGD:
     $$\theta_{t+1} = \Pi_\Theta\left(\theta_t - \eta_t \nabla \ell^B_t(\theta_t)\right), \quad \text{với } \nabla \ell^B_t(\theta_t) = \frac{1}{B_t}\sum_{k=1}^{B_t}\nabla \ell(\theta_t, z_{t,k})$$
3. **Mục tiêu tối ưu của Lịch trình Learning Rate**:
   - Thiết kế quy tắc chọn chuỗi bước nhảy thích ứng $\{\eta^*_t\}_{t=1}^T$ (chỉ dùng thông tin quá khứ $\boldsymbol{z}_{[t-1]}$) sao cho **tổng Dynamic Regret kỳ vọng** trên toàn chân trời $T$ đạt giá trị nhỏ nhất:
     $$\min_{\{\eta_t\}_{t=1}^T} \mathbb{E}[\text{Reg}(T)] = \min_{\{\eta_t\}_{t=1}^T} \sum_{t=1}^T \mathbb{E}\left[\bar{\ell}_t(\theta_t) - \bar{\ell}_t(\theta^*_t)\right]$$
   - *(Đối với hàm không lồi ở Mục 5, do không đảm bảo hội tụ về cực tiểu toàn cục, mục tiêu được thay bằng cực tiểu hóa tổng bình phương chuẩn gradient kỳ vọng: $\mathbb{E}[\text{Reg}(T)] = \sum_{t=1}^T \mathbb{E}[\|\nabla \bar{\ell}_t(\theta_t)\|_2^2]$).*

---

## 3. Pipeline Tổng thể của Phương pháp: Tách bạch Quá trình Inference và Training trong Never-Ending Learning

Đặc trưng cốt lõi của bài báo là hệ thống **Học trực tuyến không ngừng (Never-Ending / Online Sequential Learning)** theo cơ chế **Prequential (Test-then-Train)** (Mục 1 & Mục 2, trang 1–4): trên cùng một luồng dữ liệu biến thiên $P_t$, mỗi mini-batch $z_t$ đi vào hệ thống đều phải trải qua **Quá trình Suy luận (Inference / Testing)** trước để phục vụ người dùng và đo độ tiếc nuối thực tế, sau đó mới chuyển sang **Quá trình Huấn luyện (Training & LR Adaptation)** khi nhãn phản hồi được tiết lộ.

Sơ đồ khối tổng thể tách bạch rõ hai quá trình này được lưu tại `figures/pipeline.png`:

```mermaid
flowchart TB
    subgraph STREAM["LUỒNG DỮ LIỆU BIẾN THIÊN THEO THỜI GIAN (NEVER-ENDING STREAM)"]
        S1["Tại bước t: Phân phối P_t dịch chuyển (γ_t = ||θ*_t - θ*_{t+1}||_2)<br/>Sinh mini-batch đầu vào x_t = {x_{t,k}} (k = 1..B_t)"]
    end

    subgraph INFER["PHA 1: QUÁ TRÌNH SUY LUẬN & ĐÁNH GIÁ TRỰC TUYẾN (ONLINE INFERENCE / TEST PHASE)"]
        I1["1. Suy luận thời gian thực (Forward Prediction)<br/>Dùng trọng số hiện tại θ_t (cố định, chưa nhìn thấy nhãn bước t)<br/>Đưa ra dự đoán: ŷ_{t,k} = f(x_{t,k}; θ_t)"]
        I2["2. Thu nhận Nhãn Thực tế & Đo lường Regret (Online Evaluation)<br/>Nhận nhãn y_t (từ phản hồi người dùng / Teacher CNN trong Flow Cytometry)<br/>Tính Batch Loss ℓ^B_t(θ_t) và Dynamic Regret: reg_t = ℓ̄_t(θ_t) - ℓ̄_t(θ*_t)"]
        I1 --> I2
    end

    subgraph TRAIN["PHA 2: QUÁ TRÌNH HUẤN LUYỆN & THÍCH ỨNG LEARNING RATE (ONLINE TRAINING PHASE)"]
        T1["3. Tính Gradient Ngẫu nhiên<br/>Tính trung bình đạo hàm trên mini-batch:<br/>∇ℓ^B_t(θ_t) = (1/B_t) Σ ∇ℓ(θ_t, z_{t,k})"]
        T2["4. Ước lượng Độ trôi Phân phối (Shift Tracking)<br/>• Chế độ Oracle: dùng γ_{t-1}<br/>• Chế độ Thực tế: EMA γ̂_t = βγ̂_{t-1} + (1-β)||θ_t - θ_{t-1}||_2"]
        T3["5. Điều khiển Learning Rate Thích ứng (Adaptive Schedule η*_t)<br/>• Linear (Mục 3): Giải ODE ṽ_τ bằng Euler (Alg. 1)<br/>• Convex (Mục 4): Cắt ngưỡng [τ_{1,t}, τ_{2,t}] (Prop. 4.3)<br/>• Non-Convex (Mục 5): Công thức đóng (Eq. 25) / Adam Restart"]
        T4["6. Cập nhật Trọng số (Projected Mini-batch SGD)<br/>θ_{t+1} = Π_Θ(θ_t - η*_t ∇ℓ^B_t(θ_t))"]
        T1 --> T2 --> T3 --> T4
    end

    S1 --> I1
    I2 -->|Chuyển mẫu z_t = (x_t, y_t) sang pha học| T1
    T4 -->|Triển khai mô hình mới θ_{t+1} cho bước t+1| I1
```

### Diễn giải chi tiết sự khác biệt và liên kết giữa 2 Pipeline (Dùng cho Speaker Notes Slide 12)

| Tiêu chí | Pha 1: Quá trình Inference (Testing / Serving) | Pha 2: Quá trình Training (Adaptation / Updating) |
| :--- | :--- | :--- |
| **Thời điểm diễn ra ở bước $t$** | Diễn ra **TRƯỚC** khi biết nhãn $y_t$ (hoặc ngay khi mẫu dữ liệu $x_{t,k}$ vừa tới hệ thống). | Diễn ra **SAU** khi hệ thống đã xuất dự đoán và nhận được nhãn phản hồi $y_{t,k}$. |
| **Trạng thái mô hình & Learning Rate** | Trọng số $\theta_t$ **đóng băng tạm thời** (chỉ phụ thuộc lịch sử $\boldsymbol{z}_{[t-1]}$); không dùng learning rate $\eta_t$. | Trọng số được mở khóa để cập nhật $\theta_t \to \theta_{t+1}$ bằng bước nhảy thích ứng $\eta^*_t$. |
| **Luồng dữ liệu đi qua** | $x_{t,k} \xrightarrow{\theta_t} \hat{y}_{t,k} \xrightarrow{+ y_{t,k}} \text{reg}_t = \bar{\ell}_t(\theta_t) - \bar{\ell}_t(\theta^*_t)$. | $z_{t,k} = (x_{t,k}, y_{t,k}) \to \nabla \ell^B_t(\theta_t)$ kết hợp với độ trôi $\gamma_{t-1}$ (hoặc $\hat{\gamma}_t$) $\to \eta^*_t \to \theta_{t+1}$. |
| **Ý nghĩa trong Never-Ending Learning** | Đánh giá **hiệu năng trọn đời (lifetime performance)** thông qua tổng Dynamic Regret $\text{Reg}(T) = \sum_{t=1}^T \text{reg}_t$. Mô hình không được phép "gian lận" nhìn trước dữ liệu bước $t$. | Giúp mô hình liên tục khám phá lại không gian tham số khi có distribution shift ($\gamma_t > 0$ làm tăng $\eta^*_t$) và hội tụ mịn khi phân phối tĩnh ($\gamma_t = 0$ làm giảm $\eta^*_t \propto 1/t$). |
| **Liên hệ bài toán Y sinh (Phụ lục E)** | Mạng MLP 5 tầng ("Student" gọn nhẹ) phân loại trực tiếp tế bào tụy theo thời gian thực dựa trên 10 tín hiệu huỳnh quang RNA. | Mạng CNN lớn ("Teacher") cấp nhãn chuẩn $y_t$ trễ một nhịp; hệ thống tính Cross-Entropy Loss, kích hoạt Adam Restart / $\eta^*_t$ khi chuyển pha $P_1 \to P_4$ và cập nhật MLP. |
