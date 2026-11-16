
> **Hướng dẫn đọc nhanh**
>
> **Các mục quan trọng cần đọc kỹ:** 0.1, 0.3, 1.3–1.5, 3.3–3.4, 4.1–4.3, 5.3, 5.5, 7.1–7.2, 8.10.
>
> **Các mục còn lại có thể đọc lướt** để nắm tổng quan. Phần thuật ngữ và phụ lục công thức chỉ cần tra cứu khi cần hoặc khi bắt đầu code.

## 0. Tổng quan

### 0.1. Research Question

Trong online learning, data distribution có thể thay đổi theo thời gian khiến bộ trọng số tối ưu liên tục dịch chuyển.

Điều này tạo ra sự đánh đổi khi lựa chọn learning rate (LR):

- **LR quá nhỏ:** Learner thích nghi chậm với optimum mới.
- **LR quá lớn:** Stochastic gradient noise có thể khiến quá trình cập nhật thiếu ổn định.
- **LR cố định hoặc decaying:** Không nhất thiết phù hợp với mức distribution shift tại từng thời điểm.

**Research Question:** Làm thế nào thiết kế learning rate schedule để SGD thích nghi với distribution shift và duy trì cumulative dynamic regret thấp?

### 0.2. Liên hệ giữa các Section

- **Section 1:** Đặt vấn đề learning rate dưới distribution shift.
- **Section 2:** Định nghĩa online learning, oracle model và dynamic regret.
- **Section 3:** Sử dụng Stochastic Differential Equations (SDE) để phân tích SGD trong Online Linear Regression và xây dựng Algorithm 1.
- **Section 4:** Sử dụng convexity cùng regret analysis để thiết kế schedule cho general convex losses.
- **Section 6:** Thực nghiệm đánh giá các learning rate schedules.

Section 4 không mở rộng trực tiếp Algorithm 1 sang mọi mô hình. Tác giả phát triển một phương pháp phân tích và lựa chọn learning rate riêng cho lớp convex losses rộng hơn.

### 0.3. Logic chính của Section 4

```text
Assumption 4.1
    ↓ Điều kiện để phân tích SGD
Theorem 4.2 — Eq. (15)
    ↓ Upper bound của expected cumulative dynamic regret
Eq. (16)
    ↓ Biến đổi upper bound để thuận tiện tối ưu
Eq. (17)
    ↓ Xây dựng bài toán chọn learning rate từng bước
Proposition 4.3 — Eq. (18)–(20)
    ↓ Tính thresholds và lựa chọn learning rate
Projected SGD
    ↓ Cập nhật trọng số learner
```

**Theorem 4.7** bổ sung lower-bound analysis để đánh giá độ chặt của kết quả lý thuyết.

**Cốt lõi:** Dynamic regret là mục tiêu đánh giá. Upper bound là cơ sở thiết kế schedule. Proposition 4.3 cung cấp quy tắc chọn LR thực tế.

## 1. Những khái niệm nền tảng

### 1.1. Learner và SGD

**Learner** là hệ thống học máy đang được huấn luyện. Trong reproduction của nhóm, learner có thể là Online Logistic Regression.

**Stochastic Gradient Descent (SGD)** cập nhật trọng số theo hướng ngược gradient. Learning rate quyết định độ lớn bước cập nhật.

Paper sử dụng **projected mini-batch SGD**:

$$
\theta_{t+1}
=
\Pi_\Theta
\left(
\theta_t-\eta_t\nabla\ell_t^B(\theta_t)
\right)
\tag{3}
$$

Trong đó:

- $\theta_t$: Trọng số hiện tại của learner.
- $\eta_t$: Learning rate tại bước $t$.
- $\nabla\ell_t^B$: Gradient trung bình từ mini-batch.
- $\Pi_\Theta$: Projection đưa trọng số về parameter domain hợp lệ.

**Convergence** là quá trình tiến tới một nghiệm tối ưu. Trong môi trường có distribution shift, learner còn phải **tracking** một optimum đang thay đổi.

### 1.2. General Convex Loss

**Convex loss** là hàm mất mát có tính convex theo trọng số. Tính chất quan trọng là mọi local minimum đều là global minimum.

**General convex losses** là lớp loss functions rộng hơn squared loss của Linear Regression.

Ví dụ, Linear Regression sử dụng squared loss còn Logistic Regression tuyến tính có thể sử dụng binary cross-entropy. Cả hai đều thuộc lớp bài toán convex trong điều kiện phù hợp.

Section 4 không giới hạn ở Linear Regression và có thể phân tích cả response shift lẫn covariate shift. Tuy nhiên, không phải mọi neural network đều có convex loss.

### 1.3. Oracle Model và Expected Loss

Tại bước $t$, dữ liệu được lấy từ phân phối $P_t$.

**Expected loss** là loss kỳ vọng trên phân phối hiện tại:

$$
\bar\ell_t(\theta)
=
\mathbb E_{z\sim P_t}[\ell(\theta,z)]
$$

**Oracle model** là bộ trọng số làm expected loss nhỏ nhất:

$$
\theta_t^*
=
\arg\min_{\theta\in\mathbb R^p}
\bar\ell_t(\theta)
\tag{1}
$$

Oracle là **comparator lý thuyết**, không phải mô hình thứ hai mà learner cần huấn luyện.

- **Learner:** Mô hình thực sự học và cập nhật trọng số.
- **Oracle:** Nghiệm tối ưu của expected loss tại mỗi bước.
- **Evaluator:** So sánh learner với oracle để đánh giá hiệu quả.

Learner thực tế thường không biết true oracle. Trong synthetic experiment, nhóm có thể xác định oracle từ cơ chế sinh dữ liệu nếu thiết lập thỏa các điều kiện phù hợp.

### 1.4. Dynamic Regret

Regret tại bước $t$ là chênh lệch expected loss giữa learner và oracle:

$$
\operatorname{reg}_t
=
\bar\ell_t(\theta_t)
-
\bar\ell_t(\theta_t^*)
$$

**Cumulative dynamic regret** là tổng regret qua $T$ bước:

$$
\operatorname{Reg}(T)
=
\sum_{t=1}^{T}\operatorname{reg}_t
\tag{2}
$$

Final loss chỉ phản ánh kết quả ở thời điểm cuối. Cumulative dynamic regret phản ánh hiệu quả thích nghi trong toàn bộ quá trình học.

**Lưu ý:** Regret là chênh lệch loss, không phải khoảng cách trực tiếp giữa hai bộ trọng số.

### 1.5. Distribution Shift và Oracle Drift

Paper định lượng distribution shift thông qua mức dịch chuyển của oracle:

$$
\gamma_t
=
\|\theta_t^*-\theta_{t+1}^*\|_2
\tag{6}
$$

- $\gamma_t=0$: Oracle không dịch chuyển.
- $\gamma_t$ lớn: Hai oracle models liên tiếp cách xa nhau hơn.

Đây là **oracle drift**, không phải khoảng cách trực tiếp giữa hai data distributions. Hai distributions khác nhau vẫn có thể có cùng oracle.

## 2. Assumption 4.1 — Điều kiện phân tích SGD

Assumption 4.1 gồm ba điều kiện giúp tác giả chứng minh regret bound.

### 2.1. Bounded Gradient Variance

**Stochastic gradient noise** là sai lệch giữa gradient được ước lượng từ dữ liệu lấy mẫu và gradient của expected loss.

Paper giả định gradient variance bị chặn bởi $\sigma^2$.

Với mini-batch gồm $B_t$ mẫu i.i.d., variance bound của gradient trung bình là:

$$
\frac{\sigma^2}{B_t}
$$

Batch size lớn hơn giúp giảm ảnh hưởng của stochastic gradient noise.

**Vai trò:** Kiểm soát thành phần noise trong regret upper bound.

### 2.2. L-smoothness

L-smoothness giới hạn mức thay đổi gradient khi trọng số thay đổi:

$$
\|\nabla\bar\ell_t(\theta_1)
-
\nabla\bar\ell_t(\theta_2)\|
\leq
L\|\theta_1-\theta_2\|
$$

Điều kiện này giúp kiểm soát ảnh hưởng của SGD update đối với expected loss.

Trong Theorem 4.2, learning rate còn phải thỏa:

$$
0<\eta_t\leq\frac1L
$$

**L-smoothness và convexity là hai điều kiện khác nhau.**

### 2.3. Bounded Parameter Set

Learner được giữ trong parameter domain $\Theta$ bằng projection.

Theo giả định chính, oracle cũng thuộc $\Theta$ và diameter của miền bị chặn bởi $D_{\max}$.

Do đó:

$$
\|\theta_t-\theta_t^*\|
\leq D_{\max}
$$

**Vai trò:** Giúp kiểm soát tracking distance trong regret analysis.

Paper cũng nêu một điều kiện thay thế liên quan đến tập $\Theta_0$.

**Kết luận:** Assumption 4.1 kiểm soát gradient noise, smoothness và khoảng cách đến oracle. Đây là các điều kiện chứng minh lý thuyết, không phải ba phương pháp xử lý distribution shift.

## 3. Theorem 4.2 — Regret Upper Bound

### 3.1. Upper Bound là gì?

**Upper bound** là giới hạn trên được chứng minh bằng toán học.

Theorem 4.2 đưa ra upper bound cho **expected cumulative dynamic regret**:

$$
\mathbb E[\operatorname{Reg}(T)]
\leq U
$$

$U$ là ký hiệu minh họa cho biểu thức upper bound.

Đây là giới hạn của regret tích lũy qua $T$ bước sau khi lấy expectation, không phải bound riêng cho một SGD update.

### 3.2. Equation (15) chứng minh điều gì?

Tác giả sử dụng projected SGD cùng convexity và Assumption 4.1 để xây dựng Equation (15).

Upper bound này gồm bốn nhóm thành phần:

1. **Tracking distance:** Khoảng cách giữa learner và oracle thay đổi qua các bước.
2. **Gradient noise:** Ảnh hưởng của stochastic gradient.
3. **Distribution shift:** Ảnh hưởng của oracle drift $\gamma_t$.
4. **Interaction term:** Sự tương tác giữa oracle drift và vị trí learner sau cập nhật.

Kết quả cho thấy regret phụ thuộc vào cả learning rate, gradient noise và sự dịch chuyển của oracle.

### 3.3. Vì sao chuyển sang Equation (16)?

Equation (15) chứa các đại lượng khó sử dụng trực tiếp khi chọn LR như khoảng cách đến oracle và trọng số sau cập nhật.

Tác giả dùng $D_{\max}$, oracle drift, Cauchy–Schwarz và telescoping sum để có Equation (16).

Bound mới thuận tiện hơn để phân tích ba yếu tố:

- Sự thay đổi learning rate giữa các bước.
- Gradient noise.
- Distribution shift.

Từ Equation (16), tác giả thiết lập bài toán tối ưu learning rate từng bước trong Equation (17).

### 3.4. Vai trò thực tế của Upper Bound

Learner thường không biết true oracle hoặc actual regret tương lai nên không thể trực tiếp chọn LR tối ưu dựa trên regret thật.

Tác giả thay thế bằng việc **tối ưu regret upper bound** để suy ra quy tắc lựa chọn LR.

Cần phân biệt:

- **Dynamic regret:** Đại lượng cần giảm.
- **Upper bound:** Cơ sở toán học để thiết kế schedule.
- **Proposition 4.3:** Quy tắc lựa chọn learning rate.
- **Evaluator:** Đo hiệu quả thực nghiệm bằng regret.

**Tối ưu upper bound không bảo đảm actual regret luôn nhỏ nhất.** Learner cũng không cần tính lại toàn bộ upper bound tại mỗi bước training.

## 4. Proposition 4.3 — Learning Rate Schedule

### 4.1. Quy tắc hai Thresholds

Từ bài toán tối ưu trong Equation (17), Proposition 4.3 xây dựng hai thresholds:

- **Lower threshold:** $\tau_{1,t}$
- **Upper threshold:** $\tau_{2,t}$

Learning rate mới được chọn bằng cách so sánh previous LR với khoảng thresholds:

$$
\eta_t^*
=
\min\{
\tau_{2,t},
\max\{\tau_{1,t},\eta_{t-1}^*\}
\}
\tag{20}
$$

Có ba trường hợp:

1. Previous LR thấp hơn lower threshold thì tăng đến lower threshold.
2. Previous LR nằm trong khoảng thì giữ nguyên.
3. Previous LR cao hơn upper threshold thì giảm đến upper threshold.

Ví dụ, nếu thresholds là $[0.02,0.08]$:

- LR cũ $0.01$ → LR mới $0.02$.
- LR cũ $0.05$ → Giữ nguyên.
- LR cũ $0.10$ → LR mới $0.08$.

Đây là ví dụ minh họa, không phải kết quả thực nghiệm của paper.

### 4.2. Sáu thông tin Scheduler cần sử dụng

| Thông tin | Vai trò | Nguồn |
|---|---|---|
| **Previous LR** $\eta_{t-1}$ | Giá trị tham chiếu để điều chỉnh LR | Training history |
| **Oracle drift** $\gamma_t$ | Định lượng distribution shift | True value hoặc estimate |
| **Gradient noise** $\sigma$ | Kiểm soát ảnh hưởng của noise | Bound hoặc estimate |
| **Batch size** $B_t$ | Quyết định mức giảm gradient variance | Training configuration |
| **Smoothness** $L$ | Kiểm soát bước cập nhật | Phân tích loss hoặc estimate |
| **Distance bound** $D_{\max}$ | Chặn khoảng cách learner–oracle | Parameter domain |

Các input này được dùng để tính thresholds theo Equations (18) và (19).

**Scheduler không cần biết actual regret hiện tại để chọn LR.**

### 4.3. Remark 4.4 — Estimated Distribution Shift

True $\gamma_t$ phụ thuộc vào hai oracle models liên tiếp nên thường không biết trong online learning thực tế.

Remark 4.4 đề xuất sử dụng **Exponential Moving Average (EMA)** của độ thay đổi trọng số learner:

$$
\hat\gamma_t
=
\beta\hat\gamma_{t-1}
+
(1-\beta)
\|\theta_t-\theta_{t-1}\|_2
$$

EMA dùng những trọng số đã quan sát để ước lượng mức shift.

Tuy nhiên, trọng số learner cũng có thể thay đổi do gradient noise. Vì vậy, **estimated shift không nhất thiết bằng true oracle drift**.

Việc dùng EMA giúp schedule vận hành khi không có oracle information nhưng không tự động bảo đảm mọi theoretical guarantee vẫn giữ nguyên.

### 4.4. Remark 4.5 — Ảnh hưởng của Distribution Shift

Khi $\gamma_t$ tăng và những đại lượng khác không đổi, hai thresholds tăng.

Điều này tạo khả năng lựa chọn LR lớn hơn khi môi trường thay đổi mạnh.

Tuy nhiên, schedule không bắt buộc tăng LR ở mọi bước. Nếu previous LR vẫn nằm trong khoảng thresholds thì LR được giữ nguyên.

**Kết luận:** Proposition 4.3 điều chỉnh LR dựa trên oracle drift và các đặc tính của SGD thay vì chỉ giảm LR theo thời gian.

## 5. Runtime — Schedule hoạt động thế nào?

### 5.1. Phân biệt các loại thông tin

Trong synthetic experiment, nhóm có thể biết toàn bộ cơ chế sinh dữ liệu. Nhưng realistic online learner không được sử dụng mọi thông tin đó.

Cần phân biệt ba nhóm:

- **Known hoặc configured:** Previous LR, batch size, parameter domain và các giá trị đã thiết lập.
- **Estimated:** Shift, gradient noise và smoothness khi không biết chính xác.
- **Evaluation-only:** True oracle, true oracle drift và dynamic regret được xác định hoặc ước lượng để đánh giá.

**Nguyên tắc:** Scheduler chỉ được dùng thông tin hợp lệ tại thời điểm chọn LR. True oracle information được giữ riêng cho evaluator.

### 5.2. Khởi tạo Training

Trước khi training, nhóm cần chuẩn bị:

1. Initial weights và initial learning rate.
2. Batch size và parameter domain.
3. Giá trị hoặc estimate của $L$, $\sigma$ và $D_{\max}$.
4. Shift estimator cùng trạng thái EMA ban đầu.
5. Số training steps, random seeds và cấu hình thí nghiệm.

Những giá trị này phải được ghi rõ để bảo đảm reproducibility.

### 5.3. Một bước Training

Tại iteration $t$, quá trình diễn ra theo thứ tự:

**Bước 1 — Đọc history:** Lấy previous LR, trọng số hiện tại và những thống kê đã có.

**Bước 2 — Estimate shift:** Sử dụng EMA hoặc một bound hợp lệ để xác định thông tin shift.

**Bước 3 — Tính thresholds:** LR Scheduler dùng các tham số của Proposition 4.3 để tính $\tau_{1,t}$ và $\tau_{2,t}$.

**Bước 4 — Chọn LR:** So sánh previous LR với hai thresholds để tăng, giữ nguyên hoặc giảm LR.

**Bước 5 — Nhận mini-batch:** Learner nhận dữ liệu hiện tại rồi tính gradient.

**Bước 6 — SGD update:** Sử dụng current LR và mini-batch gradient để tạo $\theta_{t+1}$.

**Bước 7 — Lưu history:** Lưu trọng số mới, LR và các thống kê cần thiết cho bước tiếp theo.

**Evaluation:** Evaluator sử dụng trạng thái learner $\theta_t$ trước update để tính regret tại bước $t$. Kết quả đánh giá không được truyền ngược vào realistic LR Scheduler.

```text
Training history
       ↓
Parameter / Shift Estimator
       ↓
LR Scheduler — Proposition 4.3
       ↓
Current Learning Rate
       ↓
Mini-batch Gradient + Projected SGD
       ↓
New Weights
       ↓
Update History → Next Iteration

Evaluation sử dụng learner và oracle
trong một luồng riêng để đo regret.
```

### 5.4. Ví dụ một Iteration

Giả sử tại iteration 20:

- Learner đang có trọng số $\theta_{20}$.
- Previous LR bằng $0.01$.
- Shift estimator đã sử dụng history đến iteration 19.
- Scheduler tính được thresholds $[0.02,0.08]$.

Vì LR cũ nhỏ hơn lower threshold, scheduler chọn:

$$
\eta_{20}=0.02
$$

Learner sau đó nhận mini-batch tại iteration 20, tính gradient và cập nhật từ $\theta_{20}$ thành $\theta_{21}$.

Evaluator có thể so sánh expected loss của $\theta_{20}$ với oracle tại iteration 20 để tính regret.

**Trong toàn bộ quá trình chọn LR, learner không cần biết regret của iteration 20.**

### 5.5. Pseudocode tổng quát

```text
INPUT
    Initial model weights
    Initial learning rate
    Batch size
    Parameter domain
    Smoothness and gradient noise parameters
    Distance bound
    Shift estimator
    Number of training steps

INITIALIZATION
    Khởi tạo model.
    Khởi tạo previous_learning_rate.
    Khởi tạo shift_estimator.
    Khởi tạo training history.

FOR t = 1 TO T

    # A. Chuẩn bị learning rate

    history = lịch sử trước iteration t

    estimated_shift = shift_estimator(history)

    lower_threshold, upper_threshold =
        compute_thresholds(
            estimated_shift,
            gradient_noise,
            batch_size,
            smoothness,
            distance_bound
        )

    current_learning_rate =
        clamp(
            previous_learning_rate,
            lower_threshold,
            upper_threshold
        )

    # B. Đánh giá trạng thái trước cập nhật

    evaluator ghi nhận trạng thái theta_t
    và đánh giá trên distribution hiện tại
    bằng dữ liệu đánh giá riêng

    # C. Nhận dữ liệu và cập nhật mô hình

    mini_batch = data_generator.get_batch(t)

    gradient =
        logistic_model.compute_gradient(
            current_weights,
            mini_batch
        )

    new_weights =
        projected_sgd.update(
            current_weights,
            gradient,
            current_learning_rate
        )

    # D. Lưu trạng thái

    current_weights = new_weights

    previous_learning_rate =
        current_learning_rate

    training_history lưu thông tin iteration t

END FOR

OUTPUT
    Learned weight trajectory
    Learning rate trajectory
    Regret metrics
    Experiment logs
```

Đây là pseudocode minh họa cách tích hợp module, không phải nguyên văn Algorithm 1.

Cách tổ chức evaluation phải bảo đảm mô hình được đánh giá tại đúng thời điểm và không dùng thông tin tương lai khi lựa chọn LR.

## 6. Theorem 4.7 — Lower bound

### 6.1. Lower bound là gì?

Nếu upper bound cho biết regret không vượt quá một giới hạn thì **lower bound** cho biết regret không nhỏ hơn một giới hạn trong các điều kiện tương ứng.

Ví dụ minh họa:

$$
40
\leq
\mathbb E[\operatorname{Reg}(T)]
\leq100
$$

Khi đó ta có một khoảng chứa expected cumulative regret.

Nhưng cần lưu ý rằng upper bound và lower bound phải được xét dưới các điều kiện thích hợp.

### 6.2. Tại sao tác giả cần lower bound?

Một upper bound có thể đúng nhưng quá lớn so với regret thực sự.

Ví dụ, nếu regret chỉ khoảng 10 mà upper bound là một triệu thì bound đó chưa cung cấp nhiều thông tin về độ khó thật của bài toán.

Lower-bound analysis giúp đánh giá upper bound có gần với giới hạn thực tế của regret hay không.

Trong Section 4.2, tác giả chứng minh lower bound có cấu trúc tương tự upper bound và khác nhau bởi những hằng số.

Điều này hỗ trợ nhận xét regret analysis là **nearly tight** trong phạm vi điều kiện của các định lý.

### 6.3. Assumption 4.6 — Strong convexity

Để chứng minh Theorem 4.7, paper bổ sung Assumption 4.6.

Loss function phải thỏa $\mu$-strong convexity.

Strong convexity là điều kiện mạnh hơn convexity thông thường. Nó bổ sung một giới hạn về độ cong của loss function.

Theorem 4.7 còn yêu cầu learning rate thỏa điều kiện tương ứng, trong đó có:

$$
\eta_t\leq\frac1\mu
$$

Không được áp dụng kết luận lower bound này cho mọi convex loss khi thiếu strong convexity.

### 6.4. Lower bound có cần code không?

Trong implementation chính của R3:

- Theorem 4.2 cung cấp upper-bound analysis.
- Equation (17) dẫn đến bài toán thiết kế LR.
- Proposition 4.3 cung cấp schedule.
- Theorem 4.7 phục vụ đánh giá độ chặt của phân tích.

**Lower bound không phải một bước trong LR Scheduler.**

Nếu nhóm muốn kiểm chứng các regret inequalities bằng thực nghiệm thì có thể viết thêm theoretical diagnostics. Đây là công việc bổ sung, không phải điều kiện bắt buộc để chạy schedule.

## 7. Reproduction — Dữ liệu và thiết lập thí nghiệm

### 7.1. Paper dùng dữ liệu gì?

Trong Section 6.1, tác giả sử dụng **synthetic data** cho các thí nghiệm high-dimensional regression.

Đối với Section 4, tác giả thử nghiệm Proposition 4.3 trên **Logistic Regression với binary cross-entropy loss**.

Vì vậy, nhóm không nhất thiết phải tìm một dataset bên ngoài để thực hiện thí nghiệm này.

Thay vào đó, cần xây dựng Data Generator theo thiết lập của paper.

### 7.2. Dữ liệu được sinh như thế nào?

Paper mô tả một oracle path $\theta_t^*$ theo Equation (26).

Quỹ đạo này làm bộ trọng số mục tiêu thay đổi theo thời gian.

Tại từng bước:

1. Data Generator xác định trạng thái của oracle path.
2. Sinh ma trận covariates $X$ cho mini-batch.
3. Sinh response dựa trên công thức của thí nghiệm.
4. Trả mini-batch cho learner.
5. Giữ thông tin oracle riêng để evaluator có thể sử dụng.

Trong thí nghiệm Linear Regression, paper sinh các phần tử của $X$ từ phân phối Uniform trên $(-1,1)$.

Với Logistic Regression, tác giả sử dụng cách sinh covariates tương tự và mô tả response:

$$
y=
\operatorname{sigmoid}
(X\theta_t^*+\varepsilon)
$$

Điểm cần chú ý là paper viết trực tiếp công thức sigmoid cho response. Nhóm không nên tự đổi thành quy trình lấy mẫu nhãn Bernoulli mà không ghi rõ sự thay đổi.

Để reproduce sát Figure 5, cần đối chiếu lại Equation (26), cách sinh noise và các cấu hình thực nghiệm.

### 7.3. Oracle trong synthetic experiment

Data Generator có thể biết oracle path vì nhóm chủ động thiết kế quá trình sinh dữ liệu.

Tuy nhiên, cần phân biệt **trọng số dùng để sinh dữ liệu** và **nghiệm thực sự tối ưu của population expected loss**.

Hai đại lượng này trùng nhau khi cơ chế sinh dữ liệu cùng loss function đáp ứng các điều kiện thích hợp.

Vì thế, khi code Regret Evaluator, nhóm cần kiểm tra comparator được chọn có đúng với định nghĩa oracle hay không.

Không nên mặc định mọi vector dùng để sinh dữ liệu đều tự động là population-risk minimizer trong bất kỳ cấu hình noise nào.

### 7.4. Những tham số tác giả nêu cho Logistic Regression

Trong Section 6.1, paper đưa ra các giá trị:

$$
\sigma^2=\frac d4
$$

$$
L=\frac14
$$

$$
D_{\max}=d
$$

Trong đó $d$ là số chiều đặc trưng.

Đây là các giá trị tác giả sử dụng trong thiết lập Logistic Regression được mô tả.

Chúng không phải hằng số đúng cho mọi dataset hoặc mọi cách xây dựng parameter domain.

### 7.5. Paper experiment và proposed experiment

**Mục tiêu reproduce sát paper**

Ưu tiên xây dựng Logistic Regression với synthetic data theo Section 6.1 và oracle path của Equation (26). Sau đó so sánh kết quả với Figure 5.

**Thí nghiệm bổ sung của nhóm**

Nhóm có thể xây dựng thêm hai kịch bản dễ kiểm soát:

- **Smooth drift:** Oracle thay đổi từ từ theo thời gian.
- **Abrupt shift:** Oracle giữ tương đối ổn định rồi dịch chuyển đột ngột.

Những kịch bản này giúp phân tích hành vi của adaptive LR, nhưng phải được ghi rõ là thí nghiệm bổ sung nếu khác thiết lập gốc.

## 8. Implementation Architecture — Những gì nhóm cần code

### 8.1. Kiến trúc tổng quát

Một reproduction experiment hoàn chỉnh nên tách thành hai hệ thống.

**Online Training**

Chịu trách nhiệm nhận dữ liệu, tính gradient, lựa chọn LR và cập nhật trọng số.

**Offline Evaluation**

Chịu trách nhiệm đo dynamic regret, kiểm tra các metric và so sánh kết quả giữa những learning rate schedules.

Hai hệ thống được phép trao đổi những thông tin cần thiết để ghi kết quả nhưng không được để oracle information đi ngược vào realistic LR Scheduler.

### 8.2. Đề xuất cấu trúc thư mục

Đây là kiến trúc đề xuất cho R3, không phải cấu trúc bắt buộc từ paper.

```text
project/
│
├── docs/
│   └── 04_convex.md
│
├── src/
│   └── convex/
│       │
│       ├── data_generator.py
│       │   Sinh online data stream
│       │   Tạo oracle path và distribution shift
│       │
│       ├── logistic_model.py
│       │   Logistic Regression
│       │   Loss, sigmoid và gradient
│       │
│       ├── projected_sgd.py
│       │   Cập nhật trọng số theo SGD
│       │   Projection vào parameter domain
│       │
│       ├── parameter_estimator.py
│       │   Estimated shift
│       │   Gradient noise và smoothness estimates
│       │   Quản lý distance bound
│       │
│       ├── lr_scheduler.py
│       │   Proposition 4.3
│       │   Tính thresholds
│       │   Chọn learning rate mới
│       │
│       ├── regret_evaluator.py
│       │   Expected loss
│       │   Oracle loss
│       │   Dynamic regret và cumulative regret
│       │
│       ├── baselines.py
│       │   Constant LR
│       │   Decaying LR
│       │
│       └── train.py
│           Kết nối các module
│           Điều phối online training loop
│
├── experiments/
│   └── convex_experiment.py
│       Chạy nhiều cấu hình và random seeds
│       So sánh adaptive LR với baselines
│
├── tests/
│   └── test_convex.py
│       Kiểm tra loss, gradient và scheduler
│
└── results/
    └── convex/
        Metrics, logs và biểu đồ
```

### 8.3. Data Generator cần code gì?

**File:** `data_generator.py`

**Nhiệm vụ:** Tạo dữ liệu đến theo thời gian và mô phỏng distribution shift.

**Input chính:**

- Random seed.
- Số chiều đặc trưng $d$.
- Batch size.
- Số training steps.
- Cấu hình oracle path.
- Cấu hình noise.

**Output chính:**

- Mini-batch $(X_t,y_t)$ cho learner.
- Thông tin về oracle path cho evaluator.
- Metadata về distribution hiện tại nếu cần.

**Cần triển khai:**

1. Hàm tạo oracle path theo Eq. (26).
2. Hàm sinh covariates.
3. Hàm sinh response theo thí nghiệm Logistic Regression.
4. Hàm trả mini-batch tại bước $t$.
5. Hàm cung cấp independent evaluation samples khi cần.

**Điểm cần kiểm tra:** Data Generator có thể lưu true oracle nhưng không được đưa future oracle vào LR Scheduler.

### 8.4. Logistic Model cần code gì?

**File:** `logistic_model.py`

**Nhiệm vụ:** Triển khai learner dùng Logistic Regression.

**Input:**

- Trọng số hiện tại.
- Mini-batch $(X_t,y_t)$.

**Output:**

- Predictions.
- Mini-batch loss.
- Gradient theo trọng số.

Với logistic model tuyến tính:

$$
p=\operatorname{sigmoid}(X\theta)
$$

Binary cross-entropy loss được sử dụng để đo sai lệch giữa prediction và target.

Với binary cross-entropy chuẩn và soft targets hợp lệ trong $[0,1]$, gradient theo trọng số có dạng:

$$
\nabla\ell^B(\theta)
=
\frac1B X^\top(p-y)
$$

Đây là công thức implementation thông dụng cho Logistic Regression tuyến tính. Khi reproduce, cần đối chiếu với loss và cách sinh target được sử dụng trong experiment.

**Điểm cần kiểm tra:** Gradient đúng chiều, loss ổn định về số học và không có lỗi broadcasting.

### 8.5. Projected SGD cần code gì?

**File:** `projected_sgd.py`

**Nhiệm vụ:** Cập nhật trọng số theo Eq. (3).

**Input:**

- Current weights.
- Mini-batch gradient.
- Current learning rate.
- Parameter domain.

**Output:** New weights.

Quy trình:

1. Tính bước cập nhật SGD.
2. Áp dụng projection nếu trọng số vượt khỏi miền hợp lệ.
3. Trả về bộ trọng số mới.

**Điểm cần kiểm tra:** Projection đúng với parameter domain mà nhóm đã lựa chọn.

Nếu dùng một miền tham số dạng quả cầu Euclidean thì cần cài đặt projection tương ứng. Không được tùy ý thêm projection rồi giữ nguyên các assumptions mà không kiểm tra lại.

### 8.6. Parameter Estimator cần code gì?

**File:** `parameter_estimator.py`

Đây là một trong những module quan trọng nhất của R3.

**Nhiệm vụ:** Chuẩn bị những đại lượng mà LR Scheduler cần.

**Input:**

- Training history.
- Previous weights.
- Current weights đã có từ bước trước.
- Parameter configuration.
- Các thống kê gradient được phép sử dụng.

**Output:**

- Estimated shift $\hat\gamma_t$.
- Gradient noise parameter $\sigma$ hoặc estimate.
- Smoothness constant $L$ hoặc estimate.
- Distance bound $D_{\max}$.
- Batch size hiện tại.

**Cần triển khai:**

1. EMA shift estimator theo Remark 4.4.
2. Cách lấy $\sigma$ từ cấu hình hoặc một phương pháp estimate.
3. Cách lấy $L$ từ đặc tính bài toán hoặc estimate.
4. Cách xác định $D_{\max}$ từ parameter domain.
5. Kiểm tra các estimates có hợp lệ tại thời điểm chọn LR không.

**Điểm cần nhớ:** $\sigma$ và $L$ là các bound parameters trong theorem. Một sample estimate không tự động là một bound chắc chắn. Nhóm phải ghi rõ phương pháp xác định giá trị được sử dụng.

### 8.7. LR Scheduler cần code gì?

**File:** `lr_scheduler.py`

Đây là **module trung tâm của Proposition 4.3**.

**Input:**

- Previous LR.
- Estimated distribution shift.
- Gradient noise parameter.
- Batch size.
- Smoothness constant.
- Distance bound.

**Output:** Current LR.

**Quy trình xử lý:**

**Bước 1.** Tính hai đại lượng trung gian $b_{1,t}$ và $b_{2,t}$.

**Bước 2.** Tính lower threshold $\tau_{1,t}$.

**Bước 3.** Tính upper threshold $\tau_{2,t}$.

**Bước 4.** So sánh previous LR với khoảng thresholds.

**Bước 5.** Trả về current LR theo Eq. (20).

Pseudocode:

```text
function select_learning_rate(
    previous_lr,
    estimated_shift,
    gradient_noise,
    batch_size,
    smoothness,
    distance_bound
):

    b1 = estimated_shift^2
         + 2 * distance_bound * estimated_shift

    b2 = (estimated_shift + distance_bound)^2

    lower_threshold = compute_tau(b1)
    upper_threshold = compute_tau(b2)

    if previous_lr < lower_threshold:
        return lower_threshold

    if previous_lr > upper_threshold:
        return upper_threshold

    return previous_lr
```

Đây là logic của schedule. Hàm `compute_tau` cần triển khai chính xác theo Equations (18) và (19).

**Những trường hợp cần kiểm thử:**

- Previous LR thấp hơn lower threshold.
- Previous LR nằm trong khoảng.
- Previous LR cao hơn upper threshold.
- Estimated shift bằng 0.
- Gradient noise parameter bằng 0.
- LR tiến gần giới hạn $1/L$.
- Công thức thresholds gặp vấn đề numerical stability.

### 8.8. Regret Evaluator cần code gì?

**File:** `regret_evaluator.py`

**Nhiệm vụ:** Đánh giá learner bằng dynamic regret.

**Input:**

- Trọng số learner tại bước $t$.
- Oracle model hoặc comparator hợp lệ.
- Distribution hiện tại hoặc independent evaluation samples.

**Output:**

- Estimated expected loss của learner.
- Expected loss của oracle.
- Dynamic regret tại bước $t$.
- Cumulative dynamic regret đến bước $t$.

**Quy trình:**

1. Lấy snapshot trọng số learner trước SGD update của bước $t$.
2. Tính hoặc ước lượng expected loss của learner.
3. Tính hoặc ước lượng expected loss của oracle.
4. Lấy chênh lệch để có regret tại bước $t$.
5. Cộng regret vào cumulative regret.
6. Ghi kết quả để tổng hợp sau experiment.

**Quan trọng:** Loss trên training mini-batch không tự động bằng expected loss của distribution.

Nếu expected loss không có công thức giải tích, nhóm có thể dùng independent evaluation samples từ đúng distribution tại bước đó.

Cũng cần phân biệt regret của một lần chạy với **expected cumulative regret** trong theorem. Kết quả nhiều lần chạy với các random seeds khác nhau giúp ước lượng ảnh hưởng của stochasticity.

### 8.9. Baselines và Experiment Runner

**File:** `baselines.py`

Triển khai các learning rate schedules dùng để so sánh:

- Constant LR.
- Decaying LR.
- Adaptive LR theo Proposition 4.3.

**File:** `convex_experiment.py`

Điều phối toàn bộ thí nghiệm:

1. Đọc experiment configuration.
2. Khởi tạo cùng data stream cho các methods.
3. Chạy từng learning rate schedule.
4. Ghi metrics theo thời gian.
5. Lặp lại với nhiều random seeds.
6. Tổng hợp mean và variability.
7. Tạo biểu đồ và đối chiếu với paper.

### 8.10. Luồng kết nối giữa các module

```text
                    EXPERIMENT RUNNER
                            |
                            v
                     DATA GENERATOR
                            |
                  Tạo distribution P_t
                            |
                  +---------+---------+
                  |                   |
                  v                   v
             TRAINING DATA       EVALUATION DATA
                  |                   |
                  |                   +---- True oracle
                  |                          và comparator
                  |
                  v
                LEARNER
                  |
           Training history
                  |
                  v
           PARAMETER ESTIMATOR
                  |
          estimated_shift
          sigma, L, D_max
                  |
                  v
              LR SCHEDULER
             Proposition 4.3
                  |
             Current LR
                  |
                  v
             PROJECTED SGD
                  ^
                  |
          Logistic loss và gradient
                  ^
                  |
             Mini-batch t
                  |
                  v
             NEW WEIGHTS
                  |
                  v
            UPDATE HISTORY
                  |
                  v
             NEXT ITERATION


              EVALUATION PATH

    Learner weights trước update
                  |
                  v
            REGRET EVALUATOR
                  ^
                  |
          Oracle / Evaluation Data
                  |
                  v
             REGRET METRICS
                  |
                  v
             RESULTS / PLOTS
```

Sơ đồ trên biểu diễn cách các thành phần phối hợp. Trong code thực tế, Data Generator và Evaluator có thể nằm trong cùng một experiment runner nhưng quyền truy cập thông tin phải được tách rõ.

**Online Training** chỉ dùng dữ liệu được phép sử dụng tại thời điểm ra quyết định.

**Offline Evaluation** được dùng oracle information để đo kết quả khi synthetic experiment cho phép.

### 8.11. Thứ tự triển khai đề xuất

**Giai đoạn 1. Xây dựng nền tảng**

Hoàn thành Data Generator, Logistic Model và Projected SGD.

Mục tiêu là kiểm tra learner có thể nhận mini-batch và cập nhật trọng số đúng.

**Giai đoạn 2. Xây dựng LR Scheduler**

Triển khai Proposition 4.3.

Trước khi tích hợp vào training, kiểm thử riêng logic của hai thresholds bằng những bộ input cố định.

**Giai đoạn 3. Xây dựng Parameter Estimator**

Triển khai EMA shift estimator và các cách cung cấp schedule parameters.

Kiểm tra history được sử dụng đúng thời điểm và không chứa future information.

**Giai đoạn 4. Tích hợp Online Training**

Kết nối Estimator → Scheduler → Projected SGD.

Ghi lại trọng số và learning rate của từng iteration để kiểm tra hành vi.

**Giai đoạn 5. Xây dựng Evaluation**

Triển khai Regret Evaluator độc lập với scheduler.

Tính cumulative regret theo đúng định nghĩa của Section 2.

**Giai đoạn 6. Chạy Experiments**

Chạy constant LR, decaying LR và adaptive LR trên cùng data streams.

So sánh kết quả bằng cumulative regret cùng learning rate curves.

**Giai đoạn 7. Đối chiếu paper**

Đối chiếu với Section 6.1 và Figure 5.

Ghi rõ những kết quả phù hợp, những khác biệt và các giới hạn của reproduction.

### 8.12. Trọng tâm trách nhiệm R3

Trong một reproduction hoàn chỉnh, nhóm phải xây dựng toàn bộ hệ thống.

Tuy nhiên, **trọng tâm Section 4** là:

1. Hiểu Assumption 4.1 và Theorem 4.2.
2. Triển khai các công thức của Proposition 4.3.
3. Xây dựng hoặc phối hợp xây dựng Parameter Estimator.
4. Kiểm tra LR Scheduler hoạt động đúng.
5. Phối hợp với Regret Evaluator để đánh giá hiệu quả.
6. Giải thích được tại sao schedule thay đổi LR dưới distribution shift.

Các module Data Generator, Logistic Model và Experiment Runner có thể được chia sẻ hoặc phân công cho thành viên khác.

## 9. Some Questions 

### 1. Vì sao decaying learning rate không phải lúc nào cũng phù hợp khi có distribution shift?

Decaying LR làm learning rate giảm dần để hỗ trợ convergence trong những môi trường tương đối ổn định.

Khi distribution shift khiến oracle model thay đổi, LR đã giảm quá nhỏ có thể khiến learner phản ứng chậm với optimum mới. Vì vậy, một schedule chỉ giảm theo thời gian không phải lúc nào cũng thích hợp.

Adaptive schedule có thể điều chỉnh LR theo mức shift và các đặc tính của quá trình SGD.

### 2. Vì sao dynamic regret phản ánh hiệu quả online learning tốt hơn chỉ nhìn final loss?

Final loss chỉ cho biết mô hình hoạt động thế nào tại thời điểm cuối.

Cumulative dynamic regret đo tổng chênh lệch expected loss giữa learner và oracle tại từng bước.

Ví dụ, hai learner có thể đạt final loss gần giống nhau nhưng một learner từng chịu loss rất lớn trong phần đầu của quá trình học.

Dynamic regret phản ánh sự khác biệt đó nên phù hợp để đánh giá khả năng thích nghi xuyên suốt online learning.

### 3. Section 3 dùng SDE để làm gì và vì sao Section 4 cần phân tích khác?

Section 3 dùng SDE để xấp xỉ động lực học SGD trong Online Linear Regression dưới distribution shift.

Thông qua phân tích quá trình này, tác giả nghiên cứu tracking error và thiết kế Algorithm 1.

Section 4 muốn áp dụng phương pháp nghiên cứu cho một lớp convex loss functions rộng hơn. Vì vậy, tác giả sử dụng convexity cùng các giả định về SGD để xây dựng regret bounds và learning rate schedule riêng.

### 4. General convex loss khác gì với Linear Regression?

Linear Regression là một mô hình dự đoán cụ thể và thường sử dụng squared loss.

General convex loss là một lớp hàm mất mát có tính convex theo trọng số.

Lớp này bao gồm nhiều bài toán khác ngoài Linear Regression, chẳng hạn Logistic Regression tuyến tính với binary cross-entropy.

Do đó, Section 4 có phạm vi phân tích rộng hơn Section 3.

### 5. Assumption 4.1 kiểm soát ba yếu tố nào?

Assumption 4.1 gồm ba điều kiện.

**Bounded gradient variance** kiểm soát stochastic gradient noise.

**L-smoothness** giới hạn mức thay đổi gradient khi trọng số thay đổi.

**Bounded parameter set** giới hạn khoảng cách liên quan đến learner và oracle.

Nhờ các điều kiện này, tác giả có thể phân tích SGD và chứng minh regret upper bound.

### 6. Theorem 4.2 đưa ra upper bound cho regret nào?

Theorem 4.2 đưa ra upper bound cho **expected cumulative dynamic regret**.

Đây là tổng dynamic regret qua nhiều bước sau khi lấy expectation theo tính ngẫu nhiên của quá trình học.

Nó không phải một upper bound chỉ dành cho regret của một SGD update.

### 7. Vì sao tác giả không trực tiếp tối ưu actual regret?

Actual regret phụ thuộc vào expected loss của learner và oracle model.

Trong online learning thực tế, learner thường không biết chính xác oracle hiện tại và tương lai nên không thể trực tiếp chọn learning rate dựa trên actual regret tương lai.

Tác giả dùng một regret upper bound đã được chứng minh để xây dựng bài toán tối ưu learning rate theo từng bước.

Tối ưu upper bound tạo ra một quy tắc có cơ sở lý thuyết nhưng không bảo đảm actual regret của mọi lần chạy là nhỏ nhất.

### 8. Vì sao Equation (15) phải chuyển sang Equation (16)?

Equation (15) chứa các đại lượng khó quan sát tại thời điểm lựa chọn LR, chẳng hạn khoảng cách đến oracle và trọng số sau cập nhật.

Tác giả dùng $D_{\max}$, oracle drift và các bất đẳng thức toán học để xử lý những đại lượng đó.

Equation (16) có dạng thuận tiện hơn để tối ưu theo learning rate.

Đây là bước trung gian quan trọng trước khi xây dựng Equation (17) và Proposition 4.3.

### 9. Proposition 4.3 cần những thông tin nào để chọn LR?

Schedule sử dụng sáu thông tin chính:

1. Previous learning rate.
2. Distribution shift hoặc estimate của nó.
3. Gradient noise parameter $\sigma$.
4. Batch size $B_t$.
5. Smoothness constant $L$.
6. Distance bound $D_{\max}$.

Những thông tin này được dùng để tính hai thresholds.

Learner sau đó so sánh previous LR với khoảng thresholds để chọn current LR.

### 10. Learner có cần biết true regret hoặc true oracle model không?

Learner không cần tính actual dynamic regret ở mỗi bước để chọn learning rate.

True oracle chủ yếu là comparator trong phân tích lý thuyết và evaluation.

Tuy nhiên, công thức lý thuyết của Proposition 4.3 có sử dụng oracle drift $\gamma_t$, vốn thường không thể quan sát trực tiếp.

Vì vậy, realistic implementation cần dùng estimate hoặc bound hợp lệ thay cho true shift.

### 11. Hai thresholds quyết định tăng, giữ hay giảm LR thế nào?

Proposition 4.3 xây dựng lower threshold và upper threshold.

Nếu previous LR nhỏ hơn lower threshold, scheduler tăng LR lên lower threshold.

Nếu previous LR nằm trong khoảng, scheduler giữ nguyên.

Nếu previous LR lớn hơn upper threshold, scheduler giảm LR xuống upper threshold.

Quy tắc này được biểu diễn trong Equation (20).

### 12. EMA estimate có giới hạn gì?

EMA dùng độ thay đổi của trọng số learner trong những bước đã hoàn tất để estimate oracle drift.

Nhưng trọng số learner không chỉ thay đổi do distribution shift. Gradient noise cũng có thể làm trọng số dịch chuyển.

Vì vậy, estimated shift không nhất thiết bằng true $\gamma_t$.

Sử dụng EMA giúp schedule hoạt động thực tế hơn nhưng không tự động duy trì mọi theoretical guarantee.

### 13. Lower bound trong Theorem 4.7 dùng để làm gì?

Lower bound giúp đánh giá regret upper bound có chặt hay không.

Theorem 4.7 bổ sung strong convexity và chứng minh một lower-bound expression có cấu trúc gần upper bound.

Kết quả hỗ trợ nhận xét rằng regret analysis là nearly tight trong phạm vi các assumptions.

Lower bound không tham gia việc chọn LR ở runtime.

### 14. Những thành phần nào phải code để reproduce Section 4?

Một reproduction experiment hoàn chỉnh cần Data Generator để sinh dữ liệu có shift.

Logistic Model và Projected SGD đảm nhiệm quá trình học.

Parameter Estimator cung cấp các đại lượng cần thiết cho LR Scheduler.

LR Scheduler triển khai Proposition 4.3 để chọn learning rate.

Regret Evaluator đánh giá hiệu quả learner và Experiment Runner điều phối thí nghiệm cùng các baselines.

**Trọng tâm R3 là Parameter Estimator, LR Scheduler và phần phối hợp đánh giá regret.**

### 15. Làm sao tránh future-information leakage?

Phải tách biệt những thông tin được dùng để training với những thông tin chỉ phục vụ evaluation.

LR Scheduler chỉ được sử dụng history và các parameters hợp lệ tại thời điểm chọn LR.

Evaluator có thể sử dụng true oracle trong synthetic experiment để tính regret.

Không được đưa future oracle, future data hoặc evaluation results ngược vào realistic LR Scheduler.

Nếu thí nghiệm dùng true shift để kiểm tra một trường hợp lý tưởng thì phải ghi rõ đây là **oracle-information scenario**.

## 10. Thuật ngữ và ký hiệu quan trọng

**Learner:** Hệ thống Machine Learning đang cập nhật trọng số.

**SGD:** Stochastic Gradient Descent, thuật toán cập nhật trọng số bằng stochastic gradient.

**Learning rate:** Độ lớn bước cập nhật trọng số.

**Learning rate schedule:** Quy tắc thay đổi LR qua các bước training.

**Convergence:** Quá trình tiến tới một nghiệm hoặc trạng thái đích.

**Tracking:** Khả năng theo kịp optimum đang thay đổi.

**Convexity:** Tính chất bảo đảm local minimum cũng là global minimum.

**General convex loss:** Lớp loss functions convex theo trọng số.

**Stochastic gradient noise:** Sai lệch giữa sampled gradient và expected gradient.

**Oracle model $\theta_t^*$:** Bộ trọng số tối ưu theo expected loss tại bước $t$.

**Expected loss $\bar\ell_t$:** Loss kỳ vọng trên data distribution tại bước $t$.

**Dynamic regret:** Chênh lệch expected loss giữa learner và oracle.

**Cumulative dynamic regret:** Tổng dynamic regret qua nhiều bước.

**Distribution shift $\gamma_t$:** Được định lượng bằng oracle drift trong paper.

**Upper bound:** Giới hạn trên được chứng minh cho expected cumulative regret.

**Lower bound:** Giới hạn dưới được chứng minh trong các điều kiện tương ứng.

**L-smoothness:** Điều kiện giới hạn mức thay đổi gradient.

**Strong convexity:** Điều kiện convexity mạnh hơn.

**Projection:** Phép đưa trọng số về parameter domain hợp lệ.

**Thresholds:** Hai giá trị dùng để quyết định LR mới.

**SDE:** Stochastic Differential Equation, công cụ phân tích động lực học SGD trong Section 3.

**EMA:** Exponential Moving Average, phương pháp trung bình động dùng để estimate shift.

**Future-information leakage:** Việc sử dụng thông tin tương lai không hợp lệ khi đưa ra quyết định.

## Phụ lục kỹ thuật — Công thức đối chiếu paper

Phần phụ lục giữ các công thức cần thiết để implementation và đối chiếu với nghiên cứu gốc.

### A. Theorem 4.2 — Equations (15) và (16)

Đặt:

$$
D_t=\|\theta_t^*-\theta_t\|_2
$$

$$
a_t=2\eta_t-L\eta_t^2>0
$$

**Equation (15)**

$$
\begin{aligned}
\mathbb E[\operatorname{Reg}(T)]
\leq
\sum_{t=1}^{T}\mathbb E\Bigg[
&\frac{D_t^2-D_{t+1}^2}{a_t}
+\frac{\sigma^2\eta_t^2}{B_ta_t}\\
&+\frac{\gamma_t^2}{a_t}\\
&+\frac{2}{a_t}
\left\langle
\theta_t^*-\theta_{t+1}^*,
\theta_{t+1}-\theta_t^*
\right\rangle
\Bigg].
\end{aligned}
\tag{15}
$$

Equation (15) là regret upper bound ban đầu.

**Equation (16)**

Đặt $(x)_+=\max(x,0)$.

$$
\begin{aligned}
\mathbb E[\operatorname{Reg}(T)]
\leq{}&
D_{\max}^2
\mathbb E\left[
\frac1{a_1}
+
\sum_{t=2}^{T}
\left(
\frac1{a_t}-\frac1{a_{t-1}}
\right)_+
\right]\\
&+
\sum_{t=1}^{T}
\mathbb E\left[
\frac{
\sigma^2\eta_t^2/B_t
+\gamma_t^2
+2D_{\max}\gamma_t
}{a_t}
\right].
\end{aligned}
\tag{16}
$$

Equation (16) là dạng upper bound thuận tiện hơn cho việc thiết kế schedule.

### B. Equation (17) và Proposition 4.3

Đặt:

$$
a(\eta)=2\eta-L\eta^2
$$

**Equation (17)**

$$
\begin{aligned}
\eta_t^*
\in\arg\min_{0<\eta\leq1/L}
\Bigg\{
&D_{\max}^2
\left(
\frac1{a(\eta)}
-
\frac1{a(\eta_{t-1}^*)}
\right)_+\\
&+
\frac{
\sigma^2\eta^2/B_t
+\gamma_t^2
+2D_{\max}\gamma_t
}{a(\eta)}
\Bigg\}.
\end{aligned}
\tag{17}
$$

Equation (17) là bài toán tối ưu phần tương ứng của upper bound theo learning rate tại bước hiện tại.

**Equations (18) và (19)**

Đặt:

$$
b_{1,t}
=
\gamma_t^2+2D_{\max}\gamma_t
$$

$$
b_{2,t}
=
(\gamma_t+D_{\max})^2
$$

Với $\sigma>0$:

$$
\tau(b)
=
\frac{B_t}{2\sigma^2}
\left(
\sqrt{
b^2L^2+\frac{4\sigma^2b}{B_t}
}
-bL
\right)
$$

Khi đó:

$$
\tau_{1,t}=\tau(b_{1,t})
\tag{18}
$$

$$
\tau_{2,t}=\tau(b_{2,t})
\tag{19}
$$

Hai thresholds này được sử dụng trong quy tắc chọn learning rate.

**Equation (20)**

$$
\eta_t^*
=
\begin{cases}
\tau_{1,t}
&
\eta_{t-1}^*\leq\tau_{1,t}
\\
\eta_{t-1}^*
&
\tau_{1,t}\leq\eta_{t-1}^*\leq\tau_{2,t}
\\
\tau_{2,t}
&
\eta_{t-1}^*\geq\tau_{2,t}
\end{cases}
\tag{20}
$$

### C. Numerical stability của threshold formula

Công thức $\tau(b)$ chứa phép trừ giữa hai số có thể rất gần nhau.

Khi code, điều này có thể gây mất độ chính xác số học.

Với $\sigma>0$ và $b>0$, một biến đổi đại số tương đương là:

$$
\tau(b)
=
\frac{2b}{
\sqrt{b^2L^2+4\sigma^2b/B_t}+bL
}
$$

Đây là biến đổi phục vụ implementation, không phải một phương trình mới được đánh số trong paper.

Nếu $b=0$ và $\sigma>0$ thì $\tau(0)=0$.

Nếu $\sigma=0$, cần xử lý trường hợp suy biến bằng giới hạn thích hợp hoặc một quy tắc được xác định rõ.

Với $L>0$ và $b>0$, giới hạn khi $\sigma\to0$ của biểu thức thresholds là $1/L$.

Những edge cases này cần được kiểm thử trước khi tích hợp scheduler vào training.

### D. Theorem 4.7 — Equation (21)

Assumption 4.6 yêu cầu $\mu$-strong convexity.

Đặt:

$$
a'_t
=
2\left(
\eta_t
+
\frac{L}{\mu}\eta_t
-
L\eta_t^2
\right)
$$

Với các điều kiện của Theorem 4.7, bao gồm $\eta_t\leq1/\mu$:

$$
\begin{aligned}
\mathbb E[\operatorname{Reg}(T)]
\geq
\sum_{t=1}^{T}\mathbb E\Bigg[
&\frac{D_t^2-D_{t+1}^2}{a'_t}
+\frac{\sigma^2\eta_t^2}{B_ta'_t}\\
&+\frac{\gamma_t^2}{a'_t}\\
&+\frac{2}{a'_t}
\left\langle
\theta_t^*-\theta_{t+1}^*,
\theta_{t+1}-\theta_t^*
\right\rangle
\Bigg].
\end{aligned}
\tag{21}
$$

Lower bound hỗ trợ phân tích độ chặt của regret upper bound.

Nó không phải công thức được dùng để cập nhật learning rate trong runtime.

### E. Proof Sketch — Logic chính của chứng minh

**Bước 1. Projected SGD**

Phân tích sự thay đổi squared distance giữa learner và oracle sau một SGD update.

**Bước 2. Convexity và smoothness**

Sử dụng tính chất convex và L-smoothness để liên hệ gradient với expected loss gap.

**Bước 3. Gradient variance bound**

Lấy expectation và kiểm soát stochastic gradient noise bằng variance bound.

**Bước 4. Oracle drift**

Phân rã ảnh hưởng của sự thay đổi oracle và interaction term để có Equation (15).

**Bước 5. Cauchy–Schwarz và telescoping sum**

Chặn các đại lượng khó quan sát, chuyển từ Equation (15) sang Equation (16) rồi tối ưu tuần tự để suy ra Proposition 4.3.

Phần chứng minh chi tiết nằm trong Appendix B của paper.

## References

Fahrbach, M., Javanmard, A., Mirrokni, V. và Worah, P. (2023). *Learning Rate Schedules in the Presence of Distribution Shift*. Proceedings of the 40th International Conference on Machine Learning. PMLR 202.

- Paper chính thức: https://proceedings.mlr.press/v202/fahrbach23a.html
- PDF: https://proceedings.mlr.press/v202/fahrbach23a/fahrbach23a.pdf
- Phần chứng minh: Appendix B
- Thí nghiệm Logistic Regression: Section 6.1 và Figure 5
