# Comparator, sai số và giới hạn của cận lý thuyết

Trong phân phối $X\sim N(0,I)$, $\epsilon\sim N(0,s^2)$ độc lập, nhãn mềm $Y=\operatorname{sigmoid}(X^\top v_t+\epsilon)$. Rủi ro $F_t(w)=E[\operatorname{softplus}(X^\top w)-Y X^\top w]$. Cả ba metric cộng từ 1 đến T và dùng learner pre-update.

| Metric | Loss và comparator | Kiểu số |
|---|---|---|
| M0 | Hiệu BCE trung bình cùng batch, learner và $v_t$ dạng float32 | BCE, hiệu, cumsum float32 |
| M1 | $\sum_t[F_t(\theta_t)-F_t(v_t)]$, với $v_t$ sinh dữ liệu float64 | Quadrature float64 |
| M2 | $\sum_t[F_t(\theta_t)-F_t(w_t^{\star,\mathrm{approx}})]$ | Quadrature và oracle radial float64 |

M0 giữ dấu và có nhiễu theo batch. Với đường đi đã lưu, tích phân M1/M2 là deterministic. M0 dùng cast float32 khi đánh giá v, giống notebook. Lỗi tổng float32 có thể tích lũy. M2 là ước lượng dùng oracle số, và quan sát trên một learner path là một hiện thực của quá trình học.

Đặt $r=\Vert v\Vert$, $q=\sqrt{r^2+s^2}$, $h(a)=E[Z\operatorname{sigmoid}(aZ)]$ với $Z\sim N(0,1)$. Từ kỳ vọng có điều kiện của Gaussian, $E[XY]=v h(q)/q$. Phần softplus phụ thuộc chuẩn của w, và phần tuyến tính dẫn đến nghiệm hướng v với $h(\Vert w^\star\Vert)=r h(q)/q$. Với $s>0,r>0$, $h(a)/a=E[\operatorname{sigmoid}'(aZ)]$ giảm theo $a>0$. Vì vậy nghiệm radial nhỏ hơn r. Nếu r hoặc s bằng 0, oracle trùng reference. A4 dùng ball tâm 0 và kiểm oracle radial nằm trong miền đã khai báo. [Mã evaluator](../../core/convex_population.py)

Kiểm hiệu tức thời giữa hai metric population bằng $F_t(v_t)-F_t(w_t^{\star,\mathrm{approx}})$ với dung sai số học 3e−15. Test tích phân lồng hai Gaussian kiểm riêng loss, gradient radial và tính co lại của oracle. File per-run lưu ngân sách quadrature M1/M2, max ước lượng lỗi tham số oracle và tổng ước lượng suboptimality. Các ngân sách là ước lượng số, chưa có bảo đảm interval. Không dùng Monte Carlo trong evaluator mới, nên không báo MC SE. SD giữa seed chỉ hợp lệ khi có ít nhất hai seed, và không dùng ngân sách tích phân thay cho SD. [Kiểm thử](../../tests/test_convex_author_alignment.py)

## Theorem 4.2 và Proposition 4.3

Theorem 4.2 đưa cận trên kỳ vọng dynamic regret theo population oracle, với lịch LR thích nghi theo lịch sử, $a_t=2\eta_t-L\eta_t^2>0$, LR ≤ 1/L, phương sai gradient hữu hạn theo cận và điều kiện miền cho projected SGD. Eq. 15 được nới thành Eq. 16 qua diameter và drift. Eq. 17 tối ưu từng bước phần cận Eq. 16 theo LR và LR trước đó. Eq. 18–20 giải bài toán bằng hai threshold. Objective của Proposition 4.3 là biểu thức cận trên, nên các bảng thực nghiệm không kiểm chứng độ chặt của cận. [Fahrbach và cộng sự, §4.1](https://proceedings.mlr.press/v202/fahrbach23a/fahrbach23a.pdf#page=7)

Ở benchmark Gaussian, loss lồi và Hessian population $E[p(1-p)XX^\top]\preceq I/4$, vì $p(1-p)\leq1/4$. Batches được sinh IID trong mỗi pha theo thiết kế. Giá trị $\sigma^2=d/4$ dùng theo §6.1, chưa kiểm chứng như cận có điều kiện đồng đều trên learner paths của notebook. Notebook dùng LR zero lúc đầu và không chiếu, nên các giả thiết LR dương và miền chiếu không được thiết lập ở đây. Reference drift dùng kiến thức trước về quỹ đạo tổng hợp. Lịch cố định deterministic có thể biết trước ở benchmark, còn learner online thường không quan sát ref tiếp theo. EMA của Remark 4.4 dùng được theo lịch sử, chưa được xác nhận là upper bound của drift oracle. [Paper, §2 và §4.1](https://proceedings.mlr.press/v202/fahrbach23a/fahrbach23a.pdf)
