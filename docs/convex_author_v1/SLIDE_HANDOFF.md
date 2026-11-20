# Slide handoff Convex

Tài liệu bổ sung sau freeze, gửi lead duyệt trước khi nhập deck chung. Vị trí theo nhiệm vụ được giao. Chưa có Sheet 3 hoặc deck để xác minh slide 8–9 hoặc phối template. T30 phối hợp Hòa theo T18. Các hình nằm ở [gói kết quả](../results/convex-author-v1/RESULTS.md).

**Trích dẫn IEEE [1].** M. Fahrbach, A. Javanmard, V. Mirrokni, and P. Worah, “Learning Rate Schedules in the Presence of Distribution Shift,” in *Proc. 40th Int. Conf. Mach. Learn.*, PMLR, vol. 202, 2023. [Bài báo](https://proceedings.mlr.press/v202/fahrbach23a.html)

**Trích dẫn IEEE [2].** M. Fahrbach *et al*., “Learning rate schedules,” source code, commit 32ae9da816fc26e6a728b27fcd20255536b2fc41, regression notebook, cell TaJuBJa1OMW4. [Notebook](https://github.com/fahrbach/learning-rate-schedules/blob/32ae9da816fc26e6a728b27fcd20255536b2fc41/src/regression/regression_experiments.ipynb)

**Trích dẫn IEEE [3].** Nhóm ML, “Convex Author v1,” post-freeze experiment records, Nov. 20, 2026. [Manifest](../results/convex-author-v1/manifest.json)

**Trích dẫn IEEE [4].** Nhóm ML, “CP04/CP05 frozen scientific results,” protocol convex-cp03-v1, Nov. 2026. [Frozen report](../results/cp04-cp05/RESULTS.md)

## Slide 16 · Cận dynamic regret của Theorem 4.2

- Oracle minimizes $F_t$ tại từng t.
- Cận gồm khoảng cách learner–oracle, phương sai batch và biến thiên oracle.
- Đặt điều kiện convexity, smoothness, bounded domain và LR có thể chọn theo lịch sử.

**Bảng/hình.** Dùng [bảng metric và khối giả thiết](METRIC_VALIDATION.md) kèm Eq. 15 trên slide. Footnote [1, §2, §4.1].

**Speaker notes.** Ta xét loss kỳ vọng của dữ liệu tại thời điểm t, rồi lấy chênh với lựa chọn tối ưu của riêng thời điểm đó. Cận của paper áp dụng cho SGD dưới các giả thiết về miền và phương sai. Gaussian benchmark có BCE lồi và Hessian population bị chặn bởi I/4. Notebook có bước LR bằng 0 và thiếu bước chiếu, nên khi trình bày phải tách phép minh họa thuật toán khỏi việc áp dụng định lý trực tiếp.

## Slide 17 · Hai ngưỡng và tín hiệu drift

- Proposition 4.3 giải Eq. 17 bằng cách kẹp LR trước vào khoảng hai ngưỡng.
- Notebook nhìn forward shift của generating reference, nhóm dùng learner EMA.
- Zero LR và zero EMA kết hợp có thể làm learner đứng yên.

**Bảng/hình.** [Learning-rate trajectories](../results/convex-author-v1/figures/learning_rates.png), chú thích mốc cập nhật đầu tiên tại t=10 theo chỉ số slide từ 1. Footnote [1, Eqs. 17–20, Remark 4.4], [2], [3].

**Speaker notes.** Shift trong lý thuyết là thay đổi của oracle population. Notebook dùng reference đã tạo sẵn và biết cả vector tiếp theo. EMA chỉ dùng dữ liệu đã thể hiện qua learner path. Tính nhân quả là lợi thế về thông tin đầu vào, còn độ bám drift và chất lượng LR cần đo riêng. A1 giữ initialization nguyên bản nên zero trở thành trạng thái hấp thụ.

## Slide 20 · Cận trên, M0 và hai population metrics

- Eq. 16 là cận kỳ vọng, Proposition 4.3 chọn schedule cho biểu thức cận.
- M0 so batch BCE với generating reference.
- M1 dùng population risk tại reference. M2 dùng population oracle số.
- M2−M1 là hiệu chỉnh do comparator.

**Bảng/hình.** [Bảng M0/M1/M2](METRIC_VALIDATION.md) và [đường hiệu chỉnh comparator](../results/convex-author-v1/figures/comparator_correction.png). Footnote [1, §4.1], [2], [3].

**Speaker notes.** Điều kiện kỳ vọng trong paper khác câu hỏi thực nghiệm “batch hiện tại có loss cao hơn reference bao nhiêu”. Nhiễu nhãn đi qua sigmoid làm oracle population co lại so với reference. So cùng trajectory và distribution thì có thể quy phần chênh M2–M1 cho comparator. Không dùng M0 của tác giả để tính phần trăm chênh với M2 của nhóm.

## Slide 21 · Chi phí train và đánh giá

- Mỗi bước SGD O(Bd), lưu phase thay vì giữ toàn bộ dataset.
- Native d512 có tối thiểu hai ma trận lớn 1.000 MiB mỗi ma trận trước các bản sao.
- Evaluator dùng tích phân 1D và cache theo reference norm.
- Pilot d512 Convex hoàn tất với 2.000 bước.

**Bảng/hình.** [Tài nguyên protocol](PROTOCOL.md) và [per-run elapsed](../results/convex-author-v1/data/per_run.csv). Footnote [2], [3].

**Speaker notes.** Ở triển khai NumPy, covariates của một phase d512 chỉ giữ 5 MiB float32, cộng các mảng tạm và histories. Native notebook tạo cả X và reference cho mọi mẫu và có thêm TensorFlow copies. Chi phí population nằm ở tích phân với từng learner norm nên nên dùng lại cache. Thời gian pilot thuộc cấu hình Windows CPU đã ghi trong provenance.

## Slide 23 · Bốn protocol và cổng kiểm chứng

- P giữ 45 CP04 và 15 CP05 đã đóng băng.
- R chạy notebook gốc riêng trong Colab.
- I kiểm replay rồi mới chạy 45 ca. A chỉ mở sau I.
- Bảy pilot và ba RNG diagnostics thuộc namespaces kiểm tra.

**Bảng/hình.** [Inventory](../results/convex-author-v1/data/inventory.csv) và [manifest F0–F5](../results/convex-author-v1/manifest.json). Footnote [2]–[4].

**Speaker notes.** Bất biến quan trọng là source snapshot và input cùng cặp. M0 cuối từ output lưu lịch sử cho ba chiều nhỏ là sanity check. Để tuyên bố parity cần cả trajectories, losses, grad và LR cùng inputs từ Colab. Kết quả pilot chỉ dùng như chứng cứ chạy và phát hiện câu hỏi tiếp theo. T19 còn cross-run độc lập của thành viên.

## Slide 26 · Kết quả Convex và góc nhìn d512

- P có Convex thấp nhất ở d2–128 và constant thấp nhất ở d512.
- I d512, ba seed ghép cặp: M2 Convex 1205.491 ± 2.173, constant 504.027 ± 0.127.
- M0 và M1 cũng cho cùng thứ hạng tại d512 trong I.
- P và I dùng phase grid và label-noise std khác.

**Bảng/hình.** [P frozen Figure 3](../results/cp04-cp05/figures/cp04_all_dimension_baselines.png), [mean và SD của I](../results/convex-author-v1/data/method_summary.csv) và [diagnostic d512 seed 0](../results/convex-author-v1/figures/d512_diagnostics.png). Footnote [3], [4].

**Speaker notes.** Giữ nguyên figure chính của P. Bảng I báo mean và SD mẫu qua ba seed. Trong mỗi seed, ba method cùng X và y. Các đồ thị dynamics ở d512 chỉ dùng seed 0. Cùng thứ hạng giữa P và I cho thấy d512 là tín hiệu đáng kiểm tra, trong khi phase grid, label-noise std và cách chạy khác nhau giới hạn kết luận nhân quả về CP04.

## Slide 29 · Sensitivity và ablation có đối chứng

- CP05 đã đo ell và noise trong protocol P.
- A kiểm bốn can thiệp trên ba seed và ba dimension.
- EMA giữ zero-update ở chín ca. Ở d512, M2 giảm 679.547 so với adaptive Convex, còn cao hơn constant.
- Thay LR đầu ở d512 giảm M2 khoảng 3.588; phép chiếu không được kích hoạt.

**Bảng/hình.** [CP05 noise](../results/cp04-cp05/figures/cp05_noise_sensitivity.png), [bảng A1–A4](../results/convex-author-v1/data/ablation_summary.csv) và [gradient/drift d512](../results/convex-author-v1/figures/d512_gradient_drift.png). Footnote [1, Remark 4.4], [3], [4].

**Speaker notes.** CP05 dùng ba seed trong P. A1 giữ mọi yếu tố khác và đóng băng learner trong thiết kế zero-start. Warm start sẽ là thiết kế riêng. A2 đổi cả data-generating distribution, vì vậy so sánh với control cần giữ rõ comparator tương ứng. Các can thiệp của A giới hạn nhận định về I và không tách được nguyên nhân của P.

## Slide 30 · Đối chiếu code tác giả và code nhóm

- Cell 23 chọn một run v4 mỗi d, không chạy baseline.
- Notebook dùng 200 phase points, lặp 10 bước, noise std=.1.
- P dùng ell=8, std=sqrt(.1), EMA, initial LR=1, projection.
- Colab native hoàn tất cả năm dimension. Replay d2/8/32 khớp LR từng bước và sai khác M0 tối đa 2.48e−5.

**Bảng/hình.** [Source audit matrix](SOURCE_AUDIT.md) và [Fidelity table](../results/convex-author-v1/data/fidelity.csv). Footnote [1, §6.1], [2]–[4].

**Speaker notes.** Nhóm giữ notebook source ngoài repo. Lần chạy Colab cho M0 d128 bằng 180.8743 và d512 bằng 1207.9739. Fixture từ replica instrumented kiểm chứng replay 2.000 bước ở ba dimension nhỏ hơn. So sánh này dùng M0 trên cùng batch và reference. Hòa kiểm lại source-audit và bằng chứng Colab trước khi nhập vào tài liệu T18.
