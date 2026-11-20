# Kết quả Convex sau đóng băng

**COMPLETED_WITH_LIMITATIONS.** Hoàn thành 7 pilot và 3 RNG diagnostic, mỗi ca 2.000 bước. Colab R còn 5 ca. I và A đang chờ fixture parity ở d=2/8/32. [Inventory theo họ](data/inventory.csv) · [Cổng kiểm chứng và W&B](manifest.json)

## Pilot, seed 0

| d | Method | M0 | M1 | M2 |
|---:|---|---:|---:|---:|
| 2 | Convex | 1.035101 | 1.007194 | 1.007335 |
| 8 | Convex | 3.763831 | 3.774716 | 3.774993 |
| 32 | Convex | 47.809792 | 47.881472 | 47.881848 |
| 128 | Convex | 180.454391 | 181.604014 | 181.604411 |
| 512 | Convex | 1212.026001 | 1205.753149 | 1205.753518 |
| 512 | Constant | 504.170685 | 503.880314 | 503.880683 |
| 512 | Inverse-square-root | 527.088989 | 527.193086 | 527.193455 |

Các ca d512 dùng cùng hash X và y, cùng weights zero. Convex có M2 cao hơn constant 701.873 và LR ≥3.96 trong 99.55% bước. Norm post-update lớn nhất 7.817, dưới bán kính 256 của A4. Pilot d128 cũng ở vùng LR ≥3.96 trong 99.55% bước. Quan sát này giúp nhận diện độ nhạy cần kiểm tra ở d512. Mức LR cao có thể đi cùng sai số SGD/tracking lớn ở d/B=2, và cần can thiệp để tách cơ chế. [Per-run CSV](data/per_run.csv) · [Phương pháp và thống kê](data/method_summary.csv) · [Hình d512](figures/d512_diagnostics.png)

P có cùng thứ hạng mô tả ở d512 theo population-oracle metric B, với Convex 1385.009, constant 504.895, inverse-sqrt 520.381 qua ba seed. P khác phase grid, std, signal và projection. Đồ thị đặt các giá trị M2/B cạnh nhau với phân phối ghi rõ. Tính phần trăm cải thiện qua hai protocol sẽ trộn các khác biệt này. [P frozen](../cp04-cp05/RESULTS.md) · [Đối chiếu mô tả](figures/project_context_m2.png)

Trong pilot d512, Convex bắt đầu cập nhật ở batch 10 và mọi phương pháp thấy cùng 199 lần nhảy reference. Ở batch 901–1100, độ lớn forward shift trung bình là 0.000594. Convex giữ LR trung bình 3.9986, có norm gradient trung bình 0.5176, norm cập nhật trung bình 2.0697 và cộng thêm 65.4663 vào M2. Constant có norm gradient 0.03619, norm cập nhật 0.00362 và cộng thêm 0.00774 vào M2 trong cùng cửa sổ. Gradient ở đây được đo tại learner path của từng phương pháp, nên chênh lệch gradient bao gồm trạng thái learner và nhiễu batch. Projection không được bật trong pilot và không có lần chiếu nào. Cửa sổ ít drift cho thấy rate cao vẫn đi cùng cập nhật lớn. Nó chưa xác định nguyên nhân của chênh lệch trong CP04. [Cửa sổ d512](data/d512_windows.csv) · [Gradient, update, drift và M2 mỗi bước](figures/d512_gradient_drift.png)

## Figure 5-bottom và kiểm tra parity

| d | M0 lưu trong snapshot notebook | M0 NumPy MT19937 độc lập | Sai khác tuyệt đối của số cuối đã in |
|---:|---:|---:|---:|
| 2 | 0.98582214 | 0.985822439 | 2.99e−7 |
| 8 | 3.7352686 | 3.735282898 | 1.43e−5 |
| 32 | 48.065575 | 48.065574646 | 3.54e−7 |
| 128 | N/A | N/A | N/A |
| 512 | N/A | N/A | N/A |

Ba đối chiếu trên dùng đúng cấu hình seed theo phase trong source và so với số cuối đã in trong snapshot. Chưa có tensor hoặc đường đi Colab mới để tính sai số tại từng bước. Table fidelity giữ trống các trường chỉ áp dụng khi replay đã ghép cặp. Không dùng số lịch sử làm test truth. [Fidelity table](data/fidelity.csv) · [Source audit](../../convex_author_v1/SOURCE_AUDIT.md)

## Comparator và ablation

Hiệu M2−M1 ở d512 là khoảng 0.000369 cho cả ba phương pháp. Nó chỉ phụ thuộc comparator và phân phối nên không giải thích chênh lệch học ở pilot. Ngân sách quadrature M2 lớn nhất trong 10 ca đo là 3.98e−8, và max oracle parameter error ước lượng là 1.90e−8. Không dùng các số này làm SD giữa seed. [Metric validation](../../convex_author_v1/METRIC_VALIDATION.md)

A1 có trạng thái hấp thụ. Khi drift và LR đều bằng 0, $\\tau_1(0)=0$ và kẹp LR trước bằng 0 giữ nguyên LR. SGD giữ nguyên weights, nên learner EMA tiếp tục bằng 0 ở bước kế. Quy nạp áp dụng cho toàn horizon. Test ở d512 chạy 100 batch với gradient khác 0 tại mọi bước và xác nhận LR, shift, weights trước/sau đều bằng 0. Chưa có kết quả ma trận A1–A4. CSV pairs và summary chỉ có header vì gate chưa mở. Warm start sẽ đổi cả initialization lẫn tín hiệu drift so với I, nên chỉ phù hợp với một đối chứng bổ sung riêng. Kết quả zero-start không khái quát cho mọi causal EMA. [Protocol A](../../convex_author_v1/PROTOCOL.md) · [Paired data](data/ablation_pairs.csv)

## Hình đã đo

- [Đường M0 theo mẫu Figure 5-bottom](figures/adaptive_m0.png)
- [Convex và matched baselines ở d512](figures/matched_methods.png)
- [P và pilot theo M2, hai phân phối riêng](figures/project_context_m2.png)
- [Trajectory hai tọa độ d128](figures/trajectory_d128.png)
- [Learning rate theo d](figures/learning_rates.png)
- [Diagnostics d512](figures/d512_diagnostics.png)
- [Gradient và reference drift d512](figures/d512_gradient_drift.png)
- [Hiệu chỉnh comparator M2−M1](figures/comparator_correction.png)

Chưa có đường cong R ghép cặp và đồ thị A. Toàn bộ đường pilot là một seed trên mỗi ca, không hiển thị SD.
