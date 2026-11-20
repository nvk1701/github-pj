# Kết quả thực nghiệm Convex V5

Bộ này tách ba lớp chứng cứ. R chạy nguyên notebook TensorFlow trong Colab. I và A dùng triển khai NumPy của nhóm và dữ liệu được ghép cặp. P là kết quả CP04/CP05 bất biến, dùng một phân phối và lưới thời gian khác. [Protocol](PROTOCOL.md) · [Inventory](data/inventory.csv) · [Manifest](manifest.json)

## R và parity F2/F3

| d | R native M0 cuối | Sai số M0 tích lũy lớn nhất ở fixture chung |
|---:|---:|---:|
| 2 | 0.9858183861 | 4.709e−6 |
| 8 | 3.7352774143 | 8.345e−6 |
| 32 | 48.0655975342 | 2.480e−5 |
| 128 | 180.8743438721 | N/A |
| 512 | 1207.9738769531 | N/A |

Ở d2/8/32, cả 2.000 batch khớp chính xác về X, y, reference, gamma, ngưỡng và LR. Sai khác tuyệt đối lớn nhất của loss, gradient và weights lần lượt là 1.19e−7, 5.96e−8 và 2.38e−7. R đánh giá M0 trên batch của MT19937 theo notebook. I dùng PCG64 độc lập ghép giữa phương pháp, nên giá trị R và mean I minh họa hai giao thức chạy, còn đối chiếu từng bước dùng các fixture. [Fidelity](data/fidelity.csv) · [Native và replay M0](figures/colab_m0.png) · [Native trajectories](data/native_curves.csv)

## I · Mean ± SD mẫu của M2

Mỗi ô có ba seed. Giá trị nhỏ hơn nghĩa là cumulative population gap nhỏ hơn trên cùng phân phối.

| d | Convex | Constant | Inverse-sqrt |
|---:|---:|---:|---:|
| 2 | 1.0073 ± 0.0014 | 7.7864 ± 0.0218 | 31.3175 ± 0.0325 |
| 8 | 3.7636 ± 0.0116 | 56.2355 ± 0.3687 | 92.4544 ± 0.0531 |
| 32 | 47.8951 ± 0.0253 | 186.3040 ± 0.0882 | 207.3436 ± 0.0959 |
| 128 | 181.2661 ± 0.9833 | 336.9341 ± 0.2204 | 361.7788 ± 0.1190 |
| 512 | 1205.4913 ± 2.1730 | 504.0267 ± 0.1265 | 527.0837 ± 0.1081 |

Cùng d512, mean M0 của Convex, constant, inverse-sqrt lần lượt là 1208.6551, 504.2291 và 527.1986. Mean M1 tương ứng là 1205.4909, 504.0263 và 527.0834. Trong mỗi cặp dimension/seed, X và y khớp hash giữa ba method. Mean, SD và giá trị từng run được lưu cho cả M0, M1 và M2. [Bảng đủ ba metric](data/method_summary.csv) · [Theo seed](data/per_run.csv) · [Đối chiếu matched methods](figures/matched_methods.png)

M2 hiệu chỉnh điểm tối ưu quần thể trong chính phân phối của ca. M2−M1 ở d512 xấp xỉ 0.000369 và giống giữa các method. Khi phân phối thay đổi, như ablation label noise, reference correction cũng thay đổi. [Comparator correction](figures/comparator_correction.png) · [Định nghĩa và sai số tích phân](../../convex_author_v1/METRIC_VALIDATION.md)

## A · Paired ΔM2 so với Convex của I

Mỗi ô là mean ± SD mẫu trên ba chênh lệch cùng seed. Dấu âm là gap thấp hơn control.

| Can thiệp | d8 | d128 | d512 |
|---|---:|---:|---:|
| EMA nhân quả | +87.632 ± 0.012 | +179.806 ± 0.983 | −679.547 ± 2.173 |
| Label-noise std tăng lên sqrt(0.1) | +0.452 ± 0.021 | +8.808 ± 0.073 | +36.982 ± 0.421 |
| LR đầu 1.0 | −1.558 ± 0.015 | −2.491 ± 0.022 | −3.588 ± 1.833 |
| Chiếu lên ball bán kính d/2 | 0 ± 0 | 0 ± 0 | 0 ± 0 |

EMA khởi đầu 0 giữ LR và weights bằng 0 ở toàn bộ chín ca, kể cả khi gradient tồn tại. Trạng thái hấp thụ khiến d8 và d128 xấu đi. Ở d512, zero-update có M2 525.944 thấp hơn adaptive Convex nhưng cao hơn constant 504.027; đây là hành vi cold-start của tổ hợp thay đổi đã khai báo. Với noise, X và base noise được ghép, nhưng y và phân phối đánh giá đổi theo can thiệp. LR đầu 1 giúp giảm M2 theo cả ba nhóm trong mẫu này. Phép chiếu không tác động lên update nào tại các seed và dimension đã chạy. [Chênh lệch từng seed](data/ablation_pairs.csv) · [Tổng hợp ba metric](data/ablation_summary.csv) · [EMA](figures/ablation_ema.png) · [Noise](figures/ablation_noise.png) · [LR đầu](figures/ablation_initial_lr.png) · [Chiếu](figures/ablation_projection.png)

## d512 và giới hạn so với P

Ở batch 901–1100, forward reference shift trung bình chỉ khoảng 0.000594. Convex duy trì LR gần 4 trên cả ba seed. Norm update trung bình trong cửa sổ là 2.06–2.09 và mỗi seed cộng thêm khoảng 64.86–65.85 M2. Đây là bằng chứng định lượng về động học ở vùng ít drift, đi cùng khoảng cách Convex–constant trong I. Thay LR đầu hoặc thêm phép chiếu theo A không lấp khoảng cách. A1 thay đổi tín hiệu drift trong trường hợp learner rơi vào điểm hấp thụ, và A2 đổi phân phối. Các kết quả này giới hạn cách giải thích, chưa tách được nguyên nhân của CP04. [Cửa sổ d512](data/d512_windows.csv) · [Gradient, drift và M2](figures/d512_gradient_drift.png) · [Đối chiếu P/I theo phân phối](figures/project_context_m2.png)

P báo M2-style B với phase grid, std và cơ chế drift riêng. R báo M0 thực nghiệm. Chỉ so sánh trong các cặp chung đại lượng và dữ liệu đã nêu. Các pilot dùng threshold float64 ổn định từ preflight giữ nguyên record cũ và nằm riêng với pilot validation dùng mixed precision. [CP04/CP05 frozen](../cp04-cp05/RESULTS.md)

## Phát hành Forge

Đã seal 88 bundle offline gồm 7 pilot gốc, 45 I và 36 A. Đối soát chỉ đọc Forge chưa thấy ID tương ứng. Upload trong lượt này bị bộ duyệt chặn vì cần ủy quyền trực tiếp cho payload và đích. Trạng thái hiện tại được ghi theo từng ID. [Sổ đối soát Forge](data/forge_release_verification.csv)
