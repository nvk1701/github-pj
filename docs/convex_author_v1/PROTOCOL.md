# Thiết kế nghiên cứu Convex sau mốc đóng băng

| Họ | Thiết kế | Điều kiện để công bố | Kích thước |
|---|---|---|---:|
| P | Frozen `convex-cp03-v1` | Giữ hash và provenance gốc | 45 CP04 + 15 CP05 |
| R | Chạy Figure 5-bottom notebook bất biến trong Colab riêng | Export native theo dimension, runtime và source snapshot | 5 |
| I | B=256, T=2000, std=.1, reference-forward, không chiếu, 5 dimension × seeds 0/1/2 × 3 phương pháp | F0–F5, bao gồm replay Colab | 45 |
| A | Convex I tại d=8/128/512 × seeds 0/1/2 × 4 can thiệp | I hoàn thành, đối chứng từng seed | 36 |
| Pilot | Convex seed 0 tại cả 5 d và hai baseline tại d512 | Tương thích nội bộ và chạy hết horizon | 7 |
| RNG diagnostic | Independent MT19937 protocol, d=2/8/32 | Tham chiếu output lịch sử ở mức end-point | 3 |

[Cấu hình khóa](../../configs/convex_author_v1.yaml) và [bảng đối chiếu nguồn](SOURCE_AUDIT.md) là đặc tả đọc được. Cổng F2/F3 dùng toàn bộ replay fixture ở d=2/8/32, với X bitwise và tolerance tuyệt đối/cộng tương đối định trước trong YAML cho các trường còn lại. Sai khác kernel float32, thứ tự tổng hoặc phép nhân BLAS có thể tích lũy. Báo cáo fixture lưu max, median, p99 và số vượt ngưỡng. M0 cumulative có tolerance lớn hơn loss từng bước vì là tổng float32 của 2000 số.

I/A dùng RNG PCG64 riêng cho X và noise bằng hai child stream của `SeedSequence([seed, d])`, cùng weights zero. Trộn chung thông tin `method` vào seed sẽ phá cặp đối chứng. Baseline của nhóm là .1 và .1/√(t+1), với t đánh chỉ số từ 0. Không tune bằng evaluation seeds. Dùng MT19937 reset theo phase chỉ trong diagnostic để tránh đồng nhất RNG mặc định NumPy với TensorFlow hoặc stream I.

| Can thiệp | Thay đổi | Đối chứng và diễn giải |
|---|---|---|
| A1 EMA | Thay forward reference drift bằng learner EMA beta=.9. Learner pre-update tại t chỉ phụ thuộc batch đến t−1. | Native init giữ nguyên. Zero LR → theta đứng yên → EMA zero → zero LR. Đây là kết quả của can thiệp đơn theo cấu hình đã chọn. |
| A2 noise | Std từ .1 sang sqrt(.1), cùng X và base Gaussian noise. | M0/M1/M2 riêng trên mỗi phân phối. Delta gồm thay đổi mục tiêu và quá trình học. |
| A3 initial LR | Dùng LR trước khi kẹp bằng 1 ở bước đầu. | Sau đó áp dụng cùng Eq. 20. Kiểm early loss và trace LR. |
| A4 projection | Chiếu sau cập nhật lên ball bán kính d/2. | Báo số update chạm biên. M2 oracle nằm trong ball được kiểm trước khi tái sử dụng oracle radial. |

OAT ước lượng hiệu ứng theo đúng control đã chọn. Ví dụ chuyển sang EMA và thêm warm start là thiết kế tương tác có ý nghĩa riêng. Hai thay đổi đồng thời vẫn cần protocol bridge, gồm các khác biệt rời rạc của P nếu mục tiêu là quy chiếu toàn bộ protocol.

## Tài nguyên

Ở d512, một phase X có 2560×512 số, 5 MiB float32 hoặc 10 MiB tạm float64 khi sinh. Histories theta, post-theta, gradient, reference mỗi mảng khoảng 3.91 MiB. Một batch cần O(Bd), tính threshold O(1), EMA thêm O(d). Train giữ O(10Bd+Td) numbers, publication giữ curves hoặc tọa độ cần thiết. Population evaluator thêm quadrature 1D theo từng norm learner, cache phần phụ thuộc reference và phân phối. Số phép tính train O(TBd), chạy tuần tự, mặc định BLAS một thread nếu biến môi trường chưa đặt. Chứng cứ native có footprint riêng. [Hướng dẫn Colab](AUTHOR_NATIVE_EXECUTION.md)
