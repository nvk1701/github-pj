# T8 — Đóng góp và tính mới của bài báo

Bảng phục vụ [Issue11](https://github.com/nvk1701/github-pj/issues/11). Số[1]–[5] theo [references](references.md), còn chờ thống nhất thứ tự trích dẫn toàn nhóm. Đóng góp tác giả được tách khỏi phần triển khai của dự án.

| Đóng góp | Loại | Điểm mới và nguồn | Giới hạn |
|---|---|---|---|
| Động lực SGD với mục tiêu Linear thay đổi | Lý thuyết/kỹ thuật | SDE có shift và phương trình moment; [1]§3, Proposition3.1, Theorem3.2 | Xấp xỉ dưới mô hình và scaling assumptions đã nêu |
| Điều khiển scalar envelope | Thuật toán/lý thuyết | Chọn schedule bằng optimal control, triển khai Euler substeps; [1]Theorem3.3, Algorithm1, Eq12–13 | Tối ưu envelope không đồng nghĩa tối ưu mọi finite SGD |
| Schedule hai ngưỡng cho Convex | Thuật toán/lý thuyết | Tối thiểu hóa bound tuần tự và chiếu LR trước; [1]Theorem4.2, Proposition4.3, Eq15–20 | Cần smoothness, variance/distance bounds và thông tin hợp lệ |
| Lower bound bổ sung | Lý thuyết | Đánh giá độ chặt với giả thiết curvature thêm; [1]Assumption4.6, Theorem4.7, Eq21 | Strong convexity không mặc định đúng cho mọi logistic setting |
| Tiêu chí non-convex | Lý thuyết/thuật toán | Bound theo gradient norm và rate tối thiểu hóa bound; [1]§5, Theorem5.1, Eq22–25 | Stationarity không bảo đảm global optimum |
| Minh họa thực nghiệm có shift | Thực nghiệm | Regression và neural-network examples; [1]§6.1, Figures4–5, AppendixE/Figure6 | Không chứng minh ưu thế trên mọi dữ liệu |

## Đối chiếu với nghiên cứu trước

[Bối cảnh T4](01_context.md), §3–4, so sánh static decay, dynamic OCO và continuous-time analysis. Không được hiểu rằng mọi nghiên cứu trước đều giả định phân phối tĩnh. Yang[2]§3.2 đã xét minimizer thay đổi với noisy gradients; Besbes[3]§3–4 đã dùng variation budgets và restarted optimization. Dynamic regret và thích nghi nonstationarity không phải khái niệm do[1] lần đầu đề xuất.

Điểm cần so sánh là local-rate thresholds và shift-aware control analysis dưới đúng assumptions. Feedback và variation model khác nhau nên không suy ra dominance hoặc xếp hạng regret rates vô điều kiện. Cách diễn đạt mạnh trong tài liệu T4 thuộc phần review của Khoa; không sửa tài liệu đó trong lượt consolidation.

## Phần công việc của dự án

Gaussian population evaluator float64, numerical oracle, A/B diagnostics, immutable attempts/checksums, verifier, CP04 và CP05 là triển khai/bằng chứng của dự án[4], không phải tính mới của tác giả. Đúng threshold algebra cũng không biến EMA/noise/comparator của dự án thành exact reproduction notebook[5].

Khi trình bày cần nêu mục tiêu tối ưu và assumptions, phân biệt bound với observed curves, đồng thời giữ kết quả baseline thắng tại d512. T8 sẵn sàng cho review; nghiệm thu và đồng bộ với T4 vẫn cần người phụ trách.
