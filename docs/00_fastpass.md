# Ghi chú đọc lướt (Fast Pass) - Khoa

## 1. Bài báo hứa hẹn/giải quyết điều gì?
Bài báo giải quyết bài toán thiết kế **lịch trình tốc độ học (learning rate schedules)** tối ưu cho thuật toán SGD trong hệ thống học trực tuyến (online learning) khi đối mặt với sự thay đổi phân phối dữ liệu liên tục (distribution shift)[cite: 7]. 
Cụ thể, bài báo hứa hẹn sẽ giảm thiểu "độ hối tiếc động" (dynamic regret) cho 3 cấp độ bài toán[cite: 7]:
*   **Hồi quy tuyến tính:** Tìm ra tốc độ học tối ưu chính xác tuyệt đối nhờ phân tích bằng Phương trình vi phân ngẫu nhiên (SDE).
*   **Hàm mất mát lồi (Convex):** Đề xuất lịch trình tốc độ học mới và thiết lập ranh giới (chặn trên/chặn dưới) cho độ hối tiếc.
*   **Hàm không lồi (Non-convex / Deep Learning):** Đề xuất lịch trình tối thiểu hóa độ hối tiếc dựa trên chuẩn độ dốc (gradient norm).

## 2. Kết luận cốt lõi của bài báo là gì?
Phát hiện và kết luận quan trọng nhất là: Trực giác thường thấy trong Machine Learning là giảm dần tốc độ học theo thời gian để mô hình hội tụ[cite: 7]. Tuy nhiên, **khi cảnh quan mất mát (loss landscape) thay đổi do dữ liệu bị dịch chuyển, lịch trình tốc độ học tối ưu thường phải TĂNG LÊN**[cite: 7, 15]. Việc tăng tốc độ học (hoặc duy trì mức cao) giúp mô hình tăng cường khám phá không gian trọng số và cho phép thuật toán tối ưu hóa mạnh mẽ, quyết liệt hơn để kịp thời thích ứng với dữ liệu mới[cite: 7, 15].

## 3. Danh sách các thuật ngữ / khái niệm cần thảo luận thêm
(Do mới học ML nên cần nhóm thảo luận và làm rõ các khái niệm toán học sâu)
*   **Stochastic Differential Equations (SDE):** Phương trình vi phân ngẫu nhiên là gì và tại sao lại dùng nó để mô phỏng thuật toán SGD ở phần hồi quy tuyến tính?[cite: 7]
*   **Gradient norm:** Tại sao trong hàm không lồi (mạng neural), việc đo độ hối tiếc lại dùng "chuẩn gradient" của mô hình dự đoán thay vì so sánh trực tiếp với mô hình tối ưu $\theta^*_t$?[cite: 7, 14]
*   **Flow cytometry:** Bài toán phân tích tế bào dòng chảy sinh học ở phần thực nghiệm (Appendix E) cụ thể là làm gì và ý nghĩa của nó?[cite: 15, 28]
