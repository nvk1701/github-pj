## Bảng Ánh Xạ Thí Nghiệm (Experiments Mapping)

Dưới đây là bảng tổng hợp các thí nghiệm tái hiện lại bài báo "Learning Rate Schedules in the Presence of Distribution Shift", mã nguồn tương ứng, liên kết theo dõi trên Weights & Biases (W&B) và kết luận chính.

| Thí nghiệm | Script/Notebook | W&B run | Kết luận 1 dòng |
| :--- | :--- | :--- | :--- |
| **1. Hồi quy tuyến tính** (Linear Regression) | `experiments/linear_regression.py` | [Tên_Run_Linear](Link_W&B_của_bạn_vào_đây) | *(Điền sau: ví dụ: Tốc độ học tăng vọt khi phân phối dịch chuyển, bám sát SDE)* |
| **2. Hàm lồi tổng quát** (Convex Loss) | `experiments/convex.py` | [Tên_Run_Convex](Link_W&B_của_bạn_vào_đây) | *(Điền sau: ví dụ: Tốc độ học dạng ngưỡng hoạt động hiệu quả theo chặn trên của Regret)* |
| **3. Hàm không lồi** (Non-convex / Neural Net) | `experiments/neural_net.py` | [Tên_Run_NN](Link_W&B_của_bạn_vào_đây) | *(Điền sau: ví dụ: Restart Adam giúp mạng nơ-ron thoát khỏi cực tiểu cục bộ cũ)* |
