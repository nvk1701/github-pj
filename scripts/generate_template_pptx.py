r"""
================================================================================
FILE: scripts/generate_template_pptx.py
MỤC ĐÍCH:
    Tạo tự động file khung thuyết trình chuẩn `slides/template.pptx` (Task T20)
    gồm đúng 32 slides Widescreen 16:9 theo kế hoạch Sheet 3 của nhóm, đảm bảo
    đồng bộ 100% giữa cả 3 thành viên (TV1: Khoa, TV2: Khôi, TV3: Hòa) về:
    - Màu nền (Background `#F8FAFC`), Thanh Header (`#1E3A8A`), Bố cục 2 cột.
    - Font chữ chuẩn (`Arial`), cỡ chữ Tiêu đề, Nội dung, Khung hình ảnh/biểu đồ.
    - Thanh Footer trích dẫn chuẩn IEEE đánh số ở chân mỗi slide (không dùng slide
      References riêng theo đúng quy định của giảng viên).
    - Khung Speaker Notes (<= 500 từ/slide) kèm gợi ý nội dung chi tiết cho từng trang.

CÁCH CHẠY:
    `& './.venv-t2-clean/Scripts/python.exe' scripts/generate_template_pptx.py`
================================================================================
"""

from pathlib import Path
from typing import Dict, List
from xml.sax.saxutils import escape
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Danh mục trích dẫn chuẩn IEEE dùng ở chân từng slide
IEEE_CITATIONS = {
    1: '[1] M. Fahrbach, A. Javanmard, V. Mirrokni, and P. Worah, "Learning Rate Schedules in the Presence of Distribution Shift," in Proc. 40th Int. Conf. Mach. Learn. (ICML), PMLR, vol. 202, pp. 9523-9546, 2023.',
    2: '[2] H. Robbins and S. Monro, "A stochastic approximation method," Ann. Math. Statist., vol. 22, no. 3, pp. 400-407, 1951; R. Ge et al., NeurIPS, 2019.',
    3: '[3] T. Yang, L. Zhang, R. Jin, and J. Yi, "Tracking slowly moving clairvoyant: Optimal dynamic regret of online learning with true and noisy gradient," in Proc. ICML, PMLR, vol. 48, pp. 449-457, 2016.',
    4: '[4] O. Besbes, Y. Gur, and A. Zeevi, "Non-stationary stochastic optimization," Oper. Res., vol. 63, no. 5, pp. 1227-1244, 2015; B. Shi, W. J. Su, and M. I. Jordan, arXiv:2004.06977, 2020.',
    5: '[5] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in Proc. ICLR, 2015; A. Bastidas-Ponce et al., "Comprehensive single cell mRNA profiling...," Development, vol. 146, no. 12, 2019.',
}

# Định nghĩa chi tiết 32 slides theo đúng Sheet 3 ("3. Kế hoạch slide")
SLIDES_SPEC: List[Dict[str, object]] = [
    {
        "num": 1,
        "section": "M2 — MỞ ĐẦU",
        "owner": "TV3 (Hòa)",
        "title": "Learning Rate Schedules in the Presence of Distribution Shift",
        "left_title": "Thông tin Đồ án Giữa kỳ (Milestone 2 & Lab 1)",
        "left_bullets": [
            "Bài báo gốc: Matthew Fahrbach, Adel Javanmard, Vahab Mirrokni, Pratik Worah (Google Research & USC, ICML 2023) [1].",
            "Phạm vi thực nghiệm: FULL SCOPE (100% cả 3 bài toán: Linear Regression, Logistic Regression, Flow Cytometry MLP + Sensitivity).",
            "Thành viên thực hiện: TV1: Khoa (Trưởng nhóm - Linear & SDE) | TV2: Khôi (Convex & W&B) | TV3: Hòa (Non-convex MLP & Code gốc).",
            "Tài nguyên Public: GitHub Repository & Weights & Biases Project (tag: lab1).",
        ],
        "right_title": "Sơ đồ Tổng quan Ba Chế độ Tối ưu dưới Dịch chuyển Phân phối",
        "right_placeholder": "[TV3 chèn hình/sơ đồ tổng quan: Online Stream P_t -> (1) Linear SDE Alg.1 | (2) Convex 2-Threshold Prop.4.3 | (3) Non-convex & Adam Restart]",
        "cites": [1],
        "notes": "Giới thiệu đề tài đồ án giữa kỳ môn Học máy: Đọc hiểu chuyên sâu (Milestone 2) và Tái hiện thực nghiệm Full Scope (Lab 1) bài báo ICML 2023 của Fahrbach et al. [1]. Nêu rõ thông điệp xuyên suốt: Khi phân phối dữ liệu dịch chuyển theo thời gian, tốc độ học tối ưu cần tăng lên thay vì chỉ giảm dần về 0.",
    },
    {
        "num": 2,
        "section": "M2 — MỞ ĐẦU",
        "owner": "TV3 (Hòa)",
        "title": "Mục lục & Lộ trình Báo cáo (Milestone 2 Retelling + Lab 1 Full Scope)",
        "left_title": "Cấu trúc 5 Phần của Slide Deck (32 Slides)",
        "left_bullets": [
            "Phần 1 (Slides 3–7): Bối cảnh, Động lực thực tế (E-commerce CTR) & Phân tích 3 phương pháp tiền nhiệm.",
            "Phần 2 (Slides 8–9): Tóm tắt Đóng góp & Phân loại Tính mới (Theoretical / Technical / Empirical Novelty).",
            "Phần 3 (Slides 10–21): Phương pháp 5 lớp (Ký hiệu, Pipeline 2 pha, Linear SDE, Convex Thresholds, Non-convex, Định lý & Độ phức tạp).",
            "Phần 4 (Slides 22–30): Thực nghiệm Lab 1 Full Scope (Thiết lập W&B, Fig 4, Fig 5-top, Fig 5-bottom, Fig 6, Độ nhạy & Đối chiếu code gốc).",
            "Phần 5 (Slides 31–32): Kết luận, Hạn chế lý thuyết & Bảng đóng góp thành viên kèm link W&B/GitHub.",
        ],
        "right_title": "Sơ đồ Hành trình Từ Lý thuyết đến Thực nghiệm",
        "right_placeholder": "[TV3 vẽ lưu đồ ngang 5 bước: Context & Gap -> Contributions -> 5-Layer Method -> Full Scope Reproduction -> Critique & Takeaways]",
        "cites": [1],
        "notes": "Trình bày lộ trình 32 slides kết hợp liền mạch giữa Milestone 2 (Retelling không cần mở PDF) và Lab 1 (Tái hiện 100% thực nghiệm bài báo kèm kiểm chứng trên Weights & Biases).",
    },
    {
        "num": 3,
        "section": "M2 — CONTEXT & MOTIVATION",
        "owner": "TV1 (Khoa)",
        "title": "Vấn đề Nghiên cứu: SGD trong Hệ thống Học Trực tuyến có Trôi dạt Phân phối",
        "left_title": "Mâu thuẫn giữa Học Tĩnh (Static) và Học Trực tuyến (Never-Ending)",
        "left_bullets": [
            "Giả định truyền thống: Phân phối dữ liệu P cố định -> giảm dần tốc độ học (learning rate decay) về 0 để hội tụ vào cực tiểu tĩnh.",
            "Thực tế công nghiệp: Dữ liệu đến liên tục theo luồng P_t biến thiên -> nghiệm tối ưu thực sự θ*_t di chuyển không ngừng.",
            "Hệ quả nếu giảm η_t -> 0: Mô hình bị 'đóng băng' trọng số ở quá khứ, mất khả năng bám đuổi θ*_t mới và tích lũy Dynamic Regret lớn.",
            "Câu hỏi cốt lõi [1]: Khi phân phối dịch chuyển với biên độ γ_t = ||θ*_t - θ*_{t+1}||_2, lịch trình η*_t phải thay đổi như thế nào?",
        ],
        "right_title": "Minh họa Quỹ đạo SGD theo Tốc độ học (Tái hiện Figure 1)",
        "right_placeholder": "[Chèn hình `figures/fig1_constant_lr_trajectories.png`: So sánh η_t in {0.003, 0.01, 0.03, 0.1} bám đuổi quỹ đạo xoắn ốc θ*_t]",
        "cites": [1, 2],
        "notes": "Giải thích tại sao quy tắc giảm tốc độ học cổ điển thất bại trong môi trường học trực tuyến không ngừng (never-ending learning). Dùng hình tái hiện Figure 1 để chỉ rõ: tốc độ học nhỏ (0.003) bị tụt lại phía sau khi đích ngắm nhảy bậc, trong khi tốc độ học lớn hơn (0.1) tuy dao động mạnh hơn nhưng bám sát quỹ đạo xoắn ốc và đạt tổng độ tiếc nuối thấp nhất.",
    },
    {
        "num": 4,
        "section": "M2 — CONTEXT & MOTIVATION",
        "owner": "TV1 (Khoa)",
        "title": "Ví dụ Thực tế của Nhóm: Dự đoán CTR Thời gian thực trên Sàn Thương mại Điện tử",
        "left_title": "Bài toán Streaming CTR Prediction (Shopee / TikTok Shop)",
        "left_bullets": [
            "Luồng sự kiện liên tục: Mỗi giây nhận batch hiển thị quảng cáo z_{t,k} = (x_{t,k}, y_{t,k}); mô hình phải dự đoán trước rồi cập nhật SGD sau.",
            "Kiểu trôi 1 — Trôi mượt (Smooth Drift): Thị hiếu người dùng dịch chuyển dần từ sáng đến tối (γ_t > 0 nhỏ nhưng liên tục).",
            "Kiểu trôi 2 — Nhảy đột ngột (Bursty Shift): Sự kiện Flash Sale 0h00 hoặc xu hướng Livestream bùng nổ làm γ_t tăng vọt đột biến.",
            "Yêu cầu sống còn: Tự động tăng vọt η_t ngay khi nổ ra Flash Sale để khám phá lại không gian trọng số, rồi hạ dần khi phiên ổn định.",
        ],
        "right_title": "Sơ đồ Hai Chế độ Dịch chuyển trong Hệ thống Gợi ý",
        "right_placeholder": "[Sơ đồ so sánh Smooth Drift (buổi sáng -> tối) và Bursty Shift (Flash Sale 0h00) cùng phản ứng mong muốn của η*_t]",
        "cites": [1],
        "notes": "Trình bày ví dụ thực tế do nhóm tự xây dựng (chi tiết trong `docs/motivation_example.md`): hệ thống dự đoán tỷ lệ nhấp chuột (CTR) trực tuyến. Phân tích hậu quả kinh tế nếu mô hình dùng lịch trình 1/t cũ kỹ khi bước vào khung giờ Flash Sale 0h00.",
    },
    {
        "num": 5,
        "section": "M2 — CONTEXT & MOTIVATION",
        "owner": "TV1 (Khoa)",
        "title": "Phương pháp Tiền nhiệm 1: Lịch trình Suy giảm Cổ điển (Robbins–Monro & Step Decay)",
        "left_title": "Phân tích Ý tưởng — Hạn chế — Hệ quả Thực tế",
        "left_bullets": [
            "Ý tưởng cốt lõi [2]: Giảm η_t theo bậc O(1/t) (Robbins-Monro) hoặc chia giai đoạn (Step Decay - Ge et al., 2019) để triệt tiêu phương sai nhiễu mini-batch σ²/B_t.",
            "Hạn chế toán học: Toàn bộ chứng minh hội tụ đều giả định nghiêm ngặt P_1 = P_2 = ... = P_T (γ_t = 0 với mọi t).",
            "Hệ quả thực tế: Khi phân phối đổi hướng ở thời điểm muộn t >> 1, η_t ≈ 0 khiến bước cập nhật ||θ_{t+1} - θ_t|| ≈ 0 -> mô hình 'mù' trước dữ liệu mới.",
            "So sánh thực nghiệm: Trên bài toán Linear Regression d=8, lịch trình 1/t chịu Regret = 358.4, cao gấp 26 lần Algorithm 1 (13.4)!",
        ],
        "right_title": "Đồ thị Suy giảm 1/t dẫn đến Hiện tượng Đóng băng Trọng số",
        "right_placeholder": "[Sơ đồ minh họa η_t = η_0 / t tiến sát trục hoành khi t lớn, làm khoảng cách ||θ_t - θ*_t|| phình to sau mỗi đợt shift]",
        "cites": [1, 2],
        "notes": "Phân tích phương pháp tiền nhiệm thứ nhất (Robbins-Monro 1951, Ge et al. 2019). Nhấn mạnh nguyên tắc của Milestone 2: mọi hạn chế đều phải gắn với hệ quả thực tế (mất khả năng thích ứng ở giai đoạn muộn, buộc kỹ sư phải thiết lập lịch retrain thủ công tốn kém).",
    },
    {
        "num": 6,
        "section": "M2 — CONTEXT & MOTIVATION",
        "owner": "TV1 (Khoa)",
        "title": "Phương pháp Tiền nhiệm 2 & 3: Tối ưu Lồi Động (OCO) & Xấp xỉ SDE Tĩnh",
        "left_title": "Hai Hướng Tiếp cận Gần nhất Trước Bài báo",
        "left_bullets": [
            "Tiền nhiệm 2 — OCO với Ngân sách Biến thiên V_T (Besbes et al., 2015; Yang et al., 2016) [3], [4]: Đặt hằng số η ∝ sqrt(V_T / T) hoặc khởi động lại định kỳ mỗi Δ_T = (T/V_T)^{2/3} bước.",
            "• Hạn chế & Hệ quả: Đòi hỏi biết trước tổng chân trời T và tổng độ trôi V_T = Σ γ_t; dùng chung 1 bước nhảy cố định nên quá lớn khi tĩnh và quá nhỏ khi nhảy vọt.",
            "Tiền nhiệm 3 — Xấp xỉ SDE cho SGD (Li et al., 2017; Shi et al., 2020) [4]: Mô hình hóa SGD bằng phương trình vi phân ngẫu nhiên liên tục.",
            "• Hạn chế & Hệ quả: Chỉ xét hàm mất mát tĩnh (Y'(τ) = 0); chưa có số hạng vận tốc trôi phân phối trong phương trình SDE.",
        ],
        "right_title": "So sánh OCO Cố định (Yang 2016) vs. Thích ứng Tức thời (Fahrbach 2023)",
        "right_placeholder": "[Sơ đồ đối chiếu: Bước nhảy cố định η ∝ sqrt(V_T/T) vs. Bước nhảy thích ứng η*_t theo từng bước γ_{t-1}]",
        "cites": [1, 3, 4],
        "notes": "Phân tích sâu 2 nhánh tiền nhiệm trực tiếp: Online Convex Optimization trong môi trường động (Yang et al., 2016; Besbes et al., 2015) và phân tích SDE liên tục cho SGD (Shi et al., 2020). Chỉ rõ tại sao việc phụ thuộc vào tổng ngân sách V_T cố định không thể áp dụng trực tiếp cho hệ thống thời gian thực.",
    },
    {
        "num": 7,
        "section": "M2 — CONTEXT & MOTIVATION",
        "owner": "TV1 (Khoa)",
        "title": "Khoảng trống Nghiên cứu (The Research Gap) mà Bài báo Lấp đầy",
        "left_title": "Bảng Đối chiếu Tiền nhiệm vs. Fahrbach et al. (ICML 2023)",
        "left_bullets": [
            "1. Tính thích ứng thời gian thực: Chọn η_t hoàn toàn online từ lịch sử z_{[t-1]}, không cần biết trước chân trời T hay ngân sách V_T.",
            "2. Môi trường đối kháng thích ứng (Adaptive Adversary): Tự nhiên được phép chọn P_t dựa trên lịch sử mô hình θ_s (s < t).",
            "3. Công cụ SDE có số hạng trôi Y'(τ): Lần đầu tiên đưa vận tốc dịch chuyển nghiệm tối ưu vào SDE và giải bằng điều khiển tối ưu HJB.",
            "4. Bao phủ cả 3 lớp hàm mất mát: Hồi quy tuyến tính (SDE), Hàm lồi tổng quát (chặn trên/dưới chặt đến hằng số), và Hàm không lồi (chuẩn gradient).",
        ],
        "right_title": "Ma trận Định vị Khoảng trống Nghiên cứu",
        "right_placeholder": "[Bảng tổng hợp 4 tiêu chí: (1) Có xét Distribution Shift? (2) Thích ứng từng bước t? (3) Xấp xỉ SDE? (4) Bao phủ Non-convex?]",
        "cites": [1, 2, 3, 4],
        "notes": "Tổng kết phần Context & Motivation bằng bảng đối chiếu trực quan giữa 3 nhóm tiền nhiệm và bài báo Fahrbach et al. (2023), làm nổi bật khoảng trống nghiên cứu mà bài báo giải quyết.",
    },
    {
        "num": 8,
        "section": "M2 — CONTRIBUTIONS",
        "owner": "TV2 (Khôi)",
        "title": "Tóm tắt Ba Đóng góp Chính theo Lời Bài báo",
        "left_title": "Ba Trụ cột Đóng góp (Linear — Convex — Non-Convex)",
        "left_bullets": [
            "1. Hồi quy Tuyến tính (Mục 3): Xây dựng phương trình SDE mới xấp xỉ quỹ đạo SGD dưới dịch chuyển nhãn; dẫn xuất lịch trình tối ưu Algorithm 1 qua phương trình HJB.",
            "2. Hàm Lồi Tổng quát (Mục 4): Thiết lập cận trên Dynamic Regret (Theorem 4.2), cận dưới khớp đến hằng số cho hàm lồi mạnh (Theorem 4.7), và lịch trình 2 ngưỡng [τ_{1,t}, τ_{2,t}] (Proposition 4.3).",
            "3. Hàm Không lồi & Thực nghiệm (Mục 5 & 6): Định nghĩa Regret theo tổng bình phương chuẩn gradient; dẫn xuất bước nhảy đóng (Theorem 5.1) và kiểm chứng trên hồi quy cao chiều & Flow Cytometry MLP.",
        ],
        "right_title": "Infographic Ba Đóng góp Cốt lõi",
        "right_placeholder": "[TV2 vẽ Infographic 3 nhánh: (1) Linear SDE & Alg.1 -> (2) Convex Tight Bounds & Prop.4.3 -> (3) Non-convex Grad-Norm & Cytometry]",
        "cites": [1],
        "notes": "Trình bày lại các đóng góp chính của bài báo bằng ngôn ngữ của nhóm, chuẩn bị cho việc phân loại chi tiết và đánh giá tính mới ở Slide 9.",
    },
    {
        "num": 9,
        "section": "M2 — CONTRIBUTIONS",
        "owner": "TV2 (Khôi)",
        "title": "Phân loại Đóng góp & Đánh giá Tính mới (Novelty vs. Predecessors)",
        "left_title": "Phân loại theo 3 Chiều: Lý thuyết — Kỹ thuật — Thực nghiệm",
        "left_bullets": [
            "Đóng góp Lý thuyết (Theoretical): Cận sai số xấp xỉ SDE O(sqrt(ε)) (Prop. 3.1); cặp cận trên – cận dưới Dynamic Regret khớp nhau đến hằng số trên cùng một hàm mất mát cố định l(θ, z) (Thm. 4.2 & 4.7).",
            "Đóng góp Kỹ thuật (Technical): Rút gọn bài toán điều khiển d chiều về ODE bao trên 1 chiều ṽ_τ (Alg. 1); quy tắc kẹp 2 ngưỡng không tham số [τ_{1,t}, τ_{2,t}] kết hợp ước lượng trượt EMA γ̂_t.",
            "Đóng góp Thực nghiệm (Empirical): Chứng minh hiện tượng tốc độ học dao động nảy lên η_max khi có shift và hiệu quả của Adam Restart trên dữ liệu tế bào tụy đơn bào [5].",
        ],
        "right_title": "Bảng Đối chiếu Tính mới: Trước đây chưa làm được gì?",
        "right_placeholder": "[TV2 chèn Bảng 3 cột: Đóng góp | Loại (Lý thuyết/Kỹ thuật/Thực nghiệm) | Điểm mới vượt qua tiền nhiệm Yang 2016 / Shi 2020]",
        "cites": [1, 3, 4, 5],
        "notes": "Phân tích rõ từng đóng góp thuộc loại nào (lý thuyết, kỹ thuật, thực nghiệm) và chỉ rõ điều gì trước đây chưa thể làm được hoặc chưa từng được thử nghiệm.",
    },
    {
        "num": 10,
        "section": "M2 — METHOD (LAYER 1: FORMALISM)",
        "owner": "TV1 (Khoa)",
        "title": "Bảng Ký hiệu Thống nhất & Phát biểu Bài toán Hình thức",
        "left_title": "Hệ thống Ký hiệu Chuẩn hóa của Nhóm (`docs/notation.md`)",
        "left_bullets": [
            "t in {1..T}, B_t: Bước thời gian rời rạc, chân trời T và kích thước mini-batch.",
            "z_t = {z_{t,k}}_{k=1}^{B_t} ~ P_t: Mini-batch mẫu dữ liệu lấy từ phân phối P_t tại bước t.",
            "l̄_t(θ) = E_{z~P_t}[l(θ, z)]: Hàm mất mát kỳ vọng và nghiệm tối ưu Oracle θ*_t = argmin_θ l̄_t(θ).",
            "γ_t = ||θ*_t - θ*_{t+1}||_2 (lồi) hoặc sup_θ |l̄_t(θ) - l̄_{t+1}(θ)| (không lồi): Độ dịch chuyển phân phối.",
            "reg_t = l̄_t(θ_t) - l̄_t(θ*_t), Reg(T) = Σ_{t=1}^T reg_t: Độ tiếc nuối động tức thời và tích lũy.",
        ],
        "right_title": "Phát biểu Bài toán Tối ưu Lịch trình Tốc độ học",
        "right_placeholder": "[Khung công thức bài toán: Cập nhật θ_{t+1} = Π_Θ(θ_t - η_t ∇l^B_t(θ_t)) sao cho cực tiểu hóa E[Reg(T)] chỉ dùng lịch sử z_{[t-1]}]",
        "cites": [1],
        "notes": "Giới thiệu bảng ký hiệu thống nhất được cả 3 thành viên sử dụng xuyên suốt tài liệu, mã nguồn và slide. Nhấn mạnh tính nhân quả (non-clairvoyant): tại bước t, η_t chỉ được phép phụ thuộc vào lịch sử dữ liệu z_{[t-1]}.",
    },
    {
        "num": 11,
        "section": "M2 — METHOD (LAYER 1: FORMALISM)",
        "owner": "TV1 (Khoa)",
        "title": "Mô hình hóa Distribution Shift & Trò chơi Học Trực tuyến",
        "left_title": "Tương tác Tuần tự giữa Tự nhiên (Nature) và Bộ học (Learner)",
        "left_bullets": [
            "Mỗi vòng t = 1..T diễn ra theo thứ tự nghiêm ngặt:",
            "1. Tự nhiên chọn phân phối P_t (xác định nghiệm tối ưu θ*_t) dựa trên lịch sử z_{[t-1]}.",
            "2. Bộ học dùng trọng số hiện tại θ_t dự đoán trên batch mới z_t ~ P_t và chịu độ tiếc nuối reg_t = l̄_t(θ_t) - l̄_t(θ*_t).",
            "3. Bộ học tính gradient mẫu ∇l^B_t(θ_t), chọn bước nhảy thích ứng η_t và cập nhật θ_{t+1} = Π_Θ(θ_t - η_t ∇l^B_t(θ_t)).",
            "4. Giữa bước t và t+1, nghiệm tối ưu dịch chuyển một đoạn γ_t = ||θ*_t - θ*_{t+1}||_2.",
        ],
        "right_title": "Sơ đồ Dịch chuyển Quỹ đạo Oracle θ*_t và Trọng số Học được θ_t",
        "right_placeholder": "[Sơ đồ hình học trên không gian R^p: Hình cầu chấp nhận được Θ (đường kính D_max), bước nhảy γ_t của θ*_t và bước đuổi theo của θ_t]",
        "cites": [1, 3],
        "notes": "Giải thích sự khác biệt giữa Static Regret (so với 1 điểm cố định tốt nhất) và Dynamic Regret (so với chuỗi nghiệm động θ*_1, ..., θ*_T), cũng như cách định lượng độ dịch chuyển phân phối γ_t ở từng bước.",
    },
    {
        "num": 12,
        "section": "M2 — METHOD (LAYER 2: PIPELINE)",
        "owner": "TV1 (Khoa)",
        "title": "Pipeline Tổng thể: Tách bạch Quá trình Inference và Training",
        "left_title": "Chu trình Prequential (Test-then-Train) ở Mỗi Bước Thời gian t",
        "left_bullets": [
            "STAGE A — Online Inference & Evaluation (Pha Suy luận - Màu xanh dương):",
            "• Nhận batch mẫu mới {x_{t,k}} ~ P_t -> Dự đoán bằng mô hình đóng băng θ_t -> Nhận nhãn thực tế y_{t,k} và ghi nhận Dynamic Regret reg_t.",
            "STAGE B — Online Training & Adaptive LR (Pha Huấn luyện - Màu xanh lá):",
            "• Tính gradient trung bình mẫu ∇l^B_t(θ_t) -> Ước lượng độ trôi (Oracle γ_{t-1} hoặc EMA γ̂_t) -> Tính η*_t (Alg.1 / Prop.4.3 / Eq.25) -> Cập nhật θ_{t+1}.",
        ],
        "right_title": "Sơ đồ Khối Pipeline 2 Tầng (`figures/pipeline.png`)",
        "right_placeholder": "[Chèn hình `figures/pipeline.png` do nhóm tự vẽ bằng code tại `experiments/smoke_test.py`]",
        "cites": [1],
        "notes": "Đi từng bước qua sơ đồ khối `figures/pipeline.png`. Làm rõ yêu cầu quan trọng đối với hệ thống Never-Ending Learning: tách biệt rạch ròi pha Suy luận (Inference với θ_t đóng băng để đánh giá công bằng) và pha Huấn luyện (Training cập nhật θ_{t+1} với tốc độ học thích ứng η*_t).",
    },
    {
        "num": 13,
        "section": "M2 — METHOD (LAYER 3: LINEAR SDE)",
        "owner": "TV1 (Khoa)",
        "title": "Hồi quy Tuyến tính: Thiết lập Bài toán & Xấp xỉ SDE Liên tục (Eq. 8)",
        "left_title": "Làm gì? Vì sao cần? Hoạt động thế nào? (`docs/03_linear.md`)",
        "left_bullets": [
            "Làm gì: Xấp xỉ quỹ đạo sai số rời rạc θ_t - θ*_t bằng quá trình ngẫu nhiên liên tục X(τ) qua phương trình SDE (8) với τ = εt, η_t = εζ(εt), B_t = εν(εt).",
            "Vì sao cần: Trong miền rời rạc, tích các ma trận ngẫu nhiên khiến việc giải nghiệm đóng η*_t là bất khả thi; chuyển sang SDE cho phép dùng Giải tích Itô.",
            "Phương trình SDE (8): dX(τ) = -(ζ(τ)X(τ) + Y'(τ))dτ + (ζ(τ)/sqrt(ν(τ))) ((||X||² + σ²)I + XX^T)^{1/2} dW(τ).",
            "Bảo chứng hội tụ (Proposition 3.1): Sai số giữa ||X(tε)||² và ||θ_t - θ*_t||² bị chặn bởi O(sqrt(ε)) với xác suất cao 1 - exp(-u²).",
        ],
        "right_title": "Giải phẫu 3 Thành phần Vật lý trong Phương trình SDE (8)",
        "right_placeholder": "[Sơ đồ giải thích 3 lực tác động lên sai số X(τ): (1) Lực kéo hội tụ -ζX, (2) Lực đẩy lệch do shift -Y'(τ), (3) Nhiễu khuếch tán mini-batch dW(τ)]",
        "cites": [1, 4],
        "notes": "Trình bày chi tiết cách xây dựng phương trình SDE (8) cho hồi quy tuyến tính y_{t,k} = <x_{t,k}, θ*_t> + ε_{t,k} với x ~ N(0, I). Giải thích vì sao kỳ vọng Regret tức thời E[reg_t] = 1/2 E[||θ_t - θ*_t||²] chuyển bài toán về việc kiểm soát moment bậc hai của X(τ).",
    },
    {
        "num": 14,
        "section": "M2 — METHOD (LAYER 3: LINEAR SDE)",
        "owner": "TV1 (Khoa)",
        "title": "Hồi quy Tuyến tính: Hệ Phương trình Moment ODE & Điều khiển HJB",
        "left_title": "Từ SDE Ngẫu nhiên d Chiều về Bài toán Điều khiển Vô hướng 1 Chiều",
        "left_bullets": [
            "Theorem 3.2 (Bổ đề Itô): Dẫn xuất hệ ODE tất định cho m_τ = E[X(τ)] và v_τ = E[||X(τ)||²]:",
            "• m'_τ = -ζ(τ)m_τ - Y'(τ)  (Eq. 10)",
            "• v'_τ = ((d+1)ζ²/ν - 2ζ)v_τ + (ζ²/ν)σ²d - 2m_τ^T Y'(τ)  (Eq. 11)",
            "Quá trình bao trên ṽ_τ (Eq. 12): Dùng |m_τ^T Y'| <= ||Y'|| sqrt(v_τ) để khử hướng, thu được ODE 1 chiều ṽ'_τ.",
            "Theorem 3.3 (Phương trình HJB): Cực tiểu hóa ∫ ṽ_τ dτ dẫn tới chính sách tối ưu đóng: ζ*(τ) = min{1, ṽ_τ / (((d+1)/ν)ṽ_τ + (σ²d)/ν)} (Eq. 13).",
        ],
        "right_title": "Hành vi Lịch trình Tối ưu dưới Bursty & Smooth Shifts (Figure 2 & 3)",
        "right_placeholder": "[Chèn hình `figures/fig2_alg1_schedules.png` và `figures/fig3_no_shift_asymptotics.png`]",
        "cites": [1],
        "notes": "Giải thích bước đột phá kỹ thuật của Mục 3: dùng Bổ đề Itô để chuyển SDE ngẫu nhiên sang ODE tất định, dùng bất đẳng thức Cauchy-Schwarz lập quá trình bao trên ṽ_τ, rồi giải phương trình Hamilton-Jacobi-Bellman (HJB) thu được nghiệm đóng Eq. (13). Nêu rõ kết quả Lemma 3.6: khi không có shift (γ_t = 0), lịch trình tự động suy giảm theo quy luật cổ điển O(1/t).",
    },
    {
        "num": 15,
        "section": "M2 — METHOD (LAYER 3: LINEAR SDE)",
        "owner": "TV1 (Khoa)",
        "title": "Algorithm 1: Lịch trình Tốc độ học Tối ưu Thời gian thực cho Hồi quy Tuyến tính",
        "left_title": "Cơ chế Rời rạc hóa Forward Euler & Trực giác Toán học",
        "left_bullets": [
            "Thuật toán: Chia mỗi vòng SGD thành ceil(1/κ) bước con Euler thuận để cập nhật trạng thái phương sai v theo độ dịch chuyển γ_{t-1}:",
            "• r <- min( v B_t / ((d+1)v + σ²d), ε )",
            "• v <- v + κ(((d+1)/B_t)r² - 2r)v + κ(σ²d/B_t)r² + 2κ γ_{t-1} sqrt(v)",
            "Vì sao η*_t TĂNG khi có shift? Số hạng +2κ γ_{t-1} sqrt(v) bơm năng lượng làm v tăng vọt; mà hàm η*(v) đồng biến nghiêm ngặt theo v -> đẩy η*_t tăng lên tới trần ε!",
            "Chế độ thực tế (Remark 3.5): Khi ẩn γ_{t-1}, thay bằng trung bình trượt EMA γ̂_t = β γ̂_{t-1} + (1-β)||θ_t - θ_{t-1}||_2.",
        ],
        "right_title": "Lưu đồ Thuật toán Algorithm 1 (`LinearOptimalSchedule`)",
        "right_placeholder": "[Lưu đồ khối của Algorithm 1 tại mỗi bước t: Nhận γ_{t-1} -> Vòng lặp ceil(1/κ) bước Euler cập nhật v -> Xuất η*_t cho SGD]",
        "cites": [1],
        "notes": "Trình bày chi tiết Algorithm 1 và trả lời trực diện câu hỏi trọng tâm của Task T7: Tại sao tốc độ học tối ưu lại tăng khi có distribution shift? Khi γ_{t-1} > 0, khoảng cách kỳ vọng tới nghiệm tối ưu v tăng lên, làm tỷ số tín hiệu trên nhiễu (SNR) của gradient lớn hơn nên mô hình cần bước nhảy lớn để thu hẹp khoảng cách.",
    },
    {
        "num": 16,
        "section": "M2 — METHOD (LAYER 3: CONVEX LOSS)",
        "owner": "TV2 (Khôi)",
        "title": "Hàm Lồi Tổng quát: Thiết lập Giả thiết & Chặn trên Dynamic Regret (Theorem 4.2)",
        "left_title": "Mở rộng sang Hàm Mất mát Lồi Bất kỳ l(θ, z) (`docs/04_convex.md`)",
        "left_bullets": [
            "Làm gì & Vì sao cần: Với hàm lồi tổng quát (như Logistic Regression), không có nghiệm SDE đóng -> cần thiết lập chặn trên giải tích cho E[Reg(T)] để tối ưu tuần tự.",
            "Assumption 4.1: (i) Phương sai gradient bị chặn bởi σ²; (ii) Hàm kỳ vọng l̄_t là L-smooth; (iii) Miền tham số Θ bị chặn với đường kính D_max.",
            "Theorem 4.2 (Eq. 15–16): Với a_t = 2η_t - Lη_t² > 0 và η_t <= 1/L, tổng Regret kỳ vọng bị chặn trên bởi tổng vi phân telescospic của 1/a_t cùng phương sai nhiễu σ²η_t²/B_t và độ trôi γ_t² + 2D_max γ_t.",
        ],
        "right_title": "Sơ đồ Cấu trúc Chặn trên Regret trong Theorem 4.2",
        "right_placeholder": "[TV2 vẽ sơ đồ tách 3 thành phần của Upper Bound (16): (1) Penalty giảm bước nhảy (1/a_t - 1/a_{t-1})_+ | (2) Nhiễu mini-batch σ²/B_t | (3) Độ trôi phân phối γ_t]",
        "cites": [1],
        "notes": "Giải thích cách bài báo vượt qua khó khăn khi chuyển từ hàm bậc hai sang hàm lồi tổng quát: sử dụng tính L-smooth, phép chiếu co Π_Θ và lấy kỳ vọng có điều kiện trên filtration z_{[t-1]} để thu được cận trên khả kiến (observable upper bound) ở Công thức (16).",
    },
    {
        "num": 17,
        "section": "M2 — METHOD (LAYER 3: CONVEX LOSS)",
        "owner": "TV2 (Khôi)",
        "title": "Proposition 4.3: Lịch trình Tốc độ học Thích ứng Hai Ngưỡng [τ_{1,t}, τ_{2,t}]",
        "left_title": "Quy tắc Kẹp Ngưỡng Đóng (Công thức 18–20)",
        "left_bullets": [
            "Cực tiểu hóa hàm mục tiêu lồi (17) theo η in (0, 1/L] tại mỗi bước t sinh ra 2 ngưỡng đóng τ_{1,t} <= τ_{2,t} <= 1/L:",
            "• τ_{i,t} = (B_t / 2σ²) ( sqrt(b_{i,t}² L² + (4σ²/B_t)b_{i,t}) - b_{i,t} L ), với b_{1,t} = γ_t² + 2D_max γ_t, b_{2,t} = (γ_t + D_max)².",
            "Quy tắc cập nhật η*_t = clip(η*_{t-1}, τ_{1,t}, τ_{2,t}):",
            "• Nếu η*_{t-1} < τ_{1,t} (shift tăng mạnh): Nâng ngay η*_t lên τ_{1,t}.",
            "• Nếu τ_{1,t} <= η*_{t-1} <= τ_{2,t}: Giữ nguyên η*_t = η*_{t-1} để tránh phạt đại lượng (1/a_t - 1/a_{t-1})_+.",
            "• Nếu η*_{t-1} > τ_{2,t}: Hạ η*_t xuống τ_{2,t}.",
        ],
        "right_title": "Đồ thị Minh họa Vùng Chết (Deadzone) và Hai Ngưỡng τ_{1,t}, τ_{2,t}",
        "right_placeholder": "[TV2 vẽ đồ thị hàm mục tiêu có điểm gãy tại η_{t-1} và minh họa 2 ngưỡng τ_{1,t}, τ_{2,t} đồng biến theo độ trôi γ_t]",
        "cites": [1],
        "notes": "Trình bày Proposition 4.3 và trực giác của vùng giữ nguyên [τ_{1,t}, τ_{2,t}]. Nhấn mạnh kỹ thuật hữu tỷ hóa (rationalization) mà nhóm đã áp dụng trong `ConvexSchedule` (`core/schedules.py`) để chống hiện tượng trừ hai số gần bằng nhau (catastrophic cancellation) khi γ_t rất nhỏ.",
    },
    {
        "num": 18,
        "section": "M2 — METHOD (LAYER 3: NON-CONVEX LOSS)",
        "owner": "TV3 (Hòa)",
        "title": "Hàm Không lồi: Định nghĩa Regret theo Chuẩn Gradient & Độ trôi Toàn cục",
        "left_title": "Tại sao phải Định nghĩa lại Regret cho Mạng Nơ-ron Sâu? (Mục 5)",
        "left_bullets": [
            "Thách thức: Cảnh quan hàm mất mát không lồi chứa vô số cực tiểu địa phương (local minima) và điểm yên ngựa (saddle points); SGD không thể đảm bảo hội tụ về cực tiểu toàn cục θ*_t.",
            "Định nghĩa Regret mới (Eq. 22): Reg(T) := Σ_{t=1}^T ||∇l̄_t(θ_t)||_2² (đo tốc độ duy trì quỹ đạo gần các điểm dừng của chuỗi hàm mất mát).",
            "Định nghĩa Độ trôi toàn cục (Eq. 23): γ_t := sup_{θ in R^p} |l̄_t(θ) - l̄_{t+1}(θ)| (khoảng cách đều giữa hai bề mặt mất mát liên tiếp).",
        ],
        "right_title": "Minh họa Cực tiểu Địa phương Dịch chuyển & Ý nghĩa của ||∇l̄_t(θ_t)||²",
        "right_placeholder": "[TV3 vẽ hình bề mặt mất mát không lồi dịch chuyển từ bước t sang t+1 làm điểm cực tiểu cũ biến thành sườn dốc có ||∇l̄_{t+1}(θ_t)|| lớn]",
        "cites": [1],
        "notes": "Giải thích lý do không thể dùng l̄_t(θ_t) - l̄_t(θ*_t) cho bài toán không lồi và ý nghĩa của việc đo tổng bình phương chuẩn gradient E[||∇l̄_t(θ_t)||²].",
    },
    {
        "num": 19,
        "section": "M2 — METHOD (LAYER 3: NON-CONVEX LOSS)",
        "owner": "TV3 (Hòa)",
        "title": "Theorem 5.1 & Lịch trình Tốc độ học Đóng cho Hàm Không lồi",
        "left_title": "Chặn trên Regret (Eq. 24) và Công thức Bước nhảy Tối ưu (Eq. 25)",
        "left_bullets": [
            "Theorem 5.1: Dưới giả thiết L-smooth và phương sai gradient bị chặn bởi σ², với a_t = 2η_t - Lη_t² > 0:",
            "• E[Reg(T)] <= E[ 2l̄_1(θ_1)/a_1 + Σ_{t=2}^T 2l̄_t(θ_t)(1/a_t - 1/a_{t-1}) ] + Σ_{t=1}^T E[ (1/a_t)(Lσ²η_t²/B_t + 2γ_t) ].",
            "Công thức đóng tối ưu từng bước (Eq. 25):",
            "• η*_t = (B_t / Lσ²) ( sqrt(b_t² + (2σ²/B_t)b_t) - b_t ),  với b_t := L(γ_t + l̄_t(θ_t)).",
            "Trực giác: Khi γ_t hoặc hàm mất mát hiện tại l̄_t(θ_t) tăng vọt -> b_t tăng -> η*_t tăng lên (tiến gần 1/L) giúp mô hình thoát khỏi cực tiểu cục bộ cũ!",
        ],
        "right_title": "Đồ thị Hàm η*_t(b_t) và Liên hệ với Cơ chế Adam Restart",
        "right_placeholder": "[TV3 vẽ đồ thị η*_t tăng đơn điệu theo b_t = L(γ_t + l̄_t(θ_t)) và cơ chế Reset bộ nhớ moment bậc hai của Adam khi chuyển pha]",
        "cites": [1, 5],
        "notes": "Trình bày Theorem 5.1, công thức đóng (25) và cách xấp xỉ l̄_t(θ_t) bằng loss trung bình trên mini-batch l^B_t(θ_t). Liên hệ lý thuyết này với thủ tục Restart bộ tối ưu Adam trong thực nghiệm mạng nơ-ron sâu.",
    },
    {
        "num": 20,
        "section": "M2 — METHOD (LAYER 4: THEORETICAL PROPERTIES)",
        "owner": "TV2 (Khôi)",
        "title": "Tổng hợp Các Định lý Chính, Trực giác & Giới hạn Áp dụng",
        "left_title": "Bảng Tóm tắt 6 Kết quả Lý thuyết Cốt lõi của Bài báo",
        "left_bullets": [
            "Proposition 3.1 & Theorem 3.2–3.3 (Linear): Xấp xỉ SDE sai số O(sqrt(ε)); nghiệm điều khiển tối ưu HJB trên bao trên moment bậc hai.",
            "Lemma 3.6 (Static Limit): Khi γ_t = 0, lịch trình SDE giữ mức trần ε đến τ_* rồi suy giảm tiệm cận đúng bậc O(1/t).",
            "Theorem 4.2 & Proposition 4.3 (Convex Upper Bound): Chặn trên Regret cho hàm lồi L-smooth và quy tắc kẹp 2 ngưỡng [τ_{1,t}, τ_{2,t}].",
            "Theorem 4.7 (Strongly Convex Lower Bound): Cận dưới Regret cho hàm μ-lồi mạnh có cùng dạng hàm với Theorem 4.2, chứng tỏ phân tích chặt đến hằng số.",
            "Theorem 5.1 (Non-Convex): Chặn trên tổng bình phương chuẩn gradient và nghiệm đóng η*_t.",
        ],
        "right_title": "Bảng Phân tích Giả thiết & Khi nào Giả thiết bị Vi phạm?",
        "right_placeholder": "[TV2 lập bảng 4 cột: Định lý | Giả thiết chính | Trực giác cốt lõi | Giới hạn / Khi nào bị vi phạm (vd: x không đẳng hướng, d >> B_t, L không biết trước)]",
        "cites": [1],
        "notes": "Tổng hợp toàn bộ các định lý/bổ đề trong bài báo theo đúng yêu cầu Layer 4 của Milestone 2: phát biểu bằng lời của nhóm, nêu trực giác toán học và chỉ rõ giới hạn nơi các giả thiết bị vi phạm (chuẩn bị tiền đề cho Milestone 3).",
    },
    {
        "num": 21,
        "section": "M2 — METHOD (LAYER 5: COMPLEXITY)",
        "owner": "TV2 (Khôi)",
        "title": "Độ phức tạp Tính toán & Bộ nhớ (Time & Space Complexity Story)",
        "left_title": "Chi phí Tính toán Tăng thêm (Overhead) của Các Lịch trình Đề xuất",
        "left_bullets": [
            "Chi phí nền của Mini-batch SGD tiêu chuẩn mỗi bước: Thời gian O(B_t · d) (hoặc O(B_t · P) với mạng P tham số), Bộ nhớ O(d).",
            "Algorithm 1 (Linear): Giải ODE vô hướng 1 chiều với ceil(1/κ) bước Euler -> Thời gian O(ceil(1/κ)) = O(1) phép toán vô hướng, Bộ nhớ O(1) (chỉ lưu 1 biến thực v).",
            "Proposition 4.3 (Convex) & Theorem 5.1 (Non-convex): Tính công thức đóng τ_{1,t}, τ_{2,t} -> Thời gian O(1), Bộ nhớ O(1) (nếu dùng EMA γ̂_t cần thêm O(d) để lưu θ_{t-1}).",
            "Kết luận: Chi phí điều khiển tốc độ học hoàn toàn không đáng kể (< 0.1% thời gian chạy) so với bước tính gradient mini-batch!",
        ],
        "right_title": "Bảng So sánh Độ phức tạp Thời gian & Bộ nhớ với các Phương pháp Tiền nhiệm",
        "right_placeholder": "[TV2 chèn Bảng so sánh: Phương pháp | Thời gian cập nhật LR / bước | Bộ nhớ phụ | Yêu cầu biết trước T, V_T?]",
        "cites": [1, 2, 3],
        "notes": "Phân tích câu chuyện độ phức tạp (Complexity Story): chứng minh rằng cả 3 lịch trình trong bài báo chỉ tốn O(1) phép toán vô hướng ở mỗi vòng lặp, hoàn toàn khả thi cho các hệ thống streaming quy mô lớn.",
    },
    {
        "num": 22,
        "section": "LAB 1 — EXPERIMENTAL SETUP",
        "owner": "TV3 (Hòa)",
        "title": "Lựa chọn Phạm vi Thực nghiệm: FULL SCOPE (100% Bài báo + Bonus)",
        "left_title": "Lý do Nhóm Chọn Full Scope & Bảng Ánh xạ Thí nghiệm",
        "left_bullets": [
            "Quyết định của nhóm: Chọn FULL SCOPE — tái hiện 100% cả 3 bài toán thực nghiệm của bài báo + làm thêm 1 bài phân tích độ nhạy mở rộng.",
            "Lý do khả thi: (1) Chi phí tính toán nhẹ (toàn bộ các thí nghiệm chạy dưới 10 phút trên CPU); (2) Dữ liệu tổng hợp công thức (26) và dữ liệu sinh học `scvelo.datasets.pancreas()` tải tự động 100% bằng code.",
            "Bao phủ trọn vẹn: Thực nghiệm 1 (Linear - Fig 4 & 5-top, TV1), Thực nghiệm 2 (Logistic - Fig 5-bottom + Sensitivity, TV2), Thực nghiệm 3 (Flow Cytometry MLP - Fig 6 + Code gốc, TV3).",
        ],
        "right_title": "Bảng Tổng hợp Ma trận Thực nghiệm Full Scope",
        "right_placeholder": "[TV3 chèn Bảng: Thí nghiệm | Hình trong bài | Script trong Repo | Số Seeds | Tổng số Runs W&B | Người phụ trách]",
        "cites": [1, 5],
        "notes": "Nêu rõ lựa chọn Full Scope của nhóm theo đúng yêu cầu của Lab 1 và giải thích vì sao việc tái hiện toàn bộ bài báo kèm phân tích độ nhạy là hoàn toàn khả thi và mang lại giá trị kiểm chứng cao nhất.",
    },
    {
        "num": 23,
        "section": "LAB 1 — EXPERIMENTAL SETUP",
        "owner": "TV2 (Khôi)",
        "title": "Thiết lập Môi trường, Tracking W&B & Kỷ luật Tái lập (Reproducibility)",
        "left_title": "Kiến trúc Thực nghiệm Chuẩn hóa (`core/` + `requirements.txt` + W&B)",
        "left_bullets": [
            "Môi trường cô lập: CPython 3.12, ghim cứng phiên bản trong `requirements.txt` (PyTorch 2.11.0, NumPy 1.26.4, SciPy 1.15.2, W&B 0.31.0); kiểm tra tự động qua `scripts/check_environment.py`.",
            "Cố định hạt giống ngẫu nhiên: Mọi cấu hình trong `configs/*.yaml` chạy trên 3 seeds độc lập {0, 1, 2} để báo cáo Mean ± SD.",
            "Theo dõi thực nghiệm: Tất cả các lần chạy đều đẩy tự động lên W&B Public Project `gr_ml_29_37_37/ml-distribution-shift` với tag `lab1` thông qua module `core/tracking.py`.",
            "Kiểm thử tự động: 99/99 unit tests & integration tests trong `tests/` đạt PASS 100%.",
        ],
        "right_title": "Sơ đồ Kiến trúc Tái lập & Luồng Kiểm chứng W&B",
        "right_placeholder": "[TV2 vẽ sơ đồ: configs/*.yaml + set_seed(0,1,2) -> core/trainer.py -> core/tracking.py (init_run) -> W&B Public Dashboard + CSV/PNG]",
        "cites": [1],
        "notes": "Trình bày kỷ luật kỹ thuật của nhóm trong Lab 1: quản lý thư viện ghim phiên bản, kiểm soát RNG không rò rỉ tương lai, xác thực W&B Public và hệ thống 99 unit tests bảo vệ tính đúng đắn của code.",
    },
    {
        "num": 24,
        "section": "LAB 1 — RESULTS (LINEAR REGRESSION)",
        "owner": "TV1 (Khoa)",
        "title": "Thực nghiệm 1 (Linear): Quỹ đạo Xoắn ốc 2D & Dao động Learning Rate (Figure 4)",
        "left_title": "Kết quả Tái hiện Figure 4 (`experiments/linear_regression.py`)",
        "left_bullets": [
            "Cấu hình chuẩn Mục 6.1: d = 2, T = 2000, B_t = 256, σ² = 0.1, η_max = 0.5, k = 4, r_{1,-1}(t) xoắn vào gốc tọa độ rồi xoắn ra.",
            "Hiện tượng quan sát được khi tăng bước rời rạc hóa l in {1, 4, 16}:",
            "• l = 1 (trôi liên tục, W&B `yaomsd9p`): η_t giảm mượt về 0.0068 quanh t=1000 (khi θ*_t ở gần gốc 0) rồi tăng lại lên 0.5; Regret = 0.916 ± 0.045.",
            "• l = 4 (W&B `id9my7jj`): η_t bắt đầu dao động với 215 lần nảy lại lên trần 0.5; Regret = 0.956 ± 0.046.",
            "• l = 16 (W&B `zrhn9935`): Giữa các bước nhảy (γ_t = 0), η_t suy giảm nhanh, nhưng cứ mỗi 16 bước khi θ̂*_t nhảy bậc, η_t lập tức vọt lên đúng η_max = 0.5 (74 đỉnh); Regret = 1.218 ± 0.054.",
        ],
        "right_title": "Đồ thị Tái hiện Figure 4 (`figures/fig4_linear_trajectory_and_lr.png`)",
        "right_placeholder": "[Chèn hình `figures/fig4_linear_trajectory_and_lr.png` (2 hàng x 3 cột cho l = 1, 4, 16)]",
        "cites": [1],
        "notes": "Phân tích kết quả tái hiện Figure 4 của bài báo kèm link W&B runs (`yaomsd9p`, `id9my7jj`, `zrhn9935`). Chỉ rõ sự khớp hoàn hảo giữa hình của nhóm và hình trong bài báo: cả quỹ đạo xoắn ốc 2D (hàng trên) lẫn biên độ dao động của η_t từ 0.0065 đến η_max = 0.5 (hàng dưới).",
    },
    {
        "num": 25,
        "section": "LAB 1 — RESULTS (LINEAR REGRESSION)",
        "owner": "TV1 (Khoa)",
        "title": "Thực nghiệm 1 (Linear): Cumulative Regret theo Số chiều d (Figure 5-top)",
        "left_title": "Đối chiếu Số liệu Tái hiện Figure 5 (top) trên 3 Seeds {0, 1, 2}",
        "left_bullets": [
            "Cấu hình: l = 8, B_t = 256, T = 2000, η_max = 1/sqrt(d), quét số chiều d in {2, 8, 32, 128, 512} (15 W&B runs online của `nkhoa8508`).",
            "Kết quả Cumulative Regret cuối (Mean ± SD) & Link W&B:",
            "• d = 2:   0.89 ± 0.04 (Paper ~0.9, lệch ~1.1% | W&B `jnphcs45`) — giảm 98.1% so với lịch trình 1/t (47.20).",
            "• d = 8:   13.40 ± 0.13 (Paper ~13.3, lệch ~0.8% | W&B `6kznv467`) — giảm 96.3% so với lịch trình 1/t (358.43).",
            "• d = 32:  792.18 ± 8.50 (Paper ~800, lệch ~1.0% | W&B `1fojwok2`) — thấp hơn gấp đôi Constant LR (1743.19).",
            "• d = 128: 7049.30 ± 16.76 (W&B `hupxdej7`) | d = 512: 33034.47 ± 7.76 (W&B `c82qd3ou`).",
            "Đặc trưng đường cong: Tăng nhanh ở đầu -> đi ngang ở t in [500, 1500] (khi xoắn vào tâm 0) -> tăng lại ở cuối.",
        ],
        "right_title": "Đồ thị Figure 5 (top) & So sánh Baselines (`figures/fig5_top_*.png`)",
        "right_placeholder": "[Chèn hình `figures/fig5_top_linear_regret_and_path.png` và `figures/fig5_top_baselines_comparison.png`]",
        "cites": [1],
        "notes": "Trình bày kết quả tái hiện Figure 5 (top-left và top-right) kèm bảng so sánh với 3 lịch trình baseline và bản ước lượng trực tuyến EMA γ̂_t. Nhấn mạnh rằng ngay cả khi không biết trước γ_t và phải dùng EMA γ̂_t, Algorithm 1 vẫn đạt Regret gần như trùng khít với bản Oracle (ví dụ ở d=128: 7049.49 so với 7049.30).",
    },
    {
        "num": 26,
        "section": "LAB 1 — RESULTS (CONVEX / LOGISTIC REGRESSION)",
        "owner": "TV2 (Khôi)",
        "title": "Thực nghiệm 2 (Logistic): Quỹ đạo & Regret Hàm Lồi (Figure 5-bottom)",
        "left_title": "Kết quả 45 W&B Runs Chính thức (CP04 — `docs/results/cp04-cp05/`)",
        "left_bullets": [
            "Cấu hình: Logistic Regression với nhãn mềm y = sigmoid(Xθ*_t + ε), L = 0.25, σ² = d/4, D_max = d, bán kính chiếu R = d/2, EMA β = 0.9.",
            "Rủi ro dư tích lũy quần thể B (Mean ± SD qua 3 seeds {0, 1, 2}):",
            "• d = 2:   Convex = 0.34 ± 0.01 vs. Constant = 7.25 ± 0.04 vs. Inv-Sqrt = 29.46 ± 0.08 (W&B `38easaio`).",
            "• d = 8:   Convex = 2.03 ± 0.01 vs. Constant = 53.54 ± 0.22 vs. Inv-Sqrt = 88.42 ± 0.04 (W&B `c5mhu323`).",
            "• d = 32:  Convex = 40.04 ± 0.27 vs. Constant = 180.43 ± 0.35 (W&B `52miko4w`).",
            "• d = 128: Convex = 233.52 ± 0.59 vs. Constant = 334.95 ± 0.28 (W&B `br6md2c7`).",
            "• Phát hiện thú vị ở d = 512: Do D_max = d = 512 làm ngưỡng τ_{1,t}, τ_{2,t} bảo thủ đẩy η_t sát trần 1/L = 4.0, Convex chịu B = 1385.01 cao hơn Constant (504.90).",
        ],
        "right_title": "Đồ thị Figure 5 (bottom) & Quỹ đạo d=128 (`cp04_*.png`)",
        "right_placeholder": "[TV2 chèn hình `docs/results/cp04-cp05/figures/cp04_figure5_left.png` và `cp04_trajectory_d128.png`]",
        "cites": [1],
        "notes": "Trình bày kết quả 45 runs chính thức của Thực nghiệm 2 (Logistic Regression). Phân tích sâu phát hiện thực nghiệm ở chiều siêu cao d=512: khi đặt D_max = d quá lớn, cận trên trong Theorem 4.2 trở nên lỏng khiến ngưỡng τ_{1,t} đẩy learning rate lên gần 4.0 gây phương sai lớn — một minh chứng giá trị cho phân tích phê bình của nhóm.",
    },
    {
        "num": 27,
        "section": "LAB 1 — RESULTS (FLOW CYTOMETRY MLP)",
        "owner": "TV3 (Hòa)",
        "title": "Thực nghiệm 3 (Flow Cytometry): Ứng dụng Y sinh Phân loại Tế bào Tụy",
        "left_title": "Thiết lập Mô phỏng Dòng chảy Tế bào Đơn bào (Phụ lục E) [1], [5]",
        "left_bullets": [
            "Bộ dữ liệu thực: 4.000 tế bào tuyến tụy chuột (`scvelo.datasets.pancreas()`) với 10 marker genes: Pyy, Meg3, Malat1, Gcg, Gnas, Actb, Ghrl, Rsp3, Ins2, Hspa8.",
            "4 phân lớp tế bào (K = 4): Ductal (không tiết), Alpha, Beta, Delta (tế bào nội tiết); nhiễu nhân x_j <- x_j · U với U ~ Uniform[0.9, 1.1].",
            "Kịch bản 4 pha dịch chuyển phân phối lớp qua T = 600 bước (B_t = 64):",
            "• Pha 1 (t = 1..100):   P_1(y) = (0.0, 0.0, 0.0, 1.0) — 100% tế bào Ductal.",
            "• Pha 2 (t = 101..200): P_2(y) = (0.0, 0.0, 0.1, 0.9) — Xuất hiện tế bào Beta.",
            "• Pha 3 (t = 201..400): P_3(y) = (0.1, 0.0, 0.2, 0.7) — Tăng tỷ lệ tế bào tiết.",
            "• Pha 4 (t = 401..600): P_4(y) = (0.3, 0.5, 0.1, 0.1) — Đảo chiều sang Alpha & Delta.",
        ],
        "right_title": "Kiến trúc Mạng Student MLP 5 Tầng & Phân bố Dữ liệu (Figure 6-left)",
        "right_placeholder": "[TV3 chèn hình nhúng 2D 10-gene (Figure 6-left) và sơ đồ kiến trúc MLP: Input(10) -> 64 -> 32 -> 16 -> 8 -> 4 (ELU + Cross-Entropy)]",
        "cites": [1, 5],
        "notes": "Mô tả bài toán ứng dụng y sinh thực tế trong Phụ lục E: hệ thống Teacher-Student phân loại tế bào dòng chảy theo thời gian thực khi tỷ lệ các loại tế bào biến thiên mạnh từ vùng đầu tụy sang đuôi tụy.",
    },
    {
        "num": 28,
        "section": "LAB 1 — RESULTS (FLOW CYTOMETRY MLP)",
        "owner": "TV3 (Hòa)",
        "title": "Thực nghiệm 3 (Flow Cytometry): Hiệu quả của Cơ chế Adam Restart (Figure 6-right)",
        "left_title": "So sánh Adam Cố định vs. Adam Restart tại Mốc Dịch chuyển t in {100, 200, 400}",
        "left_bullets": [
            "4 cấu hình so sánh: `adam:0.01`, `adam-restart:0.01`, `adam:0.03`, `adam-restart:0.03` (kèm khảo sát η_0 in {0.001, 0.003, 0.1}).",
            "Kết quả chính:",
            "1. Điểm ngọt (Sweet spot) của tốc độ học: η_0 = 0.01 đạt Regret thấp nhất; η_0 = 0.1 phân kỳ sớm, còn η_0 <= 0.003 thích ứng quá chậm.",
            "2. Lợi ích rõ rệt của Adam Restart: Khởi động lại bộ đệm moment của Adam tại t = 100, 200, 400 giúp xóa bỏ tích lũy đạo hàm bậc hai cũ, làm tăng bước nhảy hiệu dụng và hạ thấp đường cong Cumulative Regret.",
            "3. Hiện tượng mắc kẹt cực tiểu cục bộ: `adam:0.03` không có restart bị kẹt ở pha 3–4 (độ dốc Regret giữ nguyên không giảm).",
        ],
        "right_title": "Đồ thị Tái hiện Figure 6 (right) — Cumulative Regret qua 600 Bước",
        "right_placeholder": "[TV3 chèn đồ thị `figures/fig6_cytometry_regret.png` có 3 đường kẻ dọc đứt nét tại t = 100, 200, 400 kèm link W&B run]",
        "cites": [1, 5],
        "notes": "Phân tích kết quả tái hiện Figure 6 (right) và giải thích cơ chế kỹ thuật vì sao việc reset trạng thái nội tại (m_t, v_t) của bộ tối ưu Adam ngay khi phát hiện distribution shift đóng vai trò tương đương với việc tăng vọt tốc độ học trong lý thuyết.",
    },
    {
        "num": 29,
        "section": "LAB 1 — BONUS SENSITIVITY ANALYSIS",
        "owner": "TV2 (Khôi)",
        "title": "Phân tích Mở rộng (Bonus): Độ nhạy theo Tần suất Trôi (l) & Nhiễu Nhãn (σ)",
        "left_title": "Kết quả 15 W&B Runs Độ nhạy tại d = 128 (`experiments/sensitivity.py`)",
        "left_bullets": [
            "Khảo sát 1 — Thay đổi tần suất và tổng độ trôi phân phối l in {1, 8, 16}:",
            "• l = 16 (Σ γ_t = 339.3): B = 141.03 ± 0.86 (W&B `ctn81a74`).",
            "• l = 8  (Σ γ_t = 688.8 - Baseline): B = 233.52 ± 0.59 (W&B `7i4l64n4`).",
            "• l = 1  (Σ γ_t = 1848.5 - Trôi liên tục): B = 330.32 ± 2.29 (W&B `adva0mi7`).",
            "Khảo sát 2 — Thay đổi độ lệch chuẩn nhiễu nhãn σ in {0.0, sqrt(0.1), 1.0}:",
            "• σ = 0.0 (Không nhiễu): B = 224.22 ± 0.67 (W&B `lbwkwdmc`).",
            "• σ = sqrt(0.1) ≈ 0.316: B = 233.52 ± 0.59 | σ = 1.0 (Nhiễu lớn): B = 282.45 ± 0.80 (W&B `lihumbm1`).",
        ],
        "right_title": "Biểu đồ Phân tích Độ nhạy (`figures/sensitivity_drift.png` & `sensitivity_noise.png`)",
        "right_placeholder": "[TV2 chèn 2 biểu đồ `figures/sensitivity_drift.png` và `figures/sensitivity_noise.png`]",
        "cites": [1],
        "notes": "Trình bày bài phân tích độ nhạy mở rộng (Task T17) với 15 runs đã xác minh trên W&B. Kết quả khẳng định tính bền vững của lịch trình thích ứng: Regret chỉ tăng nhẹ tuyến tính theo tổng độ trôi Σ γ_t và rất ít nhạy cảm với nhiễu nhãn vừa phải.",
    },
    {
        "num": 30,
        "section": "LAB 1 — DEVIATION & CODE ANALYSIS",
        "owner": "TV3 (Hòa)",
        "title": "Phân tích Độ lệch (Deviation Analysis) & Đối chiếu Code Gốc Tác giả",
        "left_title": "So sánh Bản Tự Cài đặt của Nhóm vs. Repo `fahrbach/learning-rate-schedules`",
        "left_bullets": [
            "1. Khác biệt về định nghĩa Regret thực nghiệm trong Logistic Regression:",
            "• Bài báo định nghĩa lý thuyết reg_t = l̄_t(θ_t) - l̄_t(θ*_t) (so với nghiệm tối ưu thực sự của phân phối nhiễu), nhưng code gốc tính hiệu số trực tiếp trên mẫu với vector sinh dữ liệu θ*_t (có thể âm cục bộ do nhiễu). Nhóm đã cài đặt cả hai thước đo (A: Reference gap & B: True Population Oracle gap) để đối chiếu minh bạch.",
            "2. Khác biệt về khởi tạo trạng thái v_0 và độ chia κ trong Algorithm 1:",
            "• Nhóm phát hiện điểm bất động v = 0 của phương trình Euler hiển khi γ > 0 và bổ sung bước khởi tạo giải tích, giúp tái hiện chính xác tuyệt đối Figure 2, 3, 4, 5-top.",
            "3. Khác biệt Framework: Bài báo dùng TensorFlow/Keras; nhóm tái hiện bằng NumPy/SciPy/PyTorch chuẩn hóa với seed cố định.",
        ],
        "right_title": "Bảng Tổng hợp Độ lệch Tương đối (%) qua Cả 3 Bài toán",
        "right_placeholder": "[TV3/Cả nhóm chèn Bảng đối chiếu: Hình | Metric | Số Bài báo | Số Nhóm (Mean ± SD) | Độ lệch (%) | Nguyên nhân chính]",
        "cites": [1],
        "notes": "Trình bày phần phân tích độ lệch sâu sắc (Deviation Analysis) và đối chiếu với repo gốc của tác giả (Task T18). Đây là phần thể hiện rõ nhất việc nhóm thực sự hiểu sâu bản chất toán học và phát hiện được các chi tiết ẩn trong code gốc.",
    },
    {
        "num": 31,
        "section": "TỔNG KẾT & HƯỚNG MỞ RỘNG",
        "owner": "TV1 (Khoa)",
        "title": "Kết luận, Hạn chế Lý thuyết của Bài báo & Hướng Mở rộng (Milestone 3)",
        "left_title": "Thông điệp Cốt lõi & 3 Hạn chế Lý thuyết Chuẩn bị cho Cuối kỳ",
        "left_bullets": [
            "Takeaway cốt lõi: Trong học trực tuyến có dịch chuyển phân phối, tốc độ học tối ưu phải TĂNG LÊN khi độ trôi γ_t lớn để tăng cường khám phá và đuổi kịp nghiệm mới.",
            "Hạn chế 1 (Linear SDE): Giả định hiệp phương sai đặc trưng x_{t,k} ~ N(0, I_d) đẳng hướng và không đổi (chỉ xét Label Shift, chưa xét Covariate Shift khi ma trận Σ_t biến thiên).",
            "Hạn chế 2 (Convex Bound): Cận đường kính D_max = d trở nên quá lỏng khi số chiều d = 512 lớn, làm ngưỡng τ_{1,t} đẩy tốc độ học lên quá cao (đã chứng minh thực nghiệm ở Slide 26).",
            "Hạn chế 3 (Ước lượng độ trôi): EMA γ̂_t = ||θ_t - θ_{t-1}|| bị nhiễu nặng khi batch size B_t nhỏ vì trộn lẫn bước nhảy nhiễu gradient với độ trôi thực sự.",
        ],
        "right_title": "Sơ đồ Định hướng Cải tiến cho Milestone 3 & Lab 2 (Poster Workshop)",
        "right_placeholder": "[Sơ đồ 3 hướng mở rộng cho Cuối kỳ: (1) Hiệu chỉnh D_max hiệu dụng theo chuẩn thực tế ||θ_t|| | (2) Khử nhiễu phương sai khỏi ước lượng EMA γ̂_t | (3) Thử nghiệm Covariate Shift]",
        "cites": [1],
        "notes": "Tổng kết toàn bộ bài báo cáo giữa kỳ và nêu trước 3 điểm yếu lý thuyết/thực nghiệm mà nhóm đã phát hiện trong quá trình tái hiện, làm bàn đạp trực tiếp cho phần Critique & Poster ở Milestone 3 + Lab 2 cuối kỳ.",
    },
    {
        "num": 32,
        "section": "PHỤ LỤC & TÀI NGUYÊN",
        "owner": "TV3 (Hòa)",
        "title": "Bảng Phân công Đóng góp Thành viên & Liên kết Tài nguyên Public",
        "left_title": "Đóng góp Cụ thể của Từng Thành viên (100% Đều Hiểu Toàn Bài)",
        "left_bullets": [
            "TV1 — Khoa (Trưởng nhóm): Phụ trách Context & Tiền nhiệm (T4), Ký hiệu & Pipeline 2 pha (T5), Khung `core/` (T6), Lý thuyết SDE & `LinearOptimalSchedule` (T7), Thực nghiệm 1 Fig 4 & 5-top (T14, 19 W&B runs), Kiểm tra chéo T19 & Slides 3–7, 10–15, 24–25, 31.",
            "TV2 — Khôi: Phụ trách Môi trường & W&B Tracking `core/tracking.py` (T2), Đóng góp & Độ phức tạp (T8, T10), Lý thuyết Convex & `ConvexSchedule` (T9), Thực nghiệm 2 Fig 5-bottom & Độ nhạy (T15, T17, 65 W&B runs) & Slides 8–9, 16–17, 20–21, 23, 26, 29.",
            "TV3 — Hòa: Phụ trách Lý thuyết Non-convex & Dữ liệu `scvelo` (T11), Mạng MLP 5 tầng & Adam Restart (T12), Thực nghiệm 3 Fig 6 (T16), Đối chiếu code gốc (T18), Khung & Ghép Slide Deck (T20, T23) & Slides 1–2, 18–19, 22, 27–28, 30, 32.",
        ],
        "right_title": "Liên kết Truy cập Trực tiếp (Public 100%)",
        "right_placeholder": "[Khung tổng hợp Link: (1) GitHub Repo: https://github.com/nvk1701/github-pj | (2) W&B Public Dashboard: https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift]",
        "cites": [1],
        "notes": "Slide kết thúc: khẳng định sự phối hợp chặt chẽ của cả 3 thành viên, cung cấp đường dẫn trực tiếp tới GitHub Repository và Weights & Biases Public Dashboard chứa toàn bộ các runs mang tag `lab1`.",
    },
]


def _make_text_paragraphs(lines: List[str], font_size_pt: int = 13, color_hex: str = "1E293B", bold: bool = False, bullet: bool = False) -> str:
    """Sinh chuỗi XML `<a:p>...</a:p>` chuẩn DrawingML cho một danh sách dòng văn bản."""
    sz = int(font_size_pt * 100)
    b_attr = ' b="1"' if bold else ""
    paras = []
    for line in lines:
        safe_text = escape(str(line))
        if bullet:
            p_pr = '<a:pPr marL="228600" indent="-171450" algn="l"><a:lnSpc><a:spcPct val="115000"/></a:lnSpc><a:spcBef><a:spcPts val="500"/></a:spcBef><a:buChar char="•"/></a:pPr>'
        else:
            p_pr = '<a:pPr algn="l"><a:lnSpc><a:spcPct val="115000"/></a:lnSpc><a:spcBef><a:spcPts val="300"/></a:spcBef><a:buNone/></a:pPr>'
        paras.append(
            f"<a:p>{p_pr}"
            f'<a:r><a:rPr lang="vi-VN" altLang="en-US" sz="{sz}"{b_attr} dirty="0">'
            f'<a:solidFill><a:srgbClr val="{color_hex}"/></a:solidFill>'
            f'<a:latin typeface="Arial"/><a:ea typeface="Arial"/><a:cs typeface="Arial"/>'
            f"</a:rPr><a:t>{safe_text}</a:t></a:r></a:p>"
        )
    return "".join(paras)


def _make_shape_xml(
    shape_id: int,
    name: str,
    x: int,
    y: int,
    cx: int,
    cy: int,
    fill_hex: str,
    border_hex: str | None,
    paragraphs_xml: str,
    geom: str = "rect",
    anchor: str = "t",
    pad_l: int = 137160,
    pad_t: int = 91440,
) -> str:
    """Sinh một khối `<p:sp>` chuẩn OpenXML (hình chữ nhật hoặc bo góc kèm văn bản)."""
    ln_xml = (
        f'<a:ln w="12700"><a:solidFill><a:srgbClr val="{border_hex}"/></a:solidFill></a:ln>'
        if border_hex
        else "<a:ln><a:noFill/></a:ln>"
    )
    return f"""<p:sp>
  <p:nvSpPr>
    <p:cNvPr id="{shape_id}" name="{escape(name)}"/>
    <p:cNvSpPr/>
    <p:nvPr/>
  </p:nvSpPr>
  <p:spPr>
    <a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>
    <a:prstGeom prst="{geom}"><a:avLst/></a:prstGeom>
    <a:solidFill><a:srgbClr val="{fill_hex}"/></a:solidFill>
    {ln_xml}
  </p:spPr>
  <p:txBody>
    <a:bodyPr wrap="square" lIns="{pad_l}" tIns="{pad_t}" rIns="{pad_l}" bIns="{pad_t}" anchor="{anchor}"/>
    <a:lstStyle/>
    {paragraphs_xml}
  </p:txBody>
</p:sp>"""


def _build_slide_xml(spec: Dict[str, object], total_slides: int = 32) -> str:
    """Sinh nội dung `ppt/slides/slide{i}.xml` cho một slide theo đúng thông số thiết kế chung."""
    num = int(spec["num"])
    section = str(spec["section"])
    owner = str(spec["owner"])
    title = str(spec["title"])
    left_title = str(spec["left_title"])
    left_bullets = [str(x) for x in spec["left_bullets"]]  # type: ignore[index]
    right_title = str(spec["right_title"])
    right_placeholder = str(spec["right_placeholder"])
    cite_ids = [int(c) for c in spec["cites"]]  # type: ignore[index]

    # Màu badge theo người phụ trách để dễ nhận diện trên template
    if "TV1" in owner:
        owner_bg, owner_fg = "DBEAFE", "1E40AF"
    elif "TV2" in owner:
        owner_bg, owner_fg = "DCFCE7", "166534"
    else:
        owner_bg, owner_fg = "FEF3C7", "92400E"

    # 1. Nền tổng thể của slide (#F8FAFC)
    bg_shape = _make_shape_xml(
        shape_id=2,
        name="SlideBackground",
        x=0,
        y=0,
        cx=12192000,
        cy=6858000,
        fill_hex="F8FAFC",
        border_hex=None,
        paragraphs_xml=_make_text_paragraphs([""], font_size_pt=1),
    )

    # 2. Thanh Header trên cùng (#1E3A8A)
    header_paras = (
        _make_text_paragraphs([f"{section}   |   SLIDE {num:02d}/{total_slides:02d}"], font_size_pt=10, color_hex="93C5FD", bold=True)
        + _make_text_paragraphs([title], font_size_pt=19, color_hex="FFFFFF", bold=True)
    )
    header_shape = _make_shape_xml(
        shape_id=3,
        name="HeaderBanner",
        x=0,
        y=0,
        cx=12192000,
        cy=960000,
        fill_hex="1E3A8A",
        border_hex=None,
        paragraphs_xml=header_paras,
        pad_l=320000,
        pad_t=110000,
    )

    # 3. Nhãn Người phụ trách ở góc phải trên Header
    owner_shape = _make_shape_xml(
        shape_id=4,
        name="OwnerBadge",
        x=9850000,
        y=140000,
        cx=2050000,
        cy=340000,
        fill_hex=owner_bg,
        border_hex=owner_fg,
        paragraphs_xml=_make_text_paragraphs([f"Phụ trách: {owner}"], font_size_pt=10, color_hex=owner_fg, bold=True),
        geom="roundRect",
        anchor="ctr",
        pad_l=80000,
        pad_t=30000,
    )

    # 4. Khung Cột Trái (Nội dung phân tích & Công thức / Bảng)
    left_paras = (
        _make_text_paragraphs([left_title], font_size_pt=14, color_hex="1E3A8A", bold=True)
        + _make_text_paragraphs(left_bullets, font_size_pt=12, color_hex="1E293B", bold=False, bullet=True)
    )
    left_card = _make_shape_xml(
        shape_id=5,
        name="LeftContentCard",
        x=320000,
        y=1140000,
        cx=5620000,
        cy=4960000,
        fill_hex="FFFFFF",
        border_hex="CBD5E1",
        paragraphs_xml=left_paras,
        geom="roundRect",
        pad_l=200000,
        pad_t=160000,
    )

    # 5. Khung Cột Phải (Hình vẽ / Biểu đồ / Sơ đồ tự vẽ 100%)
    right_top_paras = _make_text_paragraphs([right_title], font_size_pt=14, color_hex="0F766E", bold=True)
    right_card = _make_shape_xml(
        shape_id=6,
        name="RightVisualCard",
        x=6160000,
        y=1140000,
        cx=5712000,
        cy=4960000,
        fill_hex="FFFFFF",
        border_hex="CBD5E1",
        paragraphs_xml=right_top_paras,
        geom="roundRect",
        pad_l=200000,
        pad_t=160000,
    )

    # Khung Placeholder bên trong Cột Phải hướng dẫn chèn hình cụ thể
    placeholder_paras = _make_text_paragraphs(
        [
            "KHUNG HÌNH VẼ / BIỂU ĐỒ / BẢNG TRỰC QUAN (TỰ VẼ 100%)",
            right_placeholder,
            "(Giữ nguyên khung viền của Cột Phải, chèn ảnh PNG độ phân giải cao hoặc vẽ bảng/sơ đồ trực tiếp vào vùng này)",
        ],
        font_size_pt=11,
        color_hex="475569",
        bold=False,
    )
    right_inner_box = _make_shape_xml(
        shape_id=7,
        name="VisualPlaceholderBox",
        x=6380000,
        y=1720000,
        cx=5272000,
        cy=4160000,
        fill_hex="F1F5F9",
        border_hex="94A3B8",
        paragraphs_xml=placeholder_paras,
        geom="rect",
        anchor="ctr",
        pad_l=220000,
        pad_t=160000,
    )

    # 6. Thanh Footer Trích dẫn chuẩn IEEE ở đáy mỗi slide (yêu cầu bắt buộc của đề bài)
    cite_str = "   ".join(IEEE_CITATIONS[c] for c in cite_ids if c in IEEE_CITATIONS)
    footer_paras = _make_text_paragraphs(
        [f"IEEE Reference: {cite_str}     |     Slide {num} / {total_slides}"],
        font_size_pt=8,
        color_hex="475569",
        bold=False,
    )
    footer_shape = _make_shape_xml(
        shape_id=8,
        name="IEEEFooterBar",
        x=0,
        y=6260000,
        cx=12192000,
        cy=598000,
        fill_hex="E2E8F0",
        border_hex="CBD5E1",
        paragraphs_xml=footer_paras,
        anchor="ctr",
        pad_l=320000,
        pad_t=40000,
    )

    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0"/>
          <a:ext cx="0" cy="0"/>
          <a:chOff x="0" y="0"/>
          <a:chExt cx="0" cy="0"/>
        </a:xfrm>
      </p:grpSpPr>
      {bg_shape}
      {header_shape}
      {owner_shape}
      {left_card}
      {right_card}
      {right_inner_box}
      {footer_shape}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>"""


def _build_notes_slide_xml(spec: Dict[str, object]) -> str:
    """Sinh nội dung `ppt/notesSlides/notesSlide{i}.xml` chứa Speaker Notes chuẩn (<= 500 từ)."""
    num = int(spec["num"])
    owner = str(spec["owner"])
    title = str(spec["title"])
    notes_body = str(spec["notes"])
    notes_lines = [
        f"[SPEAKER NOTES — SLIDE {num:02d} | Phụ trách: {owner} | Giới hạn: <= 500 từ]",
        f"Tiêu đề: {title}",
        f"Nội dung thuyết minh: {notes_body}",
    ]
    paras_xml = _make_text_paragraphs(notes_lines, font_size_pt=11, color_hex="000000", bold=False)

    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:notes xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
         xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
         xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0"/>
          <a:ext cx="0" cy="0"/>
          <a:chOff x="0" y="0"/>
          <a:chExt cx="0" cy="0"/>
        </a:xfrm>
      </p:grpSpPr>
      <p:sp>
        <p:nvSpPr>
          <p:cNvPr id="2" name="Notes Placeholder"/>
          <p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr>
          <p:nvPr><p:ph type="body" idx="1"/></p:nvPr>
        </p:nvSpPr>
        <p:spPr>
          <a:xfrm><a:off x="685800" y="4343400"/><a:ext cx="5486400" cy="4114800"/></a:xfrm>
          <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
        </p:spPr>
        <p:txBody>
          <a:bodyPr/>
          <a:lstStyle/>
          {paras_xml}
        </p:txBody>
      </p:sp>
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:notes>"""


def generate_template_pptx(output_path: Path) -> None:
    """Đóng gói toàn bộ cấu trúc OpenXML thành file `slides/template.pptx`."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_slides = len(SLIDES_SPEC)

    # 1. [Content_Types].xml
    slide_overrides = "\n".join(
        f'  <Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>\n'
        f'  <Override PartName="/ppt/notesSlides/notesSlide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml"/>'
        for i in range(1, n_slides + 1)
    )
    content_types_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/presProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presProps+xml"/>
  <Override PartName="/ppt/viewProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.viewProps+xml"/>
  <Override PartName="/ppt/tableStyles.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.tableStyles+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
  <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
  <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
  <Override PartName="/ppt/notesMasters/notesMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.notesMaster+xml"/>
{slide_overrides}
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""

    # 2. _rels/.rels
    root_rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""

    # 3. ppt/presentation.xml & ppt/_rels/presentation.xml.rels
    sld_id_entries = "\n".join(
        f'    <p:sldId id="{255 + i}" r:id="rIdSlide{i}"/>' for i in range(1, n_slides + 1)
    )
    presentation_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
                xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
                saveSubsetFonts="1">
  <p:sldMasterIdLst>
    <p:sldMasterId id="2147483648" r:id="rIdMaster1"/>
  </p:sldMasterIdLst>
  <p:notesMasterIdLst>
    <p:notesMasterId r:id="rIdNotesMaster1"/>
  </p:notesMasterIdLst>
  <p:sldIdLst>
{sld_id_entries}
  </p:sldIdLst>
  <p:sldSz cx="12192000" cy="6858000" type="screen16x9"/>
  <p:notesSz cx="6858000" cy="9144000"/>
  <p:defaultTextStyle>
    <a:defPPr><a:defRPr lang="vi-VN"/></a:defPPr>
  </p:defaultTextStyle>
</p:presentation>"""

    pres_slide_rels = "\n".join(
        f'  <Relationship Id="rIdSlide{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>'
        for i in range(1, n_slides + 1)
    )
    presentation_rels_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rIdMaster1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>
  <Relationship Id="rIdNotesMaster1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesMaster" Target="notesMasters/notesMaster1.xml"/>
  <Relationship Id="rIdPresProps" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/presProps" Target="presProps.xml"/>
  <Relationship Id="rIdViewProps" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/viewProps" Target="viewProps.xml"/>
  <Relationship Id="rIdTheme1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>
  <Relationship Id="rIdTableStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/tableStyles" Target="tableStyles.xml"/>
{pres_slide_rels}
</Relationships>"""

    # 4. Slide Master, Slide Layout, Notes Master & Theme
    slide_master_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:bg><p:bgRef idx="1001"><a:schemeClr val="bg1"/></p:bgRef></p:bg>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst>
    <p:sldLayoutId id="2147483649" r:id="rIdLayout1"/>
  </p:sldLayoutIdLst>
  <p:txStyles>
    <p:titleStyle/>
    <p:bodyStyle/>
    <p:otherStyle/>
  </p:txStyles>
</p:sldMaster>"""

    slide_master_rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rIdLayout1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rIdTheme1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>"""

    slide_layout_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
             type="blank" preserve="1">
  <p:cSld name="Blank">
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>"""

    slide_layout_rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rIdMaster1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>"""

    notes_master_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:notesMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
               xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
               xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
      <p:sp>
        <p:nvSpPr>
          <p:cNvPr id="2" name="Notes Placeholder"/>
          <p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr>
          <p:nvPr><p:ph type="body" idx="1"/></p:nvPr>
        </p:nvSpPr>
        <p:spPr>
          <a:xfrm><a:off x="685800" y="4343400"/><a:ext cx="5486400" cy="4114800"/></a:xfrm>
          <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
        </p:spPr>
        <p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:endParaRPr lang="vi-VN"/></a:p></p:txBody>
      </p:sp>
    </p:spTree>
  </p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:notesStyle/>
</p:notesMaster>"""

    notes_master_rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rIdTheme1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>"""

    theme_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="MLProjectTheme">
  <a:themeElements>
    <a:clrScheme name="AcademicBlue">
      <a:dk1><a:srgbClr val="0F172A"/></a:dk1>
      <a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>
      <a:dk2><a:srgbClr val="1E3A8A"/></a:dk2>
      <a:lt2><a:srgbClr val="F8FAFC"/></a:lt2>
      <a:accent1><a:srgbClr val="1D4ED8"/></a:accent1>
      <a:accent2><a:srgbClr val="0F766E"/></a:accent2>
      <a:accent3><a:srgbClr val="B45309"/></a:accent3>
      <a:accent4><a:srgbClr val="7C3AED"/></a:accent4>
      <a:accent5><a:srgbClr val="BE123C"/></a:accent5>
      <a:accent6><a:srgbClr val="15803D"/></a:accent6>
      <a:hlink><a:srgbClr val="2563EB"/></a:hlink>
      <a:folHlink><a:srgbClr val="7C3AED"/></a:folHlink>
    </a:clrScheme>
    <a:fontScheme name="ArialStandard">
      <a:majorFont><a:latin typeface="Arial"/><a:ea typeface="Arial"/><a:cs typeface="Arial"/></a:majorFont>
      <a:minorFont><a:latin typeface="Arial"/><a:ea typeface="Arial"/><a:cs typeface="Arial"/></a:minorFont>
    </a:fontScheme>
    <a:fmtScheme name="StandardFmt">
      <a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst>
      <a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln><a:ln w="19050"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln><a:ln w="25400"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst>
      <a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst>
      <a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst>
    </a:fmtScheme>
  </a:themeElements>
</a:theme>"""

    pres_props_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presProps xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"/>"""

    view_props_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:viewProps xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"/>"""

    table_styles_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:tblStyleLst xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" def="{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}"/>"""

    core_props_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
                   xmlns:dc="http://purl.org/dc/elements/1.1/"
                   xmlns:dcterms="http://purl.org/dc/terms/"
                   xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Learning Rate Schedules in the Presence of Distribution Shift — Midterm Deck Template</dc:title>
  <dc:creator>Team TV1 (Khoa), TV2 (Khoi), TV3 (Hoa)</dc:creator>
  <dcterms:created xsi:type="dcterms:W3CDTF">2026-11-18T00:00:00Z</dcterms:created>
</cp:coreProperties>"""

    app_props_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
            xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft Office PowerPoint</Application>
  <PresentationFormat>Widescreen</PresentationFormat>
  <Slides>{n_slides}</Slides>
  <Notes>{n_slides}</Notes>
</Properties>"""

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", root_rels_xml)
        zf.writestr("docProps/core.xml", core_props_xml)
        zf.writestr("docProps/app.xml", app_props_xml)
        zf.writestr("ppt/presentation.xml", presentation_xml)
        zf.writestr("ppt/_rels/presentation.xml.rels", presentation_rels_xml)
        zf.writestr("ppt/presProps.xml", pres_props_xml)
        zf.writestr("ppt/viewProps.xml", view_props_xml)
        zf.writestr("ppt/tableStyles.xml", table_styles_xml)
        zf.writestr("ppt/theme/theme1.xml", theme_xml)
        zf.writestr("ppt/slideMasters/slideMaster1.xml", slide_master_xml)
        zf.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", slide_master_rels_xml)
        zf.writestr("ppt/slideLayouts/slideLayout1.xml", slide_layout_xml)
        zf.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", slide_layout_rels_xml)
        zf.writestr("ppt/notesMasters/notesMaster1.xml", notes_master_xml)
        zf.writestr("ppt/notesMasters/_rels/notesMaster1.xml.rels", notes_master_rels_xml)

        for spec in SLIDES_SPEC:
            idx = int(spec["num"])
            zf.writestr(f"ppt/slides/slide{idx}.xml", _build_slide_xml(spec, total_slides=n_slides))
            slide_rel = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rIdLayout1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rIdNotes1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide" Target="../notesSlides/notesSlide{idx}.xml"/>
</Relationships>"""
            zf.writestr(f"ppt/slides/_rels/slide{idx}.xml.rels", slide_rel)

            zf.writestr(f"ppt/notesSlides/notesSlide{idx}.xml", _build_notes_slide_xml(spec))
            notes_rel = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rIdNotesMaster1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesMaster" Target="../notesMasters/notesMaster1.xml"/>
  <Relationship Id="rIdSlide1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="../slides/slide{idx}.xml"/>
</Relationships>"""
            zf.writestr(f"ppt/notesSlides/_rels/notesSlide{idx}.xml.rels", notes_rel)

    size_kb = output_path.stat().st_size / 1024.0
    print(f"[OK] Generated 32-slide standard template at: {output_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    generate_template_pptx(PROJECT_ROOT / "slides" / "template.pptx")
