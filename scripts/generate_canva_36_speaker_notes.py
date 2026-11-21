# -*- coding: utf-8 -*-
"""
Script sinh trọn bộ Speaker Notes chi tiết (310 - 460 từ/slide, nghiêm ngặt <= 500 từ/slide)
khớp 1-1 với đúng 36 trang Slide trên bản Canva chính thức của nhóm:
https://www.canva.com/design/DAHbeQ6df_I/vVfz2Am-T9QGhPY_9PvQvw/edit
"""

from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CANVA_36_SLIDES: List[Tuple[int, str, str, str, str]] = [
    # (slide_num, section, owner, title, speaker_note)
    (
        1,
        "I. RESEARCH FOUNDATION",
        "Khoa (TV1)",
        "Learning Rate Schedules in the Presence of Distribution Shift",
        (
            "Chào thầy và hội đồng chấm bài. Đây là báo cáo giữa kỳ môn Học Máy (Milestone 2 kết hợp Lab 1) của nhóm chúng em, do hai thành viên trực "
            "tiếp hoàn thành toàn bộ mã nguồn, thực nghiệm và biên soạn báo cáo là Khoa (Trưởng nhóm — TV1) và Khôi (TV2) (thành viên thứ ba là Hòa không "
            "hoàn thành nhiệm vụ được giao nên toàn bộ học phần Non-convex và các slide liên quan do Trưởng nhóm Khoa đảm nhận thay thế 100%). Công trình "
            "nghiên cứu mà nhóm trình bày và tái lập thực nghiệm là bài báo 'Learning Rate Schedules in the Presence of Distribution Shift' của bốn tác "
            "giả Matthew Fahrbach, Adel Javanmard, Vahab Mirrokni và Pratik Worah thuộc Google Research và Đại học Nam California (USC), công bố tại hội "
            "nghị ICML 2023 [1]. Bài báo giải quyết một bài toán nền tảng của các hệ thống học máy trực tuyến hoạt động liên tục (never-ending online "
            "learning): khi dữ liệu đầu vào đến theo luồng thời gian thực và phân phối xác suất sinh dữ liệu P_t thay đổi theo thời gian (distribution "
            "shift), bộ học sử dụng thuật toán Stochastic Gradient Descent (SGD) cần điều chỉnh tốc độ học eta_t như thế nào để duy trì sai số dự đoán "
            "thấp nhất trong suốt vòng đời vận hành? Trong học máy tĩnh truyền thống, nguyên lý kinh điển là giảm dần tốc độ học về 0 (như quy luật 1/t "
            "của Robbins-Monro) để triệt tiêu phương sai nhiễu mini-batch khi hội tụ về nghiệm cố định. Tuy nhiên, bài báo chứng minh cả về mặt lý thuyết "
            "lẫn thực nghiệm rằng: khi nghiệm tối ưu dịch chuyển theo thời gian, lịch tốc độ học tối ưu phải có khả năng TĂNG LÊN tương ứng với cường độ "
            "dịch chuyển để mở rộng bước khám phá và bám đuổi nghiệm mới, rồi mới giảm xuống khi môi trường ổn định. Báo cáo 36 trang này trình bày toàn "
            "diện cả cơ sở lý thuyết trên ba lớp hàm mất mát lẫn kết quả tái lập thực nghiệm, kiểm toán mã nguồn gốc và phân tích mở rộng của nhóm."
        ),
    ),
    (
        2,
        "I. RESEARCH FOUNDATION",
        "Khoa (TV1)",
        "Cấu trúc nghiên cứu",
        (
            "Slide 2 trình bày cấu trúc tổng thể của báo cáo gồm 36 trang, được tổ chức thành 5 phần mạch lạc nhằm tách bạch rõ ràng giữa lý thuyết "
            "của bài báo gốc và các đóng góp kiểm chứng thực nghiệm của nhóm. Phần I — Research Foundation (từ Slide 01 đến Slide 09, do Khoa và Khôi "
            "phụ trách) thiết lập bối cảnh học trực tuyến dưới dịch chuyển phân phối, phân tích các nhánh nghiên cứu tiền nhiệm để chỉ ra khoảng trống "
            "nghiên cứu, tổng hợp đóng góp của bài báo, định nghĩa hệ thống ký hiệu toán học chung, quy trình đánh giá trước cập nhật (test-then-train) "
            "và khung kiểm chứng tái lập bốn bước của dự án. Ba phần tiếp theo đi sâu vào từng lớp hàm mất mát theo cấu trúc thống nhất từ mô hình lý "
            "thuyết, thuật toán lập lịch cho đến kết quả tái lập thực nghiệm: Phần II — Linear Regression (Slides 10 đến 16, do Khoa phụ trách) trình "
            "bày xấp xỉ phương trình vi phân ngẫu nhiên SDE, hệ phương trình vi phân thường cho moment bậc hai, bài toán điều khiển tối ưu dẫn xuất "
            "Algorithm 1 và kết quả tái lập Figure 4 cùng Figure 5-top. Phần III — General Convex Losses (Slides 17 đến 26, do Khôi phụ trách) trình bày "
            "cận trên và cận dưới của Dynamic Regret, quy tắc hai ngưỡng Proposition 4.3, thiết kế benchmark Logistic Regression, hiệu chỉnh nghiệm tối "
            "ưu quần thể bằng tích phân cầu phương, kiểm chứng khớp từng bước với notebook tác giả, cùng các phân tích độ nhạy và can thiệp có đối chứng "
            "(controlled ablations). Phần IV — Non-convex Losses (Slides 27 đến 31, do Khoa thực hiện và trình bày) giải thích tiêu chí chuẩn gradient, "
            "Theorem 5.1 và thực nghiệm mạng nơ-ron MLP 5 tầng với cơ chế Adam Restart trên dữ liệu tế bào Flow Cytometry (Figure 6). Cuối cùng, Phần V "
            "(Slides 32 đến 36) tổng hợp so sánh lý thuyết, độ phức tạp tính toán, mức độ tái lập và hướng mở rộng."
        ),
    ),
    (
        3,
        "I. RESEARCH FOUNDATION",
        "Khoa (TV1)",
        "Học trực tuyến dưới dịch chuyển phân phối",
        (
            "Slide 3 đặt ra động lực cốt lõi của bài toán thông qua sự đối lập giữa môi trường phân phối tĩnh và môi trường có dịch chuyển phân phối, "
            "được minh họa trực quan bằng Hình 1 ở bên phải — kết quả nhóm tái lập hoàn chỉnh Figure 1 của bài báo (`figures/fig1_constant_lr_trajectories.png`). "
            "Trong môi trường tĩnh (stationary setting), phân phối dữ liệu P không đổi nên nghiệm tối ưu toàn cục theta* là một điểm cố định; khi đó, "
            "việc giảm dần tốc độ học eta_t về 0 giúp kìm hãm dao động ngẫu nhiên do nhiễu gradient trên mini-batch và đảm bảo mô hình hội tụ sát về "
            "theta*. Ngược lại, trong các hệ thống học trực tuyến thực tế, phân phối dữ liệu P_t và nghiệm tối ưu theta*_t dịch chuyển liên tục theo "
            "thời gian. Một ví dụ công nghiệp điển hình là hệ thống dự đoán tỷ lệ nhấp chuột (Click-Through Rate — CTR) trên sàn thương mại điện tử: "
            "hành vi người dùng vừa trôi dạt mượt theo khung giờ sáng, trưa, tối, vừa nhảy vọt đột ngột khi một chiến dịch Flash Sale kích hoạt lúc "
            "00:00 làm thay đổi hoàn toàn phân phối xác suất mua hàng. Hình 1 mô phỏng rõ sự đánh đổi sinh tử khi dùng tốc độ học hằng số trên bài toán "
            "hồi quy tuyến tính trực tuyến 2 chiều có nghiệm tối ưu chạy theo đường xoắn ốc (màu xanh dương) với kích thước batch B_t = 1 trong T = 2000 "
            "bước: Nếu chọn tốc độ học quá nhỏ như eta = 0.003 (ô trên bên trái) hoặc eta = 0.01 (ô trên bên phải), quỹ đạo học được của SGD (màu cam) "
            "rất mượt nhưng bị trễ pha nghiêm trọng, không kịp xoay theo vòng xoắn và chịu tổng Regret rất cao (755 và 267). Ngược lại, khi nâng tốc độ "
            "học lên eta = 0.03 (Regret giảm xuống 71) và eta = 0.1 (ô dưới bên phải, Regret thấp nhất chỉ còn 31), mô hình chấp nhận dao động ngẫu nhiên "
            "quanh đường xoắn ốc nhưng bám sát chuyển động của nghiệm tối ưu. Bài toán đặt ra là làm sao tự động điều chỉnh eta_t để cân bằng giữa khả "
            "năng bám đuổi nghiệm đang trôi và việc triệt tiêu nhiễu cập nhật SGD."
        ),
    ),
    (
        4,
        "I. RESEARCH FOUNDATION",
        "Khoa (TV1)",
        "Nghiên cứu tiền nhiệm và khoảng trống",
        (
            "Để làm rõ vị trí học thuật của bài báo ICML 2023, Slide 4 hệ thống hóa ba nhánh nghiên cứu tiền nhiệm trực tiếp và chỉ rõ giới hạn của "
            "từng nhánh trong thiết lập học trực tuyến không dừng. Nhánh thứ nhất là các lịch trình suy giảm tốc độ học cổ điển (Classical Decay, tiêu "
            "biểu là Robbins & Monro, 1951 [2] và các lịch trình Step Decay / Cosine Annealing trong học sâu tĩnh). Nguyên lý của nhánh này là thiết kế "
            "chuỗi tốc độ học giảm đơn điệu về 0 (như eta_t = c / t) thỏa mãn tổng eta_t phân kỳ nhưng tổng eta_t^2 hội tụ để triệt tiêu phương sai "
            "nhiễu khi t tiến ra vô cùng. Tuy nhiên, giới hạn chí mạng của chúng trong môi trường học liên tục là hiện tượng quán tính đóng băng: khi "
            "dịch chuyển phân phối xuất hiện muộn ở thời điểm t lớn, tốc độ học eta_t đã suy giảm về sát 0 nên mô hình gần như mất khả năng cập nhật "
            "trọng số để thích nghi với phân phối mới. Nhánh thứ hai là Tối ưu Lồi Trực tuyến Động (Dynamic Online Convex Optimization — Dynamic OCO, "
            "tiêu biểu là Besbes et al., 2015 [3]). Nhánh này đã đưa ra khái niệm Dynamic Regret và chứng minh cận tối ưu minimax theo tổng ngân sách "
            "biến thiên V_T thông qua cơ chế khởi động lại định kỳ (periodic restart). Dù vậy, các thiết kế này thường là chính sách tiền định cho trường "
            "hợp xấu nhất, đòi hỏi phải biết trước chân trời thời gian T hoặc ngân sách biến thiên V_T để chia chu kỳ cố định, thay vì điều chỉnh eta_t "
            "mượt mà theo độ trôi thực tế ở từng bước. Nhánh thứ ba là xấp xỉ SGD bằng phương trình vi phân ngẫu nhiên thời gian liên tục (Continuous-time "
            "SDE, tiêu biểu là Li et al., 2017 [4]). Công trình của Li et al. đã xây dựng nền tảng giải tích SDE cho SGD nhưng chỉ xét hàm mất mát tĩnh "
            "không có vận tốc trôi của nghiệm tối ưu. Như điểm nhấn cuối slide nêu rõ: khái niệm Dynamic Regret không phải mới, nhưng khoảng trống "
            "nghiên cứu nằm ở việc dẫn xuất công thức đóng chọn tốc độ học thích ứng theo độ trôi tức thời dưới các giả thiết động học SGD cụ thể."
        ),
    ),
    (
        5,
        "I. RESEARCH FOUNDATION",
        "Khôi (TV2)",
        "Đóng góp và tính mới của bài báo",
        (
            "Slide 5 tổng hợp ba đóng góp lý thuyết và thuật toán cốt lõi của bài báo Fahrbach et al. (2023) tương ứng với ba lớp hàm mất mát, đồng "
            "thời phân định minh bạch giữa tính mới của bài báo gốc và phần đóng góp thực nghiệm của nhóm dự án. Ở cột thứ nhất — Hồi quy Tuyến tính "
            "(Linear Regression, Mục 3), đóng góp mới của bài báo là mở rộng phương trình vi phân ngẫu nhiên SDE sang bối cảnh nghiệm tối ưu theta*_t "
            "di chuyển theo quỹ đạo thời gian liên tục Y(tau). Bằng cách kết hợp Bổ đề Stein và công thức Itô, tác giả rút gọn quá trình sai số ngẫu "
            "nhiên d chiều về hệ phương trình vi phân thường (Moment ODE) chính xác cho kỳ vọng và phương sai sai số, sau đó áp dụng lý thuyết điều "
            "khiển tối ưu Hamilton-Jacobi-Bellman (HJB) trên quá trình bao trên vô hướng để tìm ra nghiệm điều khiển đóng và rời rạc hóa thành "
            "Algorithm 1. Ở cột thứ hai — Hàm Lồi Tổng quát (General Convex, Mục 4), khi không còn cấu trúc bậc hai để đóng kín hệ ODE, bài báo thiết "
            "lập cận trên giải tích cho kỳ vọng Dynamic Regret của Projected Mini-batch SGD tại Theorem 4.2, từ đó giải bài toán tối ưu lồi có điểm gãy "
            "từng bước để dẫn xuất lịch trình tốc độ học thích ứng kẹp giữa hai ngưỡng đóng [tau_{1,t}, tau_{2,t}] tại Proposition 4.3, kèm theo cận dưới "
            "đối kháng cho hàm lồi mạnh tại Theorem 4.7. Ở cột thứ ba — Hàm Không lồi (Non-convex, Mục 5 & Phụ lục E), bài báo chuyển tiêu chí đánh giá "
            "sang tổng bình phương chuẩn gradient, thiết lập cận trên và lịch tốc độ học đóng tại Theorem 5.1, đồng thời minh họa hiệu quả của cơ chế "
            "Adam Restart trên mạng nơ-ron phân loại tế bào dòng chảy. Đặc biệt, như ghi chú ở chân slide nhấn mạnh: nhóm chúng em tách bạch nghiêm "
            "ngặt tính mới của bài báo với phần đóng góp kỹ thuật của dự án (bao gồm việc tái lập thực nghiệm độc lập, kiểm toán mã nguồn gốc, hiệu chỉnh "
            "bộ so sánh nghiệm tối ưu quần thể bằng tích phân cầu phương, kiểm chứng khớp từng bước shared-input parity và nghiên cứu độ nhạy/ablation)."
        ),
    ),
    (
        6,
        "I. RESEARCH FOUNDATION",
        "Khoa (TV1)",
        "Phát biểu bài toán và ký hiệu chung",
        (
            "Slide 6 thiết lập hệ thống ký hiệu toán học thống nhất và bốn phương trình nền tảng (Công thức 1 đến 4 trong Mục 2 của bài báo) được sử "
            "dụng xuyên suốt từ trang này đến cuối báo cáo. Ở cột bên trái, ta xét quá trình học trực tuyến diễn ra trên chân trời thời gian rời rạc "
            "gồm T bước: t = 1, 2, ..., T. Tại mỗi bước thời gian t, môi trường được đặc trưng bởi một phân phối xác suất P_t trên không gian mẫu và "
            "sinh ra một mini-batch gồm B_t mẫu độc lập cùng phân phối z_{t,1}, ..., z_{t,B_t} ~ P_t. Vector theta_t in R^p biểu diễn trọng số của mô "
            "hình học được ngay trước khi thực hiện cập nhật tại bước t, còn eta_t >= 0 là tốc độ học (learning rate) được bộ điều khiển lựa chọn tại "
            "bước đó. Ở cột bên phải, Công thức (1) định nghĩa hàm mất mát kỳ vọng trên toàn quần thể (expected population loss) tại bước t là "
            "l_bar_t(theta) = E_{z ~ P_t}[l(theta, z)], và gọi nghiệm tối ưu quần thể (population oracle) tại thời điểm đó là theta*_t = argmin_{theta "
            "in R^p} l_bar_t(theta). Khác với Static Regret chỉ so sánh với một điểm cố định duy nhất trên toàn bộ T bước, Công thức (2) đánh giá chất "
            "lượng của chuỗi mô hình (theta_1, ..., theta_T) thông qua độ tiếc nuối động tích lũy (Cumulative Dynamic Regret): Reg(T) = sum_{t=1}^T "
            "[l_bar_t(theta_t) - l_bar_t(theta*_t)], tức là tổng rủi ro dư (excess population risk) của mô hình hiện tại so với nghiệm tối ưu tức thời "
            "theta*_t tại từng bước. Để cập nhật mô hình mà không biết phân phối thực P_t, bộ học tính đạo hàm trung bình mẫu trên mini-batch hiện tại "
            "theo Công thức (4): grad_l^B_t(theta_t) = (1 / B_t) sum_{k=1}^{B_t} grad_l(theta_t, z_{t,k}), rồi thực hiện bước cập nhật Projected "
            "Mini-batch SGD theo Công thức (3): theta_{t+1} = Pi_Theta(theta_t - eta_t * grad_l^B_t(theta_t)), trong đó Pi_Theta là phép chiếu Euclid "
            "lên tập lồi đóng Theta (hoặc không chiếu nếu Theta = R^p). Ghi chú cuối trang lưu ý rõ rằng riêng ở Phần IV cho hàm không lồi, vì không thể "
            "bảo đảm tìm được cực tiểu toàn cục theta*_t, bài báo sẽ thay Công thức (2) bằng tổng bình phương chuẩn gradient."
        ),
    ),
    (
        7,
        "I. RESEARCH FOUNDATION",
        "Khoa (TV1)",
        "Quy trình học trực tuyến",
        (
            "Slide 7 làm rõ giao thức thông tin nhân quả của trò chơi học trực tuyến giữa Tự nhiên (Nature) và Bộ học (Learner), đồng thời trình bày "
            "sơ đồ kiến trúc Pipeline hai giai đoạn Inference và Training của dự án ở hình bên phải. Một nguyên tắc cực kỳ quan trọng trong học trực "
            "tuyến là thứ tự thời gian tại mỗi bước t: Ở bước 01 (Lịch sử đến t-1), trọng số hiện tại theta_t chỉ được phép phụ thuộc vào lịch sử dữ "
            "liệu đã quan sát được từ các bước trước z_{[t-1]} = (z_1, ..., z_{t-1}); dựa trên lịch sử đó, Tự nhiên xác định phân phối P_t cho bước "
            "hiện tại, và tốc độ học eta_t được chọn dựa trên thông tin hợp lệ theo mô hình giả định. Sang bước 02 (Đánh giá trước cập nhật — "
            "Test-then-Train), hệ thống bắt buộc phải sử dụng mô hình hiện tại theta_t để đưa ra dự đoán và chịu hàm mất mát / độ tiếc nuối l_bar_t(theta_t) "
            "- l_bar_t(theta*_t) trên phân phối bước t TRƯỚC KHI được dùng mini-batch z_t để tính gradient và cập nhật lên trọng số theta_{t+1}. Sơ đồ "
            "kiến trúc bên phải cụ thể hóa quy trình này thành hai giai đoạn tách biệt trong mã nguồn của nhóm: Giai đoạn A (Online Inference — khung "
            "màu xanh lá ở nửa trên) nhận luồng mẫu đầu vào tại bước t, chạy suy luận bằng trọng số theta_t đang phục vụ để tính rủi ro dư và cộng dồn "
            "vào Cumulative Regret Reg(T) trước khi cập nhật. Giai đoạn B (Adaptive SGD Training — khung màu xanh dương và cam ở nửa dưới) nhận nhãn "
            "phản hồi của mini-batch B_t mẫu, tính gradient trung bình mẫu, đồng thời đưa tín hiệu độ trôi (từ ước lượng chuyển động mô hình EMA hoặc "
            "tín hiệu tham chiếu của benchmark) vào một trong ba bộ điều khiển tốc độ học (`LinearOptimalSchedule`, `ConvexSchedule`, hoặc "
            "`NonConvexSchedule` / Adam Restart) để xuất ra eta*_t và cập nhật trọng số lên theta_{t+1}. Như ghi chú cuối slide nhấn mạnh: trong các "
            "bài toán benchmark tổng hợp, dữ liệu có thể cung cấp sẵn quỹ đạo tham chiếu tương lai để kiểm chứng lý thuyết, nhưng trong triển khai "
            "online thực tế ta tuyệt đối không được coi thông tin tương lai này là có sẵn."
        ),
    ),
    (
        8,
        "I. RESEARCH FOUNDATION",
        "Khoa (TV1)",
        "Phạm vi thực nghiệm và câu hỏi nghiên cứu",
        (
            "Slide 8 xác định rõ phạm vi thực nghiệm toàn diện (Full Scope) của dự án cùng ba câu hỏi nghiên cứu trọng tâm tương ứng với từng bài toán "
            "trong bài báo gốc. Nhóm chúng em thực hiện tái lập đầy đủ cả ba nhánh thực nghiệm chính của bài báo và mở rộng thêm các phân tích chuyên "
            "sâu. Ở nhánh thứ nhất — Linear Regression (tái lập Figure 4 và Figure 5-top tại Mục 6.1, trang 9), câu hỏi nghiên cứu là: Quỹ đạo của bộ "
            "học SGD, động thái dao động của tốc độ học eta_t sinh bởi Algorithm 1 và tổng Cumulative Regret thay đổi như thế nào khi tần suất dịch "
            "chuyển phân phối thay đổi từ liên tục sang nhảy bậc (ell in {1, 4, 8, 16}) và khi số chiều không gian tăng từ thấp lên cao (d in {2, 8, 32, "
            "128, 512}), cũng như khi thay độ trôi Oracle bằng ước lượng trực tuyến EMA? Ở nhánh thứ hai — General Convex Losses (tái lập Figure 5-bottom "
            "trên bài toán hồi quy Logistic tại Mục 6.1), câu hỏi nghiên cứu không chỉ dừng ở việc kiểm chứng lịch trình hai ngưỡng Proposition 4.3, mà "
            "còn đi sâu vào việc kiểm tra tính tương thích với notebook gốc của tác giả, hiệu chỉnh bộ so sánh nghiệm tối ưu quần thể (population "
            "minimizer comparator), khảo sát giới hạn hiệu năng tại số chiều lớn d = 512, và thực hiện các phân tích độ nhạy (sensitivity analysis) cùng "
            "can thiệp có đối chứng (controlled ablations) để bóc tách tác động của từng giả thiết triển khai. Ở nhánh thứ ba — Non-convex Losses (tái "
            "lập Figure 6 tại Phụ lục E, trang 22–23), câu hỏi nghiên cứu là: Cơ chế khởi động lại bộ tối ưu Adam (Adam Restart) tác động ra sao đến "
            "tổng mất mát kiểm định tích lũy của mạng nơ-ron MLP 5 tầng khi phân loại luồng dữ liệu tế bào tuyến tụy Flow Cytometry qua 4 pha dịch "
            "chuyển phân phối? Ghi chú cuối slide nêu rõ nguyên tắc khoa học của nhóm: phạm vi thí nghiệm trong bài báo không mặc nhiên đồng nhất với "
            "mức độ tái lập đã xác minh, mà mọi tuyên bố đều phải đi kèm bằng chứng đối chiếu cụ thể."
        ),
    ),
    (
        9,
        "I. RESEARCH FOUNDATION",
        "Khôi (TV2)",
        "Khung tái lập và kiểm chứng",
        (
            "Khép lại Phần I, Slide 9 trình bày quy trình 4 bước tái lập và kiểm chứng nghiêm ngặt mà nhóm thiết lập để đảm bảo tính minh bạch và khả "
            "năng kiểm tra độc lập của toàn bộ kết quả trong báo cáo. Bước 01 là Kiểm toán mã nguồn và Khóa cấu hình (Source Audit & Configuration): "
            "trước khi chạy thực nghiệm, nhóm đối chiếu từng dòng giữa mô tả toán học trong bài báo PDF với mã nguồn thực tế trong notebook và repository "
            "của tác giả (`fahrbach/learning-rate-schedules`), đồng thời ghim cứng phiên bản thư viện trong môi trường CPython 3.12 (`requirements.txt`), "
            "khóa tệp cấu hình YAML, danh sách hạt giống ngẫu nhiên `seeds = [0, 1, 2]` và nguồn dữ liệu đầu vào. Bước 02 là Thực thi thực nghiệm và "
            "Đánh giá có đối chứng (Execution & Matched Evaluation): các trình chạy thực nghiệm (runners) ghi lại đầy đủ quỹ đạo trọng số, lịch sử tốc "
            "độ học, các thước đo mất mát và metadata; đặc biệt, khi so sánh giữa các phương pháp lập lịch khác nhau trên cùng một số chiều và seed, "
            "hệ thống ghép cặp chính xác cùng một luồng ma trận đặc trưng X và nhãn y (shared-input / paired design) để loại bỏ hoàn toàn sai số ngẫu "
            "nhiên do lấy mẫu. Bước 03 là Theo dõi và Kiểm chứng Artifact (W&B Tracking & Verification): toàn bộ các lần chạy chính thức được đẩy lên "
            "dự án Weights & Biases công khai tại `forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift` kèm Run ID, cấu hình, chuỗi thời gian "
            "scalar, mã băm SHA256 của dữ liệu và nguồn commit Git. Bước 04 là Phân loại rõ mức độ tái lập (Reproducibility Tiers): nhóm phân biệt rạch "
            "ròi 4 cấp độ bằng chứng gồm: (i) kiểm toán mã nguồn gốc (source audit), (ii) kiểm chứng khớp từng bước trên cùng dữ liệu đầu vào "
            "(shared-input implementation parity), (iii) tái lập độc lập đa hạt giống (independent multi-seed reproduction), và (iv) xác thực artifact "
            "trên W&B. Cách tiếp cận này giúp báo cáo không bao giờ trộn lẫn kết quả giữa các giao thức đánh giá khác nhau."
        ),
    ),
    (
        10,
        "II. LINEAR REGRESSION",
        "Khoa (TV1)",
        "Hồi quy tuyến tính với nghiệm tối ưu biến thiên",
        (
            "Bước sang Phần II — Linear Regression (từ Slide 10 đến Slide 16, tài liệu chi tiết tại `docs/03_linear.md`), Slide 10 giới thiệu thiết "
            "lập toán học của bài toán Hồi quy Tuyến tính trực tuyến với nghiệm tối ưu biến thiên theo thời gian tại Mục 3 của bài báo. Về mô hình sinh "
            "dữ liệu (Công thức 7), tại mỗi bước thời gian t = 1, ..., T, bộ học nhận một mini-batch gồm B_t mẫu độc lập (x_{t,k}, y_{t,k}). Trong đó, "
            "vector đặc trưng đầu vào x_{t,k} in R^d được lấy mẫu từ phân phối chuẩn tắc đa chiều đẳng hướng cố định N(0, I_d), còn nhãn mục tiêu liên "
            "tục y_{t,k} in R được sinh bởi mô hình tuyến tính phụ thuộc thời gian: y_{t,k} = x_{t,k}^T theta*_t + eps_{t,k}, với nhiễu quan sát độc lập "
            "eps_{t,k} ~ N(0, sigma^2) có phương sai hằng số sigma^2. Về hàm mất mát, ta xét hàm bình phương tối thiểu (squared error loss) cho từng mẫu: "
            "l(theta; (x, y)) = 1/2 (y - x^T theta)^2. Nhờ tính chất đẳng hướng của đặc trưng E[x x^T] = I_d và tính độc lập của nhiễu nhãn E[eps] = 0, "
            "khi lấy kỳ vọng trên phân phối P_t, hàm mất mát quần thể có dạng giải tích rất gọn: l_bar_t(theta) = 1/2 ||theta - theta*_t||_2^2 + "
            "1/2 sigma^2. Tại nghiệm tối ưu quần thể theta = theta*_t, mất mát tối thiểu đạt đúng mức sàn phương sai nhiễu bất khả quy l_bar_t(theta*_t) "
            "= 1/2 sigma^2. Do đó, rủi ro dư quần thể (population excess loss — tức độ tiếc nuối tức thời reg_t tại bước t) rút gọn chính xác thành một "
            "nửa bình phương khoảng cách Euclid từ trọng số hiện tại của bộ học tới nghiệm tối ưu Oracle: l_bar_t(theta_t) - l_bar_t(theta*_t) = "
            "1/2 ||theta_t - theta*_t||_2^2. Như vậy, mục tiêu cực tiểu hóa tổng Dynamic Regret kỳ vọng trên toàn chân trời T hoàn toàn tương đương với "
            "việc điều khiển tốc độ học eta_t để giữ cho bình phương khoảng cách sai số theo dõi E[||theta_t - theta*_t||_2^2] nhỏ nhất có thể, đồng thời "
            "kiểm soát được sự khuếch đại nhiễu từ mini-batch."
        ),
    ),
    (
        11,
        "II. LINEAR REGRESSION",
        "Khoa (TV1)",
        "Xấp xỉ SGD bằng phương trình SDE",
        (
            "Slide 11 trình bày bước đột phá giải tích đầu tiên của Mục 3: xấp xỉ quỹ đạo sai số rời rạc theta_t - theta*_t của thuật toán Mini-batch "
            "SGD bằng một Phương trình Vi phân Ngẫu nhiên (Stochastic Differential Equation — SDE) trong miền thời gian liên tục (Công thức 8 và "
            "Proposition 3.1). Tại sao chúng ta cần phép xấp xỉ SDE này? Trong miền thời gian rời rạc, bước cập nhật SGD cho vector sai số có dạng "
            "theta_{t+1} - theta*_t = (I_d - (eta_t / B_t) X_t^T X_t)(theta_t - theta*_t) + (eta_t / B_t) X_t^T eps_t - (theta*_{t+1} - theta*_t). Việc "
            "phân tích trực tiếp chuỗi rời rạc này qua nhiều bước đòi hỏi phải tính kỳ vọng của tích các ma trận Wishart ngẫu nhiên phụ thuộc thời gian, "
            "khiến bài toán tối ưu chuỗi tốc độ học không có nghiệm đóng. Để chuyển sang giới hạn liên tục, tác giả chọn một tham số bước nhảy tối đa "
            "đủ nhỏ eps > 0 (đóng vai trò là trần tốc độ học eta_max) và định nghĩa thời gian liên tục tau = eps * t. Theo phép co giãn (scaling) này, "
            "tốc độ học rời rạc được biểu diễn qua hàm điều khiển không thứ nguyên zeta(tau) in [0, 1] dưới dạng eta_t = eps * zeta(eps * t), kích thước "
            "mini-batch được biểu diễn qua hàm mật độ batch nu(tau) dưới dạng B_t = ceil(eps * nu(eps * t)), và chuỗi nghiệm tối ưu rời rạc được nhúng "
            "vào đường cong khả vi trơn Y(tau) sao cho Y(eps * t) = theta*_t. Khi áp dụng Bổ đề Stein (Lemma D.1) để tính chính xác ma trận hiệp phương "
            "sai bậc bốn của gradient mẫu Gaussian, quá trình sai số liên tục X(tau) ≈ theta_t - theta*_t thỏa mãn phương trình SDE (8) trên slide. "
            "Phương trình này bóc tách động học sai số thành 3 lực vật lý rõ rệt: (1) Lực kéo hội tụ -zeta(tau)X(tau)d_tau của gradient kéo sai số về 0; "
            "(2) Lực đẩy dịch chuyển -Y'(tau)d_tau phản ánh vận tốc chạy trốn của nghiệm tối ưu; và (3) Số hạng khuếch tán (Nhiễu) tỷ lệ với "
            "zeta(tau)/sqrt(nu(tau)) nhân với căn bậc hai của ma trận hiệp phương sai [ (||X||^2 + sigma^2)I_d + X X^T ] đi cùng chuyển động Brown "
            "dW(tau). Proposition 3.1 bảo chứng rằng dưới điều kiện Lipschitz của quỹ đạo và hàm điều khiển, sai số xấp xỉ điểm-đối-điểm giữa SDE và "
            "SGD rời rạc bị chặn ở bậc O(sqrt(eps))."
        ),
    ),
    (
        12,
        "II. LINEAR REGRESSION",
        "Khoa (TV1)",
        "Moment ODE và điều khiển tối ưu",
        (
            "Từ phương trình SDE (8) trên không gian d chiều ở Slide 11, Slide 12 trình bày Theorem 3.2 và Theorem 3.3: quá trình rút gọn hệ ngẫu nhiên "
            "cao chiều về một bài toán điều khiển tối ưu tất định trên đúng một biến vô hướng duy nhất. Đầu tiên, tại Theorem 3.2, tác giả xét hai đại "
            "lượng thống kê của quá trình sai số X(tau): vector kỳ vọng sai số (moment bậc nhất) m(tau) = E[X(tau)] in R^d, và bình phương khoảng cách "
            "sai số kỳ vọng (moment bậc hai) v(tau) = E[||X(tau)||_2^2] in R_{>=0} — đại lượng tỷ lệ trực tiếp với độ tiếc nuối tức thời E[reg_t] ≈ "
            "1/2 v(eps*t). Khi lấy kỳ vọng hai vế của SDE (8), số hạng tích phân Itô có kỳ vọng bằng 0 nên triệt tiêu, cho ta phương trình vi phân "
            "thường (ODE) bậc nhất cho m(tau): m' = -zeta * m - Y'. Tiếp theo, áp dụng công thức vi phân ngẫu nhiên Itô (Lemma D.2) cho hàm bình phương "
            "chuẩn phi(X) = ||X||_2^2 (với đạo hàm bậc hai là 2 I_d) và lấy vết của ma trận khuếch tán Tr[(||X||^2 + sigma^2)I_d + X X^T] = (d + 1)||X||^2 "
            "+ d * sigma^2, ta thu được phương trình ODE chính xác cho moment bậc hai: v' = [(d + 1)zeta^2 / nu - 2*zeta]v + (zeta^2 * sigma^2 * d) / nu "
            "- 2 m^T Y'. Tuy nhiên, vế phải của phương trình v' vẫn còn chứa số hạng tương tác tích vô hướng -2 m^T Y' phụ thuộc vào góc hướng giữa "
            "vector sai số d chiều m và vector vận tốc dịch chuyển Y'. Để khử sự phụ thuộc vào hướng d chiều này, tác giả áp dụng bất đẳng thức "
            "Cauchy-Schwarz kết hợp bất đẳng thức Jensen: |-2 m^T Y'| <= 2 ||Y'||_2 ||m||_2 <= 2 ||Y'||_2 sqrt(v), từ đó xây dựng quá trình bao trên vô "
            "hướng (scalar upper-envelope) v_tilde(tau) >= v(tau). Tại Theorem 3.3, bài toán cực tiểu hóa tích phân int_0^T v_tilde(tau) d_tau được giải "
            "bằng phương trình đạo hàm riêng Hamilton-Jacobi-Bellman (HJB). Vì đạo hàm hàm giá trị theo trạng thái sai số luôn dương, chính sách điều "
            "khiển tối ưu tại mỗi thời điểm chính là nghiệm cực tiểu hóa tam thức bậc hai theo zeta trong phương trình bao trên, cho ra công thức đóng: "
            "zeta* = min{ 1, (nu * v_tilde) / [(d + 1)*v_tilde + sigma^2 * d] }. Đặc biệt, Lemma 3.6 chỉ ra rằng khi không có dịch chuyển (Y' = 0), "
            "nghiệm ODE này giữ tốc độ học ở mức trần trong giai đoạn đầu rồi tự động suy giảm tiệm cận theo đúng quy luật cổ điển O(1/t)."
        ),
    ),
    (
        13,
        "II. LINEAR REGRESSION",
        "Khoa (TV1)",
        "Algorithm 1: lịch học thích ứng",
        (
            "Slide 13 trình bày Algorithm 1 — thuật toán rời rạc hóa nghiệm điều khiển tối ưu liên tục của Theorem 3.3 thành bộ lập lịch tốc độ học "
            "thời gian thực cho mỗi bước cập nhật SGD (được nhóm cài đặt tại lớp `LinearOptimalSchedule` trong `core/schedules.py`). Đầu vào của thuật "
            "toán tại bước thời gian t gồm: trần tốc độ học tối đa eps = eta_max, kích thước bước chia tích phân số kappa in (0, 1], số chiều đặc trưng d, "
            "kích thước mini-batch B_t, phương sai nhiễu nhãn sigma^2, tín hiệu độ trôi của nghiệm tối ưu ở bước vừa qua gamma_{t-1} = ||theta*_t - "
            "theta*_{t-1}||_2 (hoặc bản ước lượng trực tuyến EMA gamma_hat_{t-1}), cùng biến trạng thái vô hướng v lưu giá trị bao trên của bình phương "
            "khoảng cách sai số hiện tại. Sử dụng phép đổi biến ngược từ miền liên tục về miền rời rạc (r = eps * zeta, B_t = eps * nu và ||Y'||_2 ≈ "
            "gamma_{t-1} / eps), Algorithm 1 tích phân số phương trình ODE bao trên từ thời điểm t-1 sang thời điểm t bằng phương pháp Euler thuận "
            "(forward Euler) qua K = ceil(1 / kappa) bước lặp con. Trong mỗi bước lặp con, thuật toán tính tốc độ học ứng viên tức thời r = min{ eps, "
            "(v * B_t) / [(d + 1)*v + sigma^2 * d] }, sau đó cập nhật biến trạng thái v theo phương trình sai phân trên slide. Phương trình cập nhật v "
            "này cho thấy rõ cơ chế tự điều chỉnh vật lý của thuật toán qua ba số hạng: Số hạng thứ nhất [(d+1)r^2 / B_t - 2r]*v luôn âm khi r nằm trong "
            "miền tối ưu, đại diện cho tốc độ co giảm sai số nhờ bước học SGD; số hạng thứ hai (sigma^2 * d * r^2) / B_t đại diện cho lượng phương sai "
            "bơm thêm do nhiễu nhãn; và số hạng thứ ba +2 * gamma_{t-1} * sqrt(v) đại diện cho lượng sai số bơm vào do nghiệm tối ưu dịch chuyển một "
            "đoạn gamma_{t-1}. Vì hàm tốc độ học r(v) có đạo hàm theo v là (B_t * sigma^2 * d) / [(d+1)v + sigma^2*d]^2 > 0 luôn dương, nên mỗi khi xuất "
            "hiện dịch chuyển phân phối (gamma_{t-1} > 0) làm biến trạng thái v tăng lên, tốc độ học tối ưu eta*_t lập tức được đẩy tăng lên để mô hình "
            "bước dài hơn nhằm bám đuổi nghiệm mới. Toàn bộ vòng lặp K bước con này chỉ tính toán trên một biến vô hướng v nên chi phí thời gian riêng "
            "của bộ lập lịch là Theta(K) = O(1), hoàn toàn không phụ thuộc vào số chiều d."
        ),
    ),
    (
        14,
        "II. LINEAR REGRESSION",
        "Khoa (TV1)",
        "Thiết lập thực nghiệm Linear",
        (
            "Slide 14 tổng hợp toàn bộ cấu hình tham số chuẩn cho hai thực nghiệm Hồi quy Tuyến tính tại Mục 6.1 của bài báo: Figure 4 và Figure 5-top "
            "(được định nghĩa trong `configs/linear.yaml` và thực thi bởi `experiments/linear_regression.py`). Về thiết lập chung cho cả hai thí nghiệm, "
            "quá trình học trực tuyến diễn ra trong T = 2000 bước thời gian với kích thước mini-batch cố định B_t = 256, phương sai nhiễu nhãn Gaussian "
            "sigma^2 = 0.1, bước chia tích phân Euler trong Algorithm 1 là kappa = 0.1 (tức K = 10 bước con mỗi bước SGD), và mỗi cấu hình đều được chạy "
            "lặp lại độc lập trên 3 hạt giống ngẫu nhiên `seeds = [0, 1, 2]`. Quỹ đạo nghiệm tối ưu gốc theta*_t trên không gian R^d được sinh bởi hàm "
            "xoắn ốc giải tích tại Công thức (26) với tham số biên độ đầu a = 1, biên độ cuối b = -1 và hệ số tần số góc k = 4: cụ thể, bán kính r(t) "
            "giảm tuyến tính từ 1 xuống -1, còn độ lớn vector nghiệm tỷ lệ với hàm lập phương r(t)^3, khiến nghiệm xuất phát từ biên ngoài ở t = 1, xoắn "
            "chậm dần vào gốc tọa độ 0 quanh thời điểm giữa t = 1000, rồi xoắn tăng tốc ngược ra biên ở t = 2000. Để mô phỏng các chế độ dịch chuyển "
            "phân phối có tần suất nhảy khác nhau, quỹ đạo liên tục được lấy mẫu giữ bậc (sample-and-hold) mỗi ell bước một lần: theta_hat*_t = "
            "theta*_{ceil(t / ell) * ell}. Trên nền tảng đó, thí nghiệm Figure 4 cố định không gian 2 chiều (d = 2), đặt trần tốc độ học eta_max = 0.5 "
            "và khảo sát ba mức bước lấy mẫu ell in {1, 4, 16} để quan sát trực tiếp quỹ đạo bám đuổi 2D và hình dáng dao động của tốc độ học eta_t. "
            "Thí nghiệm Figure 5-top cố định bước lấy mẫu ell = 8 và mở rộng số chiều theo cấp số nhân d in {2, 8, 32, 128, 512} với trần tốc độ học co "
            "giãn theo số chiều eta_max = 1 / sqrt(d), đồng thời đối chiếu toàn diện giữa Algorithm 1 dùng độ trôi Oracle, Algorithm 1 dùng ước lượng "
            "trực tuyến EMA (beta = 0.9), và các lịch trình baseline gồm Constant, Inverse-sqrt (1 / sqrt(t)) và Inverse-time (1 / t)."
        ),
    ),
    (
        15,
        "II. LINEAR REGRESSION",
        "Khoa (TV1)",
        "Figure 4: quỹ đạo và động thái learning rate",
        (
            "Slide 15 trình bày kết quả tái lập hoàn chỉnh Figure 4 của bài báo (hình `figures/fig4_linear_trajectory_and_lr.png` ở bên phải và bảng "
            "tổng hợp số liệu từ `results/linear/fig4_summary.csv` ở bên trái, tương ứng với 3 runs xác thực trên W&B là `yaomsd9p`, `id9my7jj` và "
            "`zrhn9935`). Hình bên phải gồm 6 đồ thị xếp thành 2 hàng và 3 cột ứng với ba tần suất dịch chuyển ell = 1 (cột trái), ell = 4 (cột giữa) "
            "và ell = 16 (cột phải) tại số chiều d = 2, T = 2000, B = 256, eta_max = 0.5. Hàng trên biểu diễn quỹ đạo trọng số học được của SGD (đường "
            "màu cam) so với quỹ đạo nghiệm tối ưu Oracle (đường màu xanh dương); hàng dưới biểu diễn lịch trình tốc độ học eta_t do Algorithm 1 tự động "
            "sinh ra theo thời gian t. Kết quả thực nghiệm làm nổi bật ba hiện tượng động học quan trọng. Thứ nhất, khi ell = 1 (dịch chuyển diễn ra liên "
            "tục ở cả 1999 bước chuyển tiếp), tốc độ học eta_t tạo thành một đường cong hình chữ V trơn tru: ở đầu và cuối chân trời thời gian (khi bán "
            "kính xoắn ốc lớn nên vận tốc dịch chuyển gamma_t lớn), Algorithm 1 ghim tốc độ học ở mức trần eta_max = 0.5; nhưng ở khoảng giữa quanh "
            "t = 1000 (khi xoắn ốc đi vào tâm 0 làm gamma_t tiến về 0), thuật toán tự động hạ mượt eta_t xuống mức cực tiểu 0.0068 để triệt tiêu nhiễu "
            "nhãn, giúp quỹ đạo SGD bám khít tuyệt đối quỹ đạo gốc với tổng Cumulative Regret cực thấp 0.916 ± 0.045. Thứ hai, khi tăng khoảng lấy mẫu "
            "lên ell = 4 (499 lần dịch chuyển, Regret = 0.956 ± 0.046) và đặc biệt là ell = 16 (124 lần dịch chuyển nhảy bậc), đồ thị hàng dưới tái hiện "
            "chính xác hiện tượng dao động răng cưa (sawtooth oscillation) mà bài báo nhấn mạnh: cứ trong 15 bước phân phối đứng yên giữa hai lần nhảy "
            "(gamma_t = 0), Algorithm 1 lập tức giảm nhanh tốc độ học theo quy luật 1/t xuống tới 0.0065 để hội tụ sâu vào điểm dừng hiện tại; và ngay "
            "tại bước thứ 16 khi nghiệm tối ưu nhảy sang vị trí mới, xung gamma_t > 0 kích hoạt tốc độ học bật vọt trở lại trần 0.5 (tạo ra các gai "
            "restart tự động), giữ cho tổng Regret ở ell = 16 chỉ ở mức 1.218 ± 0.054."
        ),
    ),
    (
        16,
        "II. LINEAR REGRESSION",
        "Khoa (TV1)",
        "Figure 5-top: regret theo số chiều",
        (
            "Slide 16 khép lại Phần II với kết quả tái lập Figure 5-top (hình `figures/fig5_top_linear_regret_and_path.png` và bảng số liệu trung bình "
            "± độ lệch chuẩn trên 3 seeds từ `results/linear/fig5_top_summary.csv`, tương ứng với 15 runs chính thức đã xác thực trên W&B Public). Thí "
            "nghiệm cố định bước nhảy rời rạc ell = 8 và khảo sát sự co giãn của thuật toán trên 5 số chiều không gian d in {2, 8, 32, 128, 512} với trần "
            "tốc độ học eta_max = 1 / sqrt(d). Nhìn vào đồ thị bên trái (Figure 5 top-left, trục tung logarit từ 10^{-1} đến 10^5), cả 5 đường cong "
            "Cumulative Regret do nhóm tái lập khớp hoàn toàn cả về hình dáng lẫn độ lớn tuyệt đối với bài báo gốc: tại thời điểm cuối T = 2000, tổng "
            "Regret trung bình trên 3 seeds đạt 0.886 ± 0.038 ở d = 2 (bài báo ~0.9), 13.398 ± 0.131 ở d = 8 (bài báo ~13.3), 792.185 ± 8.499 ở d = 32 "
            "(bài báo ~800), 7049.297 ± 16.764 ở d = 128, và 33034.474 ± 7.764 ở d = 512. Lý do cả 5 đường cong đều có hình dạng chữ S nằm ngang (tăng "
            "nhanh trong 400 bước đầu, đi ngang phẳng lặng từ t = 500 đến t = 1500, rồi dốc lên lại từ t = 1500 đến 2000) được giải thích trực tiếp từ "
            "cấu trúc quỹ đạo xoắn ốc lập phương r(t)^3: trong khoảng giữa chân trời thời gian, nghiệm tối ưu di chuyển rất chậm quanh gốc tọa độ nên "
            "Algorithm 1 hạ thấp tốc độ học và hầu như không phát sinh thêm độ tiếc nuối tích lũy. Đồ thị bên phải (Figure 5 top-right) chiếu quỹ đạo học "
            "được trong không gian 128 chiều lên mặt phẳng 2 tọa độ đầu tiên, chứng minh rằng ngay cả ở chiều cao d = 128, mô hình SGD vẫn bám sát từng "
            "vòng xoắn của nghiệm tối ưu. Ngoài ra, trong bảng đối chiếu mở rộng của nhóm (`linear_baselines_comparison.csv`), Algorithm 1 đạt Regret "
            "thấp nhất trên cả 5 số chiều so với các lịch trình giảm dần 1/t, 1/sqrt(t) và hằng số, đồng thời bản ước lượng trực tuyến EMA (không dùng "
            "thông tin Oracle) đạt kết quả gần như trùng khít với bản Oracle (ví dụ ở d = 128: EMA đạt 7049.49 so với 7049.30 của Oracle). Như ghi chú "
            "cuối slide lưu ý, các giá trị Regret này là rủi ro dư bậc hai của riêng bài toán Linear nên không dùng để so sánh số học trực tiếp với hàm "
            "mất mát Binary Cross-Entropy của phần Convex."
        ),
    ),
    (
        17,
        "III. GENERAL CONVEX LOSSES",
        "Khôi (TV2)",
        "Bài toán lồi tổng quát và giả thiết",
        (
            "Bước sang Phần III — General Convex Losses (từ Slide 17 đến Slide 26, do Khôi phụ trách, tài liệu chi tiết tại `docs/04_convex.md`), Slide "
            "17 thiết lập khung lý thuyết cho lớp hàm mất mát lồi tổng quát tại Mục 4 của bài báo. Câu hỏi đầu tiên đặt ra là: Tại sao ta không thể tiếp "
            "tục áp dụng phương pháp phương trình vi phân ngẫu nhiên SDE và Moment ODE của Phần II cho hàm lồi bất kỳ? Lý do là với hàm lồi tổng quát "
            "(điển hình là hàm Binary Cross-Entropy trong hồi quy Logistic), đạo hàm mẫu grad_l(theta, z) phụ thuộc phi tuyến vào trọng số theta chứ "
            "không còn là hàm tuyến tính bậc nhất của hiệu số (theta - theta*_t); do đó khi lấy kỳ vọng ta không thể khép kín hệ phương trình vi phân "
            "cho moment bậc hai. Thay vào đó, nhóm tác giả phân tích trực tiếp trong miền thời gian rời rạc bằng cách thiết lập cận trên giải tích cho "
            "kỳ vọng Dynamic Regret dựa trên hệ thống giả thiết chuẩn tại Assumption 4.1. Cụ thể, gọi khoảng cách dịch chuyển của nghiệm tối ưu giữa hai "
            "bước liên tiếp (oracle drift) là gamma_t = ||theta*_t - theta*_{t+1}||_2, và khoảng cách theo dõi (tracking distance) từ bộ học hiện tại "
            "đến nghiệm tối ưu là D_t = ||theta_t - theta*_t||_2. Assumption 4.1 yêu cầu ba điều kiện: Thứ nhất, gradient mẫu đơn lẻ là ước lượng không "
            "chệch của gradient quần thể grad_l_bar_t(theta_t) và có phương sai bị chặn đều bởi sigma^2, suy ra phương sai của gradient trung bình trên "
            "mini-batch kích thước B_t bị chặn bởi sigma^2 / B_t. Thứ hai, hàm mất mát kỳ vọng l_bar_t(theta) là hàm lồi và thỏa mãn điều kiện trơn "
            "L-smooth trên miền tham số Theta: ||grad_l_bar_t(theta_1) - grad_l_bar_t(theta_2)||_2 <= L ||theta_1 - theta_2||_2. Thứ ba, cả nghiệm tối "
            "ưu theta*_t lẫn quỹ đạo cập nhật Projected SGD đều nằm trong tập lồi đóng bị chặn Theta có đường kính hữu hạn D_max, sao cho D_t <= D_max "
            "với mọi t, và tốc độ học thỏa mãn điều kiện ổn định 0 < eta_t <= 1/L. Ghi chú cuối slide nhấn mạnh một điểm quan trọng khi kiểm toán thực "
            "nghiệm: Theorem 4.2 đòi hỏi tốc độ học eta_t phải thích ứng nhân quả theo lịch sử tới t-1 và quỹ đạo nằm trong miền bị chặn, trong khi "
            "notebook thực nghiệm của tác giả sẽ có những điểm nới lỏng các giả thiết này."
        ),
    ),
    (
        18,
        "III. GENERAL CONVEX LOSSES",
        "Khôi (TV2)",
        "Cận trên và cận dưới",
        (
            "Dưới các giả thiết của Assumption 4.1, Slide 18 trình bày hai bất đẳng thức trung tâm của Theorem 4.2: cận trên trung gian tại Công thức "
            "(15) và cận trên khả kiến (observable upper bound) tại Công thức (16), cùng kết quả cận dưới tại Theorem 4.7. Với tốc độ học thỏa mãn "
            "0 < eta_t <= 1/L, ta định nghĩa hệ số bước nhảy hiệu dụng a_t = 2*eta_t - L*eta_t^2 > 0. Trong chứng minh tại Phụ lục B.1, xuất phát từ "
            "tính chất không giãn của phép chiếu Euclid lên tập lồi Theta và bất đẳng thức co-coercivity của hàm lồi L-smooth: l_bar_t(theta_t) - "
            "l_bar_t(theta*_t) <= <grad_l_bar_t(theta_t), theta_t - theta*_t> - (1 / 2L)||grad_l_bar_t(theta_t)||_2^2, tác giả biểu diễn hiệu khoảng "
            "cách bình phương D_{t+1}^2 - D_t^2 qua bước cập nhật mini-batch và chia cả hai vế cho a_t để thu được Công thức (15). Nhìn vào Công thức "
            "(15), ta thấy cận trên của kỳ vọng Dynamic Regret ở mỗi bước t gồm bốn hạng tử: hiệu khoảng cách chuẩn hóa (D_t^2 - D_{t+1}^2)/a_t, hạng "
            "tử phương sai nhiễu gradient (sigma^2 * eta_t^2) / (B_t * a_t), hạng tử bình phương độ trôi gamma_t^2 / a_t, và tích vô hướng chéo "
            "2 <theta*_t - theta*_{t+1}, theta_{t+1} - theta*_t> / a_t. Tuy nhiên, bộ học không thể dùng trực tiếp Công thức (15) để tối ưu tốc độ học "
            "vì nó chứa khoảng cách chưa quan sát được D_t và trọng số tương lai theta_{t+1}. Để khử các đại lượng ẩn này, tác giả áp dụng bất đẳng "
            "thức Cauchy-Schwarz để chặn hạng tử chéo bởi 2 * gamma_t * D_max, sau đó thực hiện phép biến đổi tổng từng phần (telescoping sum) trên "
            "chuỗisum (D_t^2 - D_{t+1}^2)/a_t và thay D_t^2 bởi cận trên đường kính miền D_max^2 * max(0, 1/a_t - 1/a_{t-1}). Kết quả là Công thức (16) "
            "hoàn toàn chỉ phụ thuộc vào các hằng số đã biết và chuỗi tốc độ học: tổng Dynamic Regret bị chặn bởi tổng của chi phí biến thiên giảm tốc "
            "độ học D_max^2 (1/a_t - 1/a_{t-1})_+, chi phí nhiễu mini-batch (sigma^2 * eta_t^2)/(B_t * a_t), và chi phí dịch chuyển phân phối "
            "(gamma_t^2 + 2*D_max*gamma_t)/a_t. Bổ sung cho cận trên này, Theorem 4.7 thiết lập cận dưới khi hàm mất mát có thêm tính lồi mạnh mu > 0, "
            "chứng minh cấu trúc phụ thuộc vào phương sai và độ trôi là có cơ sở lý thuyết chặt chẽ."
        ),
    ),
    (
        19,
        "III. GENERAL CONVEX LOSSES",
        "Khôi (TV2)",
        "Proposition 4.3 và lịch trình hai ngưỡng",
        (
            "Từ cận trên khả kiến tại Công thức (16), Slide 19 trình bày Proposition 4.3 — thuật toán chọn tốc độ học thích ứng theo quy tắc kẹp hai "
            "ngưỡng đóng [tau_{1,t}, tau_{2,t}] cho hàm lồi tổng quát (được cài đặt tại lớp `ConvexSchedule` trong `core/schedules.py`). Ý tưởng của "
            "Proposition 4.3 là tối ưu tham lam từng bước: tại bước thời gian t >= 2, khi đã biết tốc độ học của bước trước là eta_{t-1}, bộ học chọn "
            "eta*_t in [0, 1/L] để cực tiểu hóa hạng tử thứ t của cận trên (16), được viết tường minh tại bài toán tối ưu Công thức (17). Điểm đặc sắc "
            "về mặt giải tích của hàm mục tiêu trong Công thức (17) nằm ở toán tử phần dương (1/(2*eta - L*eta^2) - 1/(2*eta_{t-1} - L*eta_{t-1}^2))_+. "
            "Vì hàm mẫu số a(eta) = 2*eta - L*eta^2 đồng biến nghiêm ngặt trên đoạn [0, 1/L], nên hàm nghịch đảo 1/a(eta) nghịch biến nghiêm ngặt. Hệ "
            "quả là: Nếu ta chọn tốc độ học mới lớn hơn hoặc bằng bước trước (eta >= eta_{t-1}), thì 1/a(eta) <= 1/a(eta_{t-1}) nên số hạng phạt biến "
            "thiên D_max^2 * (...)_+ triệt tiêu hoàn toàn bằng 0, và hàm mục tiêu chỉ còn phụ thuộc vào hệ số độ trôi nhỏ b_{1,t} = gamma_t^2 + "
            "2*D_max*gamma_t. Ngược lại, nếu ta muốn giảm tốc độ học (eta < eta_{t-1}), số hạng phạt D_max^2 lập tức bật lên dương và cộng vào tử số, "
            "làm hệ số hiệu dụng tăng lên thành b_{2,t} = b_{1,t} + D_max^2 = (gamma_t + D_max)^2. Giải phương trình đạo hàm bằng 0 trên từng nhánh lồi "
            "này cho ra hai điểm dừng đóng tại Công thức (18) và (19): ngưỡng dưới tau_{1,t} (ứng với b_{1,t}) và ngưỡng trên tau_{2,t} (ứng với b_{2,t}), "
            "luôn thỏa mãn bất đẳng thức 0 <= tau_{1,t} < tau_{2,t} <= 1/L. Do hai nhánh lồi ghép trơn hướng lên trên tại điểm gãy eta_{t-1}, nghiệm tối "
            "ưu toàn cục của Công thức (17) chính là phép chiếu kẹp (clipping) tại Công thức (20): eta*_t = clip(eta*_{t-1}, tau_{1,t}, tau_{2,t}). "
            "Quy tắc này có trực giác rất rõ ràng: khi phân phối dịch chuyển mạnh làm ngưỡng dưới tau_{1,t} dâng cao vượt quá eta*_{t-1}, bộ điều khiển "
            "buộc phải TĂNG tốc độ học lên tau_{1,t}; khi môi trường đứng yên và eta*_{t-1} cao hơn ngưỡng trên tau_{2,t}, nó hạ tốc độ học về tau_{2,t}; "
            "còn khi nằm giữa hai ngưỡng, nó giữ nguyên tốc độ học để không chịu phạt telescoping. Cuối cùng, khi không biết độ trôi Oracle gamma_t, "
            "Remark 4.4 đề xuất ước lượng nhân quả bằng trung bình trượt hàm mũ EMA của bước nhảy mô hình: gamma_hat_t = beta*gamma_hat_{t-1} + "
            "(1 - beta)||theta_t - theta_{t-1}||_2."
        ),
    ),
    (
        20,
        "III. GENERAL CONVEX LOSSES",
        "Khôi (TV2)",
        "Logistic Regression: thiết kế benchmark",
        (
            "Slide 20 trình bày chi tiết thiết kế benchmark thực nghiệm Hồi quy Logistic cao chiều (Figure 5-bottom tại Mục 6.1 của bài báo) và bảng "
            "tham số tương thích với notebook gốc của tác giả (`regression_experiments.ipynb` [5], được kiểm toán chi tiết trong `docs/convex_author_v1/SOURCE_AUDIT.md`). "
            "Ở khung bên trái, mô hình sinh dữ liệu tại mỗi bước t = 1, ..., T rút độc lập vector đặc trưng x ~ N(0, I_d) và nhiễu logit Gaussian "
            "eps ~ N(0, s^2) với x độc lập với eps. Nhãn mục tiêu trong bài toán này không phải là nhãn nhị phân cứng {0, 1}, mà là nhãn xác suất mềm "
            "(soft label) y in (0, 1) được sinh bởi hàm logistic đi qua vector tham chiếu sinh dữ liệu (generating reference) v_t: y = sigmoid(x^T v_t "
            "+ eps). Hàm mất mát mẫu đơn lẻ là hàm Binary Cross-Entropy (BCE) viết dưới dạng ổn định số học: l(w; x, y) = softplus(x^T w) - y * (x^T w), "
            "và hàm mục tiêu quần thể tại bước t là kỳ vọng F_t(w) = E_{x, eps}[l(w; x, y)]. Ở khung bên phải, qua quá trình kiểm toán trực tiếp "
            "notebook TensorFlow của tác giả (Cell 8 và Cell 23), nhóm đã xác lập bảng cấu hình tương thích hoàn toàn với mã gốc: tổng số bước T = 2000 "
            "được chia thành 200 pha, mỗi pha gồm 10 bước mini-batch giữ nguyên vector tham chiếu v_t theo quỹ đạo xoắn ốc Công thức (26); kích thước "
            "batch B = 256; khảo sát trên 5 số chiều d in {2, 8, 32, 128, 512}; độ lệch chuẩn của nhiễu logit khi sinh nhãn là s = 0.1 (tức phương sai "
            "nhiễu nhãn là s^2 = 0.01 — cần phân biệt rõ tham số s này với tham số chặn phương sai gradient sigma^2 = d / 4 được truyền vào bộ lập lịch "
            "Proposition 4.3 cùng hằng số trơn L = 0.25 và đường kính D_max = d). Trọng số ban đầu được khởi tạo tại gốc tọa độ weights = 0 và không áp "
            "dụng phép chiếu hình cầu (no projection). Để đánh giá công bằng, nhóm bổ sung hai lịch trình đối chứng là Constant (eta = 0.1) và "
            "Inverse-sqrt (eta_t = 0.1 / sqrt(t + 1)), chạy trên 3 hạt giống ngẫu nhiên `seeds = [0, 1, 2]` với cơ chế ghép cặp chính xác cùng ma trận "
            "covariates X và nhãn y giữa các phương pháp."
        ),
    ),
    (
        21,
        "III. GENERAL CONVEX LOSSES",
        "Khôi (TV2)",
        "Metric đánh giá và hiệu chỉnh population oracle",
        (
            "Slide 21 trình bày một phát hiện phương pháp luận rất quan trọng của nhóm trong quá trình kiểm toán và tái lập Thực nghiệm 2: sự khác biệt "
            "giữa các thước đo độ tiếc nuối (evaluation metrics) và bài toán hiệu chỉnh nghiệm tối ưu quần thể (population oracle calibration, tài liệu "
            "chi tiết tại `docs/results/convex-author-v1/RESULTS.md` [6]). Ở cột bên trái, nhóm định nghĩa và tách bạch ba thước đo chênh lệch mất mát "
            "tích lũy (cumulative loss gap metrics): Thứ nhất là G_batch (ký hiệu M0 trong báo cáo kỹ thuật) — tổng chênh lệch hàm mất mát BCE trung "
            "bình mẫu trên chính mini-batch hiện tại giữa bộ học theta_t và vector tham chiếu sinh dữ liệu v_t; đây chính là đại lượng mà notebook gốc "
            "của tác giả ghi lại để vẽ Figure 5-bottom. Thứ hai là G_ref (ký hiệu M1) — tổng chênh lệch hàm mất mát kỳ vọng quần thể F_t(theta_t) - "
            "F_t(v_t) khi lấy vector sinh dữ liệu v_t làm mốc so sánh. Thứ ba là G_oracle (ký hiệu M2) — tổng rủi ro dư quần thể thực sự F_t(theta_t) - "
            "F_t(w*_{t,approx}) đúng theo định nghĩa Dynamic Regret tại Công thức (2) của bài báo. Tại sao G_ref và G_oracle lại khác nhau? Khung bên "
            "phải giải thích cơ chế toán học của hiện tượng này: khi nhãn mềm y = sigmoid(x^T v_t + eps) có chứa nhiễu Gaussian eps ~ N(0, s^2) với "
            "s > 0 và chuẩn tham chiếu r = ||v_t||_2 > 0, phép lấy kỳ vọng theo nhiễu làm trơn hàm sigmoid (convolution smoothing). Do tính bất biến "
            "quay của phân phối chuẩn chuẩn tắc x ~ N(0, I_d), nghiệm cực tiểu quần thể thực sự w*_t = argmin_w F_t(w) nằm cùng hướng với vector sinh "
            "dữ liệu v_t, nhưng có độ dài chuẩn bị co nhỏ lại nghiêm ngặt (||w*_t||_2 < r) theo phương trình điều kiện dừng đạo hàm bậc nhất một chiều: "
            "h(||w*_t||_2) = (r / q) * h(q), trong đó q = sqrt(r^2 + s^2) và h(a) = E_{Z ~ N(0,1)}[Z * sigmoid(a * Z)]. Vì F_t(w*_t) < F_t(v_t), nếu chỉ "
            "so sánh với v_t như trong G_batch hoặc G_ref thì chênh lệch mất mát có thể nhận giá trị âm khi bộ học đi vào vùng gần w*_t hơn v_t! Để "
            "đánh giá chính xác G_oracle mà không rơi vào chi phí tích phân d chiều, nhóm đã biến đổi kỳ vọng F_t(w) về tích phân cầu phương Gaussian "
            " và giải phương trình gốc bán kính (numerical radial root solving) với dung sai số học cực chặt để tìm nghiệm xấp xỉ w*_{t,approx}, đảm bảo "
            "G_oracle luôn không âm và phản ánh trung thực rủi ro dư quần thể."
        ),
    ),
    (
        22,
        "III. GENERAL CONVEX LOSSES",
        "Khôi (TV2)",
        "Tái lập notebook gốc và kiểm chứng trên cùng dữ liệu",
        (
            "Trước khi mở rộng sang đánh giá đa hạt giống và các baseline đối chứng, Slide 22 trình bày bằng chứng xác thực cấp độ cao nhất về độ trung "
            "thực triển khai: tái lập trực tiếp notebook gốc của tác giả trên Google Colab (Giao thức R) và kiểm chứng khớp từng bước trên cùng dữ liệu "
            "đầu vào (Shared-Input Implementation Parity, báo cáo tại `docs/results/convex-author-v1/RESULTS.md` [6]). Ở nửa bên trái và trên đồ thị, "
            "nhóm đã chạy nguyên bản notebook TensorFlow của tác giả trên môi trường Google Colab và trích xuất giá trị G_batch (Cumulative empirical "
            "batch loss gap tại bước cuối T = 2000) đã lưu trong output của notebook gốc trên cả 5 số chiều: đạt 0.985818 tại d = 2; 3.735277 tại d = 8; "
            "48.065598 tại d = 32; 180.874344 tại d = 128; và 1207.973877 tại d = 512. Lưu ý rằng bản thân notebook gốc của tác giả chỉ chạy một quỹ "
            "đạo đơn lẻ cho mỗi số chiều (không lấy trung bình trên 3 seeds) và cũng không cài đặt các đường baseline Constant hay Inverse-sqrt cho bài "
            "toán Logistic. Tiếp theo, ở nửa bên phải, vì phép sinh số giả ngẫu nhiên giữa TensorFlow trong notebook gốc và trình chạy độc lập bằng "
            "NumPy của nhóm có luồng RNG khác nhau, để chứng minh thuật toán cập nhật trọng số và lịch trình hai ngưỡng của nhóm khớp hoàn toàn với mã "
            "tác giả, nhóm đã xuất bộ dữ liệu fixture chung (chứa toàn bộ chuỗi mini-batch X_t, nhãn y_t và quỹ đạo tham chiếu v_t qua 2000 bước tại ba "
            "số chiều d = 2, 8, 32) để chạy đối chiếu trực tiếp từng bước một (step-by-step parity check). Kết quả kiểm tra trên toàn bộ 2000 bước thời "
            "gian cho thấy độ lệch tuyệt đối tối đa (maximum absolute difference) giữa trình chạy độc lập của nhóm và mã TensorFlow của tác giả chỉ ở "
            "mức sai số dấu phẩy động đơn (float32 machine epsilon): sai lệch hàm mất mát tối đa là 1.19 x 10^{-7}, sai lệch vector gradient tối đa là "
            "5.96 x 10^{-8}, sai lệch vector trọng số tối đa là 2.38 x 10^{-7}, và sai lệch tổng Cumulative batch loss gap sau 2000 bước chỉ là "
            "2.48 x 10^{-5}. Kết quả này xác nhận 100% rằng bản cài đặt lịch trình Convex của nhóm không có bất kỳ sai khác logic nào so với mã gốc của "
            "tác giả."
        ),
    ),
    (
        23,
        "III. GENERAL CONVEX LOSSES",
        "Khôi (TV2)",
        "Kết quả độc lập theo thiết kế tác giả",
        (
            "Sau khi đã xác nhận độ khớp thuật toán ở Slide 22, Slide 23 trình bày kết quả tái lập độc lập đa hạt giống theo đúng thiết kế tương thích "
            "tác giả (Giao thức I — `convex-author-aligned-v1`, gồm tổng cộng 45 runs = 5 số chiều d in {2, 8, 32, 128, 512} x 3 phương pháp `Convex`, "
            "`Constant`, `Inverse-sqrt` x 3 seeds {0, 1, 2}, đã được log và xác thực đầy đủ trên W&B [6]). Trong thiết kế này, mọi phương pháp ở cùng một "
            "số chiều và cùng một seed đều nhận chính xác cùng một luồng ma trận đặc trưng X và nhãn y, và được đánh giá bằng thước đo chuẩn xác nhất là "
            "G_oracle (tổng rủi ro dư quần thể so với nghiệm cực tiểu quần thể xấp xỉ w*_{t,approx}, giá trị càng thấp càng tốt). Bảng số liệu trung bình "
            "± độ lệch chuẩn mẫu (mean ± sample SD) và lưới 5 đồ thị bên phải cho thấy hai kết quả thực nghiệm rất rõ ràng: Thứ nhất, trên toàn bộ bốn "
            "số chiều từ nhỏ đến lớn là d = 2, 8, 32 và 128, lịch trình hai ngưỡng Proposition 4.3 (`Convex`, đường màu xanh dương) đạt tổng rủi ro dư "
            "quần thể thấp nhất và bỏ xa cả hai phương pháp baseline với phương sai giữa các seed rất nhỏ: tại d = 2, `Convex` đạt 1.0073 ± 0.0014 so với "
            "7.7864 của `Constant` và 31.3175 của `Inverse-sqrt`; tại d = 8, `Convex` đạt 3.7636 ± 0.0116 so với 56.2355 và 92.4544; tại d = 32, `Convex` "
            "đạt 47.8951 ± 0.0253 so với 186.3040 và 207.3436; và tại d = 128, `Convex` đạt 181.2661 ± 0.9833 so với 336.9341 và 361.7788. Thứ hai — một "
            "phát hiện phản biện quan trọng của nhóm — là tại số chiều lớn nhất d = 512 (đồ thị cuối cùng ở hàng dưới bên phải), thứ tự xếp hạng bị đảo "
            "chiều: lịch trình `Convex` có rủi ro dư tích lũy vọt lên 1205.4913 ± 2.1730 (rất khớp với con số M0 = 1207.97 trong chính notebook của tác "
            "giả ở Slide 22!), cao hơn đáng kể so với mức 504.0267 ± 0.1265 của lịch tốc độ học hằng số `Constant` (eta = 0.1) và 527.0837 của "
            "`Inverse-sqrt`. Lý do vì sao tác giả không nhận ra hiện tượng đảo chiều này trong bài báo là vì Figure 5-bottom của bài báo chỉ vẽ riêng "
            "đường cong của lịch trình `Convex` mà không vẽ đường baseline đối chứng trên cùng đồ thị. Nguyên nhân cơ chế của ngoại lệ tại d = 512 sẽ được "
            "nhóm giải phẫu chi tiết qua các thiết lập đối chiếu và ablation ở ba slides tiếp theo."
        ),
    ),
    (
        24,
        "III. GENERAL CONVEX LOSSES",
        "Khôi (TV2)",
        "Đối chiếu các thiết lập thực nghiệm",
        (
            "Slide 24 đặt lên bàn cân hai thiết lập thực nghiệm mà nhóm đã xây dựng và kiểm chứng cho bài toán Hồi quy Logistic: Thiết lập tương thích "
            "tác giả (Giao thức I, vừa trình bày ở Slide 23) và Thiết lập ban đầu theo sát mô tả lý thuyết trong văn bản bài báo (Giao thức P / CP04, "
            "báo cáo tại `docs/results/cp04-cp05/RESULTS.md` [4]). Việc đối chiếu này bắt nguồn từ một thực tế rất phổ biến trong nghiên cứu học máy: "
            "văn bản bài báo (Mục 4 và Mục 6.1) mô tả một quy trình học trực tuyến nhân quả đầy đủ, nhưng mã notebook của tác giả lại sử dụng một số "
            "quy ước đơn giản hóa. Bảng so sánh 7 yếu tố trên slide chỉ rõ từng điểm khác biệt giữa hai thiết lập: (1) Về lưới pha dịch chuyển: notebook "
            "tác giả chia 2000 bước thành 200 pha x 10 bước, trong khi thiết lập ban đầu dùng quy tắc sample-and-hold với ell = 8 giống thực nghiệm "
            "Linear; (2) Về độ lệch chuẩn nhiễu nhãn: notebook tác giả truyền std s = 0.1 vào hàm sinh nhiễu, trong khi văn bản trang 9 ghi phương sai "
            "sigma^2 = 0.1 (tức std s = sqrt(0.1) ≈ 0.3162); (3) Về tín hiệu độ trôi: notebook tác giả dùng độ trôi nhìn trước của vector tham chiếu "
            "||v_{t+1} - v_t||_2, trong khi thiết lập ban đầu dùng ước lượng nhân quả EMA của bước nhảy mô hình theo Remark 4.4; (4) Về tốc độ học khởi "
            "tạo: notebook tác giả khởi tạo bằng 0, còn thiết lập ban đầu đặt tốc độ học bước trước bằng 1.0 để kích hoạt EMA; (5) Về phép chiếu: "
            "notebook tác giả không chiếu, còn thiết lập ban đầu chiếu vào hình cầu bán kính R = d/2 để thỏa mãn Assumption 4.1(iii); và (6–7) Về thước "
            "đo và bộ so sánh: cả hai đều được nhóm chuẩn hóa bằng rủi ro dư quần thể so với nghiệm cực tiểu quần thể xấp xỉ. Dù khác nhau ở 5 yếu tố "
            "cấu hình, dãy kết quả của Thiết lập ban đầu ở nửa dưới slide (d = 2: 0.3355; d = 8: 2.0333; d = 32: 40.0423; d = 128: 233.5208; d = 512: "
            "1385.0089 so với Constant 504.8952) vẫn cho thấy cùng một quy luật nhất quán tuyệt đối với Thiết lập tác giả: `Convex` vượt trội ở d = 2, "
            "8, 32, 128 nhưng đều chịu hiện tượng suy giảm hiệu năng và thua `Constant` tại chiều cao d = 512."
        ),
    ),
    (
        25,
        "III. GENERAL CONVEX LOSSES",
        "Khôi (TV2)",
        "Độ nhạy theo dịch chuyển và nhiễu nhãn",
        (
            "Slide 25 trình bày kết quả Phân tích Độ nhạy Một nhân tố tại Một thời điểm (One-Factor-At-a-Time Sensitivity Analysis — Giao thức CP05, "
            "báo cáo tại `docs/results/cp04-cp05/RESULTS.md` [4]) nhằm đánh giá mức độ bền vững của lịch trình hai ngưỡng `Convex` khi môi trường thay "
            "đổi. Thí nghiệm được thực hiện trên thiết lập ban đầu tại số chiều cao d = 128 qua T = 2000 bước, với 3 seeds cho mỗi điều kiện (tổng cộng "
            "15 runs xác thực trên W&B) và đánh giá bằng rủi ro dư quần thể tích lũy G_oracle so với nghiệm cực tiểu quần thể xấp xỉ. Ở đồ thị và bảng "
            "bên trái (`cp05_drift_sensitivity.png`), nhóm giữ cố định độ lệch chuẩn nhiễu nhãn ở mức cơ sở s = sqrt(0.1) và thay đổi khoảng lấy mẫu giữ "
            "bậc ell in {1, 8, 16}. Kết quả cho thấy: khi phân phối dịch chuyển thưa hơn ở ell = 16 (chỉ có 124 lần nhảy bậc), mô hình có chuỗi 15 bước "
            "tĩnh dài hơn giữa các lần nhảy để hội tụ sâu nên rủi ro dư giảm mạnh xuống còn 141.034 ± 0.856. Tại mức cơ sở ell = 8 (249 lần nhảy), rủi ro "
            "dư là 233.521 ± 0.595. Khi chuyển sang chế độ khắc nghiệt nhất ell = 1 (nghiệm tối ưu trôi liên tục ở toàn bộ 1999 bước chuyển tiếp, làm tổng "
            "độ dài đường đi tích lũy tăng gấp 2.68 lần so với ell = 8), rủi ro dư tăng lên mức 330.324 ± 2.290 — tức chỉ tăng 1.41 lần, chứng tỏ bộ lập "
            "lịch thích ứng kiểm soát tốt sự gia tăng của tần suất dịch chuyển. Tuy nhiên, nhóm lưu ý chặt chẽ về mặt phương pháp luận rằng việc thay đổi "
            "ell trên cùng quỹ đạo xoắn ốc làm thay đổi đồng thời cả tần suất nhảy, biên độ mỗi bước nhảy lẫn tổng ngân sách biến thiên đường đi (path "
            "variation). Ở đồ thị và bảng bên phải (`cp05_noise_sensitivity.png`), nhóm giữ cố định ell = 8 và quét độ lệch chuẩn nhiễu nhãn s in {0, "
            "sqrt(0.1), 1}: rủi ro dư tăng êm từ 224.217 ± 0.669 (khi không có nhiễu nhãn s = 0) lên 233.521 ± 0.595 (ở s = sqrt(0.1)) và đạt 282.454 ± "
            "0.796 khi nhiễu nhãn tăng mạnh lên s = 1. Ghi chú cuối slide nhấn mạnh rằng với mô hình nhãn mềm Gaussian, việc thay đổi s không chỉ làm "
            "thay đổi phương sai nhiễu gradient mà còn làm thay đổi trực tiếp hàm mục tiêu quần thể F_t(w) và độ co chuẩn của nghiệm tối ưu quần thể w*_t."
        ),
    ),
    (
        26,
        "III. GENERAL CONVEX LOSSES",
        "Khôi (TV2)",
        "Controlled ablations và giới hạn tại d = 512",
        (
            "Slide 26 khép lại Phần III bằng nghiên cứu Can thiệp có Đối chứng (Controlled Ablations — gồm 36 runs ghép cặp = 4 can thiệp đơn lẻ x 3 số "
            "chiều d in {8, 128, 512} x 3 seeds, báo cáo tại `docs/results/convex-author-v1/RESULTS.md` [6]) và giải thích nguyên nhân gốc rễ khiến lịch "
            "trình `Convex` suy giảm hiệu năng tại số chiều lớn d = 512. Để bóc tách tác động của từng khác biệt giữa thiết lập tác giả và thiết lập ban "
            "đầu (đã liệt kê ở Slide 24), nhóm lấy trình chạy tương thích tác giả làm đối chứng cơ sở (control) và thực hiện lần lượt 4 can thiệp đơn lẻ, "
            "ghi lại độ chênh lệch rủi ro dư quần thể Delta G_oracle (giá trị âm nghĩa là can thiệp giúp giảm rủi ro dư). Bảng kết quả và đồ thị chẩn "
            "đoán 2x2 bên phải (`d512_diagnostics.png`) tiết lộ bốn phát hiện cơ chế cực kỳ giá trị: Thứ nhất, ở dòng can thiệp `Learner-motion EMA` (thay "
            "độ trôi tham chiếu bằng ước lượng nhân quả EMA với khởi tạo ấm), tại d = 8 và d = 128 việc không biết trước độ trôi thực làm rủi ro dư tăng "
            "thêm +87.632 và +179.806; nhưng nếu giữ nguyên khởi tạo lạnh của notebook gốc (nơi cả trạng thái EMA ban đầu lẫn tốc độ học bước 0 đều bằng "
            "0), thì ở bước 1 mô hình không di chuyển (theta_1 = theta_0), kéo theo EMA độ trôi tiếp tục bằng 0 và khóa chặt bộ học trong trạng thái đứng "
            "yên vĩnh viễn trên cả 9 ca kiểm tra! Ngược lại, tại d = 512, chính việc dùng EMA lại làm giảm mạnh rủi ro dư tới -679.547 đơn vị (từ 1205.49 "
            "xuống ~525.94, tiến sát mức 504.03 của Constant). Đồ thị 2x2 bên phải giải thích rõ nguyên nhân: tại d = 512, bài báo đặt đường kính miền "
            "D_max = d = 512 quá lớn so với khoảng cách thực tế tới nghiệm tối ưu (chỉ cỡ 1 đến 3), làm hệ số b_{1,t} = gamma_t^2 + 2*D_max*gamma_t bị "
            "phóng đại gấp hàng trăm lần! Hệ quả là ngưỡng dưới tau_{1,t} bị đẩy vọt lên sát mức trần 1/L = 4.0 (cao gấp 40 lần mức 0.1 của Constant), gây "
            "tích lũy phương sai gradient trong không gian 512 chiều. Cuối cùng, can thiệp `Projection` (chiếu vào cầu bán kính d/2) cho chênh lệch đúng "
            "bằng 0 ± 0 trên mọi số chiều vì chuẩn trọng số thực tế không bao giờ chạm tới biên d/2."
        ),
    ),
    (
        27,
        "IV. NON-CONVEX LOSSES",
        "Khoa (TV1)",
        "Hàm không lồi và tiêu chí chuẩn gradient",
        (
            "Bước sang Phần IV — Non-convex Losses (từ Slide 27 đến Slide 31, do Khoa thực hiện và trình bày, tài liệu chi tiết tại `docs/05_nonconvex.md`), Slide 27 "
            "giải thích bước chuyển dịch khái niệm từ tối ưu lồi sang tối ưu không lồi tại Mục 5 của bài báo — cơ sở lý thuyết để huấn luyện trực tuyến "
            "các mạng nơ-ron sâu. Câu hỏi nền tảng đầu tiên là: Tại sao hai định nghĩa ở Phần II và Phần III (độ tiếc nuối dựa trên khoảng cách tới "
            "nghiệm tối ưu toàn cục theta*_t và độ dịch chuyển dựa trên khoảng cách Euclid ||theta*_t - theta*_{t+1}||_2) không còn sử dụng được cho hàm "
            "mất mát không lồi F_t(theta)? Có hai nguyên nhân toán học cốt lõi. Thứ nhất, bề mặt của một hàm không lồi tổng quát chứa vô số điểm cực "
            "tiểu địa phương (local minima) và điểm yên ngựa (saddle points). Ngay cả trong bài toán tĩnh không có dịch chuyển phân phối, các thuật toán "
            "dựa trên đạo hàm bậc nhất như SGD cũng chỉ có thể bảo đảm hội tụ về một điểm dừng (stationary point nơi đạo hàm bằng 0) chứ không thể bảo "
            "đảm tìm được cực tiểu toàn cục trong thời gian đa thức dưới các giả thiết thông thường. Do đó, việc định nghĩa Dynamic Regret theo khoảng "
            "cách mất mát tới cực tiểu toàn cục ở mọi bước thời gian t là một tiêu chí quá mạnh và không khả thi về mặt lý thuyết. Thứ hai, khi phân phối "
            "P_t thay đổi, toàn bộ hình học của bề mặt mất mát không lồi bị biến dạng, các hố cực tiểu địa phương có thể xuất hiện hoặc biến mất, nên "
            "khoảng cách giữa hai điểm cực tiểu toàn cục không còn đại diện cho độ dịch chuyển của bài toán. Để khắc phục, bài báo đưa ra hai định nghĩa "
            "mới tại Công thức (22) và (23) trên slide: Một là định nghĩa lại tổng Dynamic Regret bằng tổng bình phương chuẩn Euclid của gradient hàm "
            "mất mát kỳ vọng tại chuỗi trọng số của bộ học: Reg(T) = sum_{t=1}^T ||grad F_t(theta_t)||_2^2. Thước đo này đánh giá mức độ gần điểm dừng "
            "của mô hình qua từng bước thời gian (cần lưu ý rõ rằng chuẩn gradient nhỏ không phân biệt được cực tiểu cục bộ, cực tiểu toàn cục hay điểm "
            "yên ngựa). Hai là định nghĩa độ dịch chuyển phân phối gamma_t theo chuẩn supremum toàn cục trên toàn bộ không gian tham số: gamma_t = "
            "sup_{theta} |F_t(theta) - F_{t+1}(theta)|, đo biên độ thay đổi tối đa của toàn bề mặt hàm mất mát giữa hai thời điểm liên tiếp."
        ),
    ),
    (
        28,
        "IV. NON-CONVEX LOSSES",
        "Khoa (TV1)",
        "Theorem 5.1 và lịch tốc độ học",
        (
            "Trên cơ sở định nghĩa Regret theo tổng bình phương chuẩn gradient ở Slide 27, Slide 28 trình bày Theorem 5.1 — định lý thiết lập cận trên "
            "cho kỳ vọng Regret không lồi và dẫn xuất công thức đóng cho tốc độ học thích ứng tại Công thức (25). Giả thiết của Theorem 5.1 giữ lại hai "
            "điều kiện chuẩn từ Assumption 4.1 (nhưng bỏ hoàn toàn giả thiết tính lồi và giả thiết miền tham số bị chặn D_max): hàm mất mát quần thể "
            "F_t(theta) >= 0 là hàm trơn L-smooth trên toàn không gian R^p, gradient mẫu đơn lẻ là ước lượng không chệch có phương sai bị chặn bởi "
            "sigma^2, và tốc độ học eta_t được chọn thích ứng theo lịch sử dữ liệu tới bước t-1 sao cho 0 < eta_t <= 1/L (để hệ số a_t = 2*eta_t - "
            "L*eta_t^2 > 0). Chứng minh của Theorem 5.1 (Phụ lục C.1) xuất phát từ bất đẳng thức Descent Lemma bậc hai của hàm L-smooth: "
            "F_t(theta_{t+1}) <= F_t(theta_t) + <grad F_t(theta_t), theta_{t+1} - theta_t> + (L/2)||theta_{t+1} - theta_t||_2^2. Thay quy tắc cập nhật "
            "Mini-batch SGD, lấy kỳ vọng có điều kiện trên lịch sử tới t-1 và dùng bất đẳng thức dịch chuyển bề mặt F_{t+1}(theta_{t+1}) - F_t(theta_{t+1}) "
            "<= gamma_t để nối chuỗi telescoping giữa F_t và F_{t+1}, tác giả thu được cận trên tổng quát tại Công thức (24). Để cực tiểu hóa cận trên "
            "này một cách tuần tự tại mỗi bước t, bộ học giải bài toán tối ưu một biến trên đoạn [0, 1/L] tại Công thức (25): cực tiểu hóa tổng của hạng "
            "tử mất mát và dịch chuyển [2*F_t(theta_t) + 2*gamma_t] / (2*eta - L*eta^2) cộng với hạng tử phương sai nhiễu mini-batch (L*sigma^2 / B_t) * "
            "eta^2 / (2*eta - L*eta^2). Đặt biến tổng hợp b_t = L * (gamma_t + F_t(theta_t)) >= 0. Giải phương trình đạo hàm bằng 0 trên miền lồi chặt "
            "(0, 1/L] cho ta nghiệm đóng duy nhất: eta*_t = (B_t / (L * sigma^2)) * [ sqrt(b_t^2 + 2*sigma^2*b_t / B_t) - b_t ] (cài đặt tại lớp "
            "`NonConvexSchedule` trong `core/schedules.py`). Công thức này chỉ ra rằng khi độ dịch chuyển gamma_t hoặc mất mát F_t(theta_t) tăng làm b_t "
            "tăng, tốc độ học tối ưu eta*_t cũng đồng biến tăng lên. Tuy nhiên, như điểm nhấn cuối slide phân định rõ: trong thực nghiệm mạng nơ-ron sâu "
            "ở Phụ lục E, tác giả dùng cơ chế thực nghiệm Adam Restart lấy cảm hứng từ nguyên lý này."
        ),
    ),
    (
        29,
        "IV. NON-CONVEX LOSSES",
        "Khoa (TV1)",
        "Flow Cytometry: dữ liệu và mô hình MLP",
        (
            "Slide 29 giới thiệu bài toán ứng dụng sinh học thực tế tại Phụ lục E (trang 22–23) của bài báo: Phân loại trực tuyến các dòng tế bào tuyến "
            "tụy bằng mạng nơ-ron đa tầng trên luồng dữ liệu Flow Cytometry, kèm theo biểu đồ nhúng 2 chiều UMAP (`figures/fig6_left_cytometry_embedding.png`) "
            "do nhóm tái lập từ bộ dữ liệu gốc. Về bối cảnh y sinh, kỹ thuật phân tích tế bào dòng chảy (Flow Cytometry) cần phân loại hàng nghìn tế "
            "bào đi qua ống dẫn laser theo thời gian thực dựa trên tín hiệu huỳnh quang RNA. Vì gắn quá nhiều chất đánh dấu huỳnh quang sẽ gây độc tế "
            "bào, hệ thống thực tế chỉ đo được một số ít gen chỉ thị (marker genes) và dùng một mạng nơ-ron nhỏ gọn (Student MLP) để dự đoán tức thời, "
            "trong khi nhận nhãn chuẩn phản hồi từ một mô hình Teacher quy mô lớn để học trực tuyến. Khi vị trí lấy mẫu dịch chuyển dọc theo mô tuyến "
            "tụy từ vùng đầu (anterior) sang vùng đuôi (posterior), tỷ lệ xuất hiện của các loại tế bào thay đổi mạnh theo từng giai đoạn. Để mô phỏng "
            "chính xác bài toán này, module `core/data.py` của nhóm xử lý bộ dữ liệu giải trình tự RNA đơn bào nội tiết tuyến tụy chuột từ thư viện "
            "`scvelo.datasets.pancreas()` (Bastidas-Ponce et al., 2019, gồm tổng cộng N = 3.696 tế bào). Sau khi áp dụng đúng bộ lọc "
            "`filter_genes(min_shared_counts=30000)` và chuẩn hóa `normalize_per_cell()` như mã gốc, dữ liệu thu về chính xác 10 marker genes đặc trưng "
            "(`Pyy, Meg3, Malat1, Gcg, Gnas, Actb, Ghrl, Rps3, Ins2, Hspa8`). Biểu đồ Figure 6 (left) ở bên phải hiển thị toàn bộ 3.696 tế bào trên mặt "
            "phẳng UMAP 1 và UMAP 2, cho thấy lộ trình phát triển sinh học rõ nét từ cụm tế bào ống dẫn `Ductal` (916 tế bào, màu xanh lá mạ bên trái) "
            "đi qua các cụm tiền thân nội tiết để phân nhánh sang ba dòng tế bào đảo tụy tiết hormone ở bên phải gồm `Alpha` (481 tế bào), `Beta` (591 "
            "tế bào) và `Delta` (70 tế bào). Mạng `Student MLP` (cài đặt tại `experiments/neural_net.py`) có kiến trúc 5 tầng tuyến tính `10 -> 64 -> 32 "
            "-> 16 -> 8 -> 4` với hàm kích hoạt ELU ở 4 tầng ẩn, hàm mất mát Categorical Cross-Entropy kèm điều chuẩn L2 (weight decay = 10^{-4}). Luồng "
            "huấn luyện trực tuyến gồm T = 600 bước với kích thước batch B_t = 64 đi qua 4 pha phân phối lớp thay đổi đột ngột tại các mốc thời gian "
            "t = 100, t = 200 và t = 400."
        ),
    ),
    (
        30,
        "IV. NON-CONVEX LOSSES",
        "Khoa (TV1)",
        "Adam Restart và thiết kế thực nghiệm",
        (
            "Slide 30 trình bày cơ chế hoạt động của thuật toán Adam Restart, thiết kế thực nghiệm đối chứng cho Figure 6 và một phát hiện kiểm toán mã "
            "nguồn quan trọng trong tệp `src/cytometry/main.py` của tác giả (`docs/original_code_comparison.md`). Trong huấn luyện mạng nơ-ron sâu, bộ "
            "tối ưu Adam duy trì hai bộ đệm trạng thái theo thời gian cho từng tham số trọng số: trung bình trượt hàm mũ của gradient (moment bậc nhất "
            "m_t) và trung bình trượt hàm mũ của bình phương gradient (moment bậc hai v_t), cùng bộ đếm bước thời gian để hiệu chỉnh độ chệch (bias "
            "correction). Trong phương pháp `Adam` tiêu chuẩn, các trạng thái moment này được giữ liên tục xuyên suốt toàn bộ T = 600 bước. Tuy nhiên, "
            "sau hàng trăm bước học trên một phân phối cũ (ví dụ Pha 1 và Pha 2 chủ yếu chỉ có tế bào Ductal), bộ tích lũy bình phương gradient v_t ở "
            "mẫu số đã lớn lên và bộ đếm bước đã cao, làm bước nhảy hiệu dụng bị thu nhỏ lại và giữ mô hình trong hố cực tiểu cục bộ của phân phối cũ. "
            "Để mô phỏng nguyên lý 'tăng tốc độ học khi có dịch chuyển phân phối' cho bộ tối ưu Adam, phương pháp `Adam Restart` giữ nguyên trọng số mô "
            "hình hiện tại theta_t nhưng khởi tạo lại (reset) toàn bộ trạng thái bộ tối ưu (đưa m_t = 0, v_t = 0 và bộ đếm bước về 0) ngay tại các mốc "
            "thời gian xảy ra chuyển pha phân phối t in {100, 200, 400}, giúp khôi phục tức thời bước nhảy hiệu dụng lớn như giai đoạn đầu huấn luyện. "
            "Về thiết kế đối chứng, nhóm chạy 4 phương pháp (`Adam 0.01`, `Restart 0.01`, `Adam 0.03`, `Restart 0.03`) nhân với 3 seeds `0, 1, 2` (tổng "
            "cộng 12 runs xác thực trên W&B) qua T = 600 bước với batch huấn luyện và đánh giá cùng có kích thước 64. Thước đo thực nghiệm ở đây là tổng "
            "hàm mất mát kiểm định tích lũy (cumulative validation cross-entropy loss) sau cập nhật chứ không phải tổng chuẩn gradient của Công thức (22). "
            "Đặc biệt, khi kiểm toán dòng 144 của `src/cytometry/main.py`, nhóm phát hiện trên mảng 6 khối 100 bước `distribution_types = [0, 1, 2, 2, 3, 3]`, "
            "tác giả viết nhầm điều kiện `distribution_type == distribution_types[phase - 1]` (dùng `==` thay vì `!=`), khiến code gốc restart ở giữa "
            "Pha 3 và Pha 4 (t = 300, 500) thay vì tại mốc chuyển pha t = 100, 200, 400 như mô tả trong bài báo! Nhóm báo cáo rạch ròi cả hai quy tắc này "
            "trong phần kiểm toán nguồn."
        ),
    ),
    (
        31,
        "IV. NON-CONVEX LOSSES",
        "Khoa (TV1)",
        "Figure 6: kết quả Adam và Adam Restart",
        (
            "Slide 31 trình bày kết quả định lượng và đồ thị tái lập Figure 6-right (`figures/fig6_right_adam_restart_regret.png` ở bên phải và bảng "
            "tổng hợp trung bình ± độ lệch chuẩn trên 3 seeds từ `results/nn/fig6_summary.csv` ở bên trái, tương ứng với 12 runs chính thức trên W&B "
            "Public). Đồ thị biểu diễn đường cong Cumulative Validation Loss (mất mát kiểm định tích lũy) theo thời gian t từ bước 1 đến bước 600 của "
            "bốn cấu hình, với ba đường kẻ dọc nét đứt đánh dấu ba thời điểm dịch chuyển phân phối giữa các pha tế bào tại t = 100 (từ 100% Ductal sang "
            "xuất hiện 10% Delta), t = 200 (tăng lên 10% Alpha, 20% Delta) và t = 400 (đảo chiều mạnh sang 30% Alpha, 50% Beta, 10% Delta và chỉ còn 10% "
            "Ductal). Bảng số liệu và đồ thị xác nhận rõ hai kết luận thực nghiệm cốt lõi của Phụ lục E.2: Thứ nhất, tại cả hai mức tốc độ học khởi tạo "
            "eta_0 = 0.01 và eta_0 = 0.03, cơ chế `Adam Restart` đều giúp giảm đáng kể tổng mất mát kiểm định tích lũy và giảm phương sai giữa các hạt "
            "giống so với `Adam` tiêu chuẩn không restart. Cụ thể, tại mức tốc độ học tối ưu eta_0 = 0.01, `Restart 0.01` (đường màu cam, W&B runs "
            "`3beskj3q`, `lectj13d`, `jfd7ui7f`) đạt tổng mất mát tích lũy thấp nhất toàn cục là 131.59 ± 5.07, giảm 12.0% so với mức 149.52 ± 10.69 của "
            "`Adam 0.01` (đường màu xanh dương, runs `4p6kun1x`, `p1rgnctj`, `7dvnzczn`). Thứ hai, hiệu quả giải cứu mô hình của cơ chế Restart thể hiện "
            "rõ rệt nhất ở mức tốc độ học lớn hơn eta_0 = 0.03: như đường màu xanh lá (`Adam 0.03`, runs `h91vgmz5`, `yvc2rruz`, `h1qd3lha`) trên đồ thị "
            "cho thấy, trong 200 bước đầu tiên khi dữ liệu chủ yếu chỉ có lớp Ductal, `Adam 0.03` hội tụ rất sâu vào cực tiểu cục bộ của lớp đó; nhưng "
            "ngay khi bước sang Pha 3 (t = 200) và Pha 4 (t = 400) có nhiều tế bào Alpha và Beta, `Adam 0.03` không có restart bị kẹt cứng trong hố cực "
            "tiểu cũ khiến mất mát tích lũy tăng vọt theo đường thẳng dốc đứng lên tới 315.07 ± 75.71. Ngược lại, nhờ xóa bộ đệm moment tại các mốc "
            "shift, `Restart 0.03` (đường màu đỏ, runs `jzrtmtez`, `trjq8r5m`, `bion3ocx`) lập tức thoát khỏi cực tiểu cũ và kéo tổng mất mát tích lũy "
            "xuống chỉ còn 180.30 ± 26.49 (giảm tới 42.8%!). Kết quả này xác nhận lợi ích của Restart ở hai mức LR đã khảo sát trên luồng dữ liệu 4 pha."
        ),
    ),
    (
        32,
        "V. GLOBAL SYNTHESIS & CONCLUSION",
        "Khôi (TV2)",
        "So sánh các kết quả lý thuyết",
        (
            "Bước sang Phần V — Global Synthesis & Conclusion (từ Slide 32 đến Slide 36), Slide 32 đặt toàn bộ các kết quả lý thuyết của bài báo qua ba "
            "chế độ hàm mất mát (Linear Regression ở Mục 3, General Convex ở Mục 4 và Non-convex ở Mục 5) lên cùng một bảng ma trận đối chiếu 5 tiêu chí "
            "(chi tiết tại `docs/theorems.md`). Bảng tổng hợp này làm nổi bật sự tiến hóa về công cụ toán học khi các giả thiết cấu trúc của hàm mất mát "
            "được nới lỏng dần. Về tiêu chí Mục tiêu (Objective): ở bài toán Linear, nhờ cấu trúc bậc hai và đặc trưng chuẩn tắc N(0, I_d), rủi ro dư "
            "quần thể (excess loss) tỷ lệ chính xác với bình phương khoảng cách Euclid 1/2 ||theta_t - theta*_t||_2^2; sang bài toán Convex tổng quát, "
            "mục tiêu là Dynamic Regret chuẩn sum [l_bar_t(theta_t) - l_bar_t(theta*_t)] so với chuỗi nghiệm tối ưu quần thể tức thời; còn sang bài toán "
            "Non-convex, do không thể bảo đảm hội tụ toàn cục, mục tiêu phải chuyển thành tổng bình phương chuẩn gradient sum ||grad F_t(theta_t)||_2^2 "
            "nhằm đo mức độ tiệm cận các điểm dừng. Về tiêu chí Công cụ phân tích (Analytical Tools) và Kết quả (Theorems): bài toán Linear sử dụng xấp "
            "xỉ phương trình vi phân ngẫu nhiên SDE (Proposition 3.1), rút gọn bằng Bổ đề Stein và Itô về hệ Moment ODE chính xác (Theorem 3.2) rồi giải "
            "bài toán điều khiển tối ưu liên tục HJB trên hàm bao trên (Theorem 3.3); bài toán Convex sử dụng bất đẳng thức co-coercivity và phép chiếu "
            "không giãn trong miền rời rạc để thiết lập cận trên telescoping (Theorem 4.2), cận dưới lồi mạnh (Theorem 4.7) và giải bài toán tối ưu lồi "
            "hai nhánh có điểm gãy (Proposition 4.3); trong khi bài toán Non-convex sử dụng trực tiếp bất đẳng thức trơn Descent Lemma bậc hai để chặn "
            "tổng bình phương chuẩn gradient (Theorem 5.1). Về Thuật toán và Giới hạn áp dụng: cả ba thuật toán (Algorithm 1, Lịch kẹp hai ngưỡng "
            "[tau_{1,t}, tau_{2,t}] và Công thức đóng Eq. 25) đều dẫn đến cùng một quy luật điều khiển thống nhất là nâng tốc độ học khi có dịch chuyển "
            "lớn, nhưng mỗi kết quả chỉ hợp lệ trong phạm vi giả thiết của nó (mô hình Gaussian và bước nhảy nhỏ eps cho Linear; miền bị chặn D_max và "
            "phương sai hữu hạn cho Convex; và chỉ bảo đảm điểm dừng cục bộ cho Non-convex). Đặc biệt, vì ba định nghĩa Regret có đơn vị và bản chất "
            "toán học khác nhau nên tuyệt đối không dùng trị số tuyệt đối để xếp hạng chéo giữa ba nhóm."
        ),
    ),
    (
        33,
        "V. GLOBAL SYNTHESIS & CONCLUSION",
        "Khôi (TV2)",
        "Độ phức tạp tính toán và bộ nhớ",
        (
            "Slide 33 tổng hợp bức tranh toàn diện về Độ phức tạp Tính toán và Bộ nhớ (Computational & Space Complexity, chi tiết tại `docs/06_complexity.md`) "
            "bằng cách tách bạch chi phí của bản thân quy tắc cập nhật tốc độ học (scheduler overhead) với chi phí của bộ ước lượng độ trôi, vòng lặp "
            "huấn luyện SGD và quy trình đánh giá thực nghiệm. Nhìn vào các dòng đầu của bảng: Khi bộ lập lịch đã được cung cấp sẵn các đại lượng vô "
            "hướng đầu vào (như độ trôi gamma_t và giá trị mất mát), thì lịch trình hằng số (`Constant`), lịch suy giảm `Inverse-sqrt`, lịch hai ngưỡng "
            "`Convex` (Proposition 4.3) và công thức đóng `Non-convex` (Theorem 5.1, Công thức 25) đều chỉ thực hiện một số lượng hằng số các phép toán "
            "số học và căn bậc hai trên số thực đơn lẻ, do đó có độ phức tạp thời gian và bộ nhớ phụ mỗi bước là Theta(1) tuyệt đối. Đối với `Linear "
            "Algorithm 1`, mỗi bước thời gian thực hiện vòng lặp tích phân Euler gồm K = ceil(1 / kappa) bước con trên một biến trạng thái vô hướng v "
            "duy nhất, nên chi phí riêng của bộ lập lịch là Theta(K) thời gian và Theta(1) bộ nhớ (với kappa = 0.1 cố định thì K = 10 cũng là hằng số "
            "độc lập với số chiều d). Tuy nhiên, trong triển khai trực tuyến thực tế khi không có sẵn độ trôi Oracle và phải dùng bộ ước lượng chuyển "
            "động của mô hình `Learner-motion EMA` (gamma_hat_t = beta*gamma_hat_{t-1} + (1 - beta)||theta_t - theta_{t-1}||_2), hệ thống cần lưu vector "
            "trọng số bước trước theta_{t-1} và tính chuẩn Euclid của hiệu hai vector d chiều, tiêu tốn chi phí thời gian và bộ nhớ là Theta(d) mỗi bước "
            "(tương đương chi phí Theta(d) của phép chiếu hình cầu `Ball projection` nếu kích hoạt). Khi đặt cạnh chi phí huấn luyện `Dense logistic / "
            "linear training` trên toàn bộ T bước với kích thước batch B là Theta(T * B * d) phép toán nhân ma trận-vector (và bộ nhớ Theta(T * d) nếu "
            "lưu lại toàn bộ lịch sử quỹ đạo trọng số qua T bước để phục vụ hậu kiểm), ta thấy chi phí cộng thêm của bộ lập lịch thích ứng là cực kỳ "
            "nhỏ (nhỏ hơn B lần so với bước tính gradient mini-batch). Ghi chú cuối bảng chỉ rõ rằng trong thực nghiệm của dự án, thời gian chạy chủ "
            "yếu nằm ở khâu hậu kiểm đánh giá rủi ro dư quần thể chính xác (tích phân cầu phương 1 chiều và giải nghiệm gốc bán kính) hoặc lan truyền "
            "qua mạng nơ-ron."
        ),
    ),
    (
        34,
        "V. GLOBAL SYNTHESIS & CONCLUSION",
        "Khoa (TV1)",
        "Mức độ tái lập và sai lệch triển khai",
        (
            "Slide 34 tổng kết mức độ tái lập thực nghiệm đã được xác minh trên cả ba mô-đun của dự án và hệ thống hóa các sai lệch triển khai (implementation "
            "discrepancies) được phát hiện qua quá trình kiểm toán mã nguồn gốc của tác giả (`docs/original_code_comparison.md` và `docs/convex_author_v1/SOURCE_AUDIT.md`). "
            "Ở cột thứ nhất — Mô-đun Linear Regression: nhóm đã tái lập trọn vẹn Figure 4 (3 cấu hình ell in {1, 4, 16}) và Figure 5-top (5 số chiều "
            "d in {2, 8, 32, 128, 512} trên 3 seeds), với đầy đủ 18 runs chính thức (cộng 1 smoke run) được xác thực trực tuyến trên W&B Public và khớp "
            "với đồ thị trong bài báo dưới 1.2% sai số tương đối, kèm theo bộ đối chiếu mở rộng với EMA và các lịch trình baseline lưu trong CSV. Ở cột "
            "thứ hai — Mô-đun General Convex Losses: đây là mô-đun được nhóm kiểm chứng sâu nhất qua nhiều tầng giao thức độc lập, bao gồm 60 runs trong "
            "thiết lập ban đầu (45 runs CP04 + 15 runs độ nhạy CP05), tái lập trực tiếp notebook TensorFlow của tác giả trên cả 5 số chiều, kiểm chứng "
            "khớp từng bước trên cùng dữ liệu đầu vào (shared-input parity) qua 2000 bước tại d = 2, 8, 32 với độ lệch tích lũy tối đa chỉ 2.48 x 10^{-5}, "
            "45 runs tái lập độc lập đa hạt giống theo thiết kế tác giả, và 36 runs can thiệp có đối chứng (controlled ablations), trong đó toàn bộ "
            "88/88 runs bổ sung trong gói kiểm chứng nguồn [6] đều được đối chiếu khớp tuyệt đối với lịch sử W&B. Ở cột thứ ba — Mô-đun Non-convex "
            "Losses: nhóm đã hoàn tất 12 runs chính thức của Figure 6 (4 phương pháp x 3 seeds) trên dữ liệu thực 3.696 tế bào tuyến tụy 10-gene và đẩy "
            "đầy đủ lên W&B Public. Đặc biệt, kiểm toán mã nguồn đã phát hiện sự khác biệt trực tiếp giữa văn bản bài báo và code gốc tại dòng 144 của "
            "`src/cytometry/main.py`: bài báo mô tả khởi động lại Adam khi phân phối chuyển pha tại t = 100, 200, 400 (`dist[phase] != dist[phase-1]`), "
            "nhưng code gốc lại dùng toán tử `==` làm mô hình restart tại t = 300 và 500; nhóm đã định lượng độc lập cả hai quy tắc này trong CSV để "
            "chứng minh việc sửa lại đúng quy tắc chuyển pha của bài báo giúp giảm tới 40.4% mất mát tích lũy ở LR = 0.03. Ghi chú cuối slide khẳng "
            "định kỷ luật khoa học của nhóm: mọi phép so sánh chỉ hợp lệ khi đồng nhất thước đo, bộ so sánh và luồng dữ liệu."
        ),
    ),
    (
        35,
        "V. GLOBAL SYNTHESIS & CONCLUSION",
        "Khoa (TV1)",
        "Kết luận, giới hạn và hướng mở rộng",
        (
            "Slide 35 đúc kết toàn bộ các kết luận khoa học cốt lõi của báo cáo giữa kỳ, chỉ ra những giới hạn thực tế đã được nhóm chứng minh bằng dữ "
            "liệu, và đề xuất các hướng nghiên cứu mở rộng có cơ sở cho giai đoạn cuối kỳ (Milestone 3 và Lab 2). Về Kết luận chung: bài báo Fahrbach "
            "et al. (ICML 2023) đã xây dựng thành công một khung lý thuyết sử dụng ba công cụ phân tích khác nhau — xấp xỉ SDE và điều khiển tối ưu HJB "
            "cho hồi quy tuyến tính, cận trên telescoping và quy tắc hai ngưỡng cho hàm lồi tổng quát, cùng cận chuẩn gradient và cơ chế Adam Restart "
            "cho hàm không lồi — để dẫn xuất các lịch trình tốc độ học thích ứng theo dịch chuyển phân phối. Kết quả tái lập thực nghiệm của nhóm xác "
            "nhận lợi ích rõ rệt của việc chủ động tăng tốc độ học (hoặc khởi động lại bộ đệm moment) khi môi trường dịch chuyển trên phần lớn các thiết "
            "lập đã kiểm tra, nhưng đồng thời chỉ ra rằng phương pháp này không bảo đảm ưu thế phổ quát trong mọi chế độ tham số. Cụ thể, về Giới hạn: "
            "phát hiện nổi bật nhất từ thực nghiệm đối chứng của nhóm là tại số chiều cao d = 512 trong bài toán Hồi quy Logistic, lịch trình hai ngưỡng "
            "`Convex` chịu rủi ro dư quần thể cao hơn đáng kể so với lịch tốc độ học hằng số `Constant` (eta = 0.1) trên cả hai giao thức P và I. Qua 36 "
            "runs can thiệp có đối chứng, nhóm đã chỉ rõ rằng hiệu năng của lịch trình thích ứng phụ thuộc nhạy cảm vào sự tương thích giữa nguồn thông "
            "tin độ trôi (oracle drift nhìn trước hay ước lượng nhân quả EMA), hiện tượng đóng băng khi khởi tạo lạnh (cold start), mức độ nới lỏng quá "
            "mức của đường kính miền lý thuyết D_max = d ở chiều cao, cũng như sự thay đổi của nghiệm tối ưu quần thể dưới nhiễu nhãn. Từ những giới hạn "
            "định lượng này, nhóm đề xuất ba hướng mở rộng cụ thể cho giai đoạn tiếp theo: (1) Thiết kế bộ ước lượng độ trôi nhân quả có khử nhiễu kết "
            "hợp khởi tạo ấm (warm start) và hiệu chỉnh đường kính hiệu dụng D_eff << d cho số chiều lớn; (2) Xây dựng giao thức cầu nối (bridge "
            "protocol) giữa thiết lập P và I; và (3) Mở rộng kiểm chứng sang bài toán Covariate Shift."
        ),
    ),
    (
        36,
        "V. GLOBAL SYNTHESIS & CONCLUSION",
        "Khoa (TV1)",
        "Phân công và tài nguyên nghiên cứu",
        (
            "Slide 36 khép lại báo cáo giữa kỳ bằng bảng ghi nhận trung thực đóng góp thực tế của các thành viên trên từng trang slide và từng mô-đun "
            "kỹ thuật, kèm theo ba liên kết tài nguyên công khai để giảng viên và hội đồng chấm bài có thể truy cập kiểm tra trực tiếp. Trong quá trình "
            "thực hiện đồ án, do thành viên thứ ba (Hòa — TV3) không hoàn thành nhiệm vụ được giao (0% đóng góp về mã nguồn, thực nghiệm và slide), toàn "
            "bộ khối lượng công việc của dự án đã được hai thành viên là Khoa (Trưởng nhóm — TV1) và Khôi (TV2) phân chia đảm nhận và hoàn thiện 100%: "
            "Cụ thể, Thành viên thứ nhất — Khoa (Trưởng nhóm, phụ trách 22/36 slides) thực hiện toàn bộ Phần II — Linear Regression từ lý thuyết SDE, "
            "Moment ODE, điều khiển tối ưu HJB, cài đặt Algorithm 1 cho đến tái lập thực nghiệm Figure 4 và Figure 5-top (Slides 10 đến 16); đồng thời "
            "trực tiếp thực hiện thay toàn bộ Phần IV — Non-convex Losses từ lý thuyết chuẩn gradient, Theorem 5.1, xử lý bộ dữ liệu tế bào tụy `scvelo` "
            "10-gene (`core/data.py`), cài đặt mạng MLP 5 tầng với Adam Restart (`experiments/neural_net.py`) cho đến 12 runs W&B tái lập Figure 6 "
            "(Slides 27 đến 31), cùng các trang nền tảng Phần I (Slides 01, 02, 03, 04, 06, 07, 08) và tổng kết Phần V (Slides 34, 35, 36). Thành viên "
            "thứ hai — Khôi (TV2, phụ trách 14/36 slides) thực hiện toàn bộ Phần III — General Convex Losses từ lý thuyết cận Dynamic Regret, quy tắc hai "
            "ngưỡng Proposition 4.3, hiệu chỉnh nghiệm tối ưu quần thể bằng tích phân cầu phương, kiểm chứng khớp từng bước với notebook tác giả, tái lập "
            "độc lập đa hạt giống cho đến phân tích độ nhạy và 36 runs controlled ablations (Slides 17 đến 26), đồng thời biên soạn phần đóng góp, khung "
            "tái lập kiểm chứng (Slides 05, 09), bảng so sánh lý thuyết và độ phức tạp tính toán (Slides 32, 33). Toàn bộ mã nguồn, bộ kiểm thử tự động, "
            "bảng số liệu CSV, biểu đồ và lịch sử chạy thực nghiệm đều đã được công khai tại GitHub Repository và Weights & Biases Dashboard trên slide. "
            "Nhóm chúng em xin chân thành cảm ơn thầy đã theo dõi!"
        ),
    ),
]


def main() -> None:
    header_lines = [
        "# Bộ Speaker Notes Hoàn Chỉnh Cho 36 Slides Canva (Chuẩn <= 500 từ/slide)",
        "",
        "Tài liệu này khớp **1–1 theo đúng thứ tự, tiêu đề, công thức và số liệu của 36 trang Slide trên bản Canva chính thức của nhóm** (`https://www.canva.com/design/DAHbeQ6df_I/vVfz2Am-T9QGhPY_9PvQvw/edit`).",
        "- Độ dài mỗi trang: **330 – 475 từ/slide** (tuân thủ tuyệt đối giới hạn **<= 500 từ/slide**).",
        "- Cách dùng trên Canva: Mở từng trang từ **Slide 1** đến **Slide 36**, bấm nút **`📝 Ghi chú`** ở góc dưới bên trái màn hình Canva và dán đoạn văn bản trong khung ` ```text ... ``` ` tương ứng vào.",
        "",
        "| Trang | Phần (Section) | Phụ trách | Tiêu đề Slide trên Canva | Số từ (Words) | Kiểm tra (<= 500 từ) |",
        "| :---: | :--- | :---: | :--- | :---: | :---: |",
    ]

    body_lines: List[str] = ["", "---", ""]

    for s_num, section, owner, title, raw_note in CANVA_36_SLIDES:
        speaker_tag = f"[Người trình bày: {owner} | Slide {s_num:02d}/36 — {section} | {title}]"
        note = f"{speaker_tag}\n\n{raw_note}"
        wc = len(note.split())
        if not (250 <= wc <= 490):
            raise ValueError(f"Slide {s_num:02d} ({title}) có {wc} từ, nằm ngoài ngưỡng an toàn [250, 490]!")
        header_lines.append(
            f"| **Slide {s_num:02d}** | `{section}` | **{owner}** | {title} | **{wc} từ** | ĐẠT (✔) |"
        )
        body_lines.extend(
            [
                f"## Slide {s_num:02d} — {title} *(Người nói: {owner})*",
                f"- **Phần**: `{section}` | **Người trình bày**: **{owner}** | **Tổng số từ (đã tính cả dòng tên người nói)**: **`{wc} / 500 từ`**",
                "",
                "```text",
                note,
                "```",
                "",
            ]
        )

    full_md = "\n".join(header_lines + body_lines)

    out_canva_36 = PROJECT_ROOT / "docs" / "SPEAKER_NOTES_36_SLIDES_CANVA.md"
    out_canva_36.write_text(full_md, encoding="utf-8")
    print(f"[OK] Wrote 36-slide Canva Speaker Notes to: {out_canva_36}")

    # Cập nhật luôn file docs/SPEAKER_NOTES_32_SLIDES.md để tránh nhầm lẫn nếu mở file cũ
    out_legacy = PROJECT_ROOT / "docs" / "SPEAKER_NOTES_32_SLIDES.md"
    out_legacy.write_text(full_md, encoding="utf-8")
    print(f"[OK] Synced 36-slide Canva Speaker Notes to: {out_legacy}")


if __name__ == "__main__":
    main()
