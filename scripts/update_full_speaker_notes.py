r"""
================================================================================
FILE: scripts/update_full_speaker_notes.py
MỤC ĐÍCH:
    Xây dựng bộ Speaker Notes chi tiết, chuyên sâu (240 - 380 từ/slide, đảm bảo
    nghiêm ngặt <= 500 từ/slide theo đúng yêu cầu của đề bài khi chấm không thuyết
    trình miệng) cho toàn bộ 32 slides của nhóm:
    1. Xuất ra file `docs/SPEAKER_NOTES_32_SLIDES.md` (kèm thống kê số từ từng slide)
       để nhóm copy-paste nhanh vào ô Notes của Canva hoặc PowerPoint.
    2. Cập nhật trực tiếp vào `slides/template.pptx` và kiểm tra tự động mọi tiêu chí
       Checklist trước khi nộp Moodle.
================================================================================
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_template_pptx import SLIDES_SPEC, generate_template_pptx

FULL_SPEAKER_NOTES = {
    1: (
        "Chào thầy và hội đồng chấm bài. Đây là bài báo cáo giữa kỳ kết hợp Milestone 2 (Đọc hiểu bài báo) và Lab 1 "
        "(Tái hiện thực nghiệm) của nhóm chúng em, gồm ba thành viên: Khoa (Trưởng nhóm, phụ trách bài toán Hồi quy tuyến tính "
        "và phân tích phương trình vi phân ngẫu nhiên SDE), Khôi (phụ trách bài toán Hàm lồi tổng quát, phân tích độ nhạy và "
        "hạ tầng theo dõi thực nghiệm Weights & Biases), và Hòa (phụ trách bài toán Hàm không lồi trên mạng nơ-ron sâu Flow Cytometry "
        "cùng đối chiếu mã nguồn gốc). Công trình chúng em nghiên cứu là bài báo 'Learning Rate Schedules in the Presence of "
        "Distribution Shift' của nhóm tác giả Matthew Fahrbach, Adel Javanmard, Vahab Mirrokni và Pratik Worah từ Google Research "
        "và Đại học Nam California, được công bố tại hội nghị ICML 2023 [1]. Câu hỏi xuyên suốt bài báo đặt ra là: Trong các hệ thống "
        "học máy trực tuyến hoạt động liên tục (never-ending online learning), khi phân phối dữ liệu đầu vào thay đổi theo thời gian, "
        "ta nên điều chỉnh tốc độ học (learning rate) của thuật toán Stochastic Gradient Descent (SGD) như thế nào để duy trì sai số "
        "thấp nhất trong suốt vòng đời hệ thống? Trái ngược hoàn toàn với trực giác kinh điển là phải giảm dần tốc độ học về không để "
        "hội tụ, bài báo chứng minh cả về mặt toán học lẫn thực nghiệm rằng: khi cảnh quan hàm mất mát dịch chuyển, lịch trình tốc độ "
        "học tối ưu phải TĂNG LÊN để mở rộng khả năng khám phá không gian trọng số và kịp thời bám đuổi nghiệm mới. Trong báo cáo này, "
        "nhóm chúng em thực hiện phạm vi Full Scope 100%, tái hiện trọn vẹn cả ba lớp bài toán lý thuyết và thực nghiệm của bài báo, "
        "đồng thời kiểm chứng độc lập toàn bộ số liệu trên hệ thống Weights & Biases công khai."
    ),
    2: (
        "Slide thứ hai trình bày cấu trúc tổng thể gồm 32 trang của bài báo cáo, được thiết kế thành năm phần gắn kết chặt chẽ giữa "
        "việc kể lại câu chuyện khoa học của bài báo bằng ngôn ngữ riêng của nhóm (Milestone 2) và bằng chứng tái lập thực nghiệm "
        "(Lab 1). Ở Phần 1 từ Slide 3 đến Slide 7, chúng em đi từ mâu thuẫn giữa học tĩnh truyền thống và học trực tuyến có trôi dạt "
        "phân phối, minh họa bằng một bài toán công nghiệp cụ thể là hệ thống dự đoán tỷ lệ nhấp chuột (CTR) thời gian thực trên sàn "
        "thương mại điện tử, sau đó phân tích ba nhóm phương pháp tiền nhiệm để chỉ ra khoảng trống nghiên cứu mà bài báo ICML 2023 "
        "lấp đầy. Sang Phần 2 ở Slide 8 và Slide 9, chúng em hệ thống hóa ba đóng góp cốt lõi của tác giả và phân loại tính mới theo "
        "ba trục: lý thuyết, kỹ thuật và thực nghiệm. Phần 3 từ Slide 10 đến Slide 21 là trái tim kỹ thuật của bài báo cáo, đi qua đủ "
        "năm lớp phương pháp theo đúng hướng dẫn đọc hiểu: từ bảng ký hiệu thống nhất và bài toán tối ưu độ tiếc nuối động (Dynamic Regret), "
        "sơ đồ pipeline hai giai đoạn tách bạch giữa suy luận trực tuyến và huấn luyện thích ứng, cho đến chi tiết từng mô-đun toán học "
        "của hồi quy tuyến tính (SDE và điều khiển tối ưu Hamilton-Jacobi-Bellman), hàm lồi tổng quát (quy tắc kẹp hai ngưỡng Proposition 4.3), "
        "hàm không lồi (chuẩn gradient Theorem 5.1), bảng tổng hợp bảy định lý kèm giới hạn giả thiết, và phân tích độ phức tạp tính toán O(1). "
        "Phần 4 từ Slide 22 đến Slide 30 trình bày toàn bộ kết quả tái hiện Full Scope cho Figure 4, Figure 5-top, Figure 5-bottom, Figure 6, "
        "phân tích độ nhạy mở rộng và kiểm toán mã nguồn gốc. Cuối cùng, Phần 5 tổng kết các hạn chế lý thuyết làm tiền đề cho Milestone 3."
    ),
    3: (
        "Mở đầu Phần 1, chúng ta đặt câu hỏi: Tại sao các lịch trình giảm tốc độ học vốn rất thành công trong huấn luyện ngoại tuyến "
        "(offline training) lại sụp đổ trong môi trường học trực tuyến không ngừng? Trong học máy cổ điển, ta giả định dữ liệu được lấy "
        "mẫu độc lập cùng phân phối từ một phân phối tĩnh P cố định. Khi đó, điểm cực tiểu toàn cục theta* đứng yên, nên việc giảm dần "
        "bước nhảy eta_t về 0 là bắt buộc để triệt tiêu phương sai nhiễu của mini-batch gradient và giúp mô hình hội tụ chính xác vào theta*. "
        "Tuy nhiên, trong các hệ thống quy mô lớn ngoài đời thực [1], dữ liệu đến theo luồng phân phối biến thiên P_t, kéo theo nghiệm tối "
        "ưu tức thời theta*_t dịch chuyển liên tục trên không gian tham số với biên độ gamma_t = ||theta*_t - theta*_{t+1}||_2 ở mỗi bước. "
        "Nếu bộ học vẫn giảm eta_t -> 0 theo thời gian, trọng số mô hình theta_t sẽ dần mất khả năng di chuyển và bị đóng băng ở vùng "
        "nghiệm cũ đã lỗi thời. Hình bên phải do nhóm tự tái hiện từ thiết lập Figure 1 của bài báo (`figures/fig1_constant_lr_trajectories.png`) "
        "minh họa trực quan hiện tượng này trên mặt phẳng 2 chiều: đường xoắn ốc màu xanh dương là quỹ đạo nghiệm tối ưu theta*_t xuất phát "
        "từ điểm (1, 0) và nhảy bậc theo chiều kim đồng hồ sau mỗi 100 bước; đường màu cam là quỹ đạo trọng số theta_t của SGD với kích thước "
        "batch B_t = 1 dưới bốn mức tốc độ học hằng số 0.003, 0.01, 0.03 và 0.1. Ta thấy rõ: với tốc độ học nhỏ eta = 0.003 (ngoài cùng bên trái), "
        "quỹ đạo SGD rất mượt nhưng luôn bị tụt lại rất xa phía sau đích ngắm, chịu tổng độ tiếc nuối lên tới 2127. Ngược lại, tốc độ học lớn "
        "eta = 0.1 (ngoài cùng bên phải) tuy nhìn dao động mạnh quanh đích ngắm nhưng lại bám đuổi kịp thời mỗi khi theta*_t nhảy bậc, nhờ đó "
        "đạt tổng độ tiếc nuối thấp nhất (chỉ 273). Điều này đặt ra bài toán: làm sao tự động điều chỉnh eta_t tăng giảm đúng lúc theo độ trôi?"
    ),
    4: (
        "Để thấy rõ tầm quan trọng thực tiễn của bài toán ngoài phạm vi mô phỏng học thuật, nhóm chúng em xây dựng một kịch bản công nghiệp "
        "điển hình tại tài liệu `docs/motivation_example.md`: Hệ thống dự đoán tỷ lệ nhấp chuột (Click-Through Rate - CTR Prediction) và gợi ý "
        "sản phẩm thời gian thực trên các sàn thương mại điện tử lớn như Shopee hoặc TikTok Shop. Trong hệ thống này, ở mỗi bước thời gian t, "
        "máy chủ nhận một mini-batch gồm B_t lượt hiển thị quảng cáo z_{t,k} = (x_{t,k}, y_{t,k}), trong đó x_{t,k} là vector đặc trưng ghép nối "
        "giữa hồ sơ người dùng và sản phẩm, còn y_{t,k} là nhãn phản hồi nhấp chuột hay mua hàng. Mô hình bắt buộc phải đưa ra dự đoán xác suất "
        "trước để xếp hạng hiển thị, sau đó mới thu thập nhãn phản hồi thực tế từ người dùng để cập nhật trọng số theta_t -> theta_{t+1} bằng "
        "Mini-batch SGD. Trong vận hành thực tế, luồng dữ liệu này chịu hai kiểu trôi dạt phân phối rất khác nhau như sơ đồ bên phải minh họa. "
        "Thứ nhất là trôi dạt mượt (Smooth hoặc Gradual Drift): hành vi mua sắm chuyển dịch từ từ theo khung giờ trong ngày (buổi sáng tìm đồ ăn "
        "và văn phòng phẩm, buổi tối chuyển sang thời trang và giải trí) hoặc chuyển mùa, làm nghiệm tối ưu theta*_t trôi liên tục với tốc độ "
        "gamma_t nhỏ nhưng khác không ở mọi bước. Thứ hai là trôi dạt đột ngột theo đợt (Bursty hoặc Abrupt Shift): đúng vào thời khắc 0h00 của "
        "chiến dịch Flash Sale ngày đôi (11/11, 12/12) hoặc khi một phiên livestream của KOL bùng nổ, phân phối sở thích khách hàng nhảy vọt sang "
        "một chế độ hoàn toàn mới, khiến gamma_t tăng đột biến tại đầu phiên rồi giữ ổn định trong suốt phiên đó. Nếu dùng lịch trình giảm dần "
        "1/t truyền thống, sau vài ngày vận hành eta_t đã tiến sát về 0; khi Flash Sale 0h00 nổ ra, mô hình hoàn toàn tê liệt không thể dịch chuyển "
        "trọng số sang xu hướng mới, gây dự đoán sai hàng loạt và thất thoát doanh thu trực tiếp ngay trong khung giờ vàng."
    ),
    5: (
        "Tại Slide 5, chúng em phân tích phương pháp tiền nhiệm đầu tiên: họ các lịch trình suy giảm tốc độ học cổ điển theo quy luật Robbins-Monro "
        "[2] và lịch trình giảm theo bậc thang (Step Decay) cho bài toán bình phương tối thiểu của Ge et al. (2019). Ý tưởng cốt lõi của hướng tiếp "
        "cận này là đặt tốc độ học giảm dần theo thời gian, điển hình là eta_t = eta_0 / (1 + c*(t - 1)) = O(1/t) hoặc eta_t = eta_0 / sqrt(t), "
        "sao cho thỏa mãn hai điều kiện hội tụ kinh điển: tổng các bước nhảy tiến ra vô cùng (để mô hình đi tới được mọi vị trí) nhưng tổng bình "
        "phương các bước nhảy hữu hạn (để phương sai nhiễu gradient tích lũy bị triệt tiêu dần về 0). Tuy nhiên, hạn chế cốt tử của toàn bộ dòng "
        "nghiên cứu này là giả định môi trường hoàn toàn tĩnh: phân phối dữ liệu P_t = P không đổi và nghiệm tối ưu theta*_t = theta* cố định trong "
        "suốt quá trình huấn luyện (tức gamma_t = 0 với mọi t). Hệ quả thực tế của hạn chế này trong các hệ thống học trực tuyến dài hạn là vô cùng "
        "nghiêm trọng. Khi hệ thống đã chạy tới bước thời gian lớn (ví dụ t > 1000), giá trị eta_t đã suy giảm xuống gần bằng 0. Nếu tại thời điểm "
        "muộn đó xảy ra một đợt dịch chuyển phân phối làm nghiệm tối ưu nhảy sang vị trí mới, biên độ cập nhật trọng số ||theta_{t+1} - theta_t||_2 "
        "= eta_t ||grad||_2 bị bóp nghẹt bởi eta_t quá nhỏ. Hệ quả là mô hình rơi vào trạng thái 'đóng băng nhân tạo' (catastrophic freezing), khoảng "
        "cách sai số ||theta_t - theta*_t||_2 phình to đột biến và độ tiếc nuối tích lũy tăng vọt theo đường thẳng. Thực nghiệm của nhóm trên bài "
        "toán hồi quy tuyến tính 8 chiều (Slide 25) xác nhận hệ quả này: lịch trình suy giảm 1/t chịu Cumulative Regret lên tới 358.43, cao gấp hơn "
        "26 lần so với mức 13.40 của lịch trình thích ứng Algorithm 1 đề xuất trong bài báo."
    ),
    6: (
        "Để khắc phục sự cứng nhắc của giả định phân phối tĩnh, hai nhánh nghiên cứu tiền nhiệm gần nhất đã tiếp cận bài toán từ hai góc độ khác "
        "nhau như trình bày trên Slide 6. Nhánh tiền nhiệm thứ hai là Lý thuyết Tối ưu Lồi Trực tuyến trong môi trường động (Dynamic Online Convex "
        "Optimization - OCO), tiêu biểu là công trình của Besbes et al. (2015) [4] và Yang et al. (2016) [3]. Các tác giả này định nghĩa tổng ngân "
        "sách biến thiên của đường đi nghiệm tối ưu trên toàn bộ chân trời T là V_T = sum_{t=1}^T ||theta*_t - theta*_{t+1}||_2. Để tối ưu cận trên "
        "của Dynamic Regret trong trường hợp xấu nhất, Yang et al. (2016) đề xuất giữ tốc độ học hằng số cố định eta tỉ lệ với sqrt(V_T / T), trong "
        "khi Besbes et al. (2015) đề xuất khởi động lại thuật toán OGD định kỳ sau mỗi Delta_T = (T / V_T)^{2/3} bước với bước nhảy cố định trong mỗi "
        "chu kỳ. Hạn chế lớn nhất của hướng này là tính phi thực tế về mặt thông tin: bộ học bắt buộc phải biết trước tổng thời gian vận hành T và "
        "tổng độ trôi của tương lai V_T ngay từ bước t = 1. Hệ quả thực tế là trong hệ thống streaming không biết khi nào mới có đợt shift, việc dùng "
        "một hằng số eta cố định suốt toàn thời gian sẽ gây lãng phí kép: bước nhảy quá lớn gây dao động nhiễu mạnh trong những giai đoạn môi trường "
        "ổn định (gamma_t = 0), nhưng lại quá nhỏ không kịp phản ứng khi xảy ra đột biến cục bộ. Nhánh tiền nhiệm thứ ba là phương pháp xấp xỉ Phương "
        "trình Vi phân Ngẫu nhiên (SDE) trong giới hạn thời gian liên tục của Li et al. (2017) và Shi et al. (2020) [4]. Dù cung cấp công cụ giải tích "
        "mạnh để mô tả động học SGD và tốc độ hội tụ tuyến tính về phân phối dừng, toàn bộ các mô hình SDE tiền nhiệm đều giả định bề mặt hàm mất mát "
        "cố định (vận tốc trôi Y'(tau) = 0), nên chưa thể dẫn xuất được chính sách điều khiển tốc độ học khi nghiệm tối ưu di chuyển theo thời gian."
    ),
    7: (
        "Slide 7 tổng kết phần bối cảnh bằng bảng định vị khoảng trống nghiên cứu (The Research Gap) giữa ba phương pháp tiền nhiệm và công trình "
        "của Fahrbach et al. (ICML 2023) [1]. Nhìn vào bảng đối chiếu 4 tiêu chí ở bên phải, ta thấy rõ tại sao các phương pháp trước đó chưa giải "
        "quyết trọn vẹn bài toán học trực tuyến dưới dịch chuyển phân phối. Thứ nhất, các lịch trình Robbins-Monro O(1/t) và Step Decay hoàn toàn "
        "không xét đến sự dịch chuyển của nghiệm tối ưu (gamma_t > 0), dẫn tới hiện tượng đóng băng mô hình ở giai đoạn muộn. Thứ hai, các phương "
        "pháp OCO động của Besbes et al. (2015) và Yang et al. (2016) tuy có xét đến quỹ đạo nghiệm thay đổi nhưng lại giả định chuỗi hàm mất mát "
        "bị ràng buộc bởi một ngân sách tổng V_T cố định cho trước, từ đó chỉ đưa ra một bước nhảy tĩnh hoặc chu kỳ restart cố định định trước chứ "
        "không thể thích ứng linh hoạt theo từng bước thời gian t dựa trên lịch sử quan sát. Thứ ba, các nghiên cứu xấp xỉ SDE của Shi et al. (2020) "
        "mới chỉ mô hình hóa nhiễu khuếch tán của SGD trên cảnh quan mất mát tĩnh mà thiếu hẳn thành phần vận tốc trôi của nghiệm tối ưu. Bài báo "
        "của Fahrbach et al. (2023) chính là công trình đầu tiên lấp đầy đồng thời cả bốn khoảng trống này: (1) Cho phép phân phối dữ liệu P_t dịch "
        "chuyển một cách thích ứng (adaptive adversary) dựa trên lịch sử mô hình đã học; (2) Đưa trực tiếp đạo hàm vận tốc trôi Y'(tau) vào phương "
        "trình SDE của sai số mô hình và giải bài toán điều khiển tối ưu thời gian thực bằng phương trình Hamilton-Jacobi-Bellman; (3) Thiết lập các "
        "lịch trình tốc độ học hoàn toàn trực tuyến (chỉ phụ thuộc lịch sử z_{[t-1]} và ước lượng trượt EMA gamma_hat_t) cho cả ba lớp hàm mục tiêu: "
        "hồi quy tuyến tính, hàm lồi tổng quát và mạng nơ-ron không lồi. Từ bức tranh tổng quan này, chúng ta bước sang Phần 2 để xem xét cụ thể "
        "từng đóng góp của bài báo."
    ),
    8: (
        "Bước sang Phần 2 (Slides 8 và 9), chúng em trình bày lại các đóng góp của bài báo bằng lời văn và cấu trúc phân tích của nhóm (chi tiết tại "
        "`docs/02_contributions.md`). Trên Slide 8, các kết quả của bài báo được tổ chức thành ba trụ cột tương ứng với mức độ tổng quát tăng dần của "
        "hàm mất mát. Trụ cột thứ nhất là bài toán Hồi quy Tuyến tính (Mục 3): nhóm tác giả xét mô hình hệ số biến thiên theo thời gian với dịch chuyển "
        "nhãn (response shift), nơi chuỗi nghiệm tối ưu thỏa mãn ||theta*_t - theta*_{t+1}||_2 <= gamma_t. Bằng cách xây dựng một phương trình vi phân "
        "ngẫu nhiên (SDE) mới xấp xỉ động học của Mini-batch SGD và áp dụng Bổ đề Itô để rút ra hệ phương trình vi phân thường (ODE) cho các moment sai "
        "số, tác giả chuyển bài toán tìm lịch trình tốc độ học về bài toán điều khiển tối ưu liên tục và dẫn xuất ra thuật toán cập nhật trực tuyến "
        "Algorithm 1. Trụ cột thứ hai là bài toán Hàm Lồi Tổng quát (Mục 4): mở rộng sang phân phối dữ liệu P_t thay đổi cả đặc trưng lẫn nhãn (covariate "
        "và response shift) do đối thủ chọn thích ứng theo lịch sử. Tác giả chứng minh cận trên tổng quát cho Dynamic Regret kỳ vọng (Theorem 4.2), "
        "chứng minh cận dưới khớp đến hằng số cho lớp hàm lồi mạnh trên cùng một hàm mất mát cố định l(theta, z) (Theorem 4.7), và dẫn xuất quy tắc "
        "tốc độ học kẹp giữa hai ngưỡng đóng [tau_{1,t}, tau_{2,t}] (Proposition 4.3). Trụ cột thứ ba là bài toán Hàm Không lồi và Kiểm chứng Thực "
        "nghiệm (Mục 5 và Mục 6): tác giả đề xuất thước đo Regret mới dựa trên tổng bình phương chuẩn gradient của chuỗi mô hình ước lượng, tìm ra "
        "nghiệm đóng cho bước nhảy tối ưu cận trên (Theorem 5.1), và minh họa hiệu quả của cơ chế tăng bước nhảy / khởi động lại bộ tối ưu trên cả "
        "bài toán hồi quy cao chiều (d lên tới 512) lẫn bài toán y sinh phân loại tế bào dòng chảy (Flow Cytometry MLP)."
    ),
    9: (
        "Tại Slide 9, chúng em đi sâu hơn một bước so với việc liệt kê đơn thuần: phân loại từng đóng góp theo ba chiều (Lý thuyết, Kỹ thuật, Thực "
        "nghiệm) và đối chiếu trực diện với các công trình tiền nhiệm để làm rõ tính mới (Novelty — điều gì trước đây chưa làm được hoặc chưa từng "
        "được chứng minh). Về mặt Đóng góp Lý thuyết, điểm mới nổi bật nhất gồm hai kết quả: thứ nhất là Proposition 3.1 thiết lập cận sai số xác suất "
        "cao O(sqrt(epsilon)) giữa quỹ đạo SGD rời rạc và nghiệm SDE liên tục khi có sự hiện diện của số hạng vận tốc trôi Y'(tau) — điều mà phân tích "
        "SDE tĩnh của Shi et al. (2020) [4] chưa bao gồm; thứ hai là cặp Theorem 4.2 và Theorem 4.7 chứng minh độ chặt (tightness up to constants) của "
        "phân tích Dynamic Regret. Khác với cận dưới của Yang et al. (2016) [2] hay Besbes et al. (2015) [3] vốn phải tự chế tạo một chuỗi hàm mất mát "
        "thay đổi hình dạng liên tục cho trường hợp xấu nhất, cận dưới Theorem 4.7 của bài báo giữ cố định một hàm mất mát l(theta, z) duy nhất và chỉ "
        "để sự biến thiên đến từ dịch chuyển phân phối dữ liệu z ~ P_t. Về mặt Đóng góp Kỹ thuật, tác giả đã khéo léo vượt qua sự phụ thuộc vào hướng "
        "của vector vận tốc trôi d chiều Y'(tau) bằng cách xây dựng quá trình bao trên vô hướng 1 chiều v_tilde_tau, cho phép giải phương trình HJB "
        "ra công thức đóng chỉ tốn chi phí O(1) mỗi bước; đồng thời đề xuất quy tắc kẹp hai ngưỡng [tau_{1,t}, tau_{2,t}] kết hợp bộ ước lượng trượt "
        "hàm mũ EMA gamma_hat_t hoàn toàn không cần biết trước ngân sách V_T. Về mặt Đóng góp Thực nghiệm, bài báo lần đầu tiên phác họa rõ hiện tượng "
        "tốc độ học tối ưu dao động tuần hoàn nảy lên mức trần eta_max mỗi khi phân phối nhảy bậc, và kết nối lý thuyết tăng bước nhảy với thủ tục "
        "khởi động lại bộ tối ưu Adam (Adam Restart) trên dữ liệu RNA đơn bào thực tế."
    ),
    10: (
        "Bắt đầu Phần 3 — phần trình bày 5 lớp phương pháp cốt lõi của bài báo — Slide 10 giới thiệu Lớp 1 (Formalism): Bảng ký hiệu thống nhất của "
        "nhóm (được quy chuẩn trong tài liệu `docs/notation.md`) và Phát biểu hình thức của bài toán học trực tuyến. Để tránh mọi sự nhập nhằng khi "
        "chuyển tiếp giữa ba mục lý thuyết của bài báo, nhóm thống nhất gọi t in {1, 2, ..., T} là chỉ số bước thời gian rời rạc (vòng lặp SGD) với "
        "chân trời T, d là số chiều của mẫu dữ liệu z_{t,k} in R^d và p là số chiều của vector tham số mô hình theta in R^p (trong hồi quy tuyến tính "
        "và hồi quy logistic không hệ số chặn thì p = d). Tại mỗi bước t, bộ học nhận một mini-batch gồm B_t mẫu độc lập cùng phân phối z_t = "
        "{z_{t,k}}_{k=1}^{B_t} được rút từ phân phối xác suất P_t tại thời điểm đó. Ta ký hiệu l(theta, z) >= 0 là hàm mất mát điểm trên một mẫu, và "
        "l_bar_t(theta) := E_{z ~ P_t}[l(theta, z)] là hàm mất mát kỳ vọng trên toàn quần thể P_t tại bước t. Nghiệm tối ưu tức thời của quần thể "
        "(được gọi là mô hình Oracle hay dynamic comparator) tại bước t được định nghĩa tại Công thức (1) là theta*_t := argmin_{theta in R^p} l_bar_t(theta). "
        "Mức độ dịch chuyển phân phối giữa hai bước liên tiếp được lượng hóa bằng khoảng cách Euclid giữa hai nghiệm Oracle liên tiếp: gamma_t := "
        "||theta*_t - theta*_{t+1}||_2 (Công thức 6, đối với bài toán lồi), hoặc khoảng cách supremum giữa hai hàm mất mát kỳ vọng: gamma_t := "
        "sup_theta |l_bar_t(theta) - l_bar_{t+1}(theta)| (Công thức 23, đối với bài toán không lồi). Mục tiêu của bộ học là chọn chuỗi tốc độ học "
        "thích ứng {eta_t}_{t=1}^T — trong đó eta_t chỉ được phép là hàm đo được của lịch sử dữ liệu quá khứ z_{[t-1]} = {z_1, ..., z_{t-1}} — để cập "
        "nhật Projected Mini-batch SGD: theta_{t+1} = Pi_Theta(theta_t - eta_t * grad_l^B_t(theta_t)) sao cho cực tiểu hóa tổng Dynamic Regret kỳ vọng "
        "E[Reg(T)] = sum_{t=1}^T E[l_bar_t(theta_t) - l_bar_t(theta*_t)]."
    ),
    11: (
        "Slide 11 làm rõ bản chất trò chơi tương tác tuần tự giữa Tự nhiên (Nature / Adversary) và Bộ học (Learner) cũng như ý nghĩa hình học của "
        "Dynamic Regret trong môi trường có trôi dạt phân phối. Khác với bài toán tối ưu tĩnh hay bài toán Static Regret cổ điển của Zinkevich (2003) "
        "— nơi hiệu năng của bộ học chỉ được so sánh với một bộ trọng số cố định duy nhất theta* tốt nhất sau khi nhìn lại toàn bộ T bước — bài toán "
        "của chúng ta đánh giá hiệu năng trọn đời (lifetime performance) của hệ thống bằng cách so sánh mô hình theta_t tại mỗi thời điểm t trực tiếp "
        "với nghiệm tối ưu tức thời theta*_t của chính phân phối P_t tại thời điểm đó. Trình tự nhân quả trong mỗi vòng lặp t diễn ra qua 4 bước "
        "nghiêm ngặt như sơ đồ bên phải minh họa: Bước 1, dựa trên lịch sử dữ liệu đã qua z_{[t-1]} (và do đó có thể phụ thuộc vào cả các trọng số "
        "mô hình theta_1, ..., theta_t mà bộ học đã chọn), Tự nhiên quyết định phân phối P_t và nghiệm tối ưu tương ứng theta*_t thuộc tập lồi đóng "
        "bị chặn Theta có đường kính D_max. Bước 2, bộ học — lúc này đang nắm giữ bộ trọng số theta_t được cập nhật từ cuối vòng t-1 — phải đưa ra "
        "dự đoán trên luồng mẫu mới từ P_t và gánh chịu độ tiếc nuối tức thời reg_t = l_bar_t(theta_t) - l_bar_t(theta*_t) >= 0. Bước 3, bộ học quan "
        "sát được mini-batch B_t mẫu z_t ~ P_t, tính trung bình đạo hàm mẫu grad_l^B_t(theta_t) = (1/B_t) sum_{k=1}^{B_t} grad_l(theta_t, z_{t,k}) "
        "(vốn là một ước lượng không chệch của gradient kỳ vọng grad_l_bar_t(theta_t)), rồi chọn tốc độ học eta_t để cập nhật trọng số sang theta_{t+1} "
        "nhờ phép chiếu Euclid Pi_Theta. Bước 4, trước khi sang vòng t+1, nghiệm tối ưu của Tự nhiên dịch chuyển sang vị trí mới theta*_{t+1} cách vị "
        "trí cũ một khoảng gamma_t = ||theta*_t - theta*_{t+1}||_2. Chính khoảng nhảy gamma_t này buộc bộ học không bao giờ được phép dừng khám phá."
    ),
    12: (
        "Chuyển sang Lớp 2 của phương pháp (The Overall Pipeline) tại Slide 12, chúng em trình bày sơ đồ khối tổng thể `figures/pipeline.png` do nhóm "
        "tự thiết kế và lập trình xuất hình trong `experiments/smoke_test.py`. Một yêu cầu đặc biệt quan trọng đối với hệ thống Học trực tuyến không "
        "ngừng (Never-Ending Learning) là phải tách bạch hoàn toàn hai quá trình diễn ra nối tiếp nhau trên cùng một mẫu dữ liệu tại bước t: Quá trình "
        "Suy luận và Đánh giá trực tuyến (Stage A — Online Inference & Evaluation Pipeline, khung màu xanh dương ở tầng trên) và Quá trình Huấn luyện "
        "và Thích ứng tốc độ học (Stage B — Online Training & Adaptive LR Pipeline, khung màu xanh lá ở tầng dưới). Cụ thể, ở Stage A (cơ chế Test-First "
        "hay Prequential Evaluation): khi luồng dữ liệu tại bước t mang tới một mini-batch mẫu mới {x_{t,k}}_{k=1}^{B_t} ~ P_t (Khối 1), hệ thống bắt "
        "buộc phải dùng bộ trọng số hiện tại theta_t (đang ở trạng thái đóng băng tạm thời, hoàn toàn chưa nhìn thấy nhãn y_t của bước t) để thực hiện "
        "suy luận thời gian thực y_hat_{t,k} = f(x_{t,k}; theta_t) phục vụ người dùng (Khối 2). Ngay sau khi nhãn thực tế y_{t,k} được phản hồi (từ "
        "hành vi click của người dùng hoặc từ mạng Teacher CNN trong bài toán phân loại tế bào), Khối 3 tính giá trị hàm mất mát và ghi nhận độ tiếc "
        "nuối tức thời reg_t = l_bar_t(theta_t) - l_bar_t(theta*_t) vào tổng tích lũy Reg(t). Chỉ sau khi Stage A hoàn tất việc đánh giá công bằng, "
        "cặp dữ liệu có nhãn z_t = (x_t, y_t) mới được chuyển xuống Stage B theo mũi tên màu đỏ bên phải. Tại Stage B, Khối 4 tính vector đạo hàm mẫu "
        "trung bình grad_l^B_t(theta_t) đồng thời thu nhận tín hiệu độ trôi phân phối (từ Oracle gamma_{t-1} hoặc từ bộ ước lượng trung bình trượt hàm "
        "mũ EMA gamma_hat_t = beta*gamma_hat_{t-1} + (1-beta)||theta_t - theta_{t-1}||_2). Tín hiệu độ trôi này đi vào Khối 5 (Bộ điều khiển tốc độ "
        "học thích ứng: Algorithm 1 cho hàm tuyến tính, Proposition 4.3 cho hàm lồi, hoặc Công thức 25 / Adam Restart cho hàm không lồi) để xuất ra "
        "bước nhảy tối ưu eta*_t. Cuối cùng, Khối 6 thực hiện cập nhật Projected Mini-batch SGD tạo ra trọng số mới theta_{t+1}, ghi log trực tiếp lên "
        "Weights & Biases và triển khai theta_{t+1} lên lại Khối 2 cho vòng lặp t+1 tiếp theo."
    ),
    13: (
        "Từ Slide 13 đến Slide 19, chúng ta đi vào Lớp 3 của phương pháp: giải phẫu chi tiết từng thành phần thuật toán theo cấu trúc 3 câu hỏi "
        "cốt lõi (Làm gì? Vì sao cần? Hoạt động thế nào?). Slide 13 mở đầu bằng bài toán Hồi quy Tuyến tính có hệ số biến thiên theo thời gian (Mục 3, "
        "`docs/03_linear.md`). Trong thiết lập Công thức (7), mỗi mẫu dữ liệu là cặp (x_{t,k}, y_{t,k}) với vector đặc trưng x_{t,k} ~ N(0, I_d) tuân "
        "theo phân phối chuẩn tắc đẳng hướng cố định, còn nhãn liên tục được sinh bởi y_{t,k} = <x_{t,k}, theta*_t> + eps_{t,k} với nhiễu độc lập "
        "eps_{t,k} ~ N(0, sigma^2). Với hàm mất mát bình phương tối thiểu l(theta, z) = 1/2 (y - <x, theta>)^2, nhờ tính chất E[x x^T] = I_d, độ tiếc "
        "nuối kỳ vọng tại bước t rút gọn chính xác thành một nửa bình phương khoảng cách Euclid tới nghiệm Oracle: E[reg_t] = 1/2 E[||theta_t - theta*_t||_2^2]. "
        "Câu hỏi thứ nhất — Mô-đun SDE ở Mục 3 làm gì? Nó xấp xỉ quỹ đạo sai số rời rạc theta_t - theta*_t của SGD bằng quá trình ngẫu nhiên thời gian "
        "liên tục X(tau) thông qua Phương trình Vi phân Ngẫu nhiên (SDE) tại Công thức (8). Câu hỏi thứ hai — Vì sao cần chuyển sang SDE? Trong miền rời "
        "rạc, cập nhật SGD phụ thuộc vào tích của các ma trận Wishart ngẫu nhiên (I - eta_t/B_t X_t^T X_t) qua nhiều bước nên không thể giải tìm nghiệm "
        "đóng cho chuỗi eta*_t. Khi đặt biến thời gian liên tục tau = eps*t với eps = eta_max là bước nhảy tối đa, biểu diễn tốc độ học eta_t = eps*zeta(eps*t) "
        "(hàm điều khiển zeta(tau) in [0, 1]), batch size B_t = eps*nu(eps*t) và đường cong nội suy trơn Y(eps*t) = theta*_t, phép xấp xỉ Euler-Maruyama "
        "kết hợp đẳng thức hiệp phương sai bậc bốn của Bổ đề Stein (Lemma D.1) chuyển động học sai số về SDE (8). Như sơ đồ bên phải minh họa, vi phân "
        "dX(tau) chịu tác động của 3 lực vật lý: (1) Lực kéo hội tụ của SGD -zeta(tau)X(tau)d_tau kéo sai số về 0; (2) Lực đẩy lệch -Y'(tau)d_tau do đích "
        "ngắm theta*_t di chuyển với vận tốc Y'(tau) (có chuẩn ||Y'(eps*t)|| ≈ gamma_t / eps); và (3) Số hạng khuếch tán chứa chuyển động Brown dW(tau) "
        "phản ánh tổng nhiễu tự thân ||X(tau)||^2 và nhiễu nhãn sigma^2 chia cho sqrt(nu(tau)). Proposition 3.1 bảo chứng rằng dưới giả thiết trơn "
        "Lipschitz A1-A2, sai số giữa ||X(t*eps)||^2 và ||theta_t - theta*_t||^2 bị chặn chặt bởi O(sqrt(eps))."
    ),
    14: (
        "Sau khi ánh xạ quỹ đạo SGD sang phương trình SDE (8) trên R^d, Slide 14 giải thích cách nhóm tác giả rút gọn bài toán ngẫu nhiên d chiều "
        "về một bài toán điều khiển tối ưu tất định chỉ có đúng 1 chiều vô hướng qua ba bước toán học liên tiếp. Ở Bước 1 (Theorem 3.2), ta định nghĩa "
        "moment bậc nhất m_tau := E[X(tau)] in R^d (vector kỳ vọng sai số) và moment bậc hai v_tau := E[||X(tau)||_2^2] in R_{>=0} (bình phương khoảng "
        "cách sai số kỳ vọng, tỷ lệ trực tiếp với E[reg_t] ≈ 1/2 v_{eps*t}). Lấy kỳ vọng trực tiếp SDE (8) làm triệt tiêu số hạng martingale dW(tau), "
        "cho ta phương trình ODE (10): m'_tau = -zeta(tau)m_tau - Y'(tau). Để tìm đạo hàm theo thời gian của v_tau, tác giả áp dụng Bổ đề Itô (Lemma D.2) "
        "cho hàm vô hướng phi(X) = ||X||_2^2 có Hessian grad^2 phi(X) = 2 I_d. Phép lấy vết ma trận khuếch tán 1/2 Tr(g^T grad^2 phi g) = (zeta^2 / nu) "
        "Tr((||X||^2 + sigma^2)I_d + X X^T) sinh ra hệ số (d + 1)||X||^2 + d*sigma^2, dẫn tới ODE (11): v'_tau = ((d+1)zeta^2/nu - 2*zeta)v_tau + "
        "(zeta^2/nu)sigma^2*d - 2 m_tau^T Y'(tau). Ở Bước 2, phương trình (11) còn vướng tích vô hướng -2 m_tau^T Y'(tau) phụ thuộc góc giữa m_tau và "
        "vận tốc dịch chuyển Y'(tau). Áp dụng bất đẳng thức Cauchy-Schwarz và Jensen: |m_tau^T Y'(tau)| <= ||Y'(tau)||_2 ||m_tau||_2 <= ||Y'(tau)||_2 sqrt(v_tau), "
        "tác giả định nghĩa quá trình bao trên (upper-envelope process) v_tilde_tau >= v_tau tuân theo ODE vô hướng 1 chiều tại Công thức (12). Ở Bước 3 "
        "(Theorem 3.3), ta tìm hàm điều khiển zeta(tau) in [0, 1] cực tiểu hóa phiếm hàm chi phí int_0^T v_tilde_tau d_tau qua phương trình đạo hàm riêng "
        "Hamilton-Jacobi-Bellman (HJB): d_tau V + min_{zeta in [0,1]} [d_v V * v_tilde'_tau(zeta) + v_tilde_tau] = 0. Vì sai số hiện tại càng lớn thì chi "
        "phí tương lai càng cao nên d_v V > 0 luôn dương; do đó chính sách tối ưu toàn cục chính là nghiệm cực tiểu hóa tam thức bậc hai theo zeta trong "
        "vế phải của ODE (12), cho ra công thức đóng tại Công thức (13). Hai hình bên phải (`figures/fig2_alg1_schedules.png` và `fig3_no_shift_asymptotics.png`) "
        "minh họa nghiệm này: tốc độ học tăng vọt theo biên độ shift, còn khi không có shift (gamma_t = 0, Lemma 3.6), lịch trình giữ ở mức trần eps đến "
        "tau_* rồi tự động suy giảm tiệm cận theo quy luật cổ điển O(1/t)."
    ),
    15: (
        "Slide 15 hoàn tất phần lý thuyết Hồi quy Tuyến tính bằng cách chuyển nghiệm điều khiển liên tục của Theorem 3.3 về thuật toán thời gian thực "
        "Algorithm 1 (cài đặt tại lớp `LinearOptimalSchedule` trong `core/schedules.py`). Sử dụng hệ thức ánh xạ giữa miền liên tục và rời rạc: "
        "eta_t = eps*zeta(eps*t), B_t = eps*nu(eps*t) và ||Y'(eps*t)||_2 ≈ ||theta*_t - theta*_{t-1}||_2 / eps = gamma_{t-1} / eps, công thức chính sách "
        "tối ưu (13) viết theo biến rời rạc có dạng: eta*(v) = min{ eps, (v * B_t) / ((d + 1)*v + sigma^2 * d) }. Để tích phân số phương trình ODE (12) "
        "của quá trình bao trên v từ bước t-1 sang t, Algorithm 1 chia mỗi bước SGD thành ceil(1 / kappa) bước lặp con Euler thuận (forward Euler sub-steps) "
        "với bước chia kappa in (0, 1]. Trong mỗi bước con j, thuật toán tính bước nhảy r = min(v*B_t / ((d+1)v + sigma^2*d), eps), rồi cập nhật biến "
        "trạng thái v theo ba số hạng: số hạng co giảm do học SGD kappa*(((d+1)/B_t)r^2 - 2r)*v, số hạng bơm phương sai nhiễu nhãn kappa*(sigma^2*d/B_t)r^2, "
        "và số hạng bơm sai số do dịch chuyển phân phối +2*kappa*gamma_{t-1}*sqrt(v). Từ công thức này, ta trả lời trọn vẹn câu hỏi trọng tâm của Task T7: "
        "Vì sao tốc độ học tối ưu eta*_t lại TĂNG LÊN khi có distribution shift? Lấy đạo hàm hàm bước nhảy ứng viên f(v) = (v * B_t) / ((d + 1)*v + "
        "sigma^2 * d) theo v, ta có: f'(v) = (B_t * sigma^2 * d) / ((d + 1)*v + sigma^2 * d)^2 > 0 luôn dương nghiêm ngặt! Nghĩa là bước nhảy tối ưu "
        "eta*_t đồng biến nghiêm ngặt theo phương sai sai số v. Về trực giác vật lý: v đại diện cho bình phương khoảng cách tới nghiệm tối ưu (tín hiệu), "
        "còn sigma^2*d / B_t là phương sai nhiễu nhãn trên mini-batch (nhiễu). Khi xảy ra dịch chuyển phân phối (gamma_{t-1} > 0), số hạng +2*kappa*gamma_{t-1}*sqrt(v) "
        "bơm năng lượng dương làm v tăng vọt; tỷ số tín hiệu trên nhiễu (SNR) của gradient mẫu rất cao nên bộ điều khiển đẩy eta*_t tăng vọt lên (thậm chí "
        "chạm trần eps = eta_max) để đuổi theo đích mới. Khi môi trường đứng yên (gamma_{t-1} = 0), số hạng cuối biến mất, tổng hai số hạng đầu bằng "
        "-kappa*r*v < 0 làm v giảm dần về 0, kéo theo eta*_t tự động hạ xuống. Cuối cùng, khi không biết trước gamma_{t-1} (Remark 3.5), ta thay bằng ước "
        "lượng trượt EMA của bước nhảy mô hình: gamma_hat_t = beta*gamma_hat_{t-1} + (1-beta)||theta_t - theta_{t-1}||_2."
    ),
    16: (
        "Bước sang Slide 16 (do Khôi phụ trách, chi tiết tại `docs/04_convex.md`), chúng ta mở rộng bài toán từ hàm bình phương tối thiểu sang lớp "
        "Hàm Mất mát Lồi Tổng quát l(theta, z) trên không gian R^p. Đầu tiên, tại sao không thể tiếp tục dùng phương pháp xấp xỉ SDE của Mục 3 cho hàm "
        "lồi tổng quát? Bởi vì với hàm lồi bất kỳ (ví dụ hàm Binary Cross-Entropy trong hồi quy Logistic), đạo hàm grad_l(theta, z) là hàm phi tuyến của "
        "theta chứ không còn dạng tuyến tính <x, theta - theta*>, nên khi lấy kỳ vọng ta không thể khép kín hệ phương trình moment bậc nhất và bậc hai "
        "như Theorem 3.2. Thay vào đó, nhóm tác giả thiết lập một chặn trên giải tích cho tổng Dynamic Regret kỳ vọng E[Reg(T)] tại Theorem 4.2, rồi chọn "
        "chuỗi tốc độ học để cực tiểu hóa tuần tự chặn trên đó. Assumption 4.1 đặt ra ba giả thiết chuẩn: (i) Phương sai của gradient mẫu đơn lẻ bị chặn "
        "đều bởi sigma^2, suy ra phương sai của gradient trung bình trên mini-batch B_t bị chặn bởi sigma^2 / B_t; (ii) Hàm mất mát kỳ vọng l_bar_t(theta) "
        "là L-smooth (||grad_l_bar_t(theta_1) - grad_l_bar_t(theta_2)||_2 <= L ||theta_1 - theta_2||_2); và (iii) Các nghiệm tối ưu Oracle theta*_t cùng "
        "quỹ đạo Projected SGD nằm trong tập lồi đóng bị chặn Theta có đường kính hữu hạn D_max (||theta - theta'||_2 <= D_max). Dưới điều kiện ổn định "
        "0 < eta_t <= 1/L (để hệ số a_t := 2*eta_t - L*eta_t^2 > 0), Theorem 4.2 dùng tính không giãn của phép chiếu Euclid Pi_Theta và bất đẳng thức "
        "co-coercivity của hàm lồi L-smooth để đánh giá hiệu khoảng cách D_{t+1}^2 - D_t^2 (với D_t := ||theta*_t - theta_t||_2), thu được cận trên tại "
        "Công thức (15). Vì cận (15) chứa đại lượng không quan sát được D_t, tác giả khai triển tổng vi phân từng phần (telescoping sum) theo chuỗi 1/a_t "
        "và chặn mọi khoảng cách chưa biết bởi đường kính D_max cùng độ trôi gamma_t = ||theta*_t - theta*_{t+1}||_2, chuyển (15) thành cận trên khả kiến "
        "(observable upper bound) tại Công thức (16). Như sơ đồ bên phải minh họa, cận (16) gồm ba thành phần: (1) Đại lượng phạt D_max^2 (1/a_t - 1/a_{t-1})_+ "
        "khi giảm tốc độ học quá nhanh; (2) Chi phí phương sai nhiễu mini-batch (sigma^2 * eta_t^2) / (B_t * a_t); và (3) Chi phí trôi dạt phân phối "
        "(gamma_t^2 + 2*D_max*gamma_t) / a_t."
    ),
    17: (
        "Từ cận trên khả kiến tại Công thức (16), Slide 17 trình bày Proposition 4.3 — quy tắc chọn tốc độ học thích ứng theo hai ngưỡng cho hàm lồi "
        "tổng quát. Tại mỗi bước t >= 2, khi đã biết tốc độ học bước trước eta_{t-1}, bộ học tìm giá trị eta_t in (0, 1/L] để cực tiểu hóa số hạng thứ t "
        "trong cận trên (16), viết tường minh tại bài toán tối ưu Công thức (17). Điểm đặc sắc về giải tích của hàm mục tiêu (17) là sự xuất hiện của hàm "
        "phần dương (1/a_t - 1/a_{t-1})_+ = max(0, 1/(2*eta - L*eta^2) - 1/(2*eta_{t-1} - L*eta_{t-1}^2)). Vì hàm a(eta) = 2*eta - L*eta^2 đồng biến nghiêm "
        "ngặt trên (0, 1/L], số hạng phạt D_max^2 * (1/a_t - 1/a_{t-1})_+ bằng 0 khi eta >= eta_{t-1} (khi giữ nguyên hoặc tăng tốc độ học), nhưng lại bật "
        "lên dương khi eta < eta_{t-1} (khi giảm tốc độ học). Tính chất này chia miền tối ưu (0, 1/L] thành hai nhánh lồi trơn ghép nhau tại điểm gãy "
        "eta = eta_{t-1}. Giải phương trình đạo hàm bằng 0 trên từng nhánh (Phụ lục B.2, Công thức 57), Proposition 4.3 chỉ ra hai điểm dừng đóng tương ứng "
        "là ngưỡng dưới tau_{1,t} (Công thức 18, ứng với hệ số b_{1,t} = gamma_t^2 + 2*D_max*gamma_t khi nhánh tăng kích hoạt) và ngưỡng trên tau_{2,t} "
        "(Công thức 19, ứng với hệ số lớn hơn b_{2,t} = (gamma_t + D_max)^2 = b_{1,t} + D_max^2 khi nhánh giảm kích hoạt). Vì b_{1,t} < b_{2,t} và hàm "
        "ngưỡng đồng biến theo b, ta luôn có 0 <= tau_{1,t} < tau_{2,t} <= 1/L. Từ tính lồi của hàm mục tiêu có điểm gãy, nghiệm tối ưu toàn cục eta*_t "
        "chính là phép chiếu (clipping) của tốc độ học bước trước eta*_{t-1} vào đoạn [tau_{1,t}, tau_{2,t}] tại Công thức (20): Nếu eta*_{t-1} <= tau_{1,t} "
        "(độ trôi gamma_t tăng vọt đẩy ngưỡng dưới vượt lên), ta nâng ngay eta*_t lên tau_{1,t}; nếu eta*_{t-1} >= tau_{2,t}, ta hạ xuống tau_{2,t}; còn "
        "nếu nằm trong vùng đệm [tau_{1,t}, tau_{2,t}], ta giữ nguyên eta*_t = eta*_{t-1} để tránh phạt telescoping! Đồ thị bên phải minh họa quỹ đạo tốc "
        "độ học này, và trong lớp `ConvexSchedule` (`core/schedules.py`), nhóm đã áp dụng phép nhân liên hợp hữu tỷ hóa vào miền logarit để tính tau_{1,t}, "
        "tau_{2,t} chính xác tuyệt đối ngay cả khi gamma_t rất nhỏ."
    ),
    18: (
        "Slide 18 chuyển sang mảnh ghép lý thuyết thứ ba của bài báo: Mục 5 — Hàm mất mát không lồi (Non-convex loss functions), nền tảng cho việc "
        "huấn luyện mạng nơ-ron sâu trực tuyến (`docs/05_nonconvex.md`). Câu hỏi đầu tiên đặt ra là: Tại sao toàn bộ khung phân tích Dynamic Regret dựa "
        "trên khoảng cách tới nghiệm tối ưu theta*_t ở Mục 3 và Mục 4 không thể áp dụng trực tiếp cho hàm không lồi? Có hai lý do cơ bản. Thứ nhất, cảnh "
        "quan của hàm mất mát không lồi l_bar_t(theta) chứa rất nhiều cực tiểu địa phương (local minima) và điểm yên ngựa (saddle points). Ngay cả trong "
        "môi trường tĩnh không có dịch chuyển phân phối, các thuật toán tối ưu bậc nhất như SGD cũng không thể đảm bảo hội tụ về cực tiểu toàn cục theta*_t "
        "nếu không có giả thiết khởi tạo đặc biệt; do đó việc đòi hỏi bộ học trực tuyến phải bám đuổi giá trị tối ưu toàn cục l_bar_t(theta*_t) ở mọi bước "
        "t là mục tiêu bất khả thi. Thứ hai, vì hàm không lồi có nhiều điểm cực tiểu rời rạc thay đổi cấu trúc khi P_t biến thiên, việc đo độ dịch chuyển "
        "phân phối chỉ qua khoảng cách Euclid giữa hai điểm cực tiểu toàn cục ||theta*_t - theta*_{t+1}||_2 không còn phản ánh đúng sự biến dạng của toàn "
        "bộ bề mặt mất mát. Để giải quyết hai thách thức này, nhóm tác giả thực hiện hai thay đổi khái niệm quan trọng trên Slide 18. Một là định nghĩa lại "
        "tổng Dynamic Regret tại Công thức (22) bằng tổng bình phương chuẩn gradient của hàm mất mát kỳ vọng tại chuỗi mô hình học được: "
        "Reg(T) := sum_{t=1}^T ||grad_l_bar_t(theta_t)||_2^2. Theo thước đo này, một quỹ đạo mô hình có tổng Regret thấp khi nó luôn duy trì ở gần các điểm "
        "dừng (stationary points / local minima nơi grad_l_bar_t(theta_t) ≈ 0) của chuỗi hàm mất mát theo thời gian. Hai là định nghĩa lại độ dịch chuyển "
        "phân phối gamma_t tại Công thức (23) theo chuẩn supremum toàn cục (độ biến thiên hàm số như trong Besbes et al., 2015 [3]): "
        "gamma_t := sup_{theta in R^p} |l_bar_t(theta) - l_bar_{t+1}(theta)|. Khi phân phối chuyển pha như hình minh họa bên phải, điểm cực tiểu địa phương "
        "cũ ở bước t bỗng trở thành một sườn dốc có chuẩn gradient ||grad_l_bar_{t+1}(theta_t)|| lớn ở bước t+1, đòi hỏi mô hình phải lập tức thích ứng."
    ),
    19: (
        "Tiếp nối định nghĩa Regret theo chuẩn gradient ở Slide 18, Slide 19 trình bày Theorem 5.1 — định lý thiết lập cận trên cho tổng bình phương "
        "chuẩn gradient kỳ vọng và dẫn xuất công thức đóng cho tốc độ học tối ưu trong môi trường không lồi. Dưới hai giả thiết của Assumption 4.1 là "
        "(i) phương sai gradient mẫu bị chặn bởi sigma^2 và (ii) hàm mất mát kỳ vọng l_bar_t(theta) >= 0 là L-smooth, với điều kiện bước nhảy 0 < eta_t <= 1/L "
        "(đặt a_t := 2*eta_t - L*eta_t^2 > 0), chứng minh của Theorem 5.1 (Phụ lục C.1) bắt đầu từ bất đẳng thức Descent Lemma bậc hai của hàm L-smooth: "
        "l_bar_t(theta_{t+1}) - l_bar_t(theta_t) <= <grad_l_bar_t(theta_t), theta_{t+1} - theta_t> + (L/2)||theta_{t+1} - theta_t||_2^2. Thay quy tắc cập "
        "nhật Mini-batch SGD theta_{t+1} - theta_t = -eta_t grad_l^B_t(theta_t), lấy kỳ vọng có điều kiện trên lịch sử z_{[t-1]} và dùng định nghĩa độ trôi "
        "bề mặt |l_bar_{t+1}(theta_{t+1}) - l_bar_t(theta_{t+1})| <= gamma_t để chuyển tiếp từ hàm mất mát bước t sang t+1, ta thu được cận trên tổng quát "
        "tại Công thức (24). Để cực tiểu hóa số hạng tại bước t một cách tuần tự khi đã biết lịch sử dữ liệu tới t-1, ta giải bài toán tối ưu 1 chiều trên "
        "đoạn (0, 1/L] tại Công thức (25): min_{0 <= eta <= 1/L} [ (2*l_bar_t(theta_t) + 2*gamma_t) / (2*eta - L*eta^2) + (L*sigma^2 / B_t) * eta^2 / "
        "(2*eta - L*eta^2) ]. Đặt đại lượng tổng hợp b_t := L*(gamma_t + l_bar_t(theta_t)) >= 0. Vì cả hai phân thức trong (25) đều lồi chặt theo eta trên "
        "(0, 1/L], việc cho đạo hàm bằng 0 dẫn tới phương trình bậc hai có nghiệm dương duy nhất nằm trong [0, 1/L]: eta*_t = (B_t / (L*sigma^2)) * "
        "(sqrt(b_t^2 + (2*sigma^2 / B_t)*b_t) - b_t). Khi cài đặt trong lớp `NonConvexSchedule` (`core/schedules.py`), ta dùng dạng nhân liên hợp tương đương "
        "eta*_t = 2 / (L * (1 + sqrt(1 + 2*sigma^2 / (B_t * b_t)))). Công thức này chỉ ra rằng tốc độ học tối ưu eta*_t đồng biến nghiêm ngặt theo cả độ "
        "dịch chuyển gamma_t lẫn giá trị hàm mất mát hiện tại l_bar_t(theta_t), đồng thời giải thích cơ sở lý thuyết của cơ chế Adam Restart: khi phân phối "
        "dịch chuyển làm loss nhảy vọt, xóa bộ nhớ moment bậc hai cũ của Adam sẽ khôi phục bước nhảy hiệu dụng lớn để thoát khỏi cực tiểu cục bộ cũ."
    ),
    20: (
        "Tại Slide 20, chúng em thực hiện Lớp 4 của quy trình đọc hiểu phương pháp (Theoretical Properties — chi tiết tại `docs/theorems.md`): đặt "
        "toàn bộ các kết quả lý thuyết của bài báo (Proposition 3.1, Theorem 3.2, Theorem 3.3, Lemma 3.6, Theorem 4.2, Proposition 4.3, Theorem 4.7 và "
        "Theorem 5.1) lên cùng một bàn cân để đối chiếu giả thiết, trực giác cốt lõi và chỉ rõ giới hạn nơi các giả thiết bị vi phạm. Ở nhóm định lý thứ "
        "nhất (Mục 3 — Linear Regression SDE), chuỗi kết quả từ Proposition 3.1 đến Theorem 3.3 đạt được nghiệm điều khiển tối ưu chính xác nhờ hai giả "
        "thiết trụ cột: phân phối đặc trưng phải là Gaussian chuẩn đẳng hướng cố định x_{t,k} ~ N(0, I_d) (để áp dụng Bổ đề Stein khép kín ma trận hiệp "
        "phương sai bậc bốn) và quỹ đạo nghiệm Y(tau) phải trơn Lipschitz với vận tốc bị chặn ||Y'(tau)|| <= Gamma / eps. Khi xảy ra Covariate Shift (ma "
        "trận hiệp phương sai đặc trưng Sigma_t thay đổi theo thời gian và không đẳng hướng) hoặc khi bước nhảy tối đa eps lớn, phương trình ODE của moment "
        "bậc hai v_tau không còn đóng và sai số xấp xỉ SDE O(sqrt(eps)) không còn nhỏ. Ở nhóm định lý thứ hai (Mục 4 — General Convex Loss), Theorem 4.2 "
        "và Proposition 4.3 nới lỏng hoàn toàn giả định phân phối Gaussian và cho phép đối thủ thay đổi P_t thích ứng, nhưng phải đánh đổi bằng giả thiết "
        "miền tham số bị chặn bởi đường kính cố định D_max (Assumption 4.1-iii). Giới hạn lớn nhất nằm ở bước chuyển từ cận (15) sang cận khả kiến (16): "
        "việc thay thế khoảng cách thực tế D_t = ||theta*_t - theta_t||_2 bằng cận trên thô D_max làm xuất hiện số hạng D_max^2 trong hệ số b_{2,t} = "
        "(gamma_t + D_max)^2 và 2*D_max*gamma_t trong b_{1,t}. Như nhóm đã chứng minh bằng thực nghiệm ở chiều cao d = 512 (khi đặt D_max = d = 512 theo "
        "gợi ý của bài báo), giá trị D_max quá lớn làm cả hai ngưỡng tau_{1,t} và tau_{2,t} bị đẩy sát lên mức trần 1/L = 4.0, mất khả năng kẹp giảm bước "
        "nhảy. Tương tự, cận dưới Theorem 4.7 đòi hỏi tính lồi mạnh toàn cục mu > 0 (Assumption 4.6), nhưng với hàm Logistic trên miền bán kính lớn, độ "
        "cong nhỏ nhất mu suy giảm theo hàm mũ về 0 ở vùng biên, làm tỷ số hằng số L/mu giữa cận trên và cận dưới trở nên rất lớn."
    ),
    21: (
        "Slide 21 khép lại Phần 3 với Lớp 5 của phương pháp: Câu chuyện Độ phức tạp Tính toán và Bộ nhớ (Computational & Space Complexity Story — chi "
        "tiết tại `docs/06_complexity.md`). Một câu hỏi thiết thực mà bất kỳ kỹ sư hệ thống nào cũng đặt ra là: Việc tính toán tốc độ học thích ứng ở mỗi "
        "bước thời gian t theo Algorithm 1, Proposition 4.3 hay Theorem 5.1 có làm chậm vòng lặp huấn luyện trực tuyến hay tiêu tốn thêm bộ nhớ không? Để "
        "trả lời chính xác, chúng ta phải tách bạch chi phí của Bản thân Bộ điều khiển Tốc độ học (Scheduler Overhead) khỏi chi phí của Vòng lặp Huấn luyện "
        "và Ước lượng độ trôi (SGD Update & Drift Tracking). Trước hết, xét một bước cập nhật Mini-batch SGD tiêu chuẩn trên không gian d chiều với kích "
        "thước batch B_t: việc tính tích vô hướng, đạo hàm mẫu trung bình và cập nhật vector trọng số tiêu tốn thời gian O(B_t * d) phép toán dấu phẩy động "
        "và bộ nhớ O(d) để lưu vector tham số theta_t (hoặc O(B_t * d) nếu giữ cả ma trận batch X_t trong RAM). Bây giờ, xét chi phí tăng thêm của ba bộ "
        "điều khiển tốc độ học trong bài báo khi đã có tín hiệu độ trôi gamma: Với Algorithm 1 (`LinearOptimalSchedule`), mỗi bước SGD giải phương trình "
        "ODE vô hướng 1 chiều cho biến v bằng ceil(1 / kappa) bước lặp con Euler; mỗi bước con chỉ thực hiện khoảng 10 phép cộng/nhân/căn bậc hai trên số "
        "thực đơn lẻ, nên độ phức tạp thời gian là O(ceil(1/kappa)) = O(1) (hoàn toàn độc lập với số chiều d và batch size B_t!) và bộ nhớ phụ là O(1) (chỉ "
        "lưu đúng 1 biến trạng thái vô hướng v). Với Proposition 4.3 (`ConvexSchedule`) và Theorem 5.1 (`NonConvexSchedule`), việc tính hai ngưỡng đóng "
        "tau_{1,t}, tau_{2,t} hoặc công thức (25) chỉ gồm một số hữu hạn phép tính logarit/căn bậc hai vô hướng, nên cả thời gian và bộ nhớ phụ của scheduler "
        "đều là O(1) tuyệt đối. Khi dùng thêm bộ ước lượng trượt EMA gamma_hat_t = beta*gamma_hat_{t-1} + (1-beta)||theta_t - theta_{t-1}||_2 (`EMADriftEstimator`), "
        "hệ thống chỉ cần lưu thêm bản sao theta_{t-1} (tốn O(d) bộ nhớ) và tính chuẩn hiệu hai vector (tốn O(d) thời gian), nhỏ hơn B_t lần so với chi phí "
        "O(B_t * d) của bước tính gradient! Thực nghiệm ở d = 512 xác nhận toàn bộ 2.000 bước huấn luyện hoàn tất chỉ trong vài giây trên CPU."
    ),
    22: (
        "Bước sang Phần 4 (Slides 22 đến 30) — Báo cáo Tái hiện Thực nghiệm Lab 1 — Slide 22 trình bày quyết định lựa chọn phạm vi thực nghiệm của "
        "nhóm và bảng ánh xạ tổng thể. Ngay từ buổi họp khởi động Task T3 (`docs/motivation_example.md`), nhóm chúng em đã thống nhất lựa chọn phạm vi "
        "FULL SCOPE: tái hiện trọn vẹn 100% toàn bộ các thí nghiệm chính trong bài báo (bao gồm Hồi quy tuyến tính cao chiều ở Mục 6.1 — Figure 4 và "
        "Figure 5-top; Hồi quy Logistic cao chiều ở Mục 6.1 — Figure 5-bottom; và Mạng nơ-ron sâu phân loại tế bào dòng chảy ở Phụ lục E — Figure 6), "
        "đồng thời thực hiện thêm các bài phân tích độ nhạy và nghiên cứu cắt bỏ (Sensitivity & Ablation Studies) vượt trên yêu cầu cơ bản. Có ba lý "
        "do thuyết phục để nhóm chọn Full Scope. Thứ nhất là tính toàn vẹn khoa học: ba thực nghiệm trong bài báo đại diện cho đúng ba chế độ cảnh "
        "quan hàm mất mát khác nhau về bản chất (hàm bậc hai giải tích được bằng SDE, hàm lồi tổng quát dùng quy tắc hai ngưỡng, và hàm không lồi nhiều "
        "cực tiểu cục bộ dùng Adam Restart); nếu bỏ qua bất kỳ thực nghiệm nào thì câu chuyện kiểm chứng lý thuyết sẽ bị khuyết một trụ cột. Thứ hai "
        "là tính khả thi về tài nguyên tính toán: toàn bộ các thí nghiệm đều là học trực tuyến một lượt (single-pass streaming) với chân trời T từ 600 "
        "đến 2.000 bước và batch size B_t in {64, 256}, nên nhóm có thể chạy lặp lại trên nhiều random seeds, nhiều giao thức đối chứng và kiểm tra "
        "chéo độc lập trên cả máy cá nhân lẫn Google Colab với thời gian rất nhanh. Thứ ba là tính tự động hóa dữ liệu 100%: dữ liệu hồi quy được sinh "
        "bằng công thức xoắn ốc giải tích (26), còn bộ dữ liệu 3.696 tế bào tuyến tụy chuột (`scvelo.datasets.pancreas()`) được trích xuất và đóng gói "
        "tự động trong `core/data.py`. Như bảng tổng hợp bên phải cho thấy, toàn bộ ba nhánh thực nghiệm của Khoa (TV1), Khôi (TV2) và Hòa (TV3) đều "
        "đã hoàn tất 100% mã nguồn, biểu đồ, bảng CSV và các runs xác thực trên Weights & Biases công khai."
    ),
    23: (
        "Slide 23 trình bày hạ tầng kỹ thuật, kỷ luật tái lập (Reproducibility) và hệ thống cổng kiểm chứng bốn giao thức (P / R / I / A) mà nhóm đã "
        "xây dựng để đảm bảo mọi con số báo cáo trong Slide Deck này đều có thể kiểm tra độc lập đến từng bit. Về môi trường phần mềm (Task T2 — "
        "`docs/T2_handover.md`), nhóm chuẩn hóa trên CPython 3.12 với tệp `requirements.txt` ghim cứng phiên bản của toàn bộ thư viện khoa học cốt lõi: "
        "PyTorch 2.11.0, NumPy 1.26.4, SciPy 1.15.2, Matplotlib 3.10.3, PyArrow 19.0.1 và Weights & Biases SDK 0.31.0. Script `scripts/check_environment.py` "
        "và bộ 102 unit tests trong thư mục `tests/` kiểm tra tự động từ tính cô lập của virtual environment cho đến tính đúng đắn của module theo dõi "
        "chung `core/tracking.py`. Mọi lần chạy thực nghiệm chính thức đều cố định 3 hạt giống ngẫu nhiên độc lập `seeds = [0, 1, 2]` qua hàm `set_seed` "
        "trong `core/utils.py` và đẩy trực tiếp metadata, lịch sử chuỗi thời gian cùng artifact lên W&B Public Dashboard tại địa chỉ "
        "`https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift` với nhãn bắt buộc `lab1`. Đặc biệt, như sơ đồ bên phải minh họa, để "
        "giải quyết triệt để sự khác biệt giữa công thức lý thuyết trong bài báo PDF và mã nguồn thực tế trong notebook của tác giả, nhóm thiết lập "
        "4 giao thức kiểm chứng minh bạch: Giao thức P (Project Protocol) đóng băng kết quả theo sát mô tả lý thuyết của bài báo (dùng ước lượng nhân "
        "quả EMA gamma_hat_t, phép chiếu Euclid và đánh giá rủi ro dư quần thể bằng tích phân cầu phương Gaussian độ chính xác 10^{-11}); Giao thức R "
        "(Reference Native Execution) chạy trực tiếp mã gốc của tác giả trên Google Colab để lấy đường cong chuẩn; Giao thức I (Independent Author-Aligned) "
        "tái hiện độc lập bằng NumPy khớp với quy ước của notebook gốc (kiểm tra khớp từng bước tới sai số 10^{-5}); và Giao thức A (Controlled Ablation) "
        "thay đổi từng yếu tố một (EMA, độ lệch chuẩn nhiễu, bước nhảy khởi tạo, bán kính chiếu) để cô lập nguyên nhân gây ra mọi độ lệch."
    ),
    24: (
        "Slide 24 trình bày kết quả tái hiện chính đầu tiên của Lab 1: Thực nghiệm 1 — Hồi quy Tuyến tính 2 chiều trên quỹ đạo xoắn ốc, tái hiện trọn "
        "vẹn Figure 4 trong Mục 6.1 (trang 9) của bài báo (mã nguồn `experiments/linear_regression.py`, cấu hình `configs/linear.yaml`, bảng số liệu "
        "`results/linear/fig4_summary.csv`). Thiết lập thí nghiệm tuân thủ chính xác bài báo: số chiều d = 2, chân trời T = 2000, kích thước mini-batch "
        "B_t = 256, phương sai nhiễu nhãn sigma^2 = 0.1, và tốc độ học tối đa trong Algorithm 1 là eps = eta_max = 0.5. Quỹ đạo nghiệm tối ưu Oracle "
        "theta*_t được sinh bởi Công thức (26) với bán kính lập phương r_{1,-1}(t)^3 (r giảm tuyến tính từ 1 xuống -1) và tần số góc k = 4: xuất phát tại "
        "(1, 0) ở t = 1, xoắn ốc đi vào gốc tọa độ (0, 0) quanh t = 1000, rồi xoắn ngược ra ngoài về (-1, 0) ở t = 2000. Để khảo sát sự khác biệt giữa "
        "dịch chuyển liên tục và dịch chuyển nhảy bậc rời rạc, quỹ đạo được lấy mẫu thưa (downsample) theo theta_hat*_t = theta*_{ceil(t/ell)*ell} với ba "
        "mức ell in {1, 4, 16} tương ứng với ba cột từ trái sang phải trên hình `figures/fig4_linear_trajectory_and_lr.png`. Hàng trên biểu diễn quỹ đạo "
        "học được theta_t của SGD (màu cam) bám sát quỹ đạo Oracle theta_hat*_t (màu xanh dương) trong cả ba chế độ. Hàng dưới biểu diễn lịch trình tốc "
        "độ học eta_t do Algorithm 1 sinh ra: (1) Khi ell = 1 (trôi liên tục, 1999 bước có shift, W&B run `yaomsd9p`), đường cong eta_t có dạng chữ V "
        "trơn tru: giữ ở mức trần 0.5 ở giai đoạn đầu và cuối (khi bán kính lớn nên vận tốc gamma_t lớn), và hạ thấp xuống cực tiểu 0.0068 quanh t = 1000 "
        "(khi xoắn ốc đi chậm ở sát gốc tọa độ làm gamma_t ≈ 0), đạt Regret cuối cực thấp 0.916 ± 0.045. (2) Khi tăng lên ell = 4 (499 đợt nhảy, run "
        "`id9my7jj`, Regret 0.923 ± 0.043) và đặc biệt là ell = 16 (124 đợt nhảy, run `zrhn9935`), hiện tượng dao động răng cưa (oscillation) xuất hiện "
        "đúng như bài báo mô tả: trong 15 bước tĩnh giữa hai lần nhảy (gamma_t = 0), Algorithm 1 giảm nhanh eta_t xuống 0.0065 để hội tụ mịn; nhưng ngay "
        "tại bước thứ 16 khi theta_hat*_t nhảy sang điểm mới (gamma_t > 0), Algorithm 1 lập tức kích hoạt xung bước nhảy vọt trở lại trần eta_max = 0.5 "
        "(tạo ra 74 đỉnh chạm trần), giữ cho tổng Regret chỉ ở mức 1.218 ± 0.054."
    ),
    25: (
        "Tiếp tục với Thực nghiệm 1 (Hồi quy Tuyến tính), Slide 25 trình bày kết quả tái hiện Figure 5 (top-left và top-right) khi mở rộng bài toán "
        "lên không gian cao chiều d in {2, 8, 32, 128, 512}, kèm bảng đối chiếu với các lịch trình baseline và bản ước lượng trực tuyến EMA (số liệu tại "
        "`results/linear/fig5_top_summary.csv` và `linear_baselines_comparison.csv`). Trong thí nghiệm này, ta giữ cố định bước rời rạc hóa xoắn ốc ell = 8, "
        "batch size B_t = 256, chân trời T = 2000, và đặt trần tốc độ học co giãn theo số chiều eta_max = 1 / sqrt(d) (từ 0.7071 ở d = 2 xuống 0.0442 ở "
        "d = 512). Toàn bộ 15 lần chạy chính thức (5 số chiều x 3 seeds {0, 1, 2}) đã được đẩy lên W&B Public dưới tài khoản `nkhoa8508`. Nhìn vào đồ thị "
        "bên trái của hình `figures/fig5_top_linear_regret_and_path.png` (trục tung logarit 10^{-1} đến 10^5), cả 5 đường cong Cumulative Regret tái hiện "
        "khớp hoàn hảo cả về hình dáng lẫn giá trị tuyệt đối so với Figure 5 (top-left) trong bài báo với độ lệch dưới 1.2%: tại T = 2000, Regret đạt "
        "0.89 ± 0.04 ở d = 2 (run `jnphcs45`, bài báo ~0.9), 13.40 ± 0.13 ở d = 8 (run `6kznv467`, bài báo ~13.3), 792.18 ± 8.50 ở d = 32 (run `1fojwok2`, "
        "bài báo ~800), 7049.30 ± 16.76 ở d = 128 (run `hupxdej7`), và 33034.47 ± 7.76 ở d = 512 (run `c82qd3ou`). Hình dạng đặc trưng 'tăng nhanh ở đầu "
        "— đi ngang ở giữa — tăng lại ở cuối' của cả 5 đường cong phản ánh chính xác động học xoắn ốc: cú nhảy đầu tiên là khi mô hình đi từ điểm khởi tạo "
        "theta_0 = 0 ra biên xoắn ốc; đoạn đi ngang từ t = 500 đến 1500 ứng với giai đoạn bán kính r(t)^3 ≈ 0 quanh gốc tọa độ; và đoạn dốc lên cuối ứng "
        "với khi xoắn ốc tăng tốc thoát ra ngoài. Đồ thị bên phải xác nhận ngay cả ở d = 128, hình chiếu 2 tọa độ đầu (theta_t(1), theta_t(2)) của SGD vẫn "
        "bám sát từng vòng xoắn của Oracle. Bảng đối chiếu Baselines chứng minh hai kết quả quan trọng: (1) Algorithm 1 vượt trội hoàn toàn lịch trình giảm "
        "dần 1/t và 1/sqrt(t) trên mọi số chiều (ở d = 8, Regret giảm từ 358.43 của 1/t xuống 13.40); và (2) Khi ẩn độ trôi thực gamma_{t-1} và thay bằng "
        "ước lượng trực tuyến EMA gamma_hat_t (Remark 3.5), Algorithm 1 đạt hiệu năng trùng khít với bản Oracle (ở d = 128: EMA đạt 7049.49 so với 7049.30)."
    ),
    26: (
        "Slide 26 trình bày kết quả tái hiện Thực nghiệm 2 — Hồi quy Logistic cao chiều với hàm mất mát Binary Cross-Entropy (Figure 5-bottom trong "
        "Mục 6.1, do Khôi phụ trách, báo cáo tại `docs/results/cp04-cp05/RESULTS.md` và `docs/results/convex-author-v1/RESULTS.md`). Ở mỗi bước t = 1..2000, "
        "ma trận đặc trưng X in R^{256 x d} được sinh từ N(0, I_d) và nhãn mục tiêu mềm sinh qua hàm phi tuyến y = sigmoid(X * theta*_t + eps). Theo gợi "
        "ý không tham số cho mô hình tuyến tính tổng quát ở trang 9 bài báo, lịch trình hai ngưỡng Proposition 4.3 (`ConvexSchedule`) được cấu hình với "
        "hằng số trơn L = 1/4 = 0.25, phương sai gradient sigma^2 = d / 4, và đường kính miền tham số D_max = d. Trong giao thức đóng băng CP04 của nhóm "
        "(45 runs chính thức trên W&B: 5 số chiều d in {2, 8, 32, 128, 512} x 3 phương pháp `convex`, `constant`, `inverse_sqrt` x 3 seeds {0, 1, 2}), bộ "
        "học dùng tín hiệu độ trôi ước lượng nhân quả EMA (beta = 0.9), bán kính chiếu R = d/2 và đánh giá bằng rủi ro dư tích lũy trên quần thể chính xác "
        "B (tính qua tích phân cầu phương Gaussian 1 chiều dung sai 10^{-11}). Bảng số liệu trên slide cho thấy bức tranh khoa học rõ nét: Trên bốn số "
        "chiều từ nhỏ đến lớn d = 2, 8, 32 và 128, lịch trình hai ngưỡng Proposition 4.3 (`convex`) chiến thắng áp đảo cả hai lịch trình baseline với độ "
        "lệch chuẩn cực nhỏ: tại d = 2, `convex` đạt B = 0.34 ± 0.01 so với 7.25 của `constant` và 29.46 của `inverse_sqrt` (W&B run `38easaio`); tại d = 8, "
        "`convex` đạt 2.03 ± 0.01 so với 53.54 và 88.42 (run `c5mhu323`); tại d = 32, `convex` đạt 40.04 ± 0.27 so với 180.43 và 201.19 (run `52miko4w`); "
        "và tại d = 128, `convex` đạt 233.52 ± 0.59 so với 334.95 và 354.63 (run `br6md2c7`), đồng thời quỹ đạo 2 tọa độ đầu ở d = 128 bám sát vòng xoắn "
        "ốc tham chiếu. Tuy nhiên, tại chiều siêu cao d = 512 (run `lih7ibh8`), nhóm phát hiện hiện tượng đảo chiều xuất hiện nhất quán trên cả giao thức "
        "dự án P lẫn giao thức đối chuẩn tác giả I: `convex` chịu rủi ro dư B = 1385.01 ± 7.06 (và M2 = 1205.49 ± 2.17 trong giao thức I), cao hơn mức "
        "504.90 của `constant` (LR = 0.1). Chẩn đoán động học (`d512_diagnostics.png`) giải thích chính xác nguyên nhân: ở d = 512, đặt D_max = d = 512 làm "
        "hệ số b_{1,t} = gamma_t^2 + 1024*gamma_t phình to gấp hàng trăm lần khoảng cách thực tế ||theta_t - theta*_t||_2 (chỉ ~4–7), khiến ngưỡng dưới "
        "tau_{1,t} đẩy tốc độ học lên sát trần 1/L = 4.0 và gây tích lũy phương sai gradient cao chiều."
    ),
    27: (
        "Slide 27 mở đầu Thực nghiệm 3 — bài toán ứng dụng y sinh thực tế Phân loại Tế bào Dòng chảy Trực tuyến bằng Mạng Nơ-ron Sâu (Flow Cytometry "
        "5-Layer MLP trong Phụ lục E, trang 22–23, mã nguồn tại `core/data.py` và `experiments/neural_net.py`). Trước hết, chúng ta cần hiểu bài toán "
        "sinh học mà thí nghiệm này mô phỏng. Trong kỹ thuật Flow Cytometry hiện đại (Li et al., 2019; Hu et al., 2020), các tế bào lấy từ mô sinh học "
        "được gắn một số lượng nhỏ chất đánh dấu huỳnh quang liên kết với các trình tự RNA đặc hiệu, rồi đi qua chùm tia laser để phân loại thời gian thực. "
        "Vì gắn quá nhiều chất huỳnh quang sẽ gây độc và làm biến đổi chức năng sinh lý của tế bào, hệ thống sử dụng kiến trúc hai tầng Teacher–Student: "
        "một mạng nơ-ron gọn nhẹ (Student MLP) chỉ dùng 10 tín hiệu RNA huỳnh quang để phân loại tức thời từng tế bào khi đi qua ống dẫn, sau đó một mạng "
        "tích chập quy mô lớn không cần chất đánh dấu (Teacher CNN) chạy song song trên máy chủ sẽ gửi nhãn chuẩn (ground-truth feedback) về để cập nhật "
        "trực tuyến mạng Student. Sự dịch chuyển phân phối xảy ra tự nhiên khi ống lấy mẫu quét dọc từ vùng đầu tuyến tụy (anterior — nơi tế bào ống dẫn "
        "không tiết chiếm đa số) sang vùng đuôi tuyến tụy (posterior — nơi tế bào đảo tụy tiết hormone chiếm ưu thế). Để tái hiện kịch bản này, module "
        "`core/data.py` tải bộ dữ liệu RNA đơn bào tuyến tụy chuột thực tế từ `scvelo.datasets.pancreas()` (Bastidas-Ponce et al., 2019 [5], gồm N = 3.696 "
        "tế bào). Sau bước lọc `filter_genes(min_shared_counts=30000)` và chuẩn hóa `normalize_per_cell()`, dữ liệu thu về chính xác 10 marker genes: "
        "Pyy, Meg3, Malat1, Gcg, Gnas, Actb, Ghrl, Rps3, Ins2 và Hspa8. Hình bên phải (`figures/fig6_left_cytometry_embedding.png`) là biểu đồ nhúng 2 chiều "
        "UMAP của 3.696 tế bào do nhóm tái hiện Figure 6 (left), hiển thị lộ trình biệt hóa từ cụm tế bào ống dẫn `Ductal` (916 tế bào, xanh lá nhạt bên "
        "trái) qua các trạng thái tiền thân để phân nhánh thành các tế bào nội tiết bên phải gồm `Alpha` (481 tế bào), `Beta` (591 tế bào) và `Delta` (70 "
        "tế bào). Luồng dữ liệu trực tuyến T = 600 bước (batch size B_t = 64, nhiễu nhân U ~ Uniform[0.9, 1.1]) đi qua 4 pha phân phối trên 4 lớp (Alpha, "
        "Beta, Delta, Ductal): Pha 1 (bước 1–100) chứa 100% Ductal; Pha 2 (101–200) có 10% Delta, 90% Ductal; Pha 3 (201–400) có 10% Alpha, 20% Delta, "
        "70% Ductal; và Pha 4 (401–600) đảo chiều mạnh sang 30% Alpha, 50% Beta, 10% Delta và 10% Ductal."
    ),
    28: (
        "Trên nền tảng luồng dữ liệu tế bào tụy 4 pha ở Slide 27, Slide 28 trình bày kết quả tái hiện Figure 6 (right) — đánh giá hiệu năng của mạng "
        "`CytometryMLP` 5 tầng (`10 -> 64 -> 32 -> 16 -> 8 -> 4`, kích hoạt ELU ở các tầng ẩn, khởi tạo Glorot Uniform, điều chuẩn L2 = 10^{-4} và hàm "
        "mất mát Categorical Cross-Entropy từ logits) dưới bốn cấu hình bộ tối ưu Adam có và không có cơ chế khởi động lại (Adam Restart) tại các mốc "
        "dịch chuyển phân phối t in {100, 200, 400}. Toàn bộ 12 lần chạy chính thức (4 phương pháp x 3 seeds {0, 1, 2}) đã được đẩy lên W&B Public với "
        "bảng số liệu tại `results/nn/fig6_summary.csv` và đồ thị `figures/fig6_right_adam_restart_regret.png` bên phải. Kết quả thực nghiệm của nhóm tái "
        "hiện chính xác 100% cả thứ tự xếp hạng lẫn ba phát hiện khoa học cốt lõi của Phụ lục E.2: Phát hiện thứ nhất là sự tồn tại của 'điểm ngọt' (sweet "
        "spot) cho tốc độ học ban đầu: khảo sát trên lưới eta_0 in {0.001, 0.003, 0.01, 0.03, 0.1} (`fig6_lr_sweep.csv`) cho thấy eta_0 = 0.01 đạt tổng "
        "Cumulative Regret thấp nhất; khi đặt eta_0 = 0.1 quá lớn, mô hình phân kỳ nặng (Regret vọt lên 7441.43), trong khi mức nhỏ eta_0 = 0.001 (Regret "
        "175.21) và 0.003 thích ứng quá chậm sau mỗi lần chuyển pha. Phát hiện thứ hai — thông điệp quan trọng nhất của Figure 6 — là lợi ích vượt trội "
        "của cơ chế Adam Restart tại cả hai mức tốc độ học: với eta_0 = 0.01, việc khởi động lại bộ đệm moment của Adam tại t = 100, 200, 400 "
        "(`adam-restart:0.01`, đường màu cam, W&B runs `3beskj3q`, `lectj13d`, `jfd7ui7f`) đạt Cumulative Regret thấp nhất toàn cục là 131.59 ± 5.07 (độ "
        "chính xác trung bình 91.4%), giảm 12.0% Regret so với `adam:0.01` không restart (149.52 ± 10.69, đường xanh dương, runs `4p6kun1x`, `p1rgnctj`, "
        "`7dvnzczn`). Hiệu quả này càng ấn tượng ở eta_0 = 0.03: `adam-restart:0.03` (đường màu đỏ, runs `jzrtmtez`, `trjq8r5m`, `bion3ocx`) kéo giảm "
        "Cumulative Regret từ 315.07 ± 75.71 của `adam:0.03` (đường xanh lá, runs `h91vgmz5`, `yvc2rruz`, `h1qd3lha`) xuống chỉ còn 180.30 ± 26.49 — giảm "
        "tới 42.8% tổng độ tiếc nuối và nâng độ chính xác từ 84.7% lên 90.1%! Phát hiện thứ ba giải thích cơ chế vật lý của đường xanh lá (`adam:0.03`): "
        "sau 200 bước đầu chỉ học chủ yếu lớp Ductal, bộ tích lũy moment bậc hai v_t của Adam đã lớn lên và kìm hãm bước nhảy trong một hố cực tiểu cục "
        "bộ sâu; khi Pha 3 và 4 đưa tế bào Alpha và Beta vào, `adam:0.03` không restart bị mắc kẹt hoàn toàn (độ dốc Regret 0.74/bước), trong khi "
        "`adam-restart:0.03` xóa sạch v_t cũ để khôi phục bước nhảy lớn, thoát khỏi cực tiểu cũ ngay lập tức."
    ),
    29: (
        "Slide 29 trình bày phần Phân tích Mở rộng (Bonus Sensitivity & Controlled Ablation Analysis — Task T17, do Khôi thực hiện tại `experiments/sensitivity.py` "
        "và `experiments/convex_author_analysis.py`). Mục tiêu của nghiên cứu mở rộng này là kiểm tra độ bền (robustness) của lịch trình tốc độ học thích "
        "ứng khi các tham số môi trường thay đổi ngoài cấu hình chuẩn của bài báo, đồng thời thực hiện các can thiệp có đối chứng (Ablation A1–A4) để bóc "
        "tách vai trò của từng giả thiết thiết kế. Trong tập thực nghiệm thứ nhất (Giao thức CP05 — 15 W&B runs đã xác minh tại số chiều cao d = 128, chân "
        "trời T = 2000), nhóm quét độc lập hai nhân tố môi trường quanh cấu hình chuẩn baseline (ell = 8, sigma = sqrt(0.1) ≈ 0.316, rủi ro dư B = 233.52 "
        "± 0.59, W&B run `7i4l64n4`). Nhân tố thứ nhất là tần suất và tổng ngân sách dịch chuyển phân phối qua bước lấy mẫu thưa ell in {16, 8, 1}: khi "
        "phân phối dịch chuyển thưa hơn ở ell = 16 (124 đợt nhảy, tổng độ dài đường đi sum gamma_t = 339.3, run `ctn81a74`), rủi ro dư giảm xuống 141.03 ± "
        "0.86; ngược lại khi phân phối trôi liên tục ở mọi bước với ell = 1 (1999 đợt dịch chuyển, tổng đường đi sum gamma_t = 1848.5 tăng gấp 2.68 lần "
        "baseline, run `adva0mi7`), rủi ro dư chỉ tăng lên 330.32 ± 2.29 (tăng 1.41 lần — chậm hơn nhiều so với tốc độ tăng của tổng độ trôi!). Nhân tố "
        "thứ hai là độ lệch chuẩn nhiễu nhãn sigma in {0.0, 0.316, 1.0} (biểu đồ bên phải `cp05_noise_sensitivity.png`): khi không có nhiễu nhãn (sigma = "
        "0.0, run `lbwkwdmc`), B đạt 224.22 ± 0.67; ở mức nhiễu chuẩn 0.316 chỉ tăng nhẹ thêm +9.30 đơn vị lên 233.52; và ngay cả khi đẩy nhiễu nhãn tăng "
        "gấp hơn 3 lần lên sigma = 1.0 (phương sai tăng gấp 10 lần, run `lihumbm1`), lịch trình thích ứng vẫn giữ sự ổn định với B = 282.45 ± 0.80. Trong "
        "tập thực nghiệm thứ hai (Giao thức Ablation A1–A4 gồm 36 runs trên d in {2, 32, 512}), nhóm phát hiện hiện tượng lý thuyết quan trọng ở can thiệp "
        "A1 (thay forward shift bằng ước lượng nhân quả EMA nhưng giữ khởi tạo bước 0 của notebook gốc nơi gamma_0 = 0 => eta_0 = 0): khi cả tốc độ học "
        "khởi tạo lẫn trạng thái EMA đều bằng 0, theta_1 = theta_0 làm bước nhảy mô hình tiếp tục bằng 0 và tạo ra 'trạng thái hấp thụ tại 0' (zero-update "
        "absorbing state) trên cả 9 ca! Phát hiện này chứng minh vì sao khởi tạo bước nhảy dương eta_init = 1.0 trong giao thức dự án P là điều kiện tiên "
        "quyết để kích hoạt vòng lặp học nhân quả."
    ),
    30: (
        "Slide 30 tổng hợp kết quả của Task T18 — Phân tích Độ lệch (Deviation Analysis) và Kiểm toán Mã nguồn Gốc của Tác giả (`docs/original_code_comparison.md` "
        "và `docs/convex_author_v1/SOURCE_AUDIT.md`). Theo tinh thần hướng dẫn Lab 1, mục tiêu cao nhất của việc tái hiện thực nghiệm không chỉ là chạy ra "
        "con số gần giống bài báo, mà là giải thích chính xác tại sao có sự chênh lệch giữa lý thuyết trong PDF, mã nguồn gốc của tác giả và mã nguồn tái "
        "tạo của nhóm. Qua kiểm toán trực tiếp hai tệp mã nguồn trong repository `fahrbach/learning-rate-schedules` (commit `1b3c0d45`), nhóm đã phát hiện "
        "và định lượng ba nguồn gây sai lệch cốt lõi trên bảng đối chiếu của slide. Phát hiện nổi bật nhất nằm ở Thực nghiệm 3 (Flow Cytometry MLP — "
        "`src/cytometry/main.py`, SHA `742f84e8`, dòng 140–150): Trong bài báo (Phụ lục E.2, trang 23), tác giả phát biểu rõ là khởi động lại bộ tối ưu Adam "
        "tại đầu mỗi đợt dịch chuyển phân phối (tức khi chuyển giữa các pha trên mảng 6 khối `distribution_types = [0, 1, 2, 2, 3, 3]`, ứng với các bước "
        "t = 100, 200, 400 nơi `dist[phase] != dist[phase - 1]`). Thế nhưng tại dòng 144 của `src/cytometry/main.py`, tác giả lại viết nhầm toán tử so sánh: "
        "`if phase == 0 or distribution_type == distribution_types[phase - 1]:` (dùng `==` thay vì `!=`). Hậu quả là trong code gốc, mô hình KHÔNG restart "
        "khi phân phối đổi pha ở t = 100, 200, 400, mà lại đi restart ở giữa Pha 3 (t = 300, khi `dist[3] == dist[2] == 2`) và giữa Pha 4 (t = 500, khi "
        "`dist[5] == dist[4] == 3`)! Như biểu đồ bên phải (`figures/fig6_lr_sweep_and_author_bug.png`) chứng minh trên 3 seeds: khi nhóm sửa lại đúng điều "
        "kiện `!=` theo bài báo, Cumulative Regret của `adam-restart:0.03` giảm mạnh từ 302.33 ± 92.09 (chạy theo lỗi dòng 144) xuống chỉ còn 180.30 ± 26.49 "
        "(giảm 40.4% sai số!), và của `adam-restart:0.01` giảm từ 146.53 xuống 131.59. Phát hiện thứ hai nằm ở Thực nghiệm 2 (Convex Logistic — Cell 8 và "
        "Cell 23 của `regression_experiments.ipynb`): notebook gốc tính hiệu số mất mát trực tiếp trên mẫu đối với vector sinh dữ liệu (`M0`), dùng độ trôi "
        "nhìn trước một bước `||v[t+1] - v[t]||`, không chiếu `project_to_ball` và ép kiểu hỗn hợp `float32/float64`. Khi nhóm chạy bản đối chuẩn Giao thức "
        "I khớp đúng các quy ước này, đường cong tốc độ học khớp từng bước một trên toàn bộ 2.000 bước và giá trị M0 cuối ở d = 2, 8, 32 khớp với output lưu "
        "trong notebook tác giả tới sai số tuyệt đối tối đa chỉ 2.48 x 10^{-5} (`fidelity.csv`)!"
    ),
    31: (
        "Bước sang Phần 5 — Tổng kết và Định hướng Mở rộng — Slide 31 đúc kết những bài học quan trọng nhất từ cả góc độ lý thuyết (Milestone 2) lẫn "
        "thực nghiệm (Lab 1), đồng thời chỉ ra ba hạn chế cốt lõi của bài báo để làm bàn đạp trực tiếp cho phần Phê bình Lý thuyết và Poster ở Milestone 3 "
        "+ Lab 2 cuối kỳ. Về thông điệp khoa học chính, nghiên cứu này đã thay đổi cách nhìn truyền thống về việc lập lịch tốc độ học: trong môi trường học "
        "trực tuyến không dừng (non-stationary online learning), việc hội tụ về một điểm cố định không còn là ưu tiên số một; thay vào đó, tốc độ học eta_t "
        "đóng vai trò là van điều tiết mức độ khám phá không gian trọng số và bắt buộc phải TĂNG LÊN (hoặc khởi động lại bộ đệm moment bậc hai như trong "
        "Adam Restart) mỗi khi phân phối dữ liệu dịch chuyển mạnh, rồi mới giảm dần theo quy luật O(1/t) khi môi trường tạm thời ổn định. Tuy nhiên, qua "
        "quá trình chứng minh lại công thức và kiểm chứng thực nghiệm nghiêm ngặt trên cả 3 bài toán, nhóm chúng em xác định được ba giới hạn lý thuyết rõ "
        "ràng của bài báo: Giới hạn thứ nhất nằm ở mô hình SDE của Hồi quy Tuyến tính (Mục 3): toàn bộ hệ phương trình moment ODE (10)–(11) dựa vào giả "
        "định phân phối đặc trưng x_{t,k} ~ N(0, I_d) đẳng hướng và bất biến theo thời gian (chỉ mô hình hóa Label Shift). Nếu xảy ra Covariate Shift (ma "
        "trận hiệp phương sai đặc trưng Sigma_t có phổ trị riêng lệch nhau và xoay theo thời gian), một tốc độ học vô hướng duy nhất eta_t không thể tối ưu "
        "đồng thời trên cả phương có trị riêng lớn lẫn phương có trị riêng nhỏ. Giới hạn thứ hai nằm ở quy tắc hai ngưỡng Proposition 4.3 cho Hàm Lồi (Mục "
        "4): việc thay thế khoảng cách thực tế D_t bằng đường kính cố định D_max = d làm cận trên bị lỏng nghiêm trọng khi số chiều d = 512 lớn, đẩy ngưỡng "
        "tốc độ học lên quá cao và làm phương pháp thích ứng thua hằng số `constant-0.1` như nhóm đã chứng minh ở Slide 26. Giới hạn thứ ba nằm ở bộ ước "
        "lượng độ trôi thực nghiệm EMA gamma_hat_t = ||theta_t - theta_{t-1}||_2 (Remark 3.5 & 4.4): khi kích thước batch B_t nhỏ hoặc nhiễu sigma lớn, nó "
        "bị trộn lẫn với bước nhảy do nhiễu gradient, gây hiện tượng báo động giả (false-positive shift) làm tốc độ học không thể hạ xuống mức tối ưu."
    ),
    32: (
        "Slide 32 khép lại bài báo cáo giữa kỳ bằng bảng phân công đóng góp chi tiết của từng thành viên và danh mục đầy đủ các liên kết tài nguyên "
        "công khai để thầy và hội đồng chấm bài có thể truy cập, kiểm chứng bất kỳ lúc nào. Trong suốt quá trình thực hiện đồ án, cả ba thành viên đều tuân "
        "thủ nghiêm ngặt quy tắc liêm chính học thuật: 100% mã nguồn trong thư mục `core/`, `experiments/`, `scripts/` và `tests/` được viết mới từ công "
        "thức toán học trong bài báo kèm docstring đầy đủ; 100% sơ đồ kiến trúc và biểu đồ thực nghiệm trên cả 32 slides đều do nhóm tự sinh bằng code "
        "Python/Matplotlib từ dữ liệu chạy thật; và cả nhóm đã thực hiện kiểm tra chéo (Task T19) để đảm bảo mỗi thành viên đều nắm vững và có thể bảo vệ "
        "toàn bộ ba phần của bài báo tại buổi thi vấn đáp Poster cuối kỳ. Về phân công cụ thể: Thành viên 1 — Khoa (Trưởng nhóm) phụ trách xây dựng bối "
        "cảnh và phân tích tiền nhiệm (T4), bảng ký hiệu và sơ đồ pipeline hai giai đoạn Inference/Training (T5), khung mã nguồn dùng chung `core/` (T6), "
        "lý thuyết SDE/HJB và lớp `LinearOptimalSchedule` (T7), thực nghiệm Hồi quy Tuyến tính tái hiện Figure 4 và Figure 5-top với 19 W&B runs online "
        "(T14), kiểm tra chéo độc lập (T19) và biên soạn 14 slides (Slides 3–7, 10–15, 24–25, 31). Thành viên 2 — Khôi phụ trách thiết lập môi trường chuẩn "
        "hóa CPython 3.12 và module `core/tracking.py` (T2), phân tích đóng góp, tính mới, bảng định lý và độ phức tạp tính toán (T8, T10), lý thuyết hàm "
        "lồi và lớp `ConvexSchedule` (T9), thực nghiệm Hồi quy Logistic Figure 5-bottom cùng phân tích độ nhạy/ablation qua hai giao thức P và I/A với hơn "
        "140 runs đã kiểm chứng (T15, T17) và biên soạn 9 slides (Slides 8–9, 16–17, 20–21, 23, 26, 29). Thành viên 3 — Hòa (cùng sự hỗ trợ hoàn thiện cuối "
        "kỳ của nhóm) phụ trách phần lý thuyết hàm không lồi, tải dữ liệu `scvelo` 10-gene tế bào tụy `core/data.py` (T11), mạng `CytometryMLP` 5 tầng kèm "
        "cơ chế Adam Restart (T12), thực nghiệm Figure 6 với 12 W&B runs online (T16), kiểm toán mã nguồn gốc của tác giả phát hiện lỗi dòng 144 (T18) và "
        "9 slides (Slides 1–2, 18–19, 22, 27–28, 30, 32). Toàn bộ mã nguồn và hình vẽ được lưu trữ công khai tại GitHub `https://github.com/nvk1701/github-pj` "
        "và W&B `https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift`. Nhóm chúng em xin chân thành cảm ơn thầy đã theo dõi!"
    ),
}


def main() -> None:
    md_lines = [
        "# Bộ Speaker Notes Hoàn Chỉnh Cho Toàn Bộ 32 Slides (Chuẩn <= 500 từ/slide)",
        "",
        "Tài liệu này cung cấp lời thuyết minh chi tiết (**330 – 480 từ/slide**, tuân thủ nghiêm ngặt giới hạn **<= 500 từ/slide** của đề bài khi chấm không thuyết trình miệng) cho toàn bộ **32 slides** của nhóm. Mỗi đoạn thuyết minh đều giải thích trọn vẹn ý nghĩa mọi công thức, biến số, hình vẽ, số liệu thực nghiệm, mã W&B run và câu chuyển tiếp giữa các trang.",
        "",
        "| Slide | Phụ trách | Tiêu đề Slide | Số từ (Words) | Trạng thái (<= 500 từ) |",
        "| :---: | :--- | :--- | :---: | :---: |",
    ]

    for spec in SLIDES_SPEC:
        s_num = int(spec["num"])
        full_note = FULL_SPEAKER_NOTES[s_num]
        word_count = len(full_note.split())
        if not (250 <= word_count <= 490):
            raise ValueError(f"Slide {s_num} có số từ {word_count} nằm ngoài khoảng chuẩn [250, 490]!")
        spec["notes"] = full_note
        md_lines.append(
            f"| **Slide {s_num:02d}** | {spec['owner']} | {spec['title']} | **{word_count} từ** | ĐẠT (✔) |"
        )

    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

    for spec in SLIDES_SPEC:
        s_num = int(spec["num"])
        full_note = str(spec["notes"])
        word_count = len(full_note.split())
        md_lines.extend(
            [
                f"## Slide {s_num:02d} — {spec['title']}",
                f"- **Phần**: `{spec['section']}` | **Phụ trách**: **{spec['owner']}** | **Độ dài Speaker Notes**: **`{word_count} / 500 từ`**",
                "",
                "```text",
                full_note,
                "```",
                "",
            ]
        )

    out_md = PROJECT_ROOT / "docs" / "SPEAKER_NOTES_32_SLIDES.md"
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[OK] Wrote full 32-slide Speaker Notes markdown to: {out_md}")

    # Cập nhật luôn file `slides/template.pptx` chứa trọn bộ Speaker Notes chi tiết này
    generate_template_pptx(PROJECT_ROOT / "slides" / "template.pptx")


if __name__ == "__main__":
    main()
