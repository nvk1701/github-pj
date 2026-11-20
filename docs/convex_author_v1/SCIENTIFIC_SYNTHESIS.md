# Kết luận khoa học và bàn giao tích hợp

Proposition 4.3 chọn LR bằng cách tối ưu từng bước một cận kỳ vọng cho projected SGD. Notebook áp dụng quy tắc hai ngưỡng vào benchmark tổng hợp với shift của generating reference sắp tới, LR khởi đầu 0, SGD không chiếu và M0 thực nghiệm. Colab tái lập đủ năm ca R. F2/F3 đối chiếu 2.000 batch trên cùng tensor ở d2/8/32, cho LR khớp chính xác và sai khác M0 tích lũy tối đa 2.48e−5. Dữ liệu này kiểm chứng hành vi thực thi đã định nghĩa, còn Theorem 4.2 vẫn phụ thuộc các giả thiết về miền, smoothness, phương sai và thông tin drift. [Source audit](SOURCE_AUDIT.md) · [Fidelity](../results/convex-author-v1/data/fidelity.csv)

P đặt câu hỏi khác về ứng dụng online, khi learner chỉ quan sát lịch sử của mình. EMA là estimator nhân quả được gợi ý ở Remark 4.4 và phụ thuộc động lực learner. Khi chuyển riêng sang EMA với LR native ban đầu 0, toàn bộ chín ca rơi vào trạng thái zero-update. Thay LR đầu giúp giảm gap nhẹ, phép chiếu không tác động đến update trong mẫu, và noise tăng làm đổi phân phối lẫn gap. [Thiết kế đối chứng](PROTOCOL.md) · [Paired ablation](../results/convex-author-v1/data/ablation_summary.csv)

M2 đo chênh lệch đến population minimizer gần đúng trên cùng phân phối. Nó giúp tách thay đổi do comparator khỏi thay đổi learner. Khi nhãn sigmoid có Gaussian noise, conditional mean bị làm trơn và optimum radial co lại. M0 phù hợp kiểm notebook. M1 phù hợp tách hiệu chỉnh comparator. M2 phù hợp đối chiếu expected-risk trên từng phân phối đã nêu. [Metric validation](METRIC_VALIDATION.md)

Ở d512, constant có M2 thấp hơn Convex trên cả ba seed I dù Convex được cấp forward reference shift. A giữ được khoảng cách lớn khi thay riêng LR đầu hoặc chiếu. Động học ít drift vẫn có LR gần 4 và update lớn. Các kết quả này ghi nhận hiện tượng trong thiết kế author-aligned, còn nguyên nhân của CP04 bị giới hạn bởi khác biệt phân phối, phase grid và tín hiệu drift. [Kết quả](../results/convex-author-v1/RESULTS.md)

## Ánh xạ cho Sheet 4 và các issue

| Issue / ô điền | Nguồn có thể trích | Điều kiện tích hợp |
|---|---|---|
| T9 và T11, phương trình/thiết lập | [Bảng paper/author/I/P](SOURCE_AUDIT.md) | Ghi rõ unknown của runtime, T18 yêu cầu chạy Colab riêng |
| T15, cột W&B và thống kê P | [Provenance 60 run](../results/cp04-cp05/data/run_provenance.csv) · [45 ca mean/SD](../results/cp04-cp05/data/cp04_method_summary.csv) | Giữ số đã đóng băng |
| T15, giá trị paper và relative deviation P | N/A | P báo M2/B và khác phân phối so với R/M0 |
| T15, phụ lục bổ sung | [I/A theo seed](../results/convex-author-v1/data/per_run.csv) · [Manifest](../results/convex-author-v1/manifest.json) | Thêm bảng R/I/A sau freeze, cần lead duyệt |
| T18, đề xuất cho `docs/08_original_code.md` | [Source audit](SOURCE_AUDIT.md) · [Colab contract](AUTHOR_NATIVE_EXECUTION.md) | Hòa duyệt/đối chiếu trước khi tích hợp |
| T19 | [Clean clone và đối chiếu](REPRODUCE.md) | Teammate chạy và tự ký bằng chứng |
| T22 | [Slide handoff](SLIDE_HANDOFF.md) | Khôi/Hòa và lead duyệt vị trí |

Workbook và deck không có trong workspace đã kiểm tra, nên chưa xác minh allocation Sheet 3 của slide 8–9 và chưa tạo deck dựa trên template. Slide handoff dùng các vị trí do nhiệm vụ cung cấp.
