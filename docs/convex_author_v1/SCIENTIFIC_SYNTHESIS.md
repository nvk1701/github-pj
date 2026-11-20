# Kết luận khoa học và bàn giao tích hợp

Proposition 4.3 chọn LR bằng cách tối ưu từng bước một cận kỳ vọng cho projected SGD. Author notebook mang quy tắc hai ngưỡng vào benchmark tổng hợp với shift của generating reference sắp tới, LR khởi đầu 0, không chiếu và M0 thực nghiệm. Audit snapshot và ba đầu mút MT19937 độc lập giúp đối chiếu cơ chế. Reproduction R và replay F2/F3 vẫn cần chứng cứ Colab. [Source audit](SOURCE_AUDIT.md) · [Fidelity](../results/convex-author-v1/data/fidelity.csv)

P đặt câu hỏi khác về ứng dụng online, khi learner chỉ quan sát lịch sử của mình. EMA là estimator nhân quả được gợi ý ở Remark 4.4, phụ thuộc động lực learner và có thể bị nhiễu hoặc rơi vào trạng thái 0. Phép thay riêng cùng native init minh họa phụ thuộc giữa design choices. Cần kết quả A1–A4 mới có thể định lượng những khác biệt giữa author-aligned control và nhóm. [Thiết kế đối chứng](PROTOCOL.md)

M2 đo chênh lệch đến population minimizer gần đúng trên cùng phân phối. Nó giúp tách thay đổi do comparator khỏi thay đổi learner. Khi nhãn sigmoid có Gaussian noise, conditional mean bị làm trơn và optimum radial co lại. M0 phù hợp kiểm notebook. M1 phù hợp tách hiệu chỉnh comparator. M2 phù hợp đối chiếu expected-risk trên từng phân phối đã nêu. [Metric validation](METRIC_VALIDATION.md)

Ở pilot d512, constant có M2 thấp hơn Convex dù Convex đã được cấp forward reference shift. Kết quả này cho thấy hiện tượng d512 có thể xuất hiện trong cả P và một thiết kế dùng shift đặc quyền. Một seed, hai protocol khác phân phối và chưa có ablation chính thức giới hạn mức khái quát hóa. [Kết quả](../results/convex-author-v1/RESULTS.md)

## Ánh xạ cho Sheet 4 và các issue

| Issue / ô điền | Nguồn có thể trích | Điều kiện tích hợp |
|---|---|---|
| T9 và T11, phương trình/thiết lập | [Bảng paper/author/I/P](SOURCE_AUDIT.md) | Ghi rõ unknown của runtime, T18 yêu cầu chạy Colab riêng |
| T15, cột W&B và thống kê P | [Provenance 60 run](../results/cp04-cp05/data/run_provenance.csv) · [45 ca mean/SD](../results/cp04-cp05/data/cp04_method_summary.csv) | Giữ số đã đóng băng |
| T15, giá trị paper và relative deviation P | N/A | P báo M2/B và khác phân phối so với R/M0 |
| T15, phụ lục bổ sung | [7 pilot, 3 diagnostic](../results/convex-author-v1/data/per_run.csv) · [Manifest](../results/convex-author-v1/manifest.json) | Ghi pilot, seed 0, ngày sau freeze, cần lead duyệt |
| T18, đề xuất cho `docs/08_original_code.md` | [Source audit](SOURCE_AUDIT.md) · [Colab contract](AUTHOR_NATIVE_EXECUTION.md) | Hòa duyệt/đối chiếu trước khi tích hợp |
| T19 | [Clean clone và đối chiếu](REPRODUCE.md) | Teammate chạy và tự ký bằng chứng |
| T22 | [Slide handoff](SLIDE_HANDOFF.md) | Khôi/Hòa và lead duyệt vị trí |

Workbook và deck không có trong workspace đã kiểm tra, nên chưa xác minh allocation Sheet 3 của slide 8–9 và chưa tạo deck dựa trên template. Slide handoff dùng các vị trí do nhiệm vụ cung cấp.
