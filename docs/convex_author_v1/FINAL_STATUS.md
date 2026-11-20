# Trạng thái kiểm chứng ngày 20/11/2026

| Hạng mục | Trạng thái | Bằng chứng |
|---|---|---|
| P, 45 CP04 + 15 CP05 | VERIFIED | 19 file publication y hệt theo SHA-256 trước/sau. Artifact lịch sử tiếp tục dùng manifest cũ |
| R, 5 ca Colab | BLOCKED_EXTERNAL_EXECUTION | Không có runtime điều khiển được cấp. Ba số cuối đã lưu trong snapshot |
| F0 audit | VERIFIED | Đọc blob tác giả tại commit bất biến và đối chiếu source P |
| F1/F5 toán và regression | VERIFIED | 132 unit tests, gồm 124 test cũ và 8 test mới |
| F2/F3 | UNVERIFIED | Cần fixture Colab matched full horizon ở d2/8/32 |
| F4 pilot | VERIFIED | 7 pilot và 3 RNG diagnostic có histories 2.000 bước, finite và hash |
| I, 45 ca | BLOCKED_EXTERNAL_EXECUTION | Gate F2/F3 chưa mở |
| A, 36 ca | BLOCKED_EXTERNAL_EXECUTION | Cần I hợp lệ trước |
| W&B pilot | COMPLETED_WITH_LIMITATIONS | SDK finish 7/7 offline. Forge API xác thực quyền đọc; upload pilot chờ ủy quyền cụ thể |
| T19 cross-run teammate | NOT_STARTED | Chờ người được phân công thực thi độc lập |
| T22 lead review | NOT_STARTED | Handoff sẵn, deck và workbook giữ nguyên |

Đã thử `--full --wandb-mode offline`. Full chạy suite và pilot rồi trả mã 2 vì thiếu fixture. Các history raw khoảng 41.1 MiB, ở `outputs/convex-author-v1/runs`. Quan sát peak working set của process chính sau ba pilot d512 là 336896000 bytes (321.3 MiB). Đây là phép đọc trong lúc chạy, gồm imports trước bước tạo hình báo cáo, không phải đo peak của process W&B con.

Forge endpoint `https://api.forge.coreweave.com` được kiểm tra chỉ đọc bằng SDK, có viewer xác thực và đọc được frozen run `38easaio` ở trạng thái finished. Bộ duyệt tự động từ chối yêu cầu đồng bộ 7 SDK artifacts vì chưa có ủy quyền rõ cho payload và đích. Không thử đồng bộ qua đường khác. Không có URL remote mới được xác minh. [Run manifest](../results/convex-author-v1/manifest.json) · [Handoff tái lập](REPRODUCE.md)

Clean-clone smoke đã dựng lại 10/10 history NPZ khớp SHA-256 và kiểm 15/15 publication hashes. Import checksum/replay chạy thử bằng fixture tổng hợp nội bộ riêng, gồm trường hợp payload bị sửa, không dùng làm bằng chứng F2/F3. Script Colab helper được kiểm tra cú pháp; thực thi ở Colab vẫn chờ runtime.

## Preflight bổ sung ngày 20/11/2026

Worktree research sạch tại `23a608139eb94ccf6322e506a70411759f9351e6`. Đã kiểm trực tiếp 19 SHA-256 frozen và 15 SHA-256 publication. Có 7 pilot, 3 diagnostic MT19937, 7 bundle SDK offline. Hai NPZ tìm thấy ở thư mục self-check mang nhãn dữ liệu tổng hợp nội bộ. Runtime Colab và export thật chưa có. Trạng thái ban đầu của F2/F3 là UNVERIFIED. Không chạy I/A trước khi cổng này mở.

Máy kiểm tra có 20 CPU logic, RAM tổng 15,73 GiB, RAM khả dụng 1,80 GiB tại thời điểm preflight, ổ D còn 146,11 GiB. Dùng Windows 11 và môi trường Python 3.12.2 đã pin. Máy đủ cho tuần tự NumPy, còn thực thi TensorFlow notebook nguyên bản thuộc Colab riêng. W&B và Git chỉ kiểm tra đọc hoặc cục bộ theo yêu cầu pre-push hiện tại.

## Kết quả kiểm toán pre-push

**READY_TO_PUSH_WITH_DISCLOSED_LIMITATIONS.** `--full --wandb-mode disabled` trên output mới qua 137 test rồi dừng ở mã 2 do thiếu fixture F2/F3. Ma trận dự kiến có 81 case key duy nhất. R đã xác minh 0/5, I 0/45, A 0/36, tất cả các ca còn thiếu đang bị chặn. Hoàn tất 7/7 pilot và 3/3 MT19937 diagnostic, không lỗi. [Inventory](../results/convex-author-v1/data/inventory.csv)

Clone sạch của mã kiểm toán dựng lại 10/10 history NPZ trùng SHA-256 với cả đợt trước và đợt này, kể cả dữ liệu ghép cặp. Tám hình PNG tái tạo có SHA-256 giống nhau giữa hai checkout. Hai hình mới dùng history có sẵn cho comparator correction và d512. 21 dòng thống kê pilot được kiểm trực tiếp từ bảng per-run. [Báo cáo](../results/convex-author-v1/RESULTS.md)

19 file CP04/CP05 giữ nguyên byte trong worktree. So với manifest nguồn CP05.6, sáu PNG khớp trực tiếp và 12 text payload khớp sau chuẩn hóa CRLF→LF như Git checkout; bản thân manifest giữ SHA-256 từ preflight. Không đổi protocol hoặc kết luận lịch sử.

Forge API xác thực quyền đọc trong lần kiểm toán, xác nhận được run frozen tham chiếu đang finished. Bảy ID pilot offline chưa có trạng thái remote nào được xác minh. Bảy SDK bundle hiện có khớp checksum và liên kết đúng history SHA-256. Phiên pre-push giữ cục bộ mọi output; không thực hiện push, merge hay upload. [Danh sách bundle và đích](../results/convex-author-v1/data/wandb_offline_sync.csv)

Để hoàn tất về khoa học, người có Colab cần chạy script T18 trong runtime riêng và giao 5 cặp native, 3 cặp diagnostic cùng log thực thi. Sau khi parity đạt, tiếp tục I và A bằng resume. T19 yêu cầu cross-run độc lập của teammate, T22 chờ xác nhận và tích hợp của lead. [Colab](AUTHOR_NATIVE_EXECUTION.md) · [Lệnh tái lập và push có ủy quyền](REPRODUCE.md)

`--report-only` cho lại đủ 17 hash artifact sinh tự động. `--resume` chạy lại 137 test, đọc 10 record hiện có và giữ nguyên hash lẫn thời gian sửa từng record trước khi về mã 2. Gói công bố kèm docs và bảng W&B có 20 hash đã seal.
