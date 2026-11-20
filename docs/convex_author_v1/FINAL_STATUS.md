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
