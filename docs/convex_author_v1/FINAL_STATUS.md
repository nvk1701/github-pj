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

## Preflight kết nối Colab MCP và Forge

| Hạng mục | Trạng thái | Bằng chứng |
|---|---|---|
| Môi trường Codex | VS Code extension cục bộ trên Windows | Originator `codex_vscode`, executable thuộc extension `openai.chatgpt`, CLI đi kèm `0.169.0-alpha.11.1` |
| Cấu hình MCP | Đã thêm `google-colab` | Entry trong cấu hình cá nhân của Codex trỏ tới `https://colaboratory.googleapis.com/mcp` qua Streamable HTTP. Không cài thêm phần mềm |
| Khám phá tools | 10 tool từ endpoint thật | `tools/list` trả HTTP 200 gồm `list_runtimes`, `list_runtime_specs`, `create_runtime`, `get_runtime`, `create_session`, `delete_session`, `list_sessions`, `execute_code`, `delete_runtime`, `get_subscription` |
| OAuth | Chờ client đăng ký trước | `list_runtimes` trả HTTP 401. `codex mcp add` đã lưu entry nhưng login tự động dừng vì Google từ chối Dynamic Client Registration. Metadata public chỉ định issuer `https://accounts.google.com/` và scope `https://www.googleapis.com/auth/colaboratory` |
| Colab Python và truyền tệp | `BLOCKED_EXTERNAL_EXECUTION` | Chưa có OAuth session xác thực. Chưa gọi tạo runtime, chưa chạy Python từ xa, chưa có checksum truyền hai chiều. Cổng Colab vẫn đóng |
| Forge W&B | `WANDB_WRITE_READY` ở mức quyền API và chuẩn bị sync | Viewer xác thực đọc được run frozen `38easaio` ở trạng thái `finished`. Team và project đều trả `readOnly=false`; trường project được SDK định nghĩa theo caller. Kiểm SHA-256 đủ 7 bundle và SDK `sync --dry-run` chọn đúng 7. Lần kiểm tra này không gửi bundle. 0/7 ID pilot được xác minh từ xa |

Người dùng cần chọn Google Cloud project được phép sử dụng, bật Colab API (`colaboratory.googleapis.com`) và tạo OAuth client loại **Desktop app**. Với ứng dụng ở chế độ Testing, thêm tài khoản đăng nhập vào Test users. Trang tạo client của Google Cloud Console đã được yêu cầu mở trong trình duyệt. Tải JSON vào thư mục riêng ngoài repo, chỉ thông báo đường dẫn cục bộ để tiếp tục cấu hình và mở OAuth consent. Thông tin xác thực và token runtime giữ ngoài Git và chat. [Google Cloud OAuth setup](https://docs.cloud.google.com/mcp/set-up-authentication-mcp-servers) · [Google Colab MCP reference](https://developers.google.com/colab/api/reference/mcp) · [Codex MCP](https://learn.chatgpt.com/docs/extend/mcp?surface=ide)

Sau OAuth, kiểm tra `list_runtimes` có xác thực. Dùng phiên Jupyter riêng trên runtime CPU phù hợp, hoặc tạo một runtime đúng quota nếu chưa có. Smoke test sẽ chạy Python trả một nonce mới, rồi dùng `get_runtime` và HTTP proxy của Google để đưa một tệp thử từ worktree lên runtime, biến đổi tệp bằng Python và tải kết quả về thư mục output cục bộ. Kiểm SHA-256 hai chiều và giữ token proxy trong bộ nhớ của client truyền file. Chỉ mở cổng `COLAB_EXECUTION_READY` sau khi đủ bằng chứng. R → parity → I → A tiếp tục theo protocol hiện hành.

Quyền ghi Forge đã được server báo qua metadata; upload thực tế vẫn cần phê duyệt riêng. [Manifest của 7 bundle](../results/convex-author-v1/data/wandb_offline_sync.csv)

## Khôi phục OAuth · V5

Codex CLI `0.169.0-alpha.11.1` trong VS Code hỗ trợ dùng Desktop client đăng ký trước qua override per-process. Entry `google-colab` toàn cục chỉ cấu hình endpoint. Kiểm tra URL do cùng lệnh login tạo ở chế độ không mở trình duyệt xác nhận client ID khớp file Desktop hiện có, scope đúng Colab, callback loopback HTTP `127.0.0.1` có cổng và PKCE S256. Không lưu giá trị OAuth trong repo. Tên `RemoteControlApp` trên màn hình Google có thể là tên branding trong project. Không có bằng chứng cần đổi client hoặc dùng lại dynamic registration.

Browser lần trước báo `403 access_denied` và nêu tài khoản không thuộc Test users của ứng dụng ở chế độ Testing. Tiến trình Codex kết thúc vì hết thời gian chờ callback. Chưa có phiên OAuth xác thực. Không có Google Cloud admin connector hoặc gcloud/ADC cục bộ phù hợp để đọc hay sửa Audience. Đã mở trang Audience của đúng project chứa Desktop client. Người có quyền cần thêm đúng tài khoản đã chọn ở màn hình 403 vào **Test users**, bấm **Save**, rồi xác nhận để mở lại consent. API enablement và trạng thái phiên Colab sẽ được kiểm tiếp sau khi nhận callback. [Audience](https://support.google.com/cloud/answer/15549945?hl=en) · [Branding](https://support.google.com/cloud/answer/15549049?hl=en)

R 0/5, F2/F3 UNVERIFIED, I 0/45, A 0/36 và Forge mới 0/88. Đã kiểm lại đủ 19 checksum frozen, 20 checksum publication và 7 checksum pilot SDK. Giữ kiểm thử 137/137 đã ghi ở bản pre-push làm baseline. Chạy lại suite sau khi có thay đổi mã/thực thi các cổng phụ thuộc. Không có upload, push hay merge trong vòng xác minh này.

### Cập nhật sau khi thêm Test user

**OAuth đã xác thực.** Bản CLI hiện tại thiếu khả năng đọc OAuth client secret qua tùy chọn tham chiếu biến môi trường đã thử. Codex hoàn tất callback với trường `oauth.client_secret` trong cấu hình tạm bên ngoài repo, chỉ cho tài khoản Windows hiện tại. Cấu hình tạm đã được xóa. Phiên được đọc lại từ Windows keyring bằng cùng Codex home riêng, và MCP báo `oAuth`.

**Cổng Colab vẫn đóng.** `list_runtimes`, `list_runtime_specs` và `get_subscription` đều trả 403 kèm lý do “API has not been used”. Google Cloud yêu cầu bật API sản phẩm cho project dùng MCP. Đã mở Colab API Library cho đúng project OAuth để chủ dự án bấm **Enable**. Khi API được bật, tái sử dụng phiên đã lưu và kiểm tra Python cùng checksum qua runtime proxy, sau đó mới mở R → F2/F3 → I → A. [Google Cloud Service Usage](https://docs.cloud.google.com/mcp/enable-disable-mcp-servers) · [Colab MCP](https://developers.google.com/colab/api/reference/mcp)

### API retry và Colab smoke test

Sau khi chủ dự án xác nhận Colab API ở trạng thái Enabled, chính phiên OAuth đã lưu trả thành công `list_runtime_specs` và `list_runtimes`. Không còn lỗi để trích consumer, reason hoặc link kích hoạt từ lần gọi hiện tại. Không thay đổi quota header, IAM, trạng thái API hay OAuth client. Tạo một runtime đủ điều kiện `VARIANT_CPU / SHAPE_STANDARD / NONE` khi danh sách runtime rỗng.

`COLAB_EXECUTION_READY`: Python chạy qua MCP trên Linux, Python 3.13.18, NumPy 1.26.4, TensorFlow 2.21.0, google-colab 1.0.0. Runtime báo SHA-256 đầu vào `08b82e94629cb7e9cd896aea2dacc819fc8a2372da992e3e12d13cf8f26163a4` và trả phép tính 285. Tệp do Python ghi được tải về qua HTTPS runtime proxy và khớp digest dự kiến `c2cabedef7784f636f1e042e9932fceee8514fce98100e99cc9f77aa8a32e5e7`. Bằng chứng nằm trong `outputs/colab-mcp-preflight/`. Kết quả smoke mở cổng chạy notebook gốc, còn R và F2/F3 cần artifact riêng.

### Kiểm toán kết quả V5

R 5/5, F2/F3 VERIFIED trên d2/8/32, I 45/45 và A 36/36. Đã kiểm cả 91 history cục bộ, g?m 10 ca validation, và đối chiếu độc lập 108 paired differences cùng 66 dòng thống kê. M2−M1 khớp hiệu chỉnh comparator với residual lớn nhất 4.44e−16 mỗi batch và 4.36e−12 trên các prefix tích lũy. Full suite đạt 138/138, clone sạch tái tạo 10 history validation với SHA-256 trùng khớp, và 19 tệp CP04/CP05 giữ nguyên byte. [Kết quả](../results/convex-author-v1/RESULTS.md) · [Native](AUTHOR_NATIVE_EXECUTION.md) · [Metric audit](../results/convex-author-v1/data/independent_validation.json)

`READY_TO_PUSH_WITH_DISCLOSED_LIMITATIONS`. Đủ 88 SDK bundle offline, 0/88 Forge remote verified. Bộ duyệt tự động chặn upload vì chưa có xác nhận trực tiếp cho payload 7 pilot, 45 I và 36 A cùng đích `gr_ml_29_37_37/ml-distribution-shift` qua `api.forge.coreweave.com`. Không push, merge hay upload trong lượt kiểm toán này. T19 và duyệt slide vẫn là các bước của nhóm. [Trạng thái theo ID](../results/convex-author-v1/data/forge_release_verification.csv)
