# Tái lập từ clone sạch

Dùng CPython 3.12 và `requirements.txt` đã pin. Lần chạy đo trên Windows, Python 3.12.2, NumPy 1.26.4, SciPy 1.15.3, Matplotlib 3.10.3 và W&B 0.31.0. Xem package/version, thread env, hash mã nguồn, config và frozen files trong [provenance](../results/convex-author-v1/provenance.json).

```powershell
git clone --branch research/convex-author-v1 https://github.com/nvk1701/github-pj.git convex-author-review
Set-Location convex-author-review
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:OPENBLAS_NUM_THREADS = '1'
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe scripts/run_convex_author_full.py --plan
.\.venv\Scripts\python.exe scripts/run_convex_author_full.py --smoke --wandb-mode disabled
.\.venv\Scripts\python.exe scripts/run_convex_author_full.py --report-only
```

`--plan` ghi 81 ca khóa. `--smoke` chạy 7 pilot và 3 diagnostic full horizon trong namespace riêng, rồi xuất CSV/PNG. `--full` thêm test suite repo và kiểm F2/F3. Thiếu fixture trả mã 2 sau pilot và báo cáo. `--resume` tái dùng record hoàn thành khớp source/config/môi trường và SHA-256 NPZ. Giữ nguyên `--output-root` để resume. Các báo cáo đồng bộ vào `<output-root>/publication/`, với output mặc định nằm dưới `outputs/convex-author-v1`. CSV lịch sử toàn độ phân giải nằm trong NPZ của từng ca. Nếu worker bị kill cứng, kiểm PID trong `.writer` và chỉ xóa marker của ca đó sau khi đã xác nhận process chết. Với record đã seal bị sai checksum, cách an toàn là chạy vào output root mới, giữ nguyên chứng cứ cũ.

Dùng `--evidence-root` để import export Colab. [Contract T18](AUTHOR_NATIVE_EXECUTION.md)

`--wandb-mode offline` dùng `core/tracking.py` và ledger theo ca. `SDK_FINISHED` là SDK đóng run cục bộ. Online cần endpoint Forge được cấu hình bên ngoài `WANDB_BASE_URL=https://api.forge.coreweave.com` và thông tin xác thực đã provision qua SDK hay Secrets. Chạy lại cùng mode giữ ledger có sẵn. Nếu thấy `SDK_STARTED` sau gián đoạn thì đối soát một run trước mọi retry để tránh duplicate. Nếu đã sync một run offline, đối chiếu ID và checksum tại Forge trước khi tạo một run online riêng. Trạng thái `REMOTE_API_VERIFIED` cần backend state finished và checksum khớp.

## Đối chiếu chéo

T19 yêu cầu tự thực thi từ clone sạch với danh tính thành viên phụ trách, ghi SHA, môi trường, manifests và delta trong cùng distribution/stream. Repo ghi các kiểm tra tự động và quy trình của nhóm. Phiên này không có xác nhận mới từ teammate. Đề nghị dùng `per_run.csv` theo case key, M0 và M2 cùng SHA logic, tránh so exact bytes qua nền tảng khác. [Issue T19](https://github.com/nvk1701/github-pj/issues/23)

Đã xác minh tự động từ clone Git sạch của commit công bố `50b3bbf9066b6a4def736aa23715d01cb84ddccf` bằng cùng môi trường Python đã pin. `--smoke` dựng lại 10 ca, khớp chính xác cả metrics, hash X/y và 10 hash NPZ. Kiểm đủ 15 SHA-256 của manifest publication trong clone. Đây là self-check tự động tại cùng máy.

## Kiểm toán bản pre-push

Lần kiểm toán dùng `--output-root outputs/convex-author-release-audit` với `--wandb-mode disabled`. Giữ `outputs/convex-author-v1` làm nguồn chứng cứ của lần chạy trước. `--resume` kiểm metadata, source/config hash, checksum NPZ, shape, dtype, finiteness, nối trajectory và cả ba cumulative metric identities. Record đã seal có checksum lệch hoặc NPZ thiếu record đều dừng kiểm chứng và giữ payload. Chọn output root mới khi cần chạy lại ca đó. Báo cáo xác minh hash X/y trước khi gộp theo method hoặc tính chênh lệch ablation.

### Chạy lại bản đã khớp Colab

Giữ `outputs/convex-author-release-audit` làm archive preflight. Với mã đã ép kiểu theo ngưỡng Colab, dùng thư mục mới và import bộ export gốc kèm SHA-256:

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
python scripts/run_convex_author_full.py --resume --wandb-mode disabled --output-root outputs/convex-author-verified --evidence-root outputs/colab-author-external
python scripts/run_convex_author_full.py --report-only --output-root outputs/convex-author-verified --evidence-root outputs/colab-author-external
```

Môi trường Colab và SHA-256 của native artifact ở publication manifest. Pipeline chỉ chạy I và A sau khi validate thành công fixture F2/F3. Việc xuất 7 pilot và 3 chẩn đoán cho mã mới phục vụ validation, còn các bundle W&B pilot cũ tiếp tục dùng ID và provenance đã seal. Clone sạch của commit mã tính toán `94c5bd9` tái tạo cả 10 history validation trùng SHA-256 với run chính.

### W&B chỉ đồng bộ sau khi được ủy quyền

[Manifest đồng bộ](../results/convex-author-v1/data/wandb_offline_sync.csv) liệt kê đúng bảy SDK bundle ban đầu cùng run ID, SHA-256 file và digest history. Paths tương đối với worktree gốc. Clone chỉ có manifest và kết quả tổng hợp. Khi có ủy quyền rõ cho đích và nội dung, đối soát trước từng ID ở project Forge, kiểm checksum, rồi đồng bộ **thư mục offline của ID đó**. Ví dụ cho một run:

```powershell
$env:WANDB_BASE_URL = 'https://api.forge.coreweave.com'
$Row = Import-Csv docs/results/convex-author-v1/data/wandb_offline_sync.csv | Where-Object run_id -eq '4otmjkml'
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $Row.bundle).Hash.ToLowerInvariant() -ne $Row.bundle_sha256) { throw 'Bundle checksum mismatch' }
python -m wandb sync --yes (Split-Path -Parent $Row.bundle)
```

Đối chiếu lại ID, trạng thái `finished`, group và `validated_history_sha256` trên Forge. Nếu lần sync bị gián đoạn hoặc ID đã hiện diện, đối soát trước mọi lần thử tiếp theo. Bản V5 đã xác minh 88 run qua API. Để kiểm lại metric, đọc summary và history qua `Run.scan_history` cho `_step`, `M0`, `M1`, `M2`, `lr`. History có 100 dòng tại các bước chia hết cho 20. Đối chiếu với các giá trị tương ứng trong history cục bộ theo SHA-256 đã lưu. CSV lưu thêm SHA-256 của chuỗi JSON UTF-8 các dòng API theo thứ tự step, với key được sắp xếp và dấu phân cách JSON không thêm khoảng trắng. [Đối soát theo ID](../results/convex-author-v1/data/forge_release_verification.csv)

### Push khi đã có ủy quyền riêng

Chạy từ research worktree. Dừng nếu worktree chưa sạch, tên nhánh khác, origin khác, hoặc ref nghiên cứu đã tồn tại trên remote.

```powershell
if ((git branch --show-current) -ne 'research/convex-author-v1') { throw 'Wrong branch' }
if (git status --porcelain) { throw 'Working tree is dirty' }
if ((git remote get-url --push origin) -ne 'https://github.com/nvk1701/github-pj.git') { throw 'Wrong origin' }
git diff --check 8686740adca5f6879775e52a500b28cb203123d0 HEAD
if ($LASTEXITCODE -ne 0) { throw 'Diff check failed' }
$ResearchRef = git ls-remote --heads origin refs/heads/research/convex-author-v1
if ($LASTEXITCODE -ne 0 -or $ResearchRef) { throw 'Remote was not checked or branch already exists' }
git push -u origin refs/heads/research/convex-author-v1:refs/heads/research/convex-author-v1
```
