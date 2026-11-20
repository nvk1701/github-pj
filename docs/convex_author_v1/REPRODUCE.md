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
