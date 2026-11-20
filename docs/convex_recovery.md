# Hướng dẫn recovery và kiểm tra chéo Convex

Giao thức khoa học frozen vẫn là `convex-cp03-v1`, source`7e0499c450a3e503ab09d4c6c3d272e6576b4432`. Verifier bảo trì không được dùng SHA của tool để gán lại nguồn tạo60 runs cũ. Nhánh bàn giao hiện hành là `feat/convex`; chỉ dùng SHA release sau khi đã được công bố.

## Checkout và môi trường

Nhánh hiện hành đã tích hợp rename từ main: `configs/*.yaml → configs/test.yaml`. Fresh Windows checkout bình thường được hỗ trợ; không cần global Git configuration. CPython3.12 là target đã kiểm tra. Cài requirements trong environment riêng, chạy từ repository root:

```powershell
python -m pip check
python -B -m scripts.check_environment --require-isolated
python -B -m scripts.smoke_tracking --mode offline --member Khoa
python -B experiments/convex_official.py --config configs/convex_cp03.yaml --dry-run
```

Offline smoke kiểm tra SDK local, không chứng minh quyền ghi hoặc independent reproduction. Thiết lập WANDB_DIR/DATA_DIR/CACHE_DIR/CONFIG_DIR ở nơi có quyền ghi; endpoint Forge là `https://api.forge.coreweave.com`. Login bằng tài khoản cá nhân qua SDK, không đưa key vào CLI/config/report. Public view không đồng nghĩa quyền ghi.

### Immutable frozen source trên Windows

Frozen7e và CP057131 vẫn chứa literalpath không hợp lệ; rename mới không thay nội dung của commit lịch sử. Chỉ áp dụng bootstrap sau trong clone mới, không dùng trên worktree đang có thay đổi:

```powershell
git clone --no-checkout https://github.com/nvk1701/github-pj.git convex-frozen
Set-Location convex-frozen
$sourceSha = '7e0499c450a3e503ab09d4c6c3d272e6576b4432'
git symbolic-ref HEAD refs/heads/reproduce-frozen
git update-ref refs/heads/reproduce-frozen $sourceSha
git -c core.protectNTFS=false read-tree $sourceSha
git -c core.protectNTFS=false update-index --skip-worktree -- 'configs/*.yaml'
git update-index --no-skip-worktree -- configs/convex.yaml configs/convex_cp03.yaml configs/convex_cp03_sensitivity.yaml
git checkout-index -a
git status --short
git rev-parse HEAD
```

Mỗi bước phải thành công; dừng nếu có lỗi hoặc HEAD/status không đúng. Exception NTFS chỉ theo lệnh, file literal không được materialize; không thay global safety. Nếu có dubious ownership, chỉ cấp safe.directory theo đúng clone được tin cậy.

## Attempts và trạng thái bền vững

Official matrix có45 logical cells:5 dimensions ×3methods ×3seeds. Root output phải mới. Mỗi lần chạy có UUID attempt riêng; không resume, silent skip, tự retrain hoặc upload retry. Config/source/science hashes phải nhất quán theo protocol; sensitivity khác điều kiện có science hash riêng, không gộp vào45-cell standard matrix.

Status/config JSON được fsync và atomic replace. attempts.jsonl append-only; failure records không bị xóa. checksums.json đóng dấu config/CSV/NPZ/PNG trước upload. Local computation, SDK finish, server finish, artifact verification và public access là trạng thái riêng. Directory tồn tại không chứng minh success.

## Failure và reconcile

1. Giữ nguyên batch root, payload, ledger và failure evidence. Không chạy lại vào root cũ.
2. Init/finish/upload chưa rõ phải reconcile trước retry. Run discovery dùng attempt identity; zero/multiple matches là ambiguity, không tự tạo run thay thế.
3. Server failed/running không phải verified success. Chỉ retry failed/unstarted cells sau khi kiểm tra, vào root mới với source/protocol được duyệt.
4. Remote success không thay thế computation local còn thiếu. SDK-finished=false vẫn giữ nếu lời gọi không trả về; reconcile chỉ bổ sung trạng thái quan sát thật.
5. Aggregate yêu cầu đúng coverage và một successful attempt cho mỗi key. Duplicate successes fail mặc định; explicit selection cần review. Corruption fail closed, failure không bị giấu.

Read-only là lựa chọn phù hợp để kiểm tra batch frozen bằng tool mới:

```powershell
python -B experiments/convex_reconcile.py verify PATH_TO_CELL --read-only --download-root NEW_EXTERNAL_DOWNLOAD_PATH
```

PATH_TO_CELL nhận relative hoặc absolute path; `seed0` từ thư mục cha đã có regression. Download root phải mới và nằm ngoài toàn input batch. Chỉ root này nhận downloads; proof/status/ledger đầu vào không bị ghi dù success hay failure. Không có run/artifact W&B mới. Nếu bỏ `--read-only`, verifier ghi observation proofs/status/ledger theo recovery contract.

Verifier yêu cầu đúng một COMMITTED `convex-evidence` với tên KEY-ATTEMPT:v0 và metadata attempt/config/science đúng. Chỉ backend `wandb-history` của cùng run được nhận diện riêng; duplicate science, type lạ, wrong attempt/version vẫn fail. Source/config/protocol/logical identity, completed server state, exact file manifest, fresh download, SDK verify và SHA256 đều được kiểm tra.

`head`/`training_source_sha` mô tả nguồn huấn luyện; `verification_tool_sha`/`verification_tool_file_sha256` mô tả tool thực thi. Raw working CRLF hash có thể khác Git LF blob hash; phải nêu convention trước so sánh.

## T19 và giới hạn

Khoa tự thực hiện cross-check, ghi account/machine/date, immutable source/tool SHA, command, env/BLAS, coverage, tolerances, output và actual/reference differences. Codex local suite hoặc read-only download không thay independent human reproduction. CP04/CP05 CSV có60 real URLs và hashes trong [provenance](results/cp04-cp05/data/run_provenance.csv). Directly comparable paper scalar vẫn N/A; dữ liệu dự án không phải exact author reproduction.

T19 report trên main là human declaration cho phạm vi đã ghi; full60 và các hướng Linear/NN cần evidence tương ứng. Không tự dùng credentials khác, thu hẹp scope hoặc nới tolerance để báo PASS. Shared root chỉ có một writer; hard kill có thể để lại pending status, cần điều tra/reconcile. Hướng dẫn này không tự cho phép experiment, merge hoặc nghiệm thu issue.
