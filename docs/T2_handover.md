# T2 — Bàn giao source, kiểm thử và phát hành

Cập nhật 13/11/2026, Asia/Saigon. Giữ SDK **wandb0.31.0** sau hotfix key Forge.
Source T2 đã kiểm thử local; **chưa đủ DoD Issue #6** vì Colab chưa end-to-end
và chưa có đủ ba account được xác minh.

## 1. Phạm vi và baseline

MAIN: D:/HOCBAI/HK7/ML/Project/github-pj, branch main.
DEVELOPMENT: D:/HOCBAI/HK7/ML/Project/github-pj-t2, branch T2-environment-wandb.
HEAD trước publish ở cả hai và origin/main:
`c978824e34ff29b83b0a00933cdfad1c897d5558`.
Fetch đầu phiên không có commit mới từ teammate.

Chỉ 11 file T2 được phép commit: requirements.txt, .gitignore, core/tracking.py,
scripts/__init__.py, scripts/check_environment.py, scripts/smoke_tracking.py,
tests/test_tracking.py, tests/test_tracking_offline.py,
notebooks/T2_environment_wandb.ipynb, README.md, docs/T2_handover.md.
R3 không thuộc T2, không stage. Tài liệu audit bên ngoài repo cũng không commit.

R3 tại DEVELOPMENT/docs/04_convex.md giữ nguyên SHA-256:
`40e8b1b2efc950879698f6dee08176a2c4024306d06693e9f9ef978b2babdbb5`.
Không sửa data/drift/schedules/trainer/utils hoặc experiments.
Giữ sparse patterns `/*`, `!/configs/`; literal configs/*.yaml vẫn nguyên
trong Git, không sửa path của lead. MAIN vẫn là common repository của DEVELOPMENT.

## 2. Polish và thay đổi logic

Comment/docstring Python trong tất cả file T2 và code notebook dùng tiếng Việt
không dấu; phần hướng dẫn Markdown dùng tiếng Việt có dấu. Nội dung ngắn, tập
trung vào lifecycle, metadata, environment isolation, commit metric và cleanup.
Tên API/biến/functions không đổi. Bảng mapping README gốc của lead giữ nguyên.

Logic tracking, secret validation, naming/grouping, Lab1/Lab2, config immutability
và seed policy giữ nguyên sau polish. Không thêm scientific computation.

Thay đổi logic duy nhất của source chính là **ZIP upload trong notebook**:
- Chuẩn hóa backslash thành slash trước khi xác định destination.
- Kiểm toàn archive trước khi ghi; chặn absolute/drive/UNC paths, '..', symlinks,
  normalized duplicates và file-directory collisions.
- Giữ destination trong source root, mở file bằng 'xb' để không ghi đè.
- Nhận cả directory entries Windows lẫn POSIX.
- Năm tests mới chạy chính hàm lấy từ AST notebook, không gọi Colab/login.
  Fixture giữ raw backslash để kiểm được hành vi Linux ngay cả trên Windows.

.gitignore thêm *.zip cho source archive sinh ra; giữ ignore venv/cache/wandb/
credentials. Không ignore source hoặc notebook.

## 3. Tracking contract

`init_run(config)` nhận flat Mapping, trả W&B Run trực tiếp.
Config > environment > defaults cho project/entity/name/mode/group/job_type.
Defaults: entity gr_ml_29_37_37, project ml-distribution-shift, online, lab1.
Scientific settings có thể nested nếu JSON-safe; không ép nested config["wandb"].

Một global run active mỗi process: wandb.run is run; wandb.log và run.log cùng đích.
Helper raise khi init lần hai trước finish, không tự finish/reuse/retry.
Sau finish có thể init lại. Dùng public reinit="return_previous" sau guard;
không private hacks, online resume="never", offline không resume.
Các thí nghiệm song song cần process riêng.

Tên tự sinh: [experiment-]method-seedN; explicit name/WANDB_NAME ưu tiên.
group/job_type optional. Lab1 mặc định; lab2 phải explicit, không tự thêm lab1;
conflicting lab tags bị chặn, tag được deduplicate và caller config không bị sửa.

Metadata tối đa depth10, 10,000 nodes và 64KiB, string keys, finite JSON-safe
values; tuple copy thành list. Reject cycles, models/tensors, unsupported SDK
controls và common secret aliases. Blacklist không bảo đảm nhận diện mọi secret.
API key/login do SDK và Secrets quản lý ngoài helper. Seed chỉ ghi nhận, không
tự seed RNG; loss/regret/schedule do experiment/trainer tính.

## 4. Environment và pins đã kiểm thử

Interpreter: D:/HOCBAI/HK7/ML/Project/.venv-t2-clean/Scripts/python.exe.
CPython3.12.2, Windows11 x64; include-system-site-packages=false.
Toàn bộ 10 distribution nằm trong venv riêng; không sửa Python hệ thống hoặc env cũ.

Pins trong requirements: numpy1.26.4, scipy1.15.3, scikit-learn1.6.1,
torch2.11.0, matplotlib3.10.3, pandas2.2.3, pyyaml6.0.3,
wandb0.31.0, datasets3.6.0, pyarrow19.0.1.
Torch thực tế là 2.11.0+cpu, cùng release với pin.
Python3.12 là target; không claim Python3.9 hoặc Colab3.13 tương thích.

Máy có i7-12700H, RTX3060 Laptop6GiB, driver546.30. Torch CPU không chứa CUDA;
GPU execution SKIPPED. Không nâng driver/CUDA hoặc tải Torch lại ở phiên polish.

## 5. Tests chạy lại sau polish

Mọi lệnh chạy từ DEVELOPMENT bằng interpreter trên, exit0:

| Check | Result | Bằng chứng |
|---|---|---|
| Python source AST | PASS | Tất cả file Python T2 hợp lệ |
| scripts.check_environment --require-isolated --gpu | PASS CPU; SKIPPED GPU | 10 imports/pins, CPU interoperability, Agg/YAML |
| pip check | PASS | No broken requirements found |
| unittest discover -s tests -v | PASS | 32 tests: 26 unit cũ + 5 ZIP + 1 real SDK offline |
| Global/object logging và lifecycle | PASS | SDK0.31.0, hai ID a5jqy4sg / 6jet0sns; Lab2/disabled checks giữ nguyên |
| Synthetic86 API-key parser | PASS local | Không login/HTTP verify và không đọc key thật |
| scripts.smoke_tracking --mode offline --member Khoi | PASS | 9kcfssxc, đúng team/project, 3 steps, completed=True |
| Notebook JSON + code AST | PASS static | Không outputs/execution count, guard Python3.12 giữ nguyên |
| ZIP Windows/POSIX và unsafe-path cases | PASS | Traversal/drive/UNC/symlink/duplicate/collision bị chặn |
| Secret-pattern scan và git diff --check | PASS heuristic/static | Không credential thật hoặc output artifacts trong whitelist |

Integration subprocess chỉ mode offline, artifacts tạm; public SDK teardown và
cwd ổn định tránh Windows lock. Test không dùng private history parser; metric
được commit rõ bằng commit=True trước khi đọc public summary.

## 6. W&B online và Colab

API mặc định SDK0.31.0: **https://api.forge.coreweave.com**.
Dashboard: https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift.
UI URL khác API host; WANDB_BASE_URL có thể cấu hình bằng endpoint chính thức.
SDK mới nhận key hợp lệ từ40 ký tự trở lên; parser đã test fake86 sau lỗi exact40
của SDK0.19.11. Không dùng key thật trong tests hoặc config.

**VS Code online smoke: PASS theo xác nhận của người dùng**:
[run kc5iilt2](https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift/runs/kc5iilt2).
PUBLIC cũng do người dùng xác nhận trên UI. Codex không chạy thêm online run,
không đọc/xin key thật và chưa xác minh đủ ba uploader riêng.

Colab vẫn đang kiểm thử. Người dùng đã tới Python guard và thấy runtime
**Python3.13.18**, không khớp target3.12. Guard không bị bỏ; cần chọn runtime3.12
tương thích trước khi đi tiếp. **COLAB END-TO-END CHƯA PASS**.
ZIP normalization được test local, không biến thành bằng chứng Colab runtime.

Notebook có source ZIP hoặc clone/checkout published SHA chứa T2, SOURCE=existing
sau restart, requirements/Torch guard, offline tests trước login và Colab Secrets.
Key lấy qua google.colab.userdata.get("WANDB_API_KEY"), không print key.
Torch mismatch dừng mặc định; CPU install cần opt-in, không tự tải CUDA.

Muốn tự kiểm tra online, chạy trong terminal tương tác sau khi có membership/write:

```powershell
$env:WANDB_BASE_URL = 'https://api.forge.coreweave.com'
& '../.venv-t2-clean/Scripts/python.exe' -m wandb login --host https://api.forge.coreweave.com --relogin --verify
& '../.venv-t2-clean/Scripts/python.exe' -m scripts.smoke_tracking --mode online --member Khoi
```

Nhập key tại prompt SDK, không truyền vào command. Mỗi người dùng key riêng;
member label không thay cho authenticated uploader. Khoa/Hòa cần xác nhận
membership/quyền ghi và run hợp lệ trước khi đủ acceptance.

## 7. Commit và push policy được người dùng cho phép

Commit trên DEVELOPMENT bằng whitelist 11 file, không git add . hoặc -A.
Nhóm commit:
1. chore: configure reproducible Python environment — requirements.txt, .gitignore.
2. feat: implement shared W&B tracking — core/tracking.py.
3. test: add tracking validation and smoke tests — scripts/, tests/ và notebook.
   Notebook đi cùng tests vì ZIP tests lấy hàm trực tiếp từ notebook.
4. docs: add Colab verification and T2 setup guide — README.md, docs/T2_handover.md.

Chỉ fast-forward MAIN nếu sạch và remote chưa diverge; push main non-force,
không rewrite teammate history. Nếu remote đổi hoặc branch protection từ chối thì
dừng push, không bypass. SHA thực tế tra bằng git log sau phát hành; không ghi
một SHA tự tham chiếu giả trong report được commit.

## 8. Những điều kiện còn thiếu

- Colab Python3.12, cài requirements và smoke online end-to-end có bằng chứng.
- Khôi/Khoa/Hòa mỗi account có run lab1 cùng project, URL và uploader xác minh.
- Membership/quyền ghi từng người; public view không đồng nghĩa write access.
- GPU/Linux runtime compatibility chưa có bằng chứng.
- Scientific experiments vẫn thuộc task khác, không được triển khai trong T2.

Chưa đánh dấu Issue #6 completed chỉ vì source được commit/push.
