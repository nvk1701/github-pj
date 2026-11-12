## Bảng Ánh Xạ Thí Nghiệm (Experiments Mapping)

Dưới đây là bảng tổng hợp các thí nghiệm tái hiện lại bài báo "Learning Rate Schedules in the Presence of Distribution Shift", mã nguồn tương ứng, liên kết theo dõi trên Weights & Biases (W&B) và kết luận chính.

| Thí nghiệm | Script/Notebook | W&B run | Kết luận 1 dòng |
| :--- | :--- | :--- | :--- |
| **1. Hồi quy tuyến tính** (Linear Regression) | `experiments/linear_regression.py` | [Tên_Run_Linear](Link_W&B_của_bạn_vào_đây) | *(Điền sau: ví dụ: Tốc độ học tăng vọt khi phân phối dịch chuyển, bám sát SDE)* |
| **2. Hàm lồi tổng quát** (Convex Loss) | `experiments/convex.py` | [Tên_Run_Convex](Link_W&B_của_bạn_vào_đây) | *(Điền sau: ví dụ: Tốc độ học dạng ngưỡng hoạt động hiệu quả theo chặn trên của Regret)* |
| **3. Hàm không lồi** (Non-convex / Neural Net) | `experiments/neural_net.py` | [Tên_Run_NN](Link_W&B_của_bạn_vào_đây) | *(Điền sau: ví dụ: Restart Adam giúp mạng nơ-ron thoát khỏi cực tiểu cục bộ cũ)* |

## T2 — Environment & W&B

Target chính thức: **CPython 3.12**. Chín package của Issue #6 và PyArrow được
ghim trong `requirements.txt`, nguồn phiên bản chung duy nhất. Không hỗ trợ
Python 3.9 với bộ pin này; không coi metadata compatibility là runtime test.

### Cài vào venv riêng

Chọn CPython thật (Windows PATH có thể ưu tiên MSYS), tạo venv **không**
`--system-site-packages`. Không sửa Python hệ thống hoặc env project khác.

```powershell
& 'C:/Users/LENOVO/AppData/Local/Programs/Python/Python312/python.exe' -m venv ../.venv-t2-clean
# Chon build CPU cung pin de tranh tai CUDA khi khong can.
& '../.venv-t2-clean/Scripts/python.exe' -m pip install --no-deps torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
& '../.venv-t2-clean/Scripts/python.exe' -m pip install -r requirements.txt
& '../.venv-t2-clean/Scripts/python.exe' -m scripts.check_environment --require-isolated --gpu
& '../.venv-t2-clean/Scripts/python.exe' -m pip check
& '../.venv-t2-clean/Scripts/python.exe' -m unittest discover -s tests -v
```

Thay đường dẫn interpreter bằng CPython 3.12 trên máy bạn. Dùng lại venv đã tạo
nếu đúng project, không tạo lại đè lên env đang dùng. Linux: tạo venv bằng
`python3.12 -m venv ../.venv-t2-clean`, executable là `../.venv-t2-clean/bin/python`.
Chọn wheel CPU trước rồi cài requirements cho profile CPU. Release
`torch==2.11.0` chấp nhận wheel `2.11.0+cpu` hoặc build CUDA cùng release;
build/device thực tế phải được ghi nhận. Không đổi pin theo máy cá nhân.

Windows hiện đã kiểm thử clean venv, Torch CPU; GPU tensor test **SKIPPED** vì
build CPU không chứa CUDA, dù máy có GPU. Không tự upgrade driver/CUDA.
Colab CPU/GPU: xem notebook; mismatch Torch phải dừng mặc định. Build GPU cần
runtime/driver tương thích và test CUDA riêng; chưa có bằng chứng Colab/GPU PASS.
PyArrow 19.0.1 giữ stack NumPy 1.26 sau lỗi import với PyArrow 26.0.1 đã quan sát.
Dependency bắc cầu chưa lock theo platform; không thêm TF/scVelo vào T2.

### Project và quyền truy cập

Entity **`gr_ml_29_37_37`**, project **`ml-distribution-shift`**:
[W&B dashboard](https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift).
PUBLIC do người dùng xác nhận trên UI; public view không cấp quyền ghi.
Khoa/Hòa chưa xác nhận membership; trạng thái lời mời/quyền ghi chưa xác minh.
Owner kiểm tra đúng account, gửi lời mời nếu chưa có; từng người chấp nhận.

Mỗi người dùng key riêng qua SDK login hoặc secret manager; không đưa key vào
source, config, CLI args, terminal history hoặc báo cáo. Local có thể dùng
`python -m wandb login` và nhập tại prompt của SDK. Không gửi key cho Codex.
Không dùng một account rồi đổi `--member` để giả ba uploader.

### Shared API và lifecycle

```python
import wandb
from core.tracking import init_run

run = init_run({
    "experiment": "linear", "method": "optimal", "seed": 0,
    "group": "linear-figure5", "tags": ["linear"],
})
try:
    assert wandb.run is run
    wandb.log({"loss": 0.5}, step=0, commit=True)
    run.log({"learning_rate": 0.01}, step=1, commit=True)
except BaseException:
    try:
        run.finish(exit_code=1)
    except Exception:
        pass  # Giu loi chinh neu cleanup cung loi.
    raise
else:
    run.finish()
```

Hai cách log ghi vào cùng active global run. Với explicit step, SDK mặc định
`commit=False`; dùng `commit=True` khi muốn đóng record ngay. Muốn gộp nhiều
metrics cùng step thì tự điều khiển commit. Helper không tính scientific metrics.

`init_run(config)` nhận flat Mapping; config > environment > defaults cho
project/entity/name/mode/group/job_type. Defaults là team/project trên, mode online,
Lab1. Các scientific settings khác được giữ trong config JSON-safe, có thể nested.
Không ép `config["wandb"]`. Optional `experiment`, `method`, `seed`, `group`,
`job_type`; explicit name (hoặc WANDB_NAME) ưu tiên hơn generated name.
Tên mặc định `{experiment}-{method}-seed{seed}`; thiếu experiment giữ fallback
`{method}-seed{seed}`, method mặc định experiment, seed thiếu hiển thị unset.
Seed chỉ được ghi nhận; caller tự seed Python/NumPy/Torch cho training.

Lab1 mặc định thêm `lab1`. `{"lab": "lab2"}` chỉ thêm `lab2`; tags trùng được
khử, tags caller không bị sửa. `tags=["lab2"]` mà thiếu lab2 explicit hoặc trộn
lab1/lab2 sẽ raise ValueError. Group optional, không ép mọi thí nghiệm cùng group.

Chỉ một run active trong process. Gọi helper lần hai khi run trước còn active
raise RuntimeError trước SDK, không finish/reuse run cũ. Sau finish có thể init lại.
SDK dùng public `reinit="return_previous"` cùng guard, tránh notebook tự finish
hoặc `create_new` bỏ global binding; guard mới là chính sách cấm reuse.
Không tự retry; online dùng resume="never". Parallel experiments dùng process riêng.

Config không bị mutate. Metadata phải JSON-safe: string keys, primitive finite
values, Mapping/list/tuple (tuple được copy thành list); tối đa depth 10, 10,000
nodes và 64 KiB payload cuối. Không truyền tensors/model/dataset object hoặc
credentials. Common secret aliases bị chặn recursively; đây **không phải** bộ
phát hiện mọi secrets. Chỉ gửi scientific metadata đã kiểm tra vào project PUBLIC.
Các SDK controls như id/resume/reinit/dir/settings/base_url/config/wandb/notes
bị từ chối trong config; endpoint/auth do SDK environment quản lý.
Identifier, mode, seed, tags hoặc lab sai sẽ fail trước init. SDK errors truyền
về caller, không tự fallback offline rồi báo upload thành công.

### Smoke và nghiệm thu online

```powershell
& '../.venv-t2-clean/Scripts/python.exe' -m scripts.smoke_tracking --mode offline --member Khoi
$env:WANDB_ENTITY = 'gr_ml_29_37_37'
$env:WANDB_PROJECT = 'ml-distribution-shift'
& '../.venv-t2-clean/Scripts/python.exe' -m scripts.smoke_tracking --mode online --member Khoi
```

CLI offline mặc định; hỗ trợ --entity/--project override. W&B/cache artifacts
được ignore. Unit tests mock SDK; integration test dùng SDK thật trong fresh
subprocess, tạo hai offline runs và kiểm global/object logging, Lab2, lifecycle
cùng disabled behavior. Test không sync, không login và không claim upload.

Online chỉ in URL sau finish thành công. Khôi/Khoa/Hòa mỗi người chạy bằng
account riêng, lưu run URL và kiểm authenticated uploader trong dashboard;
`member` chỉ là label. Kiểm tags lab1/smoke-test, ba metric steps và
`smoke/completed=true`. Người dùng đã xác nhận smoke online trên VS Code thành
công: [run kc5iilt2](https://forge.coreweave.com/wandb/gr_ml_29_37_37/ml-distribution-shift/runs/kc5iilt2).
Codex không chạy lại online và chưa xác minh đủ ba uploader riêng.

### API endpoint và troubleshooting

Hotfix 13/11/2026: pin **`wandb==0.31.0`**, hỗ trợ Python >=3.10, đã kiểm thử
Python3.12.2. Bản 0.19.11 thực tế từ chối key mới 86 ký tự vì bắt đúng 40 ký tự.
Source validator của 0.31.0 nhận key định dạng hợp lệ từ 40 ký tự trở lên;
regression test dùng chuỗi giả 86 ký tự, không gọi login/verify hay đọc key thật.
[Validator đúng phiên bản](https://github.com/wandb/wandb/blob/v0.31.0/wandb/sdk/lib/wbauth/validation.py).

SDK mặc định API **https://api.forge.coreweave.com**, hỗ trợ `WANDB_BASE_URL`;
UI **https://forge.coreweave.com/wandb/...** không phải API base. Helper không
hardcode endpoint. Để loại cấu hình host cũ và xác minh bằng tài khoản của bạn,
chạy thủ công trong terminal tương tác (không đưa key vào command):

```powershell
$env:WANDB_BASE_URL = 'https://api.forge.coreweave.com'
& '../.venv-t2-clean/Scripts/python.exe' -m wandb login --host https://api.forge.coreweave.com --relogin --verify
```

Chỉ nhập key tại prompt của SDK. `--verify` quan trọng vì CLI0.31.0 mặc định
có thể lưu credential mà không kiểm tra server. Codex không chạy lệnh login này.
Nếu env khác đang dùng SDK cũ, cài lại theo requirements bằng đúng interpreter;
không dùng `pip`/`wandb` từ PATH khác. Restart notebook/kernel sau khi nâng SDK.
Local parser/offline PASS không chứng minh key thật, quyền ghi hoặc upload online.

Auth error: kiểm own key/login của đúng host; permission error: kiểm membership
và quyền ghi trong đúng team/project. Network/API errors: kiểm endpoint và log
đã lọc credentials. Không tạo project khác hay fallback offline để che lỗi.

### Colab

Notebook: `notebooks/T2_environment_wandb.ipynb`.
**COLAB RUNTIME NOT TESTED**; JSON/AST PASS không phải runtime PASS.
Runtime Colab ban đầu người dùng quan sát là Python **3.13.18**, khác target
3.12. Notebook giữ guard; cần chọn runtime tương thích trước khi tiếp tục.
Trước publish: upload ZIP source DEVELOPMENT, loại .git/venv/cache/wandb/secrets.
Sau review và user tự push: clone rồi checkout SHA published chứa T2; không giả
định remote main đã có uncommitted code. SOURCE=existing cho restart.

Notebook đọc runtime/version, chặn mismatch Torch trước install; chỉ khi chủ động
bật ALLOW_CPU_TORCH_INSTALL mới cài wheel CPU đúng pin, không tự tải CUDA.
Runtime không khớp requirements không được coi là reproducible PASS.
Colab Secrets: lưu WANDB_API_KEY, cấp notebook access; code lấy qua
`google.colab.userdata.get("WANDB_API_KEY")`, không print key.
Offline tests chạy trước login; online cell dùng đúng team/project và hai API log.

Bằng chứng môi trường, gates và commit plan: [docs/T2_handover.md](docs/T2_handover.md).
