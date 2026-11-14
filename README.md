## Bảng Ánh Xạ Thí Nghiệm (Experiments Mapping)

Dưới đây là bảng tổng hợp các thí nghiệm tái hiện lại bài báo "Learning Rate Schedules in the Presence of Distribution Shift", mã nguồn tương ứng, liên kết theo dõi trên Weights & Biases (W&B) và kết luận chính.

| Thí nghiệm | Script/Notebook | W&B run | Kết luận 1 dòng |
| :--- | :--- | :--- | :--- |
| **1. Hồi quy tuyến tính** (Linear Regression) | `experiments/linear_regression.py` | [Tên_Run_Linear](Link_W&B_của_bạn_vào_đây) | *(Điền sau: ví dụ: Tốc độ học tăng vọt khi phân phối dịch chuyển, bám sát SDE)* |
| **2. Hàm lồi tổng quát** (Convex Loss) | `experiments/convex.py` | [Tên_Run_Convex](Link_W&B_của_bạn_vào_đây) | *(Điền sau: ví dụ: Tốc độ học dạng ngưỡng hoạt động hiệu quả theo chặn trên của Regret)* |
| **3. Hàm không lồi** (Non-convex / Neural Net) | `experiments/neural_net.py` | [Tên_Run_NN](Link_W&B_của_bạn_vào_đây) | *(Điền sau: ví dụ: Restart Adam giúp mạng nơ-ron thoát khỏi cực tiểu cục bộ cũ)* |

## Cấu Trúc Thư Mục & Giải Thích Chức Năng Từng File (Dành Cho Cả Nhóm)

Dưới đây là sơ đồ tổ chức repository và hướng dẫn nhanh để các thành viên (**TV1: Khoa**, **TV2: Khôi**, **TV3: Hòa**) tra cứu chức năng của từng thư mục/file:

```text
PROJECT/
├── README.md                          # Tổng quan dự án, Bảng ánh xạ thí nghiệm, Hướng dẫn T2 & Cấu trúc repo
├── requirements.txt                   # Danh sách thư viện ghim phiên bản (CPython 3.12, PyTorch 2.11, W&B 0.31)
├── core/                              # [KHUNG CODE DÙNG CHUNG CHO CẢ 3 BÀI TOÁN]
│   ├── tracking.py                    # (T2 - Khôi) Khởi tạo & quản lý W&B run dùng chung (`init_run`)
│   ├── utils.py                       # (T6 - Khoa) Cố định seed (`set_seed`), đọc/ghi YAML, phép chiếu Euclid `project_to_ball`
│   ├── drift.py                       # (T6 - Khoa) Sinh quỹ đạo xoắn ốc Eq. 26 (`generate_spiral_trajectory`), tính shift `gamma_t` & `EMADriftEstimator`
│   ├── schedules.py                   # (T6 - Khoa) Giao diện `BaseSchedule` & các lịch trình cơ sở (`Constant`, `1/t`, `1/sqrt(t)`)
│   ├── trainer.py                     # (T6 - Khoa) Vòng lặp Never-Ending Online SGD (`OnlineSGDTrainer`) cho Linear & Logistic + log W&B
│   └── data.py                        # (T11 - Hòa) Tải dữ liệu tế bào tụy `scvelo.datasets.pancreas()` & sinh luồng 4 pha P1->P4
├── docs/                              # [TÀI LIỆU ĐỌC HIỂU BÀI BÁO (MILESTONE 2) & BÀN GIAO]
│   ├── 00_fastpass.md                 # (T0) Ghi chú đọc lướt (Fast Pass) nắm ý tưởng cốt lõi bài báo
│   ├── motivation_example.md          # (T3) Thống nhất phạm vi Full Scope, phân công & Ví dụ thực tế (CTR thương mại điện tử)
│   ├── 01_context.md                  # (T4 - Khoa) Bối cảnh, động lực thực tế & phân tích 3 phương pháp tiền nhiệm
│   ├── notation.md                    # (T5 - Khoa) Bảng ký hiệu thống nhất, Bài toán hình thức & Giải thích 2 Pipeline (Inference + Training)
│   ├── T2_handover.md                 # (T2 - Khôi) Tài liệu bàn giao môi trường, kiểm thử W&B và quy ước commit
│   └── references.md                  # (T10 - Khôi) Danh mục tài liệu tham khảo chuẩn IEEE
├── experiments/                       # [CÁC SCRIPT CHẠY THỰC NGHIỆM CHÍNH (LAB 1)]
│   ├── smoke_test.py                  # (T5, T6 - Khoa) Kiểm thử nhanh (<1 phút) khung `core/`, tích hợp `init_run` & vẽ `figures/pipeline.png`
│   ├── linear_regression.py           # (T14 - Khoa) Thực nghiệm 1: Tái hiện Figure 4 và Figure 5 (top) cho Linear Regression
│   ├── convex.py                      # (T15 - Khôi) Thực nghiệm 2: Tái hiện Figure 5 (bottom) cho Logistic Regression
│   ├── sensitivity.py                 # (T17 - Khôi) Phân tích mở rộng (Bonus): Quét độ nhạy theo tốc độ trôi `gamma` và nhiễu `sigma`
│   └── neural_net.py                  # (T16 - Hòa) Thực nghiệm 3: Tái hiện Figure 6 (Flow Cytometry MLP 5 tầng với Adam Restart)
├── figures/                           # [HÌNH VẼ & ĐỒ THỊ TỰ TẠO ĐỂ ĐƯA VÀO SLIDE DECK]
│   ├── pipeline.png                   # (T5 - Khoa) Sơ đồ Pipeline tổng thể tách rõ Stage A (Online Inference) & Stage B (Online Training)
│   └── smoke_test.png                 # (T6 - Khoa) Đồ thị kiểm thử quỹ đạo xoắn ốc 2D, learning rate và Cumulative Regret
├── scripts/                           # [SCRIPT KIỂM TRA MÔI TRƯỜNG & TRACKING (T2)]
│   ├── check_environment.py           # (T2 - Khôi) Kiểm tra phiên bản CPython 3.12, venv cô lập và các thư viện đã ghim
│   └── smoke_tracking.py              # (T2 - Khôi) Kiểm tra kết nối W&B (`--mode offline` / `--mode online`) cho từng thành viên
├── notebooks/                         # [COLAB NOTEBOOKS]
│   └── T2_environment_wandb.ipynb     # (T2 - Khôi) Notebook kiểm tra môi trường và W&B trên Google Colab
└── tests/                             # [UNIT TESTS & INTEGRATION TESTS]
    ├── test_tracking.py               # (T2 - Khôi) 31 unit tests kiểm tra logic `core/tracking.py`, `smoke_tracking.py` và giải nén ZIP
    └── test_tracking_offline.py       # (T2 - Khôi) Integration test với W&B SDK ở chế độ offline
```

### Hướng dẫn nhanh cách ghép thuật toán vào `core/` (Dành cho T7, T9, T12)
1. **Cách viết lịch trình tốc độ học mới**: Kế thừa lớp `BaseSchedule` trong `core/schedules.py`, cài đặt hàm `step(self, t, **kwargs) -> float` (nhận `gamma_prev`, `batch_size`, `d`, `noise_std`, `loss_val`) và hàm `reset(self)`.
2. **Cách chạy vòng lặp Online SGD và đẩy log lên W&B**:
   ```python
   from core.drift import generate_spiral_trajectory
   from core.schedules import ConstantSchedule
   from core.tracking import init_run
   from core.trainer import OnlineSGDTrainer
   from core.utils import set_seed

   set_seed(42)
   theta_star, gamma_seq = generate_spiral_trajectory(T=2000, d=128, ell=8)
   schedule = ConstantSchedule(lr=0.05)

   run = init_run({
       "experiment": "linear",
       "method": schedule.name,
       "seed": 42,
       "mode": "offline",  # Đổi thành "online" khi đẩy kết quả thật lên W&B
       "tags": ["linear"],
   })
   try:
       trainer = OnlineSGDTrainer(d=128, schedule=schedule, loss_type="linear", batch_size=256, use_wandb=True)
       history = trainer.run(theta_star_path=theta_star, gamma_seq=gamma_seq, seed=42)
   finally:
       run.finish()
   ```

---

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
