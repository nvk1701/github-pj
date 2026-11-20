# Convex: mã nguồn, kết quả và hướng dẫn hiện hành

Tài liệu cập nhật ngày 18/11/2026. Nhánh bàn giao là `feat/convex`; SHA chính xác nằm trong báo cáo công bố. Các ghi chú CP01–CP05.7 giữ nguyên phạm vi lịch sử.

## Bài toán và thuật toán

Mô hình logistic được cập nhật bằng mini-batch SGD trên luồng có response shift. Covariates có phân phối Gaussian N(0,I); nhãn mềm là sigmoid của tích vô hướng với generating reference cộng nhiễu Gaussian. Quỹ đạo Eq.26 dùng cubic radius và sample-and-hold. ConvexSchedule chọn hai ngưỡng theo Proposition4.3 và chiếu learning rate trước vào khoảng ngưỡng; shift thực tế của learner được ước lượng bằng EMA .9.

Giao thức frozen `convex-cp03-v1`: T=2000, B=256, E=512, seeds0/1/2, dimensions2/8/32/128/512; noise std=sqrt(.1), L=.25, variance profile=d/4, eta_init=1. Ball projection có radius=d/2 và diameter=d. Constant/inverse-sqrt dùng LR0=.1, không tuning. Shared legacy `inst_regret`/`cum_regret` có clipping; không dùng thay cho metric official.

Primary B là tổng population risk trừ risk tại numerical approximate oracle. Secondary A dùng generating reference; khi có nhiễu, reference không mặc định là population minimizer. Signed Monte Carlo excess được giữ nguyên. Sample SD qua ba seeds, conditional MC SE và sai số số học là ba đại lượng khác nhau. Quadrature/oracle diagnostics không phải chứng chỉ sai số nghiêm ngặt.

## Nguồn và kết quả đã xác minh

| Nội dung | Mốc bất biến |
|---|---|
| Huấn luyện60 runs CP04/CP05 | `7e0499c450a3e503ab09d4c6c3d272e6576b4432` |
| Publication19 files | `4d6bce54e68e559d77cb64bd5dab5eeb1cf7fac6` |
| CP05.7 verifier/docs | `37e2492` và `1316573` |
| Audit relative-path fix | `ccc269ce94d3bf2b5108a0452793472f694feea6` |

[Kết quả frozen](results/cp04-cp05/RESULTS.md), [60 URLs và hashes](results/cp04-cp05/data/run_provenance.csv), [publication manifest](results/cp04-cp05/publication_manifest.json), [đối chiếu ba hình CP05](convex_figure_provenance.md).

CP04 hoàn thành45 cells. Convex có mean B thấp nhất tại d2/8/32/128; constant thắng tại d512:504.8952 so với Convex1385.0089. Kết quả thua được giữ nguyên. CP05 thực hiện15 runs Convex-only d128, gồm năm điều kiện OFAT và ba seeds. [Bảng điều kiện](results/cp04-cp05/data/cp05_condition_summary.csv) mô tả đúng phạm vi; YAML sensitivity cũ là kế hoạch rộng hơn, không phải resolved configs của15 runs.

Ell thay đổi đồng thời số lần chuyển, kích thước bước nhảy và path variation; không chỉ là tốc độ drift. Noise thay đổi cả target distribution và comparator. Ba seeds và OFAT không chứng minh ưu thế phổ quát, độ bền trên mọi bài toán hoặc nguyên nhân kết quả d512.

## Entry points và kiểm tra không huấn luyện

Chạy từ repository root bằng environment CPython3.12 đã cài requirements:

```powershell
python -m pip check
python -B -m scripts.check_environment --require-isolated
python -B -m scripts.smoke_tracking --mode offline --member Khoa
python -B experiments/convex_official.py --config configs/convex_cp03.yaml --dry-run
python -B experiments/convex_reconcile.py verify --help
python -B experiments/sensitivity.py --help
```

`.venv/Scripts` chứa executable environment, khác repo `scripts/`. Dùng `-m scripts.smoke_tracking`; member chỉ là nhãn, không phải uploader identity. Offline smoke không chứng minh quyền ghi online. SDK/cache nên dùng thư mục ngoài source checkout.

| Script | Phạm vi |
|---|---|
| `experiments/convex_official.py` | Official plan, durable attempts, population/MC arrays và artifact provenance |
| `experiments/convex.py` | Diagnostic runner cũ; giữ API mà official pipeline/tests sử dụng |
| `experiments/convex_diagnostics.py` | Chẩn đoán reference và numerical population oracle |
| `experiments/convex_cross_check.py` | Mã cross-check từ main8310, giữ nguyên nội dung; có replay khi chạy, không phải official uploader |
| `experiments/convex_reconcile.py` | Verifier và strict aggregate; `--read-only` không ghi proof/status/ledger đầu vào |
| `experiments/sensitivity.py` | Chỉ đọc CSV frozen và xuất hình/bảng mới; không train, không tạo W&B run |

Lệnh xuất derived figures khi cần: `python -B experiments/sensitivity.py --output-dir outputs/sensitivity-derived`. Root phải chưa tồn tại và nằm ngoài publication; trùng đường dẫn sẽ lỗi. Hình d512 mới là tổng kết cuối của đúng ba methods, không thay hình loss dynamics frozen. Provenance ghi SHA khoa học, publication, tool HEAD, script hash, source CSV hashes và output hashes.

## Khả năng chạy lại và giới hạn

[Hướng dẫn recovery/T19](convex_recovery.md) phân biệt training source và verification tool. Nhánh hiện hành đã nhận rename `configs/*.yaml` thành `configs/test.yaml` từ main5026; Windows checkout bình thường được hỗ trợ. Immutable frozen/CP057 SHAs vẫn cần no-checkout bootstrap.

Kết quả chỉ là Figure-5-style của dự án. [Paper](https://proceedings.mlr.press/v202/fahrbach23a/fahrbach23a.pdf) dùng Gaussian; [author notebook bất biến](https://github.com/fahrbach/learning-rate-schedules/blob/1b3c0d456d70c9236190ac512bbf5168527fc837/src/regression/regression_experiments.ipynb) khác về noise std, true shift, comparator, first step, projection và numeric/RNG/path. Scalar paper và relative deviation giữ N/A. Không tuyên bố EMA+d/4 đã chứng minh mọi uniform assumption của theorem.

W&B read-only ngày18/11 ghi nhận85 runs:65 dangkhoi và20 nkhoa8508, gồm18 Linear reproduction và hai smoke-test jobs mới. 60 runs khoa học cũ giữ source/config/science hashes. [Báo cáo CP056](results/cp04-cp05/WANDB_TEAM_READINESS.md) là snapshot17/11, không phải live count hiện tại. Group đề xuất: linear/convex/nonconvex; job_type mô tả mục đích. Metadata cũ không được chỉnh lại.

Khoa xác nhận own-account identity, môi trường, command, source/tool SHA và coverage T19. Hòa còn cần evidence account/write/Colab, executable data/NN contract và cross-check độc lập. Full Scope declaration có trong motivation_example; không tự chứng minh nghiệm thu. README của Khoa, workbook và issue acceptance vẫn cần review người phụ trách. [T8](02_contributions.md), [T10](06_complexity.md) và [references](references.md) phân biệt đóng góp paper với công việc dự án.
