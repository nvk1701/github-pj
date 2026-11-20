# Đối chiếu provenance của ba hình CP05

Publication gốc tại commit `4d6bce54e68e559d77cb64bd5dab5eeb1cf7fac6` được giữ nguyên19 files. Main8310 có ba hình khác bytes, không kèm manifest gốc. Consolidation giữ bản publication gốc; main variant vẫn reachable trong parent8310, không xóa lịch sử.

| Hình | SHA256 publication gốc | SHA256 bản main8310 | Kết luận |
|---|---|---|---|
| `figures/cp05_d512_loss_dynamics.png` | `ca29019803da844f058e1f419b8317ce7414d3e481d4562a92cd979a627808e6` | `2191f15fdb1ffa2ad5c58aadfcf50ae43cfd463d9ea8f01067b09aaa06c289e8` | Bản main vẽ15 nhóm nhưng ghi d512; giữ loss-dynamics gốc |
| `figures/cp05_drift_sensitivity.png` | `75eb16a9cf57262f728192511f4eef7b5097a077c7ac893c76a6cc1df792aad3` | `4f119afd0532f4b81b363cf65028eae2612dae245c81456f1e06c8fbb5dc50d4` | Bản main là bar chart trình bày lại cùng CSV; không thay publication frozen |
| `figures/cp05_noise_sensitivity.png` | `539b059c8497e6da8bc0dc5d7bc7443ffabe61a83808a769045ab4e04db533d8` | `899f7d27d124849abb2f21d851084f6f94f0b0ddfc88b126c7b394fabc4b7b9e` | Bản main là bar chart trình bày lại cùng CSV; không thay publication frozen |

## Nguồn dữ liệu và nguyên nhân

Raw60 NPZ final A/B được đối chiếu read-only với CP04/CP05 per-seed CSV, kèm240 payload hashes trong proofs cũ. CSV mean/sampleSD và nguồn7e giữ nguyên. Lỗi xác nhận nằm ở plotting/path/copy của main, không phải chỉnh scientific arrays hay W&B runs.

`origin/main:experiments/sensitivity.py` tại8310 ghi thẳng vào ba đường dẫn publication. Khối d512 đọc toàn bộ15 method/dimension groups mà không lọc dimension; hai bar plots sensitivity khác style và bố cục so với hình gốc nhưng dùng cùng bảng mean/SD. Không cần đính chính số liệu publication gốc; cần phân biệt bản trình bày lại và sửa nhãn/phạm vi hình sai trên nhánh bàn giao.

## Output policy hiện hành

`experiments/sensitivity.py` chỉ xuất vào root chưa tồn tại, từ chối root trong/bao trùm publication và lỗi nếu root đã có. Default là `outputs/sensitivity-derived`. Hình mới `d512_method_summary.png` lọc đúng ba nhóm512, có mean/sampleSD; không tự nhận là loss-dynamics curves.

Output provenance ghi training SHA7e, publication4d, tool HEAD thực tế, script file SHA256, canonical-LF source CSV hashes và hashes sáu derived payloads. Các hình được tạo bằng code đang sửa phải được nhận diện bằng file hash, không chỉ HEAD chưa commit. Root output mới không được commit cùng raw logs/NPZ. Việc xuất từ CSV không huấn luyện hay sửa60 runs.

Các plotting scripts CP05 gốc được giữ trong evidence local `_reports/cp05_evidence`; chúng không phải module được publish. Figure hashes trong publication manifest và nguồn raw/CSV là căn cứ kiểm tra bản frozen. Không gán final maintenance SHA thành SHA đã tạo kết quả cũ.
