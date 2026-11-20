# Colab riêng cho T18

**R = 5/5 VERIFIED.** Đã chạy nguyên bản Figure 5-bottom trong runtime Colab CPU qua MCP. Tệp xuất của mỗi dimension đủ 2.000 batch, khớp SHA-256 và thỏa đẳng thức tích lũy M0 theo float32. Runtime dùng Linux, Python 3.13.18, NumPy 1.26.4 và TensorFlow 2.21.0.

| d | M0 cuối Colab native |
|---:|---:|
| 2 | 0.9858183861 |
| 8 | 3.7352774143 |
| 32 | 48.0655975342 |
| 128 | 180.8743438721 |
| 512 | 1207.9738769531 |

Các cặp JSON/NPZ và log thực thi ở `outputs/colab-author-external/`. Bộ này được chuyển qua HTTPS runtime proxy và kiểm checksum tại worktree. Fixture parity được chạy trong một kernel riêng trên cùng runtime.

## Chạy nguyên bản

Mở [snapshot trong Colab](https://colab.research.google.com/github/fahrbach/learning-rate-schedules/blob/32ae9da816fc26e6a728b27fcd20255536b2fc41/src/regression/regression_experiments.ipynb) hoặc mở notebook Python trống trong Colab, upload riêng [script xuất chứng cứ của nhóm](../../scripts/export_convex_author_colab.py) và chạy trong cell:

```python
%run /content/export_convex_author_colab.py --kind native
```

Script tải notebook bất biến, kiểm Git blob, thực thi nguyên văn các cell 0, 2, 3, 6, 8, 23 trong Colab. Không cần clone nhóm vào Colab. Dùng CPU, giữ runtime và output gốc hoặc lưu notebook đã chạy, bảng phiên bản, thời gian UTC, log, memory và các cặp `native-d{2,8,32,128,512}.{json,npz}`.

Giới hạn bộ nhớ ở d=512: một ma trận float32 512000×512 là 1000 MiB. Có cả X và ma trận v lặp theo mẫu. Cell còn giữ các phase trước concatenate (thêm tối đa gần 2000 MiB), input cũ d128 trong lúc tính biểu thức vế phải, TF input/tensor copies và plotting. Dự trù runtime Colab ≥16 GiB, script kiểm còn ≥10 GiB lúc bắt đầu. Nếu OOM hoặc timeout, ghi log và dừng ca đó. Các ca hoàn thành chỉ có thể phục hồi nếu runtime còn sống. Cắt loop, stream hoặc đổi dtype sẽ thành author-adapted và cần chứng minh tương đương riêng trước khi sử dụng.

## Fixture chẩn đoán

Dùng runtime Colab riêng, cùng cấu hình phần mềm nếu muốn đối chiếu trực tiếp với R, và lưu fixture trong cùng thư mục export. Chạy script ở chế độ:

```python
%run /content/export_convex_author_colab.py --kind diagnostic
```

Chế độ này giới hạn loop ở d=2,8,32 và thêm tracing chỉ đọc tại hai điểm quanh SGD update. Không thay đổi training formula. Xuất đầy đủ 2000 batch mỗi dimension, bao gồm X, y, initial, reference, gamma, thresholds, LR, losses, gradient, theta trước/sau và M0. File `diagnostic-d*.json` chứa hash NPZ và có thể liên kết hash native tương ứng trong cùng thư mục, nếu có. Lưu thêm notebook/log thực thi và path để reviewer truy chứng.

Import các cặp JSON/NPZ vào một thư mục ngoài kho, sau đó chạy:

```powershell
python scripts/run_convex_author_full.py --report-only --output-root outputs/convex-author-release-audit --evidence-root path/to/colab-evidence
python scripts/run_convex_author_full.py --resume --wandb-mode disabled --output-root outputs/convex-author-release-audit --evidence-root path/to/colab-evidence
```

Importer kiểm URL nguồn, commit/blob, SHA-256 notebook, runtime Google Colab, thời gian UTC, phiên bản, checksum payload, dtype/shape/finite, thứ tự dữ liệu và replay. Cần đủ cặp JSON/NPZ cho từng ca. Sidecar là metadata do exporter ghi. Người duyệt vẫn cần log hoặc notebook output của lần thực thi để truy chứng runtime. F2 và F3 cần cả ba fixture đầy đủ. Full matrix được mở cổng sau khi F1, F2/F3, F4 và F5 đều có bằng chứng. Lưu bộ notebook/log gốc ngoài Git.

SHA-256 của 4.539.968 byte notebook gốc đã được kiểm bằng lần tải chỉ đọc: `c1493733e42d3226146458625431491585f0babffb8db109defdb235f13ac874`.
