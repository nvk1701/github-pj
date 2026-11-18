# T10 — Độ phức tạp tính toán và chi phí triển khai

Tài liệu phục vụ [Issue13](https://github.com/nvk1701/github-pj/issues/13). Các đếm phép toán được suy ra từ update/source, không phải định lý runtime được trích nguyên từ paper. Số nguồn theo [references](references.md). Regret bound khác computational complexity.

## Mô hình chi phí

T là horizon, d là số chiều, B là training batch và E là evaluation batch. Phép toán scalar fixed-precision có chi phí hằng; gradient oracle có chi phí C_g, projection tổng quát có C_Pi(d). Ball projection của dự án tốn Theta(d). K=ceil(1/kappa) là số Euler substeps; không cố định K khi so các mức rời rạc hóa. Chi phí tuning, tìm bound hoặc distribution oracle không được bao gồm nếu chưa nêu riêng.

| Thành phần | Tính rate | Update/chi phí thêm | State và nguồn |
|---|---|---|---|
| Constant/inverse-sqrt | Theta(1) | C_g+Theta(d)+C_Pi(d) | ModelTheta(d); [4]ConstantSchedule/InverseSqrtSchedule |
| Convex hai ngưỡng | Theta(1), scalar exp/log/sqrt | Cùng gradient/projection, estimator tính riêng | SchedulerTheta(1); [1]Proposition4.3, [4]ConvexSchedule |
| Learner-motion EMA | Không phải update học riêng | Theta(d) difference/norm | Previous learnerTheta(d); [4]EMADriftEstimator |
| Linear Algorithm1 | Theta(K) | Gradient/gamma tính riêng | EnvelopeTheta(1); [1]Algorithm1, không phải chi phí giải HJB tổng quát |
| Non-convex scalar rate | Theta(1) nếu đã có inputs | Không gồm loss/shift/variance estimation và DNN gradient | Rate stateTheta(1); [1]Eq25 |
| Yang noisy-gradient update | Theta(1) | C_g+Theta(d)+C_Pi(d) | ModelTheta(d); [2]§3.2Eq13, không tính bandit/META variants |
| Besbes restarted OGD | Theta(1), counter/rate | C_g+Theta(d)+C_Pi(d), resetTheta(d) | [3]§3–4; tối đaceil(T/Delta) resets |

Convex schedule-only có overhead hằng, nhưng vector estimator và gradient/projection vẫn có chi phí. Restart OGD thêm O(d ceil(T/Delta)), không nhân mỗi gradient call với Delta. Các phương pháp không mặc định có cùng assumptions, variation metric hoặc tốc độ thực nghiệm.

## Dense logistic training và lưu trữ

[4] `core/convex_eval.py:logistic_loss_gradient` tính Xtheta và X-transpose residual với Theta(Bd), loss scalar Theta(B); Gaussian sampling Theta(Bd). Update, ball projection và EMA thêm Theta(d). Tổng training là Theta(TBd), không có d-by-d Hessian. O(1) scalar schedule không biến toàn SGD thành O(T).

Online-only cần model/previous vectorTheta(d), active batchTheta(Bd). Frozen runner lưu đầy đủ theta/reference/oracle paths nên trajectory storageTheta(Td), không chỉTheta(d). Không lưu mọi training batch nên không có claim data storageTheta(TBd).

## Evaluation và numerical oracle

[4] `evaluate_history`, `gaussian_population_oracle`, `PopulationEvaluator`, `_integral`, `_moment` tách replay/independent MC khỏi training. Với số loss/comparator cố định, replay/MC tốnTheta(T(B+E)d). Dashboard stride20 không giảm full-resolution evaluation xuốngT/20.

Gọi U<=T là số exact reference norms trong cache, I_o là số root iterations; Q_o/Q_r và Q_theta,t là số integrand evaluations. Chúng phụ thuộc tolerance và geometry; adaptive quadrature không có chi phí cố định độc lập tolerance.

| Thành phần | Chi phí | Bộ nhớ/lưu trữ |
|---|---|---|
| Reference path/radial embedding | O(Td) | Theta(Td) |
| Chuẩn bị oracle/cache | O(Td+U(I_o Q_o+Q_r)) | O(Td+U) |
| Learner population risk | O(Td+sum_t Q_theta,t) | O(T) scalar arrays ngoài paths |
| Replay và independent MC | Theta(T(B+E)d) | O(max(B,E)d) active batches |
| Scalar metrics/CSV | Theta(T) | Theta(T) |
| Stored trajectories | Theta(Td) | NPZ compression phụ thuộc dữ liệu |

Rotational symmetry giảm integral về1D nhưng norms/dot products vẫnO(d). Evaluator không dimension-free. Bound bộ nhớ hữu ích: O(Td+max(B,E)d+T+U), chưa gồm Python/SDK/plot overhead và temporary library buffers. RSS đo được không phải số byte chính xác của các mảng này.

## Quy mô và thời gian đã đo

CP04 có9 runs mỗi dimension, sum(d)=682: proxy9*T*B*sum(d)=3.142.656.000 feature visits. CP05 có15 runs d128:983.040.000. Replay/MC dùngB+E cho proxy9.427.968.000 và2.949.120.000. Đây không phải FLOPs, oracle-call count hoặc dự báo runtime; nonlinear functions, integrals, nhiều passes và I/O còn tách riêng.

[Kết quả frozen](results/cp04-cp05/RESULTS.md) và báo cáo local CP04_EXECUTION_REPORT/CP05_NIGHT_EXECUTION_REPORT ghi: CP04 runner24,03 phút, verifier14,12 phút, toàn gate/results44,51 phút, sampled process-tree peak501,6MiB; CP05 runner7,86 phút, verifier4,36 phút, peak428,9MiB. Báo cáo nằm ở workspace `_reports`, cần cung cấp riêng khi review số đo. Không phải author runtime, benchmark với[2]/[3], cost theorem hoặc GPU speedup. Network/artifacts/plotting có thể chi phối wall time độc lập SGD.

Tóm tắt: scalar scheduleO(1), EMAO(d), dense trainingTheta(TBd). DNN cần cost model forward/backward riêng. Không có bằng chứng runtime speedup phổ quát. IEEE numbering và diễn đạt T4 còn cần whole-team review; không sửa workbook hoặc code Khoa/Hòa.
