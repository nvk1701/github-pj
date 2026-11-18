# Convex Sprint 1: hợp đồng khoa học và cách chạy

Implementation độc lập từ [PDF PMLR chính thức](https://proceedings.mlr.press/v202/fahrbach23a/fahrbach23a.pdf), Section 4, Proposition 4.3, Eq. (18)–(20). Không đọc/copy implementation tác giả. Bản R3 `docs/04_convex.md` được copy nguyên byte từ DEVELOPMENT; các đính chính ở tài liệu này, không sửa âm thầm R3.

## Schedule

```python
from core.schedules import ConvexSchedule
s = ConvexSchedule(L=.25, D_max=2, gradient_variance=.5, eta_init=1)
eta = s.step(1, gamma=.01, batch_size=256)
s.reset()
```

`gradient_variance` là **sigma²**, không phải sigma và không phải nhiễu nhãn. Constructor xác thực finite/domain; `step` yêu cầu t nguyên liên tiếp bắt đầu 1, đúng một trong `gamma` và `gamma_prev`, B nguyên dương. Các kwargs từ shared trainer như noise_std/loss_val không điều khiển schedule. `thresholds(gamma,B)` trả `(tau1,tau2)` không đổi state; `last_thresholds`, `eta_prev`, `last_t` hỗ trợ kiểm tra. Reset về eta_init trước seed mới.

Với v=sigma², g là shift signal, D=Dmax:

\[
b_1=g^2+2Dg,\quad b_2=(g+D)^2,\qquad
\tau(b)=\frac{B}{2v}\left(\sqrt{b^2L^2+4vb/B}-bL\right).
\]

Nhân liên hợp rồi chia cho bL cho dạng tương đương khi b,v>0:

\[
\tau(b)=\frac{2b}{\sqrt{b^2L^2+4vb/B}+bL}
=\frac{2}{L[1+\sqrt{1+4v/(BbL^2)}]}.
\]

Code tính dạng cuối trong log space để tránh cancellation, overflow g² và tỷ số cực trị. Chọn eta bằng ba trường hợp Eq.20. Có tests công thức gốc số nhỏ, Decimal độ chính xác cao và minimization độc lập Eq.17. Không dùng epsilon floor hoặc heuristic thay thresholds.

Biên: g=0,v>0 cho tau1=0; v=0,b>0 cho tau=1/L. Khi cả g=0 và v=0, objective không bắt buộc một LR duy nhất: convention giữ eta_prev trong [0,1/L]. eta_init>0 nên run không khởi đầu eta=0. Nếu threshold upper không thể biểu diễn dương trong float, raise ValueError thay vì trả NaN/Inf hoặc tự chọn LR. Scalar LR nằm trong (0,1/L].

## Integration và metric

Tái dùng nguyên `OnlineSGDTrainer` và spiral/EMA/baselines/projection. **Không sửa trainer, utils, drift hoặc tracking**. Thêm riêng `core/convex_eval.py` và runner, không tạo training loop thứ hai.

Trainer hiện log logistic gap bị clip về 0. Runner tắt logging nội bộ bằng `use_wandb=False`, không export `history.inst_regret/cum_regret`. Nó dùng `theta_path` trước cập nhật và evaluator riêng, sau đó log metric đã đặt tên rõ qua `init_run`.

- `train_batch_loss`: mean BCE training batch, kiểm lại bằng replay default_rng(seed).
- `train_empirical_excess`: signed training batch loss trừ reference loss, không clip.
- `eval_mc_excess`: signed mean paired loss difference trên batch evaluation độc lập.
- `eval_cum_mc_excess`: cumsum các estimates trước cập nhật, **không gọi true dynamic regret**.
- `eval_mc_excess_se`, cumulative SE: sai số MC conditional theo learned path, không thay seed-to-seed std.

Replay có assertion khớp batch loss shared trainer; nếu cách tiêu thụ RNG bên Khoa đổi thì fail rõ, không silently so hai streams khác nhau. Evaluation seed lấy từ SeedSequence([training_seed,eval_seed_stream,d]); không tiêu thụ training RNG. Cùng d/seed cho các methods có chung train/eval samples. NPZ/CSV giữ signed metric; cumulative có thể giảm, không ép monotonic.

Nếu epsilon khác 0 trong y=sigmoid(X reference+epsilon), reference không mặc nhiên là population minimizer. Test Gauss-Hermite 1D xác nhận gradient tại reference khác 0 và có vector gần đó loss thấp hơn. Với noise=0, gradient tại reference bằng 0. Sprint này dùng generating reference, chưa solve true oracle; Figure 5 true-regret reproduction vẫn còn giới hạn comparator.

## Gamma, projection và assumptions

- Mặc định `shift_mode: estimated`: EMA của theta_t−theta_(t−1), chỉ dùng updates đã hoàn tất. Scheduler nhận adapter gamma_prev từ trainer nhưng metadata ghi đây là EMA, không gọi true gamma_t.
- Optional `lagged_reference`: privileged reference transition gamma_(t−1), first 0. Đây không phải gamma_t của công thức paper. Không triển khai direct future gamma_t qua trainer, không tự lookahead.
- Ball radius R=d/2, Dmax=d: diameter=2R=Dmax. Spiral nằm trong ball với d≥2 và default amplitudes vì norm reference≤sqrt(ceil(d/2))≤d/2. Đây là lựa chọn miền tham số của nhóm, không claim tác giả dùng radius này. Config khác phải qua checks diameter và path containment.
- Profile L=.25, gradient_variance=d/4 theo tham số thí nghiệm; không claim đã chứng minh bound variance uniform trong mọi noisy model. True-oracle/domain guarantee cũng chưa được suy ra từ generating reference.
- Label noise config dùng std=sqrt(.1), diễn giải N(0,.1) là variance .1 và logistic thừa hưởng noise setup linear. Đây là reproduction assumption có ghi metadata; shared trainer default std=.1 là protocol khác.
- R3 dòng 624 nói X~Uniform(-1,1), trong khi PDF trang 9 ghi Gaussian N(0,1). Runner giữ Gaussian đúng sampler shared core. Không normalize X theo d và không đổi soft labels thành Bernoulli.

## Chạy CPU

Dùng interpreter tuyệt đối `.venv-t2-clean/Scripts/python.exe` ở Project, chạy từ worktree Convex. Không cài thêm packages. CLI chỉ chấp nhận disabled/offline; không hỗ trợ online trong Sprint 1.

```powershell
& 'D:/HOCBAI/HK7/ML/Project/.venv-t2-clean/Scripts/python.exe' -B -m unittest discover -s tests -v
& 'D:/HOCBAI/HK7/ML/Project/.venv-t2-clean/Scripts/python.exe' -B experiments/convex.py --steps 64 --dimensions 2 --seeds 0 --batch-size 32 --eval-batch-size 256 --mode disabled --output-dir outputs/convex-smoke
```

Config `configs/convex.yaml` chuẩn bị dimensions 2,8,32,128,512 và seeds0,1,2, B256,T2000,ell8; **không chạy full matrix trước khi đo pilot**. File literal `configs/*.yaml` vẫn tracked và sparse-excluded, config thật là file mới riêng. Outputs mặc định nằm trong `outputs/` đã được ignore. Mỗi method/d/seed có config.json (source hashes và assumptions), metrics.csv, history.npz, summary.json, figure.png. Không overwrite run directory cũ; chọn output khác để chạy lại. Summary chung có mean và sample std giữa seeds, single seed std=null.

Fairness mới ở stream/init/domain/protocol; default baseline LR không phải kết quả tuning tối ưu. Trước kết luận adaptive tốt hơn, cần tuning budget độc lập, cross-reproduction và đối chiếu comparator/noise/projection với nhóm. Smoke PASS không chứng minh Figure 5 reproduced. Không online W&B, commit, push hoặc đóng Issue trong Sprint 1.
