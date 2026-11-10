## Theoretical Framework

### Problem Setup

* Parameter vector: $\theta \in \mathbb{R}^p$
* Data stream: at each round $t$, a batch $z_{t,1}, \dots, z_{t,B_t} \sim P_t$ is drawn.
* Expected loss: $\bar{\ell}_t(\theta) := \mathbb{E}_{P_t}[\ell(\theta, z_{t,k})]$
* Mini-batch empirical loss: $\ell_{B_t}(\theta) := \frac{1}{B_t} \sum_{k=1}^{B_t} \ell(\theta, z_{t,k})$
* Model update: $\theta_{t+1} = \theta_t - \eta_t \nabla \ell_{B_t}(\theta_t)$

### Assumptions

1. **$L$-smoothness**: $\bar{\ell}_t$ is differentiable and $\lVert \nabla \bar{\ell}_t(\theta) - \nabla \bar{\ell}_t(\theta') \rVert \le L\lVert \theta - \theta' \rVert$ for all $\theta, \theta' \in \mathbb{R}^p$.
2. **Bounded Variance**: $\mathbb{E}_{z \sim P_t}[\lVert \nabla \ell(\theta, z) - \nabla \bar{\ell}_t(\theta) \rVert^2] \le \sigma^2$ for all $\theta \in \mathbb{R}^p$.
3. **Step Size Bound**: $\eta_t \le 1/L$, ensuring $a_t := 2\eta_t - L\eta_t^2 > 0$.

### Non-Convex Regret and Step Size Derivation

* **Regret metric**: $\text{Reg}(T) := \sum_{t=1}^T \lVert \nabla \bar{\ell}_t(\theta_t) \rVert^2$
* **Shift magnitude**: $\gamma_t := \sup_{\theta \in \mathbb{R}^p} |\bar{\ell}_t(\theta) - \bar{\ell}_{t+1}(\theta)|$
* **Optimal adaptive step size**:

$$
\eta_t^* = \frac{B_t}{L\sigma^2} \left( \sqrt{b_t^2 + \frac{2\sigma^2}{B_t}b_t} - b_t \right), \quad \text{where } b_t := L\left(\gamma_t + \bar{\ell}_t(\theta_t)\right)
$$

## Neural Network Benchmark Specifications

### Experimental Protocol Table

| Parameter / Component | Exact Specification from Paper | Notes / Implementation Values |
| :--- | :--- | :--- |
| **Dataset Source** | `scVelo` single-cell RNA-seq (murine pancreas) | 4,000 cells (Bastidas-Ponce et al., 2019) |
| **Feature Dimensions** | 10 RNA expression markers | *Pyy, Meg3, Malat1, Gcg, Gnas, Actb, Ghrl, Rsp3, Ins2, Hspa8* |
| **Class Labels ($K=4$)** | 4 cell types | `ductal`, `alpha`, `beta`, `delta` |
| **Input Perturbation** | Multiplicative uniform noise | $x_j \leftarrow x_j \cdot U$, where $U \sim \text{Uniform}[0.9, 1.1]$ |
| **Mixture Regimes ($P_t$)** | 4 sequential distribution stages ($T = 600$) | • Steps 1–100: $P_1(y) = (0.0, 0.0, 0.0, 1.0)$<br />• Steps 101–200: $P_2(y) = (0.0, 0.0, 0.1, 0.9)$<br />• Steps 201–400: $P_3(y) = (0.1, 0.0, 0.2, 0.7)$<br />• Steps 401–600: $P_4(y) = (0.3, 0.5, 0.1, 0.1)$ |
| **Model Architecture** | 5-hidden-layer feedforward MLP | Layer sizes: `[64, 32, 16, 8, 4]` |
| **Hidden Activation** | Exponential Linear Unit (`ELU`) | Applied after all hidden dense layers |
| **Output Layer** | Linear dense layer (4 units) | Unnormalized class logits |
| **Loss Function** | Categorical Cross-Entropy | `tf.keras.losses.CategoricalCrossentropy(from_logits=True)` |
| **Base Optimizer** | Adam | Kingma & Ba (2014) |
| **Mini-Batch Size ($B_t$)** | 64 fresh samples per step | Evaluated online (one-pass streaming) |
| **Total Steps ($T$)** | 600 gradient steps | Split into horizons: $100 + 100 + 200 + 200$ |
| **Evaluated Baselines** | Fixed vs. Restarting schedules | • Fixed Adam: $\eta_0 \in \{0.001, 0.003, 0.01, 0.03, 0.1\}$<br />• Restart Adam: $\eta_0 \in \{0.01, 0.03\}$ with reset at $t \in \{100, 200, 400\}$ |
| **Primary Metric** | Cumulative dynamic regret | $\text{Reg}(T) = \sum_{t=1}^T (\bar{\ell}_t(\theta_t) - \bar{\ell}_t(\theta_t^*))$ |

## Targets for Reproduction

1. **Figure 6 (left)**:
   * 2D embedding of the 10-marker single-cell RNA dataset.
   * Clusters colored according to the 4 cell lineages (`ductal`, `alpha`, `beta`, `delta`).
2. **Figure 6 (right)**:
   * Cumulative regret curve spanning steps $t = 0$ to $t = 600$.
   * Four benchmark curves: `adam:0.01`, `adam-restart:0.01`, `adam:0.03`, `adam-restart:0.03`.
   * Vertical dashed demarcation lines at distribution shift boundaries ($t = 100, 200, 400$).

## Key Experimental Findings

1. **Sweet Spot in Learning Rate**:
   * $\eta_0 = 0.01$ achieves minimal regret across fixed step sizes.
   * $\eta_0 = 0.1$ diverges early due to instability.
   * Rates below $0.01$ adapt too slowly to shifting label proportions.
2. **Benefit of Optimizer Resets**:
   * Resetting Adam state buffers upon distribution shift noticeably lowers cumulative regret for both $\eta_0 = 0.01$ and $\eta_0 = 0.03$.
   * Resetting clears stale second-order moment accumulators that otherwise restrict step sizes when loss gradients suddenly spike.
3. **Local Minima Trapping**:
   * Standard Adam without adaptation gets trapped in outdated local minima during later shifts (observable as constant non-diminishing slopes in regret during the final phases of `adam:0.03`).