### 1. Mechanics of "Adam Restart"

* **Paper statement**: Section 6.2 states that the baseline restarts Adam by *"resetting its parameters at the beginning of a distribution shift"*.
* **Ambiguity**:
  1. Does this mean resetting the first and second moment accumulators ($m_t \leftarrow 0$, $v_t \leftarrow 0$) and resetting the timestep counter ($t \leftarrow 0$ for bias correction)?
  2. Or does it instantiate a brand-new optimizer object entirely?
  3. Does the learning rate follow a warm-up schedule upon reset, or jump immediately to $\eta_0$?
* **Recommended default**: Clear the optimizer state buffers ($m_t, v_t$) and reset the internal step counter to $0$ while retaining the base learning rate $\eta_0$ and current model weights $\theta_t$.

### 2. Computation of Oracle Loss $\bar{\ell}_t(\theta_t^*)$ in Figure 6 (right)

* **Paper statement**: The y-axis in Figure 6 (right) reports cumulative regret:

$$
\text{Reg}(T) = \sum_{t=1}^T (\bar{\ell}_t(\theta_t) - \bar{\ell}_t(\theta_t^*))
$$

* **Ambiguity**: The paper does not specify how $\theta_t^*$ (the oracle optimal parameters for distribution $P_t$) is computed.
* *Option A*: Was an independent model trained to convergence offline on stationary batches sampled from each mixture $P_i$?
* *Option B*: Was the optimal loss assumed to be approximately zero ($\bar{\ell}_t(\theta_t^*) \approx 0$), making the plot effectively the cumulative empirical loss?
* *Option C*: Was $\theta_t^*$ evaluated on an infinite-sample approximation?
* **Action item**: Decide whether to pre-train stationary oracle models for each of the 4 regimes or approximate regret using cumulative classification loss.

### 3. Dimensionality Reduction for Figure 6 (left)

* **Paper statement**: Figure 6 (left) shows a 2D projection of the 10 RNA markers color-coded by cell type.
* **Ambiguity**: The reduction method (UMAP, t-SNE, or PCA) is not explicitly named.
* **Recommended default**: The `scVelo` pancreatic dataset comes with pre-computed UMAP coordinates (`adata.obsm['X_umap']`). If using custom projections, compute UMAP directly on the 10-dimensional space.

### 4. Mini-Batch Data Generation Pipeline

* **Paper statement**: Samples are generated from a mixture of 4,000 cells perturbed by $U \sim \text{Uniform}[0.9, 1.1]$ i.i.d.
* **Ambiguity**:
  1. For each step $t$, are cells sampled with replacement from the subset of the 4,000 cells that match label $y \sim P_t(y)$?
  2. Are the 10 RNA counts pre-normalized (e.g., library size normalized, log1p transformed) before applying multiplicative noise $U$?
* **Recommended default**: Apply standard `scv.pp.filter_and_normalize` preprocessing, then draw categorical labels $y \sim P_t$, pick random cells matching $y$, and multiply expression values by $U[0.9, 1.1]$.

### 5. Theory-to-Practice Gap (Section 5 vs. Section 6.2)

* **Observation**: Section 5 provides an analytical schedule $\eta_t^*$ for SGD on non-convex objectives, but the neural network experiment in Section 6.2 evaluates Adam with fixed learning rates and manual restarts.
* **Reason**: Dynamic estimation of the smoothness parameter $L$, gradient variance $\sigma^2$, and global loss shift $\gamma_t = \sup_\theta |\bar{\ell}_t(\theta) - \bar{\ell}_{t+1}(\theta)|$ is non-trivial for deep neural networks.
* **Action item**: Verify whether the project goal is solely reproducing Figure 6 (right) with Adam/Adam-restart, or evaluating an empirical surrogate of the closed-form schedule $\eta_t^*$ on the same MLP benchmark.