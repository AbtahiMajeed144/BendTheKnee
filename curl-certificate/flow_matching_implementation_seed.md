# Implementation Seed Instructions: Curl-Certificate Fine-Tuning
## Phase 1: 2D Closed-Form Testbed (Theory Validation)

### Objective
Visualize the theoretical origin of vorticity without neural networks by analytically deriving the exact empirical field for an 8-Gaussian mixture.

### Seed Instructions
1.  **Setup the Grid:** Define a 2D meshgrid (e.g., bounds `[-5, 5]`).
2.  **Define the Dataset:** Create an exact 8-point Gaussian mixture as $p_1$.
3.  **Compute Exact Softmax:** Calculate $w_i(x,t) \propto \mathcal{N}(x; t x_i, (1-t)^2 I)$ analytically over the grid for a given $t$ (e.g., $t=0.5$).
4.  **Derive the Field:** Construct $v_{emp}(x,t) = \sum w_i v_i$.
5.  **Compute & Plot Curl:** Use `numpy.gradient` to approximate the Jacobian over the grid. Extract $A_{emp} = \frac{1}{2}(J - J^\top)$. Plot the Q-criterion $|A|_F^2$ as a heatmap. You will observe hotspots exactly at the Voronoi boundaries (softmax ridges) between the target Gaussians.

---

## Phase 2: CIFAR-10 Scale-Up (Kaggle T4 Environment)

### Environment Constraints (Kaggle dual T4, 16GB VRAM each)
*   The Hutchinson trace estimator requires simultaneous storage of the forward graph, JVP tangent graph, and VJP cotangent graph.
*   **VRAM Allocation:** A standard DiT-S/2 consumes ~2.5 GB at batch size 128. With the curl penalty, this scales to ~7.5 GB per instance.
*   **Stacking Limit:** To maintain optimal ~25% VRAM headroom, deploy a maximum of **1 model instance per T4 GPU** at batch size 128 (yielding 2 concurrent experiments per Kaggle session).

### Pretrained Weights Selection
**Do not train from scratch.** Fine-tuning isolated the effect of the curl penalty on pre-existing rotational artifacts.

**Recommended Base:** TorchCFM (`pip install torchcfm`)
*   **Why:** Provides the most mathematically transparent, unabstracted exact OT-CFM implementation, making JVP/VJP injection via `torch.func` plug-and-play.
*   **Source:** [github.com/atong01/conditional-flow-matching](https://github.com/atong01/conditional-flow-matching)
*   **Alternative:** SiT (Scalable Interpolant Transformers) pretrained checkpoints for DiT-S/2.

### Critical Implementation Warning: The Parameterization Trap
Before applying `torch.func.jvp` and `vjp`, audit the pretrained checkpoint's raw output.
1.  **Direct Velocity ($v_\theta$):** If the network predicts the velocity field directly, you may apply the Hutchinson estimator to the raw output.
2.  **Target Prediction ($\hat{x}_1$):** If the network stabilizes training by predicting the target data point, you **MUST** analytically reconstruct the velocity field within the forward pass wrapper:
    $$v_{derived}(x,t) = \frac{\hat{x}_1(x,t) - (1-\sigma_{min})x}{1 - t}$$
    Apply the curl penalty to $v_{derived}$. Taking the Jacobian of the data prediction tensor will yield nonsensical gradients and catastrophic training collapse.

### Fine-Tuning Hyperparameters
*   **Epochs:** 20 to 50 epochs (typically enough for a pretrained model to adapt path trajectories to a new regularizer).
*   **Loss:** $\mathcal{L} = \mathcal{L}_{CFM} + \lambda \mathcal{L}_{Curl}$. Start with $\lambda$ relatively small (e.g., $10^{-3}$) to prevent over-regularization dominating the base vector field anchoring.
