# Execution Blueprint: Curl-Guided Memorization Probe

## 1. The Pretrained Imperative & Seed Repositories
[cite_start]To prove that finite-sample curl is the true obstruction, all experiments must initiate from off-the-shelf pretrained Flow Matching weights[cite: 215, 219, 222]. [cite_start]Do not train from scratch, as this alters the fundamental optimization trajectory.

**Target Repositories:**
1.  [cite_start]**SiT (Scalable Interpolant Transformers)** [cite: 197]
    * [cite_start]*Source:* `github.com/willisma/SiT` [cite: 198]
    * [cite_start]*Advantage:* A 1:1 architectural match for DiT-S/2, natively trained using stochastic interpolants[cite: 197, 199]. [cite_start]Forward pass directly outputs velocity $v(x,t)$, making it trivial to wrap with the JVP/VJP penalty[cite: 199]. [cite_start]Checkpoints available via Hugging Face[cite: 200].
2.  [cite_start]**TorchCFM** [cite: 202]
    * [cite_start]*Source:* `github.com/atong01/conditional-flow-matching` (`pip install torchcfm`) [cite: 203]
    * *Advantage:* The community standard for exact OT-CFM math. [cite_start]Highly optimized, minimal implementations with pretrained CIFAR-10 checkpoints available[cite: 203, 204, 205]. [cite_start]Least abstracted framework, ideal for deep mathematical modifications[cite: 210].

---

## 2. Critical Implementation Warning: The Parameterization Trap
[cite_start]Before evaluating the Jacobian, you must verify the exact parameterization of the downloaded model[cite: 206]. 

[cite_start]If the model predicts the target data point $\hat{x}_1$ (for stability) rather than the velocity field $v_\theta$, you must analytically construct the velocity field under the hood before applying the penalty[cite: 207]:
$$v_\theta(x,t) = \frac{\hat{x}_1(x,t) - (1-\sigma_{min})x}{1 - t}$$
[cite_start]Vorticity $A$ is a property of the vector field itself[cite: 208]. [cite_start]Taking the Jacobian of the data-prediction output directly will yield mathematically meaningless gradients and ruin the evaluation[cite: 209].

---

## 3. The Three-Phase Pipeline
### Phase A: Analytical Sandbox (Zero Compute)
* [cite_start]**Goal:** Visually validate the mathematical identity[cite: 406].
* [cite_start]**Method:** Construct a 2D 8-Gaussian mixture with one extreme outlier[cite: 407, 142]. [cite_start]Compute the analytical softmax weights and exactly derive the empirical curl matrix $A$[cite: 172]. 
* [cite_start]**Output:** Plot the spatial Q-criterion heatmap to reveal the "ring of fire" (vorticity ridge) isolating the memorized outlier[cite: 409, 410, 144].

### Phase B: Inference & Hutchinson Estimation
* [cite_start]**Goal:** Scale to image space (CIFAR-10) using pretrained weights[cite: 411, 412].
* [cite_start]**Method:** Intercept the velocity output during standard Euler generation[cite: 413, 414]. [cite_start]Since full Jacobian computation is $O(d)$, approximate the Frobenius norm of the antisymmetric matrix $|A|_F^2 = -\text{Tr}(A^2)$ using Hutchinson's trace estimator[cite: 107, 108, 109, 351, 415].
* **Code Implementation:**
    $$\text{Tr}(A^2) \approx \mathbb{E}_{\epsilon}\big[ \epsilon^\top A^2 \epsilon \big]$$
    Utilize `torch.func.jvp` and `torch.func.vjp` with a random Rademacher vector $\epsilon$[cite: 111, 415]. *Note: This requires PyTorch to hold the forward, tangent, and cotangent graphs in memory simultaneously, tripling VRAM usage per instance (e.g., DiT-S/2 jumps to ~7.5 GB at batch size 128)*[cite: 180, 181].

### Phase C: Metric Validation (The Kill-Shot)
* [cite_start]**Metric Formulation:** Define the "Path Vorticity Score" for sample $x_0$ as the time-integrated curl[cite: 417]:
    $$V(x_0) = \int_{0}^{0.5} |A|_F^2(x_t, t) dt$$
* **Validation:** Generate 10,000 CIFAR-10 images. Isolate the top 100 (highest curl) and bottom 100 (lowest curl)[cite: 418, 419]. 
* **Proof:** Calculate the pixel-space $L_2$ nearest-neighbor distance to the training set for both clusters[cite: 420]. A strong correlation proves the probe operates as a mathematically guaranteed, training-data-free memorization detector[cite: 421, 434, 435].