# Execution Blueprint v2.0: The Architectural Failure Probe

## 1. The Core Paradigm: Measuring Architectural Breakdown
[cite_start]Our mathematical stress-test proved that the true theoretical OT-CFM velocity field is exactly the gradient of a scalar potential, possessing absolutely zero curl everywhere, regardless of the data distribution[cite: 304, 305, 319]. 

[cite_start]Therefore, our intervention does not measure the geometry of the data; it measures the failure of the neural network[cite: 307, 310, 366]. [cite_start]Standard architectures (like UNets and DiTs) lack the inductive bias to model infinitely steep, purely irrotational gradients[cite: 327]. [cite_start]When forced to fit an isolated, memorized outlier, the network's approximation capacity buckles, taking the path of least resistance and "spinning out" into rotational, flux-null curly fields[cite: 308, 328, 332]. 

[cite_start]This implementation leverages that exact architectural failure as a zero-compute, training-data-free memorization detector[cite: 368].

---

## 2. The Target Repository: TorchCFM
To rigorously test this without confounding variables, we must use a codebase that is mathematically transparent and relies on unconstrained architectures. [cite_start]Do not use multiple models for the initial Proof of Concept (PoC)[cite: 340]. [cite_start]Stick exclusively to TorchCFM[cite: 348].

* [cite_start]**Source:** `github.com/atong01/conditional-flow-matching` (installable via `pip install torchcfm`)[cite: 232].
* [cite_start]**The Architecture:** It provides highly optimized, minimal implementations of DiT and UNet applied to CIFAR-10 using Exact OT-CFM[cite: 233]. [cite_start]Pretrained checkpoints are readily available[cite: 234].
* [cite_start]**The Advantage:** It is the least abstracted framework available, exposing the raw ODE/PDE components exactly where you need to inject the Hutchinson estimator, making our inference probe effectively plug-and-play[cite: 350].

---

## 3. The Three-Phase Pipeline

### Phase A: The Control Baseline & Toy Neural Failure (2D Sandbox)
Before touching CIFAR-10, we must perfectly replicate the theory in a 2D environment.
1.  **The Mathematical Control:** Create a 2D 8-Gaussian mixture with one extreme outlier. [cite_start]Compute the analytical empirical velocity field $v_{emp}$ using the `logsumexp` trick for numerical stability[cite: 275, 510, 511]. [cite_start]**Expected Output:** The spatial Q-criterion (curl) must evaluate to machine epsilon (e.g., $10^{-29}$) everywhere[cite: 320]. [cite_start]This proves the theoretical baseline is perfectly irrotational[cite: 318, 319].
2.  **The Neural Failure:** Train a simple, unconstrained MLP (Multi-Layer Perceptron) on this exact same 2D distribution using standard Flow Matching loss. 
3.  **The Visualization:** Plot the spatial Q-criterion of the *trained MLP's* vector field. **Expected Output:** A localized spike in vorticity ("ring of fire") exclusively isolating the memorized outlier, proving that standard networks generate curl where they fail to fit steep gradient cliffs.

### Phase B: The Inference Interception (CIFAR-10)
[cite_start]Scale to image space using an off-the-shelf, pretrained TorchCFM DiT on CIFAR-10[cite: 163, 164].
1.  [cite_start]**Execution:** Run a standard ODE generation loop (e.g., Euler method)[cite: 164].
2.  [cite_start]**The Estimator:** At each step $t \in [0.1, 0.5]$, intercept the velocity output[cite: 152, 165]. [cite_start]Compute the Frobenius norm of the antisymmetric matrix $|A|_F^2$ using Hutchinson's trace estimator[cite: 166, 474, 475, 476].
    $$\text{Tr}(A^2) \approx \mathbb{E}_{\epsilon}\big[ \epsilon^\top A^2 \epsilon \big]$$
3.  **Code Implementation:** Utilize `torch.func.jvp` and `torch.func.vjp` with a random Rademacher vector $\epsilon$[cite: 477, 478, 479]. *(Note: Be aware of the VRAM footprint; this requires holding the forward, tangent, and cotangent graphs in memory simultaneously, increasing VRAM usage by 2.5x to 3x per instance [cite: 548, 549]).*

### Phase C: Metric Validation (The Kill-Shot)
To publish, we must unequivocally prove the correlation between high neural curl and data regurgitation.
1.  [cite_start]**The Metric:** Define the **Path Vorticity Score** $V(x_0)$ for a generated sample $x_0$ as the time-integrated curl during the early stages of generation (e.g., $t=0$ to $t=0.5$), effectively filtering out benign, high-frequency manifold curl[cite: 168].
2.  **The Test:** Generate 10,000 CIFAR-10 images. [cite_start]Sort them by their $V(x_0)$ score[cite: 169]. [cite_start]Isolate the top 100 (highest architectural failure) and bottom 100 (lowest architectural failure)[cite: 170].
3.  [cite_start]**The Kill-Shot:** Calculate the pixel-space $L_2$ nearest-neighbor distance to the original CIFAR-10 training set for both groups[cite: 171]. [cite_start]A strong correlation—showing high-curl images are significantly closer to the training data—cements this metric as a reliable, training-data-free memorization probe[cite: 172, 367, 368].