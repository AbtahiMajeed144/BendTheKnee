# Flow Matching & The Curl-Guided Memorization Probe: Theoretical Foundations

## 1. Context & Research Vision
[cite_start]The objective of this research program is to explore post-processing and inference methodologies in Flow Matching (FM) to detect and reduce "knots" within the velocity field[cite: 251, 253, 254]. [cite_start]The fundamental premise is that straightening the vector field reduces the local truncation error of ODE solvers, directly minimizing the Number of Function Evaluations (NFE) required for high-quality generation[cite: 252, 268]. 

[cite_start]Our initial exploration mapped the 2023–2026 literature, identifying that while acceleration penalties (like Iso-FM and OAT-FM) [cite: 277, 278] [cite_start]and temporal schedulers (like BézierFlow) [cite: 281] [cite_start]are saturated, the geometric property of **curl (vorticity)** remains an open and highly exploitable seam[cite: 286, 287]. [cite_start]We have pivoted toward a zero-compute, inference-time "Curl-Guided Memorization Probe," translating the statistical problem of overfitting into a pure geometric diagnostic[cite: 379, 436, 437].

---

## 2. The Master Identity: Defining the "Knot"
[cite_start]Let $v_\theta(x,t)$ be the learned velocity field[cite: 262]. [cite_start]We decompose the Jacobian $\nabla v = S + A$ into its symmetric (strain-rate) and antisymmetric (vorticity) components[cite: 263]. 

The Lagrangian acceleration organizing particle dynamics is given by the master identity:
$$a(x,t) = \partial_t v + (\nabla v)v = \partial_t v + \nabla\big(\tfrac{1}{2}|v|^2\big) + 2A(x,t)v(x,t)$$
[cite_start]*(Note: $(\nabla v)v = (\nabla v)^\top v + 2Av$ and $(\nabla v)^\top v = \nabla \tfrac12|v|^2$)*[cite: 264].

[cite_start]The component $2Av$ represents the "swirl" or Lamb vector[cite: 272]. [cite_start]This rotational excess is the formal definition of our "knot"[cite: 275]. [cite_start]We rely on the Q-criterion, $\tfrac12(|A|_F^2 - |S|_F^2)$, and the Frobenius norm $|A|_F^2$ as spatial-temporal knot detectors[cite: 274].

---

## 3. The Geometry of Memorization
[cite_start]We mathematically establish that empirical FM fields inherently generate localized vorticity around isolated target data[cite: 450].

The empirical field $v_{emp}(x,t)$ is a softmax-weighted mixture of conditional optimal transport fields ($v_i$):
$$v_{emp}(x, t) = \sum_{i=1}^N w_i(x, t) v_i(x, t)$$
[cite_start]Because the underlying conditional fields are straight lines (gradient fields), they possess zero curl[cite: 382]. [cite_start]Thus, the entire antisymmetric vorticity matrix $A$ arises exclusively from the mixing weights $\nabla w_i^\top$[cite: 382]:
$$A_{emp}(x, t) = \frac{1}{2} \sum_{i=1}^N \Big( v_i \nabla w_i^\top - \nabla w_i v_i^\top \Big)$$

By expanding the spatial gradient of the Gaussian coupling weights $\nabla_x w_i$, we derive the exact geometric anatomy of a knot:
$$A_{emp}(x,t) = \frac{t}{2(1-t)^3} \sum_{i=1}^N w_i \Big( (x_i - x)(x_i - \mathbb{E}_w[X])^\top - (x_i - \mathbb{E}_w[X])(x_i - x)^\top \Big)$$
[cite_start]This proves that vorticity scales linearly with the spatial distance between a target data point $x_i$ and the collective posterior mean $\mathbb{E}_w[X]$[cite: 390]. [cite_start]Memorized outliers cause this term to explode, creating a massive, localized curl signature ("ring of fire") early in the generation timeline[cite: 392, 393, 401, 410].

---

## 4. Foundational Literature & Competitor Map
To defend this thesis, an agent must be deeply familiar with the following base papers:

* [cite_start]**Lim, S. H. (May 2026).** *On The Hidden Biases of Flow Matching Samplers* (arXiv:2512.16768)[cite: 224, 239, 240].
    * [cite_start]*Relevance:* Mathematically proves empirical minimizers are non-gradient fields with flux-null degrees of freedom, validating that vorticity is a finite-sample optimization artifact[cite: 224, 225].
* **Kornilov, N., et al. (2024)[cite_start].** *Optimal Flow Matching: Learning Straight Trajectories in Just One Step* (NeurIPS 2024)[cite: 226, 241, 242].
    * *Relevance:* Establishes that true OT trajectories are inherently straight gradient fields, enforcing this via Input Convex Neural Networks (ICNNs)[cite: 227, 228].
* [cite_start]**Khan, T. (Apr 2026).** *Isokinetic Flow Matching for Pathwise Straightening of Generative Flows* (arXiv:2604.04491)[cite: 229, 243, 244].
    * *Relevance:* Introduces Iso-FM, our primary competitor, which penalizes total acceleration. [cite_start]We must argue how our curl penalty avoids the distribution shift caused by total acceleration penalties[cite: 230, 231].
* **Yue, A., et al. (Sept 2025)[cite_start].** *OAT-FM: Optimal Acceleration Transport for Improved Flow Matching* (arXiv:2509.24936)[cite: 232, 245, 246].
    * *Relevance:* Defines acceleration transport conditions for achieving straightness[cite: 233].
* **Petrović, K., et al. (Oct 2025)[cite_start].** *Curly Flow Matching for Learning Non-gradient Field Dynamics* (arXiv:2510.26645)[cite: 234, 247, 248].
    * [cite_start]*Relevance:* Essential for understanding the opposite mathematical goal—intentionally modeling systems with inherent non-gradient dynamics (e.g., biology)[cite: 235, 236].