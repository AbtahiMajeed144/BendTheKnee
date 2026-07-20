import numpy as np
import matplotlib.pyplot as plt

def analytical_curl_probe():
    print("Executing Phase A: Analytical Sandbox (Zero Compute)")
    # 1. Setup 2D data: 8 points in a circle (inliers) + 1 extreme outlier
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    radius = 2.0
    inliers = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    outlier = np.array([[10.0, 10.0]])
    data = np.vstack([inliers, outlier]) # Shape: (9, 2)
    N = data.shape[0]

    # 2. Setup 2D Grid
    x = np.linspace(-4, 12, 200)
    y = np.linspace(-4, 12, 200)
    X, Y = np.meshgrid(x, y)
    grid = np.stack([X.ravel(), Y.ravel()], axis=1) # Shape: (40000, 2)

    # 3. Compute empirical A matrix analytically
    t = 0.1 # early timestep
    
    # Calculate w_i for all grid points and all data points
    # grid: (M, 2), data: (N, 2) -> scaled_dist: (M, N)
    scaled_grid = grid[:, None, :] # (M, 1, 2)
    scaled_data = t * data[None, :, :] # (1, N, 2)
    
    dist_sq = np.sum((scaled_grid - scaled_data)**2, axis=2) # (M, N)
    logits = -dist_sq / (2 * (1-t)**2)
    
    # numerically stable softmax
    logits_max = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    weights = exp_logits / np.sum(exp_logits, axis=1, keepdims=True) # (M, N)

    # E_w[X] = sum_i w_i * x_i
    E_w = np.sum(weights[:, :, None] * data[None, :, :], axis=1) # (M, 2)

    # Compute A_emp
    A_emp = np.zeros((grid.shape[0], 2, 2))
    scalar_factor = t / (2 * (1-t)**3)
    
    for i in range(N):
        xi = data[i] # (2,)
        diff1 = xi[None, :] - grid # (M, 2)
        diff2 = xi[None, :] - E_w # (M, 2)
        
        # Outer products -> (M, 2, 2)
        term1 = diff1[:, :, None] * diff2[:, None, :]
        term2 = diff2[:, :, None] * diff1[:, None, :]
        
        A_emp += weights[:, i, None, None] * (term1 - term2)
        
    A_emp *= scalar_factor
    
    # Frobenius norm squared of the Antisymmetric matrix
    F_norm_sq = np.sum(A_emp**2, axis=(1, 2)) # (M,)
    F_norm_sq = F_norm_sq.reshape(X.shape)
    
    # 4. Plot Heatmap
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(X, Y, F_norm_sq, shading='auto', cmap='magma')
    plt.colorbar(label='Frobenius Norm of Vorticity $|A|_F^2$')
    plt.scatter(inliers[:, 0], inliers[:, 1], c='cyan', label='Inliers', edgecolors='k', zorder=5)
    plt.scatter(outlier[:, 0], outlier[:, 1], c='red', label='Outlier', marker='*', s=300, edgecolors='k', zorder=5)
    plt.title(f'Spatial Vorticity (Curl) Heatmap at t={t}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('phase_a_heatmap.png', dpi=300)
    print("Saved Phase A heatmap to phase_a_heatmap.png. The 'ring of fire' around the outlier should be visible.")

if __name__ == "__main__":
    analytical_curl_probe()
