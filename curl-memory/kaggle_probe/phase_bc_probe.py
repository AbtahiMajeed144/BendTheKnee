import os
import torch
import numpy as np
from torchvision import datasets, transforms
from tqdm import tqdm
from torchcfm.models.unet.unet import UNetModelWrapper

def get_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    kaggle_path = "/kaggle/input/datasets/ayush1220/cifar10/cifar10/train"
    if os.path.exists(kaggle_path):
        print(f"Loading CIFAR-10 from Kaggle dataset: {kaggle_path}")
        dataset = datasets.ImageFolder(root=kaggle_path, transform=transform)
    else:
        dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    
    print("Extracting CIFAR-10 training set for NN computation...")
    train_images = []
    # Only load first 50k to avoid huge memory spike if needed, but CIFAR10 is 50k.
    # Using DataLoader for faster extraction
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=500, shuffle=False, num_workers=2)
    for imgs, _ in tqdm(dataloader, desc="Loading CIFAR-10"):
        train_images.append(imgs)
    train_images = torch.cat(train_images, dim=0) # (50000, 3, 32, 32)
    return train_images

def euler_sampler_with_vorticity(model, device, batch_size=32, NFE=10):
    model.eval()
    
    x0 = torch.randn(batch_size, 3, 32, 32, device=device)
    xt = x0.clone()
    
    dt = 1.0 / NFE
    V_x0 = torch.zeros(batch_size, device=device)
    
    t_vals = torch.linspace(0, 1, NFE + 1, device=device)
    
    for i in range(NFE):
        t = t_vals[i] * torch.ones(batch_size, device=device)
        
        # 1. Compute velocity
        with torch.no_grad():
            vt = model(t, xt)
        
        # 2. Compute Hutchinson Trace Estimator for |A|_F^2
        if t_vals[i].item() <= 0.5:
            xt_grad = xt.clone().requires_grad_(True)
            
            def get_vf_grad(x):
                return model(t, x)
            
            epsilon = torch.randn_like(xt_grad)
            
            # Forward + JVP
            _, jvp_out = torch.func.jvp(get_vf_grad, (xt_grad,), (epsilon,))
            
            # Forward + VJP
            _, vjp_fn = torch.func.vjp(get_vf_grad, xt_grad)
            vjp_out = vjp_fn(epsilon)[0]
            
            # |A|_F^2 approx || 0.5 * (J*eps - J^T*eps) ||^2
            A_eps = 0.5 * (jvp_out - vjp_out)
            A_norm_sq = torch.sum(A_eps ** 2, dim=(1, 2, 3))
            
            V_x0 += A_norm_sq.detach() * dt
            
            # Explicitly delete intermediate graphs to free memory
            del xt_grad, epsilon, jvp_out, vjp_out, vjp_fn, A_eps, A_norm_sq
            
        # Step
        xt = xt + vt * dt
        
    return xt, V_x0

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    weight_path = "/kaggle/working/otcfm-weights/otcfm_cifar10_weights_step_400000.pt"
    
    net_model = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=128,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(device)
    
    if os.path.exists(weight_path):
        print(f"Loading weights from {weight_path}")
        checkpoint = torch.load(weight_path, map_location=device)
        net_model.load_state_dict(checkpoint["ema_model"])
    else:
        print(f"WARNING: Weights not found at {weight_path}. Using random weights.")
        
    net_model.eval()
    
    # CRITICAL FIX: Freeze all model parameters.
    # We only need gradients w.r.t input 'xt' for JVP/VJP. 
    # Freezing parameters prevents autograd from saving massive activation graphs for weight updates.
    for param in net_model.parameters():
        param.requires_grad = False
    
    # Generation Settings
    TOTAL_IMAGES = 10000
    BATCH_SIZE = 4 # Reduced drastically to avoid OOM with torch.func.jvp
    NFE = 10
    
    all_generated_images = []
    all_V_x0 = []
    
    print(f"Generating {TOTAL_IMAGES} images with NFE={NFE} and computing Path Vorticity Score...")
    num_batches = TOTAL_IMAGES // BATCH_SIZE
    
    for _ in tqdm(range(num_batches), desc="Generating Batches"):
        xt, V_x0 = euler_sampler_with_vorticity(net_model, device, batch_size=BATCH_SIZE, NFE=NFE)
        # Denormalize to [0, 1]
        xt_denorm = (xt + 1.0) / 2.0
        xt_denorm = torch.clamp(xt_denorm, 0.0, 1.0)
        
        all_generated_images.append(xt_denorm.cpu())
        all_V_x0.append(V_x0.cpu())
        
        # Free memory aggressively to prevent OOM
        del xt, V_x0, xt_denorm
        torch.cuda.empty_cache()
        
    all_generated_images = torch.cat(all_generated_images, dim=0)
    all_V_x0 = torch.cat(all_V_x0, dim=0)
    
    print("Sorting images based on Path Vorticity Score V(x0)...")
    sorted_indices = torch.argsort(all_V_x0)
    
    bottom_100_idx = sorted_indices[:100]
    top_100_idx = sorted_indices[-100:]
    
    bottom_100_imgs = all_generated_images[bottom_100_idx]
    top_100_imgs = all_generated_images[top_100_idx]
    
    print(f"Top 100 V(x0) range: {all_V_x0[top_100_idx[0]]:.4f} to {all_V_x0[top_100_idx[-1]]:.4f}")
    print(f"Bottom 100 V(x0) range: {all_V_x0[bottom_100_idx[0]]:.4f} to {all_V_x0[bottom_100_idx[-1]]:.4f}")
    
    # Load and denormalize training images
    train_images = get_dataloaders()
    train_images = (train_images + 1.0) / 2.0
    train_images = torch.clamp(train_images, 0.0, 1.0)
    
    train_flat = train_images.view(50000, -1)
    top_100_flat = top_100_imgs.view(100, -1)
    bottom_100_flat = bottom_100_imgs.view(100, -1)
    
    def compute_nn_l2_distance(query_flat, dataset_flat, batch_size=10):
        nn_dists = []
        dataset_flat = dataset_flat.to(device)
        for i in tqdm(range(0, query_flat.size(0), batch_size), desc="Computing NN"):
            q_batch = query_flat[i:i+batch_size].to(device)
            q_sq = torch.sum(q_batch**2, dim=1, keepdim=True)
            d_sq = torch.sum(dataset_flat**2, dim=1).unsqueeze(0)
            dot = torch.mm(q_batch, dataset_flat.t())
            dist_sq = q_sq + d_sq - 2*dot
            dist_sq = torch.clamp(dist_sq, min=0.0)
            dists = torch.sqrt(dist_sq)
            min_dists, _ = torch.min(dists, dim=1)
            nn_dists.append(min_dists.cpu())
            
        dataset_flat = dataset_flat.cpu()
        return torch.cat(nn_dists).mean().item()
        
    print("Computing L2 NN distances for Top 100 (Highest Curl) ...")
    top_100_mean_dist = compute_nn_l2_distance(top_100_flat, train_flat)
    
    print("Computing L2 NN distances for Bottom 100 (Lowest Curl) ...")
    bottom_100_mean_dist = compute_nn_l2_distance(bottom_100_flat, train_flat)
    
    print("-" * 50)
    print("RESULTS (The Kill-Shot)")
    print("-" * 50)
    print(f"Mean L2 NN Distance for Top 100 (Memorized/High Curl): {top_100_mean_dist:.4f}")
    print(f"Mean L2 NN Distance for Bottom 100 (Novel/Low Curl):   {bottom_100_mean_dist:.4f}")
    print("-" * 50)
    
    if top_100_mean_dist < bottom_100_mean_dist:
        print("Success: High curl correlates with closer distance to training set (Memorization).")
    else:
        print("Warning: Correlation not observed as expected.")
        
if __name__ == "__main__":
    main()
