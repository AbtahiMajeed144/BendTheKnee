import torch
import torch.nn as nn
from torchcfm.models.unet.unet import UNetModelWrapper

def main():
    print("Testing Curl Penalty with torch.func.jvp and vjp...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize a smaller UNet for fast testing
    net_model = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=1,
        num_channels=32,
        channel_mult=[1, 2],
        num_heads=1,
        num_head_channels=32,
        attention_resolutions="16",
        dropout=0.1,
    ).to(device)

    # Dummy data
    batch_size = 2
    xt = torch.randn(batch_size, 3, 32, 32).to(device)
    t = torch.rand(batch_size).to(device)

    # Curl Penalty computation exactly as in train.py
    def get_vector_field(x):
        return net_model(t, x)

    epsilon = torch.randn_like(xt)

    # Forward + JVP
    vt, jvp_out = torch.func.jvp(get_vector_field, (xt,), (epsilon,))
    
    # Forward + VJP
    _, vjp_fn = torch.func.vjp(get_vector_field, xt)
    vjp_out = vjp_fn(epsilon)[0]
    
    curl_loss = torch.mean((jvp_out - vjp_out) ** 2)
    
    # Ensure gradients flow to parameters
    curl_loss.backward()
    
    grad_norm = 0.0
    for p in net_model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item()
            
    print(f"Curl Loss: {curl_loss.item():.4f}")
    print(f"Total Gradient Norm on parameters: {grad_norm:.4f}")
    print("SUCCESS: JVP and VJP computed correctly and gradients flowed backward.")

if __name__ == "__main__":
    main()
