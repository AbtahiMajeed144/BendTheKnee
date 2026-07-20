"""Load the pretrained torchcfm CIFAR-10 UNet and wrap it as a pure velocity field.

CERTIFICATE TRACK: use `cfm_cifar10_weights_step_400000.pt` (I-CFM, independent coupling) —
its target field is provably irrotational, so the certificate is valid. `otcfm_...` has
coupling-induced curl and is only used to *measure* that bias in E3.

Verified facts baked in here: velocity parameterization, GroupNorm (per-sample -> batched
JVP/VJP is correct), signature model(t, x), EMA weights are the converged ones.
"""

from __future__ import annotations

from typing import Callable

import torch

# Config confirmed from examples/images/cifar10/train_cifar10.py (both cfm_ and otcfm_).
CIFAR_UNET_CONFIG = dict(
    dim=(3, 32, 32),
    num_res_blocks=2,
    num_channels=128,
    channel_mult=[1, 2, 2, 2],
    num_heads=4,
    num_head_channels=64,
    attention_resolutions="16",
    dropout=0.1,
)


def _clean_state_dict(sd: dict) -> dict:
    """Strip a DataParallel 'module.' prefix if present."""
    if any(k.startswith("module.") for k in sd):
        return {k[len("module.") :] if k.startswith("module.") else k: v for k, v in sd.items()}
    return sd


def load_cifar_unet(
    weight_path: str,
    device: torch.device | str = "cuda",
    use_ema: bool = True,
    dtype: torch.dtype = torch.float32,
):
    """Build the UNet, load weights (EMA by default), eval + freeze. Returns the module.

    Freezing params means autograd only ever differentiates w.r.t. the input x, which is
    exactly what the estimators need and keeps the JVP/VJP graphs small.
    """
    from torchcfm.models.unet.unet import UNetModelWrapper

    model = UNetModelWrapper(**CIFAR_UNET_CONFIG).to(device=device, dtype=dtype)
    ckpt = torch.load(weight_path, map_location=device)

    if isinstance(ckpt, dict):
        key = None
        if use_ema and "ema_model" in ckpt:
            key = "ema_model"
        elif "net_model" in ckpt:
            key = "net_model"
        elif "model" in ckpt:
            key = "model"
        sd = ckpt[key] if key is not None else ckpt
    else:  # a bare module was pickled
        sd = ckpt.state_dict()

    model.load_state_dict(_clean_state_dict(sd))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def make_velocity_fn(model, t: float, device=None) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a pure function vf(x) = model(t, x) with t fixed and broadcast per sample.

    Suitable as the `vf` argument to every estimator. t is expanded to the batch inside so
    torch.func.jvp/vjp differentiate only through x.
    """
    def vf(x: torch.Tensor) -> torch.Tensor:
        tt = torch.full((x.shape[0],), float(t), device=x.device, dtype=x.dtype)
        return model(tt, x)

    return vf
