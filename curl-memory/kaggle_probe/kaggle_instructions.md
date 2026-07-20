# Kaggle Setup & Execution Guide: Curl-Guided Memorization Probe

This directory contains the proof of concept scripts for the Curl-Guided Memorization Probe. These scripts are optimized for execution within a Kaggle Notebook environment.

## 1. Environment Setup

Before running the scripts, make sure you install `torchcfm` and its dependencies. Create a cell in your Kaggle Notebook and run:

```bash
!pip install torchcfm matplotlib tqdm
```

## 2. Downloading the Pretrained Weights

You will need the pretrained CIFAR-10 OTCFM weights. Run the following cell to download them directly into the Kaggle environment:

```bash
# Create a directory to hold the weights
!mkdir -p /kaggle/working/otcfm-weights

# Download the 400k step OTCFM weights from the official repository release
!wget -O /kaggle/working/otcfm-weights/otcfm_cifar10_weights_step_400000.pt https://github.com/atong01/conditional-flow-matching/releases/download/1.0.4/otcfm_cifar10_weights_step_400000.pt
```

## 3. Running Phase A: Analytical Sandbox

This phase is zero-compute (runs on CPU) and demonstrates the theoretical mechanism of the curl probe using a 2D mixture.

```bash
!python /kaggle/working/curl-memory/kaggle_probe/phase_a_analytical.py
```
*Output: `phase_a_heatmap.png` which you can display in the notebook using `IPython.display.Image`.*

## 4. Running Phase B & C: Inference Probe and Validation

This phase uses the pretrained model, computes the Hutchinson Trace of the curl during generation, sorts the 10,000 samples by their Path Vorticity Score, and compares the top 100 and bottom 100 against the full CIFAR-10 training set.

Make sure your Kaggle Notebook has **GPU (T4 x2 or P100)** enabled.

```bash
!python /kaggle/working/curl-memory/kaggle_probe/phase_bc_probe.py
```

### Notes on Memory (VRAM)
The `phase_bc_probe.py` script uses `torch.func.jvp` and `torch.func.vjp` which significantly increases VRAM usage. The batch size is defaulted to 32 to fit comfortably in a Kaggle T4 GPU. The process of generating 10,000 images will take a while, but it will safely track progress using `tqdm`.
