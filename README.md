# nanogpt-fp8

An educational project exploring FP8 training with NVIDIA Transformer Engine. This is a modified version of nanoGPT that uses Transformer Engine layers for FP8/NVFP4 training, with architecture based on nanochat (561M parameters).

## What This Is

This project started as Karpathy's nanoGPT and evolved through experimentation with modern training techniques. The goal is to understand and test FP8 training, distributed training strategies, and various optimizations on newer NVIDIA GPUs (RTX 5090, RTX Pro 6000 - both sm120).

The codebase uses the FineWeb10B dataset following the modded-nanogpt approach by kellerjordan.

## Features

The model uses NVIDIA Transformer Engine with FP8 (DelayedScaling) or NVFP4 quantization. Training is distributed via FSDP2 (or DDP as fallback) with the Muon optimizer from Microsoft's Dion library for hidden layer weights and AdamW for embeddings.

Architecture includes RoPE positional embeddings, RMSNorm, QK normalization, logit softcapping (like Gemma2), and squared ReLU activation. Weight initialization follows the approach from arxiv.org/abs/2310.17813.

Other features include activation checkpointing for the LM head, cosine LR schedule with warmup, Muon momentum scheduling, MFU calculation, and wandb logging.

## Development History

Started from a basic transformer training on tinyshakespeare. Added DDP for multi-GPU support, then gradient accumulation and grad norm logging. Experimented with CUDA graphs and compiled optimizer. Added AMP and tested torch native Muon on H100.

Added FP8 support with compute capability checks. Got NVFP4 working. Switched to nanochat architecture (~500M params). Added fineweb dataset and distributed dataloader. Implemented FSDP, then upgraded to FSDP2. Integrated Muon from Dion with FSDP2 compatibility.

Added wandb logging and MFU tracking. Separated optimizers for different parameter groups. Implemented activation checkpointing for LM head to save VRAM. Various fixes for QK normalization (settled on RMSNorm - L2 hurts convergence), dataloader, and timing calculations.

## Requirements

- PyTorch with CUDA
- NVIDIA Transformer Engine (2.9+)
- bitsandbytes
- wandb
- Dion (Microsoft's optimizer library)

## Usage

Set up the environment:
```bash
uv venv
uv pip install -r requirements.txt
```

Download the dataset (number is how many 100M token chunks to download, max 103):
```bash
python data/cachedfineweb10b.py 9
```

Edit [run.sh](run.sh) to set the number of GPUs (`--nproc-per-node`), then:
```bash
bash run.sh
```

## Configuration

Main settings in [train.py](train.py):

- `USE_FP8` / `USE_NVFP4` - quantization mode
- `USE_FSDP2` / `USE_DDP` - distributed strategy
- `USE_AC_LM_HEAD` - activation checkpointing for LM head
- `total_batch_size`, `batch_size`, `block_size` - batch configuration
- `n_layer`, `n_embd`, `n_head` - model size (defaults: 20 layers, 1280 hidden, 10 heads)

## Results

Training on 8x RTX Pro 6000 with batch_size=16 reached val loss <3.28 in 36.5 minutes.

## Known Issues

**PyTorch dispatches sm89 kernels on sm120 GPUs.** Found this with the profiler. To use sm120 fully, you need PyTorch compiled with sm120 support which takes hours to build from source.

**No model checkpointing.** Training state and model weights are not saved. Training must complete in one run.

**No inference code.** The generate method exists but there's no standalone inference script or loading from checkpoints.

## Files

- [train.py](train.py) - main training script
- [train_orig.py](train_orig.py) - original nanoGPT for reference
- [run.sh](run.sh) - launch script for torchrun
- [data/cachedfineweb10b.py](data/cachedfineweb10b.py) - downloads FineWeb10B dataset
- [data/openwebtext-1M/](data/openwebtext-1M/) and [data/openwebtext-100k/](data/openwebtext-100k/) - alternative smaller datasets
