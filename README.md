# TokCom v5.2

Token Compression with VAE and Diffusion Model for text generation and reconstruction.

## Overview

TokCom v5.2 is a two-stage model that combines:
1. **VAE (Variational Autoencoder)**: Compresses token sequences into latent vectors
2. **Diffusion Model**: Learns to denoise latent vectors using flow matching

### Architecture

- **Tokenizer**: `meta-llama/CodeLlama-7b-hf`
- **Sequence Length**: 1024 tokens
- **Chunk Size**: 4 tokens per latent vector
- **Num Chunks**: 256 latent vectors per sequence
- **Hidden Size**: 896
- **Transformer Layers**: 24 (both VAE and Diffusion)

## Installation

```bash
pip install torch transformers datasets tqdm wandb
```

## Project Structure

```
TokCom_v5.2_zy/
├── model.py        # Model definitions (VAE, DenoisedModel)
├── dataset.py      # WikiText dataset processing
├── train.py        # Training script with wandb logging
├── inference.py    # Inference and generation script
├── config.py       # Model configuration (JSON)
└── README.md
```

## Training

### Quick Start

```bash
python train.py
```

### Training Configuration

Edit `TrainConfig` in `train.py`:

```python
class TrainConfig:
    # Dataset
    dataset_name: str = "wikitext-2-raw-v1"
    tokenizer_path: str = "meta-llama/CodeLlama-7b-hf"
    max_length: int = 1024
    chunk_size: int = 4

    # Training
    batch_size: int = 8
    num_workers: int = 4

    # Stage 1: VAE Training
    vae_epochs: int = 5
    vae_lr: float = 1e-4

    # Stage 2: Diffusion Training
    diffusion_epochs: int = 10
    diffusion_lr: float = 3e-5
    noise_std: float = 0.05

    # Sampling
    sample_every: int = 1
    num_sample_steps: int = 50
    num_samples: int = 2

    # Wandb
    wandb_project: str = "TokCom-v5.2"
```

### Training Stages

**Stage 1: VAE Training**
- Trains encoder-decoder to reconstruct token sequences
- Loss: Cross-entropy reconstruction loss
- Output: `checkpoints/best_vae.pt`

**Stage 2: Diffusion Training**
- Freezes VAE, trains denoising model
- Uses flow matching: `x_t = (1-t)*x0 + t*x1`
- Loss: Cross-entropy through frozen VAE decoder
- Output: `checkpoints/best_diffusion.pt`

### Wandb Logging

The training script logs:
- Batch-level losses (`vae/batch_loss`, `diffusion/batch_ce_loss`, `diffusion/batch_mse_loss`)
- Epoch-level metrics (train/val losses, learning rate)
- Sample reconstructions and generations as tables

## Inference

### Generate from Noise

```bash
python inference.py \
    --checkpoint checkpoints/best_diffusion.pt \
    --mode generate \
    --num_steps 50 \
    --batch_size 2
```

### Reconstruct Text

```bash
python inference.py \
    --checkpoint checkpoints/best_diffusion.pt \
    --mode reconstruct \
    --input_text "Your input text here"
```

### Encode to Latent Vectors

```bash
python inference.py \
    --checkpoint checkpoints/best_diffusion.pt \
    --mode encode \
    --input_text "Your input text here" \
    --output_file latents.pt
```

### Inference Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--checkpoint` | Path to checkpoint file | Required |
| `--mode` | `generate`, `reconstruct`, or `encode` | `generate` |
| `--input_text` | Input text (for reconstruct/encode) | None |
| `--num_chunks` | Number of chunks to generate | 256 |
| `--num_steps` | Denoising steps | 50 |
| `--batch_size` | Batch size for generation | 1 |
| `--device` | Device (cuda/cpu) | auto |
| `--output_file` | Output file path | None |

## Checkpoint Format

Checkpoints contain:

```python
{
    'epoch': int,
    'stage': str,  # 'vae' or 'diffusion'
    'val_loss': float,
    'model_args': {
        'hidden_size': 896,
        'num_hidden_layers': 24,
        'num_attention_heads': 14,
        'num_key_value_heads': 2,
        'intermediate_size': 4864,
        'vocab_size': int,
        'omni_token_id': int,
        ...
    },
    'config': {
        'tokenizer_path': str,
        'max_length': 1024,
        'chunk_size': 4,
        'noise_std': 0.05,
        ...
    },
    'vae_model_state_dict': dict,
    'denoised_model_state_dict': dict,
}
```

## Loading Models Programmatically

```python
from model import TokComVAE, DenoisedModel, ModelArgs
from inference import load_checkpoint, generate, reconstruct, encode, decode

# Load models
vae_model, denoised_model, model_args, config = load_checkpoint(
    "checkpoints/best_diffusion.pt",
    device="cuda"
)

# Generate from noise
token_ids = generate(
    vae_model, denoised_model,
    num_chunks=256,
    chunk_size=4,
    hidden_size=model_args.hidden_size,
    noise_std=config['noise_std'],
    num_steps=50,
    batch_size=1,
    device="cuda"
)

# Decode to text
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'])
text = tokenizer.decode(token_ids[0], skip_special_tokens=True)
```

## Model Details

### VAE Encoder
- Input: Token IDs `(batch, chunk_size)`
- Output: Latent vector `(batch, hidden_size)`
- Uses last position as output vector

### VAE Decoder
- Input: Latent vector `(batch, hidden_size)`
- Output: Logits `(batch, chunk_size, vocab_size)`
- Uses Omni token for sequence generation

### Diffusion Model
- Architecture: 24-layer Transformer with adaLN-Zero
- Conditioning: Timestep embedding
- Prediction: Clean data `x1` from noisy input `x_t`

### Flow Matching
- Forward: `x_t = (1-t)*x0 + t*x1`
- `x0`: Noise ~ N(0, noise_std^2)
- `x1`: Clean latent from VAE encoder
- `t`: Timestep ~ U[0, 1]

## License

MIT License
