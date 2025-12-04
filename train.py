import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer
import os
from tqdm import tqdm
import wandb

from dataset import WikiTextDataset, make_collate_fn
from model import TokComVAE, DenoisedModel, ModelArgs, sample_timesteps, sample_noise
from inference import generate, reconstruct, decode


# ============================================================================
# Training Configuration
# ============================================================================

class TrainConfig:
    # Dataset
    dataset_name: str = "wikitext-2-raw-v1"
    tokenizer_path: str = "meta-llama/CodeLlama-7b-hf"
    max_length: int = 1024
    chunk_size: int = 4  # window_length

    # Training
    batch_size: int = 8
    num_workers: int = 4

    # Stage 1: VAE Training
    vae_epochs: int = 5
    vae_lr: float = 1e-4
    vae_weight_decay: float = 0.01

    # Stage 2: Diffusion Training
    diffusion_epochs: int = 10
    diffusion_lr: float = 3e-5
    diffusion_weight_decay: float = 0.01
    noise_std: float = 0.05  # Standard deviation for x0 noise

    # Model
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Checkpointing
    save_dir: str = "./checkpoints"

    # Wandb
    wandb_project: str = "TokCom-v5.2"
    wandb_run_name: str = None  # Auto-generated if None

    # Sampling
    sample_every: int = 1  # Sample every N epochs
    num_sample_steps: int = 50  # Number of denoising steps for sampling
    num_samples: int = 2  # Number of samples to generate


# ============================================================================
# Training Functions
# ============================================================================

def train_vae_epoch(
    model: TokComVAE,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    dtype: torch.dtype,
    epoch: int,
    global_step: int
):
    """Train VAE for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"VAE Train Epoch {epoch}")
    for batch in pbar:
        # input_ids shape: (batch_size, num_chunks, chunk_size)
        input_ids = batch['input_ids'].to(device)
        batch_size, num_chunks, chunk_size = input_ids.shape

        # Flatten chunks for processing: (batch_size * num_chunks, chunk_size)
        input_ids_flat = input_ids.view(-1, chunk_size)

        # Forward pass through VAE
        with torch.autocast(device_type=device, dtype=dtype):
            logits, latent_vector = model(input_ids_flat, decode_len=chunk_size)
            # logits shape: (batch_size * num_chunks, chunk_size, vocab_size)

            # Compute reconstruction loss (cross-entropy)
            logits_flat = logits.view(-1, logits.size(-1))  # (B*num_chunks*chunk_size, vocab_size)
            targets_flat = input_ids_flat.view(-1)  # (B*num_chunks*chunk_size,)
            loss = F.cross_entropy(logits_flat, targets_flat)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        global_step += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Log batch loss to wandb
        wandb.log({
            "vae/batch_loss": loss.item(),
            "vae/global_step": global_step,
        })

    avg_loss = total_loss / num_batches
    return avg_loss, global_step


@torch.no_grad()
def validate_vae_epoch(
    model: TokComVAE,
    dataloader: DataLoader,
    device: str,
    dtype: torch.dtype
):
    """Validate VAE for one epoch."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc="VAE Validation")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        batch_size, num_chunks, chunk_size = input_ids.shape
        input_ids_flat = input_ids.view(-1, chunk_size)

        with torch.autocast(device_type=device, dtype=dtype):
            logits, latent_vector = model(input_ids_flat, decode_len=chunk_size)
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = input_ids_flat.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat)

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / num_batches
    return avg_loss


def train_diffusion_epoch(
    vae_model: TokComVAE,
    denoised_model: DenoisedModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    dtype: torch.dtype,
    epoch: int,
    global_step: int,
    noise_std: float = 0.05
):
    """Train diffusion model for one epoch with frozen VAE."""
    vae_model.eval()  # VAE is frozen
    denoised_model.train()
    total_loss = 0.0
    total_mse_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Diffusion Epoch {epoch}")
    for batch in pbar:
        # input_ids shape: (batch_size, num_chunks, chunk_size)
        input_ids = batch['input_ids'].to(device)
        batch_size, num_chunks, chunk_size = input_ids.shape

        # Flatten chunks: (batch_size * num_chunks, chunk_size)
        input_ids_flat = input_ids.view(-1, chunk_size)

        with torch.no_grad():
            # Get latent vectors from VAE encoder (x1 = clean latent)
            with torch.autocast(device_type=device, dtype=dtype):
                x1 = vae_model.encoder(input_ids_flat)  # (B*num_chunks, hidden_size)

        # Reshape to sequence: (batch_size, num_chunks, hidden_size)
        x1 = x1.view(batch_size, num_chunks, -1)

        # Sample timesteps t ~ U[0, 1]
        t = sample_timesteps(batch_size, device=device)

        # Sample noise x0 ~ N(0, noise_std^2)
        x0 = sample_noise(x1.shape, device=device, std=noise_std)

        # Create noisy data using flow matching interpolation: x_t = (1-t)*x0 + t*x1
        t_expanded = t[:, None, None]  # (batch_size, 1, 1)
        x_t = (1 - t_expanded) * x0 + t_expanded * x1

        # Forward pass through denoised model
        with torch.autocast(device_type=device, dtype=dtype):
            # Predict x1 from x_t
            x1_pred = denoised_model(x_t, t)  # (batch_size, num_chunks, hidden_size)

            # MSE loss (for logging only, no gradient)
            with torch.no_grad():
                mse_loss = F.mse_loss(x1_pred, x1)

            # Decode x1_pred through VAE decoder to compute cross-entropy loss
            # Flatten x1_pred: (batch_size * num_chunks, hidden_size)
            x1_pred_flat = x1_pred.view(-1, x1_pred.size(-1))

            # Pass through VAE decoder (frozen, no grad for decoder params)
            # But we need grad to flow back to denoised_model through x1_pred
            logits = vae_model.decoder(x1_pred_flat, decode_len=chunk_size, return_logits=True)
            # logits shape: (batch_size * num_chunks, chunk_size, vocab_size)

            # Compute cross-entropy loss (this is the training loss)
            logits_flat = logits.view(-1, logits.size(-1))  # (B*num_chunks*chunk_size, vocab_size)
            targets_flat = input_ids_flat.view(-1)  # (B*num_chunks*chunk_size,)
            loss = F.cross_entropy(logits_flat, targets_flat)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(denoised_model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_mse_loss += mse_loss.item()
        num_batches += 1
        global_step += 1
        pbar.set_postfix({
            'ce_loss': f'{loss.item():.4f}',
            'mse': f'{mse_loss.item():.6f}'
        })

        # Log batch loss to wandb
        wandb.log({
            "diffusion/batch_ce_loss": loss.item(),
            "diffusion/batch_mse_loss": mse_loss.item(),
            "diffusion/global_step": global_step,
        })

    avg_loss = total_loss / num_batches
    avg_mse = total_mse_loss / num_batches
    return avg_loss, avg_mse, global_step


@torch.no_grad()
def validate_diffusion_epoch(
    vae_model: TokComVAE,
    denoised_model: DenoisedModel,
    dataloader: DataLoader,
    device: str,
    dtype: torch.dtype,
    noise_std: float = 0.05
):
    """Validate diffusion model for one epoch."""
    vae_model.eval()
    denoised_model.eval()
    total_loss = 0.0
    total_mse_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Diffusion Validation")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        batch_size, num_chunks, chunk_size = input_ids.shape
        input_ids_flat = input_ids.view(-1, chunk_size)

        with torch.autocast(device_type=device, dtype=dtype):
            x1 = vae_model.encoder(input_ids_flat)

        x1 = x1.view(batch_size, num_chunks, -1)

        t = sample_timesteps(batch_size, device=device)
        x0 = sample_noise(x1.shape, device=device, std=noise_std)
        t_expanded = t[:, None, None]
        x_t = (1 - t_expanded) * x0 + t_expanded * x1

        with torch.autocast(device_type=device, dtype=dtype):
            x1_pred = denoised_model(x_t, t)
            mse_loss = F.mse_loss(x1_pred, x1)

            x1_pred_flat = x1_pred.view(-1, x1_pred.size(-1))
            logits = vae_model.decoder(x1_pred_flat, decode_len=chunk_size, return_logits=True)
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = input_ids_flat.view(-1)
            ce_loss = F.cross_entropy(logits_flat, targets_flat)

        total_loss += ce_loss.item()
        total_mse_loss += mse_loss.item()
        num_batches += 1
        pbar.set_postfix({
            'val_ce': f'{ce_loss.item():.4f}',
            'val_mse': f'{mse_loss.item():.6f}'
        })

    avg_loss = total_loss / num_batches
    avg_mse = total_mse_loss / num_batches
    return avg_loss, avg_mse


@torch.no_grad()
def sample_and_log_vae(
    vae_model: TokComVAE,
    val_loader: DataLoader,
    tokenizer,
    device: str,
    dtype: torch.dtype,
    epoch: int,
    num_samples: int = 2
):
    """Sample reconstructions from VAE and log to wandb."""
    vae_model.eval()

    # Get a batch from validation set
    batch = next(iter(val_loader))
    input_ids = batch['input_ids'].to(device)
    batch_size, num_chunks, chunk_size = input_ids.shape

    # Take only num_samples
    input_ids = input_ids[:num_samples]

    # Flatten to (num_samples, seq_len)
    input_ids_flat = input_ids.view(num_samples, -1)

    # Reconstruct
    reconstructed_ids = reconstruct(vae_model, input_ids_flat, chunk_size)

    # Log samples
    table_data = []
    for i in range(num_samples):
        original_text = tokenizer.decode(input_ids_flat[i], skip_special_tokens=True)
        reconstructed_text = tokenizer.decode(reconstructed_ids[i], skip_special_tokens=True)

        # Calculate token accuracy
        accuracy = (input_ids_flat[i] == reconstructed_ids[i]).float().mean().item() * 100

        # Truncate for display
        original_text = original_text[:300] + "..." if len(original_text) > 300 else original_text
        reconstructed_text = reconstructed_text[:300] + "..." if len(reconstructed_text) > 300 else reconstructed_text

        table_data.append([i+1, original_text, reconstructed_text, f"{accuracy:.2f}%"])

    # Log to wandb
    table = wandb.Table(columns=["Sample", "Original", "Reconstructed", "Token Accuracy"], data=table_data)
    wandb.log({f"vae/samples_epoch_{epoch}": table, "vae/epoch": epoch})

    print(f"  VAE samples logged to wandb")


@torch.no_grad()
def sample_and_log_diffusion(
    vae_model: TokComVAE,
    denoised_model: DenoisedModel,
    val_loader: DataLoader,
    tokenizer,
    model_args: ModelArgs,
    config: TrainConfig,
    epoch: int,
    num_samples: int = 2
):
    """Sample generations from diffusion model and log to wandb."""
    vae_model.eval()
    denoised_model.eval()

    # Get a batch from validation set for reconstruction comparison
    batch = next(iter(val_loader))
    input_ids = batch['input_ids'].to(config.device)
    batch_size, num_chunks, chunk_size = input_ids.shape

    # 1. Generate from noise
    print(f"  Generating {num_samples} samples from noise...")
    generated_ids = generate(
        vae_model, denoised_model,
        num_chunks=num_chunks,
        chunk_size=chunk_size,
        hidden_size=model_args.hidden_size,
        noise_std=config.noise_std,
        num_steps=config.num_sample_steps,
        batch_size=num_samples,
        device=config.device
    )

    # Log generated samples
    gen_table_data = []
    for i in range(num_samples):
        generated_text = tokenizer.decode(generated_ids[i], skip_special_tokens=True)
        generated_text = generated_text[:500] + "..." if len(generated_text) > 500 else generated_text
        gen_table_data.append([i+1, generated_text])

    gen_table = wandb.Table(columns=["Sample", "Generated Text"], data=gen_table_data)
    wandb.log({f"diffusion/generated_epoch_{epoch}": gen_table, "diffusion/epoch": epoch})

    # 2. Reconstruction through diffusion (encode -> add noise -> denoise -> decode)
    input_ids_sample = input_ids[:num_samples]
    input_ids_flat = input_ids_sample.view(num_samples, -1)

    # Encode to latent
    input_ids_chunked = input_ids_sample.view(-1, chunk_size)
    with torch.autocast(device_type=config.device, dtype=config.dtype):
        x1 = vae_model.encoder(input_ids_chunked)
    x1 = x1.view(num_samples, num_chunks, -1)

    # Add noise (t=0.5 as example)
    t = torch.full((num_samples,), 0.5, device=config.device)
    x0 = sample_noise(x1.shape, device=config.device, std=config.noise_std)
    t_expanded = t[:, None, None]
    x_t = (1 - t_expanded) * x0 + t_expanded * x1

    # Denoise
    x1_pred = denoised_model(x_t, t)

    # Decode
    reconstructed_ids = decode(vae_model, x1_pred, chunk_size)

    # Log reconstruction samples
    recon_table_data = []
    for i in range(num_samples):
        original_text = tokenizer.decode(input_ids_flat[i], skip_special_tokens=True)
        reconstructed_text = tokenizer.decode(reconstructed_ids[i], skip_special_tokens=True)
        accuracy = (input_ids_flat[i] == reconstructed_ids[i]).float().mean().item() * 100

        original_text = original_text[:300] + "..." if len(original_text) > 300 else original_text
        reconstructed_text = reconstructed_text[:300] + "..." if len(reconstructed_text) > 300 else reconstructed_text

        recon_table_data.append([i+1, original_text, reconstructed_text, f"{accuracy:.2f}%"])

    recon_table = wandb.Table(columns=["Sample", "Original", "Denoised Reconstruction", "Token Accuracy"], data=recon_table_data)
    wandb.log({f"diffusion/reconstructed_epoch_{epoch}": recon_table, "diffusion/epoch": epoch})

    print(f"  Diffusion samples logged to wandb")


def save_best_checkpoint(
    vae_model: TokComVAE,
    denoised_model: DenoisedModel,
    model_args: ModelArgs,
    config: TrainConfig,
    epoch: int,
    val_loss: float,
    stage: str,
    save_dir: str
):
    """Save best checkpoint based on validation loss."""
    os.makedirs(save_dir, exist_ok=True)

    # Save model args as dict
    model_args_dict = {
        'hidden_size': model_args.hidden_size,
        'num_hidden_layers': model_args.num_hidden_layers,
        'num_attention_heads': model_args.num_attention_heads,
        'num_key_value_heads': model_args.num_key_value_heads,
        'intermediate_size': model_args.intermediate_size,
        'rms_norm_eps': model_args.rms_norm_eps,
        'rope_theta': model_args.rope_theta,
        'vocab_size': model_args.vocab_size,
        'omni_token_id': model_args.omni_token_id,
    }

    # Save training config as dict
    config_dict = {
        'dataset_name': config.dataset_name,
        'tokenizer_path': config.tokenizer_path,
        'max_length': config.max_length,
        'chunk_size': config.chunk_size,
        'noise_std': config.noise_std,
    }

    checkpoint = {
        'epoch': epoch,
        'stage': stage,
        'val_loss': val_loss,
        'model_args': model_args_dict,
        'config': config_dict,
        'vae_model_state_dict': vae_model.state_dict(),
        'denoised_model_state_dict': denoised_model.state_dict(),
    }

    path = os.path.join(save_dir, f"best_{stage}.pt")
    torch.save(checkpoint, path)
    print(f"Best {stage} checkpoint saved to {path} (val_loss: {val_loss:.4f})")


# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    config = TrainConfig()

    # Initialize wandb
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config={
            # Dataset
            "dataset_name": config.dataset_name,
            "tokenizer_path": config.tokenizer_path,
            "max_length": config.max_length,
            "chunk_size": config.chunk_size,
            # Training
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            # VAE
            "vae_epochs": config.vae_epochs,
            "vae_lr": config.vae_lr,
            "vae_weight_decay": config.vae_weight_decay,
            # Diffusion
            "diffusion_epochs": config.diffusion_epochs,
            "diffusion_lr": config.diffusion_lr,
            "diffusion_weight_decay": config.diffusion_weight_decay,
            "noise_std": config.noise_std,
            # Model
            "device": config.device,
            "dtype": str(config.dtype),
        }
    )

    print("=" * 60)
    print("TokCom v5.2 Training")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Dtype: {config.dtype}")
    print(f"Tokenizer: {config.tokenizer_path}")
    print(f"Dataset: {config.dataset_name}")
    print(f"Wandb run: {wandb.run.name}")
    print("=" * 60)

    # ========================================================================
    # 1. Create Dataset
    # ========================================================================
    print("\n[Step 1] Creating dataset...")

    train_dataset = WikiTextDataset(
        dataset_name=config.dataset_name,
        chunk_size=config.chunk_size,
        tokenizer_path=config.tokenizer_path,
        split="train",
        max_length=config.max_length
    )

    val_dataset = WikiTextDataset(
        dataset_name=config.dataset_name,
        chunk_size=config.chunk_size,
        tokenizer_path=config.tokenizer_path,
        split="validation",
        max_length=config.max_length
    )

    # Create dataloaders
    collate_fn = make_collate_fn(train_dataset.eos_token_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # ========================================================================
    # 2. Initialize Models
    # ========================================================================
    print("\n[Step 2] Initializing models...")

    # Load tokenizer to get vocab size
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
    tokenizer_vocab_size = tokenizer.vocab_size
    # Add 1 for omni token
    vocab_size = tokenizer_vocab_size + 1
    omni_token_id = tokenizer_vocab_size  # omni_token_id = vocab_size of tokenizer
    print(f"Tokenizer vocabulary size: {tokenizer_vocab_size}")
    print(f"Model vocabulary size (with omni token): {vocab_size}")
    print(f"Omni token ID: {omni_token_id}")

    # Update ModelArgs with correct vocab size and omni token id
    model_args = ModelArgs(vocab_size=vocab_size, omni_token_id=omni_token_id)

    # Initialize VAE model
    vae_model = TokComVAE(model_args).to(config.device)
    print(f"VAE model parameters: {sum(p.numel() for p in vae_model.parameters()):,}")

    # Initialize Denoised model
    denoised_model = DenoisedModel(model_args).to(config.device)
    print(f"Denoised model parameters: {sum(p.numel() for p in denoised_model.parameters()):,}")

    # ========================================================================
    # 3. Stage 1: VAE Training
    # ========================================================================
    print("\n" + "=" * 60)
    print("[Stage 1] VAE Training")
    print("=" * 60)

    vae_optimizer = AdamW(
        vae_model.parameters(),
        lr=config.vae_lr,
        weight_decay=config.vae_weight_decay
    )
    vae_scheduler = CosineAnnealingLR(vae_optimizer, T_max=config.vae_epochs)

    best_vae_val_loss = float('inf')
    vae_global_step = 0

    for epoch in range(1, config.vae_epochs + 1):
        train_loss, vae_global_step = train_vae_epoch(
            vae_model, train_loader, vae_optimizer,
            config.device, config.dtype, epoch, vae_global_step
        )
        vae_scheduler.step()

        # Validation
        val_loss = validate_vae_epoch(
            vae_model, val_loader, config.device, config.dtype
        )

        current_lr = vae_scheduler.get_last_lr()[0]
        print(f"Epoch {epoch}/{config.vae_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - LR: {current_lr:.2e}")

        # Log to wandb
        wandb.log({
            "vae/train_loss": train_loss,
            "vae/val_loss": val_loss,
            "vae/learning_rate": current_lr,
            "vae/epoch": epoch,
        })

        # Save best checkpoint
        if val_loss < best_vae_val_loss:
            best_vae_val_loss = val_loss
            save_best_checkpoint(
                vae_model, denoised_model, model_args, config,
                epoch, val_loss, "vae", config.save_dir
            )
            wandb.log({"vae/best_val_loss": best_vae_val_loss})

        # Sample and log to wandb
        if epoch % config.sample_every == 0:
            sample_and_log_vae(
                vae_model, val_loader, tokenizer,
                config.device, config.dtype, epoch, config.num_samples
            )

    print(f"VAE training completed! Best val loss: {best_vae_val_loss:.4f}")

    # ========================================================================
    # 4. Stage 2: Diffusion Training (Freeze VAE)
    # ========================================================================
    print("\n" + "=" * 60)
    print("[Stage 2] Diffusion Training (VAE Frozen)")
    print("=" * 60)

    # Freeze VAE
    for param in vae_model.parameters():
        param.requires_grad = False
    vae_model.eval()
    print("VAE model frozen.")

    diffusion_optimizer = AdamW(
        denoised_model.parameters(),
        lr=config.diffusion_lr,
        weight_decay=config.diffusion_weight_decay
    )
    diffusion_scheduler = CosineAnnealingLR(diffusion_optimizer, T_max=config.diffusion_epochs)

    best_diffusion_val_loss = float('inf')
    diffusion_global_step = 0

    for epoch in range(1, config.diffusion_epochs + 1):
        train_loss, train_mse, diffusion_global_step = train_diffusion_epoch(
            vae_model, denoised_model, train_loader,
            diffusion_optimizer, config.device, config.dtype,
            epoch, diffusion_global_step, config.noise_std
        )
        diffusion_scheduler.step()

        # Validation
        val_loss, val_mse = validate_diffusion_epoch(
            vae_model, denoised_model, val_loader,
            config.device, config.dtype, config.noise_std
        )

        current_lr = diffusion_scheduler.get_last_lr()[0]
        print(f"Epoch {epoch}/{config.diffusion_epochs} - Train CE: {train_loss:.4f} - Val CE: {val_loss:.4f} - Train MSE: {train_mse:.6f} - Val MSE: {val_mse:.6f} - LR: {current_lr:.2e}")

        # Log to wandb
        wandb.log({
            "diffusion/train_ce_loss": train_loss,
            "diffusion/val_ce_loss": val_loss,
            "diffusion/train_mse_loss": train_mse,
            "diffusion/val_mse_loss": val_mse,
            "diffusion/learning_rate": current_lr,
            "diffusion/epoch": epoch,
        })

        # Save best checkpoint
        if val_loss < best_diffusion_val_loss:
            best_diffusion_val_loss = val_loss
            save_best_checkpoint(
                vae_model, denoised_model, model_args, config,
                epoch, val_loss, "diffusion", config.save_dir
            )
            wandb.log({"diffusion/best_val_loss": best_diffusion_val_loss})

        # Sample and log to wandb
        if epoch % config.sample_every == 0:
            sample_and_log_diffusion(
                vae_model, denoised_model, val_loader, tokenizer,
                model_args, config, epoch, config.num_samples
            )

    print(f"Diffusion training completed! Best val loss: {best_diffusion_val_loss:.4f}")

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best VAE checkpoint: {config.save_dir}/best_vae.pt (val_loss: {best_vae_val_loss:.4f})")
    print(f"Best Diffusion checkpoint: {config.save_dir}/best_diffusion.pt (val_loss: {best_diffusion_val_loss:.4f})")
    print("=" * 60)

    # Log final summary to wandb
    wandb.log({
        "final/best_vae_val_loss": best_vae_val_loss,
        "final/best_diffusion_val_loss": best_diffusion_val_loss,
    })

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
