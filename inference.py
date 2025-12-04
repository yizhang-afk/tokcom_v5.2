import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import argparse
from typing import Optional, List

from model import TokComVAE, DenoisedModel, ModelArgs, sample_noise


def load_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        vae_model, denoised_model, model_args, config
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model args
    model_args = ModelArgs(**checkpoint['model_args'])
    config = checkpoint['config']

    print(f"Model config:")
    print(f"  vocab_size: {model_args.vocab_size}")
    print(f"  hidden_size: {model_args.hidden_size}")
    print(f"  num_hidden_layers: {model_args.num_hidden_layers}")
    print(f"  chunk_size: {config['chunk_size']}")
    print(f"  noise_std: {config['noise_std']}")

    # Initialize models
    vae_model = TokComVAE(model_args).to(device)
    denoised_model = DenoisedModel(model_args).to(device)

    # Load weights
    vae_model.load_state_dict(checkpoint['vae_model_state_dict'])
    denoised_model.load_state_dict(checkpoint['denoised_model_state_dict'])

    # Set to eval mode
    vae_model.eval()
    denoised_model.eval()

    print("Models loaded successfully!")
    return vae_model, denoised_model, model_args, config


@torch.no_grad()
def encode(
    vae_model: TokComVAE,
    input_ids: torch.Tensor,
    chunk_size: int
) -> torch.Tensor:
    """
    Encode input tokens to latent vectors using VAE encoder.

    Args:
        vae_model: VAE model
        input_ids: Input token ids of shape (batch_size, seq_len)
        chunk_size: Size of each chunk

    Returns:
        Latent vectors of shape (batch_size, num_chunks, hidden_size)
    """
    batch_size, seq_len = input_ids.shape
    num_chunks = seq_len // chunk_size

    # Reshape to chunks: (batch_size, num_chunks, chunk_size)
    input_ids_chunked = input_ids.view(batch_size, num_chunks, chunk_size)

    # Flatten for encoder: (batch_size * num_chunks, chunk_size)
    input_ids_flat = input_ids_chunked.view(-1, chunk_size)

    # Encode
    latent_vectors = vae_model.encoder(input_ids_flat)  # (batch_size * num_chunks, hidden_size)

    # Reshape back: (batch_size, num_chunks, hidden_size)
    latent_vectors = latent_vectors.view(batch_size, num_chunks, -1)

    return latent_vectors


@torch.no_grad()
def decode(
    vae_model: TokComVAE,
    latent_vectors: torch.Tensor,
    chunk_size: int
) -> torch.Tensor:
    """
    Decode latent vectors to token ids using VAE decoder.

    Args:
        vae_model: VAE model
        latent_vectors: Latent vectors of shape (batch_size, num_chunks, hidden_size)
        chunk_size: Size of each chunk

    Returns:
        Decoded token ids of shape (batch_size, seq_len)
    """
    batch_size, num_chunks, hidden_size = latent_vectors.shape

    # Flatten: (batch_size * num_chunks, hidden_size)
    latent_flat = latent_vectors.view(-1, hidden_size)

    # Decode to logits
    logits = vae_model.decoder(latent_flat, decode_len=chunk_size, return_logits=True)
    # logits: (batch_size * num_chunks, chunk_size, vocab_size)

    # Get token ids by argmax
    token_ids = logits.argmax(dim=-1)  # (batch_size * num_chunks, chunk_size)

    # Reshape: (batch_size, num_chunks, chunk_size) -> (batch_size, seq_len)
    token_ids = token_ids.view(batch_size, num_chunks, chunk_size)
    token_ids = token_ids.view(batch_size, -1)

    return token_ids


@torch.no_grad()
def denoise(
    denoised_model: DenoisedModel,
    x_noisy: torch.Tensor,
    num_steps: int = 50,
    noise_std: float = 0.05
) -> torch.Tensor:
    """
    Denoise latent vectors using the diffusion model with Euler sampling.

    Flow matching: x_t = (1-t)*x0 + t*x1
    We start from t=0 (pure noise x0) and integrate to t=1 (clean data x1)

    Args:
        denoised_model: Denoising model
        x_noisy: Initial noisy latent vectors (x0) of shape (batch_size, num_chunks, hidden_size)
        num_steps: Number of denoising steps
        noise_std: Standard deviation used for initial noise

    Returns:
        Denoised latent vectors (x1) of shape (batch_size, num_chunks, hidden_size)
    """
    batch_size = x_noisy.shape[0]
    device = x_noisy.device

    # Start from x0 (noise)
    x_t = x_noisy

    # Euler integration from t=0 to t=1
    dt = 1.0 / num_steps

    for step in range(num_steps):
        t = step * dt
        t_tensor = torch.full((batch_size,), t, device=device)

        # Predict x1 from current x_t
        x1_pred = denoised_model(x_t, t_tensor)

        # Compute velocity: v = x1 - x0
        # For flow matching: dx/dt = x1 - x0
        # So x_{t+dt} = x_t + (x1_pred - x0) * dt
        # But we don't have x0 explicitly, we use: v = (x1_pred - x_t) / (1 - t) when t < 1
        if t < 1.0 - 1e-6:
            velocity = (x1_pred - x_t) / (1.0 - t)
            x_t = x_t + velocity * dt
        else:
            x_t = x1_pred

    return x_t


@torch.no_grad()
def generate(
    vae_model: TokComVAE,
    denoised_model: DenoisedModel,
    num_chunks: int,
    chunk_size: int,
    hidden_size: int,
    noise_std: float = 0.05,
    num_steps: int = 50,
    batch_size: int = 1,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Generate new sequences from random noise.

    Args:
        vae_model: VAE model
        denoised_model: Denoising model
        num_chunks: Number of chunks to generate
        chunk_size: Size of each chunk
        hidden_size: Hidden size of latent vectors
        noise_std: Standard deviation for initial noise
        num_steps: Number of denoising steps
        batch_size: Number of sequences to generate
        device: Device

    Returns:
        Generated token ids of shape (batch_size, seq_len)
    """
    # Sample initial noise x0
    x0 = sample_noise((batch_size, num_chunks, hidden_size), device=device, std=noise_std)

    # Denoise to get x1
    x1 = denoise(denoised_model, x0, num_steps=num_steps, noise_std=noise_std)

    # Decode to tokens
    token_ids = decode(vae_model, x1, chunk_size)

    return token_ids


@torch.no_grad()
def reconstruct(
    vae_model: TokComVAE,
    input_ids: torch.Tensor,
    chunk_size: int
) -> torch.Tensor:
    """
    Reconstruct input through VAE (encode then decode).

    Args:
        vae_model: VAE model
        input_ids: Input token ids of shape (batch_size, seq_len)
        chunk_size: Size of each chunk

    Returns:
        Reconstructed token ids of shape (batch_size, seq_len)
    """
    # Encode
    latent_vectors = encode(vae_model, input_ids, chunk_size)

    # Decode
    reconstructed_ids = decode(vae_model, latent_vectors, chunk_size)

    return reconstructed_ids


def main():
    parser = argparse.ArgumentParser(description="TokCom v5.2 Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--mode", type=str, choices=["generate", "reconstruct", "encode"], default="generate",
                        help="Inference mode: generate (from noise), reconstruct (encode-decode), encode (get latents)")
    parser.add_argument("--input_text", type=str, default=None, help="Input text for reconstruct/encode mode")
    parser.add_argument("--num_chunks", type=int, default=256, help="Number of chunks for generation (default: 256 = 1024 tokens)")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--output_file", type=str, default=None, help="Output file path (optional)")

    args = parser.parse_args()

    # Load models
    vae_model, denoised_model, model_args, config = load_checkpoint(args.checkpoint, args.device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'])

    chunk_size = config['chunk_size']
    noise_std = config['noise_std']
    max_length = config['max_length']

    print(f"\nInference mode: {args.mode}")
    print("=" * 60)

    if args.mode == "generate":
        # Generate from noise
        print(f"Generating {args.batch_size} sequence(s) with {args.num_chunks} chunks ({args.num_chunks * chunk_size} tokens)...")
        print(f"Using {args.num_steps} denoising steps")

        token_ids = generate(
            vae_model, denoised_model,
            num_chunks=args.num_chunks,
            chunk_size=chunk_size,
            hidden_size=model_args.hidden_size,
            noise_std=noise_std,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            device=args.device
        )

        # Decode to text
        print("\nGenerated sequences:")
        print("-" * 60)
        for i in range(args.batch_size):
            text = tokenizer.decode(token_ids[i], skip_special_tokens=True)
            print(f"\n[Sequence {i+1}]")
            print(text[:500] + "..." if len(text) > 500 else text)

        if args.output_file:
            with open(args.output_file, 'w') as f:
                for i in range(args.batch_size):
                    text = tokenizer.decode(token_ids[i], skip_special_tokens=True)
                    f.write(f"[Sequence {i+1}]\n{text}\n\n")
            print(f"\nOutput saved to {args.output_file}")

    elif args.mode == "reconstruct":
        if args.input_text is None:
            print("Error: --input_text is required for reconstruct mode")
            return

        print(f"Input text: {args.input_text[:100]}..." if len(args.input_text) > 100 else f"Input text: {args.input_text}")

        # Tokenize input
        input_ids = tokenizer.encode(args.input_text, add_special_tokens=False, return_tensors="pt")
        input_ids = input_ids.to(args.device)

        # Pad or truncate to max_length
        if input_ids.shape[1] < max_length:
            padding = torch.full((1, max_length - input_ids.shape[1]), tokenizer.eos_token_id, device=args.device)
            input_ids = torch.cat([input_ids, padding], dim=1)
        else:
            input_ids = input_ids[:, :max_length]

        # Reconstruct
        reconstructed_ids = reconstruct(vae_model, input_ids, chunk_size)

        # Decode
        original_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        reconstructed_text = tokenizer.decode(reconstructed_ids[0], skip_special_tokens=True)

        print("\nOriginal:")
        print("-" * 60)
        print(original_text[:500] + "..." if len(original_text) > 500 else original_text)

        print("\nReconstructed:")
        print("-" * 60)
        print(reconstructed_text[:500] + "..." if len(reconstructed_text) > 500 else reconstructed_text)

        # Calculate accuracy
        match = (input_ids[0] == reconstructed_ids[0]).float().mean().item()
        print(f"\nToken accuracy: {match * 100:.2f}%")

    elif args.mode == "encode":
        if args.input_text is None:
            print("Error: --input_text is required for encode mode")
            return

        print(f"Input text: {args.input_text[:100]}..." if len(args.input_text) > 100 else f"Input text: {args.input_text}")

        # Tokenize input
        input_ids = tokenizer.encode(args.input_text, add_special_tokens=False, return_tensors="pt")
        input_ids = input_ids.to(args.device)

        # Pad or truncate to max_length
        if input_ids.shape[1] < max_length:
            padding = torch.full((1, max_length - input_ids.shape[1]), tokenizer.eos_token_id, device=args.device)
            input_ids = torch.cat([input_ids, padding], dim=1)
        else:
            input_ids = input_ids[:, :max_length]

        # Encode
        latent_vectors = encode(vae_model, input_ids, chunk_size)

        print(f"\nLatent vectors shape: {latent_vectors.shape}")
        print(f"Latent vectors mean: {latent_vectors.mean().item():.6f}")
        print(f"Latent vectors std: {latent_vectors.std().item():.6f}")
        print(f"Latent vectors min: {latent_vectors.min().item():.6f}")
        print(f"Latent vectors max: {latent_vectors.max().item():.6f}")

        if args.output_file:
            torch.save(latent_vectors, args.output_file)
            print(f"\nLatent vectors saved to {args.output_file}")


if __name__ == "__main__":
    main()
