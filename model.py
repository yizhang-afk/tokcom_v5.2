import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

@dataclass
class ModelArgs:
    hidden_size: int = 896
    num_hidden_layers: int = 24  # Qwen2.5-0.5B architecture (matching config.py)
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    intermediate_size: int = 4864
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    vocab_size: int = 151936  # Total vocabulary size
    omni_token_id: int = 151665 # Repurposed ID for Omni token


# ============================================================================
# Diffusion Model Components
# ============================================================================

class TimestepEmbedding(nn.Module):
    """
    Embeds timestep t into a continuous representation using sinusoidal encoding.
    """
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor):
        """
        Args:
            t: Timestep tensor of shape (bsz,) with values in [0, 1]
        Returns:
            Timestep embeddings of shape (bsz, dim)
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, dtype=torch.float32, device=t.device) / half_dim
        )
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        # If dim is odd, pad with zeros
        if self.dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        return embedding


class AdaLNZero(nn.Module):
    """
    Adaptive Layer Normalization with Zero initialization (adaLN-Zero).
    Modulates the layer norm output based on timestep embedding.
    """
    def __init__(self, hidden_size: int, cond_dim: int):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=1e-6)

        # MLP to generate scale and shift from timestep
        self.scale_shift_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * hidden_size)
        )

        # Initialize scale to 0 and shift to 0 (zero initialization)
        nn.init.zeros_(self.scale_shift_mlp[-1].weight)
        nn.init.zeros_(self.scale_shift_mlp[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
        Args:
            x: Input tensor of shape (bsz, seq_len, hidden_size)
            cond: Conditioning tensor (timestep embedding) of shape (bsz, cond_dim)
        Returns:
            Modulated tensor of shape (bsz, seq_len, hidden_size)
        """
        # Normalize
        x_norm = self.norm(x)

        # Generate scale and shift
        scale_shift = self.scale_shift_mlp(cond)  # (bsz, 2 * hidden_size)
        scale, shift = scale_shift.chunk(2, dim=-1)  # Each: (bsz, hidden_size)

        # Apply modulation
        # scale starts at 0, so (1 + scale) starts at 1
        # shift starts at 0
        return x_norm * (1 + scale[:, None, :]) + shift[:, None, :]


class TransformerBlockWithAdaLN(nn.Module):
    """
    Transformer block with adaLN-Zero conditioning on timestep.
    """
    def __init__(self, args: ModelArgs, cond_dim: int):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)

        # Replace standard layer norms with adaLN
        self.adaln_attn = AdaLNZero(args.hidden_size, cond_dim)
        self.adaln_mlp = AdaLNZero(args.hidden_size, cond_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
        Args:
            x: Input tensor of shape (bsz, seq_len, hidden_size)
            cond: Timestep embedding of shape (bsz, cond_dim)
        Returns:
            Output tensor of shape (bsz, seq_len, hidden_size)
        """
        # Self-attention with adaLN
        h = x + self.self_attn(self.adaln_attn(x, cond))

        # MLP with adaLN
        out = h + self.mlp(self.adaln_mlp(h, cond))

        return out


# ============================================================================
# VAE Components
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class RoPE(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0, max_seq_len: int = 4096):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        t = torch.arange(max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, q, k):
        seq_len = q.shape[2]
        cos = self.cos_cached[:, :, :seq_len, ...]
        sin = self.sin_cached[:, :, :seq_len, ...]
        
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)
            
        q_out = (q * cos) + (rotate_half(q) * sin)
        k_out = (k * cos) + (rotate_half(k) * sin)
        return q_out, k_out

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.num_kv_heads = args.num_key_value_heads
        self.head_dim = args.hidden_size // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(args.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(args.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(args.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, args.hidden_size, bias=False)
        
        self.rope = RoPE(self.head_dim, theta=args.rope_theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = self.rope(q, k)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Bidirectional attention - all tokens can attend to all tokens
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn_output)

class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def forward(self, x):
        h = x + self.self_attn(self.input_layernorm(x))
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out

class Encoder(nn.Module):
    def __init__(self, args: ModelArgs, embed_tokens: nn.Module):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.num_hidden_layers)])
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor):
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)

        # Use only the last position as output vector
        output_vector = h[:, -1, :]  # Shape: (bsz, hidden_size)

        return output_vector

class Decoder(nn.Module):
    def __init__(self, args: ModelArgs, embed_tokens: nn.Module):
        super().__init__()
        self.args = args
        self.embed_tokens = embed_tokens
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.num_hidden_layers)])
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def forward(self, latent_vector: torch.Tensor, decode_len: int, return_logits: bool = True):
        # 1. Use the latent vector directly (no sampling)
        z = latent_vector.unsqueeze(1)  # Shape: (bsz, 1, hidden_size)

        # 2. Prepare the decoder input sequence
        bsz = z.shape[0]
        # Get the embedding for the Omni token
        omni_token_tensor = torch.full((bsz, 1), self.args.omni_token_id, dtype=torch.long, device=z.device)
        omni_embedding = self.embed_tokens(omni_token_tensor)  # Shape: (bsz, 1, hidden_size)

        # Repeat Omni embedding for the rest of the sequence
        remaining_len = decode_len - 1
        if remaining_len > 0:
            omni_seq = omni_embedding.repeat(1, remaining_len, 1)
            # Concatenate the latent vector z with the Omni sequence
            decoder_input_embeds = torch.cat([z, omni_seq], dim=1)
        else:
            decoder_input_embeds = z

        # 3. Pass through decoder transformer
        h = decoder_input_embeds
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)

        # 4. Apply lm_head to get logits for reconstruction
        if return_logits:
            logits = self.lm_head(h)  # Shape: (bsz, decode_len, vocab_size)
            return logits
        else:
            return h

class TokComVAE(nn.Module):
    def __init__(self, args: ModelArgs = ModelArgs()):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.encoder = Encoder(args, self.embed_tokens)
        self.decoder = Decoder(args, self.embed_tokens)

        # Tie the weights of the lm_head to the embedding layer
        self.decoder.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, decode_len: int = None, return_logits: bool = True):
        """
        Forward pass through the model.

        Args:
            input_ids: Input token IDs, shape (bsz, seq_len)
            decode_len: Length of decoded sequence (if None, uses input_ids.shape[1])
            return_logits: Whether to return logits (True) or hidden states (False)

        Returns:
            If return_logits=True: logits of shape (bsz, decode_len, vocab_size)
            If return_logits=False: hidden states of shape (bsz, decode_len, hidden_size)
            Also returns latent_vector for loss computation
        """
        if decode_len is None:
            decode_len = input_ids.shape[1]

        latent_vector = self.encoder(input_ids)
        outputs = self.decoder(latent_vector, decode_len, return_logits=return_logits)
        return outputs, latent_vector



# ============================================================================
# Diffusion Model
# ============================================================================

class DenoisedModel(nn.Module):
    """
    Denoising model for latent space diffusion.
    - 24 Transformer layers with adaLN-Zero conditioning
    - Bidirectional attention
    - Variable length input/output
    - Predicts clean data x1 from noisy input
    """
    def __init__(self, args: ModelArgs = ModelArgs()):
        super().__init__()
        self.args = args

        # Timestep embedding
        self.timestep_embed_dim = args.hidden_size
        self.timestep_embedding = TimestepEmbedding(self.timestep_embed_dim)

        # Timestep MLP to expand embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(self.timestep_embed_dim, args.hidden_size * 4),
            nn.SiLU(),
            nn.Linear(args.hidden_size * 4, args.hidden_size)
        )

        # Transformer layers with adaLN-Zero
        self.layers = nn.ModuleList([
            TransformerBlockWithAdaLN(args, args.hidden_size)
            for _ in range(args.num_hidden_layers)
        ])

        # Final layer norm (no conditioning)
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        # Output projection (predicts the clean data x1)
        self.output_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=True)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Forward pass through the denoising model.

        Args:
            x: Noisy latent vectors of shape (bsz, seq_len, hidden_size)
            t: Timesteps of shape (bsz,) with values in [0, 1]

        Returns:
            Predicted clean data x1 of shape (bsz, seq_len, hidden_size)
        """
        # Embed timestep
        t_embed = self.timestep_embedding(t)  # (bsz, timestep_embed_dim)
        t_cond = self.time_mlp(t_embed)  # (bsz, hidden_size)

        # Pass through transformer layers with adaLN conditioning
        h = x
        for layer in self.layers:
            h = layer(h, t_cond)

        # Final normalization
        h = self.norm(h)

        # Output projection - predict clean data x1
        output = self.output_proj(h)

        return output


# ============================================================================
# Diffusion Utilities
# ============================================================================

def sample_timesteps(batch_size: int, device: torch.device,
                     t_min: float = 1e-4, t_max: float = 1.0 - 1e-4) -> torch.Tensor:
    """
    Sample timesteps from uniform distribution [t_min, t_max].

    TokCom v5: 从 0 到 1 均匀采样，两端有非常小的截断避免数值问题。

    Args:
        batch_size: Number of timesteps to sample
        device: Device to create tensor on
        t_min: Minimum timestep value (default 1e-4, small truncation at 0)
        t_max: Maximum timestep value (default 1-1e-4, small truncation at 1)

    Returns:
        Timesteps of shape (batch_size,) with values in [t_min, t_max]
    """
    # Uniform distribution [t_min, t_max]
    t = torch.rand(batch_size, device=device) * (t_max - t_min) + t_min

    return t


def sample_noise(shape: tuple, device: torch.device, std: float = 0.05) -> torch.Tensor:
    """
    Sample noise from normal distribution with mean=0 and specified std.

    Args:
        shape: Shape of noise tensor
        device: Device to create tensor on
        std: Standard deviation of noise

    Returns:
        Noise tensor of specified shape
    """
    return torch.randn(shape, device=device) * std



