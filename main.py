import math
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from einops import rearrange, repeat
from typing import Optional, Tuple, List

class PreRMSNorm(nn.Module):
    """Pre-RMSNorm as specified in the paper for improved training."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Compute RMS along last dimension
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight

class SwiGLU(nn.Module):
    """SwiGLU activation as used in LLaMA and specified in the paper."""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim) 
        self.w3 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        gate = F.silu(self.w1(x))
        return self.w3(gate * self.w2(x))

class CausalTransformerLayer(nn.Module):
    """Causal Transformer layer with pre-RMSNorm and SwiGLU."""
    def __init__(self, 
                 dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.pre_norm1 = PreRMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.pre_norm2 = PreRMSNorm(dim)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Pre-norm for attention
        normed_x = self.pre_norm1(x)
        x = x + self.attn(normed_x, normed_x, normed_x, attn_mask=mask)[0]
        
        # Pre-norm for MLP
        normed_x = self.pre_norm2(x)
        x = x + self.mlp(normed_x)
        return x

class SigmaVAE(nn.Module):
    """σ-VAE with fixed variance as described in Section 2.3."""
    def __init__(self,
                 input_dim: int,
                 latent_dim: int, 
                 hidden_dim: int,
                 c_sigma: float = 1.0,
                 beta: float = 1.0):
        super().__init__()
        self.c_sigma = c_sigma
        self.beta = beta
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)  # μ only, no σ
        )
        
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode input to get μ
        mu = self.encoder(x)
        
        # Sample σ from N(0, c_σ) for each example
        sigma = torch.randn_like(mu[..., :1]) * math.sqrt(self.c_sigma)
        sigma = sigma.expand_as(mu)
        
        # Sample z using reparameterization trick
        epsilon = torch.randn_like(mu)
        z = mu + sigma * epsilon
        
        return z, mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu = self.encode(x)
        x_recon = self.decode(z)
        
        # Loss components: reconstruction + KL divergence (β-VAE formulation)
        recon_loss = F.mse_loss(x_recon, x, reduction='none').sum(dim=-1).mean()
        kl_loss = self.beta * (mu.pow(2).sum(dim=-1).mean())
        
        return x_recon, recon_loss, kl_loss

class NextTokenDiffusion(nn.Module):
    """Next-token diffusion head as described in Section 2.1."""
    def __init__(self,
                 dim: int,
                 hidden_dim: int,
                 num_steps: int = 1000,
                 min_beta: float = 1e-4,
                 max_beta: float = 0.02):
        super().__init__()
        
        # Noise predictor network
        self.predictor = nn.Sequential(
            PreRMSNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        )
        
        # Setup noise schedule
        self.num_steps = num_steps
        betas = torch.linspace(min_beta, max_beta, num_steps)
        alphas = 1 - betas
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(torch.cumprod(alphas, dim=0)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - torch.cumprod(alphas, dim=0)))

    def q_sample(self, x: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward diffusion process."""
        if noise is None:
            noise = torch.randn_like(x)
            
        sqrt_alphas = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        return sqrt_alphas * x + sqrt_one_minus_alphas * noise

    def predict_noise(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Predict noise given noisy input and conditioning."""
        # Combine input with conditioning
        x_cond = torch.cat([x, cond], dim=-1)
        return self.predictor(x_cond)

    def loss(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Training loss for denoising diffusion."""
        b = x.shape[0]
        t = torch.randint(0, self.num_steps, (b,), device=x.device)
        noise = torch.randn_like(x)
        x_noisy = self.q_sample(x, t, noise)
        noise_pred = self.predict_noise(x_noisy, t, cond)
        return F.mse_loss(noise_pred, noise)

class LatentLM(nn.Module):
    """Complete LatentLM model as described in the paper."""
    def __init__(self,
                 vocab_size: int,
                 hidden_dim: int,
                 num_layers: int,
                 num_heads: int,
                 max_seq_len: int,
                 continuous_dim: int,
                 latent_dim: int,
                 vae_hidden_dim: int,
                 diffusion_steps: int = 1000):
        super().__init__()
        
        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        
        # σ-VAE for continuous data
        self.vae = SigmaVAE(continuous_dim, latent_dim, vae_hidden_dim)
        
        # Causal Transformer layers
        self.layers = nn.ModuleList([
            CausalTransformerLayer(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Output heads
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        self.diffusion = NextTokenDiffusion(latent_dim, hidden_dim, diffusion_steps)
        
        self.final_norm = PreRMSNorm(hidden_dim)

    def forward(self,
               tokens: torch.Tensor,
               continuous_data: Optional[torch.Tensor] = None,
               diffusion_cond: Optional[torch.Tensor] = None) -> dict:
        """
        Forward pass handling both discrete and continuous data.
        Returns dict with relevant outputs and losses.
        """
        # Initial embeddings
        x = self.token_emb(tokens) + self.pos_emb[:, :tokens.shape[1]]
        
        # Create causal mask
        mask = torch.triu(torch.ones(tokens.shape[1], tokens.shape[1], dtype=torch.bool, device=tokens.device), 1)
        
        # Process through Transformer layers
        for layer in self.layers:
            x = layer(x, mask=mask)
        
        x = self.final_norm(x)
        outputs = {'hidden_states': x}
        
        # Language modeling head
        lm_logits = self.lm_head(x)
        outputs['lm_logits'] = lm_logits
        
        # Process continuous data if provided
        if continuous_data is not None:
            # VAE encoding/decoding
            z, mu = self.vae.encode(continuous_data)
            x_recon = self.vae.decode(z)
            
            # VAE losses
            recon_loss = F.mse_loss(x_recon, continuous_data)
            kl_loss = 0.5 * torch.sum(mu.pow(2) + torch.log(self.vae.c_sigma) - 1)
            outputs['vae_loss'] = recon_loss + self.vae.beta * kl_loss
            
            # Diffusion loss if conditioning is provided
            if diffusion_cond is not None:
                diff_loss = self.diffusion.loss(z, diffusion_cond)
                outputs['diffusion_loss'] = diff_loss
        
        return outputs

    @torch.no_grad()
    def generate(self,
                prompt_tokens: torch.Tensor,
                max_new_tokens: int,
                temperature: float = 1.0,
                continuous_positions: Optional[List[int]] = None) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Autoregressive generation of both discrete tokens and continuous data.
        Returns generated tokens and continuous data (if any).
        """
        tokens = prompt_tokens.clone()
        continuous_outputs = [] if continuous_positions else None
        
        for i in range(max_new_tokens):
            # Get predictions
            outputs = self(tokens[:, -1024:])  # Limit context for efficiency
            logits = outputs['hidden_states'][:, -1]
            
            # Sample next token
            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            tokens = torch.cat([tokens, next_token], dim=1)
            
            # Generate continuous data at specified positions
            if continuous_positions and i in continuous_positions:
                # Use diffusion to generate latent vector
                z = torch.randn(1, self.vae.latent_dim, device=tokens.device)
                cond = outputs['hidden_states'][:, -1]
                
                # Denoise through diffusion steps
                for t in reversed(range(self.diffusion.num_steps)):
                    z = self.diffusion.denoise_step(z, t, cond)
                
                # Decode to continuous data
                continuous_data = self.vae.decode(z)
                continuous_outputs.append(continuous_data)
        
        return tokens, continuous_outputs

def create_latent_lm(config: dict) -> LatentLM:
    """Helper function to create LatentLM model from config."""
    return LatentLM(
        vocab_size=config['vocab_size'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        max_seq_len=config['max_seq_len'],
        continuous_dim=config['continuous_dim'],
        latent_dim=config['latent_dim'],
        vae_hidden_dim=config['vae_hidden_dim']
    )