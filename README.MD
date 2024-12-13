# LatentLM

A unified framework for multimodal language modeling that handles both discrete (text, code) and continuous data (images, audio, video) using causal Transformers.

## Features

- Unified processing with causal Transformers
- σ-VAE tokenizer for continuous data
- Next-token diffusion for latent vector generation
- Supports text, images, and speech

## Quick Start

```bash
pip install torch einops
```

## Basic Usage

```python
# Create model
config = {
    'vocab_size': 50257,
    'hidden_dim': 1024,
    'num_layers': 24,
    'num_heads': 16,
    'max_seq_len': 4096,
    'continuous_dim': 768,
    'latent_dim': 256,
    'vae_hidden_dim': 1024
}

model = create_latent_lm(config)

# Training
outputs = model(tokens, continuous_data)
loss = outputs['lm_loss'] + outputs['vae_loss'] + outputs['diffusion_loss']
loss.backward()

# Generation
tokens, continuous_outputs = model.generate(
    prompt_tokens=prompt,
    max_new_tokens=100,
    temperature=0.8
)
```

## Performance

- Image Generation: FID 2.24, IS 253.8
- Text-to-Speech: 10× fewer decoding steps than VALL-E 2
- Competitive with specialized models

## Citation

```bibtex
@article{sun2024multimodal,
  title={Multimodal Latent Language Modeling with Next-Token Diffusion},
  author={Sun, Yutao and et al.},
  journal={arXiv preprint arXiv:2412.08635},
  year={2024}
}
