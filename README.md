# pytorch-transformer-from-scratch

Implementation of the Transformer architecture ("Attention Is All You Need", Vaswani et al., 2017) built entirely from raw PyTorch.

Two variants are built from a shared core:
- **Encoder-only** (BERT-style, bidirectional attention) — trained for text classification
- **Decoder-only** (GPT-style, causal attention) — trained for autoregressive text generation
