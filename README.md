# Mini-Gpt

Mini-Gpt is an educational, character-level GPT-style language model built in
PyTorch. It is designed to show the full training and generation pipeline in a
simple, readable way.

## What This Project Does

- Loads text data (`input.txt`) and learns patterns in character sequences.
- Trains an autoregressive transformer model to predict the next character.
- Generates new text by sampling one token at a time from model probabilities.
- Moves from a simple bigram baseline to a multi-layer transformer model.

## Project Components

### 1) Data Pipeline

- Reads Tiny Shakespeare text from `input.txt`.
- Builds a character vocabulary from unique characters in the dataset.
- Creates:
  - `stoi` (string-to-index mapping)
  - `itos` (index-to-string mapping)
  - `encode`/`decode` functions for tokenization and reconstruction
- Converts encoded text into a PyTorch tensor.
- Splits dataset into training and validation sets.
- Uses random mini-batches via `get_batch(split)` with:
  - `x`: current context tokens
  - `y`: next-token targets (shifted by one position)

### 2) Baseline Model (Bigram)

- Starts with a bigram language model as a minimal baseline.
- Uses an embedding table to directly map each token to next-token logits.
- Trains with cross-entropy loss.
- Samples text autoregressively to validate that the pipeline works end-to-end.

### 3) Transformer Model

- Implements a decoder-only GPT-style architecture:
  - Token embeddings
  - Positional embeddings
  - Stacked transformer blocks
  - Final layer norm
  - Linear language modeling head

#### Core submodules

- `Head`: single self-attention head (key/query/value projections)
- `MultiHeadAttention`: parallel attention heads + output projection
- `FeedFoward`: position-wise MLP (`Linear -> ReLU -> Linear`)
- `Block`: attention + feed-forward with residual connections and layer norm

### 4) Training & Evaluation

- Optimizer: `AdamW`
- Loss: next-token cross-entropy
- Periodic evaluation on both train/validation (`estimate_loss`)
- Uses context cropping to `block_size` during generation
- Generates long-form text from a single start token

## Techniques Used

- Character-level tokenization
- Autoregressive next-token prediction
- Causal masking using lower-triangular attention mask
- Scaled dot-product self-attention
- Multi-head attention
- Positional embeddings
- Residual connections
- Layer normalization
- Feed-forward expansion (`4 * n_embd`)
- Dropout (configurable)
- AdamW optimization
- Mini-batch stochastic training

## Related Keywords

These are relevant concepts around this project and possible next-step
extensions:

- LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- 4-bit quantization (QLoRA-style workflows) for memory-efficient training/inference
- Mixed precision training
- Gradient checkpointing
- Tokenization with BPE/SentencePiece
- Instruction fine-tuning

## Main Hyperparameters (Notebook)

- `batch_size = 16`
- `block_size = 32`
- `max_iters = 5000`
- `eval_interval = 100`
- `learning_rate = 1e-3`
- `n_embd = 64`
- `n_head = 4`
- `n_layer = 4`
- `dropout = 0.0`

## Output

After training, the model can generate Shakespeare-like character text by
sampling from predicted probability distributions at each time step.

---
Credit: This project was learned from Andrej Karpathy's YouTube content.
