# =============================================================================
# config.py — All hyperparameters live here so you only need to change one file
# =============================================================================

import torch

# --------------- Device Setup ------------------------------------------------
# Use GPU (CUDA) if available, otherwise fall back to CPU
# GPU training is much faster; on Colab, go to Runtime > Change runtime type > GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --------------- Model Size --------------------------------------------------
# These control how "big" our Transformer is
n_embd   = 128   # Each token is represented as a vector of this size (embedding dimension)
n_head   = 4     # Number of attention heads (each head looks at the sequence differently)
n_layer  = 2     # Number of stacked Transformer blocks (depth of the model)
dropout  = 0.2   # During training, randomly zero out 20% of connections to prevent memorisation

# --------------- Training Settings -------------------------------------------
batch_size    = 16    # How many sequences we process at once (higher = faster but more memory)
block_size    = 64    # How many tokens the model can "see" at once (context window)
max_iters     = 500   # Total number of training steps
eval_interval = 50    # How often we check train/val loss (every 50 steps)
eval_iters    = 100   # How many batches we average to estimate loss (more = more accurate estimate)
learning_rate = 3e-4  # How big each gradient update step is (3e-4 is a safe default for Adam)

# --------------- Extrapolation Test lengths ----------------------------------
# Part 3(c): we train on L_train=512 tokens and test on longer sequences
# to see if the positional encoding generalises beyond training length
L_train = 512   # Context length used during extrapolation training
L_test  = 1024  # Longer context to test extrapolation (can also try 2048)
