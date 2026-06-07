# =============================================================================
# model.py — The Modular Transformer Language Model
#
# This file defines the Transformer model in a modular way so you can
# swap in different attention mechanisms and positional encodings easily.
#
# Structure of a Transformer (simplified):
#
#   Input tokens → Token Embeddings + Positional Encoding
#       ↓
#   [Block 1: Attention → Add & Norm → FeedForward → Add & Norm]
#   [Block 2: ...]
#   ...
#   [Block N: ...]
#       ↓
#   Final LayerNorm → Linear head → Logits (probability over vocab)
#       ↓
#   Loss (cross-entropy vs target tokens)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    n_embd, n_head, n_layer, dropout, block_size, device
)
from attention import MultiHeadAttention, SlidingWindowAttention, LinearAttention, MultiQueryAttention
from positional import (
    SinusoidalPositionalEncoding,
    MultiHeadAttentionRoPE,
    ALiBiAttention,
    RelativeAttention
)


# =============================================================================
# FeedForward Network (same in all variants)
# =============================================================================

class FeedForward(nn.Module):
    """
    A simple two-layer MLP (Multi-Layer Perceptron) applied after attention.

    Why do we need this?
    Attention is great at GATHERING information across the sequence,
    but it's essentially just weighted averaging — it's linear.
    The FeedForward network lets each position PROCESS its gathered info
    with non-linear transformations.

    Architecture:
        Linear(n_embd → 4*n_embd) → ReLU → Linear(4*n_embd → n_embd) → Dropout
    The middle layer is 4x larger — this is a common choice from the original paper.
    """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # expand: gives model room to "think"
            nn.ReLU(),                        # non-linearity: makes the model expressive
            nn.Linear(4 * n_embd, n_embd),  # compress back to original size
            nn.Dropout(dropout),              # regularisation
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# Transformer Block (one layer)
# =============================================================================

class Block(nn.Module):
    """
    One Transformer block = Attention + FeedForward, each with residual connections.

    Residual connections (x = x + sublayer(x)) are crucial:
    - They allow gradients to flow directly back through many layers (prevents vanishing gradients)
    - The model can "skip" a block if it doesn't help (learns identity easily)

    LayerNorm normalises each token's embedding to have mean=0, std=1.
    We use "Pre-LN" style: apply LayerNorm BEFORE attention/feedforward.
    This is more stable to train than the original "Post-LN" style.
    """

    def __init__(self, n_embd, n_head, attention_module):
        """
        attention_module: an already-constructed attention object (MultiHeadAttention, etc.)
        This is how we make the block "swappable" — just pass in a different attention.
        """
        super().__init__()
        self.sa   = attention_module          # self-attention (swappable!)
        self.ffwd = FeedForward(n_embd)       # feedforward network (fixed)
        self.ln1  = nn.LayerNorm(n_embd)     # normalise before attention
        self.ln2  = nn.LayerNorm(n_embd)     # normalise before feedforward

    def forward(self, x):
        # Pre-LN attention with residual connection
        # 1. Normalise x
        # 2. Pass through attention
        # 3. Add residual (original x) back
        x = x + self.sa(self.ln1(x))   # "communicate" — tokens share info

        # Pre-LN feedforward with residual connection
        x = x + self.ffwd(self.ln2(x)) # "think" — each token processes its info
        return x


# =============================================================================
# Convolutional Block (Part 4 — Conv + Attention Hybrid)
# =============================================================================

class ConvBlock(nn.Module):
    """
    A 1D Convolutional block placed BEFORE each attention block (Part 4a design).

    Why convolution?
    - Attention is great at long-range dependencies
    - But it's relatively expensive and doesn't directly encode LOCAL structure
    - Convolution is cheap and excellent at local patterns (n-grams, character patterns)
    - Adding Conv before attention lets the model first extract local features,
      then use attention to combine them globally

    This is a depthwise separable convolution for efficiency:
    - Depthwise conv: applies one filter PER channel (cheap)
    - Pointwise (1x1) conv: mixes channels after depthwise (cheap)
    - Together they approximate a full convolution at a fraction of the cost
    """

    def __init__(self, n_embd, kernel_size=3):
        super().__init__()
        # Depthwise conv: groups=n_embd means one filter per channel
        self.depthwise = nn.Conv1d(
            in_channels=n_embd,
            out_channels=n_embd,
            kernel_size=kernel_size,
            padding=kernel_size // 2,  # 'same' padding: output length = input length
            groups=n_embd              # depthwise: each channel filtered independently
        )
        # Pointwise conv: 1x1 convolution mixes across channels
        self.pointwise = nn.Conv1d(n_embd, n_embd, kernel_size=1)
        self.norm      = nn.LayerNorm(n_embd)  # normalise output
        self.act       = nn.GELU()              # GELU is slightly smoother than ReLU

    def forward(self, x):
        # x shape: (B, T, C) — but Conv1d expects (B, C, T), so we transpose
        residual = x                             # save for residual connection
        x_conv   = x.transpose(1, 2)            # (B, C, T)
        x_conv   = self.depthwise(x_conv)       # (B, C, T) — local feature extraction
        x_conv   = self.act(x_conv)             # activation
        x_conv   = self.pointwise(x_conv)       # (B, C, T) — channel mixing
        x_conv   = x_conv.transpose(1, 2)       # back to (B, T, C)
        x_conv   = self.norm(x_conv + residual) # residual + normalise
        return x_conv


class ConvAttentionBlock(nn.Module):
    """
    Part 4(a): Conv1D layer BEFORE each attention block.
    Structure: ConvBlock → Attention → FeedForward
    """

    def __init__(self, n_embd, n_head, attention_module, kernel_size=3):
        super().__init__()
        self.conv = ConvBlock(n_embd, kernel_size)    # local feature extraction
        self.attn = attention_module                  # long-range attention
        self.ffwd = FeedForward(n_embd)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = self.conv(x)                # first: extract local n-gram features
        x = x + self.attn(self.ln1(x)) # then: attend globally
        x = x + self.ffwd(self.ln2(x)) # finally: process
        return x


# =============================================================================
# The Full Transformer Language Model
# =============================================================================

class GPTLanguageModel(nn.Module):
    """
    A modular GPT-style language model.

    Language modelling = predicting the next token given all previous tokens.
    We train by giving the model sequences and asking it to predict one step ahead.

    Parameters:
        vocab_size      : number of unique characters in our dataset
        attention_type  : which attention mechanism to use (see ATTENTION_TYPES)
        positional_type : which positional encoding to use (see POSITIONAL_TYPES)
        use_conv        : if True, add Conv1D before each attention block (Part 4)
        block_size_override : use a different context length than config (for extrapolation test)
    """

    # Mapping from string name → attention class constructor
    # This lets us select attention by name (e.g. "sliding_window") instead of importing manually
    ATTENTION_TYPES = {
        'standard'      : MultiHeadAttention,
        'sliding_window': SlidingWindowAttention,
        'linear'        : LinearAttention,
        'multi_query'   : MultiQueryAttention,
        'rope'          : MultiHeadAttentionRoPE,  # RoPE handles positional encoding internally
        'alibi'         : ALiBiAttention,          # ALiBi handles positional encoding internally
        'relative'      : RelativeAttention,       # Relative PE handles positional internally
    }

    def __init__(self, vocab_size, attention_type='standard', positional_type='sinusoidal',
                 use_conv=False, block_size_override=None):
        super().__init__()

        self.vocab_size      = vocab_size
        self.attention_type  = attention_type
        self.positional_type = positional_type
        self.use_conv        = use_conv
        # Allow overriding block_size for extrapolation experiments
        self.ctx_len         = block_size_override if block_size_override else block_size

        # --- Token Embedding Table ---
        # Converts each integer token (character index) into a dense vector of size n_embd.
        # Think of it as a lookup table: token_id → embedding vector
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # --- Positional Encoding ---
        # Only used for attention types that don't handle position internally
        # (RoPE, ALiBi, Relative PE bake position into attention itself)
        self.positional_encoding = self._build_positional_encoding(positional_type)

        # --- Transformer Blocks ---
        # Stack n_layer blocks, each with the chosen attention type
        self.blocks = nn.Sequential(
            *[self._build_block(use_conv) for _ in range(n_layer)]
        )

        # --- Final LayerNorm ---
        # Applied after all blocks, before the output projection.
        # Stabilises the final representations.
        self.ln_f = nn.LayerNorm(n_embd)

        # --- Language Model Head ---
        # Projects from n_embd → vocab_size to get unnormalised scores (logits)
        # for each possible next token.
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # --- Weight Initialisation ---
        # Normal distribution with small std stabilises early training.
        self.apply(self._init_weights)

    def _build_positional_encoding(self, positional_type):
        """
        Returns a positional encoding module, or None if the attention
        type handles position internally (RoPE, ALiBi, Relative PE).
        """
        # These attention types encode position inside their attention operation
        # so we don't need a separate positional encoding added to embeddings
        no_external_pe = {'rope', 'alibi', 'relative'}

        if self.attention_type in no_external_pe:
            return None  # position handled inside attention

        if positional_type == 'sinusoidal':
            return SinusoidalPositionalEncoding(n_embd, max_len=5000)
        else:
            # Default to sinusoidal if unknown type given
            return SinusoidalPositionalEncoding(n_embd, max_len=5000)

    def _build_block(self, use_conv):
        """
        Build one Transformer block with the chosen attention module.
        """
        head_size = n_embd // n_head  # each head works on a slice of the embedding

        # Instantiate the chosen attention class
        attention_cls    = self.ATTENTION_TYPES[self.attention_type]
        attention_module = attention_cls(n_head, head_size)

        if use_conv:
            # Part 4: wrap with ConvBlock before attention
            return ConvAttentionBlock(n_embd, n_head, attention_module)
        else:
            return Block(n_embd, n_head, attention_module)

    def _init_weights(self, module):
        """
        Initialise linear and embedding weights with small random values.
        This helps the model start training stably — if weights start too large,
        gradients can explode; too small, they can vanish.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Forward pass: given token indices, predict next tokens.

        idx     : (B, T) tensor of token indices
        targets : (B, T) tensor of next token indices (shifted by 1), used for loss
        """
        B, T = idx.shape

        # --- 1. Token Embeddings ---
        # Convert each integer token → embedding vector
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)

        # --- 2. Add Positional Encoding (if applicable) ---
        if self.positional_encoding is not None:
            # Get positional encodings for positions 0..T-1
            pos_emb = self.positional_encoding(T).to(device)  # (T, n_embd)
            # Add to token embeddings (broadcasting over batch dimension B)
            x = tok_emb + pos_emb  # (B, T, n_embd)
        else:
            # Position is encoded inside attention (RoPE/ALiBi/Relative)
            x = tok_emb  # (B, T, n_embd)

        # --- 3. Pass through all Transformer blocks ---
        x = self.blocks(x)  # (B, T, n_embd)

        # --- 4. Final normalisation ---
        x = self.ln_f(x)  # (B, T, n_embd)

        # --- 5. Project to vocabulary size ---
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # --- 6. Compute loss (if targets provided) ---
        if targets is None:
            loss = None
        else:
            # Reshape for cross_entropy: (B*T, vocab_size) vs (B*T,)
            B, T, C = logits.shape
            logits  = logits.view(B * T, C)   # flatten batch and time
            targets = targets.view(B * T)     # flatten batch and time
            # Cross-entropy loss: measures how surprised the model is by the true next token
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Generate text autoregressively:
        Given a starting sequence, predict the next token, append it, repeat.

        idx: (B, T) starting token indices
        max_new_tokens: how many new tokens to generate
        """
        for _ in range(max_new_tokens):
            # Crop context to last ctx_len tokens (model can't handle longer)
            idx_cond = idx[:, -self.ctx_len:]

            # Forward pass — we don't need loss during generation
            logits, _ = self(idx_cond)

            # Only look at the LAST time step's logits (predicting the NEXT token)
            logits = logits[:, -1, :]  # (B, vocab_size)

            # Convert logits to probabilities via softmax
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)

            # Sample the next token from the probability distribution
            # (multinomial = sample one index proportional to probs)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append the new token to the sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx
