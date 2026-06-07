# =============================================================================
# positional.py — Positional Encodings
#
# Transformers process all tokens in parallel (unlike RNNs which go step by step).
# This means the model has NO idea about word ORDER unless we inject position info.
# Positional encodings solve this by adding a position-dependent signal to embeddings.
#
# This file implements:
#   1. Sinusoidal PE  — original "Attention Is All You Need" (baseline)
#   2. RoPE           — Rotary Position Embedding (Part 3 variant)
#   3. ALiBi          — Attention with Linear Biases (Part 3 variant)
#   4. Relative PE    — Relative Positional Encoding (Part 3 variant)
# =============================================================================

import torch
import torch.nn as nn
import math
from config import n_embd, block_size


# =============================================================================
# PART 1 — Sinusoidal Positional Encoding (Baseline)
# =============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """
    The original fixed (non-learnable) positional encoding from "Attention Is All You Need".

    Idea: encode position p using sin/cos waves of different frequencies.
    Different dimensions of the embedding get different frequencies:
        PE[pos, 2i]   = sin(pos / 10000^(2i/d_model))
        PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))

    Why sin/cos? These are smooth, bounded functions that the model can learn to decode.
    The pattern is unique for every position and generalises to positions not seen in training
    (at least somewhat — this is what Part 3c tests).
    """

    def __init__(self, n_embd, max_len=5000):
        super().__init__()

        # Create a table of shape (max_len, n_embd) — one row per position
        pe = torch.zeros(max_len, n_embd)

        # position = column vector [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)

        # div_term handles the frequency scaling: 10000^(2i/d_model)
        # We use exp(log(...)) for numerical stability
        div_term = torch.exp(
            torch.arange(0, n_embd, 2).float() * (-math.log(10000.0) / n_embd)
        )  # shape: (n_embd/2,)

        # Even indices get sin, odd indices get cos
        pe[:, 0::2] = torch.sin(position * div_term)  # 0, 2, 4, ...
        pe[:, 1::2] = torch.cos(position * div_term)  # 1, 3, 5, ...

        # register_buffer: saved as part of the model but NOT trained (no gradient)
        self.register_buffer('pe', pe)

    def forward(self, T):
        # Return positional encodings for the first T positions
        # Shape: (T, n_embd)
        return self.pe[:T]


# =============================================================================
# PART 3 — RoPE: Rotary Position Embedding
# =============================================================================

class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Position Embedding (RoPE) — from "RoFormer" (Su et al., 2021).

    Key insight: instead of ADDING position info to token embeddings (like sinusoidal PE),
    RoPE ROTATES the Query and Key vectors in 2D subspaces based on position.

    Why rotation? When you compute Q·K (the attention score between positions m and n),
    the rotation cancels in a specific way so that the score only depends on the
    RELATIVE position (m - n), not absolute positions m and n separately.
    This makes the model naturally aware of relative distances.

    Extrapolation: RoPE often generalises better to sequence lengths beyond training
    because it encodes relative rather than absolute position.
    """

    def __init__(self, head_size, max_len=5000):
        super().__init__()
        self.head_size = head_size

        # --- Precompute the rotation angles ---
        # theta_i = 1 / 10000^(2i / head_size)
        # These are the same frequencies as sinusoidal PE, applied to pairs of dimensions
        theta = 1.0 / (10000 ** (torch.arange(0, head_size, 2).float() / head_size))
        # positions: 0, 1, 2, ..., max_len-1
        pos   = torch.arange(max_len).float()
        # angles[p, i] = p * theta_i — shape: (max_len, head_size/2)
        angles = torch.outer(pos, theta)

        # Stack sin and cos: interleave so we can rotate pairs (dim 2i, dim 2i+1)
        # cos_sin shape: (max_len, head_size)
        cos = torch.cos(angles).repeat_interleave(2, dim=-1)  # repeat each value twice: [c0,c0,c1,c1,...]
        sin = torch.sin(angles).repeat_interleave(2, dim=-1)

        self.register_buffer('cos', cos)  # (max_len, head_size)
        self.register_buffer('sin', sin)  # (max_len, head_size)

    @staticmethod
    def rotate_half(x):
        """
        Rotate a tensor by 90° in each 2D subspace.
        If x = [x1, x2, x3, x4, ...], this returns [-x2, x1, -x4, x3, ...]
        This is the "half rotation" used in the RoPE formula.
        """
        # Split into even and odd dimensions
        x1 = x[..., 0::2]   # even dims:  x1, x3, x5, ...
        x2 = x[..., 1::2]   # odd dims:   x2, x4, x6, ...
        # Interleave: -x2, x1, -x4, x3, ...
        return torch.stack([-x2, x1], dim=-1).flatten(-2)

    def apply_rope(self, x, T):
        """
        Apply RoPE to a tensor x of shape (B, H, T, hs).
        Formula: x_rotated = x * cos(pos) + rotate_half(x) * sin(pos)
        """
        # Grab the first T rows of cos/sin tables
        cos = self.cos[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, hs) — broadcast over B, H
        sin = self.sin[:T].unsqueeze(0).unsqueeze(0)
        # Apply rotation: standard RoPE formula
        return x * cos + self.rotate_half(x) * sin

    def forward(self, q, k):
        """
        Rotate Q and K with position-dependent rotations.
        Q and K shape: (B, H, T, head_size)
        Returns rotated Q and K of the same shape.
        """
        T = q.shape[2]
        q_rot = self.apply_rope(q, T)
        k_rot = self.apply_rope(k, T)
        return q_rot, k_rot


class MultiHeadAttentionRoPE(nn.Module):
    """
    Multi-Head Attention with Rotary Position Embedding (RoPE).
    Drop-in replacement for standard MultiHeadAttention.
    RoPE is applied to Q and K before computing attention scores.
    """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size

        # Standard projections
        self.q_proj   = nn.Linear(n_embd, num_heads * head_size, bias=False)
        self.k_proj   = nn.Linear(n_embd, num_heads * head_size, bias=False)
        self.v_proj   = nn.Linear(n_embd, num_heads * head_size, bias=False)
        self.out_proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout  = nn.Dropout(0.2)

        # The RoPE encoder — applies rotation to Q and K
        self.rope = RotaryPositionalEncoding(head_size)

        # Causal mask
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        H, hs   = self.num_heads, self.head_size

        # Project Q, K, V and reshape to (B, H, T, hs)
        q = self.q_proj(x).view(B, T, H, hs).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, hs).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, hs).transpose(1, 2)

        # === Apply RoPE to Q and K ===
        # This bakes position information INTO the Q·K dot products
        q, k = self.rope(q, k)

        # Standard scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) * (hs ** -0.5)  # (B, H, T, T)
        scores = scores.masked_fill(self.tril[:T, :T].unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
        wei    = F.softmax(scores, dim=-1)
        wei    = self.dropout(wei)

        out = wei @ v                                     # (B, H, T, hs)
        out = out.transpose(1, 2).contiguous().view(B, T, H * hs)
        out = self.out_proj(out)
        return out


# need F for MultiHeadAttentionRoPE
import torch.nn.functional as F


# =============================================================================
# PART 3 — ALiBi: Attention with Linear Biases
# =============================================================================

class ALiBiAttention(nn.Module):
    """
    ALiBi — Attention with Linear Biases (Press et al., 2021).
    Paper: "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"

    Core idea: instead of adding positional encodings to token embeddings,
    add a PENALTY directly to attention scores based on distance between tokens.
    The penalty grows LINEARLY with distance: score(i,j) -= slope * (i - j)

    Each head has a different slope, so different heads attend at different distances.
    Slopes are fixed (not learned) — they follow a geometric sequence.

    Why does this extrapolate well?
    The bias is defined for ANY distance, not just ones seen during training.
    So even if we train with L=512, the bias works correctly at L=2048.
    """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size

        # Compute ALiBi slopes — one per head
        # Slopes follow geometric sequence: 2^(-8/H), 2^(-16/H), ..., 2^(-8)
        # where H = num_heads
        slopes = self._get_alibi_slopes(num_heads)  # (H,)
        self.register_buffer('slopes', slopes)

        # Standard Q, K, V projections — NO positional encoding added to embeddings
        self.q_proj   = nn.Linear(n_embd, num_heads * head_size, bias=False)
        self.k_proj   = nn.Linear(n_embd, num_heads * head_size, bias=False)
        self.v_proj   = nn.Linear(n_embd, num_heads * head_size, bias=False)
        self.out_proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout  = nn.Dropout(0.2)

    @staticmethod
    def _get_alibi_slopes(num_heads):
        """
        Generate ALiBi slopes for each head.
        Formula from the paper: slope_h = 2^(−8h/H) for h = 1..H
        """
        # ratio = 2^(-8/H)
        ratio  = 2 ** (-8 / num_heads)
        # slopes = [ratio^1, ratio^2, ..., ratio^H]
        slopes = torch.tensor([ratio ** i for i in range(1, num_heads + 1)])
        return slopes  # shape (H,)

    def _get_alibi_bias(self, T, device):
        """
        Build the (H, T, T) bias matrix where bias[h, i, j] = -slope_h * (i - j).
        For causal attention we only care about j <= i (past tokens).
        """
        # Position indices: 0, 1, ..., T-1
        pos = torch.arange(T, device=device)
        # Distance matrix: dist[i, j] = i - j  (how far back j is from i)
        dist = pos.unsqueeze(1) - pos.unsqueeze(0)  # (T, T)
        # Only keep distances >= 0 (causal: can't attend to future)
        dist = dist.clamp(min=0)  # (T, T)
        # Multiply by slopes: (H, 1, 1) * (1, T, T) → (H, T, T)
        bias = -self.slopes.view(-1, 1, 1) * dist.unsqueeze(0)  # (H, T, T)
        return bias  # negative = penalty for attending far away

    def forward(self, x):
        B, T, C = x.shape
        H, hs   = self.num_heads, self.head_size

        # Project Q, K, V and reshape to (B, H, T, hs)
        q = self.q_proj(x).view(B, T, H, hs).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, hs).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, hs).transpose(1, 2)

        # Compute raw attention scores
        scores = (q @ k.transpose(-2, -1)) * (hs ** -0.5)  # (B, H, T, T)

        # === Add ALiBi bias — this replaces positional encoding ===
        alibi_bias = self._get_alibi_bias(T, x.device)     # (H, T, T)
        scores     = scores + alibi_bias.unsqueeze(0)       # (B, H, T, T)

        # Causal mask: future tokens get -inf
        causal_mask = torch.tril(torch.ones(T, T, device=x.device))
        scores      = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))

        wei = F.softmax(scores, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v                                      # (B, H, T, hs)
        out = out.transpose(1, 2).contiguous().view(B, T, H * hs)
        out = self.out_proj(out)
        return out


# =============================================================================
# PART 3 — Relative Positional Encoding (Shaw et al., 2018)
# =============================================================================

class RelativeAttention(nn.Module):
    """
    Attention with Relative Position Representations (Shaw et al., 2018).
    Paper: "Self-Attention with Relative Position Representations"

    Standard attention: score(i,j) = Q_i · K_j
    Relative PE:        score(i,j) = Q_i · K_j  +  Q_i · r_{i-j}
                        where r_{i-j} is a LEARNED embedding for relative distance (i-j)

    Key difference from RoPE/ALiBi:
    - RoPE bakes position into Q and K via rotation
    - ALiBi adds a fixed bias based on distance
    - Relative PE adds LEARNED embeddings based on relative distance

    The model learns what "2 positions apart" or "5 positions apart" MEANS for the task.
    We clip distances to a maximum (max_relative_pos) to keep the table small.
    """

    def __init__(self, num_heads, head_size, max_relative_pos=16):
        super().__init__()
        self.num_heads        = num_heads
        self.head_size        = head_size
        self.max_relative_pos = max_relative_pos  # clip distances beyond this

        self.q_proj   = nn.Linear(n_embd, num_heads * head_size, bias=False)
        self.k_proj   = nn.Linear(n_embd, num_heads * head_size, bias=False)
        self.v_proj   = nn.Linear(n_embd, num_heads * head_size, bias=False)
        self.out_proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout  = nn.Dropout(0.2)

        # Learnable relative position embeddings
        # We have 2*max_relative_pos + 1 possible distances:
        #   -max_relative_pos, ..., -1, 0, 1, ..., max_relative_pos
        # Index 0 = -max_relative_pos, index max_relative_pos = 0, etc.
        num_embeddings = 2 * max_relative_pos + 1
        self.rel_pos_emb = nn.Embedding(num_embeddings, head_size)

    def _get_rel_pos_matrix(self, T, device):
        """
        Build a (T, T) matrix of relative position INDICES (clamped and shifted).
        rel_pos[i, j] = clamp(i - j, -max, max) + max   (shifted to be >= 0 for indexing)
        """
        idx      = torch.arange(T, device=device)
        rel_dist = idx.unsqueeze(1) - idx.unsqueeze(0)  # (T, T): i - j
        # Clamp to [-max_relative_pos, max_relative_pos]
        rel_dist = rel_dist.clamp(-self.max_relative_pos, self.max_relative_pos)
        # Shift so minimum index is 0 (required for nn.Embedding lookup)
        rel_idx  = rel_dist + self.max_relative_pos     # (T, T), values in [0, 2*max+1)
        return rel_idx

    def forward(self, x):
        B, T, C = x.shape
        H, hs   = self.num_heads, self.head_size

        q = self.q_proj(x).view(B, T, H, hs).transpose(1, 2)  # (B, H, T, hs)
        k = self.k_proj(x).view(B, T, H, hs).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, hs).transpose(1, 2)

        # --- Standard content-based attention scores ---
        scores = (q @ k.transpose(-2, -1)) * (hs ** -0.5)  # (B, H, T, T)

        # --- Relative position bias ---
        rel_idx = self._get_rel_pos_matrix(T, x.device)    # (T, T)
        r       = self.rel_pos_emb(rel_idx)                 # (T, T, hs): learned embedding per pair

        # Add relative position contribution: Q_i · r_{i-j}
        # q: (B, H, T, hs) → sum over hs → (B, H, T, T)
        # einsum 'bhtd, tsd -> bhts': for each (batch, head, query t, key s),
        #   dot product of q[t] with r[t,s]
        rel_scores = torch.einsum('bhtd,tsd->bhts', q, r) * (hs ** -0.5)
        scores     = scores + rel_scores                    # (B, H, T, T)

        # Causal mask
        causal_mask = torch.tril(torch.ones(T, T, device=x.device))
        scores      = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))

        wei = F.softmax(scores, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v                                      # (B, H, T, hs)
        out = out.transpose(1, 2).contiguous().view(B, T, H * hs)
        out = self.out_proj(out)
        return out
