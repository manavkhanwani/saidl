# =============================================================================
# attention.py — Attention mechanisms
#
# This file implements:
#   1. Standard Self-Attention (baseline, from "Attention Is All You Need")
#   2. Sliding Window / Local Attention  (Part 2 variant A)
#   3. Linear Attention  (Part 2 variant B)
#   4. Multi-Query Attention (MQA)  (Part 2 variant C)
#
# Each class is a drop-in replacement for the others — same input, same output.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import n_embd, n_head, dropout, block_size


# =============================================================================
# PART 1 — Standard Self-Attention Head (Baseline)
# =============================================================================

class Head(nn.Module):
    """
    One head of standard causal (masked) self-attention.

    How self-attention works (plain English):
      - Every token produces three vectors: a Query (Q), a Key (K), and a Value (V).
      - Q = "what am I looking for?"
      - K = "what information do I have?"
      - V = "here is my actual content"
      - We score every pair (Q_i, K_j) to decide how much token i should attend to token j.
      - "Causal" masking means token i can only attend to tokens 0..i (no peeking at the future).
      - The final output is a weighted sum of all V vectors, weighted by the attention scores.
    """

    def __init__(self, head_size):
        super().__init__()
        # Linear layers that project from embedding space into head space.
        # bias=False is standard practice for attention projections.
        self.key   = nn.Linear(n_embd, head_size, bias=False)  # K projection
        self.query = nn.Linear(n_embd, head_size, bias=False)  # Q projection
        self.value = nn.Linear(n_embd, head_size, bias=False)  # V projection

        # register_buffer means this tensor is NOT a learnable parameter —
        # it's just a constant mask we reuse every forward pass.
        # tril creates a lower-triangular matrix of 1s:
        #   [[1,0,0],
        #    [1,1,0],
        #    [1,1,1]]
        # Position (i,j) is 1 if token i is allowed to attend to token j.
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        # Dropout randomly zeroes some attention weights during training.
        # This acts like regularisation, stopping the model from over-relying on
        # any single attention connection.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (Batch, Time/sequence-length, Channels/embedding-dim)
        B, T, C = x.shape

        # --- Step 1: Compute Keys, Queries, Values ---
        k = self.key(x)    # shape: (B, T, head_size)
        q = self.query(x)  # shape: (B, T, head_size)
        v = self.value(x)  # shape: (B, T, head_size)

        # --- Step 2: Compute raw attention scores ---
        # (B, T, head_size) @ (B, head_size, T) → (B, T, T)
        # Each entry [b, i, j] = dot product of query_i and key_j
        # = "how much should position i attend to position j?"
        # We divide by sqrt(head_size) to prevent the dot products from getting
        # very large, which would push softmax into regions with tiny gradients.
        scale = k.shape[-1] ** -0.5   # = 1 / sqrt(head_size)
        wei = q @ k.transpose(-2, -1) * scale  # (B, T, T)

        # --- Step 3: Causal (future) masking ---
        # Replace any position where tril==0 with -infinity.
        # After softmax, -inf becomes 0, so future tokens get zero attention weight.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)

        # --- Step 4: Softmax → attention weights that sum to 1 per row ---
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)       # randomly drop some weights during training

        # --- Step 5: Weighted sum of Values ---
        # (B, T, T) @ (B, T, head_size) → (B, T, head_size)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention: run several attention heads in parallel and concatenate results.

    Why multiple heads?
    Each head can focus on a different type of relationship in the sequence.
    For example, one head might learn syntax, another might track coreference.
    Concatenating all heads' outputs and projecting back gives richer representations.
    """

    def __init__(self, num_heads, head_size):
        super().__init__()
        # Create a list of independent attention heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

        # After concatenating all heads, project back to n_embd.
        # Input size = head_size * num_heads = n_embd (by design)
        self.proj    = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Run all heads on x and concatenate along the last dimension
        # Each head output: (B, T, head_size) → concat → (B, T, head_size * num_heads)
        out = torch.cat([h(x) for h in self.heads], dim=-1)

        # Project back to (B, T, n_embd) and apply dropout
        out = self.dropout(self.proj(out))
        return out


# =============================================================================
# PART 2 — Attention Variant A: Sliding Window / Local Attention
# =============================================================================

class SlidingWindowAttention(nn.Module):
    """
    Sliding Window (Local) Attention.

    Problem with standard attention: every token attends to ALL previous tokens.
    For long sequences this is O(T²) — very slow and memory hungry.

    Idea: each token only attends to a small local window of its nearest neighbours.
    This makes attention O(T * window_size) instead of O(T²), which is much cheaper.

    Think of it like reading: you understand a word mostly from the words around it,
    not from a sentence 500 words ago.
    """

    def __init__(self, num_heads, head_size, window_size=16):
        super().__init__()
        self.num_heads   = num_heads
        self.head_size   = head_size
        self.window_size = window_size  # How many past tokens each token can attend to

        # Projections for Q, K, V — all heads at once for efficiency
        self.q_proj = nn.Linear(n_embd, num_heads * head_size, bias=False)
        self.k_proj = nn.Linear(n_embd, num_heads * head_size, bias=False)
        self.v_proj = nn.Linear(n_embd, num_heads * head_size, bias=False)

        # Output projection back to n_embd
        self.out_proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        H  = self.num_heads
        hs = self.head_size

        # Project and reshape: (B, T, H*hs) → (B, H, T, hs)
        # This lets each head operate independently on (T, hs) slices
        q = self.q_proj(x).view(B, T, H, hs).transpose(1, 2)  # (B, H, T, hs)
        k = self.k_proj(x).view(B, T, H, hs).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, hs).transpose(1, 2)

        # --- Build causal mask limited to window_size ---
        # We create a (T, T) boolean mask where True means "allowed to attend".
        # Condition: j <= i (causal) AND i - j < window_size (local)
        idx = torch.arange(T, device=x.device)
        # idx[None, :] = row indices (query positions), idx[:, None] = col indices (key positions)
        # Wait, let's do it more clearly:
        row = idx.unsqueeze(1)  # shape (T, 1) — query index
        col = idx.unsqueeze(0)  # shape (1, T) — key index
        # A query at position `row` can attend to a key at position `col` if:
        #   col <= row          (causal: don't look at future)
        #   row - col < window  (local: only look at recent tokens)
        local_mask = (col <= row) & ((row - col) < self.window_size)  # (T, T) bool
        # Convert to float attention bias: 0 where allowed, -inf where blocked
        attn_bias = torch.zeros(T, T, device=x.device)
        attn_bias = attn_bias.masked_fill(~local_mask, float('-inf'))  # (T, T)

        # --- Scaled dot-product attention with local mask ---
        scale = hs ** -0.5
        # (B, H, T, hs) @ (B, H, hs, T) → (B, H, T, T)
        scores = (q @ k.transpose(-2, -1)) * scale
        scores = scores + attn_bias  # add -inf to blocked positions
        wei    = F.softmax(scores, dim=-1)
        wei    = self.dropout(wei)

        # (B, H, T, T) @ (B, H, T, hs) → (B, H, T, hs)
        out = wei @ v
        # Reshape back: (B, H, T, hs) → (B, T, H*hs) → (B, T, n_embd)
        out = out.transpose(1, 2).contiguous().view(B, T, H * hs)
        out = self.out_proj(out)
        return out


# =============================================================================
# PART 2 — Attention Variant B: Linear Attention
# =============================================================================

class LinearAttention(nn.Module):
    """
    Linear Attention (kernel-based, O(T) complexity).

    Standard softmax attention: score(Q,K) = softmax(Q @ K^T / sqrt(d))
    This is O(T²) because we compute a T×T matrix.

    Linear attention replaces softmax with a kernel trick:
        softmax(Q,K) ≈ φ(Q) · φ(K)^T
    where φ is a feature map (we use ELU+1 here, which is cheap and stable).

    The magic: instead of computing φ(Q) @ (φ(K)^T @ V) with the T×T matrix,
    we compute φ(K)^T @ V first (a small d×d matrix), then multiply by φ(Q).
    This is O(T * d²) — linear in T!

    Trade-off: approximation quality may be lower than softmax for short sequences,
    but it scales much better to long sequences.
    """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size

        self.q_proj   = nn.Linear(n_embd, num_heads * head_size, bias=False)
        self.k_proj   = nn.Linear(n_embd, num_heads * head_size, bias=False)
        self.v_proj   = nn.Linear(n_embd, num_heads * head_size, bias=False)
        self.out_proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout  = nn.Dropout(dropout)

    @staticmethod
    def feature_map(x):
        """
        φ(x) = ELU(x) + 1
        This ensures all values are positive (needed so the kernel approximation
        stays non-negative, like real attention weights).
        ELU(x) = x if x>0, else exp(x)-1
        Adding 1 shifts the range to [0, ∞).
        """
        return F.elu(x) + 1

    def forward(self, x):
        B, T, C = x.shape
        H  = self.num_heads
        hs = self.head_size

        # Project and reshape → (B, H, T, hs)
        q = self.q_proj(x).view(B, T, H, hs).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, hs).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, hs).transpose(1, 2)

        # Apply the feature map to Q and K (replaces softmax)
        q = self.feature_map(q)  # (B, H, T, hs)
        k = self.feature_map(k)  # (B, H, T, hs)

        # --- Causal linear attention via prefix sums ---
        # For causal attention we need: output_i = (Σ_{j≤i} k_j^T v_j) applied to q_i
        # We do this with a running cumulative sum (prefix sum) over time.
        # kv_cumsum[i] = Σ_{j=0}^{i} k_j ⊗ v_j  (outer product, shape hs×hs per head)
        # k_cumsum[i]  = Σ_{j=0}^{i} k_j          (shape hs per head)

        # Outer product: (B, H, T, hs, 1) * (B, H, T, 1, hs) → (B, H, T, hs, hs)
        kv = torch.einsum('bhti,bhtj->bhtij', k, v)  # (B, H, T, hs, hs)
        # Cumulative sum along time dimension gives us the prefix sums
        kv_cumsum = torch.cumsum(kv, dim=2)   # (B, H, T, hs, hs)
        k_cumsum  = torch.cumsum(k,  dim=2)   # (B, H, T, hs)

        # Compute numerator: q_i · (Σ_{j≤i} k_j v_j)
        # einsum: q is (B,H,T,hs), kv_cumsum is (B,H,T,hs,hs) → output (B,H,T,hs)
        num = torch.einsum('bhti,bhtij->bhtj', q, kv_cumsum)  # (B, H, T, hs)

        # Compute denominator (normalisation): q_i · (Σ_{j≤i} k_j)
        # This makes the outputs sum to roughly 1, like softmax
        den = (q * k_cumsum).sum(dim=-1, keepdim=True) + 1e-6  # (B, H, T, 1), +eps for stability

        # Normalised output
        out = num / den  # (B, H, T, hs)
        out = self.dropout(out)

        # Reshape back → (B, T, n_embd)
        out = out.transpose(1, 2).contiguous().view(B, T, H * hs)
        out = self.out_proj(out)
        return out


# =============================================================================
# PART 2 — Attention Variant C: Multi-Query Attention (MQA)
# =============================================================================

class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention (MQA) — from "Fast Transformer Decoding" (Shazeer, 2019).

    Standard multi-head attention: each head has its OWN K and V projections.
    → lots of memory for caching K/V during generation.

    MQA idea: share a SINGLE K and V across all heads, but keep separate Q per head.
    This dramatically reduces the KV-cache size during inference (makes generation faster).
    Quality is usually very close to full multi-head attention.

    Memory: standard MHA needs 2 * num_heads * head_size memory for K/V per token.
            MQA only needs 2 * head_size (shared K/V) — num_heads times smaller!
    """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size

        # Each head has its own Query projection
        self.q_proj = nn.Linear(n_embd, num_heads * head_size, bias=False)
        # But only ONE Key and ONE Value projection shared across all heads
        self.k_proj = nn.Linear(n_embd, head_size, bias=False)  # single K
        self.v_proj = nn.Linear(n_embd, head_size, bias=False)  # single V

        self.out_proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout  = nn.Dropout(dropout)

        # Causal mask (same as standard attention)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        H  = self.num_heads
        hs = self.head_size

        # Q is per-head: (B, T, H*hs) → (B, H, T, hs)
        q = self.q_proj(x).view(B, T, H, hs).transpose(1, 2)  # (B, H, T, hs)

        # K and V are shared (single head): (B, T, hs)
        k = self.k_proj(x)  # (B, T, hs) — ONE key for all heads
        v = self.v_proj(x)  # (B, T, hs) — ONE value for all heads

        # Expand K and V to match all heads for batched matmul:
        # (B, T, hs) → (B, 1, T, hs) → (B, H, T, hs)
        k = k.unsqueeze(1).expand(-1, H, -1, -1)  # broadcast to all heads
        v = v.unsqueeze(1).expand(-1, H, -1, -1)

        # --- Scaled dot-product attention (same as standard) ---
        scale  = hs ** -0.5
        scores = (q @ k.transpose(-2, -1)) * scale  # (B, H, T, T)

        # Apply causal mask
        scores = scores.masked_fill(self.tril[:T, :T].unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
        wei    = F.softmax(scores, dim=-1)  # (B, H, T, T)
        wei    = self.dropout(wei)

        # Weighted aggregation of values
        out = wei @ v  # (B, H, T, hs)

        # Reshape back → (B, T, n_embd)
        out = out.transpose(1, 2).contiguous().view(B, T, H * hs)
        out = self.out_proj(out)
        return out
