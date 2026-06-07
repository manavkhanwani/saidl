# =============================================================================
# train.py — Main Training Script
#
# This script runs ALL experiments for Parts 1-5:
#   Part 1: Baseline Transformer with sinusoidal PE
#   Part 2: 3 attention variants (Sliding Window, Linear, Multi-Query)
#   Part 3: 3 positional encoding variants (RoPE, ALiBi, Relative PE) + extrapolation test
#   Part 4: Conv+Attention hybrids (Part 4 combined with best attention from Part 2)
#   Part 5: Results table logged to results/results.csv
#
# HOW TO RUN:
#   python train.py
#
# Results appear in:
#   results/results.csv  — comparison table of all experiments
#   results/log.txt      — full training log
# =============================================================================

import torch
import torch.nn as nn
import time
import csv
import os

from config import (
    batch_size, block_size, max_iters, eval_interval, eval_iters,
    learning_rate, device, n_embd, n_head, n_layer, L_train, L_test
)
from model import GPTLanguageModel

# Create output directory for results
os.makedirs('results', exist_ok=True)

# Set random seed for reproducibility (same seed = same results every run)
torch.manual_seed(1337)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(path='data/train.txt'):
    """
    Load the text dataset and build a character-level vocabulary.

    Character-level modelling: each character is one token.
    The model learns to predict the next CHARACTER given the previous ones.
    This is simpler than word-level (no need for a tokeniser).
    """
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Find all unique characters in the text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"[Data] Vocab size: {vocab_size} unique characters")
    print(f"[Data] Dataset size: {len(text):,} characters")

    # Create lookup tables: character → integer and integer → character
    stoi = {ch: i for i, ch in enumerate(chars)}  # string to int
    itos = {i: ch for i, ch in enumerate(chars)}  # int to string

    # Encoder: convert a string to a list of integers
    encode = lambda s: [stoi[c] for c in s]
    # Decoder: convert a list of integers back to a string
    decode = lambda l: ''.join([itos[i] for i in l])

    # Convert the entire text to a tensor of integers
    data = torch.tensor(encode(text), dtype=torch.long)

    # Split into 90% train, 10% validation
    # We never touch the validation set during training — it's for honest evaluation
    n          = int(0.9 * len(data))
    train_data = data[:n]
    val_data   = data[n:]

    return train_data, val_data, vocab_size, encode, decode


# Load data once at the top level (shared across all experiments)
train_data, val_data, vocab_size, encode, decode = load_data()


def get_batch(split, ctx_len=None):
    """
    Sample a random batch of (input, target) pairs from the dataset.

    split: 'train' or 'val'
    ctx_len: override context length (for extrapolation experiments)

    How it works:
    - Pick random starting positions in the data
    - Input  x: tokens at positions [start, start+ctx_len)
    - Target y: tokens at positions [start+1, start+ctx_len+1)  (shifted by 1)
    - The model learns: "given x, predict y" (next-token prediction)
    """
    ctx = ctx_len if ctx_len else block_size
    data   = train_data if split == 'train' else val_data
    # Random starting positions for each sequence in the batch
    ix     = torch.randint(len(data) - ctx, (batch_size,))
    # Stack sequences into a batch tensor
    x      = torch.stack([data[i    : i + ctx    ] for i in ix])  # inputs
    y      = torch.stack([data[i + 1: i + ctx + 1] for i in ix])  # targets
    x, y   = x.to(device), y.to(device)
    return x, y


@torch.no_grad()  # no_grad means we don't track gradients (saves memory during evaluation)
def estimate_loss(model, ctx_len=None):
    """
    Estimate average training and validation loss over many batches.

    We average over eval_iters batches to get a stable estimate —
    a single batch has high variance due to random sampling.
    """
    out = {}
    model.eval()  # switch to eval mode (disables dropout)
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y       = get_batch(split, ctx_len=ctx_len)
            logits, loss = model(X, Y)
            losses[k]  = loss.item()
        out[split] = losses.mean().item()
    model.train()  # switch back to training mode (re-enables dropout)
    return out


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_model(attention_type='standard', positional_type='sinusoidal',
                use_conv=False, ctx_len=None, experiment_name='baseline'):
    """
    Train one model configuration and return metrics.

    Returns a dict with:
        train_loss      : final training loss
        val_loss        : final validation loss
        train_time_s    : training time in seconds
        throughput      : tokens per second processed during training
        peak_memory_mb  : peak GPU memory used (0 on CPU)
        experiment_name : label for this experiment
    """
    ctx = ctx_len if ctx_len else block_size
    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"  Attention:  {attention_type}")
    print(f"  Positional: {positional_type}")
    print(f"  Conv:       {use_conv}")
    print(f"  Context:    {ctx}")
    print(f"{'='*60}")

    # Build the model with the specified configuration
    model = GPTLanguageModel(
        vocab_size       = vocab_size,
        attention_type   = attention_type,
        positional_type  = positional_type,
        use_conv         = use_conv,
        block_size_override = ctx
    ).to(device)

    # Print total parameter count
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params/1e6:.2f}M")

    # AdamW is Adam with weight decay — a common choice for Transformers
    # Weight decay adds a small L2 penalty, preventing weights from growing too large
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Reset GPU memory stats (for peak memory tracking)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    total_tokens = 0  # track tokens processed for throughput calculation
    start_time   = time.time()
    final_losses = {'train': float('inf'), 'val': float('inf')}

    # ----- Main Training Loop -----
    for step in range(max_iters):

        # Every eval_interval steps, evaluate and print progress
        if step % eval_interval == 0 or step == max_iters - 1:
            losses = estimate_loss(model, ctx_len=ctx)
            final_losses = losses
            print(f"  Step {step:4d}/{max_iters} | "
                  f"train loss: {losses['train']:.4f} | "
                  f"val loss:   {losses['val']:.4f}")

        # --- Forward pass ---
        xb, yb        = get_batch('train', ctx_len=ctx)
        logits, loss  = model(xb, yb)
        total_tokens += xb.numel()  # count tokens processed

        # --- Backward pass ---
        optimizer.zero_grad(set_to_none=True)  # clear previous gradients
        loss.backward()                         # compute gradients
        optimizer.step()                        # update weights

    # ----- Post-training metrics -----
    end_time     = time.time()
    elapsed      = end_time - start_time
    throughput   = total_tokens / elapsed  # tokens per second

    # Peak GPU memory in MB (0 if on CPU)
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        peak_mem_mb = 0.0

    print(f"\n  >>> RESULTS for {experiment_name}")
    print(f"      Train loss:    {final_losses['train']:.4f}")
    print(f"      Val loss:      {final_losses['val']:.4f}")
    print(f"      Training time: {elapsed:.1f}s")
    print(f"      Throughput:    {throughput:.0f} tokens/sec")
    print(f"      Peak memory:   {peak_mem_mb:.1f} MB")

    return {
        'experiment'   : experiment_name,
        'attention'    : attention_type,
        'positional'   : positional_type,
        'conv'         : use_conv,
        'context_len'  : ctx,
        'train_loss'   : round(final_losses['train'], 4),
        'val_loss'     : round(final_losses['val'],   4),
        'train_time_s' : round(elapsed,    1),
        'throughput'   : round(throughput, 0),
        'peak_mem_mb'  : round(peak_mem_mb, 1),
    }


# =============================================================================
# EXTRAPOLATION TEST (Part 3c)
# =============================================================================

def extrapolation_test(attention_type, positional_type, experiment_name):
    """
    Part 3(c): Train on L_train tokens, evaluate on L_test tokens.

    This tests whether the positional encoding can generalise BEYOND its training length.
    - Good extrapolation: val perplexity at L_test similar to L_train
    - Poor extrapolation: val perplexity explodes at L_test (encoding breaks down)

    Perplexity = exp(cross-entropy loss) — lower is better.
    """
    print(f"\n--- Extrapolation Test: {experiment_name} ---")
    print(f"    Training context:    {L_train}")
    print(f"    Evaluation context:  {L_test}")

    # Train the model with L_train context length
    model = GPTLanguageModel(
        vocab_size       = vocab_size,
        attention_type   = attention_type,
        positional_type  = positional_type,
        block_size_override = L_train
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for step in range(max_iters):
        xb, yb       = get_batch('train', ctx_len=L_train)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Evaluate at TRAINING length (should be good)
    losses_train_len = estimate_loss(model, ctx_len=L_train)

    # Now evaluate at LONGER length L_test
    # We need to update the model's ctx_len so generate/attention uses longer sequences
    model.ctx_len = L_test
    # For the causal mask in standard attention, we need to be careful;
    # sinusoidal PE can handle any length since we use max_len=5000 in the buffer.
    # RoPE and ALiBi are designed to extrapolate naturally.
    try:
        losses_test_len = estimate_loss(model, ctx_len=L_test)
        val_loss_extrap = losses_test_len['val']
    except Exception as e:
        print(f"    [Warning] Extrapolation evaluation failed: {e}")
        val_loss_extrap = float('nan')

    import math
    ppl_train = math.exp(min(losses_train_len['val'], 20))  # clamp to avoid overflow
    ppl_extrap = math.exp(min(val_loss_extrap, 20)) if not (val_loss_extrap != val_loss_extrap) else float('nan')

    print(f"    Val perplexity at L_train={L_train}: {ppl_train:.2f}")
    print(f"    Val perplexity at L_test={L_test}:  {ppl_extrap:.2f}")

    return {
        'experiment'       : f"extrap_{experiment_name}",
        'attention'        : attention_type,
        'positional'       : positional_type,
        'L_train'          : L_train,
        'L_test'           : L_test,
        'ppl_at_L_train'   : round(ppl_train,  2),
        'ppl_at_L_test'    : round(ppl_extrap, 2) if ppl_extrap == ppl_extrap else 'NaN',
    }


# =============================================================================
# MAIN — Run all experiments
# =============================================================================

if __name__ == '__main__':

    all_results     = []  # collects metrics for the comparison table
    extrap_results  = []  # collects extrapolation test results

    # -------------------------------------------------------------------------
    # PART 1 — Baseline: Standard Transformer + Sinusoidal PE
    # -------------------------------------------------------------------------
    print("\n\n" + "="*60)
    print("PART 1 — BASELINE: Standard Transformer")
    print("="*60)
    r = train_model(
        attention_type  = 'standard',
        positional_type = 'sinusoidal',
        experiment_name = 'Part1_Baseline'
    )
    all_results.append(r)

    # -------------------------------------------------------------------------
    # PART 2 — Attention Variants
    # -------------------------------------------------------------------------
    print("\n\n" + "="*60)
    print("PART 2 — ATTENTION VARIANTS")
    print("="*60)

    # Variant A: Sliding Window (Local) Attention
    r = train_model(
        attention_type  = 'sliding_window',
        positional_type = 'sinusoidal',
        experiment_name = 'Part2A_SlidingWindow'
    )
    all_results.append(r)

    # Variant B: Linear Attention
    r = train_model(
        attention_type  = 'linear',
        positional_type = 'sinusoidal',
        experiment_name = 'Part2B_LinearAttention'
    )
    all_results.append(r)

    # Variant C: Multi-Query Attention
    r = train_model(
        attention_type  = 'multi_query',
        positional_type = 'sinusoidal',
        experiment_name = 'Part2C_MultiQuery'
    )
    all_results.append(r)

    # -------------------------------------------------------------------------
    # PART 3 — Positional Encoding Variants
    # -------------------------------------------------------------------------
    print("\n\n" + "="*60)
    print("PART 3 — POSITIONAL ENCODING VARIANTS")
    print("="*60)

    # RoPE: Rotary Position Embedding (handles PE inside attention)
    r = train_model(
        attention_type  = 'rope',
        positional_type = 'rope',   # PE is internal; label for clarity
        experiment_name = 'Part3A_RoPE'
    )
    all_results.append(r)

    # ALiBi: Attention with Linear Biases (handles PE inside attention)
    r = train_model(
        attention_type  = 'alibi',
        positional_type = 'alibi',
        experiment_name = 'Part3B_ALiBi'
    )
    all_results.append(r)

    # Relative PE (handles PE inside attention)
    r = train_model(
        attention_type  = 'relative',
        positional_type = 'relative',
        experiment_name = 'Part3C_RelativePE'
    )
    all_results.append(r)

    # -------------------------------------------------------------------------
    # PART 3c — Extrapolation Test
    # -------------------------------------------------------------------------
    print("\n\n" + "="*60)
    print("PART 3c — EXTRAPOLATION TEST")
    print(f"Train on context={L_train}, evaluate on context={L_test}")
    print("="*60)

    # Sinusoidal PE: generally poor extrapolation (fixed frequencies, no relative bias)
    e = extrapolation_test('standard', 'sinusoidal', 'Sinusoidal')
    extrap_results.append(e)

    # ALiBi: designed specifically for length extrapolation
    e = extrapolation_test('alibi', 'alibi', 'ALiBi')
    extrap_results.append(e)

    # RoPE: generally good extrapolation due to relative nature
    e = extrapolation_test('rope', 'rope', 'RoPE')
    extrap_results.append(e)

    # -------------------------------------------------------------------------
    # PART 4 — Conv + Attention Hybrids
    # -------------------------------------------------------------------------
    print("\n\n" + "="*60)
    print("PART 4 — CONV + ATTENTION HYBRIDS")
    print("Combining 1D convolution with best attention variant")
    print("="*60)

    # Hybrid: Conv + Standard Attention
    r = train_model(
        attention_type  = 'standard',
        positional_type = 'sinusoidal',
        use_conv        = True,
        experiment_name = 'Part4A_Conv_Standard'
    )
    all_results.append(r)

    # Hybrid: Conv + Sliding Window Attention (a good combo for efficiency)
    r = train_model(
        attention_type  = 'sliding_window',
        positional_type = 'sinusoidal',
        use_conv        = True,
        experiment_name = 'Part4B_Conv_SlidingWindow'
    )
    all_results.append(r)

    # -------------------------------------------------------------------------
    # PART 5 — Save Results Table
    # -------------------------------------------------------------------------
    print("\n\n" + "="*60)
    print("PART 5 — RESULTS TABLE")
    print("="*60)

    # Write main results CSV
    csv_path = 'results/results.csv'
    fieldnames = ['experiment', 'attention', 'positional', 'conv',
                  'context_len', 'train_loss', 'val_loss',
                  'train_time_s', 'throughput', 'peak_mem_mb']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    # Write extrapolation results CSV
    extrap_csv_path = 'results/extrapolation_results.csv'
    extrap_fields = ['experiment', 'attention', 'positional', 'L_train', 'L_test',
                     'ppl_at_L_train', 'ppl_at_L_test']
    with open(extrap_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=extrap_fields)
        writer.writeheader()
        writer.writerows(extrap_results)

    # Print a nicely formatted table to the console
    print(f"\n{'Experiment':<35} {'Val Loss':>10} {'Time(s)':>9} {'Tokens/s':>10} {'Mem(MB)':>9}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['experiment']:<35} {r['val_loss']:>10.4f} {r['train_time_s']:>9.1f} "
              f"{r['throughput']:>10.0f} {r['peak_mem_mb']:>9.1f}")

    print(f"\n\nExtrapolation Results:")
    print(f"{'Encoding':<20} {'PPL at L_train':>16} {'PPL at L_test':>14}")
    print("-" * 55)
    for e in extrap_results:
        print(f"{e['positional']:<20} {str(e['ppl_at_L_train']):>16} {str(e['ppl_at_L_test']):>14}")

    print(f"\n\nResults saved to: {csv_path}")
    print(f"Extrapolation results saved to: {extrap_csv_path}")
    print("\nDone! All experiments complete.")
