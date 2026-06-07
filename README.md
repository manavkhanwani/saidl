# Core ML Assignment — Long-Context Sequence Modelling

**Author:** Manav  
**Dataset:** WikiText-2 corpus, character-level tokenisation  
**Platform:** Google Colab / Kaggle (CPU)

---

## What This Project Does

We build a modular Transformer language model and systematically compare different design choices:

1. **Part 1** — Baseline Transformer (standard self-attention + sinusoidal positional encoding)
2. **Part 2** — Three alternative attention mechanisms (Sliding Window, Linear, Multi-Query)
3. **Part 3** — Three positional encoding variants (RoPE, ALiBi, Relative PE) + extrapolation test
4. **Part 4** — Hybrid architectures adding 1D convolution before attention
5. **Part 5** — Comparison table of all results (loss, perplexity, speed)

---

## Dataset

We use the **WikiText-2** corpus with **character-level tokenisation**. Each unique character is one token (vocabulary size ≈ 96). This is simpler than word-level tokenisation (no tokeniser needed) and allows the model to be fully self-contained.

This is **not** subword/BPE tokenisation — the model predicts one character at a time.

---

## File Structure

```
submission/
│
├── config.py               # All hyperparameters (change things here)
├── attention.py            # Attention: Standard, Sliding Window, Linear, Multi-Query
├── positional.py           # PE: Sinusoidal, RoPE, ALiBi, Relative
├── model.py                # Full Transformer model (modular, swappable components)
├── train.py                # Runs all experiments, saves results
│
├── data/
│   ├── train.txt           # Training text
│   ├── valid.txt           # Validation text
│   └── test.txt            # Test text
│
├── readings.txt            # All experimental results with loss curves
├── generated_samples.txt   # Generated text samples from each variant
├── attention_results.txt   # Attention variant comparison table
├── positional_results.txt  # Positional encoding comparison table
├── config_summary.txt      # All hyperparameters documented
│
└── results/                # Created automatically when you run train.py
    ├── results.csv
    └── extrapolation_results.csv
```

---

## Running Experiments

```bash
cd submission/
python train.py
```

Results are saved to `results/results.csv` and `results/extrapolation_results.csv`.  
Pre-run readings are documented in `readings.txt`.

---

## Hyperparameters

| Parameter     | Value  | Description                          |
|---------------|--------|--------------------------------------|
| block_size    | 64     | Context window (tokens)              |
| n_embd        | 128    | Embedding dimension                  |
| n_head        | 4      | Attention heads                      |
| n_layer       | 2      | Transformer blocks                   |
| dropout       | 0.2    | Dropout rate                         |
| batch_size    | 16     | Sequences per batch                  |
| max_iters     | 500    | Training steps                       |
| learning_rate | 3e-4   | AdamW learning rate                  |

---

## Component Overview

### `attention.py`

| Class | Complexity | Description |
|---|---|---|
| `MultiHeadAttention` | O(T²) | Standard attention (baseline) |
| `SlidingWindowAttention` | O(T·w) | Each token attends to w nearest neighbours |
| `LinearAttention` | O(T) | Kernel approximation via ELU+1 feature map |
| `MultiQueryAttention` | O(T²) | All heads share a single K and V |

### `positional.py`

| Class | Learnable | Extrapolates | Description |
|---|---|---|---|
| `SinusoidalPositionalEncoding` | No | Poor | Fixed sin/cos waves added to embeddings |
| `MultiHeadAttentionRoPE` | No | Good | Rotates Q and K by position |
| `ALiBiAttention` | No | Excellent | Adds linear distance penalty to scores |
| `RelativeAttention` | Yes (small table) | Moderate | Learned embeddings per relative distance |

### `model.py`

`GPTLanguageModel` accepts:
- `attention_type` → which attention class to use
- `positional_type` → which PE to use  
- `use_conv` → add depthwise separable Conv1D before each attention block (Part 4)

---

## Results Summary

| Experiment               | Val Loss | PPL   | Time (s) |
|--------------------------|----------|-------|----------|
| Baseline (Std+Sinusoidal)| 2.2391   |  9.38 |    87.3  |
| Sliding Window           | 2.2614   |  9.60 |    79.6  |
| Linear Attention         | 2.3882   | 10.89 |    74.2  |
| Multi-Query              | 2.2508   |  9.50 |    82.1  |
| RoPE                     | 2.2183   |  9.19 |    91.4  |
| ALiBi                    | 2.2244   |  9.25 |    89.7  |
| Relative PE              | 2.2377   |  9.37 |    93.2  |
| Conv + Standard          | 2.2071   |  9.09 |   104.8  |
| Conv + Sliding Window    | 2.2302   |  9.30 |    97.3  |

Full readings, loss curves, and generated samples: see `readings.txt`, `generated_samples.txt`.

---

## Key Concepts

**Perplexity** — exp(cross-entropy loss). Lower is better. A perplexity of 9.38 means the model is, on average, as uncertain as choosing between ~9 equally likely characters.

**Extrapolation** — can the model handle sequences longer than it was trained on? Sinusoidal PE fails badly; ALiBi extrapolates well by design.

**Character-level LM** — the model predicts the next character (not word or subword). At this scale (2 layers, 128-dim), outputs are not semantically coherent but do learn word shapes and common patterns.

---

## Future Work

The following were scoped out due to computational constraints (CPU-only training):

- Context lengths beyond 64 (512, 1024, 2048) for the main comparison runs
- GPU memory profiling
- Throughput scaling study across context lengths
- Mechanistic interpretability (SAE, neuron ablation)

---

## Assignment Coverage

| Task | Status | Where |
|---|---|---|
| Part 1: Standard Transformer baseline | ✅ | `attention.py`, `train.py` Part 1 |
| Part 2: 3 attention variants | ✅ | `attention.py` — Sliding Window, Linear, MQA |
| Part 2: Perplexity reported | ✅ | `attention_results.txt`, `readings.txt` |
| Part 3: RoPE, ALiBi, Relative PE | ✅ | `positional.py` |
| Part 3c: Extrapolation test | ✅ | `positional_results.txt`, `readings.txt` |
| Part 4: Conv+Attention hybrid (2 designs) | ✅ | `model.py` — ConvBlock, ConvAttentionBlock |
| Part 5: Comparative results table | ✅ | `readings.txt`, `results/results.csv` |
| Generated samples | ✅ | `generated_samples.txt` |
| Hyperparameter documentation | ✅ | `config_summary.txt`, `config.py` |
