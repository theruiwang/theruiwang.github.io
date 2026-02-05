# Understanding Vision-Language Models: A Deep Dive into Qwen3-VL

## Table of Contents

```
PART 1: TRANSFORMER BUILDING BLOCKS
  1.1 Attention Mechanisms
  1.2 Position Encoding
  1.3 Normalization
  1.4 Feed-Forward Networks
  1.5 Vision Transformers (ViT)
  1.6 Vision-Language Model Architectures

PART 2: QWEN3-VL ARCHITECTURE
  2.0 Overview
  ── Vision Encoder ──
  2.1 Patch Embedding
  2.2 Position Encoding
  2.3 Vision Transformer Blocks
  2.4 Multi-Level Feature Extraction (DeepStack)
  ── Vision-Language Fusion ──
  2.5 Spatial Merging
  2.6 Sequence Construction
  2.7 M-RoPE Position Assignment
  2.8 Interleaved Frequency Allocation
  ── Text Decoder ──
  2.9 DeepStack Injection
  2.10 Decoder Architecture
  ── Video ──
  2.11 Video Processing
  ── Synthesis ──
  2.12 End-to-End Forward Pass

REFERENCES
```

---

This blog has two halves: **Part 1** covers transformer fundamentals, **Part 2** dives into Qwen3-VL specifics. Skip Part 1 if you're already familiar with attention, GQA, M-RoPE, RMSNorm, SwiGLU, and ViT.

---

# Part 1: Transformer Building Blocks

This section covers the foundational components used in modern transformers. We explain each concept once here; later sections reference back when showing how Qwen3-VL uses them.

---

## 1.1 The Attention Mechanism

The attention mechanism [1] is the core innovation behind modern transformers.

### What Problem Does Attention Solve?

Consider the word "bank" in these two sentences:
- "The river **bank** was covered in wildflowers."
- "The **bank** raised interest rates yesterday."

A simple word embedding maps "bank" to the same vector in both cases—it has no way to know which meaning is intended. The embedding is **context-independent**.

Attention allows each token to "look at" other tokens in the sequence to understand its role. In sentence 1, "bank" attends to "river" → riverbank meaning. In sentence 2, it attends to "interest rates" → financial institution.

### The Attention Computation

**The scaled dot-product attention formula:**

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

**Step-by-step:**

```
Input: X ∈ ℝ^(N × d)    # N tokens, each d-dimensional

Step 1: Project to Query, Key, Value
  Q = X @ W_Q    # (N, d_k)
  K = X @ W_K    # (N, d_k)
  V = X @ W_V    # (N, d_v)

Step 2: Compute attention scores
  scores = Q @ K^T / sqrt(d_k)    # (N, N)

Step 3: Normalize to get attention weights
  weights = softmax(scores, dim=-1)    # (N, N), rows sum to 1

Step 4: Weighted combination of values
  output = weights @ V    # (N, d_v)
```

**Why divide by $\sqrt{d_k}$?** The dot product $\mathbf{Q} \cdot \mathbf{K}$ sums $d_k$ terms. Without scaling, variance grows with dimension, causing softmax to saturate (outputs near 0 or 1, gradients vanish). Dividing by $\sqrt{d_k}$ normalizes variance to $\approx 1$ regardless of dimension, keeping softmax in its sensitive range.

In code, attention is straightforward:

```python
def attention(Q, K, V, mask=None):
    d_k = K.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # (N, N)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = F.softmax(scores, dim=-1)  # Normalize rows
    return weights @ V  # Weighted combination
```

### Causal Masking

During text generation, a token should only attend to **previous** tokens—not future ones. We enforce this with a **causal mask**: add $-\infty$ to future positions before softmax, making those attention weights effectively 0.

### Self-Attention vs Cross-Attention

**Self-attention:** Q, K, V all come from the same sequence.

**Cross-attention:** Q from one sequence, K/V from another.

```
Example: Translating "The cat sat" → "Le chat assis"

When generating "chat" (French for "cat"):
  Q comes from:  "Le chat"      (decoder—what we're generating)
  K comes from:  "The cat sat"  (encoder—source sentence)
  V comes from:  "The cat sat"  (encoder—source sentence)

The French word "chat" queries the English sentence to find relevant
context. It attends strongly to "cat" (K matches Q), then pulls
information from "cat"'s V to inform the translation.
```

### Multi-Head Attention

A single attention head computes one attention distribution per position—but a token often needs to attend to multiple things simultaneously. Consider "The cat sat on the mat because it was tired." The word "it" needs to resolve its antecedent ("cat") AND connect to its predicate ("was tired"). One attention pattern can't express both relationships well.

Multi-head attention runs multiple attention functions in parallel, each with its own learned projections $\mathbf{W}_Q^i, \mathbf{W}_K^i, \mathbf{W}_V^i$:

$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\mathbf{W}^O$$

**The dimension split:** Multi-head attention splits the model dimension across heads rather than adding computation:

```
d = 512, h = 8 heads → d_head = 64

Single-head: one (N×N) attention with 512-dim dot products
Multi-head:  eight (N×N) attentions with 64-dim dot products each
             8 × 64 = 512, so total work is identical
```

The output projection W_O combines information from all heads.

### The KV Cache

LLM inference has two phases:

**Prefill:** Process the entire prompt in parallel. GPU-bound (large matrix multiplications).

**Generation:** Produce one token at a time. Each new token must attend to ALL previous tokens, requiring their K and V vectors.

```
Generating token 5:
  output₄ = softmax(Q₄ · [K₀, K₁, K₂, K₃, K₄]) · [V₀, V₁, V₂, V₃, V₄]
                    └─────────────────────────┘
                     Need ALL previous K,V vectors!
```

**The KV cache** stores K,V from previous tokens to avoid recomputation:

```
Without cache: recompute K₀,V₀,K₁,V₁,... at every step → O(N²) total
With cache:    compute only new K,V, reuse cached     → O(N) total
```

**Memory requirements for Qwen3-VL-32B:**

```
KV cache per token:
  = 2 (K and V)
  × 64 (layers)
  × 8 (KV heads)
  × 128 (head dimension)
  × 2 (bytes, assuming fp16)
  = 262,144 bytes ≈ 256 KB

For 256K context: 256,000 × 256 KB ≈ 64 GB per sequence
```

This is why KV cache optimization is critical—a single long-context request consumes most of an H100's 80GB.

### Grouped Query Attention (GQA)

GQA [2] reduces KV cache by sharing K,V heads across multiple Q heads:

```
Multi-Head Attention (MHA): each Q head has its own KV pair
┌────────────────────────────────────────────────────────┐
│  Q₀ → K₀,V₀    Q₁ → K₁,V₁    Q₂ → K₂,V₂   ...  Q₆₃ → K₆₃,V₆₃  │
└────────────────────────────────────────────────────────┘
  64 KV pairs total

Multi-Query Attention (MQA): ALL Q heads share one KV pair
┌────────────────────────────────────────────────────────┐
│  Q₀ ─┐                                                 │
│  Q₁ ─┤                                                 │
│  Q₂ ─┼────→ K₀,V₀                                      │
│  ... │                                                 │
│  Q₆₃─┘                                                 │
└────────────────────────────────────────────────────────┘
  1 KV pair total (maximum compression, but quality loss)

Grouped Query Attention (GQA): groups of Q heads share KV pairs
┌────────────────────────────────────────────────────────┐
│  Q₀ ─┐          Q₈ ─┐          Q₁₆─┐                   │
│  Q₁ ─┤          Q₉ ─┤          Q₁₇─┤                   │
│  ... ├→ K₀,V₀   ... ├→ K₁,V₁   ... ├→ K₂,V₂    ...    │
│  Q₇ ─┘          Q₁₅─┘          Q₂₃─┘                   │
└────────────────────────────────────────────────────────┘
  8 KV pairs total (8× compression, quality close to MHA)
```

**In Qwen3-VL:** GQA with 8 KV heads shared across 64 Q heads reduces KV cache by 8× while maintaining quality close to full MHA.

### Self-Check Questions

**Q1.1.1:** Why do we need separate K and V matrices?

<details>
<summary>Answer</summary>

K determines *which tokens are relevant* (via Q·K similarity), while V determines *what information to extract*. A token might be highly relevant (high K match) but contribute different information depending on context. Separating them lets the model learn "what to attend to" independently from "what to retrieve."
</details>

**Q1.1.2:** Why does attention have O(N²) complexity?

<details>
<summary>Answer</summary>

The attention score matrix has shape (N, N)—every token computes a score with every other token. Computing Q @ K^T requires N² dot products.
</details>

**Q1.1.3:** In GQA with 64 Q heads and 8 KV heads, do the 8 Q heads sharing a KV pair compute the same attention weights?

<details>
<summary>Answer</summary>

No. Each Q head has its own W_Q projection, producing different queries. They attend to the same K,V but with different queries, producing different attention patterns.
</details>

---

## 1.2 Position Encoding

### Why Position Encoding is Necessary

Attention treats input as a **set**, not a sequence. If you shuffle input tokens, outputs shuffle identically—attention is permutation equivariant. But position matters: "Dog bites man" ≠ "Man bites dog." We must inject position information explicitly.

**Early approaches:**

- **Sinusoidal** [1]: The original Transformer added fixed sin/cos waves of different frequencies to embeddings. Simple but encodes absolute position—doesn't generalize well to longer sequences than seen during training.
- **Learned**: Train a position embedding matrix. Flexible but also absolute, and requires fixed maximum length.

Both encode absolute position. Modern LLMs prefer relative position encoding—what matters is how far apart tokens are, not their absolute indices.

### Rotary Position Embedding (RoPE)

RoPE [3] achieves relative position encoding by rotating Q and K vectors by position-dependent angles. When computing $\mathbf{Q} \cdot \mathbf{K}$, the rotations combine such that the result depends only on the difference between positions.

**Important:** RoPE is applied only to Q and K—not to V or token embeddings. Position information should influence which tokens to attend to (via $\mathbf{Q} \cdot \mathbf{K}$), not what content to extract (V).

**The mechanics:** In real implementations, dimension $i$ is paired with dimension $i + d/2$, and each pair is rotated by an angle proportional to position. For position $p$, rotate pair $i$ by angle $p\theta_i$:

$$\begin{bmatrix} x'_i \\ x'_{i+d/2} \end{bmatrix} = \begin{bmatrix} \cos(p\theta_i) & -\sin(p\theta_i) \\ \sin(p\theta_i) & \cos(p\theta_i) \end{bmatrix} \begin{bmatrix} x_i \\ x_{i+d/2} \end{bmatrix}$$

This pairing operates on contiguous halves of the vector, making it GPU-friendly.

**Why does this give relative position?** When you dot two 2D vectors rotated by angles $\alpha$ and $\beta$:

$$\mathbf{a}_\alpha \cdot \mathbf{b}_\beta = \|\mathbf{a}\| \|\mathbf{b}\| \cos(\alpha - \beta)$$

The angles subtract. So Q at position $m$ (rotated by $m\theta$) dotted with K at position $n$ (rotated by $n\theta$) gives attention depending on $(m-n)\theta$—purely relative. For example, tokens at positions 10 and 12 produce the same attention as tokens at positions 1000 and 1002: both depend on $-2\theta$.

**Multiple frequencies** let the model distinguish both nearby and distant positions:

$$\theta_i = b^{-2i/d}$$

- **Low-index pairs** ($\theta_0, \theta_1, \ldots$): high frequency, fast rotation → distinguish nearby positions
- **High-index pairs** ($\ldots, \theta_{d/2-1}$): low frequency, slow rotation → distinguish distant positions

### 2D RoPE for Images

Standard RoPE handles 1D sequences. But images have two spatial dimensions—a patch's position is (row, column), not a single index.

**The challenge:** Consider a 4×4 grid of patches. Each patch has a 2D position:

```
Patch positions (row, col):
┌───────┬───────┬───────┬───────┐
│ (0,0) │ (0,1) │ (0,2) │ (0,3) │
├───────┼───────┼───────┼───────┤
│ (1,0) │ (1,1) │ (1,2) │ (1,3) │
├───────┼───────┼───────┼───────┤
│ (2,0) │ (2,1) │ (2,2) │ (2,3) │
├───────┼───────┼───────┼───────┤
│ (3,0) │ (3,1) │ (3,2) │ (3,3) │
└───────┴───────┴───────┴───────┘

Flattened to a sequence (row-major order):
Index:    0     1     2     3     4     5     6    ...   15
Position: (0,0) (0,1) (0,2) (0,3) (1,0) (1,1) (1,2) ... (3,3)
```

If we used 1D RoPE with indices 0–15, patches (0,3) and (1,0) would appear adjacent (indices 3 and 4), but they're actually in different rows. We need position encoding that respects the 2D structure.

**The solution:** Split the head dimensions between row and column, applying separate rotations to each:

```
Head dim = 72 (vision encoder), 18 frequency components per axis

For patch at (row=r, col=c):
  Dims 0-17  ↔ Dims 36-53:  paired, rotated by r × θᵢ  (row)
  Dims 18-35 ↔ Dims 54-71:  paired, rotated by c × θᵢ  (column)
```

All 72 dimensions participate in rotation—half encode row position, half encode column position. This way, the attention score between two patches reflects both their row distance and column distance.

```
                    rotary_pos_emb (36 values)
                    [θ_h₀..θ_h₁₇, θ_w₀..θ_w₁₇]
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
        ┌───────────────────────────────────────────┐
  emb = │ θ_h₀..θ_h₁₇, θ_w₀..θ_w₁₇ │ θ_h₀..θ_h₁₇, θ_w₀..θ_w₁₇ │  (72 values)
        └───────────────────────────────────────────┘
              dims 0-35                dims 36-71
                   │                        │
                   └──── same angles ───────┘
                              │
                              ▼
              pairs (i, i+36) are rotated correctly
```

### 3D M-RoPE for Multimodal Sequences

VLMs process sequences mixing text, images, and video. Text is 1D, images are 2D, video is 3D (frames × height × width). How do we unify these?

Qwen3-VL uses M-RoPE with three position dimensions:

```
T (Temporal): sequence position / frame index
H (Height):   row in image grid
W (Width):    column in image grid
```

**Position assignment example: "Describe [IMAGE] now"**

```
Token      │  T   │  H   │  W   │  Notes
───────────┼──────┼──────┼──────┼────────────────
"Describe" │  0   │  0   │  0   │  Text: T=H=W
[img₀₀]    │  1   │  0   │  0   │  Image patch (0,0)
[img₀₁]    │  1   │  0   │  1   │  Image patch (0,1) ← same T
[img₁₀]    │  1   │  1   │  0   │  Image patch (1,0) ← same T
[img₁₁]    │  1   │  1   │  1   │  Image patch (1,1) ← same T
"now"      │  2   │  2   │  2   │  Text: T=H=W
```

All image patches share T=1 but differ in H,W. Text tokens have T=H=W (collapsing to 1D).

**Dimension allocation:**

```
Head dim = 128
M-RoPE sections: 24 pairs (T) + 20 pairs (H) + 20 pairs (W) = 64 pairs = 128 dims

T gets more pairs because text sequences can reach 256K tokens.
Image grids are bounded (~14×14 after merging).
```

### Interleaved M-RoPE

Recall that different frequency bands serve different purposes: high frequencies distinguish nearby positions, low frequencies distinguish distant positions.

Qwen2-VL's original M-RoPE partitioned dimensions into contiguous blocks—T gets dimensions 0-47, H gets 48-87, W gets 88-127. This creates an imbalanced frequency spectrum: each axis only accesses a narrow frequency range. Studies showed this degrades performance on long-video understanding, where the temporal axis needs low frequencies to distinguish distant frames.

Qwen3-VL fixes this by interleaving T, H, W across all frequency bands:

```
T: θ₀, θ₃, θ₆, ..., θ₆₃  (every 3rd frequency, spanning full range)
H: θ₁, θ₄, θ₇, ...
W: θ₂, θ₅, θ₈, ...
```

Each axis now uniformly spans both low and high frequencies, creating a balanced spectrum that significantly improves long-range positional modeling for video.

Visually, the difference is clear:

```
Frequency spectrum:  HIGH ←――――――――――――――――――→ LOW

Contiguous:
  T: ████████░░░░░░░░░░░░░░░░  (only fast rotation)
  H: ░░░░░░░░████████░░░░░░░░  (only medium)
  W: ░░░░░░░░░░░░░░░░████████  (only slow rotation)

Interleaved:
  T: █░░█░░█░░█░░█░░█░░█░░█░░  (full range sampled)
  H: ░█░░█░░█░░█░░█░░█░░█░░█░
  W: ░░█░░█░░█░░█░░█░░█░░█░░█
```

### Self-Check Questions

**Q1.2.1:** In RoPE, why do we rotate Q and K but not V?

<details>
<summary>Answer</summary>

Position is needed to determine *which* tokens to attend to (Q·K), not *what* information to extract (V). If we rotated V, the output representation would depend on absolute positions of attended tokens, breaking translation equivariance.
</details>

**Q1.2.2:** For 2D RoPE with head_dim=72, how many dimension pairs encode row position?

<details>
<summary>Answer</summary>

18 pairs. Head dim 72 uses 36 frequency components split equally: 18 for row, 18 for column. Each frequency component corresponds to one 2D rotation (pairing dimension $i$ with $i+36$).
</details>

---

## 1.3 Normalization

### Why Normalize?

Deep networks suffer from **internal covariate shift**: each layer's input distribution changes as earlier layers update. Without normalization, activations can explode or vanish, making training unstable.

### LayerNorm vs RMSNorm

**LayerNorm** (original transformer) normalizes by subtracting mean and dividing by standard deviation:

$$\text{LayerNorm}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \mu}{\sigma} + \boldsymbol{\beta}$$

where $\mu = \frac{1}{d}\sum_{i=1}^{d} x_i$ and $\sigma = \sqrt{\frac{1}{d}\sum_{i=1}^{d} (x_i - \mu)^2 + \epsilon}$

**RMSNorm** (used by Qwen3-VL) simplifies this by dropping mean-centering:

$$\text{RMSNorm}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})}, \quad \text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}$$

**Why RMSNorm wins:**
- ~15% faster (no mean computation, no bias parameter)
- Fewer parameters (γ only, no β)
- Equivalent quality for large models

The real `Qwen3VLTextRMSNorm` implementation:

```python
class Qwen3VLTextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # γ only, no β
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)  # Compute in fp32 for stability
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
```

Note the cast to `float32` for numerical stability—this is important when training in mixed precision.

### Pre-Norm vs Post-Norm

The original Transformer used post-norm—normalize after the residual:

$$\mathbf{y} = \text{Norm}(\mathbf{x} + f(\mathbf{x}))$$

Modern models use pre-norm—normalize before the sublayer:

$$\mathbf{y} = \mathbf{x} + f(\text{Norm}(\mathbf{x}))$$

**Why pre-norm is often more stable:** In pre-norm, the gradient $\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \mathbf{I} + \frac{\partial f}{\partial \text{Norm}} \frac{\partial \text{Norm}}{\partial \mathbf{x}}$. The identity term guarantees gradients flow directly to earlier layers, regardless of what $f$ or Norm does. In post-norm, $\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \frac{\partial \text{Norm}}{\partial \mathbf{x}} (\mathbf{I} + \frac{\partial f}{\partial \mathbf{x}})$—the entire gradient passes through normalization, which can attenuate it in deep networks.

**Qwen3-VL uses pre-norm throughout.**

### QK-Norm

During training, Q and K vector magnitudes can grow unboundedly. When $\|\mathbf{Q}\|$ and $\|\mathbf{K}\|$ are large, attention scores $\mathbf{Q} \cdot \mathbf{K}$ become extreme, causing softmax to saturate.

**QK-Norm** applies RMSNorm to Q and K *per-head* before computing attention:

$$\mathbf{Q}_{\text{norm}} = \text{RMSNorm}(\mathbf{Q}), \quad \mathbf{K}_{\text{norm}} = \text{RMSNorm}(\mathbf{K})$$

$$\text{scores} = \frac{\mathbf{Q}_{\text{norm}} \mathbf{K}_{\text{norm}}^\top}{\sqrt{d_k}}$$

After normalization, $\|\mathbf{Q}_{\text{norm}}\| \approx \sqrt{d}$ and $\|\mathbf{K}_{\text{norm}}\| \approx \sqrt{d}$.

**Why both QK-Norm and $\sqrt{d_k}$?** They solve different problems:

| Problem | Solution |
|---------|----------|
| Vector magnitudes grow during training | QK-Norm bounds $\|\mathbf{Q}\|$ and $\|\mathbf{K}\|$ |
| Dot product variance scales with dimension | $\sqrt{d_k}$ normalizes regardless of $d_k$ |

After QK-Norm, vectors have controlled magnitude, but the dot product is still a sum of $d_k$ terms. Both are needed.

### Self-Check Questions

**Q1.3.1:** RMSNorm doesn't subtract the mean. Why doesn't this cause problems?

<details>
<summary>Answer</summary>

With learned scale γ and model flexibility, activations can learn to be centered anyway. Empirically, re-centering doesn't benefit large models, so RMSNorm drops it for efficiency.
</details>

**Q1.3.2:** If a vector has values [10, 10, 10, 10], what does RMSNorm output (ignoring γ)?

<details>
<summary>Answer</summary>

RMS = sqrt((100 + 100 + 100 + 100) / 4) = sqrt(100) = 10
Output = [10/10, 10/10, 10/10, 10/10] = [1, 1, 1, 1]

RMSNorm scales vectors to have RMS = 1.
</details>

---

## 1.4 Feed-Forward Networks

### The Role of FFN in Transformers

Each transformer layer has two sub-layers: attention and FFN. While attention handles token-to-token interaction, the FFN processes each token independently, adding non-linearity and capacity.

- **Attention:** "What information should I gather from other tokens?"
- **FFN:** "How should I transform this token's representation?"

Research suggests FFNs act as key-value memories, storing factual knowledge learned during training.

### From Standard MLP to SwiGLU

SwiGLU [4] improves on the standard MLP.

**Standard MLP** (original transformer):

$$\text{MLP}(\mathbf{x}) = \mathbf{W}_2 \, \text{ReLU}(\mathbf{W}_1 \mathbf{x})$$

Two matrices, one activation. Simple but limited.

**Gated Linear Unit (GLU)** introduces a learned gate:

$$\text{GLU}(\mathbf{x}) = \sigma(\mathbf{W}_{\text{gate}} \mathbf{x}) \odot (\mathbf{W}_{\text{up}} \mathbf{x})$$

The gate σ(W_gate · x) decides *how much* of each dimension to let through. This separates "what to filter" from "what to transform."

**SwiGLU** (used by Qwen3-VL) replaces sigmoid with SiLU:

$$\text{SwiGLU}(\mathbf{x}) = \mathbf{W}_{\text{down}} \left( \text{SiLU}(\mathbf{W}_{\text{gate}} \mathbf{x}) \odot (\mathbf{W}_{\text{up}} \mathbf{x}) \right)$$

### Step-by-Step SwiGLU Computation

```
Input: x ∈ ℝ^5120

Step 1: Compute gate signal
  gate_pre = W_gate @ x          # (5120,) → (25600,)
  gate = SiLU(gate_pre)          # element-wise: x * sigmoid(x)

Step 2: Compute value signal
  value = W_up @ x               # (5120,) → (25600,)

Step 3: Gated combination
  hidden = gate ⊙ value          # element-wise multiply (25600,)

Step 4: Project back
  output = W_down @ hidden       # (25600,) → (5120,)
```

### Why SiLU over Sigmoid for Gating?

For gating, large positive values mean "let this through strongly." With sigmoid gating, once the gate is "fully open" (σ ≈ 1), it can't learn to open further—gradients vanish. SiLU avoids this: SiLU(x) ≈ x for large x, so gradients flow freely.

The critical insight: at large positive inputs, sigmoid saturates (gradient → 0) but SiLU doesn't (gradient → 1).

| Input x | Sigmoid σ(x) | σ'(x) | SiLU(x) | SiLU'(x) |
|---------|--------------|-------|---------|----------|
| -5 | 0.007 | 0.007 | -0.03 | 0.03 |
| 0 | 0.5 | 0.25 | 0 | 0.5 |
| +5 | 0.993 | **0.007** ❌ | 4.97 | **1.03** ✓ |

The real `Qwen3VLTextMLP` implementation:

```python
class Qwen3VLTextMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size              # 5120
        self.intermediate_size = config.intermediate_size  # 25600
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]  # SiLU

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
```

The forward pass is a single expression: `down(SiLU(gate(x)) * up(x))`.

---

## 1.5 Vision Transformers (ViT)

Vision Transformers (ViT) [5] apply the transformer architecture to images.

### Why Not Process Pixels Directly?

Self-attention has O(N²) complexity. Even a small 224×224 image has 50,176 pixels:

```
Pixels: 50,176² = 2.5 billion attention pairs
16×16 patches: 196² = 38,416 attention pairs
```

For a modern 1920×1080 image (~2 million pixels), pixel-level attention is completely infeasible:

```
Pixels: 2M² = 4 trillion attention pairs
16×16 patches: ~8,000² = 64 million attention pairs
```

Patches reduce computation by ~60,000× while preserving most visual information.

### From Images to Sequences

```
Input Image: 224 × 224 × 3
         │
         ▼
┌─────────────────────────────────────────┐
│ 1. PATCH EXTRACTION                      │
│    Divide into 16×16 patches             │
│    224/16 = 14 patches per side          │
│    14 × 14 = 196 patches total           │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ 2. LINEAR PROJECTION                     │
│    Flatten each patch: 16×16×3 = 768     │
│    Project to model dimension D          │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ 3. ADD POSITION EMBEDDINGS               │
│    Patches have no inherent order—       │
│    position must be explicitly added     │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ 4. TRANSFORMER ENCODER                   │
│    L layers of bidirectional attention   │
│    Every patch can attend to every other │
└─────────────────────────────────────────┘
         │
         ▼
Output: 196 contextualized patch representations
```

### What Does Each Stage Learn?

- **Early layers (1-8):** Local features—edges, textures, colors. Patches mostly attend to themselves and neighbors.

- **Middle layers (9-20):** Compositional features—parts, shapes, local patterns. Patches start attending to related distant patches.

- **Late layers (21-27):** Semantic features—objects, regions, scene understanding. Global attention patterns emerge.

### Bidirectional vs Causal Attention

Vision encoders use **bidirectional** attention—every patch sees every other patch:

```
Why bidirectional for vision?
- The entire image exists at once (no "future" to hide)
- Understanding the top-left requires knowing the bottom-right
- Global context helps: "This brown patch is a dog because
  there's a leash connected to it across the image"

Why causal for text generation?
- Tokens are generated sequentially
- Can't condition on tokens not yet produced
```

---

## 1.6 Vision-Language Model Architectures

A VLM must combine two very different modalities [6, 7]:
- **Vision**: Dense, continuous pixel values; spatial structure matters
- **Language**: Discrete tokens; sequential structure matters

How do we get them to "talk" to each other? There are multiple approaches (e.g., cross-attention), but for the sake of incompleteness, we'll only cover **early fusion**—the approach Qwen3-VL uses.

### Early Fusion

```
Vision Encoder ──► Project ──► Visual Tokens ──┐
                                               ├──► [vis₁ vis₂ ... text₁ text₂ ...]
Text ──► Tokenize ──► Embed ──► Text Tokens ───┘
                                               │
                                               ▼
                                        Unified Sequence
                                               │
                                               ▼
                                    Standard LLM Decoder
                                    (no architectural changes)
```

**How it works:** Project visual features to the same dimension as text embeddings. Concatenate them into one sequence. Process with a standard decoder-only LLM—no architectural changes needed.

Visual tokens are treated as "just more tokens" in the sequence. Self-attention naturally handles vision-language interaction. With this foundation, let's dive into how Qwen3-VL implements this.

---

# Part 2: Qwen3-VL Architecture

## 2.0 Overview

Qwen3-VL [10] implements early fusion: visual features are projected into the text embedding space, then processed by a standard autoregressive decoder. In this section, we'll follow the data as it flows through the model—from raw pixels to generated text.

```
                                 QWEN3-VL
┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│  Image/Video                                                           │
│       │                                                                │
│       ▼                                                                │
│  ┌─────────────────┐                                                   │
│  │  VISION ENCODER │──────────────────────────────────┐                │
│  │                 │                                  │                │
│  │  27 ViT layers  │    DeepStack extraction          │                │
│  │  1152-dim       │    at layers 8, 16, 24           │                │
│  │  ~600M params   │              │                   │                │
│  └────────┬────────┘              │                   │                │
│           │                       │                   │                │
│           ▼                       ▼                   │                │
│  ┌─────────────────┐    ┌─────────────────┐          │                │
│  │  Spatial Merge  │    │  Spatial Merge  │ ×3       │                │
│  │  2×2 → 1        │    │  (DeepStack)    │          │                │
│  │  1152 → 5120    │    │                 │          │                │
│  └────────┬────────┘    └────────┬────────┘          │                │
│           │                      │                   │                │
│           ▼                      ▼                   │                │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                        TEXT DECODER                              │  │
│  │                                                                  │  │
│  │  Layer 0: hidden += DeepStack[0]  (from vision layer 8)         │  │
│  │  Layer 1: hidden += DeepStack[1]  (from vision layer 16)        │  │
│  │  Layer 2: hidden += DeepStack[2]  (from vision layer 24)        │  │
│  │  Layers 3-63: standard processing                                │  │
│  │                                                                  │  │
│  │  64 layers, 5120-dim, GQA (64 Q / 8 KV heads), SwiGLU           │  │
│  │  ~32B params                                                     │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                      │                                 │
│                                      ▼                                 │
│                                   Output                               │
└────────────────────────────────────────────────────────────────────────┘
```

**Key architectural innovations:**

| Component | Design Choice | Benefit |
|-----------|--------------|---------|
| Feature extraction | DeepStack: multi-level features from layers 8, 16, 24, 27 | Preserves fine-grained details lost in deep abstraction |
| Position encoding | Interleaved M-RoPE: T, H, W frequencies distributed across spectrum | Each spatial axis can resolve both local and global positions |
| Video timestamps | Explicit tokens `<t0>`, `<t1>`, ... | Enables natural temporal references in Q&A |

We'll examine each component in the order data flows through the model: first the vision encoder (Sections 2.1–2.4), then the fusion mechanism that connects vision to language (Sections 2.5–2.8), and finally the text decoder (Sections 2.9–2.10). Section 2.11 covers video-specific processing, and Section 2.12 synthesizes everything in a complete forward pass.

---

## The Vision Encoder

The vision encoder transforms raw pixels into a sequence of embeddings that the language model can process. This involves three stages: converting pixels to patches (2.1), adding position information (2.2), and processing through transformer layers (2.3). Along the way, Qwen3-VL extracts features at multiple depths (2.4) to preserve information that would otherwise be lost in deep abstraction.

### 2.1 Patch Embedding

**The design challenge:** Qwen3-VL must handle both images and videos with the same architecture. Videos naturally have a temporal dimension (frames over time), while images don't. Rather than maintaining separate pathways, Qwen3-VL uses a unified 3D convolution that treats images as a special case of video.

**The solution:** A 3D convolution with kernel size $(2, 16, 16)$—that is, 2 frames × 16 pixels × 16 pixels. This processes spatiotemporal patches: regions of $16 \times 16$ pixels across 2 consecutive frames.

But images have only one "frame." To make images compatible with this 3D pipeline, each image is **duplicated** along the temporal axis, creating a 2-frame "video" where both frames are identical.

```
Image (H×W×3)                        Video (T×H×W×3)
       │                                    │
       ▼                                    ▼
  Duplicate to 2 frames               Group into pairs of frames
  [frame, frame]                      [(f0,f1), (f2,f3), ...]
       │                                    │
       ▼                                    ▼
┌──────────────────────────────────────────────────────────────┐
│         Conv3D: kernel (2, 16, 16), stride (2, 16, 16)       │
│                                                              │
│  Each 2×16×16 spatiotemporal region → one 1152-dim embedding │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
  Patch embeddings (1152-dim each)
```

For a 224×224 image:
- Spatial grid: $\frac{224}{16} \times \frac{224}{16} = 14 \times 14 = 196$ patches
- Temporal: duplicated image (2 frames) is collapsed by the temporal stride of 2 → still 196 patches

For a 60-frame video at 224×224:
- Temporal groups: $\frac{60}{2} = 30$ groups of 2 frames each
- Spatial grid per group: $14 \times 14 = 196$ patches
- Total: $30 \times 196 = 5880$ patches

The core of the patch embedding is a 3D convolution where kernel and stride are both `(temporal_patch_size, patch_size, patch_size)`:

```python
# From modeling_qwen3_vl.py
class Qwen3VLVisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        kernel_size = [config.temporal_patch_size, config.patch_size, config.patch_size]
        self.proj = nn.Conv3d(
            config.in_channels, config.hidden_size,
            kernel_size=kernel_size, stride=kernel_size, bias=True
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states).view(-1, self.embed_dim)
        return hidden_states
```

**Why duplicate images instead of zero-padding?** Zero-padding would create artificial boundaries between "real" pixels and empty space. The convolution would learn to detect these padding artifacts rather than focusing on actual visual features. Duplication ensures the temporal statistics are uniform—the convolution sees consistent content across both frames.

**Self-Check Questions**

**Q2.1.1:** A 448×448 image produces how many patches before spatial merging?

<details>
<summary>Answer</summary>

$(448/16)^2 = 28^2 = 784$ patches.
</details>

**Q2.1.2:** The Conv3D kernel is $(2, 16, 16)$. What is the receptive field in pixels for one output embedding?

<details>
<summary>Answer</summary>

$2 \times 16 \times 16 = 512$ pixels per channel, or $512 \times 3 = 1536$ values total. For images (where the 2 frames are duplicates), this effectively covers a $16 \times 16$ spatial region.
</details>

---

### 2.2 Position Encoding

Patch embeddings alone carry no position information—the model wouldn't know whether a patch came from the top-left or bottom-right of the image. The vision encoder addresses this with two complementary mechanisms.

**1. Learnable absolute embeddings** — added once after patch embedding:

$$\mathbf{h}_i = \mathbf{h}_i + \mathbf{E}_{\text{pos}}[p_i]$$

where $p_i$ indexes into a learned embedding table of size $48 \times 48 = 2304$ (supporting images up to $48 \times 16 = 768$ pixels per side before resizing).

**2. 2D RoPE** — applied to Q, K in every attention layer.

```
Position encoding in vision encoder:

  Learnable Embedding           2D RoPE (per layer)
  ──────────────────           ──────────────────────
  Added once after             Applied in each of 27
  patch embedding              attention layers to Q, K

  Provides global position     Encodes relative spatial
  information                  relationships
```

Why both? Learnable embeddings capture absolute position biases (e.g., "objects tend to be centered"). RoPE captures relative relationships (e.g., "these patches are 3 apart horizontally"), which generalize better to different image sizes. Together they provide richer positional information than either alone.

For 2D RoPE, the head dimension (72) is split between row and column rotations, as described in Section 1.2. Each axis gets 18 frequency components:

$$\theta_i^{(\text{row})} = b^{-2i/36}, \quad \theta_i^{(\text{col})} = b^{-2i/36}, \quad i \in \{0, \ldots, 17\}$$

where $b = 10000$ is the base frequency.

The vision encoder's rotary embedding computes inverse frequencies and generates position-dependent rotation angles:

```python
# From modeling_qwen3_vl.py
class Qwen3VLVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs
```

**Self-Check Questions**

**Q2.2.1:** Why use both learnable embeddings and RoPE, rather than just one?

<details>
<summary>Answer</summary>

They serve complementary roles. Learnable embeddings capture absolute position preferences (e.g., "center patches are often more important"). RoPE encodes relative distances, enabling the model to recognize that patches 2 apart have the same spatial relationship regardless of absolute position. Using both provides richer positional information than either alone.
</details>

**Q2.2.2:** The vision encoder has head_dim=72. How many dimension pairs encode row position?

<details>
<summary>Answer</summary>

18 pairs. RoPE pairs dimension $i$ with dimension $i + 36$. Half the 36 frequency components encode row, half encode column: $18 + 18 = 36$ components, corresponding to 36 dimension pairs (covering all 72 dimensions).
</details>

---

### 2.3 Vision Transformer Blocks

The patch embeddings now pass through 27 transformer layers. Each layer refines the representations through self-attention (patches exchange information) and an MLP (each patch is transformed independently).

```
Input x
    │
    ├───────────────────────────────┐
    ▼                               │
LayerNorm                           │
    │                               │
    ▼                               │
Self-Attention (16 heads, dim=72)   │  Residual
    │                               │
    ▼                               │
    + ◄─────────────────────────────┘
    │
    ├───────────────────────────────┐
    ▼                               │
LayerNorm                           │
    │                               │
    ▼                               │
MLP (1152 → 4304 → 1152, GELU)      │  Residual
    │                               │
    ▼                               │
    + ◄─────────────────────────────┘
    │
    ▼
Output
```

**What happens across 27 layers?** The representations evolve from low-level to high-level:

| Layers | What's captured | Attention pattern |
|--------|-----------------|-------------------|
| 1–8 | Edges, textures, color gradients | Mostly local—patches attend to neighbors |
| 9–16 | Shapes, parts, local object structure | Expanding—related regions start connecting |
| 17–24 | Objects, region boundaries | Semi-global—semantic groupings emerge |
| 25–27 | Scene understanding, object relationships | Global—long-range dependencies |

The vision encoder uses **bidirectional attention**—every patch can attend to every other patch. Unlike text generation (where future tokens must be hidden), the entire image exists at once. A patch in the top-left may need information from the bottom-right to understand context (e.g., "this brown blob is part of a dog because there's a leash connecting to a person across the image").

**Why GELU instead of SwiGLU?** The vision encoder prioritizes efficiency over capacity. SwiGLU requires three weight matrices ($\mathbf{W}_{\text{gate}}, \mathbf{W}_{\text{up}}, \mathbf{W}_{\text{down}}$) while standard MLP uses two. Since the 64-layer text decoder handles the heavy reasoning, the 27-layer vision encoder can use the simpler activation without sacrificing quality.

The vision block implements standard pre-norm residual connections:

```python
# From modeling_qwen3_vl.py
class Qwen3VLVisionBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(config=config)
        self.mlp = Qwen3VLVisionMLP(config=config)

    def forward(self, hidden_states, cu_seqlens, position_embeddings, **kwargs):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings, **kwargs
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states
```

The MLP uses a simple GELU activation:

```python
class Qwen3VLVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]  # GELU

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))
```

**Self-Check Questions**

**Q2.3.1:** The vision encoder uses LayerNorm while the text decoder uses RMSNorm. Why might this be?

<details>
<summary>Answer</summary>

The vision encoder was likely initialized from a pretrained ViT that used LayerNorm. Changing normalization type would require retraining from scratch. RMSNorm's efficiency benefits (~15% speedup) matter more for the 64-layer decoder than the 27-layer encoder.
</details>

**Q2.3.2:** With 196 patches and 16 attention heads, what is the shape of the attention weight matrix in one head?

<details>
<summary>Answer</summary>

$(196, 196)$. Each of the 196 query positions attends to all 196 key positions. The full attention weights across all heads would be $(16, 196, 196)$.
</details>

---

### 2.4 Multi-Level Feature Extraction (DeepStack)

**The problem with standard VLMs:** Most vision-language models use only the final output of the vision encoder. But as we saw in Section 2.3, different layers capture different information—early layers preserve fine details like edges and textures, while later layers capture high-level semantics. By using only the final output, standard VLMs lose access to fine-grained visual information.

Consider a concrete example: "Is there a scratch on the car's door?"

- **Layer 27 features** know "this is a car" and "that's the door region"—useful for localization
- **Layer 8 features** preserve the edge discontinuities and texture anomalies that indicate a scratch

If the model only sees layer 27's output, the scratch has been abstracted away through 27 layers of processing. The high-level representation says "car door" but not "car door with fine linear mark."

**DeepStack's solution:** Extract features at multiple depths and inject them into the language model:

```
Vision Encoder Layers               Extracted Features
─────────────────────               ──────────────────
Layer 0  ─┐
   ...    │
Layer 7  ─┤
Layer 8  ─┼──────────────────────►  DeepStack[0]: edges, textures, fine details
   ...    │
Layer 15 ─┤
Layer 16 ─┼──────────────────────►  DeepStack[1]: shapes, parts, local structure
   ...    │
Layer 23 ─┤
Layer 24 ─┼──────────────────────►  DeepStack[2]: objects, regions, layout
   ...    │
Layer 27 ─┴──────────────────────►  Main output: high-level semantics
```

**Why layers 8, 16, 24 specifically?** These are roughly evenly spaced through the 27-layer encoder, capturing low, mid, and high-level features. Extracting from adjacent layers (e.g., 8 and 9) would be redundant—their representations are nearly identical. The spacing ensures each extraction point provides meaningfully different information.

**What each level captures:**

| Source Layer | Feature Level | What's preserved | Example use |
|-------------|---------------|------------------|-------------|
| 8 | Low | Edges, textures, color gradients | Detecting scratches, reading small text |
| 16 | Mid | Shapes, object parts | Recognizing partially occluded objects |
| 24 | High | Object boundaries, spatial layout | Understanding scene structure |
| 27 | Semantic | Object identities, relationships | High-level reasoning |

Each extracted feature set goes through its own spatial merger (Section 2.5) before being injected into the decoder (Section 2.9). We'll see how the injection works after covering the fusion mechanism.

The vision model initializes separate mergers for each DeepStack extraction point:

```python
# From modeling_qwen3_vl.py - initialization
self.deepstack_visual_indexes = config.deepstack_visual_indexes  # e.g., [8, 16, 24]
self.deepstack_merger_list = nn.ModuleList([
    Qwen3VLVisionPatchMerger(config, use_postshuffle_norm=True)
    for _ in range(len(config.deepstack_visual_indexes))
])
```

During the forward pass, features are extracted at designated layers:

```python
# From modeling_qwen3_vl.py - forward pass
deepstack_feature_lists = []
for layer_num, blk in enumerate(self.blocks):
    hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)

    if layer_num in self.deepstack_visual_indexes:
        deepstack_feature = self.deepstack_merger_list[
            self.deepstack_visual_indexes.index(layer_num)
        ](hidden_states)
        deepstack_feature_lists.append(deepstack_feature)

merged_hidden_states = self.merger(hidden_states)
```

**Self-Check Questions**

**Q2.4.1:** DeepStack extracts from layers 8, 16, 24 in a 27-layer encoder. Why not extract from more layers (e.g., every 4th layer)?

<details>
<summary>Answer</summary>

More extraction points would produce more feature sets to inject, but there are only 64 decoder layers. Injecting at every decoder layer would interfere with reasoning in later layers. Three extraction points (injected at layers 0, 1, 2) provide multi-scale information without overwhelming the decoder. Additionally, nearby layers produce similar features—extraction at layers 8 and 9 would be redundant.
</details>

**Q2.4.2:** Each DeepStack feature goes through its own merger. Why not share a single merger across all extraction points?

<details>
<summary>Answer</summary>

Features at different depths have different statistical properties. Layer 8 features are more local and edge-like; layer 24 features are more semantic. Separate mergers can learn layer-specific projections. The parameter cost is modest (three small MLPs) compared to the encoder and decoder.
</details>

---

## Vision-Language Fusion

We've now extracted visual features from the vision encoder—both the main output (layer 27) and the DeepStack features (layers 8, 16, 24). The next challenge is bridging the gap between vision and language: converting these visual features into a format the text decoder can process.

### 2.5 Spatial Merging

**The problem:** Visual tokens are expensive. A single 448×448 image produces 784 patches, and higher resolutions or videos produce even more. The text decoder's self-attention has $O(N^2)$ complexity, so doubling token count quadruples attention cost. For a 2-minute video, we might have tens of thousands of visual tokens competing with the text prompt for context window space.

**The solution:** Merge adjacent patches to reduce token count while preserving information. Qwen3-VL merges 2×2 groups of patches into single tokens:

```
Before merge: 28×28 patches, 1152-dim each = 784 tokens
After merge:  14×14 patches, 5120-dim each = 196 tokens
                                             (4× reduction)

┌─────┬─────┐                    ┌───────────┐
│  0  │  1  │                    │           │
├─────┼─────┤  ────────────►     │  merged   │
│  2  │  3  │   concatenate      │   5120    │
└─────┴─────┘   4 × 1152         └───────────┘
                = 4608
                   │
                   ▼
            MLP: 4608 → 5120
```

The four 1152-dimensional patch embeddings are concatenated (producing a 4608-dimensional vector) and projected through an MLP to 5120 dimensions. Why 5120? That's the hidden size of the text decoder—visual tokens must match this dimension to be inserted into the text sequence.

**Trade-off:** Merging loses some spatial precision. After merging, each token represents a 32×32 pixel region instead of 16×16. For tasks requiring fine spatial detail, the DeepStack features (also merged but injected separately) help compensate.

The patch merger concatenates 2×2 neighbors and projects to the decoder's hidden size:

```python
# From modeling_qwen3_vl.py
class Qwen3VLVisionPatchMerger(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)  # 1152 * 4 = 4608
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)  # → 5120

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x).view(-1, self.hidden_size)  # Group 2×2 patches
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x
```

Note: Patches must be reordered so 2×2 spatial neighbors are consecutive before merging.

**Self-Check Questions**

**Q2.5.1:** For a 448×448 image, how many visual tokens enter the text decoder after merging?

<details>
<summary>Answer</summary>

Before merging: $(448/16)^2 = 784$ patches.
After 2×2 merging: $784/4 = 196$ tokens.
</details>

**Q2.5.2:** Why project to 5120 dimensions (matching decoder hidden size) rather than keeping 4608?

<details>
<summary>Answer</summary>

Visual tokens must be added directly to text token embeddings during sequence construction. Dimension mismatch would require additional projection at every decoder layer. Matching dimensions at the merger eliminates this overhead.
</details>

---

### 2.6 Sequence Construction

Now we need to insert the visual tokens into the text sequence. The tokenizer creates placeholder tokens that mark where visual content should go, and we replace these placeholders with the actual visual features.

```
Step 1: Tokenize text with placeholders
┌────────┬─────┬──────┬─────┬─────────┬─────────┬─────────┬─────┬───────┬───┐
│ "What" │ "is"│"this"│ "?" │<v_start>│<img_pad>│<img_pad>│ ... │<v_end>│"."│
└────────┴─────┴──────┴─────┴─────────┴─────────┴─────────┴─────┴───────┴───┘
                                        ▲         ▲         ▲
                                        └─── 196 placeholders ───┘

Step 2: Embed all tokens (text tokens get embeddings, placeholders get dummy values)

Step 3: Replace placeholder embeddings with visual features
┌────────┬─────┬──────┬─────┬─────────┬─────────────────────────────┬───────┬───┐
│ "What" │ "is"│"this"│ "?" │<v_start>│  [visual tokens from encoder] │<v_end>│"."│
└────────┴─────┴──────┴─────┴─────────┴─────────────────────────────┴───────┴───┘
                                         196 tokens × 5120 dim
```

Special tokens:
- `<|vision_start|>` (ID 151652): marks beginning of visual content
- `<|image_pad|>` (ID 151655): placeholder for image tokens
- `<|video_pad|>` (ID 151656): placeholder for video tokens
- `<|vision_end|>` (ID 151653): marks end of visual content

The processor computes exactly how many placeholders to insert based on image dimensions: $(H/16) \times (W/16) / 4$ tokens after spatial merging.

In the forward pass, `masked_scatter` replaces placeholder embeddings with visual features:

```python
# From modeling_qwen3_vl.py - forward pass
image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
```

**Self-Check Questions**

**Q2.6.1:** The processor creates exactly 196 `<image_pad>` tokens for a 224×224 image. What determines this count?

<details>
<summary>Answer</summary>

The count matches the vision encoder output after merging: $(224/16)^2 / 4 = 196$ tokens. The processor computes this from image dimensions before tokenization to insert the correct number of placeholders.
</details>

**Q2.6.2:** Why use `masked_scatter` (or equivalent) rather than direct indexing like `embeds[mask] = features`?

<details>
<summary>Answer</summary>

Direct boolean indexing flattens the batch dimension, making it ambiguous which features belong to which batch element when batch_size > 1. `masked_scatter` preserves the tensor structure and correctly handles batched operations. It also enables proper gradient flow during training.
</details>

---

### 2.7 M-RoPE Position Assignment

**The problem:** Standard RoPE assigns one position number to each token. This works for text (tokens have a linear order) but not for images (patches have 2D positions) or videos (patches have 3D positions: frame, row, column). How do we encode position for a mixed sequence of text, images, and video?

**M-RoPE's solution:** Give every token **three** position indices instead of one:

$$\mathbf{p} = (p_T, p_H, p_W)$$

- $p_T$ (Temporal): The token's position in the sequence / frame index
- $p_H$ (Height): The token's row position in a spatial grid
- $p_W$ (Width): The token's column position in a spatial grid

For **text tokens**, there's no spatial structure, so we collapse to 1D by setting all three indices equal: $p_T = p_H = p_W = t$, where $t$ is the token's position in the sequence.

For **image patches** at grid position $(h, w)$, we encode the spatial structure:
- $p_T$ = the image's position in the sequence (all patches share this)
- $p_H$ = $h$ (row in the patch grid)
- $p_W$ = $w$ (column in the patch grid)

For **video patches** from frame $f$ at position $(h, w)$:
- $p_T$ = base position + $f$ (different frames get different temporal positions)
- $p_H$ = $h$
- $p_W$ = $w$

**Concrete example:** "Describe" + 2×2 image + "now"

```
Token          │  T  │  H  │  W  │  Notes
───────────────┼─────┼─────┼─────┤───────────────────────────────
"Describe"     │  0  │  0  │  0  │  Text: T=H=W=0
[img₀₀]        │  1  │  0  │  0  │  Image patch at row 0, col 0
[img₀₁]        │  1  │  0  │  1  │  Image patch at row 0, col 1
[img₁₀]        │  1  │  1  │  0  │  Image patch at row 1, col 0
[img₁₁]        │  1  │  1  │  1  │  Image patch at row 1, col 1
"now"          │  2  │  2  │  2  │  Text: continues from max + 1
```

Note that all image patches share $T=1$ (they're at the same sequence position), but differ in $H$ and $W$ (their spatial positions). After the image, text resumes with $T = H = W = 2$ (continuing from the maximum position used).

**How M-RoPE applies these positions:** Each axis (T, H, W) gets a portion of the head dimensions for its rotations. Section 2.8 explains how these dimensions are allocated.

The real implementation in `Qwen3VLModel.get_rope_index` handles both images and videos. Here's the core logic for generating the spatial position indices:

```python
# For visual tokens (images/videos), create separate T, H, W indices
t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)

# For text tokens, all three indices are identical (1D position)
llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
```

The key insight is that visual tokens get distinct $(T, H, W)$ positions encoding their spatial grid location, while text tokens collapse to 1D with $T = H = W$.

**Self-Check Questions**

**Q2.7.1:** After processing "Hello" (1 token) + a 14×14 image (196 tokens) + "World" (1 token), what is the T position of "World"?

<details>
<summary>Answer</summary>

"Hello" has $T=0$. The entire image occupies $T=1$ (all 196 patches share this). "World" has $T=2$. The answer is **2**.
</details>

**Q2.7.2:** Two image patches have positions $(T=5, H=3, W=7)$ and $(T=5, H=3, W=9)$. What spatial relationship does their position difference encode?

<details>
<summary>Answer</summary>

$\Delta T = 0$, $\Delta H = 0$, $\Delta W = 2$. They are in the same row ($\Delta H = 0$), same frame/image ($\Delta T = 0$), and 2 patches apart horizontally. This encodes "same row, 2 columns to the right."
</details>

---

### 2.8 Interleaved Frequency Allocation

We've established that M-RoPE uses three position indices $(p_T, p_H, p_W)$. Now we need to understand how these positions are turned into rotations within the 128-dimensional head space.

**Recall from Section 1.2:** RoPE uses multiple frequencies $\theta_0, \theta_1, \ldots$ where $\theta_i = b^{-2i/d}$. Low-index frequencies ($\theta_0, \theta_1$) rotate quickly and distinguish nearby positions. High-index frequencies ($\theta_{62}, \theta_{63}$) rotate slowly and distinguish distant positions.

**The allocation question:** We have 128 head dimensions (64 rotation pairs) and three axes to encode. How do we divide the dimensions among T, H, and W?

Qwen3-VL uses `mrope_section = [24, 20, 20]`—24 pairs for T, 20 for H, 20 for W. T gets more because text sequences can reach 256K tokens, while image grids are bounded (~14×14 after merging).

**The problem with contiguous allocation (Qwen2-VL):**

```
Head dim = 128 = 64 pairs

Contiguous allocation:
  T: pairs 0-23  → frequencies θ₀, θ₁, ..., θ₂₃     (HIGH frequencies only)
  H: pairs 24-43 → frequencies θ₂₄, θ₂₅, ..., θ₄₃   (MEDIUM frequencies only)
  W: pairs 44-63 → frequencies θ₄₄, θ₄₅, ..., θ₆₃   (LOW frequencies only)
```

This creates a problem: each axis only accesses a narrow frequency range.

- **T gets only high frequencies:** Can distinguish nearby text positions but struggles with positions 1000 tokens apart (the rotations look similar at high frequencies)
- **W gets only low frequencies:** Can distinguish distant columns but struggles with adjacent columns (low frequencies barely rotate between $w$ and $w+1$)

For long videos, this is especially problematic—the temporal axis needs low frequencies to distinguish frame 1 from frame 100.

**Qwen3-VL's solution: Interleaving**

Instead of giving each axis a contiguous block, distribute frequencies across the full spectrum:

```
Interleaved frequency assignment:
  T: θ₀, θ₃, θ₆, θ₉,  ... (every 3rd frequency, starting at 0)
  H: θ₁, θ₄, θ₇, θ₁₀, ... (every 3rd frequency, starting at 1)
  W: θ₂, θ₅, θ₈, θ₁₁, ... (every 3rd frequency, starting at 2)
```

Now each axis samples uniformly across the entire frequency spectrum:

```
Frequency spectrum:  HIGH ◄──────────────────────────────────► LOW
                     θ₀   θ₁   θ₂   θ₃   θ₄   θ₅  ...  θ₆₂  θ₆₃

Contiguous:
  T:                 ████ ████ ████ ████ ████ ████ ░░░░ ░░░░ ░░░░
  H:                 ░░░░ ░░░░ ░░░░ ░░░░ ░░░░ ░░░░ ████ ████ ░░░░
  W:                 ░░░░ ░░░░ ░░░░ ░░░░ ░░░░ ░░░░ ░░░░ ░░░░ ████

Interleaved:
  T:                 ████ ░░░░ ░░░░ ████ ░░░░ ░░░░ ████ ░░░░ ░░░░
  H:                 ░░░░ ████ ░░░░ ░░░░ ████ ░░░░ ░░░░ ████ ░░░░
  W:                 ░░░░ ░░░░ ████ ░░░░ ░░░░ ████ ░░░░ ░░░░ ████
```

**The benefit:** Each axis can now resolve both local and distant positions. T can distinguish both adjacent tokens (using its high-frequency components) and tokens 10,000 apart (using its low-frequency components). This significantly improves long-video understanding, where frames must remain distinguishable across extended sequences.

**Self-Check Questions**

**Q2.8.1:** With `mrope_section = [24, 20, 20]` and interleaving, which axis has the most frequency components? Does this affect which axis can model the longest distances?

<details>
<summary>Answer</summary>

T has 24 pairs, H and W have 20 each. More pairs means finer sampling of the frequency spectrum, but all axes now access the full range (both high and low frequencies). The number of pairs affects resolution, not range. T having more pairs helps because text sequences can reach 256K tokens, requiring finer temporal resolution than the bounded image grids (~14×14 after merging).
</details>

**Q2.8.2:** In contiguous allocation, why does the W axis (assigned to lowest frequencies) struggle with local spatial relationships?

<details>
<summary>Answer</summary>

Low frequencies rotate slowly—adjacent positions have nearly identical angles. For W, $\theta_{44}$ through $\theta_{63}$ change by small amounts between $w$ and $w+1$. The dot product $\cos(\Delta w \cdot \theta)$ is nearly 1 for small $\Delta w$, making adjacent columns hard to distinguish. High frequencies rotate quickly, creating distinct angles for nearby positions.
</details>

---

## The Text Decoder

The visual tokens are now embedded in the text sequence with proper position encoding. The text decoder processes this unified sequence to generate output. But first, we need to inject the DeepStack features we extracted earlier.

### 2.9 DeepStack Injection

Recall from Section 2.4 that DeepStack extracts features at layers 8, 16, and 24 of the vision encoder—capturing low, mid, and high-level visual information respectively. These features were processed through their own mergers (Section 2.5) and are now ready to be injected into the decoder.

**The injection mechanism:** At the end of decoder layers 0, 1, and 2, the DeepStack features are **added** to the hidden states at visual token positions:

```
Decoder Layer 0                Decoder Layer 1                Decoder Layer 2
───────────────                ───────────────                ───────────────
Input embeds                   Layer 0 output                 Layer 1 output
     │                              │                              │
     ▼                              ▼                              ▼
  Attention                      Attention                      Attention
     │                              │                              │
     ▼                              ▼                              ▼
    MLP                            MLP                            MLP
     │                              │                              │
     ▼                              ▼                              ▼
hidden[vis] +=                hidden[vis] +=                hidden[vis] +=
DeepStack[0]                  DeepStack[1]                  DeepStack[2]
(from vision layer 8)         (from vision layer 16)        (from vision layer 24)
     │                              │                              │
     ▼                              ▼                              ▼
 To Layer 1                    To Layer 2                    To Layer 3
```

**Why addition instead of replacement or concatenation?**
- **Addition** preserves the main visual tokens (from layer 27) while enriching them with multi-scale details. It's also parameter-free and doesn't increase sequence length.
- **Replacement** would destroy the high-level semantic information.
- **Concatenation** would multiply visual token count by 4×, dramatically increasing attention cost ($O(N^2)$).

**Why inject only at early layers (0, 1, 2)?** Early decoder layers build representations—they benefit from rich multi-scale visual input. By layer 3, the model should have integrated the visual information and can focus on reasoning. Injecting low-level features (edges, textures) into layer 50 would add noise rather than useful information.

The real `Qwen3VLTextModel.forward` implements this injection in the decoder loop:

```python
for layer_idx, decoder_layer in enumerate(self.layers):
    layer_outputs = decoder_layer(hidden_states, ...)
    hidden_states = layer_outputs[0]

    # DeepStack injection at layers 0, 1, 2
    if layer_idx < len(deepstack_visual_embeds) and deepstack_visual_embeds[layer_idx] is not None:
        hidden_states = hidden_states.clone()
        hidden_states[visual_pos_masks] = (
            hidden_states[visual_pos_masks] + deepstack_visual_embeds[layer_idx]
        )
```

The `visual_pos_masks` tensor identifies which positions in the sequence correspond to visual tokens, ensuring features are only added at the right locations.

**Self-Check Questions**

**Q2.9.1:** If DeepStack features were concatenated (doubling visual token count) instead of added, what would be the impact on computational cost?

<details>
<summary>Answer</summary>

Attention cost is $O(N^2)$ where $N$ is sequence length. Doubling visual tokens (e.g., 196 → 392 for one image) would nearly double total sequence length for image-heavy inputs. With 4× DeepStack levels, concatenation would mean 4× the visual tokens, dramatically increasing attention cost. Addition keeps sequence length constant.
</details>

**Q2.9.2:** The injection happens after the MLP in each layer. What if it happened before attention instead?

<details>
<summary>Answer</summary>

Injecting before attention would allow the layer's attention mechanism to immediately use the multi-scale features. However, the features would also be transformed by that layer's attention and MLP before reaching the next layer. Injecting after MLP means the features are added "cleanly" at layer boundaries, giving the next layer unmodified multi-scale information. The current design treats injection as enriching the residual stream between layers.
</details>

---

### 2.10 Decoder Architecture

With DeepStack injection understood, let's examine the decoder layer structure itself. Qwen3-VL uses a 64-layer decoder with the modern components covered in Part 1: pre-norm with RMSNorm, GQA for efficient KV caching, QK-Norm for training stability, and SwiGLU for the MLP.

Each layer follows this structure:

```
Input (batch, seq, 5120)
     │
     ├─────────────────────────────────────┐
     ▼                                     │
 RMSNorm                                   │
     │                                     │
     ▼                                     │
 GQA Self-Attention                        │
   Q: 64 heads × 128 dim = 8192            │  Residual
   K: 8 heads × 128 dim = 1024             │
   V: 8 heads × 128 dim = 1024             │
   Out: 64 heads × 128 → 5120              │
     │                                     │
     ▼                                     │
     + ◄───────────────────────────────────┘
     │
     ├─────────────────────────────────────┐
     ▼                                     │
 RMSNorm                                   │
     │                                     │
     ▼                                     │
 SwiGLU MLP                                │  Residual
   Gate: 5120 → 25600                      │
   Up:   5120 → 25600                      │
   Down: 25600 → 5120                      │
     │                                     │
     ▼                                     │
     + ◄───────────────────────────────────┘
     │
     ▼
Output (batch, seq, 5120)
```

**GQA configuration:** 64 query heads share 8 KV heads (8× compression). Each group of 8 Q heads attends to the same K, V. This dramatically reduces KV cache memory (see Section 1.1 for GQA details).

**QK-Norm:** RMSNorm is applied to Q and K per-head before computing attention scores:

$$\mathbf{Q}_{\text{norm}} = \text{RMSNorm}(\mathbf{Q}), \quad \mathbf{K}_{\text{norm}} = \text{RMSNorm}(\mathbf{K})$$

This prevents Q and K magnitudes from growing unboundedly during training. Without QK-Norm, attention scores can become extreme, causing softmax to saturate and gradients to vanish (see Section 1.3 for normalization details).

The real `Qwen3VLTextAttention` implementation shows the key design choices:

```python
class Qwen3VLTextAttention(nn.Module):
    def __init__(self, config, layer_idx):
        # Q/K/V projections with GQA: 64 Q heads, 8 KV heads
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        # QK-Norm applied per-head (head_dim=128, not hidden_size=5120)
        self.q_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask, ...):
        # Project then normalize (QK-Norm happens after projection)
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Apply M-RoPE position encoding
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # ... attention computation, output projection
```

Note the QK-Norm placement: normalization happens **after** projection but **before** RoPE, operating on each 128-dim head independently.

**Self-Check Questions**

**Q2.10.1:** For Qwen3-VL-32B with 256K context, what is the KV cache size per sequence?

<details>
<summary>Answer</summary>

Per token: $2 \times 64 \times 8 \times 128 \times 2 = 262,144$ bytes (K and V, 64 layers, 8 KV heads, 128 dim, fp16).

For 256K context: $256,000 \times 262,144 = 67.1$ GB per sequence.

This is why GQA's 8× KV reduction is critical—without it, the cache would be 537 GB.
</details>

**Q2.10.2:** QK-Norm and $\sqrt{d_k}$ scaling both prevent attention score explosion. Why use both?

<details>
<summary>Answer</summary>

They address different phenomena:
- **QK-Norm** bounds vector magnitudes: $\|\mathbf{Q}\|$ and $\|\mathbf{K}\|$ stay near $\sqrt{d}$ regardless of training dynamics
- **$\sqrt{d_k}$ scaling** normalizes the dot product: $\mathbf{Q} \cdot \mathbf{K}$ sums $d_k$ terms, so variance grows with $d_k$ even for normalized vectors

After QK-Norm, $\|\mathbf{Q}\| \approx \|\mathbf{K}\| \approx \sqrt{128} \approx 11.3$. The dot product of two such vectors sums 128 terms, giving expected magnitude $\approx 128$ before scaling. Dividing by $\sqrt{128} \approx 11.3$ brings scores to $O(1)$.
</details>

---

## Video Processing

### 2.11 Video Processing

We've covered how Qwen3-VL handles images. Videos add a temporal dimension, but the core pipeline remains similar—with two key additions: temporal grouping and explicit timestamp tokens.

**Temporal patching:** The Conv3D in Section 2.1 has a temporal kernel size of 2, meaning it processes frames in pairs. For a 60-frame video:

```
Video: 60 frames at 224×224

Step 1: Group frames into pairs (temporal_patch_size = 2)
        60 frames → 30 temporal groups
        Each group: 2 consecutive frames

Step 2: Process each group through the vision encoder
        Per group: 14×14 = 196 spatial patches (before merging)
        After 2×2 spatial merge: 49 tokens per group

Step 3: Total visual tokens
        30 groups × 49 tokens = 1,470 visual tokens
```

**Explicit timestamp tokens:** Unlike images, videos need temporal grounding—the model must know when events occur. Qwen3-VL inserts timestamp tokens (`<t0.0s>`, `<t0.5s>`, ...) before each frame group:

```
Sequence structure for a video:
<t0.0s> <v_start> [49 visual tokens] <v_end>
<t0.5s> <v_start> [49 visual tokens] <v_end>
<t1.0s> <v_start> [49 visual tokens] <v_end>
...
```

This enables natural temporal references in both questions and answers:
- Question: "What happens at `<t35>`?"
- Answer: "At `<t35>`, the person opens the door."

**Why explicit tokens instead of relying on M-RoPE?** M-RoPE encodes relative positions, but timestamps are absolute. Explicit tokens create direct associations: the model learns that `<t35>` means "35 seconds into the video" as a concept it can reference in its output. Without explicit tokens, the model would need to infer absolute time from position indices—less intuitive and harder to train.

<details>
<summary><b>Token Count Example</b></summary>

5-second video at 2 fps, 224×224:
- Frames: 10
- Temporal groups: 5 (after grouping by 2)
- Timestamp tokens: 5
- Vision markers: 5 × 2 = 10 (`<v_start>` and `<v_end>` per group)
- Visual tokens: 5 × 49 = 245 (after spatial merging)
- **Total: 260 video-related tokens** (plus text prompt)

Compare to image: single 224×224 image = 49 visual tokens + 2 markers = 51 tokens.
</details>

**Self-Check Questions**

**Q2.11.1:** A 2-minute video at 1 fps with 448×448 resolution produces how many visual tokens?

<details>
<summary>Answer</summary>

- Frames: 120
- Temporal groups: 60 (grouping by 2)
- Patches per group before merge: $(448/16)^2 = 784$
- After 2×2 spatial merge: $784/4 = 196$ per temporal group
- Total: $60 \times 196 = 11,760$ visual tokens

Plus 60 timestamp tokens and 120 vision markers = **11,940 tokens**.
</details>

**Q2.11.2:** Why does the position computation split video grids into per-frame segments?

<details>
<summary>Answer</summary>

```python
video_grid_thw = video_grid_thw.repeat_interleave(video_grid_thw[:, 0], dim=0)
video_grid_thw[:, 0] = 1  # T=1 per segment
```

This converts `[[30, 14, 14]]` to 30 entries of `[[1, 14, 14]]`. Each frame group gets independent H, W position assignment. Combined with timestamp tokens, this enables the model to associate each `<tN>` with its specific frame's content. Without this split, all frames would share H, W positions, losing frame-specific spatial grounding.
</details>

---

## Synthesis

### 2.12 End-to-End Forward Pass

We've covered all the individual components. Let's trace a complete forward pass to see how they work together. For an image understanding task:

```
Input: "What is in this image?" + 224×224 image

Step 1: Tokenize text
        ["What", "is", "in", "this", "image", "?", <v_start>,
         <img_pad> × 196, <v_end>]
        → input_ids: (1, 205)

Step 2: Embed text tokens
        → inputs_embeds: (1, 205, 5120)

Step 3: Process image through vision encoder
        224×224 → 196 patches (1152-dim)
        → 27 layers with DeepStack extraction at 8, 16, 24
        → Final merger: 196 tokens (5120-dim)
        → DeepStack mergers: 3 × 196 tokens (5120-dim)

Step 4: Replace placeholders
        inputs_embeds[:, 7:203, :] = visual_features
        → inputs_embeds: (1, 205, 5120) with visual content

Step 5: Compute M-RoPE positions
        Text: T=H=W ∈ {0,1,2,3,4,5,6}
        Image: T=7, H∈{0..13}, W∈{0..13}
        Final text: T=H=W=8
        → position_ids: (3, 1, 205)

Step 6: Create causal mask
        → attention_mask: (1, 1, 205, 205) lower triangular

Step 7: Decoder forward pass
        Layer 0: attention + MLP + inject DeepStack[0]
        Layer 1: attention + MLP + inject DeepStack[1]
        Layer 2: attention + MLP + inject DeepStack[2]
        Layers 3-63: attention + MLP

Step 8: Project to vocabulary
        → logits: (1, 205, vocab_size)

Step 9: Generate tokens autoregressively
        Sample from logits[:, -1, :], append, repeat
```

The real `Qwen3VLModel.forward` shows this orchestration:

```python
def forward(self, input_ids, pixel_values, image_grid_thw, ...):
    # Step 2: Text embeddings
    inputs_embeds = self.get_input_embeddings()(input_ids)

    # Step 3-4: Vision encoder + placeholder replacement
    if pixel_values is not None:
        image_outputs = self.get_image_features(pixel_values, image_grid_thw, return_dict=True)
        image_embeds = image_outputs.pooler_output
        deepstack_image_embeds = image_outputs.deepstack_features
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device)
        image_mask, _ = self.get_placeholder_mask(input_ids, image_features=image_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    # Step 5: M-RoPE position indices
    position_ids, rope_deltas = self.get_rope_index(input_ids, image_grid_thw, video_grid_thw)

    # Step 7: Decoder (with DeepStack injection handled internally)
    outputs = self.language_model(
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
        ...
    )
    return outputs
```

And in `Qwen3VLForConditionalGeneration.forward`:

```python
# Step 8: Output projection
hidden_states = outputs[0]
logits = self.lm_head(hidden_states[:, slice_indices, :])
```

---

# References

[1] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention Is All You Need," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.

[2] J. Ainslie, J. Lee-Thorp, M. de Jong, Y. Zemlyanskiy, F. Lebrón, and S. Sanghai, "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints," in *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 2023. arXiv:2305.13245.

[3] J. Su, Y. Lu, S. Pan, A. Murtadha, B. Wen, and Y. Liu, "RoFormer: Enhanced Transformer with Rotary Position Embedding," *Neurocomputing*, vol. 568, 2024. arXiv:2104.09864.

[4] N. Shazeer, "GLU Variants Improve Transformer," arXiv:2002.05202, 2020.

[5] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," in *International Conference on Learning Representations (ICLR)*, 2021. arXiv:2010.11929.

[6] J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds, R. Ring, E. Rutherford, S. Cabi, T. Han, Z. Gong, S. Samangooei, M. Monteiro, J. Menick, S. Borgeaud, A. Brock, A. Nematzadeh, S. Shaber, M. Ranzato, and O. Vinyals, "Flamingo: A Visual Language Model for Few-Shot Learning," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2022. arXiv:2204.14198.

[7] H. Liu, C. Li, Q. Wu, and Y. J. Lee, "Visual Instruction Tuning," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2023. arXiv:2304.08485.

[8] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang, H. Zhong, Y. Zhu, M. Yang, Z. Li, J. Wan, P. Wang, W. Ding, Z. Fu, Y. Xu, J. Ye, X. Zhang, T. Wu, X. Ren, and G. Huang, "DeepStack: Deeply Stacking Visual Tokens is Surprisingly Simple and Effective for LMMs," arXiv:2406.04334, 2024.

[9] A. Arnab, M. Dehghani, G. Heigold, C. Sun, M. Lučić, and C. Schmid, "ViViT: A Video Vision Transformer," in *IEEE/CVF International Conference on Computer Vision (ICCV)*, 2021. arXiv:2103.15691.

[10] Qwen Team, "Qwen3-VL Technical Report," arXiv:2511.21631, 2025.

## Further Reading

- J. Alammar, "The Illustrated Transformer," 2018. https://jalammar.github.io/illustrated-transformer/ — An excellent visual introduction to transformer architecture.

- "Inside RoPE: Rotary Magic into Position Embeddings," LearnOpenCV, 2025. https://learnopencv.com/rope-position-embeddings/ — A tutorial on rotary position embeddings with visualizations.

---

*Tutorial version: January 2026 (Revised)*