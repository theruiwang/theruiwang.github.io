# Qwen3.5 vs Qwen3-VL: A Deep Architectural Comparison

This tutorial compares the architectures of **Qwen3.5-9B-Instruct** and **Qwen3-VL-8B-Instruct** using exact code from the HuggingFace Transformers source. Both are vision-language models (VLMs), but they make fundamentally different design choices in how their text decoders process information.

All code references are from:
- `src/transformers/models/qwen3_5/modeling_qwen3_5.py`
- `src/transformers/models/qwen3_5/configuration_qwen3_5.py`
- `src/transformers/models/qwen3_vl/modeling_qwen3_vl.py`
- `src/transformers/models/qwen3_vl/configuration_qwen3_vl.py`

---

## 1. Overview & Cheat Sheet

Both models share the same high-level VLM structure: a **vision encoder** (ViT), a **text decoder** (transformer), and a **language modeling head**.

```python
# Qwen3_5Model.__init__ (modeling_qwen3_5.py, line 1404)
class Qwen3_5Model(Qwen3_5PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen3_5VisionModel._from_config(config.vision_config)
        self.language_model = Qwen3_5TextModel._from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()

# Qwen3VLModel.__init__ (modeling_qwen3_vl.py, line 957)
class Qwen3VLModel(Qwen3VLPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen3VLVisionModel._from_config(config.vision_config)
        self.language_model = Qwen3VLTextModel._from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()
```

The outer shell is identical: `visual` + `language_model` + `rope_deltas`. The differences are all inside these components.

### Summary Table

| Property | Qwen3.5-9B | Qwen3-VL-8B |
|---|---|---|
| **Text layers** | 32 (24 linear + 8 full) | 36 (all full attention) |
| **GQA** | 16Q / 4KV (4x grouping) | 32Q / 8KV (4x grouping) |
| **Head dim** | 256 | 128 |
| **intermediate_size** | 12288 | 12288 |
| **Query gating** | Yes (sigmoid) | No |
| **Partial rotary** | 0.25 (64 of 256 dims rotated) | 1.0 (all 128 dims rotated) |
| **RMSNorm** | 1-centered: `(1 + w) * norm(x)` | Standard: `w * norm(x)` |
| **DeepStack** | No | Yes (vision layers 8, 16, 24) |
| **Vision out_hidden_size** | 3584 | 4096 |
| **max_position_embeddings** | 32,768 | 262,144 |
| **rope_theta** | (from rope_params) | 5,000,000 |
| **Vocab size** | 248,320 | 151,936 |
| **Cache type** | Custom `Qwen3_5DynamicCache` (KV + conv + recurrent) | Standard `DynamicCache` |
| **mrope_section** | [11, 11, 10] | [24, 20, 20] |

---

## 2. Shared Foundation

### 2.1 SwiGLU MLP

Both models use the exact same SwiGLU MLP structure with identical expansion ratio (12288 / 4096 = 3x):

```python
# Qwen3_5MLP (modeling_qwen3_5.py, line 791)
class Qwen3_5MLP(nn.Module):
    def __init__(self, config, intermediate_size):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]  # silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

# Qwen3VLTextMLP (modeling_qwen3_vl.py, line 503)
class Qwen3VLTextMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]  # silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

The formula is: `output = down_proj(SiLU(gate_proj(x)) * up_proj(x))`. Structurally identical.

### 2.2 Vision Encoder

The vision encoders are nearly identical: 27 transformer blocks, 1152 hidden dim, 16 heads, Conv3d patch embedding, LayerNorm (not RMSNorm), GELU MLP, learned positional embeddings + 2D RoPE.

```python
# Representative VisionBlock (both models use the same structure)
# Qwen3_5VisionBlock (modeling_qwen3_5.py, line 1060)
class Qwen3_5VisionBlock(GradientCheckpointingLayer):
    def __init__(self, config, attn_implementation="sdpa"):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3_5VisionAttention(config=config)
        self.mlp = Qwen3_5VisionMLP(config=config)

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None,
                position_embeddings=None, **kwargs):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings, **kwargs)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states
```

Vision configs side-by-side:

```python
# Qwen3_5VisionConfig (configuration_qwen3_5.py, line 202)
class Qwen3_5VisionConfig(PreTrainedConfig):
    def __init__(self, depth=27, hidden_size=1152, hidden_act="gelu_pytorch_tanh",
                 intermediate_size=4304, num_heads=16, in_channels=3, patch_size=16,
                 spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=3584,
                 num_position_embeddings=2304, initializer_range=0.02, **kwargs):
        ...

# Qwen3VLVisionConfig (configuration_qwen3_vl.py, line 28)
class Qwen3VLVisionConfig(PreTrainedConfig):
    def __init__(self, depth=27, hidden_size=1152, hidden_act="gelu_pytorch_tanh",
                 intermediate_size=4304, num_heads=16, in_channels=3, patch_size=16,
                 spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=3584,
                 num_position_embeddings=2304,
                 deepstack_visual_indexes=[8, 16, 24],  # <-- only Qwen3-VL has this
                 initializer_range=0.02, **kwargs):
        ...
```

Note: The code defaults show `out_hidden_size=3584` for both, but the actual Qwen3-VL-8B-Instruct config.json overrides this to `4096` to match its text model's hidden_size of 4096.

### 2.3 3D Multimodal RoPE (MRoPE)

Both models use 3D MRoPE with an identical `apply_interleaved_mrope` function that reorganizes frequency layout from chunked `[TTT...HHH...WWW]` to interleaved `[THWTHWTHW...TT]`:

```python
# Identical in both models
# Qwen3_5TextRotaryEmbedding.apply_interleaved_mrope (modeling_qwen3_5.py, line 245)
# Qwen3VLTextRotaryEmbedding.apply_interleaved_mrope (modeling_qwen3_vl.py, line 362)
def apply_interleaved_mrope(self, freqs, mrope_section):
    """Apply interleaved MRoPE to 3D rotary embeddings.
    Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
    interleaved [THWTHWTHW...TT], preserving frequency continuity.
    args:
        x: (3, bs, seq_len, head_dim // 2)
        mrope_section: (3,)
    returns:
        x_t: (bs, seq_len, head_dim // 2)
    """
    freqs_t = freqs[0]  # just overwrite the first dimension T
    for dim, offset in enumerate((1, 2), start=1):  # H, W
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]
    return freqs_t
```

### 2.4 Other Shared Features

- **Pre-norm architecture** with residual connections (`residual + attention(norm(x))`)
- **QK-norm**: both apply RMSNorm per head dimension to queries and keys in text attention
- **No bias** in text attention projections (`attention_bias=False`)

---

## 3. Difference 1 — Hybrid Decoder (THE Biggest Difference)

This is the most significant architectural distinction: Qwen3.5 uses a **hybrid** decoder mixing linear and full attention, while Qwen3-VL uses **only** full attention.

### 3a. Layer Dispatch: Uniform vs Hybrid

**Qwen3-VL**: All 36 layers are uniform `Qwen3VLTextDecoderLayer`, each containing standard full attention:

```python
# Qwen3VLTextDecoderLayer (modeling_qwen3_vl.py, line 519)
class Qwen3VLTextDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3VLTextAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3VLTextMLP(config)
        self.input_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

**Qwen3.5**: Layers alternate between `linear_attention` and `full_attention` with a pattern determined by `full_attention_interval=4`. The layer type list is generated as follows:

```python
# configuration_qwen3_5.py, line 181
self.layer_types = layer_types
if self.layer_types is None:
    interval_pattern = kwargs.get("full_attention_interval", 4)
    self.layer_types = [
        "linear_attention" if bool((i + 1) % interval_pattern) else "full_attention"
        for i in range(self.num_hidden_layers)
    ]
```

For 32 layers with interval 4, this produces: `[lin, lin, lin, full, lin, lin, lin, full, ...]` — 24 linear + 8 full attention layers, with a full attention layer every 4th position.

The decoder layer conditionally instantiates either `linear_attn` or `self_attn`:

```python
# Qwen3_5DecoderLayer (modeling_qwen3_5.py, line 827)
class Qwen3_5DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, layer_idx)
        self.mlp = Qwen3_5MLP(config, config.intermediate_size)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                position_ids=None, past_key_values=None, cache_position=None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states, cache_params=past_key_values,
                cache_position=cache_position, attention_mask=attention_mask)
        elif self.layer_type == "full_attention":
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states, attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=past_key_values,
                cache_position=cache_position, position_embeddings=position_embeddings,
                **kwargs)

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
```

### 3b. Background: Why Linear Attention?

To understand why Qwen3.5 makes this choice, a brief primer on linear attention:

**Standard (full) attention** computes `softmax(QK^T / sqrt(d)) V` — this requires materializing the full N x N attention matrix, making both memory and compute O(N^2) in sequence length. During autoregressive decoding, you keep a growing KV cache that is O(N) per layer.

**Linear attention** removes the softmax, reformulating attention as `phi(Q) (phi(K)^T V)`. The key insight is *associativity*: instead of computing `(QK^T)V` (N x N intermediate), you compute `Q(K^T V)` (d x d intermediate). This allows maintaining a fixed-size "recurrent state" matrix S of shape `(key_dim x value_dim)`, updated as `S_t = S_{t-1} + k_t v_t^T`, with output `o_t = q_t S_t`. Memory per token: O(1). No growing KV cache.

**The trade-off**: Removing softmax loses the sharp, selective attention patterns that full attention provides. The model can no longer "look up" specific tokens precisely — the recurrent state is a lossy compression of the past.

**Hybrid approach** (what Qwen3.5 does): Use linear attention for most layers (cheap, O(1) state) but keep full attention every 4th layer (expensive, but preserves precise token lookup). This gives sub-quadratic overall cost while maintaining some full-attention "checkpoints" for precise retrieval.

### 3c. Deep Dive: Qwen3_5GatedDeltaNet (The Linear Attention Module)

The Gated Delta Rule improves on basic linear attention with three mechanisms:

#### (i) Causal Conv1d — Local Context Mixing

Before computing Q/K/V, the projected states pass through a depthwise causal convolution (kernel_size=4). This gives each position access to its 3 preceding tokens, providing short-range context that the recurrent state struggles to capture precisely:

```python
# Qwen3_5GatedDeltaNet.__init__ (modeling_qwen3_5.py, line 463)
self.conv1d = nn.Conv1d(
    in_channels=self.conv_dim,    # key_dim * 2 + value_dim
    out_channels=self.conv_dim,
    bias=False,
    kernel_size=self.conv_kernel_size,  # 4
    groups=self.conv_dim,          # depthwise
    padding=self.conv_kernel_size - 1,
)

# Forward path (modeling_qwen3_5.py, line 557)
if self.causal_conv1d_fn is not None:
    mixed_qkv = self.causal_conv1d_fn(
        x=mixed_qkv,
        weight=self.conv1d.weight.squeeze(1),
        bias=self.conv1d.bias,
        activation=self.activation,  # "silu"
        seq_idx=None,
    )
else:
    mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])
```

#### (ii) Gated Delta Rule — The Recurrence

Instead of the naive `S += k v^T` (which would grow unboundedly), the delta rule updates as:

```
S_t = decay_t * S_{t-1} + k_t * (beta_t * (v_t - S_{t-1}^T k_t))^T
```

Where:
- `decay_t = exp(-A * softplus(a_t + dt_bias))` — exponential forgetting (state decays over time)
- `beta_t = sigmoid(b_t)` — controls how much of the "error" to write
- The "delta" is `v_t - S^T k_t`: the difference between the desired value and what the state already predicts for this key — like a Hebbian/delta learning rule applied at each step

From the projections:

```python
# Qwen3_5GatedDeltaNet.__init__ (modeling_qwen3_5.py, line 505-508)
self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)  # beta
self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)  # decay gate
```

The recurrent loop from `torch_recurrent_gated_delta_rule` (modeling_qwen3_5.py, line 425):

```python
for i in range(sequence_length):
    q_t = query[:, :, i]
    k_t = key[:, :, i]
    v_t = value[:, :, i]
    g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
    beta_t = beta[:, :, i].unsqueeze(-1)

    last_recurrent_state = last_recurrent_state * g_t           # decay
    kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(-2)  # retrieve: S^T k
    delta = (v_t - kv_mem) * beta_t                              # compute error
    last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)  # update
    core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(-2)  # read: q S
```

And the decay is computed as (modeling_qwen3_5.py, line 585):

```python
beta = b.sigmoid()
g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
```

#### (iii) Chunk-wise Computation — Efficient Training

During training/prefill (full sequence available), processing token-by-token is slow. `torch_chunk_gated_delta_rule` splits the sequence into chunks of 64 tokens and processes them with a mix of intra-chunk matrix operations and inter-chunk recurrent state propagation. During autoregressive decoding (single token), it switches to the token-by-token recurrent path:

```python
# Qwen3_5GatedDeltaNet.forward (modeling_qwen3_5.py, line 590)
if not use_precomputed_states:
    core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
        query, key, value, g=g, beta=beta,
        initial_state=None, output_final_state=cache_params is not None,
        use_qk_l2norm_in_kernel=True)
else:
    core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
        query, key, value, g=g, beta=beta,
        initial_state=recurrent_state, output_final_state=cache_params is not None,
        use_qk_l2norm_in_kernel=True)
```

#### (iv) Gated Output Normalization

The output passes through `Qwen3_5RMSNormGated`: RMSNorm followed by element-wise multiplication with `SiLU(gate)`, where the gate comes from a separate projection:

```python
# Qwen3_5GatedDeltaNet.forward (modeling_qwen3_5.py, line 537, 621)
z = self.in_proj_z(hidden_states)  # gate projection
# ... after recurrence ...
core_attn_out = self.norm(core_attn_out, z)  # RMSNorm then * SiLU(z)

# Qwen3_5RMSNormGated.forward (modeling_qwen3_5.py, line 269)
def forward(self, hidden_states, gate=None):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    hidden_states = self.weight * hidden_states.to(input_dtype)
    hidden_states = hidden_states * F.silu(gate.to(torch.float32))
    return hidden_states.to(input_dtype)
```

### 3d. Head Configuration in Linear Attention

The linear attention uses a different head configuration than the full attention:

- `linear_num_key_heads=16`, `linear_num_value_heads=32`
- `linear_key_head_dim=128`, `linear_value_head_dim=128`

When value heads > key heads (32 > 16), Q and K are `repeat_interleave`d to match:

```python
# modeling_qwen3_5.py, line 586
if self.num_v_heads // self.num_k_heads > 1:
    query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
    key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
```

Recurrent state shape per head: `(key_head_dim x value_head_dim)` = `(128 x 128)` = 16K floats — fixed regardless of sequence length.

---

## 4. Difference 2 — Full Attention Design

Both models use GQA at a 4x grouping ratio and QK-norm. The differences are in head count, head dimension, and query gating.

**Qwen3.5**: `q_proj` outputs `num_heads * head_dim * 2` (double!), split into query + gate. After attention, the output is modulated by `sigmoid(gate)`:

```python
# Qwen3_5Attention.__init__ (modeling_qwen3_5.py, line 716)
class Qwen3_5Attention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.head_dim = getattr(config, "head_dim",
                                config.hidden_size // config.num_attention_heads)  # 256
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads  # 16/4 = 4
        self.scaling = self.head_dim ** -0.5
        # NOTE: 2x output dimension for query gating
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim * 2,
            bias=config.attention_bias)  # 4096 -> 16 * 256 * 2 = 8192
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias)  # 4096 -> 4 * 256 = 1024
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias)  # 4096 -> 4 * 256 = 1024
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size,
            bias=config.attention_bias)  # 4096 -> 4096
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

# Qwen3_5Attention.forward (modeling_qwen3_5.py, line 752)
    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, cache_position=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Split q_proj output into query and gate
        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        # ... attention computation ...
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)  # <-- gating!
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
```

**Qwen3-VL**: Standard attention without gating:

```python
# Qwen3VLTextAttention.__init__ (modeling_qwen3_vl.py, line 431)
class Qwen3VLTextAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.head_dim = getattr(config, "head_dim",
                                config.hidden_size // config.num_attention_heads)  # 128
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads  # 32/8 = 4
        self.scaling = self.head_dim ** -0.5
        # Standard q_proj — no doubling
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim,
            bias=config.attention_bias)  # 4096 -> 32 * 128 = 4096
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias)  # 4096 -> 8 * 128 = 1024
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias)  # 4096 -> 8 * 128 = 1024
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size,
            bias=config.attention_bias)  # 4096 -> 4096
        self.q_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

# Qwen3VLTextAttention.forward (modeling_qwen3_vl.py, line 471)
    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, cache_position=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # No gating — straightforward projection
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        # ... attention computation ...
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)  # <-- no gate!
        return attn_output, attn_weights
```

The sigmoid gating in Qwen3.5 modulates how much the attention output contributes to the residual stream. This gives the model per-position, per-dimension control over information flow — particularly useful in a hybrid architecture where full attention layers serve as "information checkpoints" between linear attention layers.

---

## 5. Difference 3 — RMSNorm Parameterization

A subtle but principled difference in how the normalization is parameterized:

**Qwen3.5**: Weight initialized to zeros, forward uses `(1.0 + weight)`:

```python
# Qwen3_5RMSNorm (modeling_qwen3_5.py, line 807)
class Qwen3_5RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))  # <-- zeros

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())  # <-- 1-centered
        return output.type_as(x)
```

Confirmed in `_init_weights` (modeling_qwen3_5.py, line 907):

```python
elif isinstance(module, Qwen3_5RMSNorm):
    init.zeros_(module.weight)
```

**Qwen3-VL**: Weight initialized to ones, forward uses `weight * norm(x)`:

```python
# Qwen3VLTextRMSNorm (modeling_qwen3_vl.py, line 381)
class Qwen3VLTextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # <-- ones
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)  # <-- standard
```

Both are identity at initialization. The `(1 + w)` parameterization means gradients flow through the `+1` offset, which can stabilize training when the model needs the norm to stay close to identity. Different gradient dynamics during training, but mathematically equivalent at init.

---

## 6. Difference 4 — Positional Encoding

Both use 3D MRoPE with the identical `apply_interleaved_mrope` function (see Section 2.3). The differences are in **how much** of the head dimension is rotated and the **frequency base**.

**Qwen3.5**: `partial_rotary_factor=0.25` — only 64 of 256 head dims are rotated:

```python
# Qwen3_5TextRotaryEmbedding.compute_default_rope_parameters (modeling_qwen3_5.py, line 194)
@staticmethod
def compute_default_rope_parameters(config=None, device=None, seq_len=None):
    base = config.rope_parameters["rope_theta"]
    partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)  # 256 * 0.25 = 64
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, ...) / dim))
    return inv_freq, 1.0  # inv_freq has dim/2 = 32 elements
```

The `apply_rotary_pos_emb` function in Qwen3.5 splits the tensor, rotates only the first part, and concatenates back:

```python
# apply_rotary_pos_emb (modeling_qwen3_5.py, line 636)
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    rotary_dim = cos.shape[-1]       # 64 (from partial_rotary_factor)
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]  # split at 64
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q_embed = torch.cat([q_embed, q_pass], dim=-1)  # [64 rotated | 192 unrotated]
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed
```

**Qwen3-VL**: Full rotation — all 128 head dims are rotated:

```python
# Qwen3VLTextRotaryEmbedding.compute_default_rope_parameters (modeling_qwen3_vl.py, line 312)
@staticmethod
def compute_default_rope_parameters(config=None, device=None, seq_len=None):
    base = config.rope_parameters["rope_theta"]
    dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    # No partial_rotary_factor — full dim = 128
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, ...) / dim))
    return inv_freq, 1.0  # inv_freq has dim/2 = 64 elements
```

And the `apply_rotary_pos_emb` in Qwen3-VL rotates the full tensor:

```python
# apply_rotary_pos_emb (modeling_qwen3_vl.py, line 402)
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)  # no split — full rotation
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

**Why partial rotation?** In Qwen3.5's hybrid architecture, only 8 of 32 layers use full attention with position encoding. The 24 linear attention layers are position-free (no RoPE). By leaving 192 of 256 dims unrotated in the full attention layers, Qwen3.5 preserves position-agnostic content dimensions that are compatible with how linear attention processes information. The mrope_section also differs: `[11, 11, 10]` for Qwen3.5 vs `[24, 20, 20]` for Qwen3-VL, reflecting the different rotary dimensions available.

---

## 7. Difference 5 — Vision-Language Fusion (DeepStack)

Both models use `masked_scatter` to replace placeholder token embeddings with vision features. But Qwen3-VL additionally uses **DeepStack** — multi-scale visual features injected into early text decoder layers.

### Qwen3-VL: DeepStack

The vision model extracts features from intermediate layers [8, 16, 24] via dedicated `PatchMerger` modules:

```python
# Qwen3VLVisionModel.__init__ (modeling_qwen3_vl.py, line 621)
class Qwen3VLVisionModel(Qwen3VLPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # ... patch_embed, pos_embed, rotary_pos_emb, blocks ...
        self.merger = Qwen3VLVisionPatchMerger(config=config, use_postshuffle_norm=False)

        self.deepstack_visual_indexes = config.deepstack_visual_indexes  # [8, 16, 24]
        self.deepstack_merger_list = nn.ModuleList([
            Qwen3VLVisionPatchMerger(
                config=config,
                use_postshuffle_norm=True,  # <-- postshuffle norm for deepstack
            )
            for _ in range(len(config.deepstack_visual_indexes))
        ])
```

During the forward pass, features are extracted at the specified layer indices:

```python
# Qwen3VLVisionModel.forward (modeling_qwen3_vl.py, line 794)
deepstack_feature_lists = []
for layer_num, blk in enumerate(self.blocks):
    hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens,
                        position_embeddings=position_embeddings, **kwargs)
    if layer_num in self.deepstack_visual_indexes:
        deepstack_feature = self.deepstack_merger_list[
            self.deepstack_visual_indexes.index(layer_num)](hidden_states)
        deepstack_feature_lists.append(deepstack_feature)

merged_hidden_states = self.merger(hidden_states)
return BaseModelOutputWithDeepstackFeatures(
    last_hidden_state=hidden_states,
    pooler_output=merged_hidden_states,
    deepstack_features=deepstack_feature_lists,
)
```

These multi-scale features are then additively injected into the first few text decoder layers:

```python
# Qwen3VLTextModel.forward (modeling_qwen3_vl.py, line 910)
for layer_idx, decoder_layer in enumerate(self.layers):
    layer_outputs = decoder_layer(hidden_states, ...)
    hidden_states = layer_outputs

    # add visual features to the hidden states of first several layers
    if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
        hidden_states = self._deepstack_process(
            hidden_states, visual_pos_masks, deepstack_visual_embeds[layer_idx])

# Qwen3VLTextModel._deepstack_process (modeling_qwen3_vl.py, line 937)
def _deepstack_process(self, hidden_states, visual_pos_masks, visual_embeds):
    visual_pos_masks = visual_pos_masks.to(hidden_states.device)
    visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
    hidden_states = hidden_states.clone()
    local_this = hidden_states[visual_pos_masks, :] + visual_embeds  # additive injection
    hidden_states[visual_pos_masks, :] = local_this
    return hidden_states
```

The top-level model passes deepstack features from vision to language:

```python
# Qwen3VLModel.forward (modeling_qwen3_vl.py, line 1274)
outputs = self.language_model(
    input_ids=None,
    position_ids=position_ids,
    attention_mask=attention_mask,
    past_key_values=past_key_values,
    inputs_embeds=inputs_embeds,
    cache_position=cache_position,
    visual_pos_masks=visual_pos_masks,         # <-- deepstack args
    deepstack_visual_embeds=deepstack_visual_embeds,  # <-- deepstack args
    **kwargs,
)
```

### Qwen3.5: No DeepStack

The Qwen3.5 vision model has a plain loop with no intermediate feature extraction:

```python
# Qwen3_5VisionModel.__init__ (modeling_qwen3_5.py, line 1095)
class Qwen3_5VisionModel(Qwen3_5PreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # ... patch_embed, pos_embed, rotary_pos_emb ...
        self.blocks = nn.ModuleList([Qwen3_5VisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen3_5VisionPatchMerger(config=config, use_postshuffle_norm=False)
        # No deepstack_merger_list!

# Qwen3_5VisionModel.forward (modeling_qwen3_5.py, line 1255)
for blk in self.blocks:
    hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens,
                        position_embeddings=position_embeddings, **kwargs)

merged_hidden_states = self.merger(hidden_states)
return BaseModelOutputWithPooling(
    last_hidden_state=hidden_states,
    pooler_output=merged_hidden_states,
    # No deepstack_features!
)
```

And the Qwen3.5 top-level model does not pass deepstack args:

```python
# Qwen3_5Model.forward (modeling_qwen3_5.py, line 1692)
outputs = self.language_model(
    input_ids=None,
    position_ids=position_ids,
    attention_mask=attention_mask,
    past_key_values=past_key_values,
    inputs_embeds=inputs_embeds,
    cache_position=cache_position,
    # No visual_pos_masks or deepstack_visual_embeds!
    **kwargs,
)
```

The vision-to-text projection dimension also differs: `out_hidden_size=3584` for Qwen3.5 vs `4096` for Qwen3-VL-8B (matching each model's text hidden_size).

---

## 8. Difference 6 — Custom Cache

**Qwen3-VL** uses the standard `DynamicCache`:

```python
# Qwen3VLTextModel.forward (modeling_qwen3_vl.py, line 872)
if use_cache and past_key_values is None and not torch.jit.is_tracing():
    past_key_values = DynamicCache(config=self.config)
```

**Qwen3.5** requires a custom `Qwen3_5DynamicCache` with four parallel stores:

```python
# Qwen3_5DynamicCache (modeling_qwen3_5.py, line 67)
class Qwen3_5DynamicCache:
    """
    A dynamic cache that can handle both the attention cache (which has a seq_len dimension)
    and the linear attention cache (which has a constant shape regardless of seq_len).
    """
    def __init__(self, config):
        super().__init__()
        self.layer_types = config.layer_types
        self.transformer_layers = [
            i for i in range(config.num_hidden_layers)
            if self.layer_types[i] == "full_attention"
        ]
        self.last_linear_layer = (
            len(self.layer_types) - 1 - self.layer_types[::-1].index("linear_attention")
        )

        # Four parallel stores, all initialized to None (lazy)
        self.conv_states = [None for _ in range(config.num_hidden_layers)]
        self.recurrent_states = [None for _ in range(config.num_hidden_layers)]
        self.key_cache = [None for _ in range(config.num_hidden_layers)]
        self.value_cache = [None for _ in range(config.num_hidden_layers)]
```

Used as:

```python
# Qwen3_5TextModel.forward (modeling_qwen3_5.py, line 1327)
if use_cache and past_key_values is None:
    past_key_values = Qwen3_5DynamicCache(config=self.config)
```

The key insight: linear attention layers don't produce KV pairs — they maintain fixed-size recurrent state (`recurrent_states`) and convolution state (`conv_states`). Full attention layers use `key_cache` and `value_cache` as usual. The custom cache manages both types in parallel.

---

## 9. Synthesis & Design Philosophy

**Qwen3.5** is **efficiency-oriented**:
- Hybrid attention for sub-quadratic inference cost (24 of 32 layers use O(1) linear attention)
- Sigmoid gating on full attention output for selective information flow
- Partial rotary encoding (only 25% of dims) to preserve position-agnostic content channels
- No DeepStack — simpler vision-language fusion pipeline
- Custom cache combining KV + recurrent state

**Qwen3-VL** is **representation-oriented**:
- All 36 layers use full attention for maximum expressivity
- DeepStack for rich multi-scale visual grounding (features from vision layers 8, 16, 24 injected into text layers 0, 1, 2)
- 262K context window (8x larger than Qwen3.5's 32K)
- Full rotary encoding over all head dimensions
- Standard DynamicCache — straightforward KV management

**What they share**: Both use the same GQA grouping ratio (4x), the same MLP structure (SwiGLU with 3x expansion), the same MRoPE interleaving scheme, QK-norm, and nearly identical vision encoders. The differences are in *how attention works*, not in the feedforward path.

The hybrid approach in Qwen3.5 represents a bet that most layers don't need the full O(N^2) attention pattern — a fixed-size recurrent state is sufficient for most information routing, with periodic full-attention layers serving as precise retrieval checkpoints. Qwen3-VL instead bets on maximum expressivity at every layer, compensating for the higher per-layer cost with architectural features like DeepStack that make each layer's computation more visually grounded.

---

## Appendix: Why Is Qwen3.5-9B Larger Than Qwen3-VL-8B?

At first glance, Qwen3.5 looks "smaller" in every way: fewer layers (32 vs 36), fewer attention heads (16Q/4KV vs 32Q/8KV), smaller vision output (3584 vs 4096). Yet it has **more** parameters. The explanation comes down to two things the summary table doesn't show: vocabulary size and the cost of linear attention modules.

### Per-Layer Attention Parameters

**Qwen3_5GatedDeltaNet** (linear attention, 24 layers):

| Module | Dimensions | Params |
|---|---|---|
| `in_proj_qkv` | 4096 → 2048\*2 + 4096 = 8192 | 33.55M |
| `in_proj_z` (gate) | 4096 → 4096 | 16.78M |
| `in_proj_a` (decay) | 4096 → 32 | 0.13M |
| `in_proj_b` (beta) | 4096 → 32 | 0.13M |
| `conv1d` (depthwise, k=4) | 8192 channels | 0.03M |
| `out_proj` | 4096 → 4096 | 16.78M |
| **Total** | | **67.4M** |

**Qwen3_5Attention** (full attention, 8 layers):

| Module | Dimensions | Params |
|---|---|---|
| `q_proj` (2x for gate) | 4096 → 16\*256\*2 = 8192 | 33.55M |
| `k_proj` | 4096 → 4\*256 = 1024 | 4.19M |
| `v_proj` | 4096 → 4\*256 = 1024 | 4.19M |
| `o_proj` | 4096 → 4096 | 16.78M |
| **Total** | | **58.7M** |

**Qwen3VLTextAttention** (36 layers):

| Module | Dimensions | Params |
|---|---|---|
| `q_proj` | 4096 → 32\*128 = 4096 | 16.78M |
| `k_proj` | 4096 → 8\*128 = 1024 | 4.19M |
| `v_proj` | 4096 → 8\*128 = 1024 | 4.19M |
| `o_proj` | 4096 → 4096 | 16.78M |
| **Total** | | **41.9M** |

The linear attention module is **60% heavier** than a standard attention layer (67.4M vs 41.9M), due to the extra gate projection (`in_proj_z`), the combined QKV projection being wider, and the decay/beta projections.

### Full Parameter Budget

Both models use `tie_word_embeddings=False`, so `embed_tokens` and `lm_head` are separate parameters.

| Component | Qwen3.5-9B | Qwen3-VL-8B | Delta |
|---|---|---|---|
| **Embeddings** (vocab × 4096) | 248320 × 4096 = 1.02B | 151936 × 4096 = 0.62B | +0.39B |
| **LM head** (4096 × vocab) | 248320 × 4096 = 1.02B | 151936 × 4096 = 0.62B | +0.39B |
| **Attention** | 8×58.7M + 24×67.4M = 2.09B | 36×41.9M = 1.51B | +0.58B |
| **MLP** (3 × 4096 × 12288/layer) | 32 × 150.99M = 4.83B | 36 × 150.99M = 5.44B | −0.60B |
| **Vision encoder** (27 blocks) | ~0.45B (no DeepStack) | ~0.58B (+ 3 DeepStack mergers) | −0.12B |
| **Norms, biases, etc.** | ~5M | ~5M | ~0 |
| **Total** | **~9.4B** | **~8.8B** | **+~0.6B** |

### The Three Drivers

1. **Vocabulary: 248K vs 152K** — The 96K extra tokens cost ~790M params across embeddings + lm_head. This single factor accounts for most of the gap.

2. **Linear attention overhead** — Each GatedDeltaNet layer (67.4M) is heavier than a standard attention layer (41.9M). The gate projection `in_proj_z` alone adds 16.8M per layer × 24 layers = 403M of "hidden" cost.

3. **Gated q_proj** — The 8 full attention layers in Qwen3.5 double their q_proj output dimension for sigmoid gating (4096 → 8192 instead of → 4096), adding ~134M.

These three factors contribute +1.36B, which more than offsets the 4 fewer MLP layers (−0.60B) and the simpler vision encoder (−0.12B), netting the ~0.6B difference.

The takeaway: "9B vs 8B" is not about the decoder being bigger — it's largely about the vocabulary and the per-layer cost of the linear attention machinery that makes Qwen3.5's inference more efficient.
