"""
TASK-007: Extract attention weights from the self-attention module.
Wraps the Attention module to compute attention manually and capture
the softmax attention weights (before output projection).
"""
from typing import Any, List, Optional

import mlx.core as mx
from mlx_lm import load, generate

# Use base for causal mask when mask is "causal"
from mlx_lm.models.base import create_causal_mask


class _CaptureAttention:
    """
    Wraps an Attention module; on each __call__ runs the same forward but
    computes attention manually and appends the softmax attention weights
    (shape batch, n_heads, seq_len, seq_len) to captured_weights.
    """

    def __init__(self, attn_module, captured_weights: List):
        self._attn = attn_module
        self._captured = captured_weights

    def __call__(
        self,
        x: mx.array,
        mask: Optional[Any] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape
        n_heads = self._attn.n_heads
        n_kv_heads = self._attn.n_kv_heads
        scale = self._attn.scale

        queries = self._attn.q_proj(x)
        keys = self._attn.k_proj(x)
        values = self._attn.v_proj(x)

        queries = queries.reshape(B, L, n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self._attn.rope(queries, offset=cache.offset)
            keys = self._attn.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self._attn.rope(queries)
            keys = self._attn.rope(keys)

        # GQA: repeat keys/values to n_heads if needed
        if n_kv_heads < n_heads:
            repeat = n_heads // n_kv_heads
            keys = mx.repeat(keys, repeat, axis=1)
            values = mx.repeat(values, repeat, axis=1)

        # scores: (B, n_heads, L, L)
        scores = mx.matmul(queries, keys.transpose(0, 1, 3, 2)) * scale

        if mask is not None:
            if mask == "causal":
                mask_arr = create_causal_mask(L)
                mask_arr = mask_arr[None, None, :, :]
            else:
                mask_arr = mask
            if mask_arr.dtype == mx.bool_:
                scores = mx.where(
                    mask_arr, scores, mx.finfo(scores.dtype).min
                )
            else:
                scores = scores + mask_arr

        weights = mx.softmax(scores, axis=-1, precise=True)
        self._captured.append(weights)

        output = mx.matmul(weights, values)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self._attn.o_proj(output)


def extract_attention_weights(
    model,
    tokenizer,
    prompt,
    layer_idx: int = 16,
    max_tokens: int = 50,
):
    """
    Run inference and capture attention weights from the self-attention
    module at the given layer. Uses one full forward (no cache) on
    prompt + generated sequence.

    Returns:
        (generated_text, attention_weights) where attention_weights has
        shape (1, n_heads, seq_len, seq_len). Example: (1, 28, 46, 46) for
        batch 1, 28 heads, 46 tokens.
    """
    captured = []
    layers = model.model.layers
    block = layers[layer_idx]
    original_attn = block.self_attn
    block.self_attn = _CaptureAttention(original_attn, captured)

    try:
        if isinstance(prompt, str):
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        else:
            prompt_tokens = list(prompt)
        prompt_tokens = mx.array(prompt_tokens)

        generated_text = generate(
            model, tokenizer, prompt_tokens.tolist(), max_tokens=max_tokens
        )

        gen_tokens = tokenizer.encode(generated_text, add_special_tokens=False)
        full_tokens = prompt_tokens.tolist() + gen_tokens
        full_tokens = mx.array(full_tokens)

        captured.clear()
        _ = model(full_tokens[None], cache=None)
        mx.eval(_)

        weights = captured[0] if captured else None
        if weights is not None:
            mx.eval(weights)

        return generated_text, weights
    finally:
        block.self_attn = original_attn


if __name__ == "__main__":
    from cognition_engine.inference import load

    model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
    prompt = "Write a short Python function."
    text, attn_weights = extract_attention_weights(
        model, tokenizer, prompt, layer_idx=16, max_tokens=40
    )
    print("Generated (first 120 chars):", (text[:120] + "..." if len(text) > 120 else text))
    print("Attention weights shape:", attn_weights.shape if attn_weights is not None else None)
