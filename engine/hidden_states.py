import mlx.core as mx
from mlx_lm import load, generate


class _CaptureLayer:
    """Wraps a layer and appends its output to a list on each call."""

    def __init__(self, layer, captured):
        self._layer = layer
        self._captured = captured

    def __call__(self, x, mask=None, cache=None):
        out = self._layer(x, mask, cache)
        self._captured.append(out)
        return out


def extract_hidden_states(model, tokenizer, prompt, layer_idx=16, max_tokens=100):
    """
    Run inference and capture the hidden state at a specific layer.
    Returns: (generated_text, hidden_state_tensor)
    Hidden state shape: (1, seq_len, hidden_dim)
    """
    captured = []
    layers = model.model.layers
    original_layer = layers[layer_idx]
    layers[layer_idx] = _CaptureLayer(original_layer, captured)

    try:
        # Tokenize prompt
        if isinstance(prompt, str):
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        else:
            prompt_tokens = list(prompt)
        prompt_tokens = mx.array(prompt_tokens)

        # Generate text
        generated_text = generate(
            model, tokenizer, prompt_tokens.tolist(), max_tokens=max_tokens
        )

        # Full sequence: prompt + generated tokens, one forward for (1, seq_len, hidden_dim)
        gen_tokens = tokenizer.encode(generated_text, add_special_tokens=False)
        full_tokens = prompt_tokens.tolist() + gen_tokens
        full_tokens = mx.array(full_tokens)

        captured.clear()
        # Force evaluation so the forward runs and the layer capture executes (MLX is lazy)
        _ = model(full_tokens[None], cache=None)
        mx.eval(_)
        hidden = captured[0] if captured else None
        if hidden is not None:
            mx.eval(hidden)  # in-place evaluation; mx.eval() returns None

        return generated_text, hidden
    finally:
        layers[layer_idx] = original_layer


def extract_hidden_states_forward_only(model, tokenizer, text: str, layer_idx: int = 16):
    """
    Get hidden state for text via a single forward pass (no generation).
    Use when comparing two texts of possibly very different length (e.g. task vs long response).
    Returns: hidden_state_tensor shape (1, seq_len, hidden_dim), or None on failure.
    """
    captured = []
    layers = model.model.layers
    original_layer = layers[layer_idx]
    layers[layer_idx] = _CaptureLayer(original_layer, captured)

    try:
        if isinstance(text, str):
            tokens = tokenizer.encode(text, add_special_tokens=True)
        else:
            tokens = list(text)
        if not tokens:
            return None
        tokens = mx.array(tokens)[None]  # (1, seq_len)
        captured.clear()
        _ = model(tokens, cache=None)
        mx.eval(_)
        hidden = captured[0] if captured else None
        if hidden is not None:
            mx.eval(hidden)
        return hidden
    finally:
        layers[layer_idx] = original_layer


if __name__ == "__main__":
    model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
    text, states = extract_hidden_states(model, tokenizer, "Write hello world in Python")
    print("Generated:", text[:200] + ("..." if len(text) > 200 else ""))
    print("Hidden state shape:", states.shape if states is not None else None)