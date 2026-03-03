import time
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")

prompt = "Write a Python class implementing a binary search tree with insert, search, and delete"

start = time.time()
response = generate(model, tokenizer, prompt=prompt, max_tokens=500)
elapsed = time.time() - start

tokens = len(tokenizer.encode(response))
tps = tokens / elapsed
print(f"{tokens} tokens in {elapsed:.1f}s = {tps:.1f} tok/s")