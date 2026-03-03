from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")

prompt = "Write a Python function that computes the nth fibonacci number"
response = generate(model, tokenizer, prompt=prompt, max_tokens=200)
print(response)