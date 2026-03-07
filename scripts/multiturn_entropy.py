"""
Multi-turn entropy tracking experiment (TASK-009).

The model is given a coding project and asked to build it step by step
over many turns. We track entropy at each turn to find where degradation begins.
"""

from engine.entropy import compute_token_entropy, get_logits_for_sequence
from mlx_lm import load, generate
import json
import mlx.core as mx

# Strategy: Build a conversation context manually.
# After each model response, append it to the context
# and ask the next question. This simulates real multi-turn use.

SYSTEM_PROMPT = "You are a coding assistant. Follow instructions precisely."

TASK_SEQUENCE = [
    "Create a Python class called TaskManager with an __init__ method that stores a list of tasks.",
    "Add a method add_task(title, priority) that adds a task dict to the list.",
    "Add a method get_tasks_by_priority(priority) that filters and returns matching tasks.",
    "Add a method complete_task(title) that marks a task as completed.",
    "Add a method get_statistics() that returns count of total, completed, and pending tasks.",
    "Now add due dates to tasks. Modify add_task to accept an optional due_date parameter.",
    "Add a method get_overdue_tasks() that returns tasks past their due date.",
    "Add a method export_to_json(filepath) that saves all tasks to a JSON file.",
    "Add a method import_from_json(filepath) that loads tasks from a JSON file.",
    "Add input validation to add_task: title must be non-empty string, priority must be 1-5.",
    "Add a method sort_tasks(by='priority', reverse=False) that sorts the task list.",
    "Add a method search_tasks(keyword) that searches task titles and returns matches.",
    "Refactor the class to use a dataclass for individual tasks instead of dicts.",
    "Add unit tests for add_task, complete_task, and get_statistics methods.",
    "Add error handling: raise custom TaskNotFoundError when completing non-existent task.",
    "Show me the complete, final version of the TaskManager class with all features.",
    "Now review your implementation. Are there any bugs? Fix them.",
    "Add type hints to every method in the class.",
    "Write a comprehensive docstring for the class and every public method.",
    "Create a CLI interface using argparse that wraps the TaskManager.",
]


def run_multiturn_experiment(model, tokenizer, turns=None):
    """
    Run a multi-turn conversation and track entropy at each turn.
    Returns: list of dicts with turn number, entropy stats, response length, etc.
    """
    if turns is None:
        turns = TASK_SEQUENCE

    conversation = SYSTEM_PROMPT + "\n"
    results = []

    for i, user_msg in enumerate(turns):
        # Build the full prompt (all history + new question)
        conversation += f"\nUser: {user_msg}\nAssistant: "

        # Generate response
        response = generate(model, tokenizer, prompt=conversation, max_tokens=300)

        # Compute entropy on the FULL conversation so far (including response)
        full_text = conversation + response
        logits = get_logits_for_sequence(model, tokenizer, full_text)
        entropy = compute_token_entropy(logits)
        mx.eval(entropy)

        # Track entropy of just the RESPONSE portion
        # (last N tokens, where N = number of tokens in response)
        response_tokens = len(tokenizer.encode(response))
        response_entropy = entropy[-response_tokens:] if response_tokens > 0 else entropy

        # MLX: eval then .item() for scalars
        mean_ent = mx.mean(response_entropy)
        max_ent = mx.max(response_entropy)
        min_ent = mx.min(response_entropy)
        mx.eval(mean_ent)
        mx.eval(max_ent)
        mx.eval(min_ent)

        result = {
            "turn": i,
            "prompt": user_msg[:80],
            "response_length": response_tokens,
            "context_length": len(tokenizer.encode(full_text)),
            "response_entropy_mean": mean_ent.item(),
            "response_entropy_max": max_ent.item(),
            "response_entropy_min": min_ent.item(),
        }
        results.append(result)

        print(
            f"Turn {i:3d} | ctx {result['context_length']:5d} tokens | "
            f"entropy mean={result['response_entropy_mean']:.4f} "
            f"max={result['response_entropy_max']:.4f}"
        )

        # Append response to conversation for next turn
        conversation += response + "\n"

        # Safety: stop if we're close to the model's context limit
        if result["context_length"] > 28000:  # ~28K, leave buffer for 32K limit
            print(f"Stopping at turn {i}: approaching context limit")
            break

    return results


if __name__ == "__main__":
    model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
    results = run_multiturn_experiment(model, tokenizer)

    # Save raw data
    with open("data/multiturn_entropy.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nCompleted {len(results)} turns. Data saved.")
