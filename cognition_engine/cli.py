"""
cognition-engine CLI — v0.1

Usage:
    python -m cognition_engine.cli --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit

    Then type your task. The model responds. CHI score shown after each turn.
"""
import argparse
from mlx_lm import load, generate
from cognition_engine.chi import CHIMonitor


def main():
    parser = argparse.ArgumentParser(description="Cognition Engine — AI that monitors its own brain")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
                        help="Model path or HuggingFace repo")
    parser.add_argument("--max-tokens", type=int, default=500,
                        help="Max tokens per response")
    args = parser.parse_args()

    print("Loading model...")
    model, tokenizer = load(args.model)
    print("Model loaded. Type your task. Ctrl+C to quit.\n")

    monitor = None
    conversation = ""

    while True:
        try:
            user_input = input("\033[94mYou:\033[0m ")  # Blue prompt
            if not user_input.strip():
                continue

            # Initialize CHI monitor on first input
            if monitor is None:
                monitor = CHIMonitor(model, tokenizer, user_input)

            # Build conversation
            conversation += f"\nUser: {user_input}\nAssistant: "

            # Generate
            response = generate(model, tokenizer, prompt=conversation,
                               max_tokens=args.max_tokens)
            conversation += response

            # Show response
            print(f"\033[92mAssistant:\033[0m {response}")  # Green

            # Compute CHI
            result = monitor.update(response, conversation)
            level = monitor.get_intervention_level()

            # Color-coded CHI display
            chi = result["chi_score"]
            if chi > 0.8:
                color = "\033[92m"  # Green
            elif chi > 0.5:
                color = "\033[93m"  # Yellow
            else:
                color = "\033[91m"  # Red

            print(f"\n{color}[CHI: {chi:.3f} | Level {level}]\033[0m "
                  f"E={result['entropy_health']:.2f} "
                  f"R={result['repetition_health']:.2f} "
                  f"D={result['drift_health']:.2f}\n")

        except KeyboardInterrupt:
            print("\n\nSession ended.")
            if monitor and monitor.chi_history:
                print(f"Turns: {len(monitor.chi_history)}")
                print(f"Final CHI: {monitor.chi_history[-1]:.3f}")
                print(f"Min CHI: {min(monitor.chi_history):.3f}")
            break


if __name__ == "__main__":
    main()
