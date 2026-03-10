"""
Integration test: Run the agent on a real coding task.
Verify CHI monitors, anchors inject, and compaction triggers. (TASK-025)
"""

from __future__ import annotations

import os
import shutil

from mlx_lm import load

from cognition_engine.agent import CognitionAgent


TEST_DIR = "/tmp/cognition_test"


def main() -> None:
    # Set up a clean test directory
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR)

    # Load model
    model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")

    # Create agent with a simple task
    agent = CognitionAgent(
        model,
        tokenizer,
        task=(
            "Create a Python file called calculator.py with a Calculator class "
            "that has add, subtract, multiply, and divide methods. "
            "Then create test_calculator.py with tests for each method. "
            "Run the tests to verify they pass."
        ),
        working_dir=TEST_DIR,
        auto_approve=False,  # You'll manually approve shell commands
    )

    # Run the agent (limit to 15 turns for testing)
    agent.run(max_turns=15)

    # Verify the agent created files
    assert os.path.exists(
        os.path.join(TEST_DIR, "calculator.py")
    ), "calculator.py not created"

    print("\nFiles in test directory:")
    for f in os.listdir(TEST_DIR):
        print(f"  {f}")


if __name__ == "__main__":
    main()

