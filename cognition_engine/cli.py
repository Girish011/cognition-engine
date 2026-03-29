"""
cognition-engine CLI — v0.1

Usage:
    # Chat mode (interactive CHI)
    python -m cognition_engine chat --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit

    # Agent mode (tool-calling loop)
    python -m cognition_engine agent --task "Create hello.py" --dir /tmp/test

    # Resume interrupted agent session
    python -m cognition_engine agent --dir /tmp/test --resume

    # Status for a working directory
    python -m cognition_engine status --dir /tmp/test
"""

from __future__ import annotations

import argparse

from mlx_lm import generate, load

from cognition_engine.agent import CognitionAgent
from cognition_engine.chi import CHIMonitor
from cognition_engine.memory import WorkingMemory
from cognition_engine.session import SessionManager


def run_chat(args: argparse.Namespace) -> None:
    print("Loading model...")
    model, tokenizer = load(args.model)
    print("Model loaded. Type your task. Ctrl+C to quit.\n")

    conversation = ""
    monitor: CHIMonitor | None = None

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


def run_agent(args: argparse.Namespace) -> None:
    model, tokenizer = load(args.model)

    working_dir = args.dir
    session_mgr = SessionManager(working_dir)

    if args.resume:
        if not session_mgr.has_interrupted_session():
            print("No interrupted session found.")
            return
        session = session_mgr.load_session()
        if not session:
            print("Failed to load session.")
            return
        print(f"Resuming: {session.task[:80]}...")
        if session.chi_final is not None:
            print(
                f"Previous progress: {session.turn_count} turns, "
                f"CHI {session.chi_final:.3f}"
            )
        else:
            print(f"Previous progress: {session.turn_count} turns")
        task = session.task
    else:
        task = args.task
        if not task:
            task = input("Enter task: ")

    agent = CognitionAgent(
        model,
        tokenizer,
        task,
        working_dir=working_dir,
        auto_approve=args.auto_approve,
        enable_steering=args.enable_steering,
    )
    agent.run(max_turns=args.max_turns, max_tokens=args.max_tokens)


def show_status(args: argparse.Namespace) -> None:
    working_dir = args.dir
    session_mgr = SessionManager(working_dir)
    memory = WorkingMemory(working_dir)

    session = session_mgr.load_session()
    if session:
        print(f"Task: {session.task[:80]}")
        print(f"Status: {session.status}")
        print(f"Turns: {session.turn_count}")
        print(f"Last CHI: {session.chi_final if session.chi_final is not None else 'N/A'}")
        if session.files_modified:
            print(f"Files: {', '.join(session.files_modified)}")
        else:
            print("Files: (none)")
    else:
        print("No active session.")

    summary = memory.get_session_summary()
    if summary["total_notes"] > 0:
        print(f"\nMemory: {summary['total_notes']} notes")
        print(f"Categories: {summary['categories']}")
        if summary["files_referenced"]:
            print(f"Files referenced: {', '.join(summary['files_referenced'][:10])}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cognition Engine — tool-calling agent with CHI monitoring"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Chat mode
    chat_parser = subparsers.add_parser(
        "chat", help="Interactive chat with CHI monitoring"
    )
    chat_parser.add_argument(
        "--model", default="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
    )
    chat_parser.add_argument("--max-tokens", type=int, default=500)

    # Agent mode
    agent_parser = subparsers.add_parser("agent", help="Run coding agent on a task")
    agent_parser.add_argument("--task", required=False, help="Task description")
    agent_parser.add_argument("--dir", default=".", help="Working directory")
    agent_parser.add_argument(
        "--model", default="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
    )
    agent_parser.add_argument("--max-turns", type=int, default=50)
    agent_parser.add_argument(
        "--max-tokens",
        type=int,
        default=2000,
        help="Max new tokens per agent turn (tool calls, code blocks)",
    )
    agent_parser.add_argument(
        "--resume", action="store_true", help="Resume interrupted session"
    )
    agent_parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Auto-approve shell commands (dangerous)",
    )
    agent_parser.add_argument(
        "--enable-steering",
        action="store_true",
        help="Enable CHI-triggered activation steering (Level 2 intervention)",
    )

    # Status
    status_parser = subparsers.add_parser("status", help="Show project status")
    status_parser.add_argument("--dir", default=".")

    args = parser.parse_args()

    if args.command == "chat":
        run_chat(args)
        return
    if args.command == "agent":
        run_agent(args)
        return
    if args.command == "status":
        show_status(args)
        return

    # Backward compatible default: no subcommand => chat mode
    run_chat(
        argparse.Namespace(
            model=getattr(args, "model", "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"),
            max_tokens=getattr(args, "max_tokens", 500),
        )
    )


if __name__ == "__main__":
    main()
