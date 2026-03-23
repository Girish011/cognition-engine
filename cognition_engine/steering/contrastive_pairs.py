"""
Contrastive pair generation for steering vector computation.

Each pair is (positive_prompt, negative_prompt) where:
- positive = model behaving well (focused, following format, coherent)
- negative = model behaving badly (drifting, ignoring format, repetitive)

The steering vector = mean(positive_activations) - mean(negative_activations)
"""

from __future__ import annotations

from typing import Any


ContrastivePair = dict[str, str]
PairSets = dict[str, list[ContrastivePair]]


# PAIR SET 1: "Stay on task" (anti-drift)
TASK_FOCUS_PAIRS: list[ContrastivePair] = [
    {
        "system": "You are a coding assistant. Task: Build a calculator in Python.",
        "positive": "I'll start by creating the Calculator class with add and subtract methods.\n\n[WRITE] calculator.py\n```\nclass Calculator:\n    def add(self, a, b):\n        return a + b\n\n    def subtract(self, a, b):\n        return a - b\n```",
        "negative": "That reminds me of a great story about mathematics. Did you know that Pythagoras was actually a philosopher? The history of calculators goes back to the abacus, which was invented in ancient Mesopotamia around 2400 BC. Speaking of ancient history...",
    },
    {
        "system": "You are a coding assistant. Task: Write unit tests for the API.",
        "positive": "I'll read the API code first to understand the endpoints.\n\n[READ] api.py\n\nNow I'll write tests for each endpoint.\n\n[WRITE] test_api.py\n```\nimport pytest\nfrom api import app\n```",
        "negative": "Testing is really important in software engineering. There are many testing frameworks available. Some popular ones include pytest, unittest, nose, and tox. Let me tell you about the differences between unit testing and integration testing...",
    },
    {
        "system": "You are a coding assistant. Task: Fix failing CI by updating lint config.",
        "positive": "I'll inspect the lint setup first.\n\n[READ] pyproject.toml\n\nThen I'll update only the lint section causing the CI failure.",
        "negative": "CI is a broad topic. Historically, Jenkins, CircleCI, and GitHub Actions have all evolved differently. Before touching config, let me explain DevOps maturity models in detail...",
    },
    {
        "system": "You are a coding assistant. Task: Add input validation to create_user().",
        "positive": "I'll update create_user() with explicit checks for email and password length, then add tests for invalid inputs.",
        "negative": "Sure, validation is important. First, here's a long explanation of why data quality matters across all businesses, from fintech to healthcare to e-commerce...",
    },
    {
        "system": "You are a coding assistant. Task: Implement caching for get_profile().",
        "positive": "I'll add a small in-memory cache keyed by user_id with TTL, then wrap get_profile() reads/writes around it.",
        "negative": "To implement caching correctly we should probably redesign the entire architecture, pick a distributed cache, benchmark cloud providers, and maybe rewrite in Rust first...",
    },
    {
        "system": "You are a coding assistant. Task: Rename old_config.json to settings.json and update references.",
        "positive": "I'll find all references, rename the file, and patch import/paths accordingly.\n\n[EXEC] rg \"old_config.json\" .",
        "negative": "I can do that later, but first here's a template of possible folder structures we might eventually use in a large monorepo...",
    },
    {
        "system": "You are a coding assistant. Task: Add pagination to GET /users endpoint.",
        "positive": "I'll update the endpoint to accept page/page_size, return metadata, and write tests for boundary cases.",
        "negative": "Pagination is one way to solve scale, but let's discuss sharding, CQRS, event sourcing, and eventual consistency before changing one endpoint...",
    },
    {
        "system": "You are a coding assistant. Task: Replace TODO stubs with working code in parser.py.",
        "positive": "I'll implement the parse_tokens() logic now and remove placeholder TODO comments only after behavior is complete.",
        "negative": "I'll keep the TODOs and add more notes for future work. Here's pseudo-implementation text without actual executable code...",
    },
    {
        "system": "You are a coding assistant. Task: Make tests deterministic by fixing random seeds.",
        "positive": "I'll set fixed seeds in both test setup and helper functions, then rerun the flaky test path.",
        "negative": "Randomness is a philosophical topic in computer science. Pseudorandom number generators have a rich history worth exploring first...",
    },
    {
        "system": "You are a coding assistant. Task: Add retry logic to HTTP client.",
        "positive": "I'll add bounded retries with exponential backoff for transient 5xx and timeout errors, then add unit tests.",
        "negative": "Instead of implementing retries now, let's brainstorm every possible reliability pattern and compare dozens of libraries abstractly...",
    },
]


# PAIR SET 2: "Follow tool format" (anti-hallucination)
TOOL_FORMAT_PAIRS: list[ContrastivePair] = [
    {
        "system": "You have tools: [READ] path, [WRITE] path, [EXEC] command, [THINK] thought.",
        "positive": "[THINK] I need to read the file first to understand its structure.\n\n[READ] main.py",
        "negative": "Let me look at the file main.py. It probably contains something like:\n```python\ndef main():\n    print('hello')\n```\nBased on what I think is in the file...",
    },
    {
        "system": "You have tools: [READ] path, [WRITE] path, [EXEC] command, [THINK] thought.",
        "positive": "[WRITE] utils.py\n```\ndef validate_input(data):\n    if not isinstance(data, dict):\n        raise ValueError('Expected dict')\n    return True\n```",
        "negative": "I would write a utility function that validates input. The function would check if the data is a dictionary and raise an error if not. Here's roughly what it would look like in pseudocode...",
    },
    {
        "system": "You have tools: [READ] path, [WRITE] path, [EXEC] command, [THINK] thought.",
        "positive": "[EXEC] pytest tests/test_api.py -q",
        "negative": "You should now run pytest tests/test_api.py -q in your terminal. I can't run it, but assume it passes.",
    },
    {
        "system": "You have tools: [READ] path, [WRITE] path, [EXEC] command, [THINK] thought.",
        "positive": "[READ] pyproject.toml",
        "negative": "I'll infer the pyproject config from common defaults instead of reading it directly.",
    },
    {
        "system": "You have tools: [READ] path, [WRITE] path, [EXEC] command, [THINK] thought.",
        "positive": "[THINK] The safest next action is to inspect failing test output before editing.",
        "negative": "I'll immediately rewrite three files at once and then maybe check tests later.",
    },
    {
        "system": "You have tools: [READ] path, [WRITE] path, [EXEC] command, [THINK] thought.",
        "positive": "[EXEC] rg \"create_user\" src",
        "negative": "The create_user function is probably in src/user.py. I'll proceed as if that's true.",
    },
    {
        "system": "You have tools: [READ] path, [WRITE] path, [EXEC] command, [THINK] thought.",
        "positive": "[WRITE] tests/test_retry.py\n```\ndef test_retry_on_timeout():\n    assert True\n```",
        "negative": "[THINK] maybe add tests\n[READ] app.py\n[EXEC] pytest\n[WRITE] tests/test_retry.py\n```\n# ...\n```\nAlso here's a long explanation.",
    },
    {
        "system": "You have tools: [READ] path, [WRITE] path, [EXEC] command, [THINK] thought.",
        "positive": "[EXEC] python -m scripts.migrate_users --dry-run",
        "negative": "I'll describe how to run migration in theory, but I won't actually execute the command.",
    },
    {
        "system": "You have tools: [READ] path, [WRITE] path, [EXEC] command, [THINK] thought.",
        "positive": "[READ] src/router.py",
        "negative": "The router likely has a /users endpoint and auth middleware; I'll patch based on that assumption.",
    },
    {
        "system": "You have tools: [READ] path, [WRITE] path, [EXEC] command, [THINK] thought.",
        "positive": "[THINK] Make one atomic file change, then run tests.",
        "negative": "I'll dump all potential plans and avoid choosing a concrete tool action.",
    },
]


# PAIR SET 3: "Generate coherent code" (anti-degeneration)
CODE_COHERENCE_PAIRS: list[ContrastivePair] = [
    {
        "system": "You are a coding assistant working on a Python project.",
        "positive": "def fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b",
        "negative": "def fibonacci(n):\n    if n <= 1:\n        return n\n    result = fibonacci(n-1) + fibonacci(n-2)\n    # TODO: add caching\n    # TODO: add error handling\n    # TODO: optimize\n    # Note: this could be improved\n    # See also: dynamic programming\n    # Reference: https://en.wikipedia.org/wiki/Fibonacci",
    },
    {
        "system": "You are a coding assistant working on a Python project.",
        "positive": "def parse_int(value: str) -> int:\n    value = value.strip()\n    if not value:\n        raise ValueError('empty input')\n    return int(value)",
        "negative": "def parse_int(value):\n    # parse maybe\n    # TODO parse better\n    return value  # might be fine?",
    },
    {
        "system": "You are a coding assistant working on a Python project.",
        "positive": "def is_even(n: int) -> bool:\n    return n % 2 == 0",
        "negative": "def is_even(n):\n    if n % 2 == 1:\n        return True\n    return False",
    },
    {
        "system": "You are a coding assistant working on a Python project.",
        "positive": "def unique_items(items: list[int]) -> list[int]:\n    seen = set()\n    out = []\n    for item in items:\n        if item not in seen:\n            seen.add(item)\n            out.append(item)\n    return out",
        "negative": "def unique_items(items):\n    # TODO uniqueness\n    # TODO edge cases\n    # TODO performance\n    return items",
    },
    {
        "system": "You are a coding assistant working on a Python project.",
        "positive": "def join_path(base: str, name: str) -> str:\n    if base.endswith('/'):\n        return f\"{base}{name}\"\n    return f\"{base}/{name}\"",
        "negative": "def join_path(base, name):\n    return base + name  # probably okay for all cases",
    },
    {
        "system": "You are a coding assistant working on a Python project.",
        "positive": "def clamp(value: float, lo: float, hi: float) -> float:\n    if lo > hi:\n        raise ValueError('lo must be <= hi')\n    return max(lo, min(value, hi))",
        "negative": "def clamp(value, lo, hi):\n    return value  # not implemented yet, but left as placeholder",
    },
    {
        "system": "You are a coding assistant working on a Python project.",
        "positive": "def chunked(seq: list[int], size: int) -> list[list[int]]:\n    if size <= 0:\n        raise ValueError('size must be positive')\n    return [seq[i:i + size] for i in range(0, len(seq), size)]",
        "negative": "def chunked(seq, size):\n    chunks = []\n    for i in range(len(seq)):\n        chunks.append(seq)\n    return chunks",
    },
    {
        "system": "You are a coding assistant working on a Python project.",
        "positive": "def retryable_status(code: int) -> bool:\n    return code in {408, 425, 429, 500, 502, 503, 504}",
        "negative": "def retryable_status(code):\n    # maybe retry all errors?\n    return code >= 400",
    },
    {
        "system": "You are a coding assistant working on a Python project.",
        "positive": "def normalize_whitespace(text: str) -> str:\n    return ' '.join(text.split())",
        "negative": "def normalize_whitespace(text):\n    text = text.replace('  ', ' ')\n    text = text.replace('  ', ' ')\n    text = text.replace('  ', ' ')\n    return text",
    },
    {
        "system": "You are a coding assistant working on a Python project.",
        "positive": "def safe_divide(a: float, b: float) -> float:\n    if b == 0:\n        raise ZeroDivisionError('b must not be zero')\n    return a / b",
        "negative": "def safe_divide(a, b):\n    if b == 0:\n        return 0  # hides bug\n    return a // b  # wrong operator for float division",
    },
]


def get_all_pairs() -> PairSets:
    """Return all contrastive pair sets."""
    return {
        "task_focus": TASK_FOCUS_PAIRS,
        "tool_format": TOOL_FORMAT_PAIRS,
        "code_coherence": CODE_COHERENCE_PAIRS,
    }


def format_pair_for_model(pair: ContrastivePair, tokenizer: Any) -> tuple[str, str]:
    """
    Format a contrastive pair into full prompts ready for tokenization.

    Uses the model's chat template to ensure hidden states are comparable.
    Returns (positive_text, negative_text) as formatted strings.
    """
    positive_messages = [
        {"role": "system", "content": pair["system"]},
        {"role": "assistant", "content": pair["positive"]},
    ]
    negative_messages = [
        {"role": "system", "content": pair["system"]},
        {"role": "assistant", "content": pair["negative"]},
    ]

    pos_text = tokenizer.apply_chat_template(
        positive_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    neg_text = tokenizer.apply_chat_template(
        negative_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return pos_text, neg_text

