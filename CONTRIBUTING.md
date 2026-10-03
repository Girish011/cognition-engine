# Contributing

Thanks for helping. This is a research-flavoured repo, so replications and negative results are as
welcome as code.

## Setup

```bash
pip install -e ".[dev]"
pytest            # MLX-dependent tests skip automatically off Apple Silicon
```

## Good first contributions

- Run an experiment from `scripts/` on your hardware or model and open an issue with the numbers.
- Take on an open experiment from [docs/FINDINGS.md](docs/FINDINGS.md#limitations-and-next-experiments).
- Add tests for modules that only have manual checks (`compaction.py`, `agent.py` helpers).
- Port the steering wrapper to another MLX model family.

## Pull requests

- Keep changes focused; add or update tests when behaviour changes.
- Experiments: commit the raw JSON under `data/`, and state model, quantization, hardware, and
  decoding settings. Say plainly when a result is n = 1.
- Be careful with claims: this repo's value is that it separates what was shown from what was hoped.

By contributing you agree your work is released under the MIT license.
