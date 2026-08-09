# Contributing

Thanks for your interest in improving **llm-wiki** (`llm-compounding-wiki` on PyPI).

## Development setup

```bash
git clone https://github.com/krishddd/llm-wiki.git
cd llm-wiki
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env
```

The import package is `llm_wiki`; the PyPI distribution is `llm-compounding-wiki`.

## Running checks

```bash
make test          # pytest, Ollama mocked, excludes integration
make lint          # ruff check + mypy
make fmt           # ruff --fix + ruff format
```

Or directly:

```bash
pytest -q -m "not integration"
ruff check llm_wiki tests
mypy llm_wiki
```

Tests marked `integration` need a live Ollama and are skipped by default; run them
with `pytest -m integration` (or `make test-integration`) against a real endpoint.

## Docs

The site is built with MkDocs Material. Preview locally:

```bash
pip install -e ".[docs]"
mkdocs serve            # http://127.0.0.1:8000
```

`README.md`, `CLAUDE.md`, and `AGENTS.md` are the single source of truth — the
docs pages embed them via snippet includes, so edit the root files, not copies.

## Pull requests

1. Branch off `main` (it is protected; the `lint-test` check must pass to merge).
2. Keep changes focused; add or update tests for behavior changes.
3. Run `make lint && make test` before pushing.
4. Note user-facing changes under `## [Unreleased]` in [CHANGELOG.md](CHANGELOG.md).

## Releasing (maintainers)

1. Move the `## [Unreleased]` notes into a new version section in `CHANGELOG.md`.
2. Bump `version` in `pyproject.toml`.
3. Merge to `main`, then cut a GitHub Release:
   ```bash
   gh release create vX.Y.Z --generate-notes
   ```
   The `publish.yml` workflow builds and publishes to PyPI via Trusted Publishing
   (OIDC — no token needed).

By contributing you agree that your contributions are licensed under the
project's [MIT License](LICENSE).
