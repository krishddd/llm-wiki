# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-08-08

### Added
- `py.typed` marker so downstream type checkers pick up the package's type hints
  (`Typing :: Typed`).
- `CHANGELOG.md` and `CONTRIBUTING.md`; `Changelog` project URL in the metadata.

### Changed
- Documentation: corrected stale `src/…` module paths to `llm_wiki/…` in
  `CLAUDE.md` and `AGENTS.md` (both are embedded into the docs site).
- CI: bumped GitHub Actions off the deprecated Node 20 runtime — `checkout@v7`,
  `setup-python@v7`, `upload-artifact@v7`, `download-artifact@v8`,
  `upload-pages-artifact@v5`, `deploy-pages@v5`.
- Packaging: expanded trove classifiers (Python 3.13, Science/Research audience,
  Web Environment, Libraries topic).

### Removed
- Default `jekyll-gh-pages.yml` workflow that conflicted with the MkDocs Pages
  deploy.

## [0.1.0] - 2026-08-08

### Added
- First public release of **llm-compounding-wiki** — a self-healing local-LLM
  wiki that compounds documents across working, episodic, semantic, and
  procedural memory tiers.
- FastAPI service (`llm_wiki.api:app`) with the full ingest → query → lint
  pipeline; multi-format loaders (PDF/DOCX/PPTX/XLSX/HTML/CSV/MD).
- `llm-wiki` console entry point (`serve` / `mcp`).
- Optional extras: `mcp`, `ocr`, `docs`, `dev`.
- PyPI publishing via GitHub Actions Trusted Publishing (OIDC).
- MkDocs Material documentation site deployed to GitHub Pages.

[Unreleased]: https://github.com/krishddd/llm-wiki/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/krishddd/llm-wiki/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/krishddd/llm-wiki/releases/tag/v0.1.0
