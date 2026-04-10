# Repository Guidelines

## Project Structure & Module Organization

Core framework code lives in `pidsmaker/`. The pipeline is split by concern: `tasks/` orchestrates stages, `config/` loads YAML runtime settings, `preprocessing/`, `featurization/`, `encoders/`, `decoders/`, `objectives/`, `detection/`, and `triage/` hold stage-specific implementations. System presets and experiment overrides live in `config/` and `config/experiments/`. Functional tests are in `tests/`. Dataset bootstrap scripts are in `dataset_preprocessing/`, helper run scripts are in `scripts/`, and user documentation is under `docs/`.

## Build, Test, and Development Commands

Use Python 3.9+.

- `python pidsmaker/main.py SYSTEM DATASET` runs a local experiment, for example `python pidsmaker/main.py velox CADETS_E3`.
- `./scripts/run.sh SYSTEM DATASET` starts the same run in the background with `--wandb`.
- `pytest -v` runs the functional suite on GPU.
- `pytest -v --device cpu -k "not (test_transformations or test_featurizations)"` runs the CPU-safe subset.
- `pytest --cov=pidsmaker tests/` reports coverage for the package.
- `pre-commit run --all-files` applies `ruff --fix`, `ruff format`, and YAML checks.
- `cd docs && ./build.sh` regenerates argument docs and builds the MkDocs site.

## Coding Style & Naming Conventions

Follow the repository Ruff settings in `pyproject.toml`: 4-space indentation and a 100-character line limit. Keep imports grouped and sorted by Ruff/isort. Use `snake_case` for functions, modules, YAML keys, and most config names; keep class names in `PascalCase`. New system presets should be added as focused YAML files such as `config/my_system.yml`, not by overloading `default.yml`.

## Testing Guidelines

Tests are functional and exercise end-to-end config combinations through `pidsmaker.main`. Add or extend `tests/test_framework.py` when changing pipeline behavior, configuration wiring, batching, featurization, or model composition. Name new tests `test_<behavior>` and keep dataset assumptions explicit. Run the CPU command above when GPU is unavailable; run full `pytest -v` before merging GPU-sensitive changes.

## Commit & Pull Request Guidelines

Recent history favors short, imperative commit subjects such as `update pipeline and utils` or `fix: resolve GMAE masking failure...`. Keep commits focused and descriptive. Before opening a PR, run `pre-commit`, the relevant `pytest` command, and the docs build if you touched `docs/` or CLI arguments. PRs should summarize the behavioral change, note config or dataset impacts, and include screenshots only when documentation or generated figures changed.
