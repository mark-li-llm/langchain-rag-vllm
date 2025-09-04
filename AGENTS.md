# Repository Guidelines

## Environment Setup
- Conda environment: always run commands inside your conda `rag` environment — do not use `base`.
- Activate: `conda activate rag` (if needed, first `source ~/miniconda3/etc/profile.d/conda.sh`).
- Install: `make install` installs into the currently active environment; ensure `rag` is active first.

## Project Structure & Module Organization
- `app/`: Core services and pipeline code (e.g., `api.py` FastAPI app, `pipeline.py`, `retrieval.py`, `embeddings.py`, `index_qdrant.py`, `prompting.py`).
- `tools/`: Operational scripts (`build_index.py`, `evaluate.py`, `load_test.py`).
- `ops/`: Docker Compose, images, and gateway (`compose.yaml`, `nginx.conf`).
- `configs/`: Environment and monitoring (`.env.example`, `prometheus.yml`).
- `Makefile`: Primary developer entrypoint; see `make help`.

## Build, Test, and Development Commands
- `make install`: Install Python dependencies (prefers `uv`, falls back to `pip`).
- `make build-index`: Index MS MARCO into Qdrant via `tools/build_index.py`.
- `make serve` | `make stop`: Start/stop full stack with Docker Compose.
- `make serve-api-dev`: Run only the API locally with auto-reload.
- `make health`: Check API, Nginx, and Qdrant health.
- `make eval-quick` | `make loadtest`: Evaluate retrieval or run load tests.
- `make lint` | `make format` | `make typecheck`: Ruff lint/format and mypy type check.

## Coding Style & Naming Conventions
- Python 3.11+, 4-space indentation, prefer type hints everywhere.
- Use Ruff for style/formatting; keep imports ordered; run `make lint` and `make format`.
- Naming: `snake_case` for files/functions, `PascalCase` for classes, `UPPER_CASE` for constants.
- Module layout: add new components under `app/` (e.g., `reranking.py`), keep script-style entrypoints in `tools/`.

## Testing Guidelines
- Frameworks: runtime validation (`make test`) plus `pytest` is available.
- Location: place unit tests in `tests/` with filenames `test_*.py`.
- Run: `pytest -q` for unit tests; `make eval-quick` for retrieval metrics; `make loadtest` for performance.
- Focus tests on `app/pipeline.py`, `app/retrieval.py`, and API routes in `app/api.py`.

## Commit & Pull Request Guidelines
- Commits: follow Conventional Commits (`feat`, `fix`, `docs`, `chore`, `release`, optional scopes). Example: `feat(api): add reranking endpoint`.
- PRs: include a clear description, linked issues, and any evaluation/load-test results. Update `README.md` and `.env` docs if configs change. Keep PRs focused and small.

## Security & Configuration Tips
- Copy `configs/.env.example` to `.env` and set `AUTH_BEARER_TOKEN`, `OPENAI_API_BASE`, and model settings. Do not commit secrets.
- Restrict access to vLLM servers; keep Qdrant auth enabled where applicable. Verify with `make health` after changes.
