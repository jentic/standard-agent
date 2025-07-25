install:          ## make sure pdm is installed and install dependencies
	@command -v pdm >/dev/null 2>&1 || { echo "PDM not found. Install it with: pip install pdm"; exit 1; }
	pdm install
test:             ## run unit tests
	. .venv/bin/activate && pytest -q
lint:             ## static checks
	. .venv/bin/activate && ruff check .
lint-strict:      ## static checks with mypy
	. .venv/bin/activate && ruff check . && mypy .
