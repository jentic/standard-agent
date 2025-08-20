install:          ## create venv + install deps (editable dev mode)
	python3 -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"
test:             ## run unit tests
	. .venv/bin/activate && pytest -q
lint:             ## static checks
	. .venv/bin/activate && ruff check .
lint-strict:      ## static checks with mypy
	. .venv/bin/activate && ruff check . && mypy .
