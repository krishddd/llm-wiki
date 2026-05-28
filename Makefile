.PHONY: install dev test lint fmt up down ingest query lint-wiki health

install:
	pip install -r requirements-dev.txt

dev:
	uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest -q --cov=src --cov-report=term-missing -m "not integration"

test-integration:
	pytest -q -m integration

lint:
	ruff check src tests
	mypy src

fmt:
	ruff check --fix src tests
	ruff format src tests

up:
	docker compose up -d --build

down:
	docker compose down

health:
	curl -sS http://localhost:8000/health | python -m json.tool

ingest:
	@test -n "$(SOURCE)" || (echo "usage: make ingest SOURCE=wiki/raw/file.pdf" && exit 1)
	curl -sS -F "files=@$(SOURCE)" http://localhost:8000/ingest | python -m json.tool

query:
	@test -n "$(Q)" || (echo "usage: make query Q='your question'" && exit 1)
	curl -sS -X POST http://localhost:8000/query -H 'content-type: application/json' -d '{"question":"$(Q)"}' | python -m json.tool

lint-wiki:
	curl -sS http://localhost:8000/lint | python -m json.tool
