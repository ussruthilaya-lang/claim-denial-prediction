.PHONY: setup up down test lint fmt

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip
	. .venv/bin/activate && pip install -e ./shared -e ./mlops_platform 2>/dev/null || true
	. .venv/bin/activate && pip install pytest ruff

up:
	docker compose up --build

down:
	docker compose down -v

test:
	pytest shared/tests phase1_baseline/tests phase2_gbm_shap/tests phase3_clinicalbert/tests phase4_rag_agentic/tests -v

lint:
	ruff check .

fmt:
	ruff format .
