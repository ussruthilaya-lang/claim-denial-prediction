.PHONY: setup test lint fmt demo

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip
	. .venv/bin/activate && pip install -e .
	. .venv/bin/activate && pip install pytest ruff

test:
	pytest shared/tests phase1_baseline/tests phase2_gbm_shap/tests phase3_clinicalbert/tests phase4_rag_agentic/tests -v

lint:
	ruff check .

fmt:
	ruff format .

demo:
	streamlit run mlops_platform/demo/app.py
