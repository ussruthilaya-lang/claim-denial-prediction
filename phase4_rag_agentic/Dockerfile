# Serves the pretrained Phase 4 retrieval-augmented denial model behind a
# FastAPI endpoint. Build from the REPO ROOT (needs shared/ + the pretrained
# artifacts/ alongside this phase's code):
#
#   docker build -f phase4_rag_agentic/Dockerfile -t phase4-api .
#   docker run -p 8000:8000 phase4-api
#   curl http://localhost:8000/health
#
# The model is NOT trained inside the image — it serves the artifacts already
# produced by `python scripts/run_phase4.py`, same as the Streamlit demo does.
FROM python:3.11-slim

WORKDIR /app

COPY phase4_rag_agentic/requirements.txt phase4_rag_agentic/requirements.txt
COPY phase4_rag_agentic/api/requirements.txt phase4_rag_agentic/api/requirements.txt

# sentence-transformers depends on torch; installing the CPU-only wheel first
# (from PyTorch's own index) avoids pulling multi-hundred-MB CUDA/nvidia
# packages we never use for CPU inference — smaller image, faster build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
    -r phase4_rag_agentic/requirements.txt \
    -r phase4_rag_agentic/api/requirements.txt

COPY shared/ shared/
COPY phase4_rag_agentic/ phase4_rag_agentic/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "phase4_rag_agentic.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
