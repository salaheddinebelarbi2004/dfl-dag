FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Build context is the repository root (see build.context in docker-compose.yml),
# so every COPY below is relative to the repo root, not this file's directory.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# docker_entrypoint.py imports p2p_node / run_with_shards (deploy/) and
# model_registry / federated_learning_system (core/), so all three are needed.
COPY core/ ./core/
COPY deploy/*.py ./
COPY scripts/ ./scripts/

ENV PYTHONPATH=/app:/app/core:/app/scripts

# Baseline model + pretrained weights
COPY global_model.npy .
COPY model.pt model.safetensors ./

# Per-client shards are bind-mounted individually (see volumes in docker-compose.yml);
# this copy is a fallback for single-container runs.
COPY real_federated_data /data/real_federated_data

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 CMD python -c "import os; exit(0 if os.path.exists('/tmp/client_ready') else 1)"

ENTRYPOINT ["python", "docker_entrypoint.py"]
