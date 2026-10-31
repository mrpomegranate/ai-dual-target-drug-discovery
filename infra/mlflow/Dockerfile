# Dedicated tracking-service image. Pin direct version; transitive lock is future hardening.
FROM python:3.12-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /uvx /bin/
RUN uv pip install --system --no-cache "mlflow==3.16.1"
RUN mkdir -p /mlflow/db /mlflow/artifacts
COPY scripts/mlflow_smoke.py /opt/scripts/mlflow_smoke.py
WORKDIR /mlflow
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow/db/mlflow.db", \
     "--artifacts-destination", "/mlflow/artifacts", \
     "--allowed-hosts", "localhost:*,127.0.0.1:*,mlflow:5000"]
