# ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest

# FROM ${BASE_IMAGE} AS builder

# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1

# WORKDIR /app

# RUN apt-get update \
# 	&& apt-get upgrade -y --no-install-recommends \
# 	&& rm -rf /var/lib/apt/lists/*

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# COPY server/           ./server/
# COPY inference.py      .
# COPY openenv.yaml      .

# EXPOSE 8000

# HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=5 \
# 	CMD curl -f http://localhost:8000/health || exit 1


# CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH="/app:$PYTHONPATH"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
	CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]