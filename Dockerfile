FROM python:3.13-slim

WORKDIR /app

# graph.py must be present for `pip install .` to succeed - pyproject.toml's
# [tool.setuptools] py-modules = ["graph"] is read at build time. Installing from
# pyproject.toml (rather than a hand-copied package list) keeps this Dockerfile from drifting
# out of sync with the project's actual declared dependencies.
COPY pyproject.toml graph.py agent.py api.py ./
RUN pip install --no-cache-dir .

# Tracked in git, unlike .env/ and outputs/ (see .gitignore) - those are supplied at runtime,
# not baked into the image: secrets via `docker run --env-file .env`, generated letters via
# a mounted volume at /app/outputs.
COPY inputs/ ./inputs/

EXPOSE 8000

CMD ["fastapi", "run", "api.py", "--host", "0.0.0.0", "--port", "8000"]
