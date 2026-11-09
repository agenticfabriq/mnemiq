# The mnemiq image: the engine and its HTTP API, with the workbench built in. examples/open-webui runs it.
# Build from the repository root: docker build -t mnemiq:local .   (.dockerignore admits only the build's inputs)

FROM node:24.15.0-bookworm-slim AS workbench
WORKDIR /build/workbench
RUN corepack enable && corepack prepare pnpm@11.15.1 --activate
# pnpm-workspace.yaml allows esbuild's install script, which vite needs; pnpm 11 refuses it otherwise.
COPY workbench/package.json workbench/pnpm-lock.yaml workbench/pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile
COPY workbench/ ./
# vite writes the build to ../src/mnemiq/server/static, the folder the engine serves.
RUN pnpm build

FROM python:3.13.9-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock LICENSE ./
RUN uv sync --frozen --no-dev --extra server --no-install-project
COPY src/ src/
COPY scripts/seed_demo.py scripts/seed_demo.py
COPY --from=workbench /build/src/mnemiq/server/static src/mnemiq/server/static
# The user needs a home: DuckDB installs its extensions under ~/.duckdb.
RUN uv sync --frozen --no-dev --extra server \
 && useradd --uid 10001 --create-home --shell /usr/sbin/nologin mnemiq \
 && mkdir -p /app/demo && chown mnemiq /app/demo
USER mnemiq
ENV PATH=/app/.venv/bin:$PATH
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --retries=6 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2)"]
CMD ["mnemiq", "serve", "--http", "--host", "0.0.0.0", "--port", "8080"]
