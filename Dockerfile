# Stage 1: Build Backend
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS backend-builder
ADD . /flare-ai-social
WORKDIR /flare-ai-social
RUN uv sync --frozen

# Stage 2: Final Image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
COPY --from=backend-builder /flare-ai-social/.venv ./.venv
COPY --from=backend-builder /flare-ai-social/src ./src
COPY --from=backend-builder /flare-ai-social/pyproject.toml .
COPY --from=backend-builder /flare-ai-social/README.md .

# Allow workload operator to override environment variables
LABEL "tee.launch_policy.allow_env_override"="GEMINI_API_KEY,GEMINI_MODEL"
LABEL "tee.launch_policy.log_redirect"="always"

CMD ["uv", "run", "start-social"]