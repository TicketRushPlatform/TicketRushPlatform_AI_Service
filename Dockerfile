FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    EVENT_SERVICE_BASE_URL=http://event-service:8080/api/v1 \
    BOOKING_SERVICE_BASE_URL=http://booking-service:8080/api/v1 \
    JWT_ALGORITHM=HS256 \
    CORS_ALLOW_ORIGINS=http://localhost:3000,http://localhost:5173 \
    OPENAI_MODEL=gpt-4o-mini \
    LOG_LEVEL=INFO

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src

EXPOSE 8000

CMD ["uv", "run", "--frozen", "--no-sync", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
