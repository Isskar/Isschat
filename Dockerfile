FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY .env /app/.env
COPY config.py /app/config.py

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "src/isschat_webapp.py", "--server.port=8501", "--server.address=0.0.0.0"]