FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /src

COPY pyproject.toml README.md /src/
COPY src /src
COPY .env /.env
COPY config.py /config.py

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "src/streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]