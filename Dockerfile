FROM ghcr.io/astral-sh/uv:0.7.8-python3.12-bookworm-slim@sha256:8973f2cef68d5d69799e37ad418c89b79c91511b7e4259697b98eba9cf714cbf

ENV PATH="/root/.local/bin:$PATH"
ENV ENVIRONMENT="production"

WORKDIR /app

COPY pyproject.toml uv.lock /app/
RUN ["uv", "sync"]

COPY src /app/src
COPY .streamlit /app/.streamlit
COPY config.py README.md /app/

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "src/isschat_webapp.py", "--server.port=8501", "--server.address=0.0.0.0"]