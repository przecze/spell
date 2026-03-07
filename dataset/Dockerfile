FROM --platform=linux/amd64 python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install uv && uv pip install --system -r requirements.txt

COPY precompute.py .
RUN python precompute.py

# ---

FROM --platform=linux/amd64 python:3.12-slim

WORKDIR /app

COPY requirements-runtime.txt .
RUN pip install uv && uv pip install --system -r requirements-runtime.txt

COPY dataset.py .
COPY .streamlit .streamlit/
COPY --from=builder /app/data /app/data

EXPOSE 8080
CMD ["streamlit", "run", "dataset.py", "--server.port=8080", "--server.address=0.0.0.0"]
