FROM --platform=linux/amd64 python:3.13-slim

WORKDIR /app

COPY api/requirements.txt requirements.txt
RUN pip install uv && \
    uv pip install --system --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    grep -v '^torch' requirements.txt > /tmp/requirements-no-torch.txt && \
    uv pip install --system --no-cache-dir -r /tmp/requirements-no-torch.txt

COPY api/main.py /app/main.py

# Bake in model checkpoints
COPY checkpoints/ /checkpoints/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
