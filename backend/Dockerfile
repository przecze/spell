FROM python:3.13-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install "uv==0.9.28" && uv pip install --system --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create checkpoints directory
RUN mkdir -p /checkpoints

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
