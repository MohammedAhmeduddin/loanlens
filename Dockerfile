FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ src/

# Install dependencies
RUN pip install -e ".[dev]" && \
    pip install langchain-text-splitters pypdf

# Copy remaining files
COPY . .

# Set environment variables
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=1
ENV PYTHONPATH=/app/src

# Start FastAPI
CMD ["uvicorn", "loanlens.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
