FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y build-essential libglib2.0-0 libsm6 libxrender1 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY app/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app/ .

# Create folders for files & Chroma DB
RUN mkdir -p /app/file_storage /app/chroma_db

EXPOSE 8080
ENV CEREBRAS_API_KEY=${CEREBRAS_API_KEY}

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]