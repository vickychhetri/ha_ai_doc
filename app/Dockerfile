FROM python:3.11-slim

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

# ----------------------------
# Install system dependencies
# ----------------------------
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        poppler-utils \
        tesseract-ocr \
        libtesseract-dev \
        libjpeg-dev \
        libpng-dev \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------
# Install Python dependencies
# ----------------------------
COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

# ----------------------------
# Copy only app code last
# This allows faster rebuilds when only code changes
# ----------------------------
COPY app/ /app/

# ----------------------------
# Create folders for storage
# ----------------------------
RUN mkdir -p /app/file_storage /app/chroma_db

# ----------------------------
# Expose port & environment
# ----------------------------
EXPOSE 8080
ENV CEREBRAS_API_KEY=${CEREBRAS_API_KEY}

# ----------------------------
# Start FastAPI
# ----------------------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
