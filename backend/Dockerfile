# 1. Use slim Python image (small but stable)
FROM python:3.11-slim

# 2. Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# 3. Ensure logs show immediately
ENV PYTHONUNBUFFERED=1

# 4. Set working directory
WORKDIR /app

# 5. Install system dependencies (important for numpy, tensorflow, pillow)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 6. Copy requirements first (Docker caching trick)
COPY requirements.txt .

# 7. Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 8. Copy your app code
COPY . .

# 9. Expose FastAPI port
EXPOSE 8000

# 10. Start server
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]