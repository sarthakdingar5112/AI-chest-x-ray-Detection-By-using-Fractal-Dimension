# FractalScan backend — CPU inference image.
# Use a Python version with torch wheels available (3.11 is a safe choice as of 2026).
FROM python:3.11-slim

WORKDIR /app

# System libs for opencv-headless / pillow / pydicom.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY weights ./weights

EXPOSE 8000
# TorchXRayVision downloads weights on first use; consider baking them into the image.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
