FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

COPY . .

ENV PYTHONUNBUFFERED=1
ENV TORCH_DEVICE=cpu
ENV RECOGNITION_BATCH_SIZE=32
ENV DETECTOR_BATCH_SIZE=6
ENV LAYOUT_BATCH_SIZE=4

EXPOSE 50051

CMD ["python", "main.py"]