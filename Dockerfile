FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p model && \
    curl -L --fail --retry 3 \
    "https://github.com/isuesss/wildguard-ai/releases/download/v1.0.0/wildlife_epoch_3.pth" \
    -o model/wildlife_epoch_3.pth

ENV MODEL_PATH=/app/model/wildlife_epoch_3.pth

EXPOSE 7860

CMD ["python", "app.py"]
