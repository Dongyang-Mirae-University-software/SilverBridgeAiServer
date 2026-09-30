FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# torch는 버전 고정 필수 — transformers(gemma3/medgemma)가 torch>=2.6 을 요구한다.
# cu121 인덱스는 2.5.1 이 상한이라, 무핀으로 두면 최신 transformers 와 충돌해 기동이 실패한다.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app
COPY .env.example /app/.env.example

RUN mkdir -p /app/models /app/uploads/snapshots /app/logs /app/data

EXPOSE 6017

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${APP_PORT:-6017}"]
