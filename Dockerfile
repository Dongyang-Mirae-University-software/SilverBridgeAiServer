FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# torch는 버전 고정 필수 — transformers(gemma3/medgemma)가 torch>=2.6 을 요구한다.
# cu121 인덱스는 2.5.1 이 상한이라, 무핀으로 두면 최신 transformers 와 충돌해 기동이 실패한다.
# 2026-11-10 전체 점검 L-4: 2.6.0 의 알려진 취약점(체크포인트 로드 RCE 등, 2.10.0 에서 고침) 때문에 2.11.0 으로 올렸다.
# cu124 인덱스는 2.6.0 이 상한이라 cu126 으로 옮겼다. 셋은 같은 판이어야 하고 torchaudio 는 cu126 에서 2.11.0 이 마지막이다.
# 2026-11-11: 2.11.0 에 남은 torch.jit.script 메모리 손상 건(2.13.0 에서 고침) 때문에 2.13.0 으로 올렸다.
# torchaudio 는 2.11.x 가 마지막 판이고 cu126 인덱스에는 2.11.0 까지만 있다. 2.11.1(PyPI, torch>=2.11 허용)을 따로 깐다.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.13.0 torchvision==0.28.0 \
    && pip install --no-cache-dir --no-deps torchaudio==2.11.1

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app
COPY .env.example /app/.env.example

RUN mkdir -p /app/models /app/uploads/snapshots /app/logs /app/data

EXPOSE 6017

CMD ["sh", "-c", "uvicorn app.main:app --host ${APP_HOST:-0.0.0.0} --port ${APP_PORT:-6017} --timeout-graceful-shutdown 20"]
