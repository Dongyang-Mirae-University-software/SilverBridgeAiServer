from __future__ import annotations

import logging
import re

# ?apiKey=... 가 접속 로그(uvicorn)에 그대로 찍히지 않게 가린다(QA AI-2).
_API_KEY_IN_URL = re.compile(r"(?i)(apikey=)[^&\s\"']+")


def redact_api_key(text: str) -> str:
    return _API_KEY_IN_URL.sub(r"\1***", text)


class ApiKeyRedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_api_key(record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(redact_api_key(a) if isinstance(a, str) else a for a in record.args)
        elif isinstance(record.args, dict):
            record.args = {k: redact_api_key(v) if isinstance(v, str) else v for k, v in record.args.items()}
        return True


def setup_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    redacting = ApiKeyRedactingFilter()
    # 로거 필터는 그 로거가 직접 만든 기록에만 적용된다 — uvicorn 은 접속(access)·WS 수락(error) 로그를 각자 남긴다.
    for name in ("uvicorn.access", "uvicorn.error", "uvicorn"):
        logger = logging.getLogger(name)
        if not any(isinstance(f, ApiKeyRedactingFilter) for f in logger.filters):
            logger.addFilter(redacting)
