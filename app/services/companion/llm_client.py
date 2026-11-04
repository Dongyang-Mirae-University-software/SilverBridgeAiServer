from __future__ import annotations

import concurrent.futures
import threading
from pathlib import Path
from typing import Protocol

from app.core.config import get_settings


class LlmUnavailable(Exception):
    """LLM 을 쓸 수 없음 (모델 파일 없음, 불러오기 실패, 시간 초과). 응답 쪽은 재시도 뒤 대체 문장으로 간다 (D20)."""


class LlmClient(Protocol):
    """신호 추출과 응답 생성이 같이 쓰는 LLM 인터페이스 (S5-4)."""

    def generate(self, prompt: str, *, temperature: float, num_predict: int, timeout: float) -> str: ...


class ExaoneLlm:
    """EXAONE 3.5 2.4B 를 transformers 로 직접 올리는 클라이언트 (D7 결정, CH-052).

    모델은 처음 쓸 때 한 번만 올린다. 실제 가중치가 없으면 LlmUnavailable 을 낸다.
    GPU 메모리는 MedGemma 와 나눠 쓰므로 S12-3에서 측정 전까지는 실제로 켜지 않는다.
    """

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self._load_error: str | None = None

    def model_dir(self) -> Path:
        s = get_settings()
        return Path(s.model_base_path) / s.companion_llm_subdir

    def _load(self):
        if self._model is not None:
            return self._tokenizer, self._model
        if self._load_error is not None:
            raise LlmUnavailable(self._load_error)
        with self._lock:
            if self._model is not None:
                return self._tokenizer, self._model
            path = self.model_dir()
            if not (path / "config.json").is_file():
                self._load_error = f"LLM 모델 폴더가 없습니다: {path}"
                raise LlmUnavailable(self._load_error)
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                dtype = torch.bfloat16 if get_settings().companion_llm_dtype == "bfloat16" else torch.float32

                tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=True)
                # 가중치는 float32(9.6GB)로 저장돼 있다. bfloat16 으로 올려 메모리를 절반으로 줄인다.
                model = AutoModelForCausalLM.from_pretrained(
                    str(path), local_files_only=True, trust_remote_code=True, dtype=dtype
                )
                model.eval()
                self._tokenizer, self._model = tokenizer, model
            except Exception as exc:
                self._load_error = f"LLM 을 불러오지 못했습니다 ({type(exc).__name__})"
                raise LlmUnavailable(self._load_error) from exc
            return self._tokenizer, self._model

    def _generate_blocking(self, prompt: str, temperature: float, num_predict: int) -> str:
        import torch

        tokenizer, model = self._load()
        messages = [{"role": "user", "content": prompt}]
        # transformers 5.x 는 return_dict 기본값이 달라서 명시한다 (입력 ID 만 꺼낸다)
        encoded = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        ids = encoded["input_ids"]
        sample = temperature > 0
        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=num_predict,
                do_sample=sample,
                temperature=temperature if sample else None,
            )
        return tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)

    def generate(self, prompt: str, *, temperature: float, num_predict: int, timeout: float) -> str:
        """시간 제한을 넘으면 LlmUnavailable. 넘긴 생성 작업은 백그라운드에서 끝날 때까지 돌아간다."""
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(self._generate_blocking, prompt, temperature, num_predict)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as exc:
            raise LlmUnavailable("LLM 응답 시간 초과") from exc
        except LlmUnavailable:
            raise
        except Exception as exc:
            raise LlmUnavailable(f"LLM 생성 실패 ({type(exc).__name__})") from exc
        finally:
            pool.shutdown(wait=False)


llm_client = ExaoneLlm()
