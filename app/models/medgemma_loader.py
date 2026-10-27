from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from app.core.config import Settings


_LOG = logging.getLogger(__name__)

QUANT_NONE = "none"
QUANT_NF4 = "nf4"
QUANT_INT8 = "int8"


def resolve_quant(value: str | None) -> str:
    """MEDGEMMA_QUANT 값. 모르는 값은 none(지금과 같은 bf16)으로 본다."""
    quant = (value or "").strip().lower()
    return quant if quant in (QUANT_NF4, QUANT_INT8) else QUANT_NONE


def quantization_kwargs(quant: str, torch: Any) -> dict[str, Any]:
    """from_pretrained 에 더할 양자화 인자. none이면 bitsandbytes 를 import 하지 않는다."""
    if quant not in (QUANT_NF4, QUANT_INT8):
        return {}
    import bitsandbytes  # noqa: F401  # type: ignore[reportMissingImports]  # 미설치면 여기서 실패 -> none으로 내려간다
    from transformers import BitsAndBytesConfig  # type: ignore[reportMissingImports]

    if quant == QUANT_INT8:
        # transformers 5.x 는 from_pretrained(load_in_8bit=True) 를 받지 않는다 - 설정 객체로만 켠다.
        return {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
    return {
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    }


def json_object_complete(text: str) -> bool:
    """prefill + 생성 문장이 완성된 JSON 객체 하나로 읽히면 True. 닫는 괄호가 아니면 파싱하지 않는다."""
    stripped = (text or "").strip()
    if not stripped.endswith("}"):
        return False
    try:
        value, _ = json.JSONDecoder().raw_decode(stripped)
    except ValueError:
        return False
    return isinstance(value, dict)


def json_stop_criteria(tokenizer: Any, input_len: int, prefill: str) -> Any:
    """JSON 객체가 닫히면 생성을 멈춘다. 닫힌 뒤 덧붙이는 설명·두 번째 객체에 토큰을 쓰지 않게 한다."""
    import torch  # type: ignore[reportMissingImports]
    from transformers import StoppingCriteria, StoppingCriteriaList  # type: ignore[reportMissingImports]

    class _JsonObjectDone(StoppingCriteria):
        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            text = prefill + tokenizer.decode(input_ids[0][input_len:], skip_special_tokens=True)
            return torch.full((input_ids.shape[0],), json_object_complete(text), dtype=torch.bool, device=input_ids.device)

    return StoppingCriteriaList([_JsonObjectDone()])


REPETITION_OFF = "off"
# generate 인자가 아니라 로더가 생성 구간 반복 금지 프로세서로 바꿔 끼우는 키.
GEN_NGRAM_KEY = "gen_ngram_size"


def resolve_json_repetition(value: str | None) -> tuple[dict[str, Any], str]:
    """MEDGEMMA_JSON_REPETITION -> (generate 인자, 로그 표기). off·빈 값·모르는 값·범위 밖은 인자 없음(지금과 같음)."""
    raw = (value or "").strip().lower()
    kind, _, number = raw.partition(":")
    try:
        if kind == "gen_ngram":
            size = int(number)
            if 2 <= size <= 20:
                return {GEN_NGRAM_KEY: size}, f"gen_ngram:{size}"
        elif kind == "penalty":
            penalty = float(number)
            if 1.0 <= penalty <= 1.5:
                return {"repetition_penalty": penalty}, f"penalty:{penalty:g}"
        elif kind == "ngram":
            size = int(number)
            if 2 <= size <= 20:
                return {"no_repeat_ngram_size": size}, f"ngram:{size}"
    except ValueError:
        pass
    return {}, REPETITION_OFF


def generated_ngram_bans(generated: list[int], size: int) -> set[int]:
    """생성된 토큰 안에서만 본 n-gram 반복 금지. 마지막 size-1개 토큰으로 시작하는 이전 n-gram의 다음 토큰을 막는다.

    HF no_repeat_ngram_size 는 프롬프트까지 보아 규칙에 적힌 고정 문장을 쓰지 못하게 한다(DGX 2026-10-26 e2) -
    그래서 프롬프트·prefill 은 보지 않는다.
    """
    if size < 2 or len(generated) < size:
        return set()
    prefix = tuple(generated[-(size - 1):])
    banned: set[int] = set()
    for start in range(len(generated) - size + 1):
        if tuple(generated[start:start + size - 1]) == prefix:
            banned.add(generated[start + size - 1])
    return banned


def generated_ngram_processor(input_len: int, size: int) -> Any:
    """generate 의 logits_processor. input_len 이후(생성 구간)만 반복 금지를 적용한다."""
    import torch  # type: ignore[reportMissingImports]
    from transformers import LogitsProcessor, LogitsProcessorList  # type: ignore[reportMissingImports]

    class _GeneratedNoRepeatNgram(LogitsProcessor):
        def __call__(self, input_ids: Any, scores: Any) -> Any:
            for row in range(input_ids.shape[0]):
                banned = generated_ngram_bans(input_ids[row][input_len:].tolist(), size)
                if banned:
                    scores[row, list(banned)] = -float("inf")
            return scores

    return LogitsProcessorList([_GeneratedNoRepeatNgram()])


class MedGemmaBusyError(RuntimeError):
    """생성 슬롯을 대기 상한 안에 얻지 못했다. 생성하지 않았으며 호출자는 고정 문구로 답한다."""


class GenerationSlot:
    """MedGemma 생성 동시 실행 수 제한(2026-10-27).

    생성이 겹치면 같은 GPU의 감지 모델(화재·흉기·낙상) 처리 시간이 늘어 프레임을 건너뛴다(10/27 측정 ×2.03).
    상담 생성 경로만 잡는다 - 감지 스레드(stream-analyze)는 이 슬롯을 모른다. limit 0 이하면 제한하지 않는다.
    """

    def __init__(self, limit: int) -> None:
        self._semaphore = threading.BoundedSemaphore(limit) if limit > 0 else None
        self._count_lock = threading.Lock()
        self._waiting = 0

    @contextmanager
    def acquire(self, timeout_sec: float) -> Iterator[None]:
        semaphore = self._semaphore
        if semaphore is None:
            yield
            return
        if not semaphore.acquire(blocking=False):
            with self._count_lock:
                self._waiting += 1
                waiting = self._waiting
            started = time.perf_counter()
            try:
                acquired = semaphore.acquire(timeout=timeout_sec)
            finally:
                with self._count_lock:
                    self._waiting -= 1
            wait_ms = int((time.perf_counter() - started) * 1000)
            # 대기 시간·대기 수만 남긴다(상담 내용·사용자 ID 금지).
            if not acquired:
                _LOG.warning("[MEDGEMMA-GEN-BUSY] waitMs=%d waiting=%d limitSec=%g", wait_ms, waiting, timeout_sec)
                raise MedGemmaBusyError("generation slot busy")
            _LOG.info("[MEDGEMMA-GEN-WAIT] waitMs=%d waiting=%d", wait_ms, waiting)
        try:
            yield
        finally:
            semaphore.release()


@dataclass
class ModelState:
    loaded: bool
    model_name: str
    gpu: bool
    load_error: str | None = None


class MedGemmaLoader:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._model: Any = None
        self._tokenizer: Any = None
        self._processor: Any = None
        self._torch = None
        self._uses_processor = False
        self._quant = QUANT_NONE  # 실제로 적재된 양자화(nf4·int8 실패 시 none)
        self._gen_slot = GenerationSlot(settings.medgemma_max_concurrent_generate)
        self._json_repetition_kwargs, self._json_repetition_label = resolve_json_repetition(
            settings.medgemma_json_repetition
        )
        if self._json_repetition_label == REPETITION_OFF and settings.medgemma_json_repetition not in ("", REPETITION_OFF):
            _LOG.warning("[MEDGEMMA-JSON-REPETITION-INVALID] -> off")
        self._state = ModelState(
            loaded=False,
            model_name=settings.resolved_chat_model_id(),
            gpu=False,
            load_error=None,
        )

    def _candidate_model_ids(self, gpu_ok: bool) -> list[str]:
        configured = self._settings.resolved_chat_model_id().strip()
        if "medgemma" in configured.lower():
            return [configured]
        cpu_fallbacks = [
            "Qwen/Qwen2.5-1.5B-Instruct",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "Qwen/Qwen2.5-0.5B-Instruct",
        ]
        gpu_fallbacks = [
            "google/gemma-2-2b-it",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "Qwen/Qwen2.5-0.5B-Instruct",
        ]
        if self._settings.chat_model_path.strip():
            return [configured]
        if gpu_ok:
            return [configured, *[item for item in gpu_fallbacks if item != configured]]
        return [item for item in cpu_fallbacks if item != configured]

    @property
    def state(self) -> ModelState:
        return self._state

    @property
    def quant(self) -> str:
        return self._quant

    def _load_processor_model(
        self, auto_model: Any, model_id: str, token: str | None, kwargs: dict[str, Any], gpu_ok: bool
    ) -> Any:
        """MedGemma 적재. MEDGEMMA_QUANT=nf4·int8 이면 그 양자화로 먼저 시도하고, 실패하면 WARN 한 줄 후 none으로 다시 올린다."""
        torch = self._torch
        quant = resolve_quant(self._settings.medgemma_quant)
        if quant != QUANT_NONE and gpu_ok:
            try:
                quant_kwargs = {k: v for k, v in kwargs.items() if k != "load_in_8bit"}
                model = auto_model.from_pretrained(model_id, token=token, **quant_kwargs, **quantization_kwargs(quant, torch))
                self._quant = quant
                return model
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("[MEDGEMMA-QUANT-FALLBACK] MEDGEMMA_QUANT=%s -> none (%s)", quant, type(exc).__name__)
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        self._quant = QUANT_NONE
        return auto_model.from_pretrained(model_id, token=token, **kwargs)

    def load(self) -> None:
        with self._lock:
            if self._state.loaded:
                return
            try:
                import torch  # type: ignore[reportMissingImports]
                from transformers import (  # type: ignore[reportMissingImports]
                    AutoModelForCausalLM,
                    AutoModelForImageTextToText,
                    AutoProcessor,
                    AutoTokenizer,
                )
            except Exception as exc:  # noqa: BLE001
                self._state.load_error = f"import 실패: {exc}"
                return

            self._torch = torch
            gpu_ok = bool(torch.cuda.is_available())
            self._state.gpu = gpu_ok

            if not gpu_ok:
                self._state.load_error = "CUDA를 사용할 수 없습니다."
                if self._settings.require_gpu:
                    raise RuntimeError(self._state.load_error)

            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            configured_dtype = dtype_map.get(self._settings.torch_dtype.lower(), torch.float16)
            dtype = torch.float16 if gpu_ok else configured_dtype

            # GPU 사용이 가능하면 전체 모델을 CUDA 0에 강제 적재한다.
            device_map: dict[str, int] | None = {"": 0} if gpu_ok else None
            low_cpu_mem_usage = True

            token = self._settings.hf_token.strip() or None
            errors: list[str] = []
            for model_id in self._candidate_model_ids(gpu_ok):
                try:
                    self._processor = None
                    self._uses_processor = "medgemma" in model_id.lower()
                    if self._uses_processor:
                        dtype = torch.bfloat16 if gpu_ok else torch.float32
                    kwargs: dict[str, Any] = {
                        "dtype": dtype,
                        "low_cpu_mem_usage": low_cpu_mem_usage,
                    }
                    if device_map is not None:
                        kwargs["device_map"] = device_map
                    if self._settings.load_in_8bit:
                        kwargs["load_in_8bit"] = True
                    if self._uses_processor:
                        self._processor = AutoProcessor.from_pretrained(model_id, token=token)
                        self._model = self._load_processor_model(AutoModelForImageTextToText, model_id, token, kwargs, gpu_ok)
                        self._tokenizer = getattr(self._processor, "tokenizer", None)
                    else:
                        self._tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
                        self._model = AutoModelForCausalLM.from_pretrained(model_id, token=token, **kwargs)
                    # bitsandbytes 양자화(4·8비트) 모델은 .to() 를 거부한다 - device_map 으로 이미 GPU 0에 있다.
                    if gpu_ok and hasattr(self._model, "to") and self._quant == QUANT_NONE:
                        self._model = self._model.to("cuda:0")
                    self._state.loaded = True
                    self._state.load_error = None
                    self._state.model_name = model_id
                    _LOG.info("[MEDGEMMA-LOAD] model=%s quant=%s", model_id, self._quant if self._uses_processor else QUANT_NONE)
                    return
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{model_id}: {exc}")
                    self._model = None
                    self._tokenizer = None
                    self._processor = None
                    self._uses_processor = False

            self._state.load_error = " | ".join(errors) if errors else "모델 로드 실패"
            self._state.loaded = False

    def ensure_loaded(self) -> None:
        if not self._state.loaded:
            self.load()

    # MedGemma 1.5는 답 앞에 "<unused94>thought ..." 내부 추론을 길게 쓰다 토큰 예산을 다 써버린다.
    # JSON 응답이 필요할 때는 어시스턴트 턴을 prefill("{")로 시작시키고 추론 시작 토큰을 금지한다.
    THOUGHT_TOKENS = ("<unused94>", "<unused95>")

    def _banned_token_ids(self) -> list[list[int]]:
        tokenizer = self._tokenizer
        if tokenizer is None:
            return []
        banned: list[list[int]] = []
        unk = getattr(tokenizer, "unk_token_id", None)
        for token in self.THOUGHT_TOKENS:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, int) and token_id >= 0 and token_id != unk:
                banned.append([token_id])
        return banned

    def generate_structured_text(
        self, system_prompt: str, user_prompt: str, prefill: str = "", *, wait_timeout_sec: float | None = None
    ) -> str:
        """생성 슬롯을 잡고 생성한다. wait_timeout_sec(없으면 MEDGEMMA_GEN_WAIT_TIMEOUT_SEC) 안에 못 잡으면 MedGemmaBusyError."""
        if not self._state.loaded or self._model is None or self._torch is None:
            raise RuntimeError(self._state.load_error or "model not loaded")
        timeout = self._settings.medgemma_gen_wait_timeout_sec if wait_timeout_sec is None else wait_timeout_sec
        with self._gen_slot.acquire(timeout):
            return self._generate_structured_text(system_prompt, user_prompt, prefill)

    def _generate_structured_text(self, system_prompt: str, user_prompt: str, prefill: str) -> str:
        model = self._model
        torch = self._torch

        if self._uses_processor:
            if self._processor is None:
                raise RuntimeError(self._state.load_error or "processor not loaded")
            generation_dtype = torch.float16 if self._state.gpu else torch.float32
            messages = [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ]
            inputs = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device, dtype=generation_dtype)
            if prefill and self._tokenizer is not None:
                prefill_ids = self._tokenizer(prefill, add_special_tokens=False, return_tensors="pt")["input_ids"].to(
                    model.device
                )
                extra = prefill_ids.shape[-1]
                inputs["input_ids"] = torch.cat([inputs["input_ids"], prefill_ids], dim=-1)
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = torch.cat(
                        [inputs["attention_mask"], torch.ones((1, extra), dtype=inputs["attention_mask"].dtype, device=model.device)],
                        dim=-1,
                    )
                if "token_type_ids" in inputs:
                    inputs["token_type_ids"] = torch.cat(
                        [inputs["token_type_ids"], torch.zeros((1, extra), dtype=inputs["token_type_ids"].dtype, device=model.device)],
                        dim=-1,
                    )
            input_len = inputs["input_ids"].shape[-1]
            limit = self._settings.medgemma_json_max_tokens
            generate_kwargs: dict[str, Any] = {
                "max_new_tokens": limit,
                "do_sample": False,
                **self._json_repetition_kwargs,
            }
            gen_ngram = generate_kwargs.pop(GEN_NGRAM_KEY, None)
            if gen_ngram:
                generate_kwargs["logits_processor"] = generated_ngram_processor(input_len, gen_ngram)
            banned = self._banned_token_ids()
            if banned:
                generate_kwargs["bad_words_ids"] = banned
            if prefill == "{" and self._tokenizer is not None:
                # max_new_tokens 는 상한으로 남기고, 완성된 JSON 객체가 나오면 바로 멈춘다.
                generate_kwargs["stopping_criteria"] = json_stop_criteria(self._tokenizer, input_len, prefill)
            with torch.inference_mode():
                outputs = model.generate(**inputs, **generate_kwargs)
            generated = outputs[0][input_len:]
            tokens = int(generated.shape[-1])
            # 측정용: 토큰 수·상한 도달 여부만 남긴다(모델 출력·상담 내용은 남기지 않는다).
            _LOG.info(
                "[MEDGEMMA-GEN] tokens=%d limit=%d hit_limit=%s rep=%s quant=%s",
                tokens, limit, tokens >= limit, self._json_repetition_label, self._quant,
            )
            text = self._processor.decode(generated, skip_special_tokens=True)
            if not text.strip():
                raise RuntimeError("empty generation")
            return f"{prefill}{text}" if prefill else text

        tokenizer = self._tokenizer
        if tokenizer is None:
            raise RuntimeError(self._state.load_error or "tokenizer not loaded")

        prompt_text = (
            f"[SYSTEM]\n{system_prompt.strip()}\n\n"
            f"[USER]\n{user_prompt.strip()}\n\n"
            "[ASSISTANT]\n"
        )
        encoded = tokenizer(prompt_text, return_tensors="pt")
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self._settings.max_new_tokens,
                do_sample=self._settings.do_sample,
                temperature=self._settings.temperature,
                top_p=self._settings.top_p,
                repetition_penalty=self._settings.repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][input_ids.shape[-1] :]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        if not text.strip():
            raise RuntimeError("empty generation")
        return text

    def warmup(self) -> None:
        if not self._state.loaded or self._model is None or self._torch is None:
            raise RuntimeError(self._state.load_error or "model not loaded")

        model = self._model
        torch = self._torch
        if self._uses_processor:
            if self._processor is None:
                raise RuntimeError(self._state.load_error or "processor not loaded")
            messages = [
                {"role": "system", "content": "짧게 대답해 주세요."},
                {"role": "user", "content": "안녕하세요"},
            ]
            inputs = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device, dtype=torch.float16 if self._state.gpu else torch.float32)
            with torch.inference_mode():
                _ = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                )
            return

        tokenizer = self._tokenizer
        if tokenizer is None:
            raise RuntimeError(self._state.load_error or "tokenizer not loaded")

        encoded = tokenizer("안녕하세요", return_tensors="pt")
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)

        with torch.no_grad():
            _ = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
