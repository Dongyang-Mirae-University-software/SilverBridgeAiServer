from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from app.core.config import Settings


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
                        self._model = AutoModelForImageTextToText.from_pretrained(model_id, token=token, **kwargs)
                        self._tokenizer = getattr(self._processor, "tokenizer", None)
                    else:
                        self._tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
                        self._model = AutoModelForCausalLM.from_pretrained(model_id, token=token, **kwargs)
                    if gpu_ok and hasattr(self._model, "to"):
                        self._model = self._model.to("cuda:0")
                    self._state.loaded = True
                    self._state.load_error = None
                    self._state.model_name = model_id
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

    def generate_structured_text(self, system_prompt: str, user_prompt: str) -> str:
        if not self._state.loaded or self._model is None or self._torch is None:
            raise RuntimeError(self._state.load_error or "model not loaded")

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
            input_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self._settings.max_new_tokens,
                    do_sample=False,
                )
            generated = outputs[0][input_len:]
            text = self._processor.decode(generated, skip_special_tokens=True)
            if not text.strip():
                raise RuntimeError("empty generation")
            return text

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
