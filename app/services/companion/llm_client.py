from __future__ import annotations

import concurrent.futures
import logging
import threading
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

from app.core.config import get_settings


logger = logging.getLogger(__name__)

# LoRA 어댑터 이름 (11/10). None = 어댑터를 끈 기본 모델 (위험·신호·요약·마지막 인사)
MOOD_ADAPTER = "mood"
REPLY_ADAPTER = "reply"


class LlmUnavailable(Exception):
    """LLM 을 쓸 수 없음 (모델 파일 없음, 불러오기 실패, 시간 초과). 응답 쪽은 재시도 뒤 대체 문장으로 간다 (D20)."""


def llm_load_kwargs(dtype, cuda_available: bool, quant: str) -> dict:
    """모델을 불러올 때 쓸 옵션. GPU가 있으면 GPU에 올린다 (CPU 메모리에 올리면 서버 메모리가 부족해진다).

    quant 가 int8 이면 8비트로 줄여 GPU 메모리를 아낀다. GPU가 없으면 CPU에 올린다.
    """
    kwargs = {"local_files_only": True, "trust_remote_code": True, "dtype": dtype}
    if cuda_available:
        kwargs["device_map"] = {"": 0}
        if quant == "int8":
            from transformers import BitsAndBytesConfig  # type: ignore[reportMissingImports]

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    return kwargs


class LlmClient(Protocol):
    """신호 추출과 응답 생성이 같이 쓰는 LLM 인터페이스 (S5-4)."""

    def generate(self, prompt: str, *, temperature: float, num_predict: int, timeout: float) -> str: ...


class AdapterGate:
    """모델 한 개에 얹은 LoRA 어댑터를 요청마다 바꿀 때 쓰는 문.

    어댑터는 모델 전체에 걸리는 상태라서, 다른 어댑터를 쓰는 호출이 겹치면 서로의 어댑터로 계산된다.
    같은 어댑터를 쓰는 호출은 함께 들어가고, 다른 어댑터가 필요하면 진행 중인 호출이 모두 끝난 뒤 바꾼다.
    다른 어댑터를 기다리는 호출이 있으면 새로 오는 호출도 줄을 선다 (한쪽이 계속 밀리지 않게).
    """

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._current: str | None = None
        self._active = 0
        self._waiting: Counter = Counter()

    @property
    def current(self) -> str | None:
        return self._current

    def _can_enter(self, name: str | None) -> bool:
        if self._active == 0:
            return True
        others_waiting = sum(n for k, n in self._waiting.items() if k != name)
        return self._current == name and others_waiting == 0

    @contextmanager
    def use(self, name: str | None, switch):
        """name 어댑터로 바꾼 상태에서 실행한다. switch(name) 은 실제로 모델의 어댑터를 바꾸는 함수."""
        with self._cv:
            self._waiting[name] += 1
            try:
                while not self._can_enter(name):
                    self._cv.wait()
            finally:
                self._waiting[name] -= 1
            if self._current != name:
                switch(name)
                self._current = name
            self._active += 1
        try:
            yield
        finally:
            with self._cv:
                self._active -= 1
                self._cv.notify_all()


def _with_timeout(fn, timeout: float, label: str, *args):
    """fn 을 따로 돌리고 timeout 초만 기다린다. 넘으면 LlmUnavailable (넘긴 작업은 백그라운드에서 끝난다)."""
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(fn, *args)
    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError as exc:
        raise LlmUnavailable(f"LLM {label} 시간 초과") from exc
    except LlmUnavailable:
        raise
    except Exception as exc:
        raise LlmUnavailable(f"LLM {label} 실패 ({type(exc).__name__})") from exc
    finally:
        pool.shutdown(wait=False)


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
        self._adapters: set[str] = set()
        self._gate = AdapterGate()

    def model_dir(self) -> Path:
        s = get_settings()
        return Path(s.model_base_path) / s.companion_llm_subdir

    def adapter_dirs(self) -> dict[str, Path]:
        s = get_settings()
        base = Path(s.model_base_path)
        return {MOOD_ADAPTER: base / s.companion_mood_lora_subdir, REPLY_ADAPTER: base / s.companion_reply_lora_subdir}

    def _attach_adapters(self, model):
        """있는 어댑터만 얹는다. 하나도 없으면 기본 모델 그대로 (재희 동작). 얹지 못하면 기본 모델로 계속 쓴다."""
        found = {k: p for k, p in self.adapter_dirs().items() if (p / "adapter_config.json").is_file()}
        if not found:
            return model
        try:
            from peft import PeftModel  # type: ignore[reportMissingImports]

            # EXAONE 3.5 원격 코드에는 입력·출력 임베딩 함수가 없어 peft 가 찾지 못한다 (DGX 시험과 같은 보정)
            body, top = type(model.transformer), type(model)
            body.get_input_embeddings = lambda self: self.wte
            top.get_input_embeddings = lambda self: self.transformer.wte
            top.get_output_embeddings = lambda self: self.lm_head
            names = list(found)
            peft_model = PeftModel.from_pretrained(model, str(found[names[0]]), adapter_name=names[0])
            for name in names[1:]:
                peft_model.load_adapter(str(found[name]), adapter_name=name)
            peft_model.eval()
            peft_model.base_model.disable_adapter_layers()  # 처음은 기본 모델 상태 (gate.current = None)
            self._adapters = set(names)
            logger.info("[COMPANION-LLM] LoRA adapters loaded: %s", ",".join(sorted(names)))
            return peft_model
        except Exception as exc:  # noqa: BLE001 - 어댑터가 없어도 대화는 기본 모델로 계속된다
            logger.error("[COMPANION-LLM] LoRA adapters not loaded (%s)", type(exc).__name__)
            self._adapters = set()
            return model

    def _switch(self, name: str | None) -> None:
        """gate 안에서만 부른다."""
        if not self._adapters:
            return
        if name is None:
            self._model.base_model.disable_adapter_layers()
        else:
            self._model.base_model.enable_adapter_layers()
            self._model.set_adapter(name)

    def has_adapter(self, name: str) -> bool:
        """어댑터를 쓸 수 있는지 (모델을 처음이면 올린다). 모델을 못 올리면 LlmUnavailable."""
        self._load()
        return name in self._adapters

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
                kwargs = llm_load_kwargs(dtype, torch.cuda.is_available(), get_settings().companion_llm_quant)
                model = AutoModelForCausalLM.from_pretrained(str(path), **kwargs)
                model.eval()
                model = self._attach_adapters(model)
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
        ids = encoded["input_ids"].to(model.device)  # 모델이 GPU에 있으면 입력도 GPU로 보낸다
        sample = temperature > 0
        with self._gate.use(None, self._switch), torch.no_grad():
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

    def _next_token_probs_blocking(self, prompt: str, prefix: str, candidates: tuple[str, ...]) -> dict[str, float]:
        """답을 만들지 않고, prefix 바로 다음 글자가 candidates 중 무엇일지의 확률을 읽는다 (후보끼리만 비교)."""
        import torch

        tokenizer, model = self._load()
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False) + prefix
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        with self._gate.use(None, self._switch), torch.no_grad():
            logits = model(**enc).logits[0, -1]
        ids = [tokenizer.encode(c, add_special_tokens=False)[0] for c in candidates]
        probs = torch.softmax(logits[ids].float(), dim=0).tolist()
        return dict(zip(candidates, probs))

    def next_token_probs(self, prompt: str, prefix: str, candidates: tuple[str, ...], timeout: float) -> dict[str, float]:
        """시간 제한을 넘으면 LlmUnavailable."""
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(self._next_token_probs_blocking, prompt, prefix, candidates)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as exc:
            raise LlmUnavailable("LLM 확률 계산 시간 초과") from exc
        except LlmUnavailable:
            raise
        except Exception as exc:
            raise LlmUnavailable(f"LLM 확률 계산 실패 ({type(exc).__name__})") from exc
        finally:
            pool.shutdown(wait=False)

    # --- 11/10: system + user 대화 형식, 어댑터 선택 (위험 v9·기분 e900·대답 cc5) ---

    def _chat_ids(self, tokenizer, system: str | None, user: str):
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                             return_dict=True)["input_ids"][0]

    def _need_adapter(self, adapter: str | None) -> None:
        if adapter is not None and adapter not in self._adapters:
            raise LlmUnavailable(f"LoRA 어댑터가 없습니다: {adapter}")

    def _message_probs_blocking(self, system, user, prefix, candidates, adapter) -> dict[str, float]:
        import torch

        tokenizer, model = self._load()
        self._need_adapter(adapter)
        # DGX 시험(risk_v9·fuse_cmp)과 같게: 대화 틀 토큰 뒤에 접두 토큰을 그대로 잇는다 (문자열로 합쳐 다시 자르지 않는다)
        prompt_ids = self._chat_ids(tokenizer, system, user)
        prefix_ids = torch.tensor(tokenizer(prefix, add_special_tokens=False)["input_ids"], dtype=prompt_ids.dtype)
        ids = torch.cat([prompt_ids, prefix_ids])[None].to(model.device)
        with self._gate.use(adapter, self._switch), torch.inference_mode():
            logits = model(input_ids=ids).logits[0, -1].float()
        cand_ids = [tokenizer(c, add_special_tokens=False)["input_ids"][0] for c in candidates]
        probs = torch.softmax(logits[cand_ids], dim=0).tolist()
        return dict(zip(candidates, probs))

    def message_token_probs(self, system: str | None, user: str, prefix: str, candidates: tuple[str, ...], *,
                            adapter: str | None = None, timeout: float) -> dict[str, float]:
        """system·user 대화 뒤 prefix 다음 첫 토큰이 candidates 중 무엇일지의 확률 (후보끼리만 비교).

        adapter 가 None 이면 어댑터를 끈 기본 모델. 시간 제한·실패·어댑터 없음은 LlmUnavailable.
        """
        return _with_timeout(self._message_probs_blocking, timeout, "확률 계산", system, user, prefix, candidates, adapter)

    def _generate_adapter_blocking(self, system, user, adapter, temperature, num_predict) -> str:
        import torch

        tokenizer, model = self._load()
        self._need_adapter(adapter)
        ids = self._chat_ids(tokenizer, system, user)[None].to(model.device)
        sample = temperature > 0
        with self._gate.use(adapter, self._switch), torch.no_grad():
            out = model.generate(
                input_ids=ids,
                max_new_tokens=num_predict,
                do_sample=sample,
                temperature=temperature if sample else None,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)

    def generate_adapter(self, user: str, *, adapter: str, system: str | None = None, temperature: float,
                         num_predict: int, timeout: float) -> str:
        """어댑터를 켜고 답을 만든다. 어댑터가 없거나 실패하면 LlmUnavailable (부르는 쪽이 기본 모델로 돌아간다)."""
        return _with_timeout(self._generate_adapter_blocking, timeout, "생성", system, user, adapter, temperature, num_predict)


llm_client = ExaoneLlm()
