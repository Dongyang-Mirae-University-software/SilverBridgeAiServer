from __future__ import annotations

import concurrent.futures
import itertools
import logging
import re
import threading
import time
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


class GateDeadline(LlmUnavailable):
    """문 앞에서 기다리다 마감 시각이 지났다 (계산하지 않고 포기한다)."""


class AdapterGate:
    """모델 한 개에 얹은 LoRA 어댑터를 요청마다 바꿀 때 쓰는 문.

    어댑터는 모델 전체에 걸리는 상태라서, 다른 어댑터를 쓰는 호출이 겹치면 서로의 어댑터로 계산된다.
    같은 어댑터를 쓰는 호출은 함께 들어가고, 다른 어댑터가 필요하면 진행 중인 호출이 모두 끝난 뒤 바꾼다.

    11/10 점검(AI #209) 반영:
    - 줄 순서(FIFO): 가장 먼저 선 호출의 어댑터가 다음 차례다. 차례가 오면 그 어댑터를 기다리던 호출이 한꺼번에 들어간다 (M-2).
    - 우선순위: 위험·기분 확률처럼 짧은 호출(priority=True)은 기다리는 생성보다 먼저 차례를 받는다.
      지금 쓰는 어댑터와 같으면 생성이 기다리고 있어도 바로 들어간다 (H-1 - 위험 판정이 대답 생성 뒤에 줄 서지 않게).
    - 마감 시각: 기다리다 지나면 GateDeadline (버려질 계산을 하지 않는다, H-5).
    - 전환이 실패해도 대기자를 깨운다 (L-3).
    """

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._current: str | None = None
        self._active = 0
        self._queue: list[tuple[int, str | None, bool]] = []
        self._tickets = itertools.count()

    @property
    def current(self) -> str | None:
        return self._current

    def _head_name(self) -> str | None:
        """다음 차례 어댑터. 우선 호출 중 가장 먼저 선 것, 없으면 전체에서 가장 먼저 선 것 (줄은 비어 있지 않을 때만 부른다)."""
        urgent = [w for w in self._queue if w[2]]
        return min(urgent or self._queue)[1]

    def _can_enter(self, name: str | None) -> bool:
        head = self._head_name()
        if self._active == 0:
            return name == head
        return name == self._current and head == self._current

    @contextmanager
    def use(self, name: str | None, switch, *, priority: bool = False, deadline: float | None = None):
        """name 어댑터로 바꾼 상태에서 실행한다. switch(name) 은 실제로 모델의 어댑터를 바꾸는 함수.

        deadline 은 time.monotonic() 기준 시각. 들어가기 전에 지나면 GateDeadline.
        """
        with self._cv:
            me = (next(self._tickets), name, priority)
            self._queue.append(me)
            try:
                while not self._can_enter(name):
                    if deadline is None:
                        self._cv.wait()
                        continue
                    rest = deadline - time.monotonic()
                    if rest <= 0:
                        raise GateDeadline("LLM 차례를 기다리다 마감 시각이 지났습니다")
                    self._cv.wait(rest)
                if self._current != name:
                    switch(name)
                    self._current = name
                self._active += 1
            finally:
                self._queue.remove(me)
                self._cv.notify_all()
        try:
            yield
        finally:
            with self._cv:
                self._active -= 1
                self._cv.notify_all()


def _with_timeout(fn, timeout: float, label: str, *args):
    """fn(*args, deadline=...) 을 따로 돌리고 timeout 초만 기다린다. 넘으면 LlmUnavailable.

    작업에도 같은 마감 시각을 넘긴다: 문 앞·생성 자리에서 기다리다 지나면 계산하지 않고, 생성은 남은 시간까지만 한다 (H-5).
    """
    deadline = time.monotonic() + timeout
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(fn, *args, deadline=deadline)
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


# EXAONE 대화 틀의 특수 토큰 꼴 ([|endofturn|] 등). 어르신 말에 들어 있으면 지운다 (L-1)
_SPECIAL = re.compile(r"\[\|[A-Za-z_]{1,24}\|\]")


def clean_user_text(text: str) -> str:
    return _SPECIAL.sub("", text)


def _drop_adapter(peft_model, name: str) -> bool:
    """반쯤 올라간 어댑터를 지운다. 지웠거나 남은 것이 없으면 True."""
    try:
        if name in getattr(peft_model, "peft_config", {}):
            peft_model.delete_adapter(name)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("[COMPANION-LLM] LoRA adapter cleanup failed: %s (%s)", name, type(exc).__name__)
        return False


def restore_base_layers(model) -> int:
    """모델 안에 끼워진 LoRA 층(peft BaseTunerLayer)을 원래 층으로 되돌린다. 되돌린 층 수를 돌려준다."""
    try:
        from peft.tuners.tuners_utils import BaseTunerLayer  # type: ignore[reportMissingImports]
    except Exception:  # noqa: BLE001 - peft 가 없으면 끼워진 층도 없다
        return 0
    restored = 0
    for name, module in list(model.named_modules()):
        if not name or not isinstance(module, BaseTunerLayer):
            continue
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, module.get_base_layer())
        restored += 1
    if restored:
        logger.warning("[COMPANION-LLM] LoRA layers removed from base model: %d", restored)
    return restored


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
        self._gen_slots: threading.BoundedSemaphore | None = None
        self._gen_slots_lock = threading.Lock()

    def model_dir(self) -> Path:
        s = get_settings()
        return Path(s.model_base_path) / s.companion_llm_subdir

    def adapter_dirs(self) -> dict[str, Path]:
        s = get_settings()
        base = Path(s.model_base_path)
        return {MOOD_ADAPTER: base / s.companion_mood_lora_subdir, REPLY_ADAPTER: base / s.companion_reply_lora_subdir}

    def _attach_adapters(self, model):
        """있는 어댑터만 얹는다. 하나도 없으면 기본 모델 그대로 (재희 동작).

        어댑터마다 따로 올리고, 실패한 어댑터만 빼고 나머지는 쓴다. 하나도 못 올리면 원본 모델에 이미 끼워진
        LoRA 층을 걷어내 깨끗한 기본 모델로 되돌린다 (11/10 점검 H-2: 걷어내지 않으면 위험·신호가 기분 LoRA로 계산됐다).
        """
        found = {k: p for k, p in self.adapter_dirs().items() if (p / "adapter_config.json").is_file()}
        self._adapters = set()
        if not found:
            return model
        try:
            from peft import PeftModel  # type: ignore[reportMissingImports]

            # EXAONE 3.5 원격 코드에는 입력·출력 임베딩 함수가 없어 peft 가 찾지 못한다 (DGX 시험과 같은 보정)
            body, top = type(model.transformer), type(model)
            body.get_input_embeddings = lambda self: self.wte
            top.get_input_embeddings = lambda self: self.transformer.wte
            top.get_output_embeddings = lambda self: self.lm_head
        except Exception as exc:  # noqa: BLE001 - 어댑터가 없어도 대화는 기본 모델로 계속된다
            logger.error("[COMPANION-LLM] LoRA adapters not loaded (%s)", type(exc).__name__)
            return model
        peft_model, loaded = None, []
        for name, path in found.items():
            try:
                if peft_model is None:
                    peft_model = PeftModel.from_pretrained(model, str(path), adapter_name=name)
                else:
                    peft_model.load_adapter(str(path), adapter_name=name)
                loaded.append(name)
            except Exception as exc:  # noqa: BLE001 - 그 어댑터만 빼고 계속한다
                logger.error("[COMPANION-LLM] LoRA adapter not loaded: %s (%s)", name, type(exc).__name__)
                if peft_model is not None and not _drop_adapter(peft_model, name):
                    loaded = []  # 지우지 못하면 어댑터를 모두 포기하고 아래에서 걷어낸다
                    break
        if peft_model is None or not loaded:
            restore_base_layers(model)
            return model
        peft_model.eval()
        peft_model.set_adapter(loaded[0])
        peft_model.base_model.disable_adapter_layers()  # 처음은 기본 모델 상태 (gate.current = None)
        self._adapters = set(loaded)
        logger.info("[COMPANION-LLM] LoRA adapters loaded: %s", ",".join(sorted(loaded)))
        return peft_model

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

    def warm_up(self) -> None:
        """모델·어댑터를 미리 올린다 (서버 시작 때 백그라운드에서, 11/10 점검 M-1).

        실패하면 실패 표시를 지워 첫 턴에서 한 번 더 시도하게 한다 (시작 때만의 일시 오류로 말벗 LLM이 계속 꺼져 있지 않게).
        """
        try:
            self._load()
        except LlmUnavailable:
            self._load_error = None
            raise

    @contextmanager
    def _gen_slot(self, deadline: float):
        """말벗 생성 동시 실행 상한 (COMPANION_LLM_GEN_CONCURRENCY, 0이면 없음). 마감까지 자리가 안 나면 GateDeadline."""
        limit = get_settings().companion_llm_gen_concurrency
        if limit <= 0:
            yield
            return
        with self._gen_slots_lock:
            if self._gen_slots is None:
                self._gen_slots = threading.BoundedSemaphore(limit)
            slots = self._gen_slots
        if not slots.acquire(timeout=max(0.0, deadline - time.monotonic())):
            raise GateDeadline("LLM 생성 자리를 기다리다 마감 시각이 지났습니다")
        try:
            yield
        finally:
            slots.release()

    @staticmethod
    def _gen_kwargs(tokenizer, temperature: float, num_predict: int, deadline: float) -> dict:
        sample = temperature > 0
        return {
            "max_new_tokens": num_predict,
            "do_sample": sample,
            "temperature": temperature if sample else None,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            # 마감까지 남은 시간만 생성한다 (넘기면 버려질 계산을 끝까지 하지 않게, H-5)
            "max_time": max(0.1, deadline - time.monotonic()),
        }

    def _generate_blocking(self, prompt: str, temperature: float, num_predict: int, *, deadline: float) -> str:
        import torch

        tokenizer, model = self._load()
        messages = [{"role": "user", "content": clean_user_text(prompt)}]
        # transformers 5.x 는 return_dict 기본값이 달라서 명시한다 (입력 ID 만 꺼낸다)
        encoded = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        ids = encoded["input_ids"].to(model.device)  # 모델이 GPU에 있으면 입력도 GPU로 보낸다
        with self._gen_slot(deadline), self._gate.use(None, self._switch, deadline=deadline), torch.no_grad():
            out = model.generate(ids, **self._gen_kwargs(tokenizer, temperature, num_predict, deadline))
        return tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)

    def generate(self, prompt: str, *, temperature: float, num_predict: int, timeout: float) -> str:
        """시간 제한을 넘으면 LlmUnavailable. 같은 마감 시각이 작업에도 걸린다."""
        return _with_timeout(self._generate_blocking, timeout, "생성", prompt, temperature, num_predict)

    def _next_token_probs_blocking(self, prompt: str, prefix: str, candidates: tuple[str, ...], *,
                                   deadline: float) -> dict[str, float]:
        """답을 만들지 않고, prefix 바로 다음 글자가 candidates 중 무엇일지의 확률을 읽는다 (후보끼리만 비교)."""
        import torch

        tokenizer, model = self._load()
        messages = [{"role": "user", "content": clean_user_text(prompt)}]
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False) + prefix
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        with self._gate.use(None, self._switch, priority=True, deadline=deadline), torch.no_grad():
            logits = model(**enc).logits[0, -1]
        ids = [tokenizer.encode(c, add_special_tokens=False)[0] for c in candidates]
        probs = torch.softmax(logits[ids].float(), dim=0).tolist()
        return dict(zip(candidates, probs))

    def next_token_probs(self, prompt: str, prefix: str, candidates: tuple[str, ...], timeout: float) -> dict[str, float]:
        """시간 제한을 넘으면 LlmUnavailable."""
        return _with_timeout(self._next_token_probs_blocking, timeout, "확률 계산", prompt, prefix, candidates)

    # --- 11/10: system + user 대화 형식, 어댑터 선택 (위험 v9·기분 e900·대답 cc5) ---

    def _chat_ids(self, tokenizer, system: str | None, user: str):
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": clean_user_text(user)}]
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                             return_dict=True)["input_ids"][0]

    def _need_adapter(self, adapter: str | None) -> None:
        if adapter is not None and adapter not in self._adapters:
            raise LlmUnavailable(f"LoRA 어댑터가 없습니다: {adapter}")

    def _message_probs_blocking(self, system, user, prefix, candidates, adapter, *, deadline: float) -> dict[str, float]:
        import torch

        tokenizer, model = self._load()
        self._need_adapter(adapter)
        # DGX 시험(risk_v9·fuse_cmp)과 같게: 대화 틀 토큰 뒤에 접두 토큰을 그대로 잇는다 (문자열로 합쳐 다시 자르지 않는다)
        prompt_ids = self._chat_ids(tokenizer, system, user)
        prefix_ids = torch.tensor(tokenizer(prefix, add_special_tokens=False)["input_ids"], dtype=prompt_ids.dtype)
        ids = torch.cat([prompt_ids, prefix_ids])[None].to(model.device)
        # 위험·기분 확률은 한 번의 forward(0.1초 안팎)라 기다리는 생성보다 먼저 들어간다 (H-1)
        with self._gate.use(adapter, self._switch, priority=True, deadline=deadline), torch.inference_mode():
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

    def _generate_adapter_blocking(self, system, user, adapter, temperature, num_predict, *, deadline: float) -> str:
        import torch

        tokenizer, model = self._load()
        self._need_adapter(adapter)
        ids = self._chat_ids(tokenizer, system, user)[None].to(model.device)
        with self._gen_slot(deadline), self._gate.use(adapter, self._switch, deadline=deadline), torch.no_grad():
            out = model.generate(input_ids=ids, **self._gen_kwargs(tokenizer, temperature, num_predict, deadline))
        return tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)

    def generate_adapter(self, user: str, *, adapter: str, system: str | None = None, temperature: float,
                         num_predict: int, timeout: float) -> str:
        """어댑터를 켜고 답을 만든다. 어댑터가 없거나 실패하면 LlmUnavailable (부르는 쪽이 기본 모델로 돌아간다)."""
        return _with_timeout(self._generate_adapter_blocking, timeout, "생성", system, user, adapter, temperature, num_predict)


llm_client = ExaoneLlm()
