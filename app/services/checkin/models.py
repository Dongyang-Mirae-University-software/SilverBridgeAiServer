"""안부 말벗 모델 묶음: SenseVoiceSmall(받아쓰기) + EXAONE 3.5 2.4B 4비트(대화·정리·정서).

- 받아쓰기: SenseVoiceSmall (2026-10-26 DGX 어르신 1,200문장 글자 오류 5.5%, 문장당 0.03초.
  faster-whisper large-v3 는 CPU 문장당 14초에 자막 문구 덧붙임·빈 결과가 있어 쓰지 않는다)
- 공감·꼬리질문은 기본 모델(추가 가중치 끔), 정리·정서는 추가 가중치를 켠 같은 모델로 한다 - 모델은 하나만 올린다.
- GPU 생성은 한 번에 하나(_gen_lock). 기다리다 넘치면 CheckinBusy 로 바로 거절한다(요청 스레드를 붙들지 않게).

테스트는 get_models() 를 가짜로 바꿔 모델 없이 돈다.
"""

from __future__ import annotations

import json
import logging
import re
import threading

from app.services.checkin.config import get_checkin_settings

log = logging.getLogger(__name__)

SUMMARY_FIELDS = ("sleep", "meal", "pain", "mood", "lonely")
SUMMARY_ALLOWED = {"sleep": {"좋음", "나쁨"}, "meal": {"먹음", "거름"}, "pain": {"있음", "없음"},
                   "mood": {"좋음", "보통", "나쁨"}, "lonely": {"있음", "없음"}}
EMOTIONS = ("기쁨", "편안", "외로움", "기운 없음", "걱정")

_SCHEMA = '{"sleep":"좋음|나쁨|모름","meal":"먹음|거름|모름","pain":"있음|없음|모름","mood":"좋음|보통|나쁨|모름","lonely":"있음|없음|모름"}'
# DGX 시험(2026-10-20 v2 + 10/22 QLoRA)과 같은 문장이어야 한다 - 추가 가중치가 이 문장으로 배웠다.
PROMPT_SUMMARY = f"""다음은 어르신이 안부 질문에 한 대답입니다. 대답을 아래 형식의 JSON 한 줄로 정리하세요. 다른 말은 쓰지 마세요.
{_SCHEMA}

판정 규칙:
- 대답에 나오지 않은 칸은 반드시 "모름"이다. 짐작해서 채우지 마라.
- sleep: 잘 잤다·그럭저럭 잤다처럼 불편을 말하지 않으면 "좋음". 못 잤다·뒤척였다·자다가 깼다면 "잘 잤다"고 했어도 "나쁨".
- meal: 조금이라도 먹었으면 "먹음". 안 먹었다·걸렀다·아직 안 먹었다면 "거름".
- pain: 아프다·쑤신다·결린다·저리다·쓰리다·지끈거린다면 "있음". 아픈 데 없다·괜찮다고 하면 "없음".
- mood: 기분 좋다·즐겁다·반갑다면 "좋음". 그냥 그렇다·괜찮다·별일 없다면 "보통". 우울·속상·걱정·귀찮음·적적·허전하면 "나쁨". 여러 감정이 나오면 마지막에 말한 지금 감정을 따른다.
- lonely: 외롭다·적적하다·허전하다·혼자라 싫다·말할 사람이 없다면 "있음". 외롭지 않다고 하거나 가족·친구와 함께 지냈다·통화했다면 "없음".
- 외로움이 "있음"이면 mood는 "나쁨"이다.

예시
대답: 눕긴 일찍 누웠는데 새벽 세 시에 깨서 못 잤어. 아침은 먹었어.
{{"sleep":"나쁨","meal":"먹음","pain":"모름","mood":"모름","lonely":"모름"}}
대답: 입맛이 없어서 저녁은 그냥 넘겼어. 요즘 혼자라 적적해.
{{"sleep":"모름","meal":"거름","pain":"모름","mood":"나쁨","lonely":"있음"}}
대답: 어깨가 좀 결려. 그래도 기분은 괜찮아.
{{"sleep":"모름","meal":"모름","pain":"있음","mood":"보통","lonely":"모름"}}
대답: 어제 아들 내외가 와서 같이 밥 먹었어. 좋았지.
{{"sleep":"모름","meal":"먹음","pain":"모름","mood":"좋음","lonely":"없음"}}"""

# 2026-11-01 정서 5가지 학습(감성대화 노년)과 같은 문장
PROMPT_EMOTION = """다음은 어르신이 안부 질문에 한 대답입니다. 어르신의 정서를 아래 다섯 가지 중 하나로 고르세요.
- 기쁨: 즐겁고 기분이 좋음, 고마움, 만족
- 편안: 평소와 같음, 특별한 일 없음, 안심함
- 외로움: 적적함, 보고 싶음, 혼자라고 느낌, 찾아오는 사람이 없음
- 기운 없음: 슬픔, 우울함, 무기력, 낙담, 마음이 아픔
- 걱정: 불안, 근심, 두려움, 짜증, 스트레스
근거가 부족하면 "모름"이라고 쓰세요. 다른 말 없이 JSON 한 줄로만 답하세요: {"emotion": "..."}"""

PROMPT_REPLY = """당신은 혼자 사시는 어르신께 안부를 여쭙는 다정한 말벗입니다.
어르신의 대답을 듣고 공감하는 짧은 한 문장과, 이어서 여쭐 짧은 질문 한 문장을 존댓말로 하세요.
- 두 문장을 넘기지 마세요. 이모지·괄호·영어를 쓰지 마세요.
- 진단하거나 약·치료를 권하지 마세요. 사실을 짐작해 단정하지 마세요.
- 어르신이 이야기를 마치고 싶어 하시면 따뜻한 인사 한 문장만 하세요."""


class CheckinBusy(RuntimeError):
    pass


class CheckinModels:
    def __init__(self) -> None:
        s = get_checkin_settings()
        import torch
        from funasr import AutoModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self._torch = torch
        self.stt = AutoModel(model=s.sensevoice_model, device=s.device, disable_update=True, log_level="ERROR")
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_use_double_quant=True)
        self.tok = AutoTokenizer.from_pretrained(s.exaone_model, trust_remote_code=True)
        lm = AutoModelForCausalLM.from_pretrained(s.exaone_model, trust_remote_code=True, quantization_config=q,
                                                  device_map={"": s.device})
        self.adapters: set[str] = set()
        paths = {name: p for name, p in (("summary", s.summary_adapter), ("emotion", s.emotion_adapter)) if p}
        if paths:
            from peft import PeftModel

            # EXAONE 3.5 remote code 에는 get_input_embeddings 가 없어 peft 가 실패한다(transformers 5)
            TM, CL = type(lm.transformer), type(lm)
            TM.get_input_embeddings = lambda self: self.wte
            CL.get_input_embeddings = lambda self: self.transformer.wte
            CL.get_output_embeddings = lambda self: self.lm_head
            first, *rest = paths.items()
            lm = PeftModel.from_pretrained(lm, first[1], adapter_name=first[0])
            for name, path in rest:
                lm.load_adapter(path, adapter_name=name)
            self.adapters = set(paths)
        self.has_adapter = bool(self.adapters)
        self.lm = lm.eval()
        self._gen_lock = threading.BoundedSemaphore(1)
        self._wait = s.busy_wait_sec
        self._reply_tokens = s.reply_max_tokens

    # ---- 받아쓰기 ----
    def transcribe(self, arr) -> str:
        out = self.stt.generate(input=arr, language="ko", use_itn=False)
        return re.sub(r"<\|[^|]*\|>", "", out[0]["text"] if out else "").strip()

    # ---- 생성 ----
    def _generate(self, system: str, user: str, *, json_mode: bool, adapter: str | None, max_new: int) -> str:
        """adapter: "summary"·"emotion" 이면 그 추가 가중치를 켜고, None 이거나 없는 이름이면 기본 모델로 생성한다."""
        torch = self._torch
        if not self._gen_lock.acquire(timeout=self._wait):
            raise CheckinBusy()
        try:
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"]
            if json_mode:
                ids = torch.cat([ids, self.tok("{", add_special_tokens=False, return_tensors="pt")["input_ids"]], -1)
            ids = ids.to(self.lm.device)
            kw = dict(max_new_tokens=max_new, do_sample=False, pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
            with torch.inference_mode():
                if adapter in self.adapters:
                    self.lm.set_adapter(adapter)  # 잠금 안이라 다른 요청과 섞이지 않는다
                    out = self.lm.generate(input_ids=ids, attention_mask=torch.ones_like(ids), **kw)
                elif self.adapters:
                    with self.lm.disable_adapter():
                        out = self.lm.generate(input_ids=ids, attention_mask=torch.ones_like(ids), **kw)
                else:
                    out = self.lm.generate(input_ids=ids, attention_mask=torch.ones_like(ids), **kw)
            text = self.tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
            return ("{" + text) if json_mode else text
        finally:
            self._gen_lock.release()

    def reply(self, question: str, answer: str, last_turn: bool) -> str:
        hint = "\n(이번이 마지막이니 질문 없이 따뜻한 인사 한 문장으로 마무리하세요.)" if last_turn else ""
        raw = self._generate(PROMPT_REPLY, f"질문: {question}\n어르신 대답: {answer}{hint}", json_mode=False,
                             adapter=None, max_new=self._reply_tokens)
        return raw

    def summarize(self, utts: list[str]) -> dict:
        raw = self._generate(PROMPT_SUMMARY, "대답:\n" + "\n".join(utts), json_mode=True, adapter="summary", max_new=120)
        return parse_summary(raw)

    def emotion(self, utts: list[str]) -> str | None:
        raw = self._generate(PROMPT_EMOTION, "대답:\n" + "\n".join(utts), json_mode=True, adapter="emotion", max_new=30)
        return parse_emotion(raw)


def parse_summary(raw: str) -> dict:
    """따옴표 없는 키도 받고, 목록 밖 값·빠진 칸은 모름(None)."""
    s = (raw or "").strip()
    s = s[: s.find("}") + 1] if "}" in s else s
    s = re.sub(r'([{,]\s*)([A-Za-z_]+)\s*:', r'\1"\2":', s)
    try:
        d = json.loads(s)
    except json.JSONDecodeError:
        d = {}
    if not isinstance(d, dict):
        d = {}
    return {f: (d.get(f) if d.get(f) in SUMMARY_ALLOWED[f] else None) for f in SUMMARY_FIELDS}


def parse_emotion(raw: str) -> str | None:
    m = re.search(r'"?emotion"?\s*:\s*"([^"]+)"', raw or "")
    value = m.group(1).strip() if m else None
    return value if value in EMOTIONS else None


_models: CheckinModels | None = None
_load_lock = threading.Lock()


def get_models() -> CheckinModels:
    global _models
    if _models is None:
        with _load_lock:
            if _models is None:
                _models = CheckinModels()
                log.info("[CHECKIN] models loaded adapters=%s", sorted(_models.adapters))
    return _models
