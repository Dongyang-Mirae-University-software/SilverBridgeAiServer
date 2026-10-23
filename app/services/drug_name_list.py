from __future__ import annotations

import bisect
import difflib
import logging
import re
from functools import lru_cache
from pathlib import Path

from app.schemas.medication_ocr_schema import MedicationOcrResponse

_LOG = logging.getLogger(__name__)

# 식약처 의약품 제품 허가정보 품목명(갱신 방법은 README "약 이름 허가 목록").
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "drug_names.txt"

SUGGESTION_LIMIT = 3
SUGGESTION_CUTOFF = 0.6
# "정" 한 글자가 수만 개 품목의 앞부분으로 잡히지 않게, 앞부분 일치는 이 길이 이상만 본다(정확 일치는 길이 무관).
PREFIX_MIN_LEN = 3

_BRACKETS = re.compile(r"\([^()]*\)|\[[^\[\]]*\]")
# 숫자(소수·"500/20" 같은 복합 포함) + 단위. 봉투에서 잘린 "m"도 단위로 본다.
_DOSE = re.compile(
    r"\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)*\s*(?:밀리그램|밀리그람|마이크로그램|그램|밀리리터|mcg|mg|ml|g|%|m)",
    re.IGNORECASE,
)
_SEPARATORS = re.compile(r"[\s\-_/·,.]")


def strip_brackets(text: str) -> str:
    """괄호·대괄호와 그 안 내용을 지운다(중첩은 안쪽부터 반복)."""
    previous = None
    while previous != text:
        previous, text = text, _BRACKETS.sub("", text)
    return text


def normalize(name: str) -> str:
    text = strip_brackets(name.replace("\r", "").replace("\n", ""))
    text = _DOSE.sub("", text)
    return _SEPARATORS.sub("", text).lower()


class DrugNameList:
    """정규화 키 → 표시용 이름. 기동 때 한 번 읽고 이후 읽기 전용이다."""

    def __init__(self, names: dict[str, str]) -> None:
        self._names = names
        self._keys = sorted(names)

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH) -> DrugNameList:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            _LOG.warning("[MED-OCR] 약 이름 허가 목록을 읽지 못해 대조를 생략한다 (%s)", type(exc).__name__)
            return cls({})
        names: dict[str, str] = {}
        for line in lines:
            key = normalize(line)
            if key:
                names.setdefault(key, strip_brackets(line).strip())
        if not names:
            _LOG.warning("[MED-OCR] 약 이름 허가 목록이 비어 있어 대조를 생략한다")
        return cls(names)

    @property
    def available(self) -> bool:
        return bool(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def check(self, name: str) -> tuple[bool | None, list[str]]:
        """(inList, suggestions). 목록이 없으면 (None, [])."""
        if not self._names:
            return None, []
        key = normalize(name)
        if not key:
            return False, []
        if key in self._names or self._has_prefix(key):
            return True, []
        suggestions: list[str] = []
        for match in difflib.get_close_matches(key, self._keys, n=SUGGESTION_LIMIT, cutoff=SUGGESTION_CUTOFF):
            display = self._names[match]
            if display not in suggestions:
                suggestions.append(display)
        return False, suggestions

    def _has_prefix(self, key: str) -> bool:
        if len(key) < PREFIX_MIN_LEN:
            return False
        index = bisect.bisect_left(self._keys, key)
        return index < len(self._keys) and self._keys[index].startswith(key)

    def annotate(self, result: MedicationOcrResponse) -> MedicationOcrResponse:
        """drugs[] 에 inList·suggestions 를 채운다. 이름 원문은 바꾸지 않는다(고르는 건 보호자)."""
        drugs = []
        for drug in result.drugs:
            in_list, suggestions = self.check(drug.name)
            drugs.append(drug.model_copy(update={"inList": in_list, "suggestions": suggestions}))
        return result.model_copy(update={"drugs": drugs})


@lru_cache
def get_drug_name_list() -> DrugNameList:
    return DrugNameList.load()
