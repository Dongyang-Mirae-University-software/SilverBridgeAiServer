"""EXAONE 원격 코드(modeling_exaone.py)를 transformers 5.13.1 에 맞게 고친다.

왜 필요한가: EXAONE 폴더의 modeling_exaone.py 는 transformers 4.x 시절 API로 쓰여 있다.
  - create_causal_mask 에 input_embeds= (5.x 는 inputs_embeds=)
  - create_causal_mask 에 cache_position= (5.x 는 받지 않음)
서버는 MedGemma 때문에 transformers 5.13.1 을 유지해야 하므로 코드 쪽을 고친다.

사용: 모델을 올린 뒤 한 번 실행한다. 이미 고쳐져 있으면 아무것도 바꾸지 않는다.
  python scripts/companion/patch_exaone_remote_code.py models/companion/llm/exaone-3.5-2.4b
  (원본은 modeling_exaone.py.orig-5.13 으로 백업한다. 이미 있으면 덮지 않는다.)
"""

import shutil
import sys
from pathlib import Path

TARGET = "modeling_exaone.py"

# 두 치환은 CRLF 줄 끝을 기준으로 한다(원본 파일 형식 그대로).
FIX_KEYWORD = (b"input_embeds=inputs_embeds", b"inputs_embeds=inputs_embeds")
FIX_CACHE_ARG = (
    b"attention_mask=attention_mask,\r\n"
    b"            cache_position=cache_position,\r\n"
    b"            past_key_values=past_key_values,\r\n"
    b"            position_ids=position_ids,\r\n"
    b"        )\r\n\r\n"
    b"        hidden_states = inputs_embeds",
    b"attention_mask=attention_mask,\r\n"
    b"            past_key_values=past_key_values,\r\n"
    b"            position_ids=position_ids,\r\n"
    b"        )\r\n\r\n"
    b"        hidden_states = inputs_embeds",
)


def patch(model_dir: Path) -> str:
    path = model_dir / TARGET
    if not path.is_file():
        raise SystemExit(f"파일이 없습니다: {path}")
    data = path.read_bytes()

    if FIX_KEYWORD[0] not in data and FIX_CACHE_ARG[0] not in data:
        return "이미 고쳐져 있습니다 (변경 없음)"

    fixed = data
    for old, new in (FIX_KEYWORD, FIX_CACHE_ARG):
        count = fixed.count(old)
        if count != 1:
            raise SystemExit(f"예상과 다른 코드입니다 (일치 {count}곳). 자동 수정하지 않습니다: {path}")
        fixed = fixed.replace(old, new)

    backup = model_dir / (TARGET + ".orig-5.13")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_bytes(fixed)
    return f"고쳤습니다 (원본 백업: {backup.name})"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("사용법: python scripts/companion/patch_exaone_remote_code.py <EXAONE 모델 폴더>")
    print(patch(Path(sys.argv[1])))
