#!/usr/bin/env python3
"""측정 전 동결: 답 생성 환경 + 채점 환경을 freeze.json 에 적고 각 파일 sha256 을 남긴다. 본문 없음."""
import hashlib, json, pathlib, subprocess, sys, datetime

H = pathlib.Path.home()
J = H / "chat-eval/judge"
HUB = H / ".cache/huggingface/hub"


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def ref(m):
    p = HUB / f"models--{m.replace('/', '--')}/refs/main"
    return p.read_text().strip() if p.exists() else None


def dist(path, names):
    out = {}
    for n in names:
        hits = sorted(pathlib.Path(path).glob(f"{n}-*.dist-info"))
        out[n] = hits[-1].name[len(n) + 1:-len(".dist-info")] if hits else None
    return out


def main():
    commit = sys.argv[1] if len(sys.argv) > 1 else "bac824b"
    code = H / f"chat-eval/aiserver-{commit}"
    vj = next((H / "chat-eval/venv_judge/lib").glob("python*/site-packages"), None)
    f = {
        "frozenAt": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "answer": {
            "aiCommit": commit,
            "codeFiles": {p: sha(code / p) for p in ["app/models/medgemma_loader.py", "app/core/config.py", "tools/eval/replay_offline.py"]},
            "env": {"MEDGEMMA_ENABLED": "true", "MEDGEMMA_QUANT": "int8", "MEDGEMMA_JSON_REPETITION": "gen_ngram:10",
                    "DEFAULT_CHAT_MODEL": "google/medgemma-1.5-4b-it", "others": "코드 기본값(gosky .env 에도 없음)"},
            "router": "stub",
            "libs": dist(H / "chat-eval/pylib_gosky", ["transformers", "tokenizers", "accelerate", "bitsandbytes"]),
            "torch": subprocess.run([str(H / "kcs-mjng-yolo/venv/bin/python"), "-c", "import torch;print(torch.__version__)"],
                                    capture_output=True, text=True).stdout.strip(),
            "model": {"google/medgemma-1.5-4b-it": ref("google/medgemma-1.5-4b-it")},
        },
        "judge": {
            "models": {m: {"rev": ref(m), "quant": q} for m, q in [("openai/gpt-oss-120b", "MXFP4"), ("Qwen/Qwen3-32B-FP8", "FP8"),
                                                                     ("LGAI-EXAONE/EXAONE-4.0-32B", "BF16")]},
            "engine": dist(vj, ["vllm", "torch"]) if vj else None,
            "files": {p.name: sha(p) for p in sorted(J.glob("*")) if p.suffix in (".py", ".txt", ".sh")},
            "temperature": 0.0, "seed": 42, "extra": {"gpt-oss-120b": {"reasoning_effort": "low"}},
            "rule": "기준별 3표 단순 다수결(2:1 은 split 표시·사람 검토 1순위), 문항 통과 = 모든 기준 다수결 충족, 영역 점수 Wilson 95%",
            "selfConsistency": "gpt-oss-120b 3회 seed 42/43/44 temperature 0.3, Krippendorff α",
        },
    }
    out = J / "freeze.json"
    out.write_text(json.dumps(f, ensure_ascii=False, indent=1))
    print("freeze.json sha256", sha(out))


if __name__ == "__main__":
    main()
