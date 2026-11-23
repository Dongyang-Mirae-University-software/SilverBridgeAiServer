"""replay_offline 을 그대로 돌리되 MedGemma 생성 호출마다 걸린 시간(ms)만 기록한다(본문 없음).
app 을 먼저 불러오므로 replay_offline 이 stub 모드에서 정하는 환경을 미리 똑같이 넣는다."""
import json, os, runpy, sys, tempfile, time
os.environ["DATABASE_URL"] = f"sqlite:///{tempfile.mkdtemp(prefix='replay_offline_')}/r.db"
os.environ.setdefault("STREAM_STATE_BACKEND", "memory")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ["FOLLOW_UP_ENABLED"] = "false"
os.environ["ANTHROPIC_API_KEY"] = ""
sys.path.insert(0, os.getcwd())
from app.models import medgemma_loader as ml
LOG = os.environ["GEN_TIME_LOG"]
_orig = ml.MedGemmaLoader.generate_structured_text
def _timed(self, *a, **k):
    t = time.perf_counter(); ok = False
    try:
        r = _orig(self, *a, **k); ok = True; return r
    finally:
        with open(LOG, "a") as f:
            f.write(json.dumps({"ms": round((time.perf_counter() - t) * 1000), "ok": ok}) + "\n")
ml.MedGemmaLoader.generate_structured_text = _timed
sys.argv = ["tools/eval/replay_offline.py"] + sys.argv[1:]
runpy.run_path("tools/eval/replay_offline.py", run_name="__main__")
