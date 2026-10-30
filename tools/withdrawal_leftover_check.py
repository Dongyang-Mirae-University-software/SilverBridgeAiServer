"""탈퇴 정리에서 빠진 회원 데이터 대조(2026-10-30 탈퇴 정리 점검 L-1) - 읽기 전용, 개수만 출력한다.

BE 회원(gosky·vkcs-linux 두 서버) ↔ AI 챗 기록(user_id·대리 상담 wardId)·AI 예약 키 행·예약 서비스 sb- 계정을 맞춰,
BE에 없는 ID가 각 저장소에 몇 개 남았는지 보여 준다. 측정 실행기 ID(eval-*)는 따로 센다. ID·내용·키는 출력하지 않는다.
정리는 하지 않는다 - 남은 것이 있으면 사용자 실행 스크립트로 지운다(AI DELETE API 또는 백업 + 가드 스크립트).

실행(ssh 별칭 gosky·vkcs-linux 가 되는 작업 컴퓨터에서): python3 tools/withdrawal_leftover_check.py
"""
from __future__ import annotations

import json
import subprocess
import sys

_BE_SQL = "SELECT json_agg(id) FROM users;"
_AI_SQL = (
    "SELECT json_build_object("
    "'chat', (SELECT json_agg(DISTINCT user_id) FROM chat_logs),"
    "'proxy', (SELECT json_agg(DISTINCT context_json::jsonb->'userContext'->>'wardId') FROM chat_logs "
    "          WHERE context_json::jsonb->'userContext'->>'wardId' IS NOT NULL),"
    "'cred', (SELECT json_agg(user_id) FROM reservation_credentials));"
)
_RESV_JS = (
    "const {PrismaClient}=require('@prisma/client');const p=new PrismaClient();"
    "p.user.findMany({where:{email:{startsWith:'sb-',endsWith:'@silverbridge.local'}},select:{email:true}})"
    ".then(r=>{console.log(JSON.stringify(r.map(x=>x.email.slice(3,-'@silverbridge.local'.length))));return p.\\$disconnect()})"
)


def _psql(host: str, container: str, db: str, sql: str) -> object:
    cmd = f'docker exec -i {container} sh -c "psql -U \\"\\$POSTGRES_USER\\" -d {db} -tA"'
    out = subprocess.run(["ssh", "-o", "BatchMode=yes", host, cmd], input=sql + "\n", capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip() or "null")


def _reservation_accounts() -> list:
    cmd = f"docker exec -w /app silverbridgereservation-reservation-api-1 node -e \"{_RESV_JS}\""
    out = subprocess.run(["ssh", "-o", "BatchMode=yes", "gosky", cmd], capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1] or "[]")


def leftover_counts(be_ids: set[str], stores: dict[str, list]) -> dict[str, dict[str, int]]:
    """저장소별로 BE에 없는 ID 수(회원 ID·측정 실행기 ID 따로)."""
    result: dict[str, dict[str, int]] = {}
    for name, ids in stores.items():
        missing = {str(i) for i in (ids or []) if i} - be_ids
        result[name] = {
            "total": len({str(i) for i in (ids or []) if i}),
            "notInBe": sum(1 for i in missing if not i.startswith("eval")),
            "eval": sum(1 for i in missing if i.startswith("eval")),
        }
    return result


def main() -> int:
    be = set(_psql("gosky", "dmu-dev-db", '\\"\\$POSTGRES_DB\\"', _BE_SQL) or []) | set(
        _psql("vkcs-linux", "dmu-dev-db", '\\"\\$POSTGRES_DB\\"', _BE_SQL) or []
    )
    ai = _psql("gosky", "silverbridge-ai-postgres", "silverbridge_ai", _AI_SQL) or {}
    stores = {
        "aiChatLogs(user_id)": ai.get("chat") or [],
        "aiProxyLogs(wardId)": ai.get("proxy") or [],
        "aiReservationCredentials": ai.get("cred") or [],
        "reservationAccounts(sb-)": _reservation_accounts(),
    }
    print(f"BE members: {len(be)}")
    for name, c in leftover_counts({str(i) for i in be}, stores).items():
        print(f"{name}: total {c['total']}, not in BE {c['notInBe']}, eval {c['eval']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
