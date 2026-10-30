from __future__ import annotations

import hashlib
import hmac
import re
import logging
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.services.chat_deadline import chat_deadline
from app.services.reservation_api_client import ReservationApiClient
from app.models.reservation_credential import ReservationCredential

_LOG = logging.getLogger(__name__)
# 회원 데이터 삭제 때 예약 서비스 키 폐기에 쓰는 전체 시간(초). BE 탈퇴 정리 호출 제한(15초) 안에 응답하도록.
_REVOKE_BUDGET_SEC = 8.0


@dataclass
class ReservationCredentialSummary:
    userId: str
    reservationEmail: str | None
    apiKeyPrefix: str
    createdAt: str
    updatedAt: str


class ReservationCredentialService:
    @staticmethod
    def _extract_prefix(api_key: str) -> str:
        raw = (api_key or "").strip()
        if not raw.startswith("sbk_") or "." not in raw:
            return ""
        prefix = raw.removeprefix("sbk_").split(".", 1)[0].strip()
        return prefix

    def upsert_credential(
        self,
        db: Session,
        *,
        user_id: str,
        reservation_email: str | None,
        api_key: str,
    ) -> ReservationCredentialSummary:
        prefix = self._extract_prefix(api_key)
        if not prefix:
            raise ValueError("유효한 reservation API key 형식이 아닙니다.")

        current = db.query(ReservationCredential).filter(ReservationCredential.user_id == user_id).first()
        if current is None:
            current = ReservationCredential(
                user_id=user_id,
                reservation_email=(reservation_email or "").strip() or None,
                api_key_prefix=prefix,
                api_key=api_key.strip(),
            )
            db.add(current)
        else:
            current.reservation_email = (reservation_email or "").strip() or None
            current.api_key_prefix = prefix
            current.api_key = api_key.strip()

        db.commit()
        db.refresh(current)
        return self._to_summary(current)

    @staticmethod
    def delete_credential(db: Session, user_id: str) -> int:
        """해당 회원의 예약 API 키 행을 지우고 지운 건수를 돌려준다(회원 탈퇴 정리용). 다른 회원 행은 건드리지 않는다."""
        deleted = db.query(ReservationCredential).filter(ReservationCredential.user_id == user_id).delete(
            synchronize_session=False
        )
        db.commit()
        return int(deleted)

    def purge_reservation_account(self, client: ReservationApiClient, user_id: str) -> dict[str, object]:
        """그 회원 예약 계정(sb-{userId})을 예약 서비스에서 지운다 - 예약(환자 이름·전화·증상)·API 키·회원 행(2026-10-30,
        예약 서비스 DELETE /users/me). 예약 서비스가 그 경로를 모르면(옛 버전) AI-{userId} 활성 키 폐기로 대신한다(#122).
        AI 쪽 행 삭제 뒤 best-effort로 부른다 - 8초 안(BE 호출 제한 15초). 가입하지 않는다.
        accountDeleted: True 지움 / False 계정 없음 / None 못 함·모름. revokedKeys: 없앤(지우거나 폐기한) 키 수, 못 하면 None.
        로그는 건수·상태 코드만(키·이메일·비밀번호·예약 내용 금지)."""
        out: dict[str, object] = {"accountDeleted": None, "revokedKeys": None}
        settings = client._settings
        if not settings.has_reservation_account_secret:
            return out
        email, password = self.account_for(user_id, settings.reservation_account_secret)
        try:
            with chat_deadline(_REVOKE_BUDGET_SEC):
                login = client.login_account(email, password)
                status = int(login.get("httpStatus") or 0)
                if status in {400, 401, 404}:
                    return {"accountDeleted": False, "revokedKeys": 0}  # 예약 계정이 없다(예약을 한 번도 안 했다)
                data = login.get("data")
                token = str(data.get("accessToken") or data.get("access_token") or "").strip() if isinstance(data, dict) else ""
                if not (200 <= status < 300 and token):
                    _LOG.info("[예약계정] 회원 데이터 삭제 - 로그인 실패 http_status=%s", status)
                    return out
                deleted = client.delete_my_account(token)
                d_status = int(deleted.get("httpStatus") or 0)
                d_data = deleted.get("data") if isinstance(deleted.get("data"), dict) else {}
                if 200 <= d_status < 300:
                    keys = d_data.get("apiKeys")
                    _LOG.info("[예약계정] 회원 데이터 삭제 - 계정 삭제(예약 %s건·키 %s건)", d_data.get("reservations"), keys)
                    return {"accountDeleted": True, "revokedKeys": keys if isinstance(keys, int) else None}
                _LOG.info("[예약계정] 회원 데이터 삭제 - 계정 삭제 실패 http_status=%s, 키 폐기로 대신", d_status)
                out["revokedKeys"] = self._revoke_ai_keys(client, token, f"AI-{user_id}")
                return out
        except Exception as exc:  # noqa: BLE001 - 마감(ChatDeadlineExceeded) 포함, AI 쪽 행 삭제는 이미 끝났다
            _LOG.info("[예약계정] 회원 데이터 삭제 건너뜀 %s", type(exc).__name__)
            return out

    @staticmethod
    def _revoke_ai_keys(client: ReservationApiClient, token: str, name: str) -> int | None:
        """name 이름의 활성 키를 모두 폐기한다. 전부 폐기했으면 건수, 목록·폐기 중 하나라도 실패하면 None."""
        listed = client.list_api_keys(token)
        rows = (listed.get("data") or {}).get("keys") if isinstance(listed.get("data"), dict) else None
        if not (200 <= int(listed.get("httpStatus") or 0) < 300) or not isinstance(rows, list):
            _LOG.info("[예약계정] 키 폐기용 목록 조회 실패 http_status=%s", listed.get("httpStatus"))
            return None
        targets = [r for r in rows if isinstance(r, dict) and r.get("name") == name and r.get("isActive") and isinstance(r.get("id"), int)]
        revoked = sum(1 for row in targets if 200 <= int(client.revoke_api_key(token, int(row["id"])).get("httpStatus") or 0) < 300)
        _LOG.info("[예약계정] 회원 데이터 삭제 - 키 폐기 %s건(대상 %s건)", revoked, len(targets))
        return revoked if revoked == len(targets) else None

    def get_api_key(self, db: Session, user_id: str) -> str | None:
        current = db.query(ReservationCredential).filter(ReservationCredential.user_id == user_id).first()
        if not current:
            return None
        return current.api_key.strip() or None

    def has_credential(self, db: Session, user_id: str) -> bool:
        return self.get_api_key(db, user_id) is not None

    PHONE_PATTERN = re.compile(r"^01[0-9]-?[0-9]{3,4}-?[0-9]{4}$")

    @staticmethod
    def account_for(user_id: str, secret: str) -> tuple[str, str]:
        """프론트 BFF와 같은 규칙의 사용자별 예약 계정. 이메일 sb-<userId>@silverbridge.local, 비밀번호 HMAC(secret, userId)."""
        email = f"sb-{user_id}@silverbridge.local"
        password = hmac.new(secret.encode("utf-8"), user_id.encode("utf-8"), hashlib.sha256).hexdigest()
        return email, password

    def ensure_credential(
        self,
        db: Session,
        *,
        user_id: str,
        client: ReservationApiClient,
        name: str | None = None,
        phone: str | None = None,
    ) -> str | None:
        """user_id(예약 당사자, 보호자가 대신하면 피보호자) 계정의 API 키. 없으면 로그인/가입해서 발급한다.

        예전엔 공용 계정(RESERVATION_SERVICE_EMAIL) 하나로 모든 예약을 넣어 웹(사용자별 계정)에서 보이지 않았다.
        같은 계정 규칙을 쓰면 챗봇·보호자 웹·피보호자 웹이 같은 예약 목록을 본다."""
        if not client._settings.has_reservation_account_secret:
            # 비밀값이 없으면 계정 비밀번호를 만들 수 없다 - 예약 기능만 끈다(다른 기능은 영향 없음). 값은 로그에 남기지 않는다.
            _LOG.warning("[예약계정] RESERVATION_ACCOUNT_SECRET 미설정 - 예약 계정 발급을 건너뜁니다")
            return None
        email, password = self.account_for(user_id, client._settings.reservation_account_secret)
        current = db.query(ReservationCredential).filter(ReservationCredential.user_id == user_id).first()
        # 저장된 키가 다른 계정(공용 계정) 것이면 버리고 다시 발급한다.
        if current and current.reservation_email == email and client.has_valid_reservation_key(current.api_key):
            return current.api_key.strip()

        login_result = client.login_account(email, password)
        login_status = int(login_result.get("httpStatus") or 0)
        if login_status in {400, 401, 404}:
            safe_phone = (phone or "").strip()
            if not self.PHONE_PATTERN.match(safe_phone):
                safe_phone = "010-0000-0000"
            client.signup_account(email, password, (name or "").strip() or "사용자", safe_phone)
            login_result = client.login_account(email, password)
            login_status = int(login_result.get("httpStatus") or 0)
        login_data = login_result.get("data")
        if not (200 <= login_status < 300 and isinstance(login_data, dict)):
            return None

        access_token = str(login_data.get("accessToken") or login_data.get("access_token") or "").strip()
        if not access_token:
            return None

        issued = client.issue_api_key(access_token, f"AI-{user_id}")
        issued_data = issued.get("data")
        if not (200 <= int(issued.get("httpStatus") or 0) < 300 and isinstance(issued_data, dict)):
            return None
        raw_api_key = str(issued_data.get("apiKey") or issued_data.get("api_key") or "").strip()
        if not raw_api_key:
            return None

        if current is None:
            current = ReservationCredential(
                user_id=user_id,
                reservation_email=email,
                api_key_prefix=self._extract_prefix(raw_api_key),
                api_key=raw_api_key,
            )
            db.add(current)
        else:
            current.reservation_email = email
            current.api_key_prefix = self._extract_prefix(raw_api_key)
            current.api_key = raw_api_key
        db.commit()
        db.refresh(current)
        self._revoke_stale_keys(client, access_token, f"AI-{user_id}", current.api_key_prefix)
        return current.api_key.strip()

    @staticmethod
    def _revoke_stale_keys(client: ReservationApiClient, access_token: str, name: str, keep_prefix: str) -> None:
        """새 키를 저장한 뒤 같은 이름의 옛 키를 폐기한다(2026-10-29 #107 점검 L-2) - 발급이 시간 초과로 끝났는데 실제로는
        만들어졌거나, 저장된 키가 무효라 다시 발급한 경우 쓰지 않는 키가 남지 않게. 다른 이름(웹 등)의 키는 건드리지 않는다.
        best-effort - 실패해도 새 키로 상담을 이어 간다. 로그에는 개수·상태 코드만(키·prefix 금지)."""
        if not keep_prefix:
            return
        try:
            listed = client.list_api_keys(access_token)
            rows = (listed.get("data") or {}).get("keys") if isinstance(listed.get("data"), dict) else None
            if not (200 <= int(listed.get("httpStatus") or 0) < 300) or not isinstance(rows, list):
                _LOG.info("[예약계정] 옛 키 목록 조회 실패 http_status=%s", listed.get("httpStatus"))
                return
            stale = [
                row for row in rows
                if isinstance(row, dict) and row.get("name") == name and row.get("isActive")
                and row.get("keyPrefix") != keep_prefix and isinstance(row.get("id"), int)
            ]
            failed = 0
            for row in stale:
                res = client.revoke_api_key(access_token, int(row["id"]))
                if not (200 <= int(res.get("httpStatus") or 0) < 300):
                    failed += 1
            if stale:
                _LOG.info("[예약계정] 옛 키 폐기 %s건(실패 %s건)", len(stale) - failed, failed)
        except Exception as exc:  # noqa: BLE001 - 마감(ChatDeadlineExceeded) 포함, 새 키 저장은 이미 끝났다
            _LOG.info("[예약계정] 옛 키 폐기 건너뜀 %s", type(exc).__name__)

    @staticmethod
    def _to_summary(item: ReservationCredential) -> ReservationCredentialSummary:
        return ReservationCredentialSummary(
            userId=item.user_id,
            reservationEmail=item.reservation_email,
            apiKeyPrefix=item.api_key_prefix,
            createdAt=item.created_at.isoformat() if item.created_at else "",
            updatedAt=item.updated_at.isoformat() if item.updated_at else "",
        )


_SERVICE = ReservationCredentialService()


def get_reservation_credential_service() -> ReservationCredentialService:
    return _SERVICE
