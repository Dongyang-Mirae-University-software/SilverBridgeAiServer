from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.services.reservation_api_client import ReservationApiClient
from app.models.reservation_credential import ReservationCredential


@dataclass
class ReservationCredentialSummary:
    userId: int
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
        user_id: int,
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

    def get_api_key(self, db: Session, user_id: int) -> str | None:
        current = db.query(ReservationCredential).filter(ReservationCredential.user_id == user_id).first()
        if not current:
            return None
        return current.api_key.strip() or None

    def has_credential(self, db: Session, user_id: int) -> bool:
        return self.get_api_key(db, user_id) is not None

    def ensure_credential(
        self,
        db: Session,
        *,
        user_id: int,
        reservation_email: str | None,
        client: ReservationApiClient,
    ) -> str | None:
        current = db.query(ReservationCredential).filter(ReservationCredential.user_id == user_id).first()
        if current and client.has_valid_reservation_key(current.api_key):
            return current.api_key.strip()

        service_email = (
            (current.reservation_email if current and current.reservation_email else "").strip()
            or client._settings.reservation_service_email.strip()
        )
        service_password = client._settings.reservation_service_password
        if not service_email or not service_password:
            return current.api_key.strip() if current and current.api_key.strip() else None

        login_result = client.login_account(service_email, service_password)
        login_status = int(login_result.get("httpStatus") or 0)
        login_data = login_result.get("data")
        if not (200 <= login_status < 300 and isinstance(login_data, dict)):
            return current.api_key.strip() if current and current.api_key.strip() else None

        access_token = str(login_data.get("accessToken") or login_data.get("access_token") or "").strip()
        if not access_token:
            return current.api_key.strip() if current and current.api_key.strip() else None

        issued = client.issue_api_key(access_token, f"AI-{user_id}")
        issued_status = int(issued.get("httpStatus") or 0)
        issued_data = issued.get("data")
        if not (200 <= issued_status < 300 and isinstance(issued_data, dict)):
            return current.api_key.strip() if current and current.api_key.strip() else None

        raw_api_key = str(issued_data.get("apiKey") or issued_data.get("api_key") or "").strip()
        if not raw_api_key:
            return current.api_key.strip() if current and current.api_key.strip() else None

        if current is None:
            current = ReservationCredential(
                user_id=user_id,
                reservation_email=service_email or None,
                api_key_prefix=self._extract_prefix(raw_api_key),
                api_key=raw_api_key,
            )
            db.add(current)
        else:
            current.reservation_email = service_email or None
            current.api_key_prefix = self._extract_prefix(raw_api_key)
            current.api_key = raw_api_key

        db.commit()
        db.refresh(current)
        return current.api_key.strip()

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
