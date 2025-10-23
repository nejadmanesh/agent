"""Client helpers for communicating with the backend API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from urllib.parse import urljoin

import requests

from .config import DashboardConfig


class APIError(RuntimeError):
    """Raised when the dashboard fails to communicate with the API."""


@dataclass(slots=True)
class AuthResult:
    """Container for authentication results returned by the backend API."""

    access_token: str
    refresh_token: str | None = None
    raw_payload: Mapping[str, Any] | None = None


class DashboardAPI:
    """Lightweight API client used by the Streamlit dashboard."""

    def __init__(self, config: DashboardConfig) -> None:
        self._config = config
        self._session = requests.Session()

    def login(self, username: str, password: str) -> AuthResult:
        """Authenticate the user and return a JWT access token."""

        payload = {"username": username, "password": password}
        url = self._build_url(self._config.login_endpoint)
        try:
            response = self._session.post(url, json=payload, timeout=10)
        except requests.RequestException as exc:  # pragma: no cover - network failure
            raise APIError("خطا در اتصال به سرویس احراز هویت.") from exc

        if response.status_code >= 400:
            raise APIError("اطلاعات ورود نامعتبر است یا سرویس در دسترس نیست.")

        data: Mapping[str, Any] = response.json()
        token = self._extract_token(data)
        return AuthResult(
            access_token=token,
            refresh_token=self._extract_refresh_token(data),
            raw_payload=data,
        )

    def fetch_metrics(self, token: str) -> Sequence[Mapping[str, Any]]:
        """Fetch dashboard metrics from the backend API."""

        url = self._build_url(self._config.metrics_endpoint)
        headers = {"Authorization": f"Bearer {token}"}

        try:
            response = self._session.get(url, headers=headers, timeout=10)
        except requests.RequestException as exc:  # pragma: no cover - network failure
            raise APIError("خطا در دریافت داده‌ها از API.") from exc

        if response.status_code >= 400:
            raise APIError("عدم دسترسی به داده‌ها. لطفاً دوباره وارد شوید.")

        data = response.json()
        if isinstance(data, Mapping) and "items" in data:
            items = data["items"]
        else:
            items = data

        if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
            raise APIError("ساختار پاسخ API نامعتبر است.")

        return items

    def _build_url(self, endpoint: str) -> str:
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            return endpoint
        return urljoin(self._config.api_base_url.rstrip("/") + "/", endpoint.lstrip("/"))

    @staticmethod
    def _extract_token(payload: Mapping[str, Any]) -> str:
        for key in ("access_token", "token", "jwt"):
            token = payload.get(key)
            if isinstance(token, str) and token:
                return token
        raise APIError("توکن دسترسی در پاسخ API یافت نشد.")

    @staticmethod
    def _extract_refresh_token(payload: Mapping[str, Any]) -> str | None:
        value = payload.get("refresh_token")
        return value if isinstance(value, str) else None


__all__ = ["APIError", "AuthResult", "DashboardAPI"]
