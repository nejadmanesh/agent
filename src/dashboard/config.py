"""Configuration helpers for the Streamlit dashboard."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_API_BASE_URL = "http://localhost:8000"
DEFAULT_LOGIN_ENDPOINT = "/auth/login"
DEFAULT_METRICS_ENDPOINT = "/metrics"
DEFAULT_FONT_FAMILY = "Vazirmatn, Vazir, IRANSans, Tahoma, Arial, sans-serif"
DEFAULT_CHART_TITLE = "روند شاخص‌های کلیدی"
DEFAULT_PAGE_TITLE = "داشبورد پایش سرویس"
DEFAULT_JWT_ALGORITHMS: tuple[str, ...] = ("HS256",)


@dataclass(slots=True)
class DashboardConfig:
    """Runtime configuration for the dashboard application."""

    api_base_url: str = DEFAULT_API_BASE_URL
    login_endpoint: str = DEFAULT_LOGIN_ENDPOINT
    metrics_endpoint: str = DEFAULT_METRICS_ENDPOINT
    jwt_secret: str | None = None
    font_family: str = DEFAULT_FONT_FAMILY
    page_title: str = DEFAULT_PAGE_TITLE
    chart_title: str = DEFAULT_CHART_TITLE
    jwt_algorithms: tuple[str, ...] = DEFAULT_JWT_ALGORITHMS

    @classmethod
    def from_environment(cls) -> "DashboardConfig":
        """Build a configuration instance from environment variables."""

        algorithms = os.getenv("DASHBOARD_JWT_ALGORITHMS")
        if algorithms:
            parsed_algorithms = tuple(
                alg.strip() for alg in algorithms.split(",") if alg.strip()
            )
            jwt_algorithms = parsed_algorithms or DEFAULT_JWT_ALGORITHMS
        else:
            jwt_algorithms = DEFAULT_JWT_ALGORITHMS

        return cls(
            api_base_url=os.getenv("DASHBOARD_API_BASE_URL", DEFAULT_API_BASE_URL),
            login_endpoint=os.getenv("DASHBOARD_LOGIN_ENDPOINT", DEFAULT_LOGIN_ENDPOINT),
            metrics_endpoint=os.getenv("DASHBOARD_METRICS_ENDPOINT", DEFAULT_METRICS_ENDPOINT),
            jwt_secret=os.getenv("DASHBOARD_JWT_SECRET"),
            font_family=os.getenv("DASHBOARD_FONT_FAMILY", DEFAULT_FONT_FAMILY),
            page_title=os.getenv("DASHBOARD_PAGE_TITLE", DEFAULT_PAGE_TITLE),
            chart_title=os.getenv("DASHBOARD_CHART_TITLE", DEFAULT_CHART_TITLE),
            jwt_algorithms=jwt_algorithms,
        )


__all__ = ["DashboardConfig"]
