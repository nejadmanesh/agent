"""Streamlit application entry point for the analytics dashboard."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import jwt
import streamlit as st
from jwt import InvalidTokenError, PyJWTError
from plotly import graph_objects as go

from .api import APIError, DashboardAPI
from .config import DashboardConfig


def run() -> None:
    """Render the dashboard UI."""

    config = DashboardConfig.from_environment()
    st.set_page_config(page_title=config.page_title, page_icon="📈", layout="wide")
    st.title(config.page_title)
    st.caption("اتصال به API و نمایش شاخص‌های کلیدی با پشتیبانی از فونت فارسی.")

    _init_session_state()
    api_client = DashboardAPI(config)

    with st.sidebar:
        st.header("حساب کاربری")
        if st.session_state.auth_token:
            _render_user_details()
            if st.button("خروج", use_container_width=True):
                _reset_auth_state()
                st.experimental_rerun()
        else:
            _render_login_form(api_client, config)

    token: str | None = st.session_state.auth_token
    if not token:
        st.info("برای مشاهده داشبورد، ابتدا وارد حساب خود شوید.")
        return

    metrics = _load_metrics(api_client, token)
    if metrics:
        _render_chart(metrics, config)
        st.subheader("جدول داده‌ها")
        st.dataframe(metrics, use_container_width=True)
    else:
        st.warning("داده‌ای برای نمایش یافت نشد.")


def _init_session_state() -> None:
    st.session_state.setdefault("auth_token", None)
    st.session_state.setdefault("auth_payload", None)
    st.session_state.setdefault("auth_claims", None)


def _render_login_form(api_client: DashboardAPI, config: DashboardConfig) -> None:
    st.write("برای ورود، نام کاربری و گذرواژه خود را وارد کنید.")
    with st.form("login-form"):
        username = st.text_input("نام کاربری")
        password = st.text_input("گذرواژه", type="password")
        submitted = st.form_submit_button("ورود")

    if not submitted:
        return

    if not username or not password:
        st.error("نام کاربری و گذرواژه نمی‌تواند خالی باشد.")
        return

    with st.spinner("در حال ورود..."):
        try:
            auth_result = api_client.login(username=username, password=password)
        except APIError as exc:
            st.error(str(exc))
            return

    st.session_state.auth_token = auth_result.access_token
    st.session_state.auth_payload = auth_result.raw_payload
    st.session_state.auth_claims = _decode_jwt(auth_result.access_token, config.jwt_secret)
    st.success("ورود با موفقیت انجام شد.")
    st.experimental_rerun()


def _render_user_details() -> None:
    claims: Mapping[str, Any] | None = st.session_state.auth_claims
    payload: Mapping[str, Any] | None = st.session_state.auth_payload

    username = None
    if claims:
        username = claims.get("preferred_username") or claims.get("username") or claims.get("sub")
    if not username and payload and isinstance(payload, Mapping):
        username = payload.get("username") or payload.get("user")

    if username:
        st.success(f"کاربر فعال: {username}")
    else:
        st.success("کاربر وارد شده است.")

    if claims:
        with st.expander("جزئیات توکن"):
            st.json(claims)


def _load_metrics(api_client: DashboardAPI, token: str) -> list[Mapping[str, Any]]:
    with st.spinner("در حال دریافت داده‌ها..."):
        try:
            raw_metrics = api_client.fetch_metrics(token)
        except APIError as exc:
            st.error(str(exc))
            return []

    metrics: list[Mapping[str, Any]] = []
    for item in raw_metrics:
        if isinstance(item, Mapping):
            metrics.append(item)
    return metrics


def _render_chart(metrics: Sequence[Mapping[str, Any]], config: DashboardConfig) -> None:
    st.subheader("نمودار تحلیلی")
    figure = _build_chart(metrics, config)
    if figure is None:
        st.warning("داده‌های دریافتی برای رسم نمودار مناسب نیست.")
        return
    st.plotly_chart(figure, use_container_width=True)


def _build_chart(metrics: Sequence[Mapping[str, Any]], config: DashboardConfig) -> go.Figure | None:
    if not metrics:
        return None

    sample = metrics[0]
    label_field = _discover_label_field(sample)
    numeric_fields = [
        key
        for key, value in sample.items()
        if isinstance(value, (int, float))
    ]

    if not numeric_fields:
        return None

    x_values = [
        str(item.get(label_field, idx + 1)) if label_field else idx + 1
        for idx, item in enumerate(metrics)
    ]

    figure = go.Figure()
    for field in numeric_fields:
        y_values = []
        for item in metrics:
            value = item.get(field)
            if isinstance(value, (int, float)):
                y_values.append(value)
        if len(y_values) != len(metrics):
            continue
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines+markers",
                name=field,
            )
        )

    if not figure.data:
        return None

    figure.update_layout(
        title=config.chart_title,
        xaxis_title=label_field or "ردیف",
        yaxis_title="مقدار",
        template="plotly_white",
        font={"family": config.font_family},
    )
    return figure


def _discover_label_field(sample: Mapping[str, Any]) -> str | None:
    for candidate in ("date", "timestamp", "label", "name", "title"):
        if candidate in sample:
            return candidate
    return None


def _decode_jwt(token: str, secret: str | None) -> Mapping[str, Any] | None:
    if not token:
        return None

    algorithm = "HS256"
    try:
        header = jwt.get_unverified_header(token)
        if isinstance(header, Mapping):
            alg_value = header.get("alg")
            if isinstance(alg_value, str) and alg_value:
                algorithm = alg_value
    except (PyJWTError, ValueError, TypeError):
        pass

    try:
        if secret:
            return jwt.decode(
                token,
                secret,
                algorithms=[algorithm],
                options={"verify_aud": False},
            )
        return jwt.decode(
            token,
            options={"verify_signature": False, "verify_aud": False},
            algorithms=[algorithm],
        )
    except InvalidTokenError:
        return None


def _reset_auth_state() -> None:
    st.session_state.auth_token = None
    st.session_state.auth_payload = None
    st.session_state.auth_claims = None


if __name__ == "__main__":
    run()
