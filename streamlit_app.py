from __future__ import annotations
from app.location_query import strip_relative_location
from app.chat_session import sync_chat_location
from app.agriculture_news import NEWS_INTENT

from app.location_controls import render_place_selector
from app.location_selection import location_context, place_options, qualified_place, scope_market_rows, market_scope_caption

import os
import re
import sys
import sqlite3
import smtplib
import subprocess
import traceback
import html
import uuid
import hmac
import hashlib
import base64
from datetime import date, datetime
from time import time
from pathlib import Path
from email.message import EmailMessage
from functools import lru_cache
from textwrap import dedent
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np
import streamlit as st
import json
from urllib.parse import urlencode
from urllib.request import urlopen
import difflib

from app.advisor import AdvisorConfig, RAGAdvisor, WESTERN_UP_CROP_BASELINES
from app.config import load_config
from app.crop_guide import build_crop_production_followup
import app.db as db_mod
from app.datagov_client import DataGovClient
from app.feedback import compact_evidence_text, validate_feedback_with_local_sources
from app.lstm_forecast import prepare_daily_series, quick_forecast, train_and_forecast
from app.weather import get_current_weather_hindi
from app.cacp import get_latest_sugarcane_frp
from app.msp import get_msp_for_crop
from app.supabase_auth import (
    SupabaseConfig,
    get_user as supabase_get_user,
    is_configured as supabase_is_configured,
    refresh_session as supabase_refresh_session,
    resend_signup_email as supabase_resend_signup_email,
    send_password_reset_email as supabase_send_password_reset_email,
    sign_in_with_password as supabase_sign_in_with_password,
    sign_out as supabase_sign_out,
    sign_up as supabase_sign_up,
)


authenticate_user = db_mod.authenticate_user
authenticate_user_status = db_mod.authenticate_user_status
create_password_reset_otp = db_mod.create_password_reset_otp
create_query_log = db_mod.create_query_log
create_user = db_mod.create_user
export_training_feedback = db_mod.export_training_feedback
feedback_exists = db_mod.feedback_exists
get_conn = db_mod.get_conn
get_feedback_queue = db_mod.get_feedback_queue
get_training_feedback_examples = db_mod.get_training_feedback_examples
get_user_by_email = db_mod.get_user_by_email
get_user_by_id = db_mod.get_user_by_id
init_db = db_mod.init_db
reset_password_with_otp = db_mod.reset_password_with_otp
review_feedback = db_mod.review_feedback
save_feedback = db_mod.save_feedback
set_verification_token_for_email = db_mod.set_verification_token_for_email
upsert_external_user = db_mod.upsert_external_user
verify_user_by_token = db_mod.verify_user_by_token


BRAND_IMAGE = Path(
    "data/raw/indian-agriculture-landscape-farmer-working-indian-rice-fields-rural-worker-vector-cartoon-backg_1396-599.avif"
)
PAGE_ICON = str(BRAND_IMAGE) if BRAND_IMAGE.exists() else "🌾"
st.set_page_config(
    page_title="KisaanAI - Agriculture Assistant",
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)
# Public news must render before readiness and authentication can stop the page.
from app.news_panel import render_news_panel
with st.sidebar:
    render_news_panel()

if BRAND_IMAGE.exists():
    st.image(str(BRAND_IMAGE), use_container_width=True)

st.title("KisaanAI - Agriculture Assistant")
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 2rem;
        max-width: 96rem;
    }
    .kisaan-hero {
        background:
            radial-gradient(circle at top right, rgba(124, 169, 91, 0.20), transparent 28%),
            linear-gradient(135deg, rgba(247, 250, 240, 0.98), rgba(234, 244, 223, 0.94));
        border: 1px solid rgba(94, 129, 63, 0.18);
        border-radius: 24px;
        padding: 1rem 1.1rem 0.9rem 1.1rem;
        margin: 0.2rem 0 1rem 0;
        box-shadow: 0 14px 30px rgba(51, 77, 32, 0.08);
    }
    .kisaan-hero-eyebrow {
        font-size: 0.78rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #5f7d3a;
        margin-bottom: 0.35rem;
        font-weight: 700;
    }
    .kisaan-hero-title {
        font-size: 1.5rem;
        line-height: 1.15;
        color: #234018;
        font-weight: 700;
        margin: 0;
    }
    .kisaan-hero-copy {
        color: #456233;
        margin: 0.45rem 0 0 0;
        font-size: 0.98rem;
    }
    .kisaan-toolbar-note {
        color: #557145;
        font-size: 0.9rem;
        margin: 0.45rem 0 0.15rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

cfg = load_config()
APP_BUILD_VERSION = "2026-06-21-white-grub-source-verify-v1"
LIVE_MARKET_CSV = Path("data/raw/live/datagov_commodity.csv")
AGMARKNET_CSV = Path("data/raw/live/agmarknet_report.csv")
AGMARKNET_CATALOG_CSV = Path("data/processed/agmarknet_catalog.csv")
AGMARKNET_AUTO_REFRESH_META = Path("data/raw/live/agmarknet_auto_refresh.json")
AGMARKNET_AUTO_REFRESH_LOG = Path("logs/agmarknet_auto_refresh.log")
AGMARKNET_AUTO_REFRESH_RETRY_MINUTES = 30
AGMARKNET_REFRESH_STATUS = Path("data/raw/live/agmarknet_refresh_status.json")
FETCH_PAGE_LIMIT = 200
FETCH_MAX_RECORDS_COMBO = 50000
FETCH_MAX_RECORDS_STATE = 50000
FETCH_COOLDOWN_SEC = 600
FAST_FETCH_LIMIT = 200
TRAINING_FEEDBACK_PATH = Path("data/processed/accepted_feedback.jsonl")
AUTH_COOKIE_NAME = "krishiai_auth"
MEDIUM_GENERATOR_MODEL = os.getenv("KISAANAI_MEDIUM_GENERATOR_MODEL") or cfg.generator_model
COMPLEX_GENERATOR_MODEL = os.getenv("KISAANAI_COMPLEX_GENERATOR_MODEL") or None


def _safe_text(value: object, fallback: str = "") -> str:
    if pd.isna(value):
        return fallback
    txt = str(value).strip()
    if not txt or txt.lower() == "nan":
        return fallback
    return txt


def _safe_float(value: object) -> float | None:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def _extract_season_from_query_text(text: str, advisor: RAGAdvisor | None = None) -> str | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    normalized = advisor._normalize_hinglish(raw) if advisor is not None else raw.lower()
    t = f"{raw.lower()} | {str(normalized).lower()}"
    season_patterns = [
        ("Rabi", [r"\brabi\b", r"रबी"]),
        ("Kharif", [r"\bkharif\b", r"खरीफ"]),
        ("Zaid", [r"\bzaid\b", r"जायद"]),
        ("Annual", [r"\bannual\b", r"सालाना", r"वार्षिक"]),
    ]
    for label, patterns in season_patterns:
        if any(re.search(pattern, t, flags=re.IGNORECASE) for pattern in patterns):
            return label
    return None


def _latest_agmarknet_report_date(path: Path) -> date | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, usecols=["rep_date"])
    except Exception:
        return None
    if "rep_date" not in df.columns or df.empty:
        return None
    dt = pd.to_datetime(df["rep_date"], errors="coerce", dayfirst=True)
    if dt.dropna().empty:
        return None
    try:
        return dt.max().date()
    except Exception:
        return None


def _load_agmarknet_auto_refresh_meta() -> dict[str, object]:
    if not AGMARKNET_AUTO_REFRESH_META.exists():
        return {}
    try:
        return json.loads(AGMARKNET_AUTO_REFRESH_META.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_agmarknet_refresh_status() -> dict[str, object]:
    if not AGMARKNET_REFRESH_STATUS.exists():
        return {}
    try:
        return json.loads(AGMARKNET_REFRESH_STATUS.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_agmarknet_auto_refresh_meta(payload: dict[str, object]) -> None:
    AGMARKNET_AUTO_REFRESH_META.parent.mkdir(parents=True, exist_ok=True)
    AGMARKNET_AUTO_REFRESH_META.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _pid_is_running(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except Exception:
        return False
    return True


def _is_streamlit_cloud_runtime() -> bool:
    sharing_mode = os.getenv("STREAMLIT_SHARING_MODE", "").strip().lower()
    cwd = str(Path.cwd())
    return bool(sharing_mode) or cwd.startswith("/mount/src/")


def _should_start_agmarknet_auto_refresh() -> tuple[bool, str]:
    if os.getenv("AGMARKNET_AUTO_REFRESH", "1").strip().lower() in {"0", "false", "no"}:
        return False, "disabled"
    if _is_streamlit_cloud_runtime() and os.getenv("AGMARKNET_AUTO_REFRESH_ON_CLOUD", "0").strip().lower() not in {"1", "true", "yes"}:
        return False, "disabled_on_streamlit_cloud"
    meta = _load_agmarknet_auto_refresh_meta()
    last_started_raw = str(meta.get("last_started_at") or "").strip()
    last_pid = int(meta.get("pid") or 0) if str(meta.get("pid") or "").strip().isdigit() else 0
    if last_pid and _pid_is_running(last_pid):
        return False, "already_running"
    today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
    latest_report_date = _latest_agmarknet_report_date(AGMARKNET_CSV)
    if latest_report_date is not None and latest_report_date >= today:
        return False, "already_latest"
    if last_started_raw:
        try:
            last_started = datetime.fromisoformat(last_started_raw)
            last_started_ist = last_started.astimezone(ZoneInfo("Asia/Kolkata"))
            elapsed_min = (datetime.now(ZoneInfo("Asia/Kolkata")) - last_started_ist).total_seconds() / 60.0
            if elapsed_min < AGMARKNET_AUTO_REFRESH_RETRY_MINUTES:
                return False, "recent_attempt"
        except Exception:
            pass
    return True, "stale_or_missing"


def _start_agmarknet_auto_refresh() -> tuple[bool, str]:
    should_start, reason = _should_start_agmarknet_auto_refresh()
    if not should_start:
        return False, reason
    cmd = [
        sys.executable,
        "-u",
        str(Path("scripts/agmarknet_daily_refresh.py")),
    ]
    env = os.environ.copy()
    env.setdefault("AGMARKNET_LOOKBACK_DAYS", "14")
    env.setdefault("AGMARKNET_MODE", "auto")
    AGMARKNET_AUTO_REFRESH_LOG.parent.mkdir(parents=True, exist_ok=True)
    log_handle = AGMARKNET_AUTO_REFRESH_LOG.open("ab")
    proc = subprocess.Popen(
        cmd,
        cwd=str(Path(".")),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    _save_agmarknet_auto_refresh_meta(
        {
            "last_started_at": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
            "pid": proc.pid,
            "mode": env.get("AGMARKNET_MODE", "auto"),
            "lookback_days": env.get("AGMARKNET_LOOKBACK_DAYS", "14"),
            "log_file": str(AGMARKNET_AUTO_REFRESH_LOG),
        }
    )
    return True, "started"


def _extract_primary_weather_condition(text: str) -> str:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    for line in lines:
        if line.startswith("- स्थिति:"):
            return line.split(":", 1)[1].strip()
    for line in lines:
        if line.startswith("- ") and "):" in line:
            after = line.split("):", 1)[1].strip()
            return after.split(",", 1)[0].strip()
    return str(text or "")


def _weather_theme_from_text(text: str) -> str:
    condition = _extract_primary_weather_condition(text).lower()
    if any(token in condition for token in ["बारिश", "फुहार", "वर्षा", "rain", "showers", "तूफान", "आंधी"]):
        return "rain"
    if any(token in condition for token in ["बादल", "कोहरा", "cloud", "fog", "धुंध"]):
        return "cloud"
    if any(token in condition for token in ["आसमान साफ", "मुख्यतः साफ", "sunny", "clear", "धूप"]):
        return "sun"
    return "cloud"


def _weather_card_label(action: str | None) -> str:
    action_key = str(action or "").strip().lower()
    if action_key in {"rain_day", "daily_rain"}:
        return "Rain Forecast"
    if action_key == "weekly":
        return "Weekly Weather"
    if action_key == "daily":
        return "Weather Forecast"
    return "Weather Update"


def _is_night_in_india() -> bool:
    hour = datetime.now(ZoneInfo("Asia/Kolkata")).hour
    return hour >= 18 or hour < 6


def _looks_like_crop_water_followup(query: str, advisor: RAGAdvisor) -> bool:
    normalized = advisor._normalize_hinglish(query)
    crop = advisor._extract_crop_from_query(normalized)
    if not crop:
        return False
    water_terms = [
        "pani",
        "paani",
        "पानी",
        "sinchai",
        "sichai",
        "sinchaai",
        "irrigation",
        "water",
        "lagega",
        "lagta",
        "kitna",
        "kitta",
        "कितना",
    ]
    return any(term in normalized for term in water_terms)


def render_weather_chat_card(text: str, action: str | None = None) -> None:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        st.write(text)
        return
    title = html.escape(lines[0])
    body = "<br>".join(html.escape(line) for line in lines[1:]) if len(lines) > 1 else ""
    theme = _weather_theme_from_text(text)
    night = _is_night_in_india()
    themes = {
        "rain": {
            "bg": "repeating-linear-gradient(-65deg, rgba(255,255,255,0.0) 0px, rgba(255,255,255,0.0) 12px, rgba(220,241,255,0.14) 12px, rgba(220,241,255,0.14) 14px, rgba(255,255,255,0.0) 14px, rgba(255,255,255,0.0) 24px), linear-gradient(135deg, #0f3554 0%, #1f5c85 55%, #4f8fb7 100%)",
            "bg_size": "160px 160px, auto",
            "border": "#8dc7ec",
            "label": _weather_card_label(action),
            "card_animation": "weatherDayRainBg 5s linear infinite",
        },
        "cloud": {
            "bg": "radial-gradient(ellipse at 16% 28%, rgba(255,255,255,0.12) 0 24px, transparent 26px), radial-gradient(ellipse at 38% 18%, rgba(255,255,255,0.11) 0 32px, transparent 34px), radial-gradient(ellipse at 62% 30%, rgba(255,255,255,0.12) 0 28px, transparent 30px), linear-gradient(135deg, #435365 0%, #66798a 60%, #97aab8 100%)",
            "bg_size": "220px 100px, 260px 120px, 240px 100px, auto",
            "border": "#d7e2ea",
            "label": _weather_card_label(action),
            "card_animation": "weatherDayCloudBg 16s ease-in-out infinite",
        },
        "sun": {
            "bg": "radial-gradient(circle at 88% 18%, rgba(255,244,190,0.65) 0 42px, rgba(255,244,190,0.0) 44px), radial-gradient(circle at 15% 28%, rgba(255,255,255,0.18) 0 2px, transparent 3px), radial-gradient(circle at 36% 18%, rgba(255,255,255,0.16) 0 2px, transparent 3px), radial-gradient(circle at 62% 34%, rgba(255,255,255,0.16) 0 2px, transparent 3px), linear-gradient(135deg, #7f4a00 0%, #c87a00 55%, #f6c54f 100%)",
            "bg_size": "auto, auto, auto, auto, auto",
            "border": "#ffe7a8",
            "label": _weather_card_label(action),
            "card_animation": "weatherDaySunBg 10s ease-in-out infinite",
        },
        "night_rain": {
            "bg": "repeating-linear-gradient(-65deg, rgba(255,255,255,0.0) 0px, rgba(255,255,255,0.0) 12px, rgba(219,236,255,0.18) 12px, rgba(219,236,255,0.18) 14px, rgba(255,255,255,0.0) 14px, rgba(255,255,255,0.0) 24px), radial-gradient(circle at 86% 18%, rgba(255,255,255,0.96) 0 18px, transparent 19px), radial-gradient(circle at 89% 16%, #0b2345 0 16px, transparent 17px), radial-gradient(circle at 14% 28%, rgba(255,255,255,0.9) 0 1.2px, transparent 1.8px), radial-gradient(circle at 24% 18%, rgba(255,255,255,0.88) 0 1.1px, transparent 1.7px), radial-gradient(circle at 38% 34%, rgba(255,255,255,0.92) 0 1.2px, transparent 1.8px), radial-gradient(circle at 56% 22%, rgba(255,255,255,0.88) 0 1.1px, transparent 1.7px), linear-gradient(135deg, #07162d 0%, #0d2950 55%, #193d69 100%)",
            "bg_size": "160px 160px, auto, auto, auto, auto, auto, auto, auto",
            "border": "#4e6f96",
            "label": _weather_card_label(action),
            "card_animation": "weatherNightRainBg 2.2s linear infinite",
        },
        "night_cloud": {
            "bg": "radial-gradient(circle at 86% 18%, rgba(255,255,255,0.96) 0 18px, transparent 19px), radial-gradient(circle at 89% 16%, #0b2345 0 16px, transparent 17px), radial-gradient(circle at 18% 24%, rgba(255,255,255,0.92) 0 1.2px, transparent 1.8px), radial-gradient(circle at 36% 18%, rgba(255,255,255,0.88) 0 1.1px, transparent 1.7px), radial-gradient(circle at 52% 32%, rgba(255,255,255,0.9) 0 1.2px, transparent 1.8px), radial-gradient(circle at 70% 22%, rgba(255,255,255,0.88) 0 1.1px, transparent 1.7px), radial-gradient(ellipse at 20% 48%, rgba(255,255,255,0.12) 0 26px, transparent 28px), radial-gradient(ellipse at 38% 58%, rgba(255,255,255,0.10) 0 34px, transparent 36px), radial-gradient(ellipse at 64% 44%, rgba(255,255,255,0.11) 0 30px, transparent 32px), linear-gradient(135deg, #08172f 0%, #173253 55%, #284a73 100%)",
            "bg_size": "auto, auto, auto, auto, auto, auto, 220px 100px, 260px 120px, 240px 100px, auto",
            "border": "#6d87a8",
            "label": _weather_card_label(action),
            "card_animation": "weatherNightCloudBg 18s ease-in-out infinite",
        },
        "night_sun": {
            "bg": "radial-gradient(circle at 86% 18%, rgba(255,255,255,0.97) 0 18px, transparent 19px), radial-gradient(circle at 89% 16%, #0b2345 0 16px, transparent 17px), radial-gradient(circle at 12% 34%, rgba(255,255,255,0.96) 0 1.3px, transparent 1.8px), radial-gradient(circle at 24% 18%, rgba(255,255,255,0.92) 0 1.2px, transparent 1.7px), radial-gradient(circle at 38% 42%, rgba(255,255,255,0.9) 0 1.2px, transparent 1.7px), radial-gradient(circle at 54% 22%, rgba(255,255,255,0.95) 0 1.3px, transparent 1.8px), radial-gradient(circle at 68% 36%, rgba(255,255,255,0.9) 0 1.2px, transparent 1.7px), radial-gradient(circle at 80% 28%, rgba(255,255,255,0.94) 0 1.3px, transparent 1.8px), linear-gradient(135deg, #041226 0%, #0b2345 50%, #163663 100%)",
            "bg_size": "auto, auto, auto, auto, auto, auto, auto, auto, auto",
            "border": "#9cb6df",
            "label": _weather_card_label(action),
            "card_animation": "weatherNightStarBg 10s ease-in-out infinite",
        },
    }
    theme_key = f"night_{theme}" if night and f"night_{theme}" in themes else theme
    cfg_theme = themes.get(theme_key, themes["cloud"])
    card_html = dedent(
        f"""
        <style>
        @keyframes weatherNightStarBg {{
            0%, 100% {{ background-position: 0 0, 0 0, 0 0, 0 0, 0 0, 0 0, 0 0, 0 0, 0 0; }}
            50% {{ background-position: 0 0, 0 0, 2px 1px, -2px 2px, 1px -1px, -1px 1px, 2px 2px, -2px -1px, 0 0; }}
        }}
        @keyframes weatherNightCloudBg {{
            0%, 100% {{ background-position: 0 0, 0 0, 0 0, 0 0, 0 0, 0 0, 0px 0px, 0px 0px, 0px 0px, 0 0; }}
            50% {{ background-position: 0 0, 0 0, 1px 0px, -1px 1px, 0px -1px, 1px 0px, 12px 0px, -10px 0px, 8px 0px, 0 0; }}
        }}
        @keyframes weatherNightRainBg {{
            0% {{ background-position: 0 -28px, 0 0, 0 0, 0 0, 0 0, 0 0, 0 0, 0 0; }}
            100% {{ background-position: 24px 28px, 0 0, 0 0, 2px 1px, -1px 1px, 1px -1px, -1px 1px, 0 0; }}
        }}
        @keyframes weatherDaySunBg {{
            0%, 100% {{ background-position: 0 0, 0 0, 0 0, 0 0, 0 0; }}
            50% {{ background-position: 0 0, 2px -1px, -2px 1px, 1px 1px, 0 0; }}
        }}
        @keyframes weatherDayCloudBg {{
            0%, 100% {{ background-position: 0px 0px, 0px 0px, 0px 0px, 0 0; }}
            50% {{ background-position: 10px 0px, -8px 0px, 6px 0px, 0 0; }}
        }}
        @keyframes weatherDayRainBg {{
            0% {{ background-position: 0 -24px, 0 0; }}
            100% {{ background-position: 26px 24px, 0 0; }}
        }}
        </style>
        <div style="
            background: {cfg_theme['bg']};
            background-size: {cfg_theme['bg_size']};
            border: 1px solid {cfg_theme['border']};
            border-radius: 18px;
            padding: 16px 18px;
            color: #ffffff;
            box-shadow: 0 10px 24px rgba(0,0,0,0.16);
            margin: 4px 0 6px 0;
            position: relative;
            overflow: hidden;
            animation: {cfg_theme['card_animation']};
            user-select: text;
            -webkit-user-select: text;
        ">
            <div style="position: relative; z-index: 1;">
            <div style="font-size: 0.76rem; letter-spacing: 0.08em; text-transform: uppercase; opacity: 0.88; margin-bottom: 8px;">
                {cfg_theme['label']}
            </div>
            <div style="font-size: 1.05rem; font-weight: 700; margin-bottom: 8px;">{title}</div>
            <div style="font-size: 0.96rem; line-height: 1.65; white-space: normal; user-select: text; -webkit-user-select: text;">{body}</div>
            </div>
        </div>
        """
    ).strip()
    st.markdown(card_html, unsafe_allow_html=True)


def render_market_panel(meta: dict | None = None, auto_chart: pd.DataFrame | None = None, auto_table: pd.DataFrame | None = None) -> None:
    if meta is None:
        meta = st.session_state.get("auto_market_meta") or {}
    if auto_chart is None:
        auto_chart = st.session_state.get("auto_chart")
    if auto_table is None:
        auto_table = st.session_state.get("auto_forecast_table")
    if auto_chart is None or auto_table is None or not meta:
        return

    latest = meta.get("latest") or {}
    commodity_label = meta.get("commodity_label", "")
    caption = meta.get("caption", "")
    market_list = meta.get("market_list", [])
    nearest_market = meta.get("nearest_market")

    price_val = _safe_float(latest.get("Modal_Price"))
    arrival_val = _safe_float(latest.get("Arrival_Qty"))
    price_unit = _safe_text(latest.get("Price_Unit"), "Rs./Quintal")
    arrival_unit = _safe_text(latest.get("Arrival_Unit"), "Metric Tonnes")
    arrival_date = _safe_text(latest.get("Arrival_Date"))

    st.markdown("---")
    st.subheader("Market Snapshot")
    if caption:
        st.caption(caption)

    left, right = st.columns([1, 1.5])
    with left:
        if commodity_label:
            st.markdown(f"**फसल:** {commodity_label}")
        if price_val is not None:
            price_display = f"₹{price_val:,.0f}"
            if price_unit:
                price_display += f" {price_unit}"
            st.metric("ताज़ा भाव", price_display)
        if arrival_val is not None:
            arrival_display = f"{arrival_val:,.1f}"
            if arrival_unit:
                arrival_display += f" {arrival_unit}"
            st.metric("आवक", arrival_display)
        if arrival_date:
            st.write(f"**तारीख:** {arrival_date}")
        if nearest_market:
            st.write(f"**निकटतम मंडी:** {nearest_market[1]}")
            st.write(f"**दूरी:** {nearest_market[0]:.1f} km")
        elif market_list:
            st.write("**उपलब्ध मंडियाँ (नमूना):**")
            st.write(", ".join(sorted(market_list)[:8]))

    with right:
        st.line_chart(auto_chart, use_container_width=True)

    if auto_table is not None and not auto_table.empty:
        display_table = auto_table.copy()
        rename_cols = {}
        if "date" in display_table.columns:
            rename_cols["date"] = "तारीख"
        if "Forecast" in display_table.columns:
            rename_cols["Forecast"] = "अनुमानित भाव"
        if rename_cols:
            display_table = display_table.rename(columns=rename_cols)
        st.dataframe(display_table, use_container_width=True, height=260)


def check_ready() -> tuple[bool, str]:
    db_exists = os.path.exists(cfg.paths["sqlite_db"])
    idx_exists = os.path.exists(cfg.paths["vector_store"])
    md_exists = os.path.exists(cfg.paths["metadata_store"])

    if not db_exists:
        return False, f"Missing SQLite DB: {cfg.paths['sqlite_db']}"
    if not idx_exists:
        return False, f"Missing vector index: {cfg.paths['vector_store']}"
    if not md_exists:
        return False, f"Missing metadata store: {cfg.paths['metadata_store']}"
    return True, "System ready"


def _build_advisor_config(**kwargs: object) -> AdvisorConfig:
    supported = set(getattr(AdvisorConfig, "__dataclass_fields__", {}).keys())
    filtered = {key: value for key, value in kwargs.items() if key in supported}
    return AdvisorConfig(**filtered)


@lru_cache(maxsize=4)
def get_advisor(_build_version: str = APP_BUILD_VERSION) -> RAGAdvisor:
    _ = _build_version
    return RAGAdvisor(
        _build_advisor_config(
            embedding_model=cfg.embedding_model,
            generator_model=MEDIUM_GENERATOR_MODEL,
            index_path=cfg.paths["vector_store"],
            metadata_path=cfg.paths["metadata_store"],
            top_k=cfg.top_k,
            db_path=cfg.paths["sqlite_db"],
            complex_generator_model=COMPLEX_GENERATOR_MODEL,
            response_cache_path=os.getenv("KISAANAI_RESPONSE_CACHE_PATH", "data/processed/query_response_cache.json"),
            response_cache_version=APP_BUILD_VERSION,
            query_cache_ttl_sec=int(os.getenv("KISAANAI_QUERY_CACHE_TTL_SEC", str(6 * 60 * 60))),
        )
    )


@lru_cache(maxsize=32)
def load_market_df(csv_path: str, mtime_ns: int) -> pd.DataFrame:
    _ = mtime_ns
    return pd.read_csv(csv_path)


@lru_cache(maxsize=64)
def build_forecast(
    csv_path: str,
    mtime_ns: int,
    commodity: str,
    state: str,
    district: str,
    horizon: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _ = mtime_ns
    df = pd.read_csv(csv_path)
    series = prepare_daily_series(
        df=df,
        date_col="Arrival_Date",
        value_col="Modal_Price",
        commodity=commodity,
        state=state,
        district=district,
    )
    result = train_and_forecast(
        series_df=series,
        horizon_days=horizon,
        lookback=30,
        epochs=40,
    )
    return result.history, result.forecast


def clear_local_caches() -> None:
    for fn in (
        get_advisor,
        _load_agmarknet_df_cached,
        _load_agmarknet_catalog_cached,
        load_market_df,
        build_forecast,
        load_commodity_catalog,
        load_location_lookup,
        load_commodity_aliases,
        load_training_feedback_memory,
        load_location_corrections,
    ):
        try:
            fn.cache_clear()
        except Exception:
            pass


def build_forecast_from_df(
    df: pd.DataFrame,
    commodity: str,
    state: str,
    district: str,
    horizon: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    series = prepare_daily_series(
        df=df,
        date_col="Arrival_Date",
        value_col="Modal_Price",
        commodity=commodity,
        state=state,
        district=district,
    )
    result = train_and_forecast(
        series_df=series,
        horizon_days=horizon,
        lookback=30,
        epochs=40,
    )
    return result.history, result.forecast


def build_quick_forecast_from_df(
    df: pd.DataFrame,
    commodity: str,
    state: str,
    district: str,
    horizon: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    series = prepare_daily_series(
        df=df,
        date_col="Arrival_Date",
        value_col="Modal_Price",
        commodity=commodity,
        state=state,
        district=district,
    )
    result = quick_forecast(
        series_df=series,
        horizon_days=horizon,
    )
    return result.history, result.forecast


def load_local_env(env_path: Path) -> dict[str, str]:
    vals: dict[str, str] = {}
    if not env_path.exists():
        return vals
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        vals[k.strip()] = v.strip().strip('"').strip("'")
    return vals


def get_setting(name: str, env_vals: dict[str, str] | None = None, default: str = "") -> str:
    env_vals = env_vals or {}
    try:
        if hasattr(st, "secrets") and name in st.secrets:
            val = st.secrets[name]
            if val is not None:
                return str(val).strip()
    except Exception:
        pass
    val = env_vals.get(name) or os.getenv(name) or default
    return str(val).strip()


def smtp_config_status() -> dict[str, bool | str]:
    env_vals = load_local_env(Path(".env"))
    app_base_url = get_setting("APP_BASE_URL", env_vals)
    smtp_host = get_setting("SMTP_HOST", env_vals)
    smtp_port = get_setting("SMTP_PORT", env_vals, "587")
    smtp_user = get_setting("SMTP_USER", env_vals)
    smtp_pass = get_setting("SMTP_PASS", env_vals)
    smtp_from = get_setting("SMTP_FROM", env_vals, smtp_user)
    return {
        "app_base_url": bool(app_base_url),
        "smtp_host": bool(smtp_host),
        "smtp_port": bool(smtp_port),
        "smtp_user": bool(smtp_user),
        "smtp_pass": bool(smtp_pass),
        "smtp_from": bool(smtp_from),
        "smtp_ready": bool(app_base_url and smtp_host and smtp_port and smtp_user and smtp_pass and smtp_from),
        "app_base_url_value": app_base_url,
        "smtp_host_value": smtp_host,
        "smtp_from_value": smtp_from,
    }


def merge_market_data(existing_path: Path, new_df: pd.DataFrame) -> pd.DataFrame:
    if existing_path.exists():
        try:
            old_df = pd.read_csv(existing_path)
            merged = pd.concat([old_df, new_df], ignore_index=True)
        except Exception:
            merged = new_df.copy()
    else:
        merged = new_df.copy()
    key_cols = [
        c
        for c in [
            "State",
            "District",
            "Market",
            "Commodity",
            "Variety",
            "Grade",
            "Arrival_Date",
            "Modal_Price",
        ]
        if c in merged.columns
    ]
    if key_cols:
        merged = merged.drop_duplicates(subset=key_cols, keep="last")
    return merged


def normalize_agmarknet_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]
    # Live Agmarknet dashboard snapshots can include both the older CSV-style
    # fields and the newer dashboard-style fields. Coalesce them first so we
    # never create duplicate logical columns like Arrival_Date/Modal_Price.
    if "reported_date" in out.columns:
        if "rep_date" in out.columns:
            out["rep_date"] = out["reported_date"].fillna(out["rep_date"])
            out = out.drop(columns=["reported_date"])
        else:
            out = out.rename(columns={"reported_date": "rep_date"})
    if "as_on" in out.columns:
        if "model_price_wt" in out.columns:
            out["model_price_wt"] = out["as_on"].fillna(out["model_price_wt"])
            out = out.drop(columns=["as_on"])
        else:
            out = out.rename(columns={"as_on": "model_price_wt"})
    col_map = {
        "state_name": "State",
        "district_name": "District",
        "market_name": "Market",
        "cmdt_name": "Commodity",
        "rep_date": "Arrival_Date",
        "model_price_wt": "Modal_Price",
        "min_price_wt": "Min_Price",
        "max_price_wt": "Max_Price",
        "unit_name_price": "Price_Unit",
        "cumm_arr": "Arrival_Qty",
        "unit_name_arrival": "Arrival_Unit",
    }
    rename = {}
    for k, v in col_map.items():
        if k in out.columns and v not in out.columns:
            rename[k] = v
    if rename:
        out = out.rename(columns=rename)
    return out


def build_agmarknet_catalog_df(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_agmarknet_df(df)
    keep = [col for col in ["State", "District", "Commodity"] if col in out.columns]
    if not keep:
        return pd.DataFrame(columns=["State", "District", "Commodity"])
    catalog = out[keep].copy()
    for col in ["State", "District", "Commodity"]:
        if col not in catalog.columns:
            catalog[col] = ""
    catalog = catalog[["State", "District", "Commodity"]].fillna("")
    catalog = catalog.astype(str).drop_duplicates().sort_values(["State", "District", "Commodity"])
    return catalog.reset_index(drop=True)


def save_agmarknet_catalog(df: pd.DataFrame, out_path: Path = AGMARKNET_CATALOG_CSV) -> None:
    catalog = build_agmarknet_catalog_df(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(out_path, index=False)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    from math import radians, sin, cos, asin, sqrt

    r = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return r * c


def _geocode_cached(name: str, admin: str | None = None) -> tuple[float, float, str] | None:
    if not name:
        return None
    cache_path = Path("data/processed/market_geocode_cache.json")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        cache = {}
    key = f"{name}|{admin or ''}".lower()
    if key in cache:
        lat, lon, label = cache[key]
        return float(lat), float(lon), label
    candidates = [name]
    if admin:
        candidates.append(f"{name}, {admin}")
    candidates.append(f"{name}, Uttar Pradesh, India")
    for cand in candidates:
        params = urlencode({"name": cand, "count": 1, "language": "en", "format": "json"})
        url = f"https://geocoding-api.open-meteo.com/v1/search?{params}"
        try:
            with urlopen(url, timeout=6) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception:
            continue
        results = payload.get("results") or []
        if not results:
            continue
        top = results[0]
        try:
            lat = float(top.get("latitude"))
            lon = float(top.get("longitude"))
        except Exception:
            continue
        label = ", ".join([p for p in [top.get("name"), top.get("admin1"), top.get("country")] if p])
        cache[key] = [lat, lon, label]
        try:
            cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        except Exception:
            pass
        return lat, lon, label
    return None


def _reverse_geocode_cached(lat: float, lon: float) -> dict[str, str] | None:
    cache_path = Path("data/processed/reverse_geocode_cache.json")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        cache = {}
    key = f"{lat:.4f},{lon:.4f}"
    if key in cache:
        return cache[key]
    params = urlencode({"format": "json", "lat": lat, "lon": lon, "zoom": 10, "addressdetails": 1})
    url = f"https://nominatim.openstreetmap.org/reverse?{params}"
    try:
        with urlopen(url, timeout=8) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    address = payload.get("address", {}) if isinstance(payload, dict) else {}
    out = {
        "district": address.get("district")
        or address.get("county")
        or address.get("state_district")
        or "",
        "state": address.get("state") or "",
        "village": address.get("village") or address.get("town") or address.get("city") or "",
    }
    cache[key] = out
    try:
        cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return out


def _forward_geocode_cached(place: str, region: str = "Uttar Pradesh") -> list[dict[str, str]]:
    if not place:
        return []
    cache_path = Path("data/processed/forward_geocode_cache.json")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        cache = {}
    key = f"{place}|{region}".lower()
    if key in cache:
        return cache[key]
    params = urlencode({"q": f"{place}, {region}, India", "format": "json", "limit": 5, "addressdetails": 1})
    url = f"https://nominatim.openstreetmap.org/search?{params}"
    try:
        with urlopen(url, timeout=8) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []
    if not payload:
        return []
    results: list[dict[str, str]] = []
    for top in payload:
        addr = top.get("address", {}) if isinstance(top, dict) else {}
        out = {
            "district": addr.get("district")
            or addr.get("county")
            or addr.get("state_district")
            or "",
            "state": addr.get("state") or "",
            "village": addr.get("village") or addr.get("town") or addr.get("city") or "",
            "lat": str(top.get("lat", "")),
            "lon": str(top.get("lon", "")),
            "importance": str(top.get("importance", "")),
        }
        results.append(out)
    cache[key] = results
    try:
        cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return results


def load_agmarknet_df() -> pd.DataFrame:
    if not AGMARKNET_CSV.exists():
        return pd.DataFrame()
    mtime_ns = AGMARKNET_CSV.stat().st_mtime_ns
    return _load_agmarknet_df_cached(mtime_ns)


@lru_cache(maxsize=4)
def _load_agmarknet_df_cached(mtime_ns: int) -> pd.DataFrame:
    _ = mtime_ns
    try:
        df = pd.read_csv(AGMARKNET_CSV)
        return normalize_agmarknet_df(df)
    except Exception:
        return pd.DataFrame()


def load_agmarknet_catalog() -> pd.DataFrame:
    if AGMARKNET_CATALOG_CSV.exists():
        mtime_ns = AGMARKNET_CATALOG_CSV.stat().st_mtime_ns
        return _load_agmarknet_catalog_cached(mtime_ns)
    if not AGMARKNET_CSV.exists():
        return pd.DataFrame(columns=["State", "District", "Commodity"])
    mtime_ns = AGMARKNET_CSV.stat().st_mtime_ns
    return _load_agmarknet_catalog_cached(mtime_ns)


@lru_cache(maxsize=4)
def _load_agmarknet_catalog_cached(mtime_ns: int) -> pd.DataFrame:
    _ = mtime_ns
    if AGMARKNET_CATALOG_CSV.exists():
        try:
            df = pd.read_csv(AGMARKNET_CATALOG_CSV)
            for col in ["State", "District", "Commodity"]:
                if col not in df.columns:
                    df[col] = ""
            return df[["State", "District", "Commodity"]].fillna("").astype(str)
        except Exception:
            pass
    raw_usecols_candidates = [
        ["state_name", "district_name", "cmdt_name"],
        ["State", "District", "Commodity"],
    ]
    df = None
    for usecols in raw_usecols_candidates:
        try:
            df = pd.read_csv(AGMARKNET_CSV, usecols=usecols)
            break
        except Exception:
            df = None
    if df is None:
        try:
            df = pd.read_csv(AGMARKNET_CSV)
        except Exception:
            return pd.DataFrame(columns=["State", "District", "Commodity"])
    out = build_agmarknet_catalog_df(df)
    try:
        save_agmarknet_catalog(out)
    except Exception:
        pass
    return out


def is_price_query(text: str) -> bool:
    t = text.lower()
    keywords = [
        "price",
        "rate",
        "mandi",
        "bhav",
        "bhaav",
        "bhao",
        "daam",
        "dam",
        "भाव",
        "दाम",
        "कीमत",
        "keemat",
        "kimat",
        "qeemat",
        "मंडी",
        "forecast",
        "भविष्य",
        "अगले",
        "आने वाले",
        "15 दिन",
        "पंद्रह दिन",
        "msp",
        "minimum support price",
        "support price",
        "न्यूनतम समर्थन मूल्य",
        "समर्थन मूल्य",
        "सरकारी भाव",
        "सरकारी रेट",
    ]
    return any(k in t for k in keywords)


def is_latest_commodity_price_query(text: str) -> bool:
    t = (text or "").lower()
    latest_markers = [
        "latest available",
        "latest price available",
        "date k hisab",
        "date ke hisab",
        "kis date ka latest",
        "konsi fasal ka price latest",
        "kaunsi fasal ka price latest",
        "which commodity has latest price",
        "which crop has latest price",
        "सबसे latest",
        "सबसे लेटेस्ट",
        "लेटेस्ट उपलब्ध",
        "ताज़ा उपलब्ध",
    ]
    commodity_markers = ["commodity", "crop", "fasal", "फसल", "price", "bhav", "भाव", "rate", "कीमत", "मंडी"]
    return any(marker in t for marker in latest_markers) and any(marker in t for marker in commodity_markers)


def wants_detailed_price_forecast(text: str) -> bool:
    t = (text or "").lower()
    keys = [
        "forecast",
        "trend",
        "prediction",
        "predict",
        "अगले",
        "आने वाले",
        "भविष्य",
        "अनुमान",
        "7 दिन",
        "15 दिन",
        "next week",
        "tomorrow",
        "kal",
        "agle",
    ]
    return any(k in t for k in keys)


def is_msp_query(text: str) -> bool:
    t = (text or "").lower()
    keys = [
        "msp",
        "minimum support price",
        "support price",
        "न्यूनतम समर्थन मूल्य",
        "समर्थन मूल्य",
        "सरकारी भाव",
        "सरकारी रेट",
        "support rate",
    ]
    return any(k in t for k in keys)


def is_crop_query(text: str) -> bool:
    t = text.lower()
    keys = [
        "which crop",
        "best crop",
        "crop to grow",
        "what crop should i grow",
        "कौन सी फसल",
        "फसल बेहतर",
        "फसल उगानी",
        "कौनसी फसल",
        "कौनसी फसल",
        "किस फसल",
        "profit",
        "profitable",
        "लाभ",
        "लाभदायक",
    ]
    return any(k in t for k in keys)


def get_sugarcane_price_fallback() -> dict[str, str | float]:
    frp = get_latest_sugarcane_frp()
    if frp:
        return {
            "price": float(frp.get("price_per_qtl") or 0),
            "season": str(frp.get("season") or ""),
            "source": "CACP FRP",
            "source_url": str(frp.get("source_url") or ""),
        }
    baseline = WESTERN_UP_CROP_BASELINES.get("Sugarcane", {})
    return {
        "price": float(baseline.get("fallback_price") or 380),
        "season": "2025-26",
        "source": "baseline/MSP-SAP estimate",
        "source_url": "",
    }


SPECIALTY_CROP_PRICE_NOTES = {
    "saffron": (
        "केसर जैसी specialty crop का भाव grade, origin और quality के हिसाब से काफी बदलता है। "
        "अगर आप specific market या state बताएं, तो targeted price lookup ज्यादा उपयोगी रहेगा।"
    ),
}


def _format_specialty_crop_price_unavailable(
    commodity_label: str,
    commodity_key: str,
    selected_district: str,
    mention_district: bool,
) -> str:
    intro = (
        f"चयनित जिले ({selected_district}) में {commodity_label} का मंडी डेटा उपलब्ध नहीं है।\n"
        if mention_district and selected_district
        else ""
    )
    base = f"{commodity_label} के लिए स्थानीय मंडी/MSP डेटा उपलब्ध नहीं मिला। "
    note = SPECIALTY_CROP_PRICE_NOTES.get(
        commodity_key,
        "इस फसल का भाव market, quality और grade के हिसाब से बदल सकता है। "
        "अगर आप specific market या state बताएं, तो मैं targeted lookup कर सकता हूँ।",
    )
    return f"{intro}{base}{note}".strip()


def _try_price_web_fallback(
    advisor: RAGAdvisor | None,
    *,
    user_query: str,
    commodity_label: str,
    commodity_key: str,
    selected_state: str,
    selected_district: str,
    mention_district: bool,
) -> tuple[str | None, list[str]]:
    if advisor is None or not hasattr(advisor, "_answer_with_web_search"):
        return None, []
    question_candidates = [
        (user_query or "").strip(),
        f"{commodity_key} mandi price India",
        f"{commodity_key} market price India",
        f"{commodity_key} current price India",
    ]
    if selected_district:
        question_candidates.insert(1, f"{commodity_key} mandi price {selected_district} India")
    context_parts = []
    if selected_district:
        context_parts.append(f"जिला: {selected_district}")
    if selected_state:
        context_parts.append(f"राज्य: {selected_state}")
    context_part = " | ".join(context_parts)
    seen: set[str] = set()
    for question in question_candidates:
        q = re.sub(r"\s+", " ", question).strip()
        if not q:
            continue
        q_norm = q.lower()
        if q_norm in seen:
            continue
        seen.add(q_norm)
        try:
            web_result = advisor._answer_with_web_search(q, context_part)
        except Exception:
            web_result = None
        if not web_result:
            continue
        answer = str(web_result.get("answer") or "").strip()
        refs = [str(r).strip() for r in (web_result.get("references") or []) if str(r).strip()]
        if not answer:
            continue
        if mention_district and selected_district and "चयनित जिले" not in answer:
            answer = f"चयनित जिले ({selected_district}) में स्थानीय मंडी डेटा उपलब्ध नहीं है।\n{answer}"
        return answer, refs
    return None, []


def is_profitability_followup_query(text: str) -> bool:
    t = text.lower()
    keys = [
        "lagat kaise",
        "lagat kaese",
        "cost kaise",
        "cost kese",
        "profit kaise",
        "profit kese",
        "revenue kaise",
        "kaise nikali",
        "kese nikali",
        "kaise nikala",
        "kese nikala",
        "लागत कैसे",
        "लाभ कैसे",
        "मुनाफा कैसे",
        "हिसाब कैसे",
        "कैसे निकाली",
        "कैसे निकाला",
    ]
    return any(k in t for k in keys)


def _best_match(query: str, options: list[str]) -> str | None:
    q = query.lower()
    q_tokens = re.findall(r"[a-z0-9]+", q)
    matches = []
    for opt in options:
        if not opt:
            continue
        o = opt.lower()
        o_tokens = re.findall(r"[a-z0-9]+", o)
        if not o_tokens:
            continue
        if len(o_tokens) == 1:
            if o_tokens[0] in q_tokens:
                matches.append(opt)
        else:
            if " ".join(o_tokens) in " ".join(q_tokens):
                matches.append(opt)
    if not matches:
        return None
    matches.sort(key=lambda x: len(x), reverse=True)
    return matches[0]


def _normalize_district_name(name: str) -> str:
    n = name.lower().strip()
    n = n.replace("district", "").replace("division", "").replace(" मंडल", "").replace(" जिला", "")
    n = re.sub(r"[^a-z0-9]+", "", n)
    return n



def _filter_location_lookup_scope(
    lookup: pd.DataFrame,
    state: str | None = None,
    district: str | None = None,
) -> pd.DataFrame:
    if lookup.empty:
        return lookup
    out = lookup.copy()
    state_norm = str(state or "").strip().lower()
    district_norm = _normalize_district_name(district or "")
    if state_norm and "state" in out.columns:
        out = out[out["state"].astype(str).str.strip().str.lower() == state_norm]
    if district_norm and "district" in out.columns:
        out = out[out["district"].astype(str).map(_normalize_district_name) == district_norm]
    return out


def _build_location_alias_map(lookup: pd.DataFrame) -> dict[str, str]:
    if lookup.empty:
        return {}
    alias_map: dict[str, str] = {}
    suffixes = {"rural", "bangar", "khadar", "khurd", "kalan"}

    def add_alias(alias: str, raw: str) -> None:
        alias_norm = _normalize_text(alias)
        if len(alias_norm) < 3:
            return
        current = alias_map.get(alias_norm)
        if current is None or len(raw) < len(current):
            alias_map[alias_norm] = raw

    def expand_aliases(tokens: list[str]) -> set[str]:
        aliases: set[str] = set()
        if not tokens:
            return aliases
        aliases.add(" ".join(tokens))
        aliases.add(tokens[0])
        if len(tokens) >= 2:
            aliases.add(" ".join(tokens[:2]))
        if len(tokens) >= 2 and tokens[-1] in suffixes:
            aliases.add(" ".join(tokens[:-1]))
        expanded: set[str] = set()
        for alias in aliases:
            alias_norm = _normalize_text(alias)
            if not alias_norm:
                continue
            expanded.add(alias_norm)
            if alias_norm.endswith("ban"):
                expanded.add(alias_norm[:-3] + "van")
            if alias_norm.endswith("van"):
                expanded.add(alias_norm[:-3] + "ban")
            if "w" in alias_norm:
                expanded.add(alias_norm.replace("w", "v"))
            if "v" in alias_norm:
                expanded.add(alias_norm.replace("v", "w"))
        return expanded

    for col in ["place", "sub_district"]:
        if col not in lookup.columns:
            continue
        for raw in lookup[col].dropna().astype(str).unique().tolist():
            raw = raw.strip()
            if not raw or raw.lower() == "nan":
                continue
            tokens = [t for t in re.findall(r"[a-z0-9]+", raw.lower()) if t]
            for alias in expand_aliases(tokens):
                add_alias(alias, raw)
    return alias_map


def _match_place_from_lookup(
    text: str,
    lookup: pd.DataFrame,
    stop: set[str] | None = None,
    commodity_tokens: set[str] | None = None,
) -> str | None:
    if not text or lookup.empty:
        return None
    alias_map = _build_location_alias_map(lookup)
    if not alias_map:
        return None
    stop = stop or set()
    commodity_tokens = commodity_tokens or set()
    raw_tokens = re.findall(r"[a-z0-9]+", str(text).lower())
    if not raw_tokens:
        return None

    candidate_norms: list[str] = []
    seen: set[str] = set()

    def add_candidate(candidate: str) -> None:
        norm = _normalize_text(candidate)
        if not norm or norm in seen:
            return
        seen.add(norm)
        candidate_norms.append(norm)

    filtered_tokens = [
        token for token in raw_tokens
        if token not in stop and token not in commodity_tokens and len(token) >= 3
    ]
    for size in range(min(4, len(filtered_tokens)), 0, -1):
        for idx in range(0, len(filtered_tokens) - size + 1):
            add_candidate(" ".join(filtered_tokens[idx : idx + size]))
    for token in filtered_tokens:
        add_candidate(token)
    if filtered_tokens:
        add_candidate(" ".join(filtered_tokens[-2:]))
        add_candidate(" ".join(filtered_tokens[-3:]))

    for norm in candidate_norms:
        if norm in alias_map:
            return alias_map[norm]

    alias_keys = list(alias_map.keys())
    for norm in candidate_norms:
        if len(norm) < 4:
            continue
        prefix_matches = [key for key in alias_keys if key.startswith(norm) or norm.startswith(key)]
        if prefix_matches:
            best_key = min(prefix_matches, key=len)
            return alias_map[best_key]

    for norm in candidate_norms:
        cutoff = 0.78 if len(norm) >= 6 else 0.9
        close = difflib.get_close_matches(norm, alias_keys, n=1, cutoff=cutoff)
        if close:
            return alias_map[close[0]]
    return None


def extract_place_from_query(query: str, lookup: pd.DataFrame | None = None) -> str | None:
    # Prefer a direct lookup match from the known location table before falling
    # back to token filtering. This keeps place parsing stable even when the
    # query contains extra words like commodity names or question words.
    q = strip_relative_location(query)
    if not q:
        return None

    stop = {
        "like", "kya", "ky", "what", "which", "kitna", "kitne", "kitni",
        "and", "or", "also", "please", "tell", "my", "for", "in", "at", "of", "to", "a", "an", "on", "with", "और", "होगा", "the", "is", "are", "will", "it", "be", "how", "today", "tomorrow", "tonight", "forecast", "next", "week", "day", "days", "rain", "rainfall", "temperature", "kal", "parso", "आज", "कल", "बारिश", "तापमान", "रहेगा", "कैसा", "है",
        "aaj", "aj", "abhi", "ka", "ki", "ke", "ko", "se", "par",
        "me", "mein", "में", "kesa", "kaisa", "hai", "h",
        "pani", "paani", "water", "sinchai", "sichai", "sinchaai", "irrigation",
        "lagta", "lagti", "lagte", "lata", "leti", "chahiye",
        "price", "rate", "mandi", "bhav", "bhaav", "bhao", "daam", "dam",
        "भाव", "कीमत", "मंडी", "मौसम", "mausam", "mosam", "weather",
        "btaye", "bataye", "bataiye", "btao", "batao", "boliye", "bolo",
        "do", "de", "dijiye", "dijie", "batayiye", "btaiye", "liye", "liyee",
        "बताएं", "बताये", "बताइए", "बताओ", "दीजिए", "दो",
    }
    alias_path = Path("data/raw/commodity_aliases.json")
    mtime_ns = alias_path.stat().st_mtime_ns if alias_path.exists() else 0
    aliases = load_commodity_aliases(mtime_ns)
    commodity_tokens = set()
    for alias_list in aliases.values():
        for alias in alias_list:
            for t in re.findall(r"[a-z0-9]+", alias.lower()):
                commodity_tokens.add(t)

    if lookup is None:
        lookup_path = Path("data/processed/location_lookup.csv")
        lookup_mtime = lookup_path.stat().st_mtime_ns if lookup_path.exists() else 0
        lookup = load_location_lookup(lookup_mtime)
    q_norm = _normalize_text(q)
    q_tokens = re.findall(r"[a-z0-9]+", q.lower())
    if q_norm and not lookup.empty:
        candidates: list[tuple[int, int, str]] = []
        for col in ["place", "sub_district", "district"]:
            if col not in lookup.columns:
                continue
            for raw in lookup[col].dropna().astype(str).unique().tolist():
                raw = raw.strip()
                if not raw or raw.lower() == "nan":
                    continue
                raw_tokens = re.findall(r"[a-z0-9]+", raw.lower())
                if not raw_tokens:
                    continue
                matched = False
                token_hits = 0
                if len(raw_tokens) == 1:
                    token = raw_tokens[0]
                    if len(token) >= 3 and token in q_tokens and token not in stop and token not in commodity_tokens:
                        matched = True
                        token_hits = 1
                else:
                    joined = " ".join(raw_tokens)
                    q_joined = " ".join(q_tokens)
                    if joined in q_joined:
                        useful = [t for t in raw_tokens if t not in stop and t not in commodity_tokens]
                        if useful:
                            matched = True
                            token_hits = len(useful)
                    else:
                        hits = [
                            t for t in raw_tokens
                            if len(t) >= 3 and t in q_tokens and t not in stop and t not in commodity_tokens
                        ]
                        if hits:
                            matched = True
                            token_hits = len(hits)
                if matched:
                    candidates.append((token_hits, len("".join(raw_tokens)), raw))
        if candidates:
            candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
            return candidates[0][2]

    lookup_match = _match_place_from_lookup(q, lookup, stop, commodity_tokens)
    if lookup_match:
        return lookup_match

    tokens = [t.strip(" ?!.,") for t in q.split() if t.strip()]
    if not tokens:
        return None

    filtered = []
    for tok in tokens:
        t = re.sub(r"[^a-zA-Z0-9ऀ-ॿ]+", "", tok).lower()
        if not t or t in stop or t in commodity_tokens:
            continue
        filtered.append(tok)

    while filtered:
        last = re.sub(r"[^a-zA-Z0-9ऀ-ॿ]+", "", filtered[-1]).lower()
        if not last or last in stop or last in commodity_tokens or len(re.sub(r"[^a-z0-9]+", "", last)) <= 1:
            filtered.pop()
        else:
            break
    if not filtered:
        return None
    candidate = " ".join(filtered)
    if not lookup.empty:
        matched_candidate = _match_place_from_lookup(candidate, lookup, stop, commodity_tokens)
        if matched_candidate:
            return matched_candidate
        district_guess, state_guess = _lookup_district_from_location(candidate, lookup)
        if district_guess or state_guess:
            return candidate
    return None



def _place_variants(place: str) -> list[str]:
    base = place.strip()
    variants = [base]
    if base.lower().endswith("e") and len(base) > 3:
        variants.append(base[:-1])
    return variants


@lru_cache(maxsize=8)
def load_commodity_catalog() -> list[str]:
    path = Path("data/raw/agmarknet_commodities.csv")
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    if "commodity_name" not in df.columns:
        return []
    names = df["commodity_name"].dropna().astype(str).unique().tolist()
    return sorted(names)


def _normalize_text(val: str) -> str:
    return "".join(ch for ch in val.lower() if ch.isalnum())


def _lookup_district_from_location(place: str, lookup: pd.DataFrame) -> tuple[str | None, str | None]:
    if not place or lookup.empty:
        return None, None
    norm = _normalize_text(place)
    if not norm:
        return None, None
    # Priority: village/place -> sub-district -> district
    for col in ["place_norm", "sub_district", "district"]:
        if col not in lookup.columns:
            continue
        if col == "place_norm":
            matches = lookup[lookup[col] == norm]
        else:
            matches = lookup[lookup[col].astype(str).map(_normalize_text) == norm]
        if matches.empty:
            continue
        up = matches[matches["state"].str.lower() == "uttar pradesh"] if "state" in matches.columns else matches
        pick = up.iloc[0] if not up.empty else matches.iloc[0]
        district = str(pick.get("district", "")).strip()
        state = str(pick.get("state", "")).strip()
        return (district or None), (state or None)
    # Fallback: contains match for place names like "Doghat Rural"
    if "place" in lookup.columns and (" " in place.strip() or len(norm) >= 5):
        contains = lookup[lookup["place"].astype(str).map(_normalize_text).str.contains(norm, na=False)]
        if not contains.empty:
            up = contains[contains["state"].str.lower() == "uttar pradesh"] if "state" in contains.columns else contains
            pick = up.iloc[0] if not up.empty else contains.iloc[0]
            district = str(pick.get("district", "")).strip()
            state = str(pick.get("state", "")).strip()
            return (district or None), (state or None)
    # Fallback: fuzzy match (handles minor spelling errors like Kurava->Kurawa)
    try:
        pool = lookup
        if "state" in lookup.columns:
            up = lookup[lookup["state"].str.lower() == "uttar pradesh"]
            if not up.empty:
                pool = up
        norms = pool["place_norm"].dropna().astype(str).unique().tolist()
        matches = difflib.get_close_matches(norm, norms, n=1, cutoff=0.8)
        if matches:
            m = pool[pool["place_norm"] == matches[0]]
            if not m.empty:
                pick = m.iloc[0]
                district = str(pick.get("district", "")).strip()
                state = str(pick.get("state", "")).strip()
                return (district or None), (state or None)
    except Exception:
        pass
    # Fallback: v/w swap
    try:
        if "place_norm" in lookup.columns:
            if "v" in norm:
                alt = norm.replace("v", "w")
            elif "w" in norm:
                alt = norm.replace("w", "v")
            else:
                alt = ""
            if alt:
                m = lookup[lookup["place_norm"] == alt]
                if not m.empty:
                    pick = m.iloc[0]
                    district = str(pick.get("district", "")).strip()
                    state = str(pick.get("state", "")).strip()
                    return (district or None), (state or None)
    except Exception:
        pass
    canonical_place = _match_place_from_lookup(place, lookup)
    if canonical_place and canonical_place.strip().lower() != place.strip().lower():
        return _lookup_district_from_location(canonical_place, lookup)
    return None, None



def _set_session_location_context(place: str | None, district: str | None, state: str | None = None) -> None:
    clean_place = str(place or "").strip()
    clean_district = str(district or "").strip()
    clean_state = str(state or "Uttar Pradesh").strip() or "Uttar Pradesh"
    st.session_state["last_location_context"] = {
        "place": clean_place,
        "district": clean_district,
        "state": clean_state,
    }



def _extract_explicit_district_from_query(query: str, lookup: pd.DataFrame) -> tuple[str | None, str | None]:
    if lookup.empty:
        return None, None
    q_tokens = re.findall(r"[a-z0-9]+", str(query or "").lower())
    if not q_tokens:
        return None, None
    q_joined = " ".join(q_tokens)
    best: tuple[int, int, str, str] | None = None
    for col in ("district", "sub_district"):
        if col not in lookup.columns:
            continue
        for raw in lookup[col].dropna().astype(str).unique().tolist():
            raw = raw.strip()
            if not raw or raw.lower() == "nan":
                continue
            raw_tokens = re.findall(r"[a-z0-9]+", raw.lower())
            if not raw_tokens:
                continue
            score = 0
            if len(raw_tokens) == 1:
                if raw_tokens[0] in q_tokens:
                    score = 1
            else:
                joined = " ".join(raw_tokens)
                if joined in q_joined:
                    score = len(raw_tokens) + 1
            if score <= 0:
                continue
            match = lookup[lookup[col].astype(str).str.strip().str.lower() == raw.lower()]
            if match.empty:
                continue
            picked = match.iloc[0]
            district = str(picked.get("district", "")).strip()
            state = str(picked.get("state", "")).strip()
            cand = (score, len(raw_tokens), district or raw, state)
            if best is None or cand[:2] > best[:2]:
                best = cand
    if best is None:
        return None, None
    return best[2] or None, best[3] or None


def _extract_location_search_hint(query: str) -> str | None:
    q = strip_relative_location(query)
    if not q:
        return None
    stop = {
        "like", "kya", "ky", "what", "which", "kitna", "kitne", "kitni",
        "and", "or", "also", "please", "tell", "my", "for", "in", "at", "of", "to", "a", "an", "on", "with", "और", "होगा", "the", "is", "are", "will", "it", "be", "how", "today", "tomorrow", "tonight", "forecast", "next", "week", "day", "days", "rain", "rainfall", "temperature", "kal", "parso", "आज", "कल", "बारिश", "तापमान", "रहेगा", "कैसा", "है",
        "aaj", "aj", "abhi", "ka", "ki", "ke", "ko", "se", "par",
        "me", "mein", "में", "kesa", "kaisa", "hai", "h",
        "pani", "paani", "water", "sinchai", "sichai", "sinchaai", "irrigation",
        "lagta", "lagti", "lagte", "lata", "leti", "chahiye",
        "price", "rate", "mandi", "bhav", "bhaav", "bhao", "daam", "dam",
        "भाव", "कीमत", "मंडी", "मौसम", "mausam", "mosam", "weather", "district", "जिला",
        "btaye", "bataye", "bataiye", "btao", "batao", "boliye", "bolo",
        "do", "de", "dijiye", "dijie", "batayiye", "btaiye", "liye", "liyee",
        "बताएं", "बताये", "बताइए", "बताओ", "दीजिए", "दो",
    }
    alias_path = Path("data/raw/commodity_aliases.json")
    mtime_ns = alias_path.stat().st_mtime_ns if alias_path.exists() else 0
    aliases = load_commodity_aliases(mtime_ns)
    commodity_tokens = {
        token
        for alias_list in aliases.values()
        for alias in alias_list
        for token in re.findall(r"[a-z0-9]+", alias.lower())
    }
    filtered = []
    for tok in q.split():
        cleaned = re.sub(r"[^a-zA-Z0-9\u0900-\u097F]+", "", tok).lower()
        if not cleaned or cleaned in stop or cleaned in commodity_tokens:
            continue
        filtered.append(tok.strip())
    if not filtered:
        return None
    return " ".join(filtered[-3:]).strip() or None


def _suggest_locations_within_scope(hint: str, lookup: pd.DataFrame, limit: int = 3) -> list[str]:
    if not hint or lookup.empty:
        return []
    hint_norm = _normalize_text(hint)
    if not hint_norm:
        return []
    option_map: dict[str, str] = {}
    for col in ["place", "sub_district"]:
        if col not in lookup.columns:
            continue
        for raw in lookup[col].dropna().astype(str).unique().tolist():
            raw = raw.strip()
            if not raw or raw.lower() == "nan":
                continue
            variants = {_normalize_text(raw)}
            token_variants = [token for token in re.findall(r"[a-z0-9]+", raw.lower()) if len(token) >= 4]
            variants.update(token_variants)
            for norm in variants:
                if not norm:
                    continue
                if norm not in option_map or len(raw) < len(option_map[norm]):
                    option_map[norm] = raw
    if not option_map:
        return []
    suggestions: list[str] = []
    for norm, raw in option_map.items():
        if norm.startswith(hint_norm) or hint_norm.startswith(norm):
            if raw not in suggestions:
                suggestions.append(raw)
        if len(suggestions) >= limit:
            return suggestions[:limit]
    close = difflib.get_close_matches(hint_norm, list(option_map.keys()), n=max(limit * 3, 5), cutoff=0.68)
    for norm in close:
        raw = option_map[norm]
        if raw not in suggestions:
            suggestions.append(raw)
        if len(suggestions) >= limit:
            break
    return suggestions[:limit]


def _resolve_query_location_with_selection(
    query: str,
    *,
    selected_state: str | None,
    selected_district: str | None,
    allow_place_lookup: bool = True,
    strict_on_hint: bool = False,
) -> dict[str, object]:
    query = strip_relative_location(query)
    lookup_path = Path("data/processed/location_lookup.csv")
    lookup_mtime = lookup_path.stat().st_mtime_ns if lookup_path.exists() else 0
    lookup = load_location_lookup(lookup_mtime)
    scoped_lookup = _filter_location_lookup_scope(lookup, selected_state, selected_district)
    selected_district_norm = _normalize_district_name(selected_district or "")

    explicit_district, explicit_state = _extract_explicit_district_from_query(query, lookup)
    if explicit_district or explicit_state:
        explicit_norm = _normalize_district_name(explicit_district or "")
        if selected_district_norm and explicit_norm and explicit_norm != selected_district_norm:
            return {
                "status": "outside_scope",
                "requested": explicit_district or query,
                "actual_district": explicit_district or "",
                "district": explicit_district or "",
                "state": explicit_state or "",
                "suggestions": _suggest_locations_within_scope(explicit_district or query, scoped_lookup),
            }
        return {
            "status": "matched",
            "place": explicit_district or None,
            "district": explicit_district or selected_district or None,
            "state": explicit_state or selected_state or None,
        }

    if not allow_place_lookup:
        return {"status": "none"}

    scoped_place = extract_place_from_query(query, scoped_lookup)
    if scoped_place:
        scoped_district, scoped_state = _lookup_district_from_location(scoped_place, scoped_lookup)
        return {
            "status": "matched",
            "place": scoped_place,
            "district": scoped_district or selected_district or None,
            "state": scoped_state or selected_state or None,
        }

    global_place = extract_place_from_query(query, lookup)
    if global_place:
        global_district, global_state = _lookup_district_from_location(global_place, lookup)
        global_norm = _normalize_district_name(global_district or "")
        if selected_district_norm and global_norm and global_norm != selected_district_norm:
            return {
                "status": "outside_scope",
                "requested": global_place,
                "actual_district": global_district or "",
                "district": global_district or "",
                "state": global_state or "",
                "suggestions": _suggest_locations_within_scope(global_place, scoped_lookup),
            }
        return {
            "status": "matched",
            "place": global_place,
            "district": global_district or selected_district or None,
            "state": global_state or selected_state or None,
        }

    hint = _extract_location_search_hint(query)
    if hint:
        suggestions = _suggest_locations_within_scope(hint, scoped_lookup)
        if suggestions or strict_on_hint:
            return {
                "status": "suggest",
                "requested": hint,
                "district": selected_district or "",
                "state": selected_state or "",
                "suggestions": suggestions,
            }
    return {"status": "none"}


def _resolve_query_location(query: str, *, allow_place_lookup: bool = True) -> tuple[str | None, str | None, str | None]:
    query = strip_relative_location(query)
    lookup_path = Path("data/processed/location_lookup.csv")
    lookup_mtime = lookup_path.stat().st_mtime_ns if lookup_path.exists() else 0
    lookup = load_location_lookup(lookup_mtime)
    district, state = _extract_explicit_district_from_query(query, lookup)
    if district or state:
        explicit_place = district or None
        return explicit_place, district, state
    if not allow_place_lookup:
        return None, None, None
    place = extract_place_from_query(query, lookup)
    if not place:
        return None, None, None
    district, state = _lookup_district_from_location(place, lookup)
    return place, district, state



def _coerce_selectbox_state(
    key: str,
    options: list[str],
    preferred: str | None = None,
    *,
    fallback: str | None = None,
) -> int:
    if not options:
        return 0
    current = st.session_state.get(key)
    if current not in options:
        target = None
        for candidate in (preferred, fallback):
            cand = str(candidate or "").strip()
            if cand and cand in options:
                target = cand
                break
        if target is None:
            target = options[0]
        st.session_state[key] = target
        current = target
    return options.index(current)


@lru_cache(maxsize=8)
def load_location_lookup(mtime_ns: int) -> pd.DataFrame:
    _ = mtime_ns
    path = Path("data/processed/location_lookup.csv")
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    for col in ["place", "place_norm", "sub_district", "district", "state"]:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("")
    return df


@lru_cache(maxsize=8)
def load_commodity_aliases(mtime_ns: int) -> dict[str, list[str]]:
    path = Path("data/raw/commodity_aliases.json")
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    cleaned = {}
    for k, v in data.items():
        if isinstance(v, list):
            cleaned[str(k).lower()] = [str(x) for x in v]
    return cleaned


@lru_cache(maxsize=8)
def load_training_feedback_memory(
    db_path: str,
    db_mtime_ns: int,
    feedback_path: str,
    feedback_mtime_ns: int,
) -> list[dict[str, object]]:
    _ = db_mtime_ns
    _ = feedback_mtime_ns
    rows: list[dict[str, object]] = []
    try:
        rows.extend(get_training_feedback_examples(db_path, limit=200))
    except Exception:
        pass
    # Exported datasets are offline snapshots; the reviewed database is authoritative.
    deduped: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (
            str(row.get("user_query") or "").strip().lower(),
            str(row.get("correction_text") or "").strip().lower(),
            str(row.get("topic") or "").strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _feedback_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9\u0900-\u097f]+", (text or "").lower()) if len(t) > 1}


def _feedback_similarity(a: str, b: str) -> float:
    a_tokens = _feedback_tokens(a)
    b_tokens = _feedback_tokens(b)
    jaccard = (len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens))) if (a_tokens and b_tokens) else 0.0
    seq = difflib.SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()
    return max(jaccard, seq)


def find_feedback_memory_hint(
    user_query: str,
    *,
    topic_hint: str | None,
    crop_hint: str | None,
    db_path: str,
    advisor: RAGAdvisor | None = None,
) -> dict[str, object] | None:
    db_file = Path(db_path)
    db_mtime_ns = db_file.stat().st_mtime_ns if db_file.exists() else 0
    feedback_mtime_ns = TRAINING_FEEDBACK_PATH.stat().st_mtime_ns if TRAINING_FEEDBACK_PATH.exists() else 0
    rows = load_training_feedback_memory(
        db_path,
        db_mtime_ns,
        str(TRAINING_FEEDBACK_PATH),
        feedback_mtime_ns,
    )
    if not rows:
        return None
    scored_rows: list[tuple[float, dict[str, object]]] = []
    crop_hint_l = (crop_hint or "").lower().strip()
    for row in rows:
        score = _feedback_similarity(user_query, str(row.get("user_query") or ""))
        row_topic = str(row.get("topic") or "").strip().lower()
        row_crop = str(row.get("crop_name") or "").strip().lower()
        if topic_hint and row_topic == topic_hint.lower():
            score += 0.18
        if crop_hint_l and row_crop and crop_hint_l in row_crop:
            score += 0.18
        if score >= 0.30:
            scored_rows.append((score, row))
    if not scored_rows:
        return None
    scored_rows.sort(key=lambda x: x[0], reverse=True)

    # Semantic rerank on the best lexical candidates using the same local embedder as the RAG path.
    if advisor is not None:
        try:
            advisor._ensure_rag_components(load_generator=False)
            if advisor.embedder is not None:
                top_candidates = scored_rows[:20]
                candidate_texts = [
                    f"{str(row.get('user_query') or '').strip()}\n{str(row.get('correction_text') or '').strip()}".strip()
                    for _, row in top_candidates
                ]
                query_vec = np.asarray(advisor.embedder.encode([user_query])[0], dtype=np.float32)
                candidate_vecs = np.asarray(advisor.embedder.encode(candidate_texts), dtype=np.float32)
                query_norm = float(np.linalg.norm(query_vec)) or 1.0
                candidate_norms = np.linalg.norm(candidate_vecs, axis=1)
                candidate_norms[candidate_norms == 0] = 1.0
                semantic_scores = (candidate_vecs @ query_vec) / (candidate_norms * query_norm)
                best_idx = int(np.argmax(semantic_scores))
                semantic_best = float(semantic_scores[best_idx])
                lexical_best, best_row = top_candidates[best_idx]
                if semantic_best >= 0.42 or lexical_best >= 0.55:
                    return best_row
        except Exception:
            pass

    best_lexical, best_row = scored_rows[0]
    return best_row if best_lexical >= 0.45 else None


def _extract_district_from_feedback_text(text: str, districts: list[str]) -> str | None:
    q = (text or "").lower()
    for d in sorted(districts, key=len, reverse=True):
        dl = d.lower()
        if dl and dl in q:
            return d
    return None


def _feedback_prefers_omit_district(text: str) -> bool:
    t = (text or "").lower()
    patterns = [
        "don't mention any district",
        "do not mention any district",
        "district mat mention",
        "jila mat likho",
        "जिला मत लिखो",
        "कोई जिला मत लिखो",
    ]
    return any(p in t for p in patterns)


def commodity_display_name(commodity: str) -> str:
    alias_path = Path("data/raw/commodity_aliases.json")
    mtime_ns = alias_path.stat().st_mtime_ns if alias_path.exists() else 0
    aliases = load_commodity_aliases(mtime_ns)
    crop = str(commodity).strip()
    crop_norm = re.sub(r"[^a-z0-9]+", "", crop.lower())
    alias_list: list[str] = []
    for key, vals in aliases.items():
        key_norm = re.sub(r"[^a-z0-9]+", "", key.lower())
        if key_norm == crop_norm or crop_norm in key_norm or key_norm in crop_norm:
            alias_list = vals
            break
    hi = None
    hinglish = None
    for alias in alias_list:
        a = str(alias).strip()
        if not a:
            continue
        if hi is None and re.search(r"[\u0900-\u097f]", a):
            hi = a
        if hinglish is None and re.fullmatch(r"[A-Za-z0-9 ()/\\-]+", a):
            a_norm = re.sub(r"[^a-z0-9]+", "", a.lower())
            if a_norm and a_norm != crop_norm and crop_norm not in a_norm and a_norm not in crop_norm:
                hinglish = a
    if hi and hinglish:
        return f"{crop} ({hi} / {hinglish})"
    if hi:
        return f"{crop} ({hi})"
    if hinglish:
        return f"{crop} ({hinglish})"
    return crop


@lru_cache(maxsize=8)
def load_location_corrections(mtime_ns: int) -> dict[str, str]:
    path = Path("data/raw/location_corrections.json")
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    cleaned = {}
    for k, v in data.items():
        if isinstance(v, str):
            cleaned[str(k).lower()] = v
    return cleaned


def save_location_correction(place: str, district: str) -> None:
    path = Path("data/raw/location_corrections.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data[str(place).lower()] = district
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_commodity_from_query(query: str, commodity_list: list[str]) -> str | None:
    from app.commodity_lookup import resolve_commodities
    matches = resolve_commodities(query, commodity_list)
    return matches[0] if len(matches) == 1 else None


def extract_entities_ner(query: str, districts: list[str], commodities: list[str]) -> tuple[str | None, str | None]:
    q = query.lower()
    district = None
    commodity = None

    # District NER: exact token/phrase match
    for d in sorted(districts, key=len, reverse=True):
        if d and re.search(rf"(?<!\\w){re.escape(d.lower())}(?!\\w)", q):
            district = d
            break

    # District fuzzy match (misspellings)
    if not district and districts:
        matches = difflib.get_close_matches(q, districts, n=1, cutoff=0.85)
        if matches:
            district = matches[0]

    # Share the agent's token-safe catalog resolver.
    commodity = resolve_commodity_from_query(query, commodities)

    return district, commodity

LOCATION_DISTRICT_MAP = {}


def extract_selection_from_query(
    query: str,
    df: pd.DataFrame,
    fallback_state: str,
    fallback_district: str,
    fallback_commodity: str,
) -> tuple[str, str, str]:
    q = query.lower()
    state = fallback_state
    district = fallback_district
    commodity = fallback_commodity
    place_provided = False

    if not df.empty:
        states = sorted(df["State"].dropna().astype(str).unique().tolist()) if "State" in df.columns else []
        districts = (
            sorted(df["District"].dropna().astype(str).unique().tolist())
            if "District" in df.columns
            else []
        )
        commodities = (
            sorted(df["Commodity"].dropna().astype(str).unique().tolist())
            if "Commodity" in df.columns
            else []
        )
        st_match = _best_match(q, states)
        if st_match:
            state = st_match
        dist_match, comm_match_ner = extract_entities_ner(query, districts, commodities)
        if dist_match:
            district = dist_match
        # Prefer geocode + reverse-geocode if a place is present
        place = extract_place_from_query(query)
        if place:
            place_provided = True
            # First try fuzzy match against known districts (handles misspellings like Sharanpur)
            if districts:
                matches = difflib.get_close_matches(place, districts, n=1, cutoff=0.8)
                if matches:
                    district = matches[0]
            # Always map via local lookup CSV (deterministic).
            lookup_path = Path("data/processed/location_lookup.csv")
            lookup_mtime = lookup_path.stat().st_mtime_ns if lookup_path.exists() else 0
            lookup = load_location_lookup(lookup_mtime)
            if not lookup.empty:
                resolved_district, resolved_state = _lookup_district_from_location(place, lookup)
                if resolved_district:
                    district = resolved_district
                if resolved_state and resolved_state.lower() != state.lower():
                    state = resolved_state
            if not district:
                st.warning(f"स्थान '{place}' का जिला नहीं मिला। कृपया स्थान या जिला स्पष्ट करें।")
        catalog = load_commodity_catalog()
        comm_match = comm_match_ner or resolve_commodity_from_query(query, catalog or commodities)
        if comm_match:
            commodity = comm_match
        else:
            comm_match = _best_match(q, commodities)
            if comm_match:
                commodity = comm_match

    # If a place was provided but district couldn't be resolved, avoid defaulting.
    if place_provided and not district:
        district = ""
    return state, district, commodity


def summarize_latest_market(df: pd.DataFrame) -> tuple[str, dict[str, str] | None]:
    if df.empty:
        return "चयनित संयोजन के लिए कोई ताज़ा बाजार डेटा नहीं मिला।", None
    df = df.copy()
    df["Arrival_Date_dt"] = pd.to_datetime(df["Arrival_Date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Arrival_Date_dt", "Modal_Price"])
    if df.empty:
        return "चयनित संयोजन के लिए वैध तारीख/दाम उपलब्ध नहीं हैं।", None
    latest = df.loc[df["Arrival_Date_dt"].idxmax()].to_dict()
    latest_dt = latest.get("Arrival_Date_dt")
    modal = latest.get("Modal_Price")
    unit = latest.get("Price_Unit", "Rs./Quintal")
    if pd.isna(unit) or not str(unit).strip() or str(unit).strip().lower() == "nan":
        unit = "Rs./Quintal"
    qty = latest.get("Arrival_Qty", None)
    arrival_unit = latest.get("Arrival_Unit", None)
    if pd.isna(arrival_unit) or not str(arrival_unit).strip() or str(arrival_unit).strip().lower() == "nan":
        arrival_unit = "Metric Tonnes"
    line = f"ताज़ा भाव ({latest_dt.date()}): {modal} {unit}"
    if not pd.isna(qty) and str(qty).strip() and str(qty).strip().lower() != "nan":
        line += f", आवक: {qty} {arrival_unit}"
    return line, latest


def summarize_latest_available_commodity(df: pd.DataFrame, state: str, district: str) -> tuple[str, list[str]]:
    if df.empty:
        return "अभी बाजार डेटा उपलब्ध नहीं है।", []
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]
    out = out[out["State"].astype(str).str.lower() == str(state).lower()]
    out = out[out["District"].astype(str).str.lower() == str(district).lower()]
    out["Arrival_Date_dt"] = pd.to_datetime(out["Arrival_Date"], errors="coerce", dayfirst=True)
    out = out.dropna(subset=["Arrival_Date_dt", "Modal_Price", "Commodity"])
    if out.empty:
        return f"{district} में किसी commodity के लिए वैध ताज़ा डेटा नहीं मिला।", []

    latest_by_commodity = (
        out.sort_values("Arrival_Date_dt", ascending=False)
        .groupby("Commodity", as_index=False)
        .head(1)
        .sort_values(["Arrival_Date_dt", "Modal_Price"], ascending=[False, False])
    )
    top = latest_by_commodity.iloc[0].to_dict()
    top_date = top.get("Arrival_Date_dt")
    top_commodity = commodity_display_name(str(top.get("Commodity") or ""))
    top_price = top.get("Modal_Price")
    top_qty = top.get("Arrival_Qty")
    top_market = str(top.get("Market") or "").strip()
    lines = [
        f"{district} में तारीख के हिसाब से सबसे ताज़ा उपलब्ध commodity: {top_commodity}",
        f"- ताज़ा तारीख: {top_date.date()}",
        f"- भाव: {top_price} {top.get('Price_Unit') or 'Rs./Quintal'}",
    ]
    if top_market:
        lines.append(f"- मंडी: {top_market}")
    if pd.notna(top_qty):
        lines.append(f"- आवक: {top_qty} {top.get('Arrival_Unit') or 'Metric Tonnes'}")

    sample_lines: list[str] = []
    for _, row in latest_by_commodity.head(5).iterrows():
        sample_lines.append(
            f"- {commodity_display_name(str(row.get('Commodity') or ''))}: {row['Arrival_Date_dt'].date()} | {row.get('Modal_Price')} {row.get('Price_Unit') or 'Rs./Quintal'}"
        )
    if sample_lines:
        lines.append("अन्य हाल की उपलब्ध commodities (नमूना):")
        lines.extend(sample_lines)
    return "\n".join(lines), ["agmarknet_report.csv"]




def latest_crop_prices_for_location(
    df: pd.DataFrame,
    state: str,
    district: str,
    limit: int = 20,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = normalize_agmarknet_df(df)
    out.columns = [c.strip() for c in out.columns]
    required = {"State", "District", "Commodity", "Arrival_Date", "Modal_Price"}
    if not required.issubset(out.columns):
        return pd.DataFrame()
    out = out[out["State"].astype(str).str.lower() == str(state).lower()]
    out = out[out["District"].astype(str).str.lower() == str(district).lower()]
    out["Arrival_Date_dt"] = pd.to_datetime(out["Arrival_Date"], errors="coerce", dayfirst=True)
    out = out.dropna(subset=["Arrival_Date_dt", "Modal_Price", "Commodity"])
    if out.empty:
        return pd.DataFrame()
    latest = (
        out.sort_values(["Arrival_Date_dt", "Modal_Price"], ascending=[False, False])
        .groupby("Commodity", as_index=False)
        .head(1)
        .sort_values(["Arrival_Date_dt", "Modal_Price"], ascending=[False, False])
        .head(limit)
        .copy()
    )
    latest["Commodity"] = latest["Commodity"].astype(str).map(commodity_display_name)
    display = pd.DataFrame({
        "फसल": latest["Commodity"],
        "ताज़ा भाव": latest["Modal_Price"],
        "इकाई": latest.get("Price_Unit", pd.Series(["Rs./Quintal"] * len(latest))).replace("", "Rs./Quintal"),
        "तारीख": latest["Arrival_Date_dt"].dt.date.astype(str),
        "मंडी": latest.get("Market", pd.Series([""] * len(latest))).fillna("").astype(str),
    })
    return display.reset_index(drop=True)

def summarize_latest_market_for_market(df: pd.DataFrame, market: str) -> tuple[str, dict[str, str] | None]:
    if df.empty or "Market" not in df.columns:
        return summarize_latest_market(df)
    sub = df[df["Market"].astype(str).str.lower() == market.lower()]
    if sub.empty:
        return summarize_latest_market(df)
    return summarize_latest_market(sub)


def filter_market_rows(
    df: pd.DataFrame,
    commodity: str,
    state: str,
    district: str,
) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]
    out = out[out["Commodity"].astype(str).str.lower() == commodity.lower()]
    out = out[out["State"].astype(str).str.lower() == state.lower()]
    out = out[out["District"].astype(str).str.lower() == district.lower()]
    out["Arrival_Date_dt"] = pd.to_datetime(out["Arrival_Date"], errors="coerce", dayfirst=True)
    out = out.dropna(subset=["Arrival_Date_dt", "Modal_Price"])
    return out


def fetch_live_df(
    api_key: str,
    resource_id: str,
    state: str,
    district: str,
    commodity: str,
    limit: int = FAST_FETCH_LIMIT,
) -> pd.DataFrame:
    client = DataGovClient(api_key=api_key, timeout_sec=25, retries=2)
    params = {}
    if state.strip():
        params["filters[State]"] = state.strip()
    if district.strip():
        params["filters[District]"] = district.strip()
    if commodity.strip():
        params["filters[Commodity]"] = commodity.strip()
    params["sort[Arrival_Date]"] = "desc"
    try:
        recs = client.fetch_records(
            resource_id=resource_id,
            limit=limit,
            max_records=limit,
            extra_params=params,
        )
        return pd.DataFrame(recs) if recs else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def current_user() -> dict | None:
    return st.session_state.get("auth_user")


def get_supabase_config() -> SupabaseConfig | None:
    env_vals = load_local_env(Path(".env"))
    url = get_setting("SUPABASE_URL", env_vals, "").strip()
    anon_key = get_setting("SUPABASE_ANON_KEY", env_vals, "").strip()
    app_base_url = get_setting("APP_BASE_URL", env_vals, "").strip()
    cfg = SupabaseConfig(url=url, anon_key=anon_key, redirect_url=app_base_url)
    return cfg if supabase_is_configured(cfg) else None


def _auth_cookie_secret() -> str:
    env_vals = load_local_env(Path(".env"))
    secret = get_setting("AUTH_COOKIE_SECRET", env_vals)
    if secret:
        return secret
    return "krishiai-dev-secret-change-me"


def _sign_auth_value(raw_value: str) -> str:
    secret = _auth_cookie_secret().encode("utf-8")
    sig = hmac.new(secret, raw_value.encode("utf-8"), hashlib.sha256).hexdigest()
    payload = f"{raw_value}.{sig}"
    return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("utf-8")


def _verify_auth_value(token: str) -> str | None:
    try:
        decoded = base64.urlsafe_b64decode(token.encode("utf-8")).decode("utf-8")
        raw_value, sig = decoded.rsplit(".", 1)
    except Exception:
        return None
    expected = hmac.new(
        _auth_cookie_secret().encode("utf-8"),
        raw_value.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected, sig):
        return None
    return raw_value


def _encode_auth_payload(payload: dict[str, object]) -> str:
    return _sign_auth_value(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))


def _decode_auth_payload(token: str) -> dict[str, object] | None:
    raw = _verify_auth_value(token)
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def set_auth_cookie_payload(payload: dict[str, object]) -> None:
    try:
        st.context.cookies[AUTH_COOKIE_NAME] = _encode_auth_payload(payload)
    except Exception:
        pass


def clear_auth_cookie() -> None:
    try:
        st.context.cookies[AUTH_COOKIE_NAME] = ""
    except Exception:
        pass


def _queue_auth_cookie_write(payload: dict[str, object] | None) -> None:
    if not payload:
        return
    st.session_state["_pending_auth_cookie_payload"] = dict(payload)
    st.session_state.pop("_pending_auth_cookie_armed", None)


def _flush_pending_auth_cookie_write() -> None:
    payload = st.session_state.get("_pending_auth_cookie_payload")
    if not isinstance(payload, dict) or not payload:
        st.session_state.pop("_pending_auth_cookie_armed", None)
        return
    if not st.session_state.get("_pending_auth_cookie_armed"):
        st.session_state["_pending_auth_cookie_armed"] = True
        return
    set_auth_cookie_payload(payload)
    st.session_state.pop("_pending_auth_cookie_payload", None)
    st.session_state.pop("_pending_auth_cookie_armed", None)


def _bootstrap_auth_session(db_path: str) -> None:
    if st.session_state.get("_auth_bootstrap_done"):
        return
    # Avoid forcing a rerun during app startup; bootstrap auth only when the
    # current Streamlit session is already rendering normally.
    handle_email_verification(db_path)
    restore_auth_from_cookie(db_path)
    st.session_state["_auth_bootstrap_done"] = True


def restore_auth_from_cookie(db_path: str) -> None:
    if current_user():
        return
    try:
        token = st.context.cookies.get(AUTH_COOKIE_NAME)
    except Exception:
        token = None
    if not token:
        return
    payload = _decode_auth_payload(str(token))
    if payload and payload.get("provider") == "supabase":
        supa = get_supabase_config()
        if not supa:
            return
        access_token = str(payload.get("access_token") or "")
        refresh_token = str(payload.get("refresh_token") or "")
        ok = False
        session = None
        if access_token:
            ok, session = supabase_get_user(supa, access_token)
            if ok and isinstance(session, dict):
                local_user = _mirror_supabase_user(db_path, session)
                if local_user:
                    st.session_state["auth_user"] = local_user
                    st.session_state["auth_session"] = {
                        "provider": "supabase",
                        "access_token": access_token,
                        "refresh_token": refresh_token,
                    }
                    return
        if refresh_token:
            ok, session = supabase_refresh_session(supa, refresh_token)
            if ok and isinstance(session, dict):
                access_token = str(session.get("access_token") or "")
                refresh_token = str(session.get("refresh_token") or refresh_token)
                user_payload = session.get("user") or {}
                local_user = _mirror_supabase_user(db_path, user_payload)
                if local_user and access_token:
                    st.session_state["auth_user"] = local_user
                    st.session_state["auth_session"] = {
                        "provider": "supabase",
                        "access_token": access_token,
                        "refresh_token": refresh_token,
                    }
                    _queue_auth_cookie_write(st.session_state["auth_session"])
                    return
        return
    raw = _verify_auth_value(str(token))
    if raw and raw.isdigit():
        user = get_user_by_id(db_path, int(raw))
        if user:
            st.session_state["auth_user"] = user


def _mirror_supabase_user(db_path: str, user_payload: dict[str, object] | None) -> dict | None:
    if not isinstance(user_payload, dict):
        return None
    external_id = str(user_payload.get("id") or "").strip()
    email = str(user_payload.get("email") or "").strip().lower()
    metadata = user_payload.get("user_metadata") if isinstance(user_payload.get("user_metadata"), dict) else {}
    username = str(metadata.get("username") or "").strip()
    display_name = str(metadata.get("display_name") or metadata.get("full_name") or "").strip()
    email_confirmed = bool(user_payload.get("email_confirmed_at") or user_payload.get("confirmed_at"))
    if not external_id or not email:
        return None
    try:
        return upsert_external_user(
            db_path,
            provider="supabase",
            external_user_id=external_id,
            email=email,
            display_name=display_name,
            username=username,
            is_verified=email_confirmed or True,
        )
    except Exception:
        return None


def build_verification_link(token: str) -> str | None:
    env_vals = load_local_env(Path(".env"))
    base_url = get_setting("APP_BASE_URL", env_vals)
    if not base_url:
        return None
    sep = "&" if "?" in base_url else "?"
    return f"{base_url}{sep}verify_token={token}"


def send_verification_email(email: str, token: str) -> tuple[bool, str]:
    env_vals = load_local_env(Path(".env"))
    smtp_host = get_setting("SMTP_HOST", env_vals)
    smtp_port = int(get_setting("SMTP_PORT", env_vals, "587"))
    smtp_user = get_setting("SMTP_USER", env_vals)
    smtp_pass = get_setting("SMTP_PASS", env_vals)
    smtp_from = get_setting("SMTP_FROM", env_vals, smtp_user)
    link = build_verification_link(token)
    if not smtp_host or not smtp_user or not smtp_pass or not smtp_from:
        return False, "SMTP is not configured."
    if not link:
        return False, "APP_BASE_URL is not configured."
    msg = EmailMessage()
    msg["Subject"] = "Verify your KisaanAI account"
    msg["From"] = smtp_from
    msg["To"] = email
    msg.set_content(
        "Namaste,\n\n"
        "Please verify your KisaanAI account before signing in.\n\n"
        f"Verification link: {link}\n\n"
        "If you did not create this account, you can ignore this email.\n"
    )
    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        return True, "Verification email sent."
    except Exception as exc:
        return False, f"Could not send verification email: {exc}"


def send_password_reset_otp_email(email: str, otp: str) -> tuple[bool, str]:
    env_vals = load_local_env(Path(".env"))
    smtp_host = get_setting("SMTP_HOST", env_vals)
    smtp_port = int(get_setting("SMTP_PORT", env_vals, "587"))
    smtp_user = get_setting("SMTP_USER", env_vals)
    smtp_pass = get_setting("SMTP_PASS", env_vals)
    smtp_from = get_setting("SMTP_FROM", env_vals, smtp_user)
    if not smtp_host or not smtp_user or not smtp_pass or not smtp_from:
        return False, "SMTP is not configured."
    msg = EmailMessage()
    msg["Subject"] = "KisaanAI password reset OTP"
    msg["From"] = smtp_from
    msg["To"] = email
    msg.set_content(
        "Namaste,\n\n"
        "Use this OTP to reset your KisaanAI password:\n\n"
        f"{otp}\n\n"
        "This OTP is valid for 15 minutes.\n"
        "If you did not request a password reset, please ignore this email.\n"
    )
    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        return True, "Password reset OTP sent."
    except Exception as exc:
        return False, f"Could not send password reset OTP: {exc}"


def _set_signed_in_user(local_user: dict, session: dict[str, object] | None = None) -> None:
    st.session_state["auth_user"] = local_user
    if session and session.get("provider") == "supabase":
        st.session_state["auth_session"] = session
        _queue_auth_cookie_write(session)


def handle_email_verification(db_path: str) -> None:
    try:
        token = st.query_params.get("verify_token")
    except Exception:
        token = None
    if not token:
        return
    ok, msg = verify_user_by_token(db_path, str(token))
    if ok:
        st.session_state["auth_notice"] = ("success", f"Email verified for {msg}. You can now sign in.")
        st.session_state["auth_mode_state"] = "Sign In"
    else:
        st.session_state["auth_notice"] = ("error", msg)
        st.session_state["auth_mode_state"] = "Create Account"
    try:
        st.query_params.clear()
    except Exception:
        pass


def render_auth_sidebar(db_path: str) -> None:
    _bootstrap_auth_session(db_path)
    st.subheader("Account Access")
    smtp_status = smtp_config_status()
    supabase_cfg = get_supabase_config()
    use_supabase = supabase_cfg is not None
    user = current_user()
    if user:
        st.success(f"Signed in as {user.get('display_name')}")
        st.caption(f"Role: {user.get('role', 'user')}")
        if st.button("Sign Out", use_container_width=True):
            auth_session = st.session_state.get("auth_session") or {}
            if use_supabase and auth_session.get("provider") == "supabase" and auth_session.get("access_token"):
                try:
                    supabase_sign_out(supabase_cfg, str(auth_session.get("access_token")))
                except Exception:
                    pass
            clear_auth_cookie()
            for key in ("auth_user", "auth_session", "chat_history", "last_structured_topic", "last_structured_context", "last_location_context"):
                st.session_state.pop(key, None)
            st.rerun()
        return

    if "auth_mode_state" not in st.session_state:
        st.session_state["auth_mode_state"] = "Sign In"
    auth_mode = st.radio(
        "Access",
        ["Sign In", "Create Account"],
        horizontal=True,
        key="auth_mode_widget",
        index=0 if st.session_state.get("auth_mode_state") == "Sign In" else 1,
    )
    if auth_mode != st.session_state.get("auth_mode_state"):
        st.session_state["auth_mode_state"] = auth_mode
    notice = st.session_state.get("auth_notice")
    if notice:
        level, text = notice
        getattr(st, level if level in {"success", "warning", "error", "info"} else "info")(text)
    if auth_mode == "Create Account":
        with st.form("create_account_form", clear_on_submit=False):
            username = st.text_input("Username", key="auth_username")
            password = st.text_input("Password", type="password", key="auth_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="auth_confirm_password")
            display_name = st.text_input("Display name", key="auth_display_name")
            email = st.text_input("Email", key="auth_email")
            submitted = st.form_submit_button("Create Account", use_container_width=True)
        if submitted:
            if password != confirm_password:
                st.session_state["auth_notice"] = ("error", "Password and confirm password do not match.")
                st.rerun()
            if use_supabase:
                ok, payload = supabase_sign_up(
                    supabase_cfg,
                    email=(email or "").strip().lower(),
                    password=password,
                    username=(username or "").strip(),
                    display_name=(display_name or "").strip() or (username or "").strip(),
                )
                st.session_state["auth_mode_state"] = "Sign In" if ok else "Create Account"
                if ok:
                    st.session_state["auth_notice"] = (
                        "success",
                        "Account created in Supabase. Please verify your email before signing in.",
                    )
                    st.session_state.pop("auth_verification_link", None)
                else:
                    st.session_state["auth_notice"] = (
                        "warning" if payload.get("code") == "client_timeout" else "error",
                        str(payload.get("msg") or payload.get("error_description") or payload.get("message") or "Could not create account in Supabase."),
                    )
                st.rerun()
            else:
                ok, msg = create_user(db_path, username, password, display_name, email)
                if ok:
                    payload = msg if isinstance(msg, dict) else {}
                    token = str(payload.get("verification_token") or "")
                    sent, send_msg = send_verification_email(str(payload.get("email") or email), token)
                    st.session_state["auth_mode_state"] = "Sign In"
                    if sent:
                        st.session_state["auth_notice"] = (
                            "success",
                            "Account created. Verification link has been sent to your email. Please verify before signing in.",
                        )
                        st.session_state.pop("auth_verification_link", None)
                    else:
                        fallback_link = build_verification_link(token)
                        st.session_state["auth_notice"] = (
                            "warning",
                            f"Account created, but email could not be sent yet. {send_msg}",
                        )
                        if fallback_link:
                            st.session_state["auth_verification_link"] = fallback_link
                    st.rerun()
                else:
                    st.session_state["auth_mode_state"] = "Create Account"
                    st.session_state["auth_notice"] = ("error", str(msg))
                    st.rerun()
    else:
        with st.form("sign_in_form", clear_on_submit=False):
            username = st.text_input("Email" if use_supabase else "Username or Email", key="auth_username")
            password = st.text_input("Password", type="password", key="auth_password")
            submitted = st.form_submit_button("Sign In", use_container_width=True)
        if submitted:
            if use_supabase:
                ok, payload = supabase_sign_in_with_password(
                    supabase_cfg,
                    email=(username or "").strip().lower(),
                    password=password,
                )
                if not ok:
                    msg = str(payload.get("msg") or payload.get("error_description") or payload.get("message") or "")
                    if "Email not confirmed" in msg:
                        st.session_state["auth_notice"] = ("warning", "Please verify your email before signing in.")
                    else:
                        st.session_state["auth_notice"] = ("error", msg or "Invalid email or password.")
                else:
                    user_payload = payload.get("user") or {}
                    local_user = _mirror_supabase_user(db_path, user_payload)
                    if not local_user:
                        st.session_state["auth_notice"] = ("error", "Could not prepare your local profile after Supabase sign-in.")
                    else:
                        _set_signed_in_user(
                            local_user,
                            {
                                "provider": "supabase",
                                "access_token": str(payload.get("access_token") or ""),
                                "refresh_token": str(payload.get("refresh_token") or ""),
                            },
                        )
                        st.rerun()
            else:
                status, user = authenticate_user_status(db_path, username, password)
                if status == "invalid" or not user:
                    st.session_state["auth_notice"] = ("error", "Invalid username/email or password.")
                elif status == "unverified":
                    st.session_state["auth_mode_state"] = "Sign In"
                    st.session_state["auth_notice"] = ("warning", "Please verify your email before signing in.")
                else:
                    _set_signed_in_user(user, {"provider": "local", "user_id": str(user["id"])})
                    st.rerun()
            st.rerun()
    verification_link = st.session_state.get("auth_verification_link")
    if not use_supabase and auth_mode == "Create Account" and verification_link:
        st.caption("Verification link preview")
        st.code(verification_link)
    with st.expander("Resend Verification Email", expanded=False):
        resend_email = st.text_input("Email for verification", key="resend_verify_email")
        if st.button("Resend Verification Link", key="resend_verify_btn", use_container_width=True):
            if use_supabase:
                ok, payload = supabase_resend_signup_email(supabase_cfg, resend_email.strip().lower())
                if ok:
                    st.success("Verification link sent.")
                else:
                    st.warning(str(payload.get("msg") or payload.get("error_description") or payload.get("message") or "Could not resend verification email."))
            else:
                ok, payload = set_verification_token_for_email(db_path, resend_email)
                if not ok:
                    st.warning(str(payload))
                else:
                    data = payload if isinstance(payload, dict) else {}
                    token = str(data.get("verification_token") or "")
                    sent, send_msg = send_verification_email(str(data.get("email") or resend_email), token)
                    if sent:
                        st.success("Verification link sent.")
                        st.session_state.pop("auth_verification_link", None)
                    else:
                        fallback_link = build_verification_link(token)
                        st.warning(send_msg)
                        if fallback_link:
                            st.code(fallback_link)
    with st.expander("Forgot Password", expanded=False):
        reset_email = st.text_input("Registered email", key="forgot_email")
        if st.button("Send reset email" if use_supabase else "Send OTP", key="send_reset_otp", use_container_width=True):
            if use_supabase:
                ok, payload = supabase_send_password_reset_email(supabase_cfg, reset_email.strip().lower())
                if ok:
                    st.success("Password reset email sent. Please use the link from your inbox.")
                else:
                    st.warning(str(payload.get("msg") or payload.get("error_description") or payload.get("message") or "Could not send password reset email."))
            else:
                ok, payload = create_password_reset_otp(db_path, reset_email)
                if not ok:
                    st.warning(str(payload))
                else:
                    data = payload if isinstance(payload, dict) else {}
                    sent, send_msg = send_password_reset_otp_email(str(data.get("email") or reset_email), str(data.get("otp") or ""))
                    if sent:
                        st.success("OTP sent to your email.")
                        st.session_state["reset_email_active"] = str(data.get("email") or reset_email).strip().lower()
                        st.session_state.pop("reset_otp_preview", None)
                    else:
                        st.warning(send_msg)
                        st.session_state["reset_email_active"] = str(data.get("email") or reset_email).strip().lower()
                        st.session_state["reset_otp_preview"] = str(data.get("otp") or "")
            active_reset_email = st.session_state.get("reset_email_active", "")
            if active_reset_email:
                st.caption(f"Resetting password for: {active_reset_email}")
                otp = st.text_input("OTP", key="reset_otp")
                new_password = st.text_input("New password", type="password", key="reset_new_password")
                confirm_password = st.text_input("Confirm new password", type="password", key="reset_confirm_password")
                if st.button("Update Password", key="update_password_btn", use_container_width=True):
                    if new_password != confirm_password:
                        st.error("Passwords do not match.")
                    else:
                        ok, msg = reset_password_with_otp(db_path, active_reset_email, otp, new_password)
                        if ok:
                            st.success(msg)
                            st.session_state["auth_mode_state"] = "Sign In"
                            st.session_state.pop("reset_email_active", None)
                            st.session_state.pop("reset_otp_preview", None)
                        else:
                            st.error(msg)
            otp_preview = st.session_state.get("reset_otp_preview")
            if otp_preview:
                st.caption("OTP preview")
                st.code(otp_preview)
    with st.expander("Auth Diagnostics", expanded=False):
        if use_supabase:
            st.write("Auth provider: Supabase")
            st.write(f"SUPABASE_URL configured: {'Yes' if bool(supabase_cfg.url) else 'No'}")
            st.write(f"SUPABASE_ANON_KEY configured: {'Yes' if bool(supabase_cfg.anon_key) else 'No'}")
            st.write(f"APP_BASE_URL configured: {'Yes' if bool(supabase_cfg.redirect_url) else 'No'}")
            if supabase_cfg.redirect_url:
                st.caption(f"APP_BASE_URL: {supabase_cfg.redirect_url}")
        else:
            st.write("Auth provider: Local + SMTP")
            st.write(f"APP_BASE_URL configured: {'Yes' if smtp_status['app_base_url'] else 'No'}")
            st.write(f"SMTP host configured: {'Yes' if smtp_status['smtp_host'] else 'No'}")
            st.write(f"SMTP user configured: {'Yes' if smtp_status['smtp_user'] else 'No'}")
            st.write(f"SMTP password configured: {'Yes' if smtp_status['smtp_pass'] else 'No'}")
            st.write(f"SMTP from configured: {'Yes' if smtp_status['smtp_from'] else 'No'}")
            if smtp_status["app_base_url"]:
                st.caption(f"APP_BASE_URL: {smtp_status['app_base_url_value']}")
            if smtp_status["smtp_host"]:
                st.caption(f"SMTP_HOST: {smtp_status['smtp_host_value']}")
            if smtp_status["smtp_from"]:
                st.caption(f"SMTP_FROM: {smtp_status['smtp_from_value']}")
            test_email = st.text_input("Send test email to", key="smtp_test_email")
            if st.button("Send Test Email", key="send_test_email_btn", use_container_width=True):
                if not test_email.strip():
                    st.warning("Enter an email address first.")
                else:
                    sent, msg = send_password_reset_otp_email(test_email.strip(), "123456")
                    if sent:
                        st.success("Test email sent.")
                    else:
                        st.error(msg)
    st.caption("Corrections are validated against local sources before they are reused for future tuning.")


def render_admin_feedback_queue(db_path: str, *, use_sidebar: bool = True) -> None:
    user = current_user() or {}
    if user.get("role") != "admin":
        return
    queue = get_feedback_queue(db_path, limit=12)
    target = st.sidebar if use_sidebar else st
    with target.expander("Feedback Review Queue", expanded=False):
        if not queue:
            st.write("No feedback waiting for review.")
            return
        for item in queue:
            st.markdown(f"**#{item['id']} · {item.get('username') or 'user'} · {item.get('validation_status')}**")
            st.caption(str(item.get("topic") or ""))
            st.write(f"Q: {item.get('user_query') or ''}")
            st.write("Original answer:")
            st.write(item.get("answer_text") or "")
            corrected_answer = st.text_area(
                "Reviewed replacement answer (complete answer, not an instruction)",
                value=item.get("correction_text") or "",
                key=f"reviewed_answer_{item['id']}",
                max_chars=2500,
            )
            st.caption("Verify facts and sources before accepting. Pesticide advice requires agricultural expertise.")
            notes = item.get("validation_notes")
            if notes:
                st.caption(notes)
            evidence_text = compact_evidence_text(item.get("evidence_json") or [])
            if evidence_text:
                st.code(evidence_text)
            c1, c2 = st.columns(2)
            if c1.button("Accept", key=f"fb_accept_{item['id']}", use_container_width=True):
                try:
                    review_feedback(db_path, int(item["id"]), int(user["id"]), "accepted", True,
                                    correction_text=corrected_answer)
                except ValueError as exc:
                    st.error(str(exc))
                    continue
                export_training_feedback(db_path, TRAINING_FEEDBACK_PATH)
                st.rerun()
            if c2.button("Reject", key=f"fb_reject_{item['id']}", use_container_width=True):
                review_feedback(db_path, int(item["id"]), int(user["id"]), "rejected", False)
                export_training_feedback(db_path, TRAINING_FEEDBACK_PATH)
                st.rerun()
            st.markdown("---")


def log_query_answer(
    user_query: str,
    composed_query: str,
    topic: str,
    answer_text: str,
    references: list[str],
    district: str,
    season: str,
    crop_name: str,
) -> int | None:
    user = current_user()
    if not user:
        return None
    return create_query_log(
        cfg.paths["sqlite_db"],
        {
            "user_id": user.get("id"),
            "session_id": st.session_state.get("session_id"),
            "user_query": user_query,
            "composed_query": composed_query,
            "topic": topic,
            "answer_text": answer_text,
            "references": references,
            "district": district,
            "season": season,
            "crop_name": crop_name,
        },
    )


def log_app_exception(stage: str, exc: Exception, user_query: str = "") -> str:
    timestamp = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S %Z")
    log_path = Path("logs/app_runtime_errors.log")
    parts = [
        f"[{timestamp}] stage={stage}",
        f"query={user_query.strip()}",
        f"error={type(exc).__name__}: {exc}",
        traceback.format_exc().strip(),
        "-" * 80,
    ]
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(parts) + "\n")
    except Exception:
        pass
    return timestamp


def render_feedback_widget(item: dict, advisor: RAGAdvisor) -> None:
    if item.get("role") != "assistant" or not item.get("query_log_id"):
        return
    topic = str(item.get("topic") or "").strip().lower()
    user = current_user()
    if not user:
        return
    query_log_id = int(item["query_log_id"])
    if feedback_exists(cfg.paths["sqlite_db"], query_log_id, int(user["id"])):
        st.caption("Feedback saved for this answer.")
        return
    with st.expander("इस उत्तर पर प्रतिक्रिया दें / Give feedback", expanded=False):
        rating = st.radio(
            "Was this answer helpful?",
            ["Helpful", "Not helpful", "Provide correction"],
            key=f"rating_{query_log_id}",
            horizontal=True,
        )
        correction = ""
        improvement = st.selectbox(
            "किस बात पर प्रतिक्रिया है?",
            ["सामान्य उपयोगिता", "हिंदी और अनुवाद", "जानकारी की शुद्धता", "छूटी हुई जानकारी", "उत्तर की लंबाई / प्रस्तुति"],
            key=f"feedback_area_{query_log_id}",
        )
        if rating in {"Not helpful", "Provide correction"}:
            correction = st.text_area(
                "What should the answer say instead?",
                key=f"correction_{query_log_id}",
                placeholder="Write the corrected answer, missing fact, or better explanation.",
                height=180,
            )
            st.caption("Please write the corrected version or the missing source-backed detail.")
        submitted = st.button("Submit feedback", key=f"submit_feedback_{query_log_id}", use_container_width=True)
        if submitted:
            if rating == "Provide correction" and not correction.strip():
                st.warning("Please add the correction before submitting feedback.")
                return
            payload = validate_feedback_with_local_sources(
                advisor=advisor,
                question=str(item.get("user_query") or ""),
                answer=str(item.get("text") or ""),
                correction=correction,
                topic=topic,
                references=item.get("references", []),
                rating=rating.lower().replace(" ", "_"),
            )
            payload["notes"] = f"Feedback area: {improvement}. " + payload["notes"]
            feedback_id = save_feedback(
                cfg.paths["sqlite_db"],
                {
                    "query_log_id": query_log_id,
                    "user_id": user.get("id"),
                    "rating": rating.lower().replace(" ", "_"),
                    "correction_text": correction.strip(),
                    "validation_status": payload["status"],
                    "validation_method": payload["method"],
                    "validation_notes": payload["notes"],
                    "evidence": payload["evidence"],
                    "guardrail_flags": payload["guardrail_flags"],
                    "is_training_eligible": payload["training_eligible"],
                },
            )
            export_training_feedback(cfg.paths["sqlite_db"], TRAINING_FEEDBACK_PATH)
            if feedback_id is None:
                st.error("Could not save feedback.")
            elif payload["status"] == "source_matched":
                st.success("Feedback saved with source evidence. Human review is required before learning from it.")
            elif payload["status"] == "rejected_guardrail":
                st.warning("Feedback was stored but blocked from training because it contains unsafe or sensitive content.")
            else:
                st.info("Feedback saved for review. We will validate it before using it to improve future answers.")
            st.rerun()


def save_advisory(
    farmer_id: str,
    district: str,
    season: str,
    crop_name: str,
    recommendation_text: str,
) -> bool:
    try:
        conn = get_conn(cfg.paths["sqlite_db"])
    except (sqlite3.Error, OSError):
        return False
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO advisories (
                farmer_id, district, season, crop_name,
                recommendation_text, confidence
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (farmer_id, district, season, crop_name, recommendation_text, 0.5),
        )
        conn.commit()
        return True
    except (sqlite3.Error, OSError):
        return False
    finally:
        conn.close()


ok, status_msg = check_ready()
if not ok:
    st.error(status_msg)
    st.info(
        "Run: `python scripts/init_db.py`, `python scripts/load_seed_data.py`, "
        "`python scripts/ingest_documents.py --input_dir data/raw`, `python scripts/build_index.py`"
    )
    st.stop()
init_db(cfg.paths["sqlite_db"])

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

agmarknet_auto_started, agmarknet_auto_reason = _start_agmarknet_auto_refresh()
if agmarknet_auto_started:
    st.session_state["agmarknet_auto_refresh_notice"] = (
        "Agmarknet auto-refresh started in background for the last 14 days."
    )

# Apply pending sidebar selection from last query (if any)
pending = st.session_state.pop("pending_selection", None)
if isinstance(pending, dict):
    if "fc_state" not in st.session_state:
        st.session_state["fc_state"] = pending.get("state", "Uttar Pradesh")
    if "fc_district" not in st.session_state:
        st.session_state["fc_district"] = pending.get("district", "Meerut")
    if "fc_commodity_override" not in st.session_state:
        st.session_state["fc_commodity_override"] = pending.get("commodity", "")

agmarknet_status = _load_agmarknet_refresh_status()
latest_report_date = _latest_agmarknet_report_date(AGMARKNET_CSV)
_bootstrap_auth_session(cfg.paths["sqlite_db"])
auth_user_snapshot = current_user()

st.markdown(
    """
    <div class="kisaan-hero">
        <div class="kisaan-hero-eyebrow">District-First Advisory</div>
        <h2 class="kisaan-hero-title">Choose a district once, then ask village or town specific questions inside it.</h2>
        <p class="kisaan-hero-copy">
            Village and town names are now constrained to the selected district. Use the district price action for a clean crop-price view.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
farmer_id = str((current_user() or {}).get("username") or "FARMER_DEMO")
preferred_crop = ""
env_vals = load_local_env(Path(".env"))
api_key = get_setting("DATA_GOV_API_KEY", env_vals)
resource_id = get_setting("DATA_GOV_RESOURCE_ID", env_vals, "35985678-0d79-46b4-9ed6-6f13308a1d24")
session_location_defaults = st.session_state.get("last_location_context", {}) or {}
session_state_default = str(session_location_defaults.get("state") or "Uttar Pradesh").strip() or "Uttar Pradesh"
session_district_default = str(session_location_defaults.get("district") or "Meerut").strip() or "Meerut"

_catalog_df = load_agmarknet_catalog()
if not _catalog_df.empty:
    _init_df = _catalog_df.copy()
elif LIVE_MARKET_CSV.exists():
    try:
        _init_df = pd.read_csv(LIVE_MARKET_CSV)
        _init_df.columns = [c.strip() for c in _init_df.columns]
    except Exception:
        _init_df = pd.DataFrame(columns=["State", "District", "Commodity"])
else:
    _init_df = pd.DataFrame(columns=["State", "District", "Commodity"])

state_options = (
    sorted(_init_df["State"].dropna().astype(str).unique().tolist())
    if "State" in _init_df.columns
    else []
)
lookup_path = Path("data/processed/location_lookup.csv")
hierarchy = load_location_lookup(lookup_path.stat().st_mtime_ns if lookup_path.exists() else 0)
if not hierarchy.empty and "state" in hierarchy:
    state_options = sorted(set(state_options) | set(hierarchy["state"].dropna().astype(str)))
if not state_options:
    state_options = ["Uttar Pradesh"]

if "show_local_prices_panel" not in st.session_state:
    st.session_state["show_local_prices_panel"] = False
st.session_state.pop("fc_planning_season", None)

toolbar_cols = st.columns([1.05, 1.05, 1.8, 0.9, 0.9, 0.9])
with toolbar_cols[4]:
    st.caption("Data")
    with st.popover("Agmarknet Status"):
        if st.session_state.get("agmarknet_auto_refresh_notice"):
            st.caption(str(st.session_state.get("agmarknet_auto_refresh_notice")))
        st.caption(f"Latest report date: {latest_report_date.isoformat() if latest_report_date else 'Unavailable'}")
        if agmarknet_status:
            st.caption(f"Last refresh status: {agmarknet_status.get('status', 'unknown')}")
            if agmarknet_status.get("updated_at"):
                st.caption(f"Last refresh time: {agmarknet_status.get('updated_at')}")
            if agmarknet_status.get("message"):
                st.caption(str(agmarknet_status.get("message")))
with toolbar_cols[5]:
    st.caption("Account")
    account_label = (
        f"{str(auth_user_snapshot.get('display_name') or auth_user_snapshot.get('username') or 'Account').strip()}"
        if auth_user_snapshot
        else "Sign in"
    )
    with st.popover(account_label):
        render_auth_sidebar(cfg.paths["sqlite_db"])
        render_admin_feedback_queue(cfg.paths["sqlite_db"], use_sidebar=False)

if not current_user():
    st.info("Use the Account menu in the top bar to sign in and start using the assistant.")
    st.stop()

loc_col1, loc_col2, town_col, loc_col3 = toolbar_cols[:4]
with loc_col1:
    st.caption("State")
    state_default = _coerce_selectbox_state(
        "fc_state",
        state_options,
        session_state_default,
        fallback="Uttar Pradesh",
    )
    selected_state = st.selectbox(
        "State",
        state_options,
        index=state_default,
        key="fc_state",
        label_visibility="collapsed",
    )

if ("State" in _init_df.columns and "District" in _init_df.columns):
    district_options = sorted(
        _init_df[_init_df["State"].astype(str) == selected_state]["District"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
else:
    district_options = []
if not hierarchy.empty and {"state", "district"}.issubset(hierarchy.columns):
    hierarchy_districts = hierarchy[hierarchy["state"].astype(str).str.casefold() == selected_state.casefold()]["district"].dropna().astype(str)
    district_options = sorted(set(district_options) | set(hierarchy_districts))
if not district_options:
    district_options = ["Meerut"]

with loc_col2:
    st.caption("District")
    district_default = _coerce_selectbox_state(
        "fc_district",
        district_options,
        session_district_default,
        fallback="Meerut",
    )
    selected_district = st.selectbox(
        "District",
        district_options,
        index=district_default,
        key="fc_district",
        label_visibility="collapsed",
    )

active_state = selected_state
active_district = selected_district
district = active_district
with town_col:
    active_location = render_place_selector(active_state, active_district)
active_place = str(active_location.get("place") or "")
sync_chat_location(st.session_state, active_location)


with loc_col3:
    st.caption("Local Prices")
    if st.button("Show Crop Prices", key="show_local_crop_prices", use_container_width=True):
        st.session_state["show_local_prices_panel"] = True

st.caption(f"Selected location: {qualified_place(active_location)}")
if active_place:
    st.caption("Weather uses the selected place or a labelled regional fallback. Prices are reported by mandi; district markets are shown if no matching town market exists.")

if (
    "State" in _init_df.columns
    and "District" in _init_df.columns
    and "Commodity" in _init_df.columns
):
    temp = _init_df.copy()
    temp = temp[temp["State"].astype(str) == selected_state]
    temp = temp[temp["District"].astype(str) == selected_district]
    commodity_options = sorted(temp["Commodity"].dropna().astype(str).unique().tolist())
else:
    commodity_options = []
if not commodity_options:
    commodity_options = [preferred_crop or "Wheat"]
commodity_fallback = preferred_crop or commodity_options[0]
# Crop choice comes from the question/conversation, not an advanced filter.
st.session_state.pop("fc_commodity_dropdown", None)
season = ""  # Only a season explicitly asked for (or a follow-up) narrows chat advice.
remembered_commodity = (st.session_state.get("last_structured_context") or {}).get("preferred_crop", "")
active_commodity = remembered_commodity if remembered_commodity in commodity_options else commodity_fallback

local_market_df = load_agmarknet_df()
if local_market_df.empty and LIVE_MARKET_CSV.exists():
    try:
        live_mtime_ns = LIVE_MARKET_CSV.stat().st_mtime_ns
        local_market_df = normalize_agmarknet_df(load_market_df(str(LIVE_MARKET_CSV), live_mtime_ns))
    except Exception:
        local_market_df = pd.DataFrame()

if st.session_state.get("show_local_prices_panel"):
    scoped_market_df, price_scope = scope_market_rows(local_market_df, active_location)
    st.caption(market_scope_caption(active_location, price_scope))
    local_prices = latest_crop_prices_for_location(scoped_market_df, active_state, active_district, limit=20)
    if local_prices.empty:
        st.info(f"{active_district} के लिए अभी ताज़ा फसल-भाव सूची उपलब्ध नहीं मिली।")
    else:
        st.markdown(f"**{active_district} में फसलों के ताज़ा भाव**")
        st.dataframe(local_prices, use_container_width=True, height=420)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_structured_topic" not in st.session_state:
    st.session_state.last_structured_topic = None
if "last_structured_context" not in st.session_state:
    st.session_state.last_structured_context = {}
if "last_location_context" not in st.session_state:
    st.session_state.last_location_context = {}
if "pending_chat_items" in st.session_state:
    pending_items = st.session_state.pop("pending_chat_items") or []
    st.session_state.chat_history.extend(pending_items)

try:
    advisor = get_advisor(APP_BUILD_VERSION)
except Exception as exc:
    error_ts = log_app_exception("advisor_init", exc)
    st.error(
        "KisaanAI startup failed while initializing the advisor. "
        f"Please retry in a minute. Logged at {error_ts}."
    )
    with st.expander("Technical details"):
        st.exception(exc)
    st.stop()

for item in st.session_state.chat_history:
    with st.chat_message(item["role"]):
        if item["role"] == "assistant" and item.get("type") != "market_panel":
            from app.translation_status import render_translation_status
            render_translation_status(st, item.get("translation"))
        if item.get("agent_trace"):
            with st.expander("Plan and agent activity"):
                st.write(item["agent_trace"]["goal"])
                st.write(" → ".join(item["agent_trace"]["plan"]))
                st.json(item["agent_trace"]["decisions"])
        if item.get("type") == "market_panel":
            render_market_panel(
                meta=item.get("market_meta") or {},
                auto_chart=item.get("market_chart"),
                auto_table=item.get("market_table"),
            )
        elif item.get("role") == "assistant" and str(item.get("topic") or "").strip().lower() == "weather":
            render_weather_chat_card(item["text"], action=item.get("weather_action"))
            refs = item.get("references", [])
            if refs:
                with st.expander("Sources Used"):
                    for src in refs:
                        st.write(f"- {src}")
        else:
            st.write(item["text"])
            refs = item.get("references", [])
            if refs:
                with st.expander("Sources Used"):
                    for src in refs:
                        st.write(f"- {src}")
        render_feedback_widget(item, advisor)

from app.chat_controls import render_chat_composer
# Reserve output above the composer, including replies generated on this rerun.
response_area = st.container()
user_query = render_chat_composer(st)

if user_query:
    with response_area:
        try:
            st.session_state.chat_history.append({"role": "user", "text": user_query})
            with st.chat_message("user"):
                st.write(user_query)

            # Fast path: user says answer is incorrect -> ask for correction details, no greeting/LLM.
            if user_query.strip().lower() in {"this is incorrect", "incorrect", "गलत", "गलत है", "sahi nahi"}:
                msg = "कृपया सही जिला/फसल/बजट लिखें ताकि मैं सही उत्तर दे सकूँ।"
                query_log_id = log_query_answer(
                    user_query=user_query,
                    composed_query=user_query,
                    topic="clarification",
                    answer_text=msg,
                    references=[],
                    district=district,
                    season=season,
                    crop_name=preferred_crop or "unknown",
                )
                st.session_state.chat_history.append({"role": "assistant", "text": msg, "references": []})
                with st.chat_message("assistant"):
                    st.write(msg)
                st.stop()

            normalized_user_query = advisor._normalize_hinglish(user_query)
            weather_intent = advisor._is_weather_intent(normalized_user_query)
            detected_query_crop = advisor._extract_crop_from_query(normalized_user_query) or ""
            intent_msp = is_msp_query(user_query)
            intent_price = is_price_query(user_query) or intent_msp
            crop_protect_followup_checker = getattr(advisor, "_is_crop_protection_followup_intent", None)
            crop_guide_followup_checker = getattr(advisor, "_is_crop_guide_followup_intent", None)
            if callable(crop_protect_followup_checker):
                crop_protection_followup = bool(crop_protect_followup_checker(normalized_user_query))
            else:
                fallback_pesticide_intent = getattr(advisor, "_is_pesticide_intent", None)
                crop_protection_followup = bool(callable(fallback_pesticide_intent) and fallback_pesticide_intent(normalized_user_query))
            if callable(crop_guide_followup_checker):
                crop_guide_followup_detected = bool(crop_guide_followup_checker(normalized_user_query))
            else:
                crop_guide_followup_detected = False

            # If the previous response asked only for a weather location, consume this input
            # only when it plausibly looks like a location reply. Otherwise continue with
            # normal routing so agri follow-up questions are not hijacked into weather.
            pending_weather_request = st.session_state.pop("pending_weather_location", None)
            if pending_weather_request and not NEWS_INTENT.search(user_query):
                place_guess = extract_place_from_query(user_query)
                looks_like_location_reply = bool(
                    place_guess
                    or advisor._looks_like_location_only(user_query)
                    or (
                        weather_intent
                        and not crop_guide_followup_detected
                        and not crop_protection_followup
                    )
                )
                if not looks_like_location_reply and (crop_guide_followup_detected or crop_protection_followup or advisor._has_agri_intent(normalized_user_query)):
                    pending_weather_request = None
                else:
                    place = place_guess or user_query.strip()
                    lookup_path = Path("data/processed/location_lookup.csv")
                    lookup_mtime = lookup_path.stat().st_mtime_ns if lookup_path.exists() else 0
                    lookup = load_location_lookup(lookup_mtime)
                    district, _state = _lookup_district_from_location(place, lookup)
                    composed_weather_query = place if not district else f"{place}, {district}, Uttar Pradesh"
                    original_weather_query = (
                        pending_weather_request.get("original_query", "")
                        if isinstance(pending_weather_request, dict)
                        else ""
                    )
                    normalized_weather_query = advisor._normalize_hinglish(original_weather_query or user_query)
                    weather_request = advisor._parse_weather_request(original_weather_query or user_query, normalized_weather_query)
                    if weather_request:
                        weather_result = advisor._answer_weather_request(
                            weather_request,
                            original_weather_query or user_query,
                            normalized_weather_query,
                            place_override=place,
                        )
                        weather = weather_result.get("answer", "")
                        weather_references = weather_result.get("references", ["Open-Meteo API"])
                        weather_action = weather_result.get("weather_action", weather_request.action)
                    else:
                        weather = get_current_weather_hindi(composed_weather_query)
                        if not weather:
                            weather = get_current_weather_hindi(composed_weather_query)
                        weather_references = ["Open-Meteo API"]
                        weather_action = "current"
                    final_answer = (
                        weather
                        if weather
                        else "अभी लाइव मौसम डेटा नहीं मिल पाया। कृपया कुछ देर बाद फिर प्रयास करें।"
                    )
                    query_log_id = log_query_answer(
                        user_query=user_query,
                        composed_query=composed_weather_query,
                        topic="weather",
                        answer_text=final_answer,
                        references=weather_references,
                        district=district or "",
                        season=season,
                        crop_name=preferred_crop or "unknown",
                    )
                    _set_session_location_context(place, district or "", "Uttar Pradesh")
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "text": final_answer,
                            "references": weather_references,
                            "query_log_id": query_log_id,
                            "topic": "weather",
                            "user_query": user_query,
                            "weather_action": weather_action,
                        }
                    )
                    with st.chat_message("assistant"):
                        render_weather_chat_card(final_answer, action=weather_action)
                        render_feedback_widget(st.session_state.chat_history[-1], advisor)
                    st.stop()

            last_ctx = st.session_state.get("last_structured_context", {}) or {}
            last_location_ctx = st.session_state.get("last_location_context", {}) or {}
            session_state_hint = (last_location_ctx.get("state") or "Uttar Pradesh").strip() or "Uttar Pradesh"
            session_district_hint = (last_location_ctx.get("district") or last_ctx.get("district") or district or "Meerut").strip() or "Meerut"
            query_place = query_place_district = query_place_state = None
            location_scope_result: dict[str, object] = {"status": "none"}
            scope_state = active_state if "active_state" in locals() else session_state_hint
            scope_district = active_district if "active_district" in locals() else session_district_hint
            if not NEWS_INTENT.search(user_query) and not crop_guide_followup_detected and not crop_protection_followup:
                location_scope_result = _resolve_query_location_with_selection(
                    user_query,
                    selected_state=scope_state,
                    selected_district=scope_district,
                    allow_place_lookup=bool(weather_intent or intent_price or not detected_query_crop),
                    strict_on_hint=bool(weather_intent),
                )
                if location_scope_result.get("status") == "matched":
                    query_place = location_scope_result.get("place") or None
                    query_place_district = location_scope_result.get("district") or None
                    query_place_state = location_scope_result.get("state") or None

            location_scope_status = str(location_scope_result.get("status") or "none")
            if location_scope_status in {"outside_scope", "suggest"}:
                requested_location = str(location_scope_result.get("requested") or "यह स्थान").strip() or "यह स्थान"
                suggestions = [
                    str(item).strip()
                    for item in (location_scope_result.get("suggestions") or [])
                    if str(item).strip()
                ]
                actual_district = str(location_scope_result.get("actual_district") or "").strip()
                if location_scope_status == "outside_scope":
                    lines = [f"'{requested_location}' चयनित जिला {scope_district} में नहीं मिला।"]
                    if actual_district and _normalize_district_name(actual_district) != _normalize_district_name(scope_district):
                        lines.append(f"यह स्थान {actual_district} जिले से जुड़ा दिख रहा है।")
                else:
                    lines = [f"चयनित जिला {scope_district} में '{requested_location}' का exact match नहीं मिला।"]
                if location_scope_status == "suggest" and suggestions:
                    lines.append(f"क्या आपका मतलब: {', '.join(suggestions[:3])}?")
                lines.append(f"कृपया {scope_district} के गांव/कस्बे का नाम लिखें या ऊपर जिला बदलें।")
                final_answer = "\n".join(lines)
                query_log_id = log_query_answer(
                    user_query=user_query,
                    composed_query=user_query,
                    topic="location_scope",
                    answer_text=final_answer,
                    references=[],
                    district=scope_district,
                    season=season,
                    crop_name=detected_query_crop or preferred_crop or "unknown",
                )
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "text": final_answer,
                        "references": [],
                        "query_log_id": query_log_id,
                        "topic": "location_scope",
                        "user_query": user_query,
                    }
                )
                with st.chat_message("assistant"):
                    st.write(final_answer)
                st.stop()
            if query_place and (query_place_district or query_place_state):
                _set_session_location_context(
                    query_place,
                    query_place_district or session_district_hint,
                    query_place_state or session_state_hint,
                )
                last_location_ctx = st.session_state.get("last_location_context", {}) or {}
                session_state_hint = (last_location_ctx.get("state") or session_state_hint).strip() or "Uttar Pradesh"
                session_district_hint = (last_location_ctx.get("district") or session_district_hint).strip() or "Meerut"

            followup_profit = (
                is_profitability_followup_query(user_query)
                and st.session_state.get("last_structured_topic") in {"crop_profitability", "crop_profitability_followup"}
            )
            followup_crop_care = (
                crop_protection_followup
                and st.session_state.get("last_structured_topic") in {"crop_guide", "crop_guide_followup", "pesticide"}
            )
            followup_crop_guide = (
                crop_guide_followup_detected
                and st.session_state.get("last_structured_topic") in {"crop_guide", "crop_guide_followup", "pesticide"}
            )
            remembered_crop_context = bool((last_ctx.get("preferred_crop") or "").strip())
            can_use_pesticide_followup_context = (
                crop_protection_followup
                and (
                    st.session_state.get("last_structured_topic") in {"crop_guide", "crop_guide_followup", "pesticide"}
                    or remembered_crop_context
                )
            )
            explicit_query_season = _extract_season_from_query_text(user_query, advisor)

            # Resolve place->district for crop intent (so profit uses correct district)
            resolved_district = session_district_hint
            if is_crop_query(user_query):
                if query_place_district:
                    resolved_district = query_place_district
            elif followup_profit and last_ctx.get("district"):
                resolved_district = last_ctx["district"]

            season_for_query = (
                explicit_query_season
                or (last_ctx.get("season", season) if (followup_profit or followup_crop_care or followup_crop_guide) else season)
            )
            preferred_crop_for_query = (
                last_ctx.get("preferred_crop", preferred_crop) if (followup_profit or followup_crop_care or followup_crop_guide) else preferred_crop
            )

            question_for_advisor = user_query.strip()
            fallback_weather_place = str(
                last_location_ctx.get("place")
                or (active_district if "active_district" in locals() else "")
                or session_district_hint
            ).strip()
            if weather_intent and not (query_place or query_place_district or query_place_state) and fallback_weather_place:
                question_for_advisor = f"{fallback_weather_place} में {question_for_advisor}"

            query_location = dict(active_location)
            if query_place:
                if str(query_place).casefold() == str(active_district).casefold():
                    query_location.update(place="", sub_district="")
                elif str(query_place).casefold() != str(active_place).casefold():
                    matches = [loc for loc in place_options(active_state, active_district).values()
                               if str(loc["place"]).casefold() == str(query_place).casefold()]
                    query_location = matches[0] if len(matches) == 1 else dict(active_location, place=query_place, sub_district="")
            resolved_district = active_district
            composed_query = (
                location_context(query_location)
                + f"जिला: {resolved_district} | मौसम: {season_for_query} | पसंदीदा फसल: {preferred_crop_for_query or 'कोई नहीं'} | "
                f"किसान का प्रश्न: {question_for_advisor}"
            )

            explicit_place = bool(extract_place_from_query(user_query))
            explicit_district = False
            query_crop_hint = detected_query_crop or preferred_crop_for_query or ""
            feedback_hint = None
            market_df = pd.DataFrame()
            if intent_price:
                market_df = load_agmarknet_df()
                feedback_hint = find_feedback_memory_hint(
                    user_query,
                    topic_hint="price",
                    crop_hint=query_crop_hint,
                    db_path=cfg.paths["sqlite_db"],
                    advisor=advisor,
                )
                if not market_df.empty and "District" in market_df.columns:
                    known_districts = sorted(market_df["District"].dropna().astype(str).unique().tolist())
                    explicit_district = bool(extract_entities_ner(user_query, known_districts, [])[0])
            feedback_district_hint = ""
            feedback_prefers_omit_district = False
            if feedback_hint:
                feedback_prefers_omit_district = _feedback_prefers_omit_district(str(feedback_hint.get("correction_text") or ""))
                if not market_df.empty and "District" in market_df.columns:
                    feedback_district_hint = _extract_district_from_feedback_text(
                        str(feedback_hint.get("correction_text") or ""),
                        sorted(market_df["District"].dropna().astype(str).unique().tolist()),
                    ) or ""
            fallback_state_for_price = session_state_hint or (active_state if "active_state" in locals() else "Uttar Pradesh")
            fallback_district_for_price = (
                session_district_hint
                or feedback_district_hint
                or (active_district if "active_district" in locals() else district)
            )
            selected_state, selected_district, selected_commodity = extract_selection_from_query(
                user_query,
                market_df,
                fallback_state=fallback_state_for_price,
                fallback_district=fallback_district_for_price,
                fallback_commodity=active_commodity if "active_commodity" in locals() else (preferred_crop or "Wheat"),
            )
            if intent_price and not explicit_place and not explicit_district and fallback_district_for_price:
                selected_district = fallback_district_for_price
                if fallback_state_for_price:
                    selected_state = fallback_state_for_price

            agentic_enabled = os.getenv("KISAANAI_AGENTIC", "1").lower() not in {"0", "false", "no"}
            if agentic_enabled or NEWS_INTENT.search(user_query):
                # Every query, including prices and mixed intents, uses the coordinator.
                intent_price = False

            if intent_price:
                # Ensure commodity is explicitly detected for price queries.
                comm_from_query = resolve_commodity_from_query(
                    user_query, load_commodity_catalog()
                )
                if not comm_from_query and is_latest_commodity_price_query(user_query):
                    final_answer, price_references = summarize_latest_available_commodity(
                        market_df,
                        selected_state,
                        selected_district,
                    )
                    query_log_id = log_query_answer(
                        user_query=user_query,
                        composed_query=composed_query,
                        topic="price",
                        answer_text=final_answer,
                        references=price_references,
                        district=selected_district,
                        season=season,
                        crop_name="latest_available_commodity",
                    )
                    st.session_state.chat_history.append(
                        {"role": "assistant", "text": final_answer, "references": price_references, "query_log_id": query_log_id, "topic": "price", "user_query": user_query}
                    )
                    with st.chat_message("assistant"):
                        st.write(final_answer)
                    st.stop()
                if not comm_from_query:
                    final_answer = (
                        "कृपया फसल/कमोडिटी का नाम बताएं (जैसे: गेहूं, गन्ना, धान)।"
                        if not intent_msp
                        else "कृपया जिस फसल का MSP चाहिए उसका नाम बताएं (जैसे: गेहूं, धान, चना)।"
                    )
                    query_log_id = log_query_answer(
                        user_query=user_query,
                        composed_query=composed_query,
                        topic="price",
                        answer_text=final_answer,
                        references=[],
                        district=selected_district,
                        season=season,
                        crop_name="unknown",
                    )
                    st.session_state.chat_history.append(
                        {"role": "assistant", "text": final_answer, "references": [], "query_log_id": query_log_id, "topic": "price", "user_query": user_query}
                    )
                    with st.chat_message("assistant"):
                        st.write(final_answer)
                    st.stop()
                if intent_msp:
                    selected_commodity = comm_from_query or selected_commodity
                    commodity_label = commodity_display_name(selected_commodity)
                    msp = get_msp_for_crop(selected_commodity)
                    if msp:
                        final_answer = (
                            f"{commodity_label} के लिए MSP (राष्ट्रीय): ₹{int(msp['msp'])}/क्विंटल.\n"
                            f"स्रोत: {msp['source_url']}"
                        )
                    else:
                        final_answer = (
                            f"{commodity_label} के लिए अभी MSP रिकॉर्ड उपलब्ध नहीं मिला। "
                            "कृपया फसल का नाम दोबारा लिखें या दूसरी फसल पूछें।"
                        )
                    query_log_id = log_query_answer(
                        user_query=user_query,
                        composed_query=composed_query,
                        topic="price",
                        answer_text=final_answer,
                        references=[],
                        district=selected_district,
                        season=season,
                        crop_name=selected_commodity or "unknown",
                    )
                    st.session_state.chat_history.append(
                        {"role": "assistant", "text": final_answer, "references": [], "query_log_id": query_log_id, "topic": "price", "user_query": user_query}
                    )
                    with st.chat_message("assistant"):
                        st.write(final_answer)
                    st.stop()
                if market_df.empty:
                    selected_commodity = comm_from_query or selected_commodity
                    commodity_label = commodity_display_name(selected_commodity)
                    msp = get_msp_for_crop(selected_commodity)
                    if msp:
                        final_answer = (
                            f"{commodity_label} के लिए MSP (राष्ट्रीय): ₹{int(msp['msp'])}/क्विंटल.\n"
                            f"स्रोत: {msp['source_url']}"
                        )
                    else:
                        final_answer = (
                            f"{commodity_label} के लिए अभी मंडी/MSP डेटा उपलब्ध नहीं मिला। "
                            "कृपया थोड़ी देर बाद फिर प्रयास करें।"
                        )
                    query_log_id = log_query_answer(
                        user_query=user_query,
                        composed_query=composed_query,
                        topic="price",
                        answer_text=final_answer,
                        references=[],
                        district=selected_district,
                        season=season,
                        crop_name=selected_commodity or "unknown",
                    )
                    st.session_state.chat_history.append(
                        {"role": "assistant", "text": final_answer, "references": [], "query_log_id": query_log_id, "topic": "price", "user_query": user_query}
                    )
                    with st.chat_message("assistant"):
                        st.write(final_answer)
                    st.stop()
                if selected_district and (query_place or explicit_place or explicit_district):
                    _set_session_location_context(
                        query_place,
                        selected_district,
                        selected_state or session_state_hint,
                    )
                if not selected_district:
                    st.session_state.pop("auto_chart", None)
                    st.session_state.pop("auto_forecast_table", None)
                    st.session_state.pop("auto_forecast_caption", None)
                    final_answer = (
                        "स्थान का जिला ऑटो‑मैप नहीं हो पाया। "
                        "कृपया सही जिला बताएं, ताकि अगली बार अपने‑आप सही जिला चुना जा सके।"
                    )
                    query_log_id = log_query_answer(
                        user_query=user_query,
                        composed_query=composed_query,
                        topic="price",
                        answer_text=final_answer,
                        references=[],
                        district="",
                        season=season,
                        crop_name=selected_commodity or "unknown",
                    )
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "text": final_answer,
                            "references": [],
                            "query_log_id": query_log_id,
                            "topic": "price",
                            "user_query": user_query,
                        }
                    )
                    with st.chat_message("assistant"):
                        st.write(final_answer)
                    st.session_state["need_location_correction"] = True
                    st.stop()
                # Use the commodity resolved from query (avoid fallback to unrelated commodity)
                selected_commodity = comm_from_query or selected_commodity
                commodity_label = commodity_display_name(selected_commodity)
                filtered = filter_market_rows(market_df, selected_commodity, selected_state, selected_district)
                mention_district_in_price_answer = bool(selected_district and (explicit_place or explicit_district))
                price_references: list[str] = []
                if filtered.empty:
                    if selected_commodity.lower() in {"sugarcane", "गन्ना"}:
                        sugarcane_price = get_sugarcane_price_fallback()
                        price = sugarcane_price.get("price")
                        season = sugarcane_price.get("season", "")
                        source_name = sugarcane_price.get("source", "")
                        src = sugarcane_price.get("source_url", "")
                        if price:
                            intro = (
                                f"चयनित जिले ({selected_district}) में {commodity_label} का मंडी डेटा उपलब्ध नहीं है।\n"
                                if mention_district_in_price_answer
                                else ""
                            )
                            final_answer = (
                                f"{intro}"
                                f"गन्ना के लिए {source_name} {season}: ₹{int(float(price))}/क्विंटल."
                            )
                            if src:
                                final_answer += f"\nस्रोत: {src}"
                        else:
                            intro = (
                                f"चयनित जिले ({selected_district}) में {commodity_label} का मंडी डेटा उपलब्ध नहीं है। "
                                if mention_district_in_price_answer
                                else ""
                            )
                            final_answer = f"{intro}CACP से FRP निकालने में समस्या आई।"
                    else:
                        msp = get_msp_for_crop(selected_commodity)
                        if msp:
                            intro = (
                                f"चयनित जिले ({selected_district}) में {commodity_label} का मंडी डेटा नहीं मिला।\n"
                                if mention_district_in_price_answer
                                else ""
                            )
                            final_answer = f"{intro}MSP (राष्ट्रीय) {msp['crop']}: ₹{int(msp['msp'])}/क्विंटल.\nस्रोत: {msp['source_url']}"
                        else:
                            web_answer, price_references = _try_price_web_fallback(
                                advisor,
                                user_query=user_query,
                                commodity_label=commodity_label,
                                commodity_key=selected_commodity,
                                selected_state=selected_state,
                                selected_district=selected_district,
                                mention_district=mention_district_in_price_answer,
                            )
                            if web_answer:
                                final_answer = web_answer
                            else:
                                commodity_key = re.sub(r"[^a-z0-9]+", "", str(selected_commodity).lower())
                                final_answer = _format_specialty_crop_price_unavailable(
                                    commodity_label,
                                    commodity_key,
                                    selected_district,
                                    mention_district_in_price_answer,
                                )
                    query_log_id = log_query_answer(
                        user_query=user_query,
                        composed_query=composed_query,
                        topic="price",
                        answer_text=final_answer,
                        references=price_references,
                        district=selected_district,
                        season=season,
                        crop_name=selected_commodity or "unknown",
                    )
                    st.session_state.chat_history.append(
                        {"role": "assistant", "text": final_answer, "references": price_references, "query_log_id": query_log_id, "topic": "price", "user_query": user_query}
                    )
                    with st.chat_message("assistant"):
                        st.write(final_answer)
                    st.stop()
                nearest_market = None
                if not filtered.empty and "Market" in filtered.columns and filtered["Market"].notna().any():
                    place = extract_place_from_query(user_query)
                    if place:
                        place_geo = _geocode_cached(place, f"{selected_district}, Uttar Pradesh")
                        if place_geo:
                            plat, plon, _ = place_geo
                            markets = (
                                filtered["Market"].dropna().astype(str).unique().tolist()
                                if "Market" in filtered.columns
                                else []
                            )
                            best = None
                            for m in markets[:30]:
                                geo = _geocode_cached(m, f"{selected_district}, Uttar Pradesh")
                                if not geo:
                                    continue
                                lat, lon, label = geo
                                dist = haversine_km(plat, plon, lat, lon)
                                if best is None or dist < best[0]:
                                    best = (dist, label, m)
                            if best:
                                nearest_market = best

                forecast_df = filtered
                if nearest_market:
                    latest_line, _latest = summarize_latest_market_for_market(filtered, nearest_market[2])
                    forecast_df = filtered[
                        filtered["Market"].astype(str).str.lower() == nearest_market[2].lower()
                    ]
                else:
                    latest_line, _latest = summarize_latest_market(filtered)
                auto_caption = f"{selected_state} / {selected_district} / {commodity_label}"
                detailed_forecast_requested = wants_detailed_price_forecast(user_query)
                auto_chart = None
                auto_table = None
                try:
                    if detailed_forecast_requested:
                        hist, fc = build_forecast_from_df(
                            df=forecast_df,
                            commodity=selected_commodity,
                            state=selected_state,
                            district=selected_district,
                            horizon=15,
                        )
                        history_tail = hist.tail(90).copy()
                        history_tail = history_tail.rename(columns={"value": "History"})
                        fc2 = fc.rename(columns={"predicted_value": "Forecast"})

                        chart_df = pd.DataFrame({"date": pd.to_datetime(history_tail["date"])})
                        chart_df["History"] = history_tail["History"].values
                        chart_df = chart_df.set_index("date")

                        fc_chart = pd.DataFrame({"date": pd.to_datetime(fc2["date"])})
                        fc_chart["Forecast"] = fc2["Forecast"].values
                        fc_chart = fc_chart.set_index("date")
                        auto_chart = chart_df.join(fc_chart, how="outer")
                        auto_table = fc2
                except Exception:
                    auto_chart = None
                    auto_table = None

                market_list = []
                if "Market" in filtered.columns:
                    market_list = (
                        filtered["Market"].dropna().astype(str).unique().tolist()
                    )
                market_answer = (
                    f"बाजार जानकारी ({selected_state} / {selected_district} / {commodity_label}):\n"
                    f"- {latest_line}\n"
                )
                if market_list:
                    sample_markets = ", ".join(sorted(market_list)[:8])
                    market_answer += f"- उपलब्ध मंडियाँ (नमूना): {sample_markets}\n"
                if detailed_forecast_requested and auto_table is not None and not auto_table.empty:
                    next_vals = auto_table["Forecast"].head(7).tolist()
                    vals_str = ", ".join([f"{v:.0f}" for v in next_vals])
                    market_answer += f"- अगले 7 दिन के अनुमानित भाव: {vals_str} Rs./Quintal\n"

                if nearest_market:
                    market_answer += (
                        f"- निकटतम मंडी (लगभग): {nearest_market[1]} ({nearest_market[0]:.1f} km)\n"
                    )

                query_log_id = log_query_answer(
                    user_query=user_query,
                    composed_query=composed_query,
                    topic="price",
                    answer_text=market_answer,
                    references=[],
                    district=selected_district,
                    season=season,
                    crop_name=selected_commodity or "unknown",
                )
                if detailed_forecast_requested and auto_table is not None:
                    market_meta = {
                        "caption": auto_caption,
                        "commodity_label": commodity_label,
                        "latest": _latest or {},
                        "market_list": market_list,
                        "nearest_market": nearest_market,
                    }
                    st.session_state["auto_chart"] = auto_chart
                    st.session_state["auto_forecast_table"] = auto_table
                    st.session_state["auto_forecast_caption"] = auto_caption
                    st.session_state["auto_market_meta"] = market_meta
                    st.session_state["pending_selection"] = {
                        "state": selected_state,
                        "district": selected_district,
                        "commodity": selected_commodity,
                    }
                    pending_chat_items = [
                        {"role": "assistant", "text": market_answer, "references": [], "query_log_id": query_log_id, "topic": "price", "user_query": user_query},
                    ]
                    if detailed_forecast_requested and auto_chart is not None:
                        pending_chat_items.append(
                            {
                                "role": "assistant",
                                "type": "market_panel",
                                "text": "",
                                "references": [],
                                "market_meta": market_meta,
                                "market_chart": auto_chart,
                                "market_table": auto_table,
                            }
                        )
                    st.session_state["pending_chat_items"] = pending_chat_items
                else:
                    st.session_state.pop("auto_chart", None)
                    st.session_state.pop("auto_forecast_table", None)
                    st.session_state.pop("auto_forecast_caption", None)
                    st.session_state.pop("auto_market_meta", None)
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "text": market_answer,
                            "references": [],
                            "query_log_id": query_log_id,
                            "topic": "price",
                            "user_query": user_query,
                        }
                    )
                    with st.chat_message("assistant"):
                        st.write(market_answer)
                        render_feedback_widget(st.session_state.chat_history[-1], advisor)
                    st.stop()

                # Re-render so the market answer and panel appear inline at this chat turn.
                st.rerun()
                final_answer = market_answer
            else:
                query_crop_context = advisor._extract_crop_from_query(advisor._normalize_hinglish(user_query)) or preferred_crop_for_query or ""
                direct_crop_followup = None
                direct_crop_protection_followup = None
                if (
                    not agentic_enabled
                    and not NEWS_INTENT.search(user_query)
                    and (
                        crop_guide_followup_detected
                        or _looks_like_crop_water_followup(user_query, advisor)
                    )
                    and advisor._has_agri_intent(normalized_user_query)
                    and not advisor._is_weather_intent(normalized_user_query)
                ):
                    advisor._ensure_rag_components(load_generator=False)
                    guide_answer, guide_sources = build_crop_production_followup(
                        normalized_user_query,
                        crop_hint=query_crop_context or last_ctx.get("preferred_crop", "") or None,
                        reasoning_generator=advisor.generator,
                    )
                    if guide_answer:
                        direct_crop_followup = {
                            "answer": guide_answer,
                            "references": guide_sources,
                            "retrieved": [],
                            "topic": "crop_guide_followup",
                        }
                if (
                    not agentic_enabled
                    and not NEWS_INTENT.search(user_query)
                    and direct_crop_followup is None
                    and can_use_pesticide_followup_context
                    and not advisor._is_weather_intent(normalized_user_query)
                ):
                    pesticide_result = advisor._structured_pesticide_advice(
                        normalized_user_query,
                        crop_hint=query_crop_context or last_ctx.get("preferred_crop", "") or None,
                    )
                    if pesticide_result and pesticide_result.get("answer"):
                        direct_crop_protection_followup = {
                            "answer": pesticide_result["answer"],
                            "references": pesticide_result.get("references", []),
                            "retrieved": pesticide_result.get("retrieved", []),
                            "topic": "pesticide",
                        }
                if agentic_enabled:
                    with st.spinner("Checking sources and coordinating specialists..."):
                        result = advisor.answer(composed_query)
                elif direct_crop_followup is not None:
                    result = direct_crop_followup
                elif direct_crop_protection_followup is not None:
                    result = direct_crop_protection_followup
                else:
                    with st.spinner("Generating recommendation..."):
                        try:
                            result = advisor.answer(composed_query)
                        except Exception as exc:
                            import traceback as _traceback

                            print(f"[KisaanAI] advisor.answer failed: {exc}")
                            _traceback.print_exc()
                            result = {
                                "answer": (
                                    "अभी उत्तर तैयार करते समय तकनीकी समस्या आई। "
                                    "कृपया सवाल थोड़ा छोटा लिखें या कुछ देर बाद फिर प्रयास करें।"
                                ),
                                "references": [],
                                "retrieved": [],
                                "topic": "clarification",
                            }
                if (
                    not agentic_enabled
                    and not NEWS_INTENT.search(user_query)
                    and str(result.get("topic") or "").strip().lower() == "weather"
                    and (crop_guide_followup_detected or _looks_like_crop_water_followup(user_query, advisor))
                    and advisor._has_agri_intent(normalized_user_query)
                ):
                    advisor._ensure_rag_components(load_generator=False)
                    guide_answer, guide_sources = build_crop_production_followup(
                        normalized_user_query,
                        crop_hint=query_crop_context or last_ctx.get("preferred_crop", "") or None,
                        reasoning_generator=advisor.generator,
                    )
                    if guide_answer:
                        result = {
                            "answer": guide_answer,
                            "references": guide_sources,
                            "retrieved": [],
                            "topic": "crop_guide_followup",
                        }
                if (
                    not agentic_enabled
                    and not NEWS_INTENT.search(user_query)
                    and str(result.get("topic") or "").strip().lower() == "weather"
                    and can_use_pesticide_followup_context
                    and not advisor._is_weather_intent(normalized_user_query)
                ):
                    pesticide_result = advisor._structured_pesticide_advice(
                        normalized_user_query,
                        crop_hint=query_crop_context or last_ctx.get("preferred_crop", "") or None,
                    )
                    if pesticide_result and pesticide_result.get("answer"):
                        result = {
                            "answer": pesticide_result["answer"],
                            "references": pesticide_result.get("references", []),
                            "retrieved": pesticide_result.get("retrieved", []),
                            "topic": "pesticide",
                        }
                from app.hindi_translation import translate_answer
                result = translate_answer(result)
                final_answer = result["answer"]
                topic = result.get("topic") or "rag"
                query_log_id = log_query_answer(
                    user_query=user_query,
                    composed_query=composed_query,
                    topic=topic,
                    answer_text=final_answer,
                    references=result.get("references", []),
                    district=resolved_district,
                    season=season_for_query,
                    crop_name=preferred_crop_for_query or "unknown",
                )
                if topic == "weather" and ("मौसम के लिए स्थान" in final_answer or "कृपया स्थान लिखें" in final_answer):
                    st.session_state["pending_weather_location"] = {"original_query": user_query}
                if topic in {"crop_profitability", "crop_profitability_followup", "crop_guide", "crop_guide_followup"}:
                    st.session_state["last_structured_topic"] = topic
                    st.session_state["last_structured_context"] = {
                        "district": resolved_district,
                        "season": season_for_query,
                        "preferred_crop": query_crop_context,
                    }
                elif topic == "pesticide":
                    st.session_state["last_structured_topic"] = topic
                    st.session_state["last_structured_context"] = {
                        "district": resolved_district,
                        "season": season_for_query,
                        "preferred_crop": query_crop_context or last_ctx.get("preferred_crop", ""),
                    }
                elif topic in {"weather", "rag", "clarification"}:
                    st.session_state["last_structured_topic"] = topic
                    if topic == "weather":
                        place_guess, weather_district, weather_state = _resolve_query_location(user_query)
                        if place_guess or weather_district:
                            _set_session_location_context(
                                place_guess or weather_district,
                                weather_district or last_location_ctx.get("district", ""),
                                weather_state or last_location_ctx.get("state", "Uttar Pradesh") or "Uttar Pradesh",
                            )

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "text": final_answer,
                    "references": ([] if intent_price else result.get("references", [])),
                    "query_log_id": (None if intent_price else query_log_id),
                    "topic": (None if intent_price else topic),
                    "user_query": user_query,
                    "weather_action": (None if intent_price else result.get("weather_action")),
                    "agent_trace": (None if intent_price else result.get("agent_trace")),
                    "translation": (None if intent_price else result.get("translation")),
                }
            )

            with st.chat_message("assistant"):
                if not intent_price and str(topic or "").strip().lower() == "weather":
                    render_weather_chat_card(final_answer, action=result.get("weather_action"))
                else:
                    st.write(final_answer)
                if not intent_price:
                    from app.translation_status import render_translation_status
                    render_translation_status(st, result.get("translation"))
                    if result.get("agent_trace"):
                        with st.expander("Plan and agent activity"):
                            trace = result["agent_trace"]
                            st.write(trace["goal"])
                            st.write(" → ".join(trace["plan"]))
                            st.json(trace["decisions"])
                    with st.expander("Sources Used"):
                        for src in result.get("references", []):
                            st.write(f"- {src}")
                    render_feedback_widget(st.session_state.chat_history[-1], advisor)

            # Correction form only when auto-mapping failed
            if st.session_state.pop("need_location_correction", False):
                with st.expander("सही जिला बताएं (एक बार)"):
                    place_guess = extract_place_from_query(user_query) or ""
                    corr_place = st.text_input("स्थान (Village/Town)", value=place_guess, key="corr_place")
                    corr_district = st.text_input("सही जिला", value="", key="corr_district")
                    if st.button("सुधार सहेजें", use_container_width=True):
                        if corr_place and corr_district:
                            save_location_correction(corr_place, corr_district)
                            clear_local_caches()
                            st.success("सुधार सहेजा गया। अगली बार यही जिला उपयोग होगा।")

            if intent_price:
                st.session_state["pending_selection"] = {
                    "state": selected_state,
                    "district": selected_district,
                    "commodity": selected_commodity,
                }

            save_advisory(
                farmer_id=farmer_id,
                district=district,
                season=season,
                crop_name=preferred_crop or "unknown",
                recommendation_text=final_answer,
            )
        except Exception as exc:
            error_ts = log_app_exception("user_query", exc, user_query=user_query)
            final_answer = (
                "अभी इस सवाल को प्रोसेस करते समय तकनीकी समस्या आई। "
                f"कृपया दोबारा प्रयास करें। Error log time: {error_ts}."
            )
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "text": final_answer,
                    "references": [],
                    "query_log_id": None,
                    "topic": "runtime_error",
                    "user_query": user_query,
                    "weather_action": None,
                }
            )
            with st.chat_message("assistant"):
                st.error(final_answer)
                with st.expander("Technical details"):
                    st.exception(exc)

_flush_pending_auth_cookie_write()
