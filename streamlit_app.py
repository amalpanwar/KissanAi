from __future__ import annotations

import os
import re
import sys
import sqlite3
import smtplib
import uuid
from datetime import date
from time import time
from pathlib import Path
from email.message import EmailMessage

import pandas as pd
import streamlit as st
import json
from urllib.parse import urlencode
from urllib.request import urlopen
import difflib

from app.advisor import AdvisorConfig, RAGAdvisor, WESTERN_UP_CROP_BASELINES
from app.config import load_config
from app.db import (
    authenticate_user,
    authenticate_user_status,
    create_password_reset_otp,
    create_query_log,
    create_user,
    export_training_feedback,
    feedback_exists,
    get_conn,
    get_feedback_queue,
    get_user_by_email,
    init_db,
    review_feedback,
    save_feedback,
    set_verification_token_for_email,
    reset_password_with_otp,
    verify_user_by_token,
)
from app.datagov_client import DataGovClient
from app.feedback import compact_evidence_text, validate_feedback_with_local_sources
from app.lstm_forecast import prepare_daily_series, train_and_forecast
from app.weather import get_current_weather_hindi
from app.cacp import get_latest_sugarcane_frp
from app.msp import get_msp_for_crop


BRAND_IMAGE = Path(
    "data/raw/indian-agriculture-landscape-farmer-working-indian-rice-fields-rural-worker-vector-cartoon-backg_1396-599.avif"
)
PAGE_ICON = str(BRAND_IMAGE) if BRAND_IMAGE.exists() else "🌾"
st.set_page_config(page_title="KrishiAI - Agriculture Assistant", page_icon=PAGE_ICON, layout="wide")
if BRAND_IMAGE.exists():
    st.image(str(BRAND_IMAGE), use_container_width=True)

st.title("KrishiAI - Agriculture Assistant")

cfg = load_config()
LIVE_MARKET_CSV = Path("data/raw/live/datagov_commodity.csv")
AGMARKNET_CSV = Path("data/raw/live/agmarknet_report.csv")
FETCH_PAGE_LIMIT = 200
FETCH_MAX_RECORDS_COMBO = 50000
FETCH_MAX_RECORDS_STATE = 50000
FETCH_COOLDOWN_SEC = 600
FAST_FETCH_LIMIT = 200
TRAINING_FEEDBACK_PATH = Path("data/processed/accepted_feedback.jsonl")


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


@st.cache_resource(show_spinner=False)
def get_advisor() -> RAGAdvisor:
    return RAGAdvisor(
        AdvisorConfig(
            embedding_model=cfg.embedding_model,
            generator_model=cfg.generator_model,
            index_path=cfg.paths["vector_store"],
            metadata_path=cfg.paths["metadata_store"],
            top_k=cfg.top_k,
            db_path=cfg.paths["sqlite_db"],
        )
    )


@st.cache_data(show_spinner=False)
def load_market_df(csv_path: str, mtime_ns: int) -> pd.DataFrame:
    _ = mtime_ns
    return pd.read_csv(csv_path)


@st.cache_data(show_spinner=False)
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
    try:
        df = pd.read_csv(AGMARKNET_CSV)
        return normalize_agmarknet_df(df)
    except Exception:
        return pd.DataFrame()


def is_price_query(text: str) -> bool:
    t = text.lower()
    keywords = [
        "price",
        "rate",
        "mandi",
        "bhav",
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
    ]
    return any(k in t for k in keywords)


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


def extract_place_from_query(query: str) -> str | None:
    # Prefer a direct lookup match from the known location table before falling
    # back to token filtering. This keeps place parsing stable even when the
    # query contains extra words like commodity names or question words.
    q = str(query).strip()
    if not q:
        return None

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
                    if len(token) >= 3 and token in q_tokens:
                        matched = True
                        token_hits = 1
                else:
                    joined = " ".join(raw_tokens)
                    q_joined = " ".join(q_tokens)
                    if joined in q_joined:
                        matched = True
                        token_hits = len(raw_tokens)
                    else:
                        hits = [t for t in raw_tokens if len(t) >= 3 and t in q_tokens]
                        if hits:
                            matched = True
                            token_hits = len(hits)
                if matched:
                    candidates.append((token_hits, len("".join(raw_tokens)), raw))
        if candidates:
            candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
            return candidates[0][2]

    tokens = [t.strip(" ?!.,") for t in q.split() if t.strip()]
    if not tokens:
        return None

    stop = {
        "kya", "ky", "what", "which", "kitna", "kitne", "kitni",
        "aaj", "aj", "abhi", "ka", "ki", "ke", "ko", "se", "par",
        "me", "mein", "में", "kesa", "kaisa", "hai", "h",
        "price", "rate", "mandi", "bhav", "daam", "dam",
        "भाव", "कीमत", "मंडी", "मौसम", "weather",
    }

    alias_path = Path("data/raw/commodity_aliases.json")
    mtime_ns = alias_path.stat().st_mtime_ns if alias_path.exists() else 0
    aliases = load_commodity_aliases(mtime_ns)
    commodity_tokens = set()
    for alias_list in aliases.values():
        for alias in alias_list:
            for t in re.findall(r"[a-z0-9]+", alias.lower()):
                commodity_tokens.add(t)

    filtered = []
    for tok in tokens:
        t = re.sub(r"[^a-zA-Z0-9\u0900-\u097F]+", "", tok).lower()
        if not t or t in stop or t in commodity_tokens:
            continue
        filtered.append(tok)

    while filtered:
        last = re.sub(r"[^a-zA-Z0-9\u0900-\u097F]+", "", filtered[-1]).lower()
        if not last or last in stop or last in commodity_tokens or len(re.sub(r"[^a-z0-9]+", "", last)) <= 1:
            filtered.pop()
        else:
            break
    if not filtered:
        return None
    return " ".join(filtered)


def _place_variants(place: str) -> list[str]:
    base = place.strip()
    variants = [base]
    if base.lower().endswith("e") and len(base) > 3:
        variants.append(base[:-1])
    return variants


@st.cache_data(show_spinner=False)
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
    if "place" in lookup.columns:
        contains = lookup[lookup["place"].astype(str).map(_normalize_text).str.contains(norm, na=False)]
        if not contains.empty:
            up = contains[contains["state"].str.lower() == "uttar pradesh"] if "state" in contains.columns else contains
            pick = up.iloc[0] if not up.empty else contains.iloc[0]
            district = str(pick.get("district", "")).strip()
            state = str(pick.get("state", "")).strip()
            return (district or None), (state or None)
    # Fallback: fuzzy match (handles minor spelling errors like Kurava->Kurawa)
    try:
        import difflib

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
    return None, None


@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner=False)
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
    if not query or not commodity_list:
        return None
    q = query.lower()
    q_tokens = re.findall(r"[a-z0-9]+", q)
    q_norm = " ".join(q_tokens)
    q_hi = query
    alias_path = Path("data/raw/commodity_aliases.json")
    mtime_ns = alias_path.stat().st_mtime_ns if alias_path.exists() else 0
    aliases = load_commodity_aliases(mtime_ns)

    best_fuzzy: tuple[float, str] | None = None
    for eng_name, alias_list in aliases.items():
        for alias in alias_list:
            alias_text = str(alias).strip()
            if not alias_text:
                continue
            a_tokens = re.findall(r"[a-z0-9]+", alias_text.lower())
            matched = False
            if a_tokens:
                if len(a_tokens) == 1:
                    matched = a_tokens[0] in q_tokens
                    if not matched:
                        close = difflib.get_close_matches(a_tokens[0], q_tokens, n=1, cutoff=0.84)
                        if close:
                            score = difflib.SequenceMatcher(None, a_tokens[0], close[0]).ratio()
                            if best_fuzzy is None or score > best_fuzzy[0]:
                                best_fuzzy = (score, eng_name)
                else:
                    matched = " ".join(a_tokens) in q_norm
            else:
                matched = alias_text in q_hi
            if matched:
                for name in commodity_list:
                    if name.lower() == eng_name.lower():
                        return name
                return eng_name.title()

    if best_fuzzy is not None:
        eng_name = best_fuzzy[1]
        for name in commodity_list:
            if name.lower() == eng_name.lower():
                return name
        return eng_name.title()

    for name in commodity_list:
        n_tokens = re.findall(r"[a-z0-9]+", name.lower())
        if not n_tokens:
            continue
        if len(n_tokens) == 1:
            if n_tokens[0] in q_tokens:
                return name
        else:
            n_norm = " ".join(n_tokens)
            if n_norm in q_norm:
                return name
    qn = _normalize_text(q)
    for name in commodity_list:
        if _normalize_text(name) in qn:
            return name
    return None


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

    # Commodity NER: alias map
    alias_path = Path("data/raw/commodity_aliases.json")
    mtime_ns = alias_path.stat().st_mtime_ns if alias_path.exists() else 0
    aliases = load_commodity_aliases(mtime_ns)
    for eng_name, alias_list in aliases.items():
        for alias in sorted(alias_list, key=len, reverse=True):
            if alias and alias.lower() in q:
                for name in commodities:
                    if name.lower() == eng_name.lower():
                        commodity = name
                        return district, commodity
                commodity = eng_name.title()
                return district, commodity

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
    msg["Subject"] = "Verify your KrishiAI account"
    msg["From"] = smtp_from
    msg["To"] = email
    msg.set_content(
        "Namaste,\n\n"
        "Please verify your KrishiAI account before signing in.\n\n"
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
    msg["Subject"] = "KrishiAI password reset OTP"
    msg["From"] = smtp_from
    msg["To"] = email
    msg.set_content(
        "Namaste,\n\n"
        "Use this OTP to reset your KrishiAI password:\n\n"
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


def handle_email_verification(db_path: str) -> None:
    token = st.query_params.get("verify_token")
    if not token:
        return
    ok, msg = verify_user_by_token(db_path, str(token))
    if ok:
        st.session_state["auth_notice"] = ("success", f"Email verified for {msg}. You can now sign in.")
        st.session_state["auth_mode"] = "Sign In"
    else:
        st.session_state["auth_notice"] = ("error", msg)
        st.session_state["auth_mode"] = "Create Account"
    try:
        st.query_params.clear()
    except Exception:
        pass


def render_auth_sidebar(db_path: str) -> None:
    st.subheader("Account Access")
    user = current_user()
    if user:
        st.success(f"Signed in as {user.get('display_name')}")
        st.caption(f"Role: {user.get('role', 'user')}")
        if st.button("Sign Out", use_container_width=True):
            for key in ("auth_user", "chat_history", "last_structured_topic", "last_structured_context", "last_location_context"):
                st.session_state.pop(key, None)
            st.rerun()
        return

    auth_mode = st.radio("Access", ["Sign In", "Create Account"], horizontal=True, key="auth_mode")
    notice = st.session_state.pop("auth_notice", None)
    if notice:
        level, text = notice
        getattr(st, level if level in {"success", "warning", "error", "info"} else "info")(text)
    username = st.text_input("Username", key="auth_username")
    password = st.text_input("Password", type="password", key="auth_password")
    display_name = ""
    email = ""
    if auth_mode == "Create Account":
        display_name = st.text_input("Display name", key="auth_display_name")
        email = st.text_input("Email", key="auth_email")
    if st.button(auth_mode, use_container_width=True):
        if auth_mode == "Create Account":
            ok, msg = create_user(db_path, username, password, display_name, email)
            if ok:
                payload = msg if isinstance(msg, dict) else {}
                token = str(payload.get("verification_token") or "")
                sent, send_msg = send_verification_email(str(payload.get("email") or email), token)
                st.session_state["auth_mode"] = "Sign In"
                st.session_state["auth_password"] = ""
                st.session_state["auth_username"] = username
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
                st.session_state["auth_mode"] = "Create Account"
                st.error(str(msg))
        else:
            status, user = authenticate_user_status(db_path, username, password)
            if status == "invalid" or not user:
                st.error("Invalid username or password.")
            elif status == "unverified":
                st.session_state["auth_mode"] = "Sign In"
                st.session_state["auth_notice"] = ("warning", "Please verify your email before signing in.")
                st.rerun()
            else:
                st.session_state["auth_user"] = user
                st.rerun()
    verification_link = st.session_state.get("auth_verification_link")
    if auth_mode == "Create Account" and verification_link:
        st.caption("Verification link preview")
        st.code(verification_link)
    with st.expander("Resend Verification Email", expanded=False):
        resend_email = st.text_input("Email for verification", key="resend_verify_email")
        if st.button("Resend Verification Link", key="resend_verify_btn", use_container_width=True):
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
        if st.button("Send OTP", key="send_reset_otp", use_container_width=True):
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
                        st.session_state["auth_mode"] = "Sign In"
                        for key in ("reset_email_active", "reset_otp", "reset_new_password", "reset_confirm_password", "reset_otp_preview"):
                            st.session_state.pop(key, None)
                    else:
                        st.error(msg)
        otp_preview = st.session_state.get("reset_otp_preview")
        if otp_preview:
            st.caption("OTP preview")
            st.code(otp_preview)
    st.caption("Corrections are validated against local sources before they are reused for future tuning.")


def render_admin_feedback_queue(db_path: str) -> None:
    user = current_user() or {}
    if user.get("role") != "admin":
        return
    queue = get_feedback_queue(db_path, limit=12)
    with st.sidebar.expander("Feedback Review Queue", expanded=False):
        if not queue:
            st.write("No feedback waiting for review.")
            return
        for item in queue:
            st.markdown(f"**#{item['id']} · {item.get('username') or 'user'} · {item.get('validation_status')}**")
            st.caption(str(item.get("topic") or ""))
            st.write(f"Q: {item.get('user_query') or ''}")
            if item.get("correction_text"):
                st.write(f"Correction: {item['correction_text']}")
            notes = item.get("validation_notes")
            if notes:
                st.caption(notes)
            evidence_text = compact_evidence_text(item.get("evidence_json") or [])
            if evidence_text:
                st.code(evidence_text)
            c1, c2 = st.columns(2)
            if c1.button("Accept", key=f"fb_accept_{item['id']}", use_container_width=True):
                review_feedback(db_path, int(item["id"]), int(user["id"]), "accepted", True)
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


def render_feedback_widget(item: dict, advisor: RAGAdvisor) -> None:
    if item.get("role") != "assistant" or not item.get("query_log_id"):
        return
    user = current_user()
    if not user:
        return
    query_log_id = int(item["query_log_id"])
    if feedback_exists(cfg.paths["sqlite_db"], query_log_id, int(user["id"])):
        st.caption("Feedback saved for this answer.")
        return
    with st.expander("Give feedback on this answer", expanded=False):
        rating = st.radio(
            "Was this answer helpful?",
            ["Helpful", "Not helpful", "Provide correction"],
            key=f"rating_{query_log_id}",
            horizontal=True,
        )
        correction = ""
        if rating in {"Not helpful", "Provide correction"}:
            correction = st.text_area(
                "What should the answer say instead?",
                key=f"correction_{query_log_id}",
                placeholder="Write the corrected answer, missing fact, or better explanation.",
            )
        if st.button("Submit feedback", key=f"submit_feedback_{query_log_id}", use_container_width=True):
            payload = validate_feedback_with_local_sources(
                advisor=advisor,
                question=str(item.get("user_query") or ""),
                answer=str(item.get("text") or ""),
                correction=correction,
                topic=item.get("topic"),
                references=item.get("references", []),
            )
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
                st.success("Feedback saved and matched to local source evidence. It is ready for future tuning.")
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
handle_email_verification(cfg.paths["sqlite_db"])

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

# Apply pending sidebar selection from last query (if any)
pending = st.session_state.pop("pending_selection", None)
if isinstance(pending, dict):
    if "fc_state" not in st.session_state:
        st.session_state["fc_state"] = pending.get("state", "Uttar Pradesh")
    if "fc_district" not in st.session_state:
        st.session_state["fc_district"] = pending.get("district", "Meerut")
    if "fc_commodity_override" not in st.session_state:
        st.session_state["fc_commodity_override"] = pending.get("commodity", "")

with st.sidebar:
    render_auth_sidebar(cfg.paths["sqlite_db"])
    render_admin_feedback_queue(cfg.paths["sqlite_db"])
    if not current_user():
        st.info("Sign in to use chat, save history, and submit corrections that help improve future answers.")
        st.stop()
    st.subheader("Commodity Forecast (15 Days)")
    farmer_id = str((current_user() or {}).get("username") or "FARMER_DEMO")
    season = st.selectbox("Planning Season", ["Rabi", "Kharif", "Annual"])
    preferred_crop = ""
    env_vals = load_local_env(Path(".env"))
    api_key = get_setting("DATA_GOV_API_KEY", env_vals)
    resource_id = get_setting("DATA_GOV_RESOURCE_ID", env_vals, "35985678-0d79-46b4-9ed6-6f13308a1d24")

    _catalog_df = load_agmarknet_df()
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
    if not state_options:
        state_options = ["Uttar Pradesh"]
    state_default = state_options.index("Uttar Pradesh") if "Uttar Pradesh" in state_options else 0
    selected_state = st.selectbox("State", state_options, index=state_default, key="fc_state")
    state_override = st.text_input("State (type override, optional)", value="", key="fc_state_override")

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
    if not district_options:
        district_options = ["Meerut"]
    district_default = district_options.index("Meerut") if "Meerut" in district_options else 0
    selected_district = st.selectbox("District", district_options, index=district_default, key="fc_district")
    district_override = st.text_input(
        "District (type override, optional)",
        value="",
        key="fc_district_override",
    )

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
    commodity_default = commodity_options.index("Wheat") if "Wheat" in commodity_options else 0
    selected_commodity_from_dropdown = st.selectbox(
        "Commodity (from data)",
        commodity_options,
        index=commodity_default,
        key="fc_commodity_dropdown",
    )
    commodity_text_override = st.text_input(
        "Commodity (type override, optional)",
        value="",
        key="fc_commodity_override",
    )
    selected_commodity = commodity_text_override.strip() or selected_commodity_from_dropdown
    active_state = state_override.strip() or selected_state
    active_district = district_override.strip() or selected_district
    active_commodity = selected_commodity
    district = active_district

    min_raw_points = 60
    max_stale_days = 30
    fast_mode = st.checkbox("Fast Forecast (no download)", value=True)

    def run_fetch(use_state_only: bool = False) -> tuple[int, int]:
        if not api_key:
            st.warning("DATA_GOV_API_KEY not found in .env")
            return 0, 0
        if not resource_id:
            st.warning("DATA_GOV_RESOURCE_ID not found in .env")
            return 0, 0

        client = DataGovClient(api_key=api_key, timeout_sec=30, retries=5)
        params = {}
        if active_state.strip():
            params["filters[State]"] = active_state.strip()
        if not use_state_only:
            if active_district.strip():
                params["filters[District]"] = active_district.strip()
            if active_commodity.strip():
                params["filters[Commodity]"] = active_commodity.strip()
        # Favor recent data to reduce payload and timeouts.
        params["sort[Arrival_Date]"] = "desc"
        recs = client.fetch_records(
            resource_id=resource_id,
            limit=FETCH_PAGE_LIMIT,
            max_records=FETCH_MAX_RECORDS_STATE if use_state_only else FETCH_MAX_RECORDS_COMBO,
            extra_params=params,
        )
        if recs:
            LIVE_MARKET_CSV.parent.mkdir(parents=True, exist_ok=True)
            new_df = pd.DataFrame(recs)
            merged = merge_market_data(LIVE_MARKET_CSV, new_df)
            merged.to_csv(LIVE_MARKET_CSV, index=False)
            st.cache_data.clear()
            return len(new_df), len(merged)
        return 0, 0

    if st.button("Refresh Selected Combination", use_container_width=True):
        with st.spinner("Fetching selected combination..."):
            try:
                combo_key = f"{active_state}|{active_district}|{active_commodity}".lower()
                last_ts = st.session_state.get(f"last_fetch_ts::{combo_key}", 0)
                if time() - last_ts < FETCH_COOLDOWN_SEC:
                    st.info("Using recent cached data; skip fetch to avoid timeout.")
                else:
                    fetched, stored = run_fetch(use_state_only=False)
                    st.session_state[f"last_fetch_ts::{combo_key}"] = time()
                    if fetched > 0:
                        st.success(f"Fetched {fetched} rows, stored {stored} unique rows.")
                    else:
                        st.warning("No records returned for selected combination.")
            except Exception as e:
                st.error(f"Fetch timed out/failed: {e}")
                st.info("Retry once, or use 'Refresh State Catalog' first and then narrow the selection.")

    if st.button("Refresh State Catalog", use_container_width=True):
        with st.spinner("Fetching all districts/commodities for selected state..."):
            try:
                state_key = f"last_fetch_state::{active_state}".lower()
                last_ts = st.session_state.get(state_key, 0)
                if time() - last_ts < FETCH_COOLDOWN_SEC:
                    st.info("Using recent cached data; skip fetch to avoid timeout.")
                else:
                    fetched, stored = run_fetch(use_state_only=True)
                    st.session_state[state_key] = time()
                    if fetched > 0:
                        st.success(f"Fetched {fetched} rows, stored {stored} unique rows.")
                    else:
                        st.warning("No records returned for selected state.")
            except Exception as e:
                st.error(f"Fetch timed out/failed: {e}")
                st.info("API is slow right now. Retry after 10-20 seconds.")

    if st.button("Refresh Agmarknet (Last 14 Days)", use_container_width=True):
        with st.spinner("Refreshing Agmarknet data (last 14 days)..."):
            try:
                import subprocess

                cmd = [
                    sys.executable,
                    str(Path("scripts/agmarknet_daily_refresh.py")),
                ]
                env = os.environ.copy()
                env["AGMARKNET_LOOKBACK_DAYS"] = "14"
                result = subprocess.run(cmd, env=env, cwd=str(Path(".")), capture_output=True, text=True)
                if result.returncode != 0:
                    st.error("Agmarknet refresh failed.")
                    st.code(result.stderr or result.stdout)
                else:
                    st.success("Agmarknet refresh completed.")
                    if result.stdout.strip():
                        st.code(result.stdout.strip())
                    st.cache_data.clear()
            except Exception as e:
                st.error(f"Agmarknet refresh failed: {e}")

    if st.button("Show 15-Day Forecast", use_container_width=True):
        try:
            if AGMARKNET_CSV.exists():
                mtime_ns = AGMARKNET_CSV.stat().st_mtime_ns
                mdf = load_market_df(str(AGMARKNET_CSV), mtime_ns)
                mdf = normalize_agmarknet_df(mdf)
            elif fast_mode:
                with st.spinner("Fetching recent data (fast mode)..."):
                    live_df = fetch_live_df(
                        api_key=api_key,
                        resource_id=resource_id,
                        state=active_state,
                        district=active_district,
                        commodity=active_commodity,
                    )
                if live_df.empty:
                    st.warning(
                        "Fast mode fetch timed out or returned no rows. "
                        "Falling back to cached CSV if available."
                    )
                    if LIVE_MARKET_CSV.exists():
                        mtime_ns = LIVE_MARKET_CSV.stat().st_mtime_ns
                        mdf = load_market_df(str(LIVE_MARKET_CSV), mtime_ns)
                    else:
                        st.error("No cached CSV available. Use Refresh Selected Combination.")
                        st.stop()
                else:
                    mdf = live_df
            else:
                if not LIVE_MARKET_CSV.exists():
                    st.info("No live commodity CSV found. Run data fetch script first.")
                    st.stop()
                mtime_ns = LIVE_MARKET_CSV.stat().st_mtime_ns
                mdf = load_market_df(str(LIVE_MARKET_CSV), mtime_ns)
                if mdf.empty:
                    st.info("Live commodity data file is empty.")
                    st.stop()

            mdf.columns = [c.strip() for c in mdf.columns]
            filtered = filter_market_rows(mdf, active_commodity, active_state, active_district)
            raw_points = int(filtered["Arrival_Date_dt"].dt.date.nunique()) if not filtered.empty else 0
            latest_dt = filtered["Arrival_Date_dt"].max().date() if raw_points > 0 else None
            st.caption(
                "Selection: "
                f"{active_state} / {active_district} / {active_commodity} | "
                f"Raw points: {raw_points} | Latest arrival date: {latest_dt if latest_dt else 'N/A'}"
            )

            if raw_points == 0:
                st.error(
                    "No rows found for selected state/district/commodity. "
                    "Click 'Refresh Selected Combination' first."
                )
            elif raw_points < int(min_raw_points):
                st.warning(
                    f"कम डेटा उपलब्ध है ({raw_points} < {int(min_raw_points)}). "
                    "अनुमान सीमित विश्वसनीय हो सकता है।"
                )
            elif latest_dt is None:
                st.error("Latest arrival date is missing after date parsing.")
            else:
                stale_days = (date.today() - latest_dt).days
                if stale_days > int(max_stale_days):
                    st.warning(
                        f"Data is stale by {stale_days} days. Forecast is generated on last available market data."
                    )

                with st.spinner("Training LSTM and generating forecast..."):
                    if fast_mode:
                        hist, fc = build_forecast_from_df(
                            df=mdf,
                            commodity=active_commodity,
                            state=active_state,
                            district=active_district,
                            horizon=15,
                        )
                    else:
                        csv_path = str(AGMARKNET_CSV if AGMARKNET_CSV.exists() else LIVE_MARKET_CSV)
                        hist, fc = build_forecast(
                            csv_path=csv_path,
                            mtime_ns=mtime_ns,
                            commodity=active_commodity,
                            state=active_state,
                            district=active_district,
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

                joined = chart_df.join(fc_chart, how="outer")
                st.line_chart(joined, use_container_width=True)
                st.dataframe(fc2, use_container_width=True, height=240)
        except Exception as e:
            st.warning(f"Forecast unavailable: {e}")

st.markdown("Ask in Hindi or English. The assistant will respond in Hindi.")

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

advisor = get_advisor()

for item in st.session_state.chat_history:
    with st.chat_message(item["role"]):
        if item.get("type") == "market_panel":
            render_market_panel(
                meta=item.get("market_meta") or {},
                auto_chart=item.get("market_chart"),
                auto_table=item.get("market_table"),
            )
        else:
            st.write(item["text"])
            refs = item.get("references", [])
            if refs:
                with st.expander("Sources Used"):
                    for src in refs:
                        st.write(f"- {src}")
            render_feedback_widget(item, advisor)

user_query = st.chat_input("अपना सवाल लिखें... (e.g., 2 एकड़, ₹50,000 बजट, रबी में कौन सी फसल बेहतर है?)")

if user_query:
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

    # If the previous response asked only for a weather location, treat this input as the location.
    if st.session_state.pop("pending_weather_location", False):
        place = user_query.strip()
        lookup_path = Path("data/processed/location_lookup.csv")
        lookup_mtime = lookup_path.stat().st_mtime_ns if lookup_path.exists() else 0
        lookup = load_location_lookup(lookup_mtime)
        district, _state = _lookup_district_from_location(place, lookup)
        weather_place = place if not district else f"{place}, {district}, Uttar Pradesh"
        weather = get_current_weather_hindi(weather_place)
        if not weather:
            weather = get_current_weather_hindi(weather_place)
        final_answer = (
            weather
            if weather
            else "अभी लाइव मौसम डेटा नहीं मिल पाया। कृपया कुछ देर बाद फिर प्रयास करें।"
        )
        query_log_id = log_query_answer(
            user_query=user_query,
            composed_query=weather_place,
            topic="weather",
            answer_text=final_answer,
            references=["Open-Meteo API"],
            district=district or "",
            season=season,
            crop_name=preferred_crop or "unknown",
        )
        st.session_state["last_location_context"] = {"place": place, "district": district or "", "state": "Uttar Pradesh"}
        st.session_state.chat_history.append(
            {"role": "assistant", "text": final_answer, "references": ["Open-Meteo API"], "query_log_id": query_log_id, "topic": "weather", "user_query": user_query}
        )
        with st.chat_message("assistant"):
            st.write(final_answer)
        st.stop()

    last_ctx = st.session_state.get("last_structured_context", {}) or {}
    last_location_ctx = st.session_state.get("last_location_context", {}) or {}
    followup_profit = (
        is_profitability_followup_query(user_query)
        and st.session_state.get("last_structured_topic") in {"crop_profitability", "crop_profitability_followup"}
    )
    followup_crop_care = (
        advisor._is_pesticide_intent(advisor._normalize_hinglish(user_query))
        and st.session_state.get("last_structured_topic") in {"crop_guide", "pesticide"}
    )

    # Resolve place->district for crop intent (so profit uses correct district)
    resolved_district = district
    if is_crop_query(user_query):
        place = extract_place_from_query(user_query)
        if place:
            lookup_path = Path("data/processed/location_lookup.csv")
            lookup_mtime = lookup_path.stat().st_mtime_ns if lookup_path.exists() else 0
            lookup = load_location_lookup(lookup_mtime)
            d_from_place, _ = _lookup_district_from_location(place, lookup)
            if d_from_place:
                resolved_district = d_from_place
    elif followup_profit and last_ctx.get("district"):
        resolved_district = last_ctx["district"]

    season_for_query = last_ctx.get("season", season) if (followup_profit or followup_crop_care) else season
    preferred_crop_for_query = (
        last_ctx.get("preferred_crop", preferred_crop) if (followup_profit or followup_crop_care) else preferred_crop
    )

    composed_query = (
        f"जिला: {resolved_district} | मौसम: {season_for_query} | पसंदीदा फसल: {preferred_crop_for_query or 'कोई नहीं'} | "
        f"किसान का प्रश्न: {user_query.strip()}"
    )

    market_df = load_agmarknet_df()
    intent_price = is_price_query(user_query)
    selected_state, selected_district, selected_commodity = extract_selection_from_query(
        user_query,
        market_df,
        fallback_state=last_location_ctx.get("state") or (active_state if "active_state" in locals() else "Uttar Pradesh"),
        fallback_district=last_location_ctx.get("district") or (active_district if "active_district" in locals() else district),
        fallback_commodity=active_commodity if "active_commodity" in locals() else (preferred_crop or "Wheat"),
    )

    if intent_price and not market_df.empty:
        # Ensure commodity is explicitly detected for price queries.
        comm_from_query = resolve_commodity_from_query(
            user_query, load_commodity_catalog()
        )
        if not comm_from_query:
            final_answer = "कृपया फसल/कमोडिटी का नाम बताएं (जैसे: गेहूं, गन्ना, धान)।"
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
        if selected_district:
            st.session_state["last_location_context"] = {"place": extract_place_from_query(user_query) or last_location_ctx.get("place", ""), "district": selected_district, "state": selected_state}
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
        if filtered.empty:
            if selected_commodity.lower() in {"sugarcane", "गन्ना"}:
                sugarcane_price = get_sugarcane_price_fallback()
                price = sugarcane_price.get("price")
                season = sugarcane_price.get("season", "")
                source_name = sugarcane_price.get("source", "")
                src = sugarcane_price.get("source_url", "")
                if price:
                    final_answer = (
                        f"चयनित जिले ({selected_district}) में {commodity_label} का मंडी डेटा उपलब्ध नहीं है।\n"
                        f"गन्ना के लिए {source_name} {season}: ₹{int(float(price))}/क्विंटल."
                    )
                    if src:
                        final_answer += f"\nस्रोत: {src}"
                else:
                    final_answer = (
                        f"चयनित जिले ({selected_district}) में {commodity_label} का मंडी डेटा उपलब्ध नहीं है। "
                        "CACP से FRP निकालने में समस्या आई।"
                    )
            else:
                msp = get_msp_for_crop(selected_commodity)
                if msp:
                    final_answer = (
                        f"चयनित जिले ({selected_district}) में {commodity_label} का मंडी डेटा नहीं मिला।\n"
                        f"MSP (राष्ट्रीय) {msp['crop']}: ₹{int(msp['msp'])}/क्विंटल.\n"
                        f"स्रोत: {msp['source_url']}"
                    )
                else:
                    final_answer = (
                        f"चयनित जिले ({selected_district}) में {commodity_label} का मंडी डेटा उपलब्ध नहीं है। "
                        "कृपया दूसरी फसल चुनें या बाद में पुनः प्रयास करें।"
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
        auto_chart = None
        auto_table = None
        try:
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
        if auto_table is not None and not auto_table.empty:
            next_vals = auto_table["Forecast"].head(7).tolist()
            vals_str = ", ".join([f"{v:.0f}" for v in next_vals])
            market_answer += f"- अगले 7 दिन के अनुमानित भाव: {vals_str} Rs./Quintal\n"

        if nearest_market:
            market_answer += (
                f"- निकटतम मंडी (लगभग): {nearest_market[1]} ({nearest_market[0]:.1f} km)\n"
            )

        if auto_chart is not None and auto_table is not None:
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
            st.session_state["pending_chat_items"] = [
                {"role": "assistant", "text": market_answer, "references": [], "query_log_id": query_log_id, "topic": "price", "user_query": user_query},
                {
                    "role": "assistant",
                    "type": "market_panel",
                    "text": "",
                    "references": [],
                    "market_meta": market_meta,
                    "market_chart": auto_chart,
                    "market_table": auto_table,
                },
            ]
        else:
            st.session_state.pop("auto_chart", None)
            st.session_state.pop("auto_forecast_table", None)
            st.session_state.pop("auto_forecast_caption", None)
            st.session_state.pop("auto_market_meta", None)

        # Re-render so the market answer and panel appear inline at this chat turn.
        st.rerun()
        final_answer = market_answer
    else:
        with st.spinner("Generating recommendation..."):
            result = advisor.answer(composed_query)
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
        if final_answer.startswith("कृपया मौसम के लिए स्थान बताएं"):
            st.session_state["pending_weather_location"] = True
        query_crop_context = advisor._extract_crop_from_query(advisor._normalize_hinglish(user_query)) or preferred_crop_for_query or ""
        if topic in {"crop_profitability", "crop_profitability_followup", "crop_guide"}:
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
                place_guess = extract_place_from_query(user_query)
                if place_guess:
                    lookup_path = Path("data/processed/location_lookup.csv")
                    lookup_mtime = lookup_path.stat().st_mtime_ns if lookup_path.exists() else 0
                    lookup = load_location_lookup(lookup_mtime)
                    weather_district, weather_state = _lookup_district_from_location(place_guess, lookup)
                    st.session_state["last_location_context"] = {
                        "place": place_guess,
                        "district": weather_district or last_location_ctx.get("district", ""),
                        "state": weather_state or last_location_ctx.get("state", "Uttar Pradesh") or "Uttar Pradesh",
                    }

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "text": final_answer,
            "references": ([] if intent_price else result.get("references", [])),
            "query_log_id": (None if intent_price else query_log_id),
            "topic": (None if intent_price else topic),
            "user_query": user_query,
        }
    )

    with st.chat_message("assistant"):
        st.write(final_answer)
        if not intent_price:
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
                    st.cache_data.clear()
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
