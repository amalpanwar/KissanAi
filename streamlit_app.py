from __future__ import annotations

import os
import re
import sys
import sqlite3
from datetime import date
from time import time
from pathlib import Path

import pandas as pd
import streamlit as st
import json
from urllib.parse import urlencode
from urllib.request import urlopen

from app.advisor import AdvisorConfig, RAGAdvisor
from app.config import load_config
from app.datagov_client import DataGovClient
from app.lstm_forecast import prepare_daily_series, train_and_forecast


st.set_page_config(page_title="KisaanAI - Western UP", page_icon="🌾", layout="wide")

st.title("KisaanAI - Western Uttar Pradesh Agriculture Assistant")
st.caption("RAG + SLM based local-language advisory for crop, budget, and yield conditions")

cfg = load_config()
LIVE_MARKET_CSV = Path("data/raw/live/datagov_commodity.csv")
AGMARKNET_CSV = Path("data/raw/live/agmarknet_report.csv")
FETCH_PAGE_LIMIT = 200
FETCH_MAX_RECORDS_COMBO = 50000
FETCH_MAX_RECORDS_STATE = 50000
FETCH_COOLDOWN_SEC = 600
FAST_FETCH_LIMIT = 200


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
        "भाव",
        "दाम",
        "कीमत",
        "मंडी",
        "forecast",
        "भविष्य",
        "अगले",
        "आने वाले",
        "15 दिन",
        "पंद्रह दिन",
    ]
    return any(k in t for k in keywords)


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
    tokens = [t.strip(" ?!.,") for t in query.split() if t.strip()]
    stop = {"aaj", "aj", "ka", "ki", "ke", "me", "mein", "में", "kesa", "kaisa", "hai", "h"}
    for tok in tokens:
        if tok.lower() not in stop and tok.lower() not in {"price", "rate", "mandi", "bhav"}:
            return tok
    return None


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


@st.cache_data(show_spinner=False)
def load_commodity_aliases() -> dict[str, list[str]]:
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


def resolve_commodity_from_query(query: str, commodity_list: list[str]) -> str | None:
    if not query or not commodity_list:
        return None
    q = query.lower()
    q_tokens = re.findall(r"[a-z0-9]+", q)
    q_norm = " ".join(q_tokens)
    # Match by full token to avoid "rice" in "price"
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
    # Normalized match (remove spaces/punct)
    qn = _normalize_text(q)
    for name in commodity_list:
        if _normalize_text(name) in qn:
            return name
    # Hinglish/Hindi aliases mapped to canonical names if present in catalog
    # Alias map from json (key is English commodity name)
    aliases = load_commodity_aliases()
    for eng_name, alias_list in aliases.items():
        for alias in alias_list:
            a_tokens = re.findall(r"[a-z0-9]+", alias.lower())
            if not a_tokens:
                continue
            if len(a_tokens) == 1:
                if a_tokens[0] in q_tokens:
                    # Return the catalog's exact name if present
                    for name in commodity_list:
                        if name.lower() == eng_name.lower():
                            return name
                    return eng_name.title()
            else:
                a_norm = " ".join(a_tokens)
                if a_norm in q_norm:
                    for name in commodity_list:
                        if name.lower() == eng_name.lower():
                            return name
                    return eng_name.title()
    return None

LOCATION_DISTRICT_MAP = {
    "doghat": "Baghpat",
    "डोघाट": "Baghpat",
    "दोगहट": "Baghpat",
    "baraut": "Baghpat",
    "बड़ौत": "Baghpat",
    "बरौत": "Baghpat",
    "biral": "Baghpat",
    "बिराल": "Baghpat",
}


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
        dist_match = _best_match(q, districts)
        if dist_match:
            district = dist_match
        # Prefer geocode + reverse-geocode if a place is present
        place = extract_place_from_query(query)
        if place:
            geo = _geocode_cached(place, "Uttar Pradesh")
            if geo:
                plat, plon, _ = geo
                rev = _reverse_geocode_cached(plat, plon)
                if rev and rev.get("district"):
                    rev_norm = _normalize_district_name(rev["district"])
                    for d in districts:
                        if _normalize_district_name(d) == rev_norm:
                            district = d
                            break
                    else:
                        # partial match fallback
                        for d in districts:
                            if rev_norm and rev_norm in _normalize_district_name(d):
                                district = d
                                break
        # Fallback to manual mapping if reverse-geocode fails
        for loc, dist in LOCATION_DISTRICT_MAP.items():
            if loc in q:
                district = dist
                break
        catalog = load_commodity_catalog()
        comm_match = resolve_commodity_from_query(query, catalog or commodities)
        if comm_match:
            commodity = comm_match
        else:
            comm_match = _best_match(q, commodities)
            if comm_match:
                commodity = comm_match

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
    qty = latest.get("Arrival_Qty", None)
    arrival_unit = latest.get("Arrival_Unit", None)
    line = f"ताज़ा भाव ({latest_dt.date()}): {modal} {unit}"
    if qty and arrival_unit:
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


def save_advisory(
    farmer_id: str,
    district: str,
    season: str,
    crop_name: str,
    recommendation_text: str,
) -> None:
    conn = sqlite3.connect(cfg.paths["sqlite_db"])
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

# Apply pending sidebar selection from last query (if any)
pending = st.session_state.pop("pending_selection", None)
if isinstance(pending, dict):
    st.session_state["fc_state"] = pending.get("state", st.session_state.get("fc_state", "Uttar Pradesh"))
    st.session_state["fc_district"] = pending.get("district", st.session_state.get("fc_district", "Meerut"))
    st.session_state["fc_commodity_override"] = pending.get("commodity", "")

with st.sidebar:
    st.subheader("Farmer Context")
    farmer_id = st.text_input("Farmer ID", value="FARMER_DEMO")
    district = st.selectbox(
        "District",
        ["Meerut", "Muzaffarnagar", "Baghpat", "Saharanpur", "Shamli", "Bulandshahr"],
    )
    season = st.selectbox("Season", ["Rabi", "Kharif", "Annual"])
    preferred_crop = st.text_input("Preferred Crop (optional)", value="Wheat")

    st.markdown("---")
    st.subheader("Commodity Forecast (15 Days)")
    env_vals = load_local_env(Path(".env"))
    api_key = os.getenv("DATA_GOV_API_KEY", "") or env_vals.get("DATA_GOV_API_KEY", "")
    resource_id = os.getenv("DATA_GOV_RESOURCE_ID", "") or env_vals.get(
        "DATA_GOV_RESOURCE_ID", "35985678-0d79-46b4-9ed6-6f13308a1d24"
    )

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
        district_options = [district]
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

    auto_chart = st.session_state.get("auto_chart")
    auto_table = st.session_state.get("auto_forecast_table")
    auto_caption = st.session_state.get("auto_forecast_caption")
    if auto_chart is not None and auto_table is not None:
        st.markdown("---")
        st.subheader("Query Forecast")
        if auto_caption:
            st.caption(auto_caption)
        st.line_chart(auto_chart, use_container_width=True)
        st.dataframe(auto_table, use_container_width=True, height=240)

st.markdown("Ask in Hindi or English. The assistant will respond in Hindi.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for item in st.session_state.chat_history:
    with st.chat_message(item["role"]):
        st.write(item["text"])
        refs = item.get("references", [])
        if refs:
            with st.expander("Sources Used"):
                for src in refs:
                    st.write(f"- {src}")

user_query = st.chat_input("अपना सवाल लिखें... (e.g., 2 एकड़, ₹50,000 बजट, रबी में कौन सी फसल बेहतर है?)")

if user_query:
    st.session_state.chat_history.append({"role": "user", "text": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    advisor = get_advisor()
    composed_query = (
        f"जिला: {district} | मौसम: {season} | पसंदीदा फसल: {preferred_crop or 'कोई नहीं'} | "
        f"किसान का प्रश्न: {user_query.strip()}"
    )

    market_df = load_agmarknet_df()
    intent_price = is_price_query(user_query)
    selected_state, selected_district, selected_commodity = extract_selection_from_query(
        user_query,
        market_df,
        fallback_state=active_state if "active_state" in locals() else "Uttar Pradesh",
        fallback_district=active_district if "active_district" in locals() else district,
        fallback_commodity=active_commodity if "active_commodity" in locals() else (preferred_crop or "Wheat"),
    )

    if intent_price and not market_df.empty:
        filtered = filter_market_rows(market_df, selected_commodity, selected_state, selected_district)
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
        auto_caption = f"{selected_state} / {selected_district} / {selected_commodity}"
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

        if auto_chart is not None and auto_table is not None:
            st.session_state["auto_chart"] = auto_chart
            st.session_state["auto_forecast_table"] = auto_table
            st.session_state["auto_forecast_caption"] = auto_caption
        else:
            st.session_state.pop("auto_chart", None)
            st.session_state.pop("auto_forecast_table", None)
            st.session_state.pop("auto_forecast_caption", None)

        market_list = []
        if "Market" in filtered.columns:
            market_list = (
                filtered["Market"].dropna().astype(str).unique().tolist()
            )
        market_answer = (
            f"बाजार जानकारी ({selected_state} / {selected_district} / {selected_commodity}):\n"
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
        # Skip RAG for price intent to avoid irrelevant filler.
        final_answer = market_answer
    else:
        with st.spinner("Generating recommendation..."):
            result = advisor.answer(composed_query)
        final_answer = result["answer"]

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "text": final_answer,
            "references": ([] if intent_price else result.get("references", [])),
        }
    )

    with st.chat_message("assistant"):
        st.write(final_answer)
        if not intent_price:
            with st.expander("Sources Used"):
                for src in result.get("references", []):
                    st.write(f"- {src}")

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
