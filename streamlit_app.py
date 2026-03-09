from __future__ import annotations

import os
import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from app.advisor import AdvisorConfig, RAGAdvisor
from app.config import load_config
from app.datagov_client import DataGovClient
from app.lstm_forecast import prepare_daily_series, train_and_forecast


st.set_page_config(page_title="KisaanAI - Western UP", page_icon="🌾", layout="wide")

st.title("KisaanAI - Western Uttar Pradesh Agriculture Assistant")
st.caption("RAG + SLM based local-language advisory for crop, budget, and yield conditions")

cfg = load_config()
LIVE_MARKET_CSV = Path("data/raw/live/datagov_commodity.csv")
FETCH_PAGE_LIMIT = 500
FETCH_MAX_RECORDS = 100000


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
        epochs=80,
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

    if LIVE_MARKET_CSV.exists():
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

    min_raw_points = st.number_input("Min Raw Date Points", min_value=30, max_value=365, value=90, step=10)
    max_stale_days = st.number_input("Max Stale Days", min_value=0, max_value=30, value=15, step=1)

    def run_fetch(use_state_only: bool = False) -> tuple[int, int]:
        if not api_key:
            st.warning("DATA_GOV_API_KEY not found in .env")
            return 0, 0
        if not resource_id:
            st.warning("DATA_GOV_RESOURCE_ID not found in .env")
            return 0, 0

        client = DataGovClient(api_key=api_key, timeout_sec=45, retries=4)
        params = {}
        if selected_state.strip():
            params["filters[State]"] = selected_state.strip()
        if not use_state_only:
            if selected_district.strip():
                params["filters[District]"] = selected_district.strip()
            if selected_commodity.strip():
                params["filters[Commodity]"] = selected_commodity.strip()
        recs = client.fetch_records(
            resource_id=resource_id,
            limit=FETCH_PAGE_LIMIT,
            max_records=FETCH_MAX_RECORDS,
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

    # If dropdowns are too narrow (e.g., only Meerut/Wheat), auto-backfill state catalog once.
    state_catalog_key = f"state_catalog_loaded::{selected_state}"
    if (
        len(district_options) <= 1
        and api_key
        and resource_id
        and not st.session_state.get(state_catalog_key, False)
    ):
        with st.spinner(f"Loading full district/commodity catalog for {selected_state}..."):
            try:
                fetched, stored = run_fetch(use_state_only=True)
                st.session_state[state_catalog_key] = True
                if fetched > 0:
                    st.caption(f"Catalog loaded for {selected_state}: fetched {fetched}, stored {stored}")
                    st.rerun()
            except Exception as e:
                st.session_state[state_catalog_key] = True
                st.warning(f"Catalog auto-load failed: {e}")

    combo_key = f"{selected_state}|{selected_district}|{selected_commodity}".lower()
    if st.session_state.get("last_combo_key") != combo_key and api_key and resource_id:
        with st.spinner("Updating data for selected combination..."):
            try:
                fetched, stored = run_fetch(use_state_only=False)
                if fetched > 0:
                    st.caption(f"Auto-updated selection data: fetched {fetched}, stored {stored}")
                st.session_state["last_combo_key"] = combo_key
            except Exception as e:
                st.warning(f"Auto refresh failed: {e}")

    if st.button("Refresh Selected Combination", use_container_width=True):
        with st.spinner("Fetching selected combination..."):
            try:
                fetched, stored = run_fetch(use_state_only=False)
                if fetched > 0:
                    st.success(f"Fetched {fetched} rows, stored {stored} unique rows.")
                else:
                    st.warning("No records returned for selected combination.")
            except Exception as e:
                st.error(f"Fetch failed: {e}")

    if st.button("Refresh State Catalog", use_container_width=True):
        with st.spinner("Fetching all districts/commodities for selected state..."):
            try:
                fetched, stored = run_fetch(use_state_only=True)
                if fetched > 0:
                    st.success(f"Fetched {fetched} rows, stored {stored} unique rows.")
                else:
                    st.warning("No records returned for selected state.")
            except Exception as e:
                st.error(f"Fetch failed: {e}")

    if not LIVE_MARKET_CSV.exists():
        st.info("No live commodity CSV found. Run data fetch script first.")
    else:
        try:
            mtime_ns = LIVE_MARKET_CSV.stat().st_mtime_ns
            mdf = load_market_df(str(LIVE_MARKET_CSV), mtime_ns)
            if mdf.empty:
                st.info("Live commodity data file is empty.")
            else:
                mdf.columns = [c.strip() for c in mdf.columns]
                if st.button("Show 15-Day Forecast", use_container_width=True):
                    filtered = filter_market_rows(mdf, selected_commodity, selected_state, selected_district)
                    raw_points = int(filtered["Arrival_Date_dt"].dt.date.nunique()) if not filtered.empty else 0
                    latest_dt = filtered["Arrival_Date_dt"].max().date() if raw_points > 0 else None
                    st.caption(
                        f"Raw points: {raw_points} | Latest arrival date: {latest_dt if latest_dt else 'N/A'}"
                    )

                    if raw_points < int(min_raw_points):
                        st.error(
                            f"Insufficient raw history for reliable LSTM forecast ({raw_points} < {int(min_raw_points)}). "
                            "Refresh data with broader filters or choose another commodity/district."
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
                            hist, fc = build_forecast(
                                csv_path=str(LIVE_MARKET_CSV),
                                mtime_ns=mtime_ns,
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

                        joined = chart_df.join(fc_chart, how="outer")
                        st.line_chart(joined, use_container_width=True)
                        st.dataframe(fc2, use_container_width=True, height=240)
        except Exception as e:
            st.warning(f"Forecast unavailable: {e}")

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
    with st.spinner("Generating recommendation..."):
        result = advisor.answer(composed_query)

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "text": result["answer"],
            "references": result.get("references", []),
        }
    )

    with st.chat_message("assistant"):
        st.write(result["answer"])
        with st.expander("Sources Used"):
            for src in result.get("references", []):
                st.write(f"- {src}")

    save_advisory(
        farmer_id=farmer_id,
        district=district,
        season=season,
        crop_name=preferred_crop or "unknown",
        recommendation_text=result["answer"],
    )
