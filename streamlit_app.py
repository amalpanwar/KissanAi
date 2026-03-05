from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from app.advisor import AdvisorConfig, RAGAdvisor
from app.config import load_config
from app.lstm_forecast import prepare_daily_series, train_and_forecast


st.set_page_config(page_title="KisaanAI - Western UP", page_icon="🌾", layout="wide")

st.title("KisaanAI - Western Uttar Pradesh Agriculture Assistant")
st.caption("RAG + SLM based local-language advisory for crop, budget, and yield conditions")

cfg = load_config()
LIVE_MARKET_CSV = Path("data/raw/live/datagov_commodity.csv")


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
def load_market_df(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


@st.cache_data(show_spinner=False)
def build_forecast(
    csv_path: str,
    commodity: str,
    state: str,
    district: str,
    horizon: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    if not LIVE_MARKET_CSV.exists():
        st.info("No live commodity CSV found. Run data fetch script first.")
    else:
        try:
            mdf = load_market_df(str(LIVE_MARKET_CSV))
            if mdf.empty:
                st.info("Live commodity data file is empty.")
            else:
                mdf.columns = [c.strip() for c in mdf.columns]
                commodities = sorted(mdf["Commodity"].dropna().astype(str).unique().tolist())
                states = sorted(mdf["State"].dropna().astype(str).unique().tolist())

                selected_commodity = st.selectbox("Commodity", commodities, index=0)
                selected_state = st.selectbox("State", states, index=0)

                district_choices = (
                    mdf[mdf["State"].astype(str) == selected_state]["District"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
                district_choices = sorted(district_choices) or ["Meerut"]
                selected_district = st.selectbox("District (Market Data)", district_choices, index=0)

                if st.button("Show 15-Day Forecast", use_container_width=True):
                    with st.spinner("Training LSTM and generating forecast..."):
                        hist, fc = build_forecast(
                            csv_path=str(LIVE_MARKET_CSV),
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
