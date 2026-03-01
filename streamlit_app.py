from __future__ import annotations

import os
import sqlite3

import streamlit as st

from app.advisor import AdvisorConfig, RAGAdvisor
from app.config import load_config


st.set_page_config(page_title="KisaanAI - Western UP", page_icon="🌾", layout="wide")

st.title("KisaanAI - Western Uttar Pradesh Agriculture Assistant")
st.caption("RAG + SLM based local-language advisory for crop, budget, and yield conditions")

cfg = load_config()


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
        )
    )


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
        f"District: {district} | Season: {season} | Preferred crop: {preferred_crop}. "
        f"Farmer question: {user_query.strip()}"
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
