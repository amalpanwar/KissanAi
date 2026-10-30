"""Autocomplete chat composer; sends one value only when a question is submitted."""
from __future__ import annotations

from pathlib import Path
import streamlit.components.v1 as components

_composer = components.declare_component("kisaan_chat_composer", path=str(Path(__file__).with_name("chat_component")))

SUGGESTIONS = [
    {"question": "आज मेरे क्षेत्र में मौसम कैसा रहेगा?", "keywords": "weather rain forecast temperature mausam mosam barish baarish pani मौसम बारिश वर्षा तापमान"},
    {"question": "आज गेहूं का मंडी भाव क्या है?", "keywords": "price prices mandi rate bhav bhaav daam market भाव कीमत मंडी दाम"},
    {"question": "गेहूं की खेती कैसे करें?", "keywords": "grow growing cultivation kheti kaise kese gehu wheat खेती उगाना बुवाई कैसे"},
    {"question": "गेहूं में दीमक की रोकथाम के लिए कौन सा कीटनाशक उपयोग करें?", "keywords": "pesticide pesticides pest disease spray insecticide keet dimak keetnashak dawa कीटनाशक कीट रोग दवा दीमक"},
]


def consume_submission(value, state):
    if not isinstance(value, dict):
        return None
    if value.get("kind") == "draft":
        return None
    text = value.get("text")
    submission_id = value.get("id")
    if not isinstance(text, str) or not text.strip() or len(text) > 4000 or not isinstance(submission_id, str):
        return None
    if not submission_id or state.get("last_chat_submission") == submission_id:
        return None
    state["last_chat_submission"] = submission_id
    return text.strip()


def render_chat_composer(st):
    epoch = st.session_state.get("chat_location_epoch", 0)
    from app.query_translation import translate_query
    preview_key = f"query_preview_{epoch}"
    preview = st.session_state.get(preview_key, {})
    value = _composer(suggestions=SUGGESTIONS, preview=preview,
                      key=f"farmer_autocomplete_chat_{epoch}", default=None)
    if isinstance(value, dict) and value.get("kind") == "draft":
        text = value.get("text", "")
        draft_id = value.get("id")
        if isinstance(text, str) and isinstance(draft_id, str) and len(text) <= 4000 and draft_id != preview.get("id"):
            location = st.session_state.get("last_location_context", {})
            names = [location.get(k, "") for k in ('place', 'district', 'state', 'sub_district')]
            result = translate_query(text, names)
            st.session_state[preview_key] = dict(result, id=draft_id, original=text)
            st.rerun()
        return None
    return consume_submission(value, st.session_state)
