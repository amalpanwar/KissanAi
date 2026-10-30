"""Autocomplete chat composer; sends one value only when a question is submitted."""
from __future__ import annotations

import json
import re
from pathlib import Path
import streamlit.components.v1 as components

_composer = components.declare_component("kisaan_chat_composer", path=str(Path(__file__).with_name("chat_component")))

SUGGESTIONS = [
    {"question": "आज मेरे क्षेत्र में मौसम कैसा रहेगा?", "keywords": "weather rain forecast temperature mausam mosam barish baarish pani मौसम बारिश वर्षा तापमान"},
]


def _commodity_aliases():
    """Read market commodity names and aliases from the repository dictionary."""
    path = Path(__file__).resolve().parents[1] / "data" / "raw" / "commodity_aliases.json"
    try:
        with path.open(encoding="utf-8") as source:
            data = json.load(source)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _display_commodity(names, fallback):
    for name in names:
        value = str(name).strip()
        if value and re.search(r"[\u0900-\u097f]", value):
            return value
    return next((str(name).strip() for name in names if str(name).strip()), fallback)


def commodity_price_suggestions():
    """Offer a mandi-price question for every commodity in the alias dictionary."""
    intent_terms = "price prices mandi rate bhav bhaav daam market भाव कीमत मंडी दाम"
    suggestions = []
    for commodity, aliases in _commodity_aliases().items():
        names = [commodity, *(aliases if isinstance(aliases, list) else [])]
        names = list(dict.fromkeys(str(name).strip() for name in names if str(name).strip()))
        display = _display_commodity(names, commodity)
        suggestions.append({
            "question": f"आज {display} का मंडी भाव क्या है?",
            "keywords": " ".join([*names, intent_terms]),
        })
    return suggestions


def crop_guide_suggestions():
    """Offer crop-specific cultivation and protection prompts for guide crops."""
    from app.crop_guide import CROP_ALIASES, CROP_NAME_HI

    section_names = {
        "Millets", "Small Millets", "Pulses", "Oilseeds", "FIBRE CROPS",
        "Forage Crops", "Green Manure Crops", "SERICULTURE", "AGRO FORESTRY",
    }
    market_aliases = _commodity_aliases()
    guide_terms = {
        "cultivation": "grow growing cultivation farming kheti kaise sowing खेती उगाना बुवाई कैसे",
        "protection": "pest pests disease diseases protection spray insecticide pesticide keet rog bachav कीट रोग बचाव दवा",
    }
    suggestions = []
    for crop, aliases in CROP_ALIASES.items():
        if crop in section_names:
            continue
        crop_names = [crop, *(str(alias) for alias in aliases)]
        normalized = {name.strip().casefold() for name in crop_names if name.strip()}
        for names in market_aliases.values():
            if isinstance(names, list) and any(str(name).strip().casefold() in normalized for name in names):
                crop_names.extend(str(name).strip() for name in names if str(name).strip())
        keywords = " ".join(dict.fromkeys(name.strip() for name in crop_names if name.strip()))
        display_name = CROP_NAME_HI.get(crop, crop)
        for kind, terms in guide_terms.items():
            question = (
                f"{display_name} की खेती कैसे करें?"
                if kind == "cultivation"
                else f"{display_name} में कीटों की रोकथाम के लिए कौन सा कीटनाशक उपयोग करें?"
            )
            suggestions.append({
                "question": question,
                "keywords": f"{keywords} {terms}",
            })
    return suggestions


SUGGESTIONS.extend(commodity_price_suggestions())
SUGGESTIONS.append({
    "question": "गेहूं में दीमक की रोकथाम के लिए कौन सा कीटनाशक उपयोग करें?",
    "keywords": "Wheat wheat gehu gehun गेहूं diimak termite termites pest pesticide insecticide कीटनाशक दीमक",
})
SUGGESTIONS.extend(crop_guide_suggestions())

def consume_submission(value, state):
    if not isinstance(value, dict):
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
    value = _composer(suggestions=SUGGESTIONS, key=f"farmer_autocomplete_chat_{epoch}", default=None)
    return consume_submission(value, st.session_state)
