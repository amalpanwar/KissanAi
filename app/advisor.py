from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass
import json
import re
import sqlite3
from zoneinfo import ZoneInfo
import csv
from pathlib import Path

from app.embeddings import Embedder
from app.generator import LocalGenerator
from app.prompting import build_prompt
from app.retriever import Retriever
from app.vector_store import NumpyVectorStore
from app.weather import get_current_weather_hindi
from app.upag_apy import load_latest_up_yield_qtl_per_acre
from app.crop_guide import build_crop_production_guide
from app.cacp import get_cacp_cost_for_crop, get_sugarcane_cost_snapshot, get_latest_sugarcane_frp
import pandas as pd

from app.location_lookup import lookup_place, lookup_place_in_text

COMMODITY_ALIAS_PATH = Path("data/raw/commodity_aliases.json")

RAW_CROP_COMMODITIES = {
    "wheat",
    "rice",
    "paddy",
    "maize",
    "bajra",
    "barley",
    "jowar",
    "mustard",
    "mustard seed",
    "potato",
    "onion",
    "garlic",
    "gram",
    "bengal gram",
    "green gram",
    "black gram",
    "arhar",
    "lentil",
    "pea",
    "peas",
    "sugarcane",
    "gur",
    "cotton",
    "soybean",
    "groundnut",
    "sesamum",
    "tomato",
    "brinjal",
    "chilli",
    "cabbage",
    "cauliflower",
}
PROCESSED_COMMODITY_WORDS = {
    "dal",
    "oil",
    "flour",
    "atta",
    "gur",
    "jaggery",
    "bran",
    "cake",
    "split",
    "milled",
}
DISEASE_ALIASES = {
    "red rot": ["red rot", "लाल सड़न", "redrot"],
    "rust": ["rust", "रतुआ"],
    "yellow rust": ["yellow rust", "stripe rust", "पीली रतुआ", "पीला रतुआ"],
    "brown rust": ["brown rust", "leaf rust", "भूरी रतुआ"],
    "black rust": ["black rust", "stem rust", "काली रतुआ"],
    "loose smut": ["loose smut", "smut", "ढीला कंडुआ", "कंडुआ"],
    "karnal bunt": ["karnal bunt", "bunt", "burnt", "करनाल बंट", "कर्नाल बंट", "बंट"],
    "powdery mildew": ["powdery mildew", "चूर्णी फफूंदी"],
    "leaf blight": ["leaf blight", "blight", "झुलसा"],
    "stem borer": ["stem borer", "तना छेदक"],
    "insect pest": [
        "borer",
        "stem borer",
        "leaf folder",
        "leaffolder",
        "hopper",
        "planthopper",
        "fly",
        "aphid",
        "mite",
        "caterpillar",
        "कीट",
    ],
}
DISEASE_HINDI_TERMS = {
    "red rot": "लाल सड़न",
    "yellow rust": "पीली रतुआ",
    "stripe rust": "पीली रतुआ",
    "brown rust": "भूरी रतुआ",
    "leaf rust": "पत्ती/भूरी रतुआ",
    "black rust": "काली रतुआ",
    "stem rust": "तना/काली रतुआ",
    "rust": "रतुआ",
    "leaf blight": "पत्ती झुलसा",
    "blight": "झुलसा",
    "loose smut": "ढीला कंडुआ",
    "karnal bunt": "कर्नाल बंट",
    "powdery mildew": "चूर्णी फफूंदी",
    "stem borer": "तना छेदक",
    "insect pest": "कीट",
    "aphid": "माहू",
    "termite": "दीमक",
}
PEST_LABEL_WORDS = {
    "aphid",
    "aphids",
    "backed",
    "brown",
    "caterpillar",
    "folder",
    "fruit",
    "gall",
    "green",
    "hopper",
    "leaf",
    "maggot",
    "mite",
    "moth",
    "plant",
    "rice",
    "shoot",
    "stem",
    "thrips",
    "white",
    "whorl",
    "yellow",
    "borer",
    "fly",
}
WESTERN_UP_CROP_BASELINES = {
    # Indicative per-acre baselines. Prices are overridden by district Agmarknet
    # when a raw crop price is available; sugarcane uses cane/SAP-style pricing,
    # not jaggery/sugar prices because those are processed commodities.
    "Sugarcane": {
        "season": "Annual",
        "yield_qtl_per_acre": 320,
        "cost_min": 85000,
        "cost_max": 115000,
        "fallback_price": 380,
        "water_need": "बहुत अधिक",
        "risk": "लंबी अवधि और अधिक पानी/मजदूरी",
        "market_aliases": ["Sugarcane"],
    },
    "Potato": {
        "season": "Rabi",
        "yield_qtl_per_acre": 95,
        "cost_min": 60000,
        "cost_max": 95000,
        "fallback_price": 900,
        "water_need": "मध्यम",
        "risk": "भंडारण और भाव गिरने का जोखिम",
        "market_aliases": ["Potato"],
    },
    "Mustard": {
        "season": "Rabi",
        "yield_qtl_per_acre": 8,
        "cost_min": 13000,
        "cost_max": 22000,
        "fallback_price": 5650,
        "water_need": "कम",
        "risk": "कम लागत, पर उपज सीमित",
        "market_aliases": ["Mustard"],
    },
    "Wheat": {
        "season": "Rabi",
        "yield_qtl_per_acre": 20,
        "cost_min": 22000,
        "cost_max": 34000,
        "fallback_price": 2425,
        "water_need": "मध्यम",
        "risk": "स्थिर फसल, लाभ मध्यम",
        "market_aliases": ["Wheat"],
    },
    "Paddy": {
        "season": "Kharif",
        "yield_qtl_per_acre": 24,
        "cost_min": 30000,
        "cost_max": 48000,
        "fallback_price": 2300,
        "water_need": "बहुत अधिक",
        "risk": "पानी और कीट/रोग दबाव अधिक",
        "market_aliases": ["Paddy(Common)", "Paddy(Basmati)", "Rice"],
    },
    "Maize": {
        "season": "Kharif/Zaid",
        "yield_qtl_per_acre": 22,
        "cost_min": 22000,
        "cost_max": 34000,
        "fallback_price": 2225,
        "water_need": "मध्यम",
        "risk": "भाव और वर्षा पर निर्भर",
        "market_aliases": ["Maize"],
    },
    "Black Gram": {
        "season": "Kharif/Zaid",
        "yield_qtl_per_acre": 5,
        "cost_min": 14000,
        "cost_max": 24000,
        "fallback_price": 7400,
        "water_need": "कम",
        "risk": "कीट दबाव और कम उपज",
        "market_aliases": ["Black Gram(Urd Beans)(Whole)"],
    },
    "Green Gram": {
        "season": "Kharif/Zaid",
        "yield_qtl_per_acre": 5,
        "cost_min": 14000,
        "cost_max": 24000,
        "fallback_price": 8682,
        "water_need": "कम",
        "risk": "कीट दबाव और कम उपज",
        "market_aliases": ["Green Gram(Moong)(Whole)"],
    },
    "Arhar": {
        "season": "Kharif",
        "yield_qtl_per_acre": 6,
        "cost_min": 18000,
        "cost_max": 30000,
        "fallback_price": 8000,
        "water_need": "कम-मध्यम",
        "risk": "लंबी अवधि और pod borer जोखिम",
        "market_aliases": ["Arhar(Tur/Red Gram)(Whole)"],
    },
    "Groundnut": {
        "season": "Kharif",
        "yield_qtl_per_acre": 9,
        "cost_min": 25000,
        "cost_max": 42000,
        "fallback_price": 6783,
        "water_need": "मध्यम",
        "risk": "रेतीली मिट्टी में बेहतर, कीट/रोग जोखिम",
        "market_aliases": ["Groundnut"],
    },
}
EXOTIC_CROP_NOTES = [
    "Saffron: Western UP open-field climate के लिए सामान्यतः उपयुक्त नहीं; इसे high-altitude/cool-dry climate चाहिए, इसलिए इसे सामान्य लाभ रैंकिंग में नहीं रखा।",
    "Dragon fruit/Strawberry/Protected vegetables: high-value हो सकते हैं, पर polyhouse/irrigation/market-linkage और बहुत अधिक शुरुआती पूंजी चाहिए; अलग business-plan query पर इन्हें अलग से compare करना बेहतर है।",
]


@dataclass
class AdvisorConfig:
    embedding_model: str
    generator_model: str
    index_path: str
    metadata_path: str
    top_k: int
    db_path: str | None = None


class RAGAdvisor:
    def __init__(self, cfg: AdvisorConfig) -> None:
        self.cfg = cfg
        self.embedder: Embedder | None = None
        self.retriever: Retriever | None = None
        self.generator: LocalGenerator | None = None
        self.top_k = cfg.top_k

    def answer(self, user_query: str) -> dict:
        context_part, farmer_question = self._split_context_and_question(user_query)
        if self._is_greeting(farmer_question) and not self._has_agri_intent(farmer_question):
            return {"answer": self._time_based_greeting(), "references": [], "retrieved": [], "topic": "greeting"}

        normalized_question = self._normalize_hinglish(farmer_question)
        loc_ctx = lookup_place_in_text(farmer_question) or lookup_place_in_text(normalized_question)
        if loc_ctx:
            district = loc_ctx.get("district")
            state = loc_ctx.get("state")
            if district and "District:" not in context_part and "जिला:" not in context_part:
                suffix = f"District: {district}"
                if state:
                    suffix = f"{suffix} | State: {state}"
                context_part = f"{context_part} | {suffix}" if context_part else suffix
        if self._is_weather_impact_intent(normalized_question, context_part):
            return {
                "answer": self._weather_impact_advice(context_part, normalized_question),
                "references": [],
                "retrieved": [],
                "topic": "weather_impact",
            }
        if self._is_weather_intent(normalized_question):
            place = (
                self._extract_location_from_question(farmer_question)
                or self._extract_location_from_question(normalized_question)
            )
            loc = lookup_place(place) if place else None
            if not loc:
                loc = lookup_place_in_text(farmer_question) or lookup_place_in_text(normalized_question)
            if loc and loc.get("place"):
                place = loc.get("place")
            if not place and not loc:
                return {
                    "answer": "मौसम के लिए स्थान नहीं मिला। कृपया स्थान लिखें (जैसे: डोघाट/बड़ौत/मेरठ)।",
                    "references": [],
                    "retrieved": [],
                    "topic": "weather",
                }
            loc = loc or (lookup_place(place) if place else None)
            district = loc.get("district") if loc else self._lookup_district_from_location(place)
            state = loc.get("state") if loc else "Uttar Pradesh"
            weather_place = place if not district else f"{place}, {district}, {state}"
            weather = get_current_weather_hindi(weather_place)
            if not weather:
                weather = get_current_weather_hindi(weather_place)
            if not weather:
                return {
                    "answer": "अभी लाइव मौसम डेटा नहीं मिल पाया। कृपया कुछ देर बाद फिर प्रयास करें।",
                    "references": [],
                    "retrieved": [],
                    "topic": "weather",
                }
            return {"answer": weather, "references": ["Open-Meteo API"], "retrieved": [], "topic": "weather"}
        profitability_followup = (
            self._is_profitability_followup_intent(normalized_question)
            or self._is_profitability_followup_intent(farmer_question)
        )
        if profitability_followup:
            crop_cost_method_answer = (
                self._answer_crop_cost_method_query(normalized_question, context_part)
                or self._answer_crop_cost_method_query(farmer_question, context_part)
            )
            if crop_cost_method_answer:
                return {
                    "answer": crop_cost_method_answer,
                    "references": ["CACP official report", "UPAG yield data"],
                    "retrieved": [],
                    "topic": "crop_profitability_followup",
                }
            return {
                "answer": self._explain_profitability_method(context_part, farmer_question),
                "references": [],
                "retrieved": [],
                "topic": "crop_profitability_followup",
            }
        sugarcane_cost_answer = (
            self._answer_sugarcane_cost_query(normalized_question, context_part)
            or self._answer_sugarcane_cost_query(farmer_question, context_part)
        )
        if sugarcane_cost_answer:
            return {
                "answer": sugarcane_cost_answer,
                "references": ["data/raw/official_sources/cacp_sugarcane_latest.pdf"],
                "retrieved": [],
                "topic": "crop_profitability_followup",
            }
        generic_cacp_cost_answer = (
            self._answer_generic_cacp_cost_query(normalized_question, context_part)
            or self._answer_generic_cacp_cost_query(farmer_question, context_part)
        )
        if generic_cacp_cost_answer:
            return {
                "answer": generic_cacp_cost_answer,
                "references": ["CACP official report"],
                "retrieved": [],
                "topic": "crop_profitability_followup",
            }
        if self._is_crop_guide_intent(normalized_question):
            guide_answer, guide_sources = build_crop_production_guide(normalized_question)
            if guide_answer:
                return {
                    "answer": guide_answer,
                    "references": guide_sources,
                    "retrieved": [],
                    "topic": "crop_guide",
                }
        if self._is_crop_choice_intent(normalized_question):
            place = self._extract_location_from_question(farmer_question)
            loc = lookup_place_in_text(farmer_question) or lookup_place_in_text(normalized_question)
            if not loc and place:
                loc = lookup_place(place)
            if loc and not place:
                place = loc.get("place")
            district_override = loc.get("district") if loc else None
            structured, sources = self._structured_crop_recommendation(
                context_part,
                normalized_question,
                district_override=district_override,
            )
            if structured:
                refs = list(sources)
                return {
                    "answer": structured,
                    "references": refs,
                    "retrieved": [],
                    "topic": "crop_profitability",
                }
        if self._is_pesticide_intent(normalized_question):
            crop_hint = self._extract_preferred_crop_from_context(context_part)
            result = self._structured_pesticide_advice(normalized_question, crop_hint=crop_hint)
            result["topic"] = "pesticide"
            return result
        if self._looks_like_location_only(farmer_question):
            return {
                "answer": "कृपया बताएं कि आप मौसम पूछ रहे हैं या भाव/कीमत?",
                "references": [],
                "retrieved": [],
                "topic": "clarification",
            }

        normalized_query = (
            f"{context_part} किसान का प्रश्न: {normalized_question}".strip()
            if context_part
            else normalized_question
        )

        try:
            self._ensure_rag_components(load_generator=False)
        except Exception:
            return {
                "answer": "अभी यह सवाल local source से नहीं निकल पाया और RAG model उपलब्ध नहीं है। कृपया सवाल में फसल/जिला साफ लिखें या थोड़ी देर बाद फिर प्रयास करें।",
                "references": [],
                "retrieved": [],
                "topic": "rag",
            }
        if self.embedder is None or self.retriever is None or self.generator is None:
            return {
                "answer": "मॉडल अभी उपलब्ध नहीं है। कृपया थोड़ी देर बाद फिर प्रयास करें।",
                "references": [],
                "retrieved": [],
                "topic": "rag",
            }

        qvec = self.embedder.encode([normalized_question])[0]
        retrieved = self.retriever.retrieve(qvec, k=self.top_k)
        prompt = build_prompt(normalized_query, retrieved)
        try:
            response = self.generator.generate(prompt)
            if self._is_low_quality_response(response):
                response = self._fallback_answer(retrieved, normalized_question)
        except Exception:
            response = self._fallback_answer(retrieved, normalized_question)
        return {
            "answer": response,
            "references": [r.get("source_file") for r in retrieved],
            "retrieved": retrieved,
            "topic": "rag",
        }

    def _is_cost_of_production_query(self, text: str) -> bool:
        t = text.strip().lower()
        phrases = [
            "kitni lagat", "lagat kitni", "cost kitni", "cost kya", "per quintal cost",
            "1 quintal", "ek quintal", "one quintal", "grow karne", "ugane me", "उगाने में",
            "production cost", "cost of production", "kitne me padta", "कितनी लागत", "लागत कितनी",
        ]
        return any(p in t for p in phrases)

    def _answer_sugarcane_cost_query(self, question: str, context_part: str) -> str | None:
        crop = self._extract_crop_from_query(question) or self._extract_preferred_crop_from_context(context_part)
        if (crop or '').lower() != 'sugarcane':
            return None
        if not self._is_cost_of_production_query(question):
            return None
        snap = get_sugarcane_cost_snapshot()
        if not snap:
            return None
        yield_info = load_latest_up_yield_qtl_per_acre('Sugarcane', season='Annual') or load_latest_up_yield_qtl_per_acre('Sugarcane', season='Rabi')
        lines = [
            'गन्ना के लिए source-backed लागत संकेत:',
        ]
        a2fl = snap.get('a2fl_basic_per_qtl')
        modified = snap.get('modified_a2fl_per_qtl')
        c2 = snap.get('c2_basic_per_qtl')
        frp = snap.get('frp_per_qtl')
        if a2fl is not None:
            lines.append(f"- CACP के अनुसार 10.25% basic recovery rate पर A2+FL लागत लगभग ₹{int(float(a2fl))}/क्विंटल है।")
        if modified is not None:
            lines.append(f"- transport + insurance जोड़ने पर modified A2+FL लागत लगभग ₹{int(float(modified))}/क्विंटल है।")
        if c2 is not None:
            lines.append(f"- अगर C2 basis देखें, तो लागत लगभग ₹{int(float(c2))}/क्विंटल है।")
        if frp is not None and modified is not None:
            margin = float(frp) - float(modified)
            lines.append(f"- CACP FRP 2025-26: ₹{int(float(frp))}/क्विंटल; यानी modified A2+FL पर लगभग ₹{int(margin)}/क्विंटल का gross margin बनता है।")
        elif frp is not None and a2fl is not None:
            margin = float(frp) - float(a2fl)
            lines.append(f"- CACP FRP 2025-26: ₹{int(float(frp))}/क्विंटल; यानी A2+FL पर लगभग ₹{int(margin)}/क्विंटल का gross margin बनता है।")
        if yield_info and yield_info.get('yield_qtl_per_acre'):
            y = float(yield_info['yield_qtl_per_acre'])
            lines.append(f"- UPAG yield संकेत: लगभग {y:.1f} qtl/acre ({yield_info.get('crop_year','')}).")
            if modified is not None:
                acre_cost = y * float(modified)
                lines.append(f"- इसी हिसाब से modified A2+FL लागत लगभग ₹{int(acre_cost)}/acre बैठती है।")
        lines.extend([
            '',
            'इसका आसान मतलब:',
            '- A2+FL: खेत पर जेब से होने वाला खर्च + परिवार की मेहनत।',
            '- C2: A2+FL के ऊपर जमीन का किराया और fixed capital का ब्याज भी जोड़कर निकाली गई पूरी लागत।',
            '- इसलिए A2+FL practical working cost दिखाती है, जबकि C2 ज्यादा broad cost दिखाती है।',
            '- exact लागत variety, recovery, पानी, labour और transport distance के हिसाब से बदल सकती है।',
        ])
        return "\n".join(lines)

    def _answer_generic_cacp_cost_query(self, question: str, context_part: str) -> str | None:
        if not self._is_cost_of_production_query(question):
            return None
        crop = self._extract_crop_from_query(question) or self._extract_preferred_crop_from_context(context_part)
        if not crop or (crop or "").lower() == "sugarcane":
            return None
        snap = get_cacp_cost_for_crop(crop)
        if not snap:
            return None
        crop_label = self._crop_display_label(crop)
        lines = [
            f"{crop_label} के लिए CACP लागत संकेत:",
            f"- CACP {str(snap.get('report_kind', '')).title()} report {snap.get('season', '')} के अनुसार A2 लागत लगभग ₹{int(float(snap['a2_per_qtl']))}/क्विंटल है।",
            f"- A2+FL लागत लगभग ₹{int(float(snap['a2fl_per_qtl']))}/क्विंटल है।",
            f"- C2 लागत लगभग ₹{int(float(snap['c2_per_qtl']))}/क्विंटल है।",
            "",
            "इसका आसान मतलब:",
            "- A2: जेब से होने वाला सीधा खर्च। इसमें बीज, खाद, दवा, सिंचाई, मशीन, मजदूरी जैसी नकद/वास्तविक लागत आती है।",
            "- FL: परिवार के लोगों की मेहनत (Family Labour) की कीमत।",
            "- A2+FL: A2 के साथ परिवार की मेहनत जोड़कर निकाली गई लागत।",
            "- C2: A2+FL के ऊपर जमीन का किराया और fixed capital का ब्याज भी जोड़कर निकाली गई पूरी लागत।",
            "",
            "व्यवहार में समझें:",
            "- A2 कम दिखेगी, क्योंकि इसमें परिवार की मेहनत और जमीन का किराया पूरा नहीं जुड़ता।",
            "- A2+FL को किसान-स्तर की अधिक practical लागत माना जा सकता है।",
            "- C2 सबसे broad लागत है, इसलिए यह आम तौर पर सबसे ज्यादा होती है।",
        ]
        return "\n".join(lines)

    def _is_profitability_followup_intent(self, text: str) -> bool:
        t = text.strip().lower()
        phrases = [
            "lagat kaese",
            "lagat kaise",
            "cost kaise",
            "cost kese",
            "profit kaise",
            "profit kese",
            "revenue kaise",
            "hisab kaise",
            "calculation kaise",
            "calculation kese",
            "kaise nikali",
            "kaise nikala",
            "kese nikali",
            "kese nikala",
            "लागत कैसे",
            "लागत किस आधार",
            "लागत कैसे निकाली",
            "लागत कैसे निकाला",
            "मुनाफा कैसे",
            "लाभ कैसे",
            "हिसाब कैसे",
            "कैसे निकाली",
            "कैसे निकाला",
        ]
        if any(p in t for p in phrases):
            return True
        has_cost_word = any(p in t for p in ("lagat", "cost", "लागत", "profit", "लाभ", "hisab", "हिसाब", "calculation"))
        has_how_word = any(p in t for p in ("kaise", "kese", "कैसे"))
        has_derived_word = any(p in t for p in ("nikali", "nikala", "निकाली", "निकाला"))
        return has_cost_word and has_how_word and has_derived_word

    def _answer_crop_cost_method_query(self, question: str, context_part: str) -> str | None:
        crop = self._extract_crop_from_query(question) or self._extract_preferred_crop_from_context(context_part)
        if not crop:
            return None
        q = question.strip().lower()
        has_cost_word = any(p in q for p in ("lagat", "cost", "लागत", "hisab", "हिसाब", "calculation"))
        has_how_word = any(p in q for p in ("kaise", "kese", "कैसे"))
        has_derived_word = any(p in q for p in ("nikali", "nikala", "निकाली", "निकाला"))
        if not (has_cost_word and has_how_word and has_derived_word):
            return None

        crop_label = self._crop_display_label(crop)
        if crop.lower() == "sugarcane":
            snap = get_sugarcane_cost_snapshot()
            yield_info = load_latest_up_yield_qtl_per_acre("Sugarcane", season="Annual") or load_latest_up_yield_qtl_per_acre("Sugarcane", season="Rabi")
            if not snap:
                return None
            lower_qtl = snap.get("modified_a2fl_per_qtl") or snap.get("a2fl_basic_per_qtl")
            upper_qtl = snap.get("modified_c2_per_qtl") or snap.get("c2_basic_per_qtl") or snap.get("c2_per_qtl")
            if lower_qtl is None or upper_qtl is None:
                return None
            season = str(snap.get("season") or "").strip()
            lines = [
                f"{crop_label} की लागत का हिसाब इस तरह निकाला गया:",
                f"- base source: CACP Sugarcane report {season}".strip(),
                f"- lower cost band: paid-out cost + family labour + transport/insurance मिलाकर लगभग ₹{int(float(lower_qtl))}/क्विंटल।",
                f"- upper cost band: पूरी लागत के हिसाब से लगभग ₹{int(float(upper_qtl))}/क्विंटल।",
            ]
            if yield_info and yield_info.get("yield_qtl_per_acre"):
                y = float(yield_info["yield_qtl_per_acre"])
                lines.extend([
                    f"- UPAG के अनुसार yield लगभग {y:.1f} qtl/acre ली गई।",
                    f"- इसलिए per acre लागत लगभग ₹{int(y * float(lower_qtl))} से ₹{int(y * float(upper_qtl))} निकाली गई।",
                ])
            lines.extend([
                "",
                "आसान मतलब:",
                "- lower band practical working cost दिखाती है।",
                "- upper band में जमीन और capital जैसी broader लागत भी जुड़ती है।",
            ])
            return "\n".join(lines)

        snap = get_cacp_cost_for_crop(crop)
        if not snap:
            return None
        a2 = snap.get("a2_per_qtl")
        a2fl = snap.get("a2fl_per_qtl")
        c2 = snap.get("c2_per_qtl")
        if a2 is None or a2fl is None or c2 is None:
            return None
        yield_info = load_latest_up_yield_qtl_per_acre(crop, season="Rabi") or load_latest_up_yield_qtl_per_acre(crop, season="Kharif") or load_latest_up_yield_qtl_per_acre(crop)
        report_kind = str(snap.get("report_kind", "")).title().strip()
        season = str(snap.get("season", "")).strip()
        lines = [
            f"{crop_label} की लागत का हिसाब इस तरह निकाला गया:",
            f"- base source: CACP {report_kind} report {season}".strip(),
            f"- A2 लगभग ₹{int(float(a2))}/क्विंटल लिया गया।",
            f"- A2+FL लगभग ₹{int(float(a2fl))}/क्विंटल लिया गया।",
            f"- C2 लगभग ₹{int(float(c2))}/क्विंटल लिया गया।",
        ]
        if yield_info and yield_info.get("yield_qtl_per_acre"):
            y = float(yield_info["yield_qtl_per_acre"])
            lines.extend([
                f"- UPAG के अनुसार yield लगभग {y:.1f} qtl/acre ली गई।",
                f"- इसलिए per acre लागत लगभग ₹{int(y * float(a2fl))} से ₹{int(y * float(c2))} निकाली गई।",
            ])
        lines.extend([
            "",
            "आसान मतलब:",
            "- A2: जेब से होने वाला सीधा खर्च।",
            "- A2+FL: A2 के साथ परिवार की मेहनत जोड़कर निकाली गई practical लागत।",
            "- C2: A2+FL के ऊपर जमीन का किराया और fixed capital का ब्याज जोड़कर निकाली गई पूरी लागत।",
        ])
        return "\n".join(lines)

    def _official_cost_range_for_profitability(self, crop: str, yield_qtl_per_acre: float) -> dict | None:
        if (crop or '').lower() == 'sugarcane':
            snap = get_sugarcane_cost_snapshot()
            if not snap:
                return None
            lower_qtl = snap.get('modified_a2fl_per_qtl') or snap.get('a2fl_basic_per_qtl')
            upper_qtl = snap.get('modified_c2_per_qtl') or snap.get('c2_basic_per_qtl') or lower_qtl
            if lower_qtl is None or upper_qtl is None:
                return None
            season = ''
            source_name = str(snap.get('source_name') or 'CACP Sugarcane report')
            m = re.search(r'(20\d{2}-\d{2})', source_name)
            if m:
                season = m.group(1)
            return {
                'cost_min': float(lower_qtl) * float(yield_qtl_per_acre),
                'cost_max': float(upper_qtl) * float(yield_qtl_per_acre),
                'source': f"CACP{(' ' + season) if season else ''}",
                'basis': 'paid-out cost + family labour (transport/insurance सहित) से पूरी लागत',
            }
        snap = get_cacp_cost_for_crop(crop)
        if not snap:
            return None
        a2fl = snap.get('a2fl_per_qtl')
        c2 = snap.get('c2_per_qtl')
        if a2fl is None or c2 is None:
            return None
        report_kind = str(snap.get('report_kind', '')).title()
        season = str(snap.get('season', '')).strip()
        src = 'CACP'
        if report_kind or season:
            src = f"CACP {report_kind} {season}".strip()
        return {
            'cost_min': float(a2fl) * float(yield_qtl_per_acre),
            'cost_max': float(c2) * float(yield_qtl_per_acre),
            'source': src,
            'basis': 'paid-out cost + family labour से पूरी लागत',
        }

    def _explain_profitability_method(self, context_part: str, question: str) -> str:
        district = self._extract_district(context_part) or "Meerut"
        season = self._extract_season(context_part) or "Rabi"
        preferred_crop = None
        pref = re.search(r"(?:पसंदीदा फसल|Preferred crop):\s*([^|]+)", context_part, flags=re.IGNORECASE)
        if pref:
            preferred_crop = pref.group(1).strip()
        lines = [
            "लागत/लाभ निकालने का तरीका:",
            f"- जिला: {district}",
            f"- मौसम: {season}",
        ]
        if preferred_crop and preferred_crop not in {"कोई नहीं", "None"}:
            lines.append(f"- संदर्भ फसल: {preferred_crop}")
        lines.extend(
            [
                "- आय का हिसाब: उपज/acre × भाव/qtl",
                "- उपज: जहाँ उपलब्ध हो वहाँ UPAG yield लिया गया; नहीं मिलने पर Western UP baseline yield लिया गया।",
                "- भाव: पहले जिला-स्तर Agmarknet भाव लिया गया; न मिलने पर baseline/MSP-SAP estimate लिया गया।",
                "- लागत: जहाँ CACP उपलब्ध है वहाँ A2+FL को lower band और C2 को upper band माना गया; नहीं मिलने पर baseline indicative range लिया गया।",
                "- फिर पानी की जरूरत और रोग/कीट दबाव के आधार पर लागत में हल्का adjustment किया गया।",
                "- अंतिम लाभ = अनुमानित आय − adjusted लागत range",
                "",
                "ध्यान दें:",
                "- यह planning estimate है, exact खेत-स्तर लेखा नहीं।",
                "- इसमें आपकी असली मजदूरी दर, सिंचाई खर्च, fertilizer bill, variety, lease rent और financing cost अलग से नहीं जोड़ी गई है।",
                "- अगर आप चाहें तो मैं अगला जवाब crop-wise लागत breakdown में दे सकता हूँ, जैसे: बीज, खाद, सिंचाई, मजदूरी, दवा अलग-अलग।",
            ]
        )
        return "\n".join(lines)

    def _is_weather_impact_intent(self, text: str, context_part: str = "") -> bool:
        t = text.strip().lower()
        weather_words = [
            "बारिश", "बारिस", "barish", "baarish", "rain", "rainfall",
            "मौसम", "mausam", "weather", "तापमान", "temperature",
            "गरमी", "गर्मी", "heat", "ठंड", "cold", "पाला", "frost",
            "हवा", "wind",
        ]
        impact_words = [
            "नुकसान", "nuksan", "nauksan", "damage", "affect", "impact",
            "asar", "असर", "खराब", "kharab", "bachav", "बचाव",
            "problem", "दिक्कत", "stress", "safe", "बचेगी", "hoga", "hogi",
        ]
        crop_words = ["फसल", "fasal", "crop"]
        has_weather = any(w in t for w in weather_words)
        has_impact = any(w in t for w in impact_words)
        has_crop = any(w in t for w in crop_words) or self._extract_crop_from_query(text) is not None
        has_ctx_crop = self._extract_preferred_crop_from_context(context_part) not in {None, "", "कोई नहीं", "None"}
        return has_weather and has_impact and (has_crop or has_ctx_crop)

    def _crop_weather_risk_type(self, crop: str) -> str:
        c = (crop or "").lower()
        if c in {"wheat", "rice", "maize", "sorghum", "millets", "bajra", "barley"}:
            return "grain"
        if c in {"sugarcane"}:
            return "cane"
        if c in {"mustard", "groundnut", "sesame", "sunflower", "soyabean", "soybean"}:
            return "oilseed"
        if c in {"green gram", "black gram", "pigeon pea", "chickpea", "pea", "lentil", "arhar", "moong", "urad", "chana"}:
            return "pulse"
        if c in {"tomato", "brinjal", "chilli", "cabbage", "cauliflower", "onion", "potato", "okra"}:
            return "vegetable"
        return "generic"

    def _weather_impact_advice(self, context_part: str, question: str) -> str:
        crop = self._extract_crop_from_query(question) or self._extract_preferred_crop_from_context(context_part)
        if not crop or crop in {"कोई नहीं", "None"}:
            return (
                "यह साधारण मौसम सवाल नहीं, फसल-प्रभाव सवाल है। "
                "कृपया फसल का नाम और अगर संभव हो तो उसकी अवस्था भी लिखें, जैसे: "
                "`गेहूं में बालियां निकल रही हैं` या `धान की रोपाई हुई है`।"
            )
        crop_label = self._crop_display_label(crop)
        risk_type = self._crop_weather_risk_type(crop)
        lines = [f"फसल-प्रभाव सलाह: {crop_label}"]
        q = question.lower()
        if any(w in q for w in ["बारिश", "बारिस", "barish", "baarish", "rain", "rainfall"]):
            if risk_type == "grain":
                lines.append("- ज्यादा बारिश या पानी भराव से lodging, fungal रोग और दाना/बालियों पर असर पड़ सकता है।")
                lines.append("- खासकर flowering या grain filling अवस्था में नुकसान ज्यादा हो सकता है।")
                lines.append("- खेत में जल निकासी साफ रखें और लगातार नमी रहने पर रोग/झुलसा/रतुआ की निगरानी करें।")
            elif risk_type == "cane":
                lines.append("- लगातार बारिश से waterlogging, जड़ों पर असर, गिरना और fungal disease का खतरा बढ़ सकता है।")
                lines.append("- गन्ने में नालियां साफ रखें ताकि पानी रुके नहीं।")
            elif risk_type == "oilseed":
                lines.append("- ज्यादा बारिश से फूल झड़ना, fungal infection और पानी भराव का खतरा बढ़ सकता है।")
                lines.append("- जल निकासी और पत्तियों/तनों पर रोग के लक्षणों की निगरानी जरूरी है।")
            elif risk_type == "pulse":
                lines.append("- ज्यादा बारिश से जड़ सड़न, फूल/फली झड़ना और fungal रोग बढ़ सकते हैं।")
                lines.append("- खेत में पानी रुकने न दें और रोग के शुरुआती लक्षण देखें।")
            elif risk_type == "vegetable":
                lines.append("- ज्यादा बारिश से पत्तियों और फलों में सड़न, fungal disease और root stress बढ़ सकता है।")
                lines.append("- मल्च/drainage और disease scouting पर ध्यान दें।")
            else:
                lines.append("- ज्यादा बारिश से पानी भराव, fungal disease और बढ़वार पर असर पड़ सकता है।")
                lines.append("- सबसे पहले खेत की drainage और पौधों पर रोग/सड़न के लक्षण देखें।")
        elif any(w in q for w in ["तापमान", "temperature", "गरमी", "गर्मी", "heat", "ठंड", "cold", "पाला", "frost"]):
            lines.append("- तापमान का असर फसल की अवस्था पर निर्भर करता है; बहुत ज्यादा गर्मी, ठंड या पाला flowering और बढ़वार को नुकसान पहुँचा सकता है।")
            lines.append("- ऐसी स्थिति में सिंचाई, mulching और stage-specific सुरक्षा उपाय जरूरी हो सकते हैं।")
        else:
            lines.append("- मौसम का असर फसल और उसकी अवस्था पर निर्भर करता है।")
        lines.append("- और सटीक सलाह के लिए फसल की अवस्था लिखें, जैसे: बुवाई, रोपाई, flowering, दाना भराव या कटाई के पास।")
        return "\n".join(lines)

    def _is_pesticide_intent(self, text: str) -> bool:
        t = text.lower()
        keys = [
            "pesticide",
            "insecticide",
            "fungicide",
            "herbicide",
            "dawai",
            "dawa",
            "dose",
            "dosage",
            "borer",
            "stem borer",
            "smut",
            "bunt",
            "burnt",
            "rot",
            "red rot",
            "wilt",
            "spot",
            "blight",
            "rust",
            "mildew",
            "fungus",
            "fungal",
            "kida",
            "kide",
            "kido",
            "keeda",
            "keede",
            "keet",
            "कीटनाशक",
            "फफूंदनाशी",
            "घासनाशी",
            "दवा",
            "डोज",
            "खुराक",
            "स्प्रे",
            "छिड़काव",
            "कीट",
            "रोग",
        ]
        return any(k in t for k in keys) or self._looks_like_pesticide_name_query(t)

    def _extract_preferred_crop_from_context(self, context_part: str) -> str | None:
        if not context_part:
            return None
        pref = re.search(r"(?:पसंदीदा फसल|Preferred crop):\s*([^|]+)", context_part, flags=re.IGNORECASE)
        if not pref:
            return None
        raw = pref.group(1).strip()
        if not raw or raw.lower() in {"कोई नहीं", "none", "na", "n/a"}:
            return None
        return self._extract_crop_from_query(raw) or raw

    def _structured_pesticide_advice(self, question: str, crop_hint: str | None = None) -> dict:
        crop = self._extract_crop_from_query(question) or crop_hint
        pesticide_name = self._extract_pesticide_name_from_query(question)
        if not pesticide_name:
            pesticide_name = self._infer_pesticide_name_from_query_tokens(question, crop=crop)
        disease_terms = self._extract_disease_terms_from_query(question)
        issue_mode = self._generic_issue_mode(question)
        if pesticide_name:
            chem_lines, chem_sources = self._extract_rows_for_pesticide_name(
                pesticide_name,
                crop=crop,
                limit=4,
            )
            if not chem_lines:
                chem_lines, chem_sources = self._extract_rows_for_pesticide_text(
                    question,
                    crop=crop,
                    limit=4,
                )
            if chem_lines:
                intro = [
                    "संरचित कीटनाशक सलाह:",
                    f"- दवा/रसायन: {pesticide_name}",
                ]
                if crop:
                    intro.append(f"- संदर्भ फसल: {self._crop_name_hi(crop)}")
                intro.append("- यह दवा इन फसलों/रोग-कीट स्थितियों में मिलती है:")
                intro.extend([f"  • {line}" for line in chem_lines[:4]])
                intro.append("- अगर आप चाहें, तो फसल या रोग का नाम लिखें; फिर मैं इसी दवा का सबसे सही dose, dilution और PHI उसी case के हिसाब से बता दूँगा।")
                return {"answer": "\n".join(intro), "references": chem_sources, "retrieved": []}
            if crop:
                return {
                    "answer": (
                        "संरचित कीटनाशक सलाह:\n"
                        f"- दवा/रसायन: {pesticide_name}\n"
                        f"- संदर्भ फसल: {self._crop_name_hi(crop)}\n"
                        "- इस फसल के लिए इस दवा का स्पष्ट structured record नहीं मिला।\n"
                        "- कृपया रोग/कीट का नाम या लक्षण लिखें, फिर मैं इस फसल के लिए सही दवा/विकल्प बता दूँगा।"
                    ),
                    "references": [],
                    "retrieved": [],
                }
        if crop and not disease_terms and self._is_generic_disease_query(question):
            common_issues = self._extract_common_crop_issues(crop, limit=3, issue_mode=issue_mode)
            sample_sources: list[str] = []
            if common_issues:
                issue_lines = "\n".join([f"  • {issue}" for issue in common_issues[:3]])
                mode_label = {
                    "fungal": "फफूंद/फंगल रोग",
                    "pest": "कीट",
                    "disease": "रोग",
                    "general": "रोग/कीट",
                }.get(issue_mode, "रोग/कीट")
                symptom_prompt = {
                    "fungal": "- अगर exact नाम नहीं पता, तो लक्षण लिखें: पत्तियों पर धब्बे, सफेद परत, काला/लाल सड़न, तना सड़ना, या सूखना।",
                    "pest": "- अगर exact नाम नहीं पता, तो लक्षण लिखें: पत्ती कटना, छेद, रस चूसना, कीड़ा दिखना, तना छेदना, या जड़ नुकसान।",
                    "disease": "- अगर exact नाम नहीं पता, तो लक्षण लिखें: धब्बे, सड़न, झुलसा, सूखना, या पत्तियों का रंग बदलना।",
                    "general": "- अगर exact नाम नहीं पता, तो लक्षण लिखें: धब्बे/सड़न/छेद/कीड़ा दिखना/पत्ती मुड़ना।",
                }.get(issue_mode, "- अगर exact नाम नहीं पता, तो मुख्य लक्षण लिखें।")
                return {
                    "answer": (
                        "संरचित कीटनाशक सलाह:\n"
                        f"- फसल: {self._crop_name_hi(crop)}\n"
                        f"- इस फसल में आम तौर पर ये 2-3 {mode_label} ज़्यादा देखे जाते हैं:\n"
                        f"{issue_lines}\n"
                        f"{symptom_prompt}\n"
                        "- आप इनमें से किसी एक का नाम या लक्षण लिखें, फिर मैं उसी के हिसाब से सही दवा, dose और PHI बता दूँगा।"
                    ),
                    "references": sample_sources,
                    "retrieved": [],
                }
        if crop:
            db_lines, db_sources = self._extract_pesticides_from_db(crop, disease_terms=disease_terms, limit=8)
            if db_lines:
                disease_label = ", ".join(disease_terms[:3]) if disease_terms else "दिए गए रोग/कीट"
                lines = [
                    "संरचित कीटनाशक सलाह:",
                    f"- फसल: {self._crop_name_hi(crop)}",
                    f"- रोग/कीट मिलान: {disease_label}",
                    "- सलाह:",
                ]
                lines.extend([f"  • {line}" for line in db_lines[:3]])
                lines.append("- समझें: `सक्रिय तत्व (a.i.)` दवा का असर करने वाला chemical हिस्सा है, `उत्पाद मात्रा (formulation)` बाजार से खरीदी जाने वाली दवा की कुल मात्रा है, और `कितने पानी में घोलें` का मतलब spray के लिए पानी की मात्रा है।")
                lines.append("- अगर कहीं `g/ml` एक साथ लिखा हो, तो source table में unit साफ़ नहीं है; ऐसे case में label/packet या कृषि अधिकारी से unit verify करके ही spray करें।")
                lines.append("- छिड़काव/बीज उपचार से पहले उत्पाद लेबल, PHI और स्थानीय कृषि अधिकारी की सलाह जरूर मिलाएँ।")
                return {"answer": "\n".join(lines), "references": db_sources, "retrieved": []}

        self._ensure_rag_components()
        if self.embedder is None or self.retriever is None:
            return {
                "answer": "पेस्ट/रोग संबंधी जानकारी अभी उपलब्ध नहीं है। कृपया बाद में प्रयास करें।",
                "references": [],
                "retrieved": [],
            }
        qvec = self.embedder.encode([question])[0]
        retrieved = self.retriever.retrieve(qvec, k=max(5, self.top_k))
        snippets = [r.get("text", "") for r in retrieved if r.get("text")]
        if not snippets:
            return {
                "answer": "पेस्ट/रोग संबंधी जानकारी नहीं मिली। कृपया फसल और रोग का नाम बताएं।",
                "references": [],
                "retrieved": [],
            }

        # Heuristic extraction from retrieved text
        text = self._clean_retrieved_text("\n".join(snippets))
        pesticides = self._extract_pesticide_names(text)
        doses = self._extract_dose_lines(text)
        waiting = self._extract_waiting_period(text)
        extra_lines = []
        extra_sources = []
        if not pesticides:
            if crop:
                extra_lines, extra_sources = self._extract_pesticides_from_db(crop, disease_terms=disease_terms)
                if not extra_lines:
                    extra_lines, extra_sources = self._extract_pesticides_from_pdfs(crop)
                if extra_lines:
                    pesticides = self._extract_pesticide_names("\n".join(extra_lines)) or pesticides
                    doses = doses or self._extract_dose_lines("\n".join(extra_lines))

        lines = []
        lines.append("संरचित कीटनाशक सलाह:")
        if pesticides:
            lines.append(f"- सुझाई गई दवाएँ: {', '.join(pesticides[:5])}")
        else:
            lines.append("- सुझाई गई दवाएँ: स्पष्ट नाम नहीं मिला (कृपया रोग/कीट बताएं)")
        if doses:
            lines.append(f"- खुराक/डोज़: {doses[0]}")
        if waiting:
            lines.append(f"- सुरक्षा अवधि (PHI): {waiting}")
        if extra_lines:
            lines.append("- स्रोत से उदाहरण पंक्तियाँ:")
            lines.extend([f"  • {l}" for l in extra_lines[:3]])
        lines.append("- छिड़काव से पहले लेबल निर्देश और राज्य सलाह देखें।")

        return {
            "answer": "\n".join(lines),
            "references": list({*(r.get("source_file") for r in retrieved), *extra_sources}),
            "retrieved": retrieved,
        }

    def _is_generic_disease_query(self, text: str) -> bool:
        t = text.lower()
        generic_markers = [
            "rog",
            "bimari",
            "disease",
            "fungus",
            "fungal",
            "fugal",
            "dawai",
            "dawa",
            "spray",
            "कीट",
            "रोग",
            "बीमारी",
            "दवा",
            "स्प्रे",
            "फफूंद",
            "फंगस",
        ]
        specific_markers = [
            "rust",
            "smut",
            "bunt",
            "blight",
            "mildew",
            "borer",
            "hopper",
            "aphid",
            "termite",
            "blast",
            "rot",
            "wilt",
            "spot",
            "leaf folder",
            "तना छेदक",
            "रतुआ",
            "झुलसा",
            "कंडुआ",
            "बंट",
            "दीमक",
        ]
        return any(k in t for k in generic_markers) and not any(k in t for k in specific_markers)

    def _is_fungal_query(self, text: str) -> bool:
        t = text.lower()
        return any(k in t for k in ["fungus", "fungal", "fugal", "फफूंद", "फंगस"])

    def _is_pest_only_query(self, text: str) -> bool:
        t = text.lower()
        pest_terms = [
            "kida", "kide", "kido", "keeda", "keede", "keet", "pest", "insect",
            "borer", "hopper", "aphid", "termite", "mite", "caterpillar",
            "कीट", "कीड़ा", "कीड़े", "दीमक", "माहू",
        ]
        disease_terms = ["fungus", "fungal", "fugal", "rog", "bimari", "disease", "फफूंद", "फंगस", "रोग", "बीमारी"]
        return any(k in t for k in pest_terms) and not any(k in t for k in disease_terms)

    def _is_disease_only_query(self, text: str) -> bool:
        t = text.lower()
        if self._is_fungal_query(text):
            return True
        return any(k in t for k in ["rog", "bimari", "disease", "रोग", "बीमारी"])

    def _generic_issue_mode(self, text: str) -> str:
        if self._is_fungal_query(text):
            return "fungal"
        if self._is_pest_only_query(text):
            return "pest"
        if self._is_disease_only_query(text):
            return "disease"
        return "general"

    def _extract_common_crop_issues(self, crop: str, limit: int = 4, issue_mode: str = "general") -> list[str]:
        if not self.cfg.db_path:
            return []
        try:
            conn = sqlite3.connect(self.cfg.db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT disease_name_en, disease_name_hi, quality_status
                FROM pesticide_recommendations
                WHERE lower(crop_name)=lower(?)
                """,
                (crop,),
            ).fetchall()
            conn.close()
        except Exception:
            return []
        ranked: list[tuple[int, str]] = []
        seen: set[str] = set()
        disease_tokens = ["rot", "rust", "smut", "bunt", "blight", "mildew", "wilt", "spot", "canker", "scab", "blast", "disease", "सड़न", "रतुआ", "झुलसा", "कंडुआ", "फफूंद"]
        pest_tokens = ["borer", "hopper", "aphid", "termite", "mite", "bug", "grub", "fly", "caterpillar", "pyrilla", "leaf folder", "तना छेदक", "दीमक", "माहू", "कीट"]
        for r in rows:
            label = (r["disease_name_hi"] or r["disease_name_en"] or "").strip()
            if not label:
                continue
            if not self._is_issue_label_useful(label):
                continue
            for part in self._split_issue_label(label):
                part_l = part.lower()
                if issue_mode == "fungal" and not any(tok in part_l for tok in disease_tokens):
                    continue
                if issue_mode == "pest" and not any(tok in part_l for tok in pest_tokens):
                    continue
                if issue_mode == "disease" and not any(tok in part_l for tok in disease_tokens):
                    continue
                cleaned = self._translate_disease_name(part)
                key = re.sub(r"[^a-z0-9\u0900-\u097F]+", "", cleaned.lower())
                if not key or key in seen:
                    continue
                seen.add(key)
                score = self._issue_priority_score(part, r["quality_status"] or "")
                ranked.append((score, cleaned))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [label for _score, label in ranked[:limit]]

    def _extract_generic_crop_samples(self, crop: str, limit: int = 4) -> tuple[list[str], list[str]]:
        if not self.cfg.db_path:
            return [], []
        try:
            conn = sqlite3.connect(self.cfg.db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT crop_name, disease_name_en, disease_name_hi, pesticide_name,
                       ai_g, formulation, dilution, dose_text, waiting_period_days,
                       ai_unit, formulation_unit, dilution_unit, waiting_period_unit,
                       unit_source, source_file, quality_status
                FROM pesticide_recommendations
                WHERE lower(crop_name)=lower(?)
                ORDER BY CASE quality_status WHEN 'valid' THEN 0 ELSE 1 END
                LIMIT 300
                """,
                (crop,),
            ).fetchall()
            conn.close()
        except Exception:
            return [], []
        lines: list[str] = []
        sources: list[str] = []
        seen: set[str] = set()
        for r in rows:
            raw_label = (r["disease_name_hi"] or r["disease_name_en"] or "").strip()
            if not raw_label or not self._is_issue_label_useful(raw_label):
                continue
            disease = r["disease_name_hi"] or self._translate_disease_name(r["disease_name_en"] or "")
            issue_key = re.sub(r"[^a-z0-9\u0900-\u097F]+", "", disease.lower())
            if not issue_key or issue_key in seen:
                continue
            dose_parts = []
            if r["ai_g"]:
                dose_parts.append(f"a.i.: {self._format_value_with_unit(r['ai_g'], r['ai_unit'])}")
            if r["formulation"]:
                dose_parts.append(f"formulation: {self._format_value_with_unit(r['formulation'], r['formulation_unit'])}")
            if r["dilution"]:
                dose_parts.append(f"पानी/घोल: {self._format_value_with_unit(r['dilution'], r['dilution_unit'])}")
            if not dose_parts and r["dose_text"]:
                dose_parts.append(r["dose_text"])
            waiting = self._format_value_with_unit(r["waiting_period_days"], r["waiting_period_unit"])
            waiting_part = f" | PHI: {waiting}" if waiting else ""
            line = f"{disease} | दवा: {r['pesticide_name'] or 'नाम उपलब्ध नहीं'}"
            if dose_parts:
                line += f" | {'; '.join(dose_parts)}"
            line += waiting_part
            lines.append(line.strip())
            seen.add(issue_key)
            if r["source_file"]:
                sources.append(r["source_file"])
            if len(lines) >= limit:
                break
        return lines, sorted(set(sources))

    def _extract_pesticides_for_issue(self, crop: str, issue: str, limit: int = 2) -> tuple[list[str], list[str]]:
        if not self.cfg.db_path:
            return [], []
        issue_norm = issue.lower().strip()
        if not issue_norm:
            return [], []
        try:
            conn = sqlite3.connect(self.cfg.db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT crop_name, disease_name_en, disease_name_hi, pesticide_name,
                       ai_g, formulation, dilution, dose_text, waiting_period_days,
                       ai_unit, formulation_unit, dilution_unit, waiting_period_unit,
                       unit_source, source_file, quality_status
                FROM pesticide_recommendations
                WHERE lower(crop_name)=lower(?)
                  AND (
                    lower(coalesce(disease_name_en, '')) LIKE ?
                    OR lower(coalesce(disease_name_hi, '')) LIKE ?
                  )
                ORDER BY CASE quality_status WHEN 'valid' THEN 0 ELSE 1 END
                LIMIT 50
                """,
                (crop, f"%{issue_norm}%", f"%{issue_norm}%"),
            ).fetchall()
            conn.close()
        except Exception:
            return [], []
        if not rows:
            return [], []
        lines = []
        sources = []
        for r in rows[:limit]:
            disease = r["disease_name_hi"] or self._translate_disease_name(r["disease_name_en"] or "")
            dose_parts = []
            if r["ai_g"]:
                dose_parts.append(f"a.i.: {self._format_value_with_unit(r['ai_g'], r['ai_unit'])}")
            if r["formulation"]:
                dose_parts.append(f"formulation: {self._format_value_with_unit(r['formulation'], r['formulation_unit'])}")
            if r["dilution"]:
                dose_parts.append(f"पानी/घोल: {self._format_value_with_unit(r['dilution'], r['dilution_unit'])}")
            if not dose_parts and r["dose_text"]:
                dose_parts.append(r["dose_text"])
            waiting = self._format_value_with_unit(r["waiting_period_days"], r["waiting_period_unit"])
            waiting_part = f" | PHI: {waiting}" if waiting else ""
            line = f"{disease} | दवा: {r['pesticide_name'] or 'नाम उपलब्ध नहीं'}"
            if dose_parts:
                line += f" | {'; '.join(dose_parts)}"
            line += waiting_part
            lines.append(line.strip())
            if r["source_file"]:
                sources.append(r["source_file"])
        return lines, list(set(sources))

    def _is_issue_label_useful(self, label: str) -> bool:
        text = label.lower()
        if text in {"none", "na", "n/a", "-"}:
            return False
        weed_tokens = [
            "amaranthus",
            "cyperus",
            "digitaria",
            "echinochloa",
            "portulaca",
            "boerhaavia",
            "chenopodium",
            "brachiaria",
            "convolvulus",
            "trianthema",
            "parthenium",
            "commelina",
            "euphorbia",
            "phyllanthus",
            "dactyloctenium",
            "setaria",
            "eleusine",
            "weed",
            "grass",
        ]
        useful_tokens = [
            "rot",
            "rust",
            "smut",
            "bunt",
            "blight",
            "mildew",
            "borer",
            "termite",
            "bug",
            "pyrilla",
            "grub",
            "spot",
            "wilt",
            "hopper",
            "folder",
            "blast",
            "disease",
        ]
        weed_hits = sum(tok in text for tok in weed_tokens)
        useful_hits = sum(tok in text for tok in useful_tokens)
        if weed_hits >= 2:
            return False
        if weed_hits >= 1 and useful_hits <= 1:
            return False
        return useful_hits > 0

    def _split_issue_label(self, label: str) -> list[str]:
        raw = str(label).replace("&", ",").replace(" and ", ",").replace("And", ",")
        raw = re.sub(r"(?i)\btermitesand\b", "Termites, ", raw)
        raw = re.sub(r"(?i)\bearlyshootborer\b", "Early shoot borer", raw)
        raw = re.sub(r"(?i)\btopborer\b", "Top borer", raw)
        raw = re.sub(r"(?i)\bwhitegrub\b", "White grub", raw)
        parts = [p.strip(" ,;/") for p in raw.split(",")]
        return [p for p in parts if p]

    def _issue_priority_score(self, label: str, quality_status: str) -> int:
        text = label.lower()
        score = 0
        if quality_status == "valid":
            score += 2
        for tok in ["red rot", "smut", "rust", "wilt", "blast", "blight", "borer", "termite", "pyrilla", "bug", "grub"]:
            if tok in text:
                score += 3
        if len(text) <= 30:
            score += 2
        if "," not in text:
            score += 1
        return score

    def _crop_name_hi(self, crop: str) -> str:
        mapping = {
            "Wheat": "गेहूं",
            "Rice": "धान",
            "Sugarcane": "गन्ना",
            "Mustard": "सरसों",
            "Pigeon pea": "अरहर",
            "Black gram": "उड़द",
            "Green gram": "मूंग",
            "Chickpea": "चना",
        }
        return mapping.get(crop, crop)

    def _load_commodity_aliases(self) -> dict[str, list[str]]:
        try:
            data = json.loads(COMMODITY_ALIAS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        out: dict[str, list[str]] = {}
        for k, v in data.items():
            if isinstance(v, list):
                out[str(k).lower()] = [str(x) for x in v if str(x).strip()]
        return out

    def _crop_display_label(self, crop: str) -> str:
        hi = self._crop_name_hi(crop)
        try:
            aliases = self._load_commodity_aliases()
        except Exception:
            aliases = {}
        crop_norm = re.sub(r"[^a-z0-9]+", "", crop.lower())
        alias_list: list[str] = []
        for key, vals in aliases.items():
            key_norm = re.sub(r"[^a-z0-9]+", "", key.lower())
            if key_norm == crop_norm or crop_norm in key_norm or key_norm in crop_norm:
                alias_list = vals
                break
        hinglish = None
        for alias in alias_list:
            a = str(alias).strip()
            if not a or re.search(r"[\u0900-\u097f]", a):
                continue
            a_norm = re.sub(r"[^a-z0-9]+", "", a.lower())
            if not a_norm or a_norm == crop_norm:
                continue
            if crop_norm in a_norm or a_norm in crop_norm:
                continue
            if re.fullmatch(r"[A-Za-z0-9 ()/\-]+", a):
                hinglish = a
                break
        if hi != crop and hinglish:
            return f"{crop} ({hi} / {hinglish})"
        if hi != crop:
            return f"{crop} ({hi})"
        if hinglish:
            return f"{crop} ({hinglish})"
        return crop

    def _extract_pesticide_names(self, text: str) -> list[str]:
        # Simple keyword-based extraction + suffix heuristic
        names = set()
        patterns = [
            r"(?i)\\b(Chlorpyrifos|Imidacloprid|Mancozeb|Carbendazim|Metalaxyl|Copper oxychloride|Azoxystrobin|Propiconazole|Thiamethoxam|Lambda-cyhalothrin)\\b",
            r"(?i)\\b(मैनकोज़ेब|कार्बेन्डाज़िम|कॉपर ऑक्सीक्लोराइड|इमिडाक्लोप्रिड|थायमेथोक्साम)\\b",
        ]
        for pat in patterns:
            for m in re.findall(pat, text):
                names.add(m)
        for m in re.findall(r"\\b([A-Za-z\\-]{5,20}(?:zeb|mide|fent|azole|thrin|phos|zole|nate|quat|sulfur|cide))\\b", text):
            names.add(m)
        return sorted(names)

    def _looks_like_pesticide_name_query(self, text: str) -> bool:
        t = str(text).lower()
        patterns = [
            r"\b[a-z0-9.+%-]{4,}(?:zeb|mide|azole|thrin|phos|quat|cide|sulfan|sulfur|mycin)\b",
            r"\bcarbendazim\b",
            r"\bmancozeb\b",
            r"\bpropiconazole\b",
            r"\bazoxystrobin\b",
            r"\bchlorpyrifos\b",
            r"\bthiamethoxam\b",
            r"\bfipronil\b",
            r"\bimidacloprid\b",
        ]
        return any(re.search(p, t) for p in patterns)

    def _extract_pesticide_name_from_query(self, text: str) -> str | None:
        q = str(text).lower()
        names = self._load_known_pesticide_names()
        if not names:
            return None
        matches = []
        for name in names:
            nl = name.lower()
            if nl and nl in q:
                matches.append(name)
        if matches:
            return max(matches, key=len)
        # fallback: match tokenized alphanumeric core
        q_compact = re.sub(r"[^a-z0-9%+.]+", "", q)
        for name in sorted(names, key=len, reverse=True):
            core = re.sub(r"[^a-z0-9%+.]+", "", name.lower())
            if core and len(core) >= 6 and core in q_compact:
                return name
        return None

    def _infer_pesticide_name_from_query_tokens(self, text: str, crop: str | None = None) -> str | None:
        if not self.cfg.db_path:
            return None
        tokens = [
            tok
            for tok in re.findall(r"[A-Za-z][A-Za-z0-9%+.\-]{4,}", str(text))
            if tok.lower() not in {
                "dawai", "dawa", "fungus", "fungal", "disease", "spray", "dose",
                "use", "kaise", "konsi", "kiski", "kis", "liye", "rog", "pest",
            }
        ]
        if not tokens:
            return None
        try:
            conn = sqlite3.connect(self.cfg.db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT DISTINCT pesticide_name
                FROM pesticide_recommendations
                WHERE pesticide_name IS NOT NULL AND trim(pesticide_name) <> ''
                """
            ).fetchall()
            conn.close()
        except Exception:
            return None
        best_name = None
        best_score = 0
        crop_l = crop.lower() if crop else ""
        for r in rows:
            name = str(r["pesticide_name"]).strip()
            nl = name.lower()
            score = 0
            for tok in tokens:
                if tok.lower() in nl:
                    score += len(tok)
            if crop_l and crop_l in nl:
                score += 1
            if score > best_score:
                best_score = score
                best_name = name
        return best_name if best_score >= 5 else None

    def _load_known_pesticide_names(self) -> list[str]:
        if not self.cfg.db_path:
            return []
        try:
            conn = sqlite3.connect(self.cfg.db_path)
            rows = conn.execute(
                """
                SELECT DISTINCT pesticide_name
                FROM pesticide_recommendations
                WHERE pesticide_name IS NOT NULL AND trim(pesticide_name) <> ''
                """
            ).fetchall()
            conn.close()
        except Exception:
            return []
        names = [str(r[0]).strip() for r in rows if r and str(r[0]).strip()]
        return sorted(set(names), key=len, reverse=True)

    def _extract_dose_lines(self, text: str) -> list[str]:
        lines = []
        for line in text.splitlines():
            if re.search(r"(ml|g|gm|gram|लीटर|ली\\.|l/ha|kg/ha|g/l)", line, flags=re.IGNORECASE):
                clean = line.strip()
                if 8 <= len(clean) <= 120:
                    lines.append(clean)
        return lines

    def _extract_waiting_period(self, text: str) -> str | None:
        m = re.search(r"(?:PHI|प्री-हार्वेस्ट|सुरक्षा अवधि)[^\\d]*(\\d+\\s*(?:दिन|days))", text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return None

    def _clean_retrieved_text(self, text: str) -> str:
        lines = []
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            letters = sum(ch.isalpha() for ch in s)
            if letters < max(6, len(s) * 0.3):
                continue
            if len(s) > 200:
                continue
            lines.append(s)
        return "\n".join(lines)

    def _extract_crop_from_query(self, text: str) -> str | None:
        t = text.lower()
        crop_aliases = {
            "Wheat": ["गेहूं", "गेहू", "gehu", "gehun", "wheat", "whaet"],
            "Rice": ["धान", "paddy", "rice", "chawal", "चावल"],
            "Sugarcane": ["गन्ना", "ganne", "ganna", "sugarcane", "गन्ने"],
            "Mustard": ["सरसों", "sarso", "sarson", "mustard"],
            "Potato": ["आलू", "aloo", "potato"],
            "Sunflower": ["सूरजमुखी", "surajmukhi", "surujmukhi", "soorajmukhi", "suryamukhi", "sunflower"],
            "Groundnut": ["मूंगफली", "mungfali", "moongfali", "groundnut", "peanut"],
            "Cotton": ["कपास", "kapas", "cotton"],
            "Maize": ["मक्का", "makka", "maize", "corn"],
            "Black gram": ["उड़द", "urad", "black gram", "blackgram"],
            "Green gram": ["मूंग", "moong", "green gram", "greengram"],
            "Pigeon pea": ["अरहर", "arhar", "tur", "tuar", "red gram", "redgram"],
            "Chickpea": ["चना", "chana", "chickpea", "bengal gram", "bengalgram"],
        }
        for crop_name, aliases in crop_aliases.items():
            if any(alias.lower() in t for alias in aliases):
                return crop_name
        return None

    def _extract_disease_terms_from_query(self, text: str) -> list[str]:
        t = text.lower()
        terms: list[str] = []
        for canonical, aliases in DISEASE_ALIASES.items():
            if any(alias.lower() in t for alias in aliases):
                terms.append(canonical)
        generic_insect_patterns = [
            r"\bkida\b",
            r"\bkide\b",
            r"\bkido\b",
            r"\bkeeda\b",
            r"\bkeede\b",
            r"\bkeet\b",
            r"कीट",
        ]
        if not terms and any(re.search(pat, t) for pat in generic_insect_patterns):
            terms.append("insect pest")
        return terms

    def _extract_pesticides_from_pdfs(self, crop: str) -> tuple[list[str], list[str]]:
        try:
            from pypdf import PdfReader
        except Exception:
            return [], []
        sources = []
        lines_out: list[str] = []
        root = Path("data/raw/all_sources")
        if not root.exists():
            return [], []
        crop_key = crop.lower()
        for pdf in root.glob("*.pdf"):
            try:
                reader = PdfReader(str(pdf))
            except Exception:
                continue
            sources.append(str(pdf))
            for i in range(min(10, len(reader.pages))):
                try:
                    page_text = reader.pages[i].extract_text() or ""
                except Exception:
                    continue
                for line in page_text.splitlines():
                    s = line.strip()
                    if not s:
                        continue
                    if crop_key in s.lower():
                        if re.search(r"(ml|g|gm|kg|l/ha|kg/ha)", s, flags=re.IGNORECASE) or self._extract_pesticide_names(s):
                            if 6 <= len(s) <= 160:
                                lines_out.append(s)
                if len(lines_out) >= 6:
                    break
            if len(lines_out) >= 6:
                break
        return lines_out[:6], sources

    def _extract_pesticides_from_db(
        self,
        crop: str,
        disease_terms: list[str] | None = None,
        limit: int = 5,
    ) -> tuple[list[str], list[str]]:
        if not self.cfg.db_path:
            return [], []
        try:
            conn = sqlite3.connect(self.cfg.db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT crop_name, disease_name_en, disease_name_hi, pesticide_name,
                       ai_g, formulation, dilution, dose_text, waiting_period_days,
                       ai_unit, formulation_unit, dilution_unit, waiting_period_unit,
                       unit_source, source_file, quality_status
                FROM pesticide_recommendations
                WHERE lower(crop_name)=lower(?)
                ORDER BY CASE quality_status WHEN 'valid' THEN 0 ELSE 1 END
                LIMIT 200
                """,
                (crop,),
            ).fetchall()
            conn.close()
        except Exception:
            return [], []
        if not rows:
            return [], []
        disease_terms = disease_terms or []
        expanded_terms = []
        for term in disease_terms:
            expanded_terms.extend(DISEASE_ALIASES.get(term, [term]))
        expanded_terms = [t.lower() for t in expanded_terms if t]

        def score_row(r: sqlite3.Row) -> int:
            text = f"{r['disease_name_en'] or ''} {r['disease_name_hi'] or ''}".lower()
            score = 0
            if expanded_terms:
                score += sum(10 for term in expanded_terms if term in text)
            if r["quality_status"] == "valid":
                score += 2
            if r["pesticide_name"]:
                score += 1
            if r["ai_g"] or r["formulation"] or r["dilution"] or r["dose_text"]:
                score += 1
            return score

        if expanded_terms:
            rows = [r for r in rows if score_row(r) >= 10]
        rows = sorted(rows, key=score_row, reverse=True)[:limit]
        if not rows:
            return [], []
        lines = []
        sources = []
        for r in rows:
            disease = r["disease_name_hi"] or self._translate_disease_name(r["disease_name_en"] or "")
            dose_parts = self._build_hindi_dose_parts(r)
            waiting = self._format_value_with_unit(r["waiting_period_days"], r["waiting_period_unit"])
            waiting_part = f" | कटाई से पहले प्रतीक्षा अवधि (PHI): {waiting}" if waiting else ""
            detail = f" | {'; '.join(dose_parts)}" if dose_parts else ""
            line = f"{disease} | दवा: {r['pesticide_name'] or 'नाम उपलब्ध नहीं'}{detail}{waiting_part}"
            lines.append(line.strip())
            if r["source_file"]:
                sources.append(r["source_file"])
        return lines, list(set(sources))

    def _extract_rows_for_pesticide_name(
        self,
        pesticide_name: str,
        crop: str | None = None,
        limit: int = 5,
    ) -> tuple[list[str], list[str]]:
        if not self.cfg.db_path or not pesticide_name:
            return [], []
        try:
            conn = sqlite3.connect(self.cfg.db_path)
            conn.row_factory = sqlite3.Row
            if crop:
                rows = conn.execute(
                    """
                    SELECT crop_name, disease_name_en, disease_name_hi, pesticide_name,
                           ai_g, formulation, dilution, dose_text, waiting_period_days,
                           ai_unit, formulation_unit, dilution_unit, waiting_period_unit,
                           source_file, quality_status
                    FROM pesticide_recommendations
                    WHERE lower(pesticide_name)=lower(?)
                      AND lower(crop_name)=lower(?)
                    ORDER BY CASE quality_status WHEN 'valid' THEN 0 ELSE 1 END
                    LIMIT ?
                    """,
                    (pesticide_name, crop, limit * 3),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT crop_name, disease_name_en, disease_name_hi, pesticide_name,
                           ai_g, formulation, dilution, dose_text, waiting_period_days,
                           ai_unit, formulation_unit, dilution_unit, waiting_period_unit,
                           source_file, quality_status
                    FROM pesticide_recommendations
                    WHERE lower(pesticide_name)=lower(?)
                    ORDER BY CASE quality_status WHEN 'valid' THEN 0 ELSE 1 END
                    LIMIT ?
                    """,
                    (pesticide_name, limit * 4),
                ).fetchall()
            conn.close()
        except Exception:
            return [], []
        lines: list[str] = []
        sources: list[str] = []
        seen: set[str] = set()
        for r in rows:
            disease = r["disease_name_hi"] or self._translate_disease_name(r["disease_name_en"] or "")
            crop_label = self._crop_name_hi(r["crop_name"] or crop or "")
            key = f"{crop_label}|{disease}"
            if key in seen:
                continue
            seen.add(key)
            dose_parts = self._build_hindi_dose_parts(r)
            waiting = self._format_value_with_unit(r["waiting_period_days"], r["waiting_period_unit"])
            waiting_part = f" | कटाई से पहले प्रतीक्षा अवधि (PHI): {waiting}" if waiting else ""
            detail = f" | {'; '.join(dose_parts)}" if dose_parts else ""
            lines.append(f"{crop_label}: {disease}{detail}{waiting_part}".strip())
            if r["source_file"]:
                sources.append(r["source_file"])
            if len(lines) >= limit:
                break
        return lines, sorted(set(sources))

    def _extract_rows_for_pesticide_text(
        self,
        text: str,
        crop: str | None = None,
        limit: int = 5,
    ) -> tuple[list[str], list[str]]:
        if not self.cfg.db_path:
            return [], []
        tokens = [
            tok.lower()
            for tok in re.findall(r"[A-Za-z][A-Za-z0-9%+.\-]{4,}", str(text))
            if tok.lower() not in {
                "dawai", "dawa", "fungus", "fungal", "disease", "spray", "dose",
                "use", "kaise", "konsi", "kiski", "kis", "liye", "rog", "pest",
                "lagti", "lagta", "me", "for",
            }
        ]
        tokens = [t for t in tokens if len(t) >= 5]
        if not tokens:
            return [], []
        try:
            conn = sqlite3.connect(self.cfg.db_path)
            conn.row_factory = sqlite3.Row
            if crop:
                rows = conn.execute(
                    """
                    SELECT crop_name, disease_name_en, disease_name_hi, pesticide_name,
                           ai_g, formulation, dilution, dose_text, waiting_period_days,
                           ai_unit, formulation_unit, dilution_unit, waiting_period_unit,
                           source_file, quality_status
                    FROM pesticide_recommendations
                    WHERE lower(crop_name)=lower(?)
                    LIMIT 800
                    """,
                    (crop,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT crop_name, disease_name_en, disease_name_hi, pesticide_name,
                           ai_g, formulation, dilution, dose_text, waiting_period_days,
                           ai_unit, formulation_unit, dilution_unit, waiting_period_unit,
                           source_file, quality_status
                    FROM pesticide_recommendations
                    LIMIT 1200
                    """
                ).fetchall()
            conn.close()
        except Exception:
            return [], []
        scored: list[tuple[int, sqlite3.Row]] = []
        for r in rows:
            pname = str(r["pesticide_name"] or "").lower()
            if not pname:
                continue
            score = 0
            for tok in tokens:
                if tok in pname:
                    score += len(tok)
            if score > 0:
                if r["quality_status"] == "valid":
                    score += 3
                scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        lines: list[str] = []
        sources: list[str] = []
        seen: set[str] = set()
        for _score, r in scored:
            disease = r["disease_name_hi"] or self._translate_disease_name(r["disease_name_en"] or "")
            crop_label = self._crop_name_hi(r["crop_name"] or crop or "")
            pname = r["pesticide_name"] or "नाम उपलब्ध नहीं"
            key = f"{crop_label}|{disease}|{pname}"
            if key in seen:
                continue
            seen.add(key)
            dose_parts = self._build_hindi_dose_parts(r)
            waiting = self._format_value_with_unit(r["waiting_period_days"], r["waiting_period_unit"])
            waiting_part = f" | कटाई से पहले प्रतीक्षा अवधि (PHI): {waiting}" if waiting else ""
            detail = f" | {'; '.join(dose_parts)}" if dose_parts else ""
            lines.append(f"{crop_label}: {disease} | दवा: {pname}{detail}{waiting_part}".strip())
            if r["source_file"]:
                sources.append(r["source_file"])
            if len(lines) >= limit:
                break
        return lines, sorted(set(sources))

    def _build_hindi_dose_parts(self, row: sqlite3.Row) -> list[str]:
        parts: list[str] = []
        ai = self._format_value_with_unit(row["ai_g"], row["ai_unit"])
        formulation = self._format_value_with_unit(row["formulation"], row["formulation_unit"])
        dilution = self._format_value_with_unit(row["dilution"], row["dilution_unit"])
        dose_text = str(row["dose_text"] or "").strip()
        if ai:
            parts.append(f"सक्रिय तत्व (a.i.): {ai}")
        if formulation:
            form_label = "उत्पाद मात्रा (formulation)"
            if self._is_ambiguous_mixed_unit(formulation):
                form_label += " - unit अस्पष्ट"
            parts.append(f"{form_label}: {formulation}")
        if dilution:
            parts.append(f"कितने पानी में घोलें: {dilution}")
        if not parts and dose_text:
            parts.append(f"खुराक: {dose_text}")
        return parts

    def _is_ambiguous_mixed_unit(self, text: str) -> bool:
        t = str(text).lower()
        return ("g/ml" in t) or ("gm/ml" in t) or ("kg/l" in t) or ("gm/l" in t)

    def _format_value_with_unit(self, value: object, unit: object) -> str:
        text = "" if value is None else str(value).replace("\xa0", " ").strip()
        if not text or text.lower() in {"nan", "none", "na", "n/a"} or text in {"-", "–", "--"}:
            return ""
        unit_text = "" if unit is None else str(unit).replace("\xa0", " ").strip()
        if not unit_text or unit_text.lower() in {"nan", "none"}:
            return text
        if re.search(r"[a-zA-Z%]", text):
            return text
        return f"{text} {unit_text}".strip()

    def _translate_disease_name(self, disease_en: str) -> str:
        text = disease_en or ""
        lower = text.lower()
        hits = []
        for key, hi in DISEASE_HINDI_TERMS.items():
            if key in lower and hi not in hits:
                hits.append(hi)
        if hits:
            return f"{' / '.join(hits)} ({self._clean_pest_label(text)})"
        return self._clean_pest_label(text)

    def _clean_pest_label(self, text: str) -> str:
        if not text:
            return ""
        cleaned = str(text).replace("\xa0", " ")
        cleaned = re.sub(r"[A-Za-z]{8,}", lambda m: self._split_agri_compound(m.group(0)), cleaned)
        cleaned = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", cleaned)
        cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def _split_agri_compound(self, token: str) -> str:
        lower = token.lower()
        words = sorted(PEST_LABEL_WORDS, key=len, reverse=True)
        out: list[str] = []
        i = 0
        while i < len(lower):
            match = next((w for w in words if lower.startswith(w, i)), None)
            if match:
                out.append(match)
                i += len(match)
            else:
                # Keep unknown stretches unchanged instead of guessing.
                j = i + 1
                while j < len(lower) and not any(lower.startswith(w, j) for w in words):
                    j += 1
                out.append(token[i:j])
                i = j
        if len(out) <= 1:
            return token
        joined = " ".join(out)
        return joined[:1].upper() + joined[1:] if token[:1].isupper() else joined

    def _fallback_answer(self, retrieved: list[dict], normalized_query: str) -> str:
        if not retrieved:
            return (
                f"समझा गया सवाल (हिंदी): {normalized_query}\n\n"
                "मॉडल से उत्तर साफ नहीं मिला। कृपया सवाल में जिला, मौसम, बजट और फसल विकल्प जोड़कर फिर से पूछें।"
            )
        snippets = [f"- {r.get('text', '')[:140]}" for r in retrieved[:3]]
        return (
            f"समझा गया सवाल (हिंदी): {normalized_query}\n\n"
            "मॉडल अस्थायी रूप से धीमा है, इसलिए संदर्भ आधारित त्वरित सलाह दी जा रही है:\n\n"
            "1) जिले और मौसम के हिसाब से मध्यम जोखिम वाली फसल चुनें।\n"
            "2) बजट को बीज, उर्वरक, सिंचाई और रोग प्रबंधन में बांटें।\n"
            "3) बाजार कीमत और भंडारण जोखिम देखकर अंतिम निर्णय लें।\n\n"
            "संदर्भ अंश:\n"
            + "\n".join(snippets)
        )

    def _is_greeting(self, text: str) -> bool:
        t = text.strip().lower()
        tokens = re.findall(r"[a-z0-9\u0900-\u097F]+", t)
        greetings = {"hello", "hi", "hey", "namaste", "नमस्ते", "राम", "ram"}
        # Handle "राम राम" / "ram ram"
        if "राम" in tokens and "राम" in tokens:
            return True
        if "ram" in tokens and "ram" in tokens:
            return True
        return any(tok in greetings for tok in tokens)

    def _has_agri_intent(self, text: str) -> bool:
        t = text.strip().lower()
        agri_words = [
            "crop",
            "fasal",
            "फसल",
            "yield",
            "उपज",
            "budget",
            "बजट",
            "wheat",
            "गेहूं",
            "sugarcane",
            "गन्ना",
            "potato",
            "आलू",
            "mustard",
            "सरसों",
        ]
        return any(w in t for w in agri_words)

    def _lookup_district_from_location(self, place: str) -> str | None:
        if not place:
            return None
        path = Path("data/processed/location_lookup.csv")
        if not path.exists():
            return None
        norm = re.sub(r"[^a-z0-9]+", "", place.lower())
        if not norm:
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except Exception:
            return None
        for col in ["place_norm", "sub_district", "district"]:
            for row in rows:
                val = row.get(col, "") or ""
                val_norm = val if col == "place_norm" else re.sub(r"[^a-z0-9]+", "", val.lower())
                if val_norm == norm:
                    district = (row.get("district") or "").strip()
                    state = (row.get("state") or "").strip()
                    if state.lower() == "uttar pradesh" or not state:
                        return district or None
        # Fallback: contains match (e.g., "Doghat Rural")
        for row in rows:
            place_val = (row.get("place") or "").strip()
            place_norm = re.sub(r"[^a-z0-9]+", "", place_val.lower())
            if norm and norm in place_norm:
                district = (row.get("district") or "").strip()
                state = (row.get("state") or "").strip()
                if state.lower() == "uttar pradesh" or not state:
                    return district or None
        # Fallback: fuzzy match for minor spelling errors
        try:
            import difflib

            up_rows = [r for r in rows if (r.get("state") or "").lower() == "uttar pradesh"]
            pool = up_rows if up_rows else rows
            norms = list({(r.get("place_norm") or "").strip() for r in pool if r.get("place_norm")})
            matches = difflib.get_close_matches(norm, norms, n=1, cutoff=0.8)
            if matches:
                for row in pool:
                    if (row.get("place_norm") or "").strip() == matches[0]:
                        district = (row.get("district") or "").strip()
                        return district or None
        except Exception:
            pass
        return None

    def _time_based_greeting(self) -> str:
        hour = datetime.now(ZoneInfo("Asia/Kolkata")).hour
        if hour < 12:
            greeting = "सुप्रभात"
        elif hour < 17:
            greeting = "नमस्कार"
        else:
            greeting = "शुभ संध्या"
        return f"{greeting}। मैं किसान एआई हूँ, मैं आपकी कैसे सहायता करूँ?"

    def _normalize_hinglish(self, text: str) -> str:
        mapping = {
            r"\bwhaet\b": "wheat",
            r"\bburnt\b": "bunt",
            r"\bkido\b": "कीट",
            r"\bkide\b": "कीट",
            r"\bkida\b": "कीट",
            r"\bkeeda\b": "कीट",
            r"\bkeede\b": "कीट",
            r"\bdawai\b": "दवा",
            r"\bdawa\b": "दवा",
            r"\bkaise\b": "कैसे",
            r"\bkese\b": "कैसे",
            r"\bkonsi\b": "कौन सी",
            r"\bkaunsi\b": "कौन सी",
            r"\bfasal\b": "फसल",
            r"\bacchi\b": "अच्छी",
            r"\bbetter\b": "बेहतर",
            r"\bhai\b": "है",
            r"\bkitna\b": "कितना",
            r"\bbudget\b": "बजट",
            r"\byield\b": "उपज",
            r"\bgehun\b": "गेहूं",
            r"\bgehu\b": "गेहूं",
            r"\bgehoo\b": "गेहूं",
            r"\bगेहु\b": "गेहूं",
            r"\bsurajmukhi\b": "सूरजमुखी",
            r"\bsurujmukhi\b": "सूरजमुखी",
            r"\bsoorajmukhi\b": "सूरजमुखी",
            r"\bsuryamukhi\b": "सूरजमुखी",
            r"\baloo\b": "आलू",
            r"\bsarso\b": "सरसों",
            r"\bwhat crop should i grow\b": "मुझे कौन सी फसल उगानी चाहिए",
            r"\bgrow\b": "उगानी",
            r"\bugaye\b": "उगाएं",
            r"\bugayen\b": "उगाएं",
            r"\bugana\b": "उगाना",
            r"\bmausam\b": "मौसम",
            r"\bbarish\b": "बारिश",
            r"\bbaarish\b": "बारिश",
            r"\baaj\b": "आज",
            r"\bjankari\b": "जानकारी",
        }
        out = text
        for pattern, replacement in mapping.items():
            out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
        return out

    def _split_context_and_question(self, text: str) -> tuple[str, str]:
        marker = "किसान का प्रश्न:"
        if marker in text:
            left, right = text.split(marker, 1)
            return left.strip(), right.strip()
        return "", text.strip()

    def _is_low_quality_response(self, text: str) -> bool:
        t = (text or "").strip()
        if len(t) < 40:
            return True
        if "�" in t:
            return True
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        if len(lines) >= 4:
            unique_ratio = len(set(lines)) / len(lines)
            if unique_ratio < 0.55:
                return True
        return False

    def _is_weather_intent(self, text: str) -> bool:
        t = text.strip().lower()
        weather_words = [
            "weather",
            "mausam",
            "maussam",
            "mausm",
            "mosam",
            "मौसम",
            "बारिश",
            "बारिस",
            "barish",
            "baarish",
            "rain",
            "rainfall",
            "temperature",
            "तापमान",
            "aaj ka mausam",
            "आज का मौसम",
            "आर्द्रता",
            "humidity",
        ]
        return any(w in t for w in weather_words)

    def _extract_district(self, context_part: str) -> str | None:
        if not context_part:
            return None
        m_hi = re.search(r"जिला:\s*([^|]+)", context_part, flags=re.IGNORECASE)
        if m_hi:
            return m_hi.group(1).strip()
        m_en = re.search(r"District:\s*([^|]+)", context_part, flags=re.IGNORECASE)
        if m_en:
            return m_en.group(1).strip()
        return None

    def _extract_location_from_question(self, question: str) -> str | None:
        if not question:
            return None
        q = question.strip()
        # Fast path: strip common weather words and stopwords, keep remaining tokens as location.
        if (
            "mausam" in q.lower()
            or "मौसम" in q
            or "weather" in q.lower()
            or "बारिश" in q
            or "बारिस" in q
            or "barish" in q.lower()
            or "baarish" in q.lower()
            or "rain" in q.lower()
        ):
            drop = {
                "aaj",
                "aj",
                "kya",
                "ka",
                "ki",
                "ke",
                "me",
                "mein",
                "में",
                "abhi",
                "aa",
                "sakti",
                "sakta",
                "sakta",
                "hai",
                "ho",
                "hoga",
                "hoon",
                "kesa",
                "kaisa",
                "h",
                "mausam",
                "maussam",
                "mosam",
                "mausm",
                "मौसम",
                "weather",
                "baarish",
                "barish",
                "rain",
                "rainfall",
                "बारिश",
                "बारिस",
                "अभी",
                "क्या",
                "सकती",
                "सकता",
                "है",
                "ka",
                "?",
                "kesa?",
                "hai?",
                "mausam?",
                "mausam.",
                "mausam,",
                "weather?",
                "weather.",
                "weather,",
            }
            tokens = [t.strip(" ?!.,") for t in re.split(r"\s+", q) if t.strip()]
            kept = [t for t in tokens if t.strip(" ?!.,").lower() not in drop]
            if kept:
                return " ".join(kept)
        q_lower = question.lower()
        if (
            "mausam" in q_lower
            or "मौसम" in question
            or "weather" in q_lower
            or "बारिश" in question
            or "बारिस" in question
            or "barish" in q_lower
            or "baarish" in q_lower
            or "rain" in q_lower
        ):
            tokens = [t.strip(" ?!.,") for t in re.split(r"\s+", question) if t.strip()]
            stop = {
                "aaj",
                "aj",
                "kya",
                "ka",
                "ki",
                "ke",
                "me",
                "mein",
                "में",
                "abhi",
                "aa",
                "sakti",
                "sakta",
                "ho",
                "hoga",
                "kesa",
                "kaisa",
                "hai",
                "h",
                "weather",
                "barish",
                "baarish",
                "rain",
            }
            for tok in tokens:
                t = tok.lower()
                if t in stop or t in {"mausam", "maussam", "mosam", "mausm", "मौसम", "weather", "बारिश", "बारिस", "क्या", "सकती", "सकता", "अभी", "है"}:
                    continue
                return tok
        # Try explicit location phrases first
        patterns = [
            r"(?:weather in|mausam in|maussam in|mosam in)\s+([a-zA-Z\\s]+)",
            r"([a-zA-Z\\s]+?)\\s+(?:me|mein|में)\\s+(?:baarish|barish|rain)",
            r"([a-zA-Z\\s]+?)\\s+(?:me|mein|में)\\s+बारिश",
            r"([\\u0900-\\u097F\\s]+?)\\s+में\\s+बारिश",
            r"([a-zA-Z\\s]+?)\\s+(?:ka|ki|ke)\\s+weather",
            r"([a-zA-Z\\s]+?)\\s+(?:ka|ki|ke)\\s+(?:mausam|maussam|mosam|mausm|मौसम)",
            r"(?:aaj|aj)?\\s*(?:ka\\s+)?weather\\s+([a-zA-Z\\s]+?)\\s+(?:me|mein|में)",
            r"(?:aaj|aj)?\\s*(?:ka\\s+)?(?:mausam|maussam|mosam|mausm|मौसम)\\s+([a-zA-Z\\s]+?)\\s+(?:me|mein|में)",
            r"([a-zA-Z\\s]+?)\\s+(?:me|mein|में)\\s+(?:ka\\s+)?weather",
            r"([a-zA-Z\\s]+?)\\s+(?:me|mein|में)\\s+(?:ka\\s+)?(?:mausam|maussam|mosam|mausm|मौसम)",
            r"([\\u0900-\\u097F\\s]+?)\\s+का\\s+मौसम",
            r"([\\u0900-\\u097F\\s]+?)\\s+की\\s+मौसम",
            r"(?:आज|अज)?\\s*(?:का\\s+)?मौसम\\s+([\\u0900-\\u097F\\s]+?)\\s+में",
            r"([\\u0900-\\u097F\\s]+?)\\s+में\\s+(?:का\\s+)?मौसम",
        ]
        for pat in patterns:
            m = re.search(pat, question, flags=re.IGNORECASE)
            if m:
                return m.group(1).strip()
        # Fallback: token after 'mausam/मौसम'
        tokens = re.split(r"\s+", question.strip())
        stop = {
            "aaj",
            "aj",
            "ka",
            "ki",
            "ke",
            "me",
            "mein",
            "में",
            "kesa",
            "kaisa",
            "hai",
            "h",
            "?",
        }
        for idx, tok in enumerate(tokens):
            t = tok.strip(" ?!.," ).lower()
            if t in {"mausam", "maussam", "mosam", "mausm", "मौसम", "weather"} and idx + 1 < len(tokens):
                cand = tokens[idx + 1].strip(" ?!.,")
                if cand and cand.lower() not in stop:
                    return cand
        # Fallback: token before 'mausam/मौसम'
        for idx, tok in enumerate(tokens):
            t = tok.strip(" ?!.," ).lower()
            if t in {"mausam", "maussam", "mosam", "mausm", "मौसम", "weather"} and idx - 1 >= 0:
                cand = tokens[idx - 1].strip(" ?!.,")
                if cand and cand.lower() not in stop:
                    return cand
            # Handle "X ka mausam" -> pick token before ka/ki/ke
            if t in {"ka", "ki", "ke"} and idx + 1 < len(tokens):
                nxt = tokens[idx + 1].strip(" ?!.,").lower()
                if nxt in {"mausam", "maussam", "mosam", "mausm", "मौसम", "weather"} and idx - 1 >= 0:
                    cand = tokens[idx - 1].strip(" ?!.,")
                    if cand and cand.lower() not in stop:
                        return cand
        # Last resort: first non-stop token
        tokens = [t.strip(" ?!.," ) for t in re.split(r"\s+", question) if t.strip()]
        stop = {
            "aaj",
            "aj",
            "ka",
            "ki",
            "ke",
            "me",
            "mein",
            "में",
            "kesa",
            "kaisa",
            "hai",
            "h",
            "mausam",
            "maussam",
            "mosam",
            "mausm",
            "मौसम",
            "weather",
        }
        for tok in tokens:
            t = tok.lower()
            if t in stop:
                continue
            return tok
        return None

    def _looks_like_location_only(self, question: str) -> bool:
        if not question:
            return False
        q = question.strip()
        if self._is_greeting(q):
            return False
        if self._is_profitability_followup_intent(q):
            return False
        if self._has_agri_intent(q) or self._is_crop_choice_intent(q):
            return False
        tokens = [t for t in re.split(r"\\s+", q) if t]
        if len(tokens) > 3:
            return False
        # If it contains any weather word, weather intent already handled.
        if self._is_weather_intent(q):
            return False
        return True

    def _extract_season(self, context_part: str) -> str | None:
        if not context_part:
            return None
        m_hi = re.search(r"मौसम:\s*([^|]+)", context_part, flags=re.IGNORECASE)
        if m_hi:
            return m_hi.group(1).strip()
        m_en = re.search(r"Season:\s*([^|]+)", context_part, flags=re.IGNORECASE)
        if m_en:
            return m_en.group(1).strip()
        return None

    def _extract_budget(self, text: str) -> float | None:
        m = re.search(r"(?:₹|rs\.?|inr)?\s*([0-9][0-9,]{3,})", text, flags=re.IGNORECASE)
        if not m:
            return None
        val = m.group(1).replace(",", "")
        try:
            return float(val)
        except ValueError:
            return None

    def _is_crop_choice_intent(self, text: str) -> bool:
        t = text.strip().lower()
        keys = [
            "what crop should i grow",
            "कौन सी फसल",
            "फसल बेहतर",
            "कौन सी crop",
            "लाभदायक फसल",
            "फसल लाभदायक",
            "लाभ वाली",
            "लाभदायक",
            "फायदे की फसल",
            "फसल फ़ायदेमंद",
            "best crop",
            "which crop",
            "crop to grow",
            "फसल उगानी",
            "profitable",
            "laabhdayak",
            "labhdayak",
            "laabh",
            "labh",
            "faayde ki",
            "fayde ki",
            "faaydemand",
            "fayemand",
            "badhiya fasal",
            "acchi fasal",
            "achhi fasal",
            "best fasal",
            "profit",
        ]
        return any(k in t for k in keys)

    def _is_crop_guide_intent(self, text: str) -> bool:
        t = text.strip().lower()
        keys = [
            "how to grow",
            "how to cultivate",
            "crop guide",
            "production guide",
            "package of practices",
            "kaise ugaye",
            "kaise ugai",
            "kaise ugayen",
            "kaise ugaaye",
            "kheti kaise kare",
            "kheti kese kare",
            "ki kheti kaise kare",
            "ki kheti kese kare",
            "ki kheti kaise karein",
            "ki kheti kese karein",
            "kheti kare",
            "kheti kese",
            "kheti kaise",
            "ki kheti",
            "खेती करें",
            "खेती कैसे करें",
            "ugane ka tarika",
            "खेती कैसे",
            "उगाने का तरीका",
            "कृषि विधि",
            "cultivation",
            "grow crop",
            "कैसे उगाएं",
            "कैसे उगाये",
        ]
        crop_markers = [
            "rice",
            "wheat",
            "sugarcane",
            "maize",
            "groundnut",
            "sesame",
            "cotton",
            "धान",
            "गेहूं",
            "gehu",
            "gehun",
            "गन्ना",
            "गन्ने",
            "ganne",
            "ganna",
            "मक्का",
            "मूंगफली",
            "तिल",
            "कपास",
            "सूरजमुखी",
            "surajmukhi",
            "surujmukhi",
            "soorajmukhi",
            "suryamukhi",
        ]
        guide_context_words = [
            "खेती",
            "kheti",
            "cultivation",
            "guide",
            "production",
            "उगाएं",
            "उगाना",
            "kare",
            "karein",
            "करें",
        ]
        return (any(k in t for k in keys) or any(w in t for w in guide_context_words)) and any(c in t for c in crop_markers)

    def _structured_crop_recommendation(
        self,
        context_part: str,
        question: str,
        district_override: str | None = None,
    ) -> tuple[str | None, list[str]]:
        if not self.cfg.db_path:
            return None, []

        district = district_override or self._extract_district(context_part) or "Meerut"
        season = self._extract_season(context_part) or "Rabi"
        budget = self._extract_budget(question)
        sources: list[str] = []
        market_prices = self._load_agmarknet_prices(district)
        if market_prices:
            sources.append("agmarknet_report.csv")

        conn = sqlite3.connect(self.cfg.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT crop_name, cost_min_inr_per_acre, cost_max_inr_per_acre,
                       market_price_inr_per_qtl, avg_yield_qtl_per_acre
                FROM crop_economics
                WHERE lower(district) = lower(?) AND lower(season) = lower(?)
                """,
                (district, season),
            ).fetchall()
            if not rows:
                rows = conn.execute(
                    """
                    SELECT crop_name, cost_min_inr_per_acre, cost_max_inr_per_acre,
                           market_price_inr_per_qtl, avg_yield_qtl_per_acre
                    FROM crop_economics
                    WHERE lower(district) = lower(?)
                    """,
                    (district,),
                ).fetchall()
        finally:
            conn.close()

        if not rows:
            baseline_answer = self._rank_from_profit_baselines(district, season, question, market_prices)
            if baseline_answer:
                sources.extend(["crop_profit_baselines", "pesticide_recommendations"])
                return baseline_answer, sources
            return self._rank_from_agmarknet_only(district, question), sources
        sources.insert(0, "crop_economics (SQLite)")

        scored: list[dict] = []
        for r in rows:
            price = float(r["market_price_inr_per_qtl"])
            price_note = ""
            crop = r["crop_name"]
            if crop in market_prices:
                price_info = market_prices[crop]
                price = float(price_info["price"])
                price_note = f" (भाव: {price:.0f} Rs./Quintal, {price_info['date']})"
            pest_label = self._estimate_pesticide_pressure(crop)
            rev = price * float(r["avg_yield_qtl_per_acre"])
            base_cost_min = float(r["cost_min_inr_per_acre"])
            base_cost_max = float(r["cost_max_inr_per_acre"])
            pmin = rev - base_cost_max
            pmax = rev - base_cost_min
            if budget is not None and float(r["cost_max_inr_per_acre"]) > budget:
                continue
            scored.append(
                {
                    "crop": crop,
                    "cost_min": base_cost_min,
                    "cost_max": base_cost_max,
                    "revenue": rev,
                    "profit_min": pmin,
                    "profit_max": pmax,
                    "price_note": price_note,
                    "pressure": pest_label,
                }
            )

        if not scored:
            return (
                f"समझा गया सवाल (हिंदी): {question}\n\n"
                f"{district} ({season}) में आपके बजट के अंदर कोई स्पष्ट फसल विकल्प नहीं मिला। "
                "कृपया बजट बढ़ाएँ या फसल विकल्प बताकर फिर पूछें।"
            ), sources

        scored = sorted(scored, key=lambda x: x["profit_min"], reverse=True)[:3]
        lines = []
        lines.append("मानदंड: मंडी MSP/भाव (Agmarknet) + उपलब्ध लागत/उपज डेटा + PPQS/MUP रोग/कीट दबाव संकेत।")
        lines.append("कीमत स्रोत: Agmarknet (district market prices)")
        for i, s in enumerate(scored, start=1):
            lines.append(
                f"{i}) {s['crop']}: लागत ₹{int(s['cost_min'])}-₹{int(s['cost_max'])}/एकड़, "
                f"अनुमानित आय ₹{int(s['revenue'])}/एकड़, "
                f"संभावित लाभ ₹{int(s['profit_min'])}-₹{int(s['profit_max'])}/एकड़"
                f"{s['price_note']} | रोग/कीट दबाव संकेत: {s['pressure']}"
            )

        budget_line = f"बजट: ₹{int(budget)} प्रति एकड़" if budget is not None else "बजट: उपलब्ध नहीं"
        return (
            f"समझा गया सवाल (हिंदी): {question}\n\n"
            f"जिला: {district} | मौसम: {season} | {budget_line}\n"
            "उपलब्ध अर्थशास्त्रीय डेटा के आधार पर सर्वोत्तम फसल विकल्प:\n"
            + "\n".join(lines)
            + "\n\nनोट: कीटनाशक की रुपये लागत उपलब्ध नहीं है, इसलिए उसे लाभ में जोड़ा/घटाया नहीं गया। अंतिम निर्णय से पहले स्थानीय मंडी भाव, पानी उपलब्धता और मिट्टी की स्थिति जरूर देखें।"
        ), sources

    def _rank_from_profit_baselines(
        self,
        district: str,
        season: str,
        question: str,
        market_prices: dict[str, dict[str, str | float]],
    ) -> str | None:
        budget = self._extract_budget(question)
        scored: list[dict] = []
        for crop, base in WESTERN_UP_CROP_BASELINES.items():
            if not self._season_matches(season, str(base["season"])):
                continue
            market = self._market_price_for_crop(crop, base, market_prices)
            price = float(market["price"])
            price_source = str(market["source"])
            yield_info = load_latest_up_yield_qtl_per_acre(crop, season=season)
            yield_qtl_per_acre = (
                float(yield_info["yield_qtl_per_acre"])
                if yield_info and yield_info.get("yield_qtl_per_acre")
                else float(base["yield_qtl_per_acre"])
            )
            revenue = price * yield_qtl_per_acre
            official_cost = self._official_cost_range_for_profitability(crop, yield_qtl_per_acre)
            if official_cost:
                cost_min = float(official_cost["cost_min"])
                cost_max = float(official_cost["cost_max"])
                cost_source = str(official_cost["source"])
                cost_basis = str(official_cost["basis"])
            else:
                cost_min = float(base["cost_min"])
                cost_max = float(base["cost_max"])
                cost_source = "baseline"
                cost_basis = "indicative range"
            if budget is not None and cost_min > budget:
                continue
            pest_pressure = self._estimate_pesticide_pressure(crop)
            water_penalty = self._water_penalty(base["water_need"])
            pest_penalty = {"कम": 0.98, "मध्यम": 1.0, "उच्च": 1.06, "अज्ञात": 1.03}.get(pest_pressure, 1.03)
            adjusted_cost_max = cost_max * water_penalty * pest_penalty
            profit_min = revenue - adjusted_cost_max
            profit_max = revenue - cost_min
            scored.append(
                {
                    "crop": crop,
                    "season": base["season"],
                    "yield": yield_qtl_per_acre,
                    "yield_source": "UPAG" if yield_info else "baseline",
                    "yield_year": (yield_info or {}).get("crop_year", ""),
                    "yield_note": (yield_info or {}).get("estimate_note", ""),
                    "price": price,
                    "price_source": price_source,
                    "price_date": market.get("date", ""),
                    "revenue": revenue,
                    "cost_min": cost_min,
                    "cost_max": adjusted_cost_max,
                    "cost_source": cost_source,
                    "cost_basis": cost_basis,
                    "profit_min": profit_min,
                    "profit_max": profit_max,
                    "pest_pressure": pest_pressure,
                    "water_need": base["water_need"],
                    "risk": base["risk"],
                }
            )
        if not scored:
            return None
        scored = sorted(scored, key=lambda x: (x["profit_min"], x["profit_max"]), reverse=True)[:5]
        lines = [
            f"जिला: {district} | मौसम: {season}",
            "लाभ रैंकिंग अब सिर्फ ₹/क्विंटल से नहीं, बल्कि उपज × भाव − लागत से निकाली गई है।",
            "लागत में जहाँ उपलब्ध हो वहाँ CACP के अनुसार paid-out cost + family labour से लेकर पूरी लागत तक का band लिया गया है; नहीं मिलने पर baseline indicative range रखा गया है।",
            "",
            "सबसे बेहतर विकल्प:",
        ]
        for i, s in enumerate(scored, start=1):
            date_part = f", {s['price_date']}" if s["price_date"] else ""
            crop_label = self._crop_display_label(str(s["crop"]))
            lines.append(
                f"{i}) {crop_label}: उपज ~{s['yield']:.1f} qtl/acre"
                f"{(' (UPAG ' + str(s['yield_year']) + (', 2nd AE' if s.get('yield_note') else '') + ')') if s['yield_source']=='UPAG' and s['yield_year'] else ''}, "
                f"भाव ₹{s['price']:.0f}/qtl ({s['price_source']}{date_part}), "
                f"आय ~₹{int(s['revenue'])}/acre, लागत ~₹{int(s['cost_min'])}-₹{int(s['cost_max'])}/acre ({s['cost_source']}, {s['cost_basis']}), "
                f"लाभ ~₹{int(s['profit_min'])}-₹{int(s['profit_max'])}/acre; पानी: {s['water_need']}, रोग/कीट दबाव: {s['pest_pressure']}"
            )
        lines.extend(
            [
                "",
                "क्यों sugarcane अलग दिखता है: इसका ₹/qtl कम होता है, लेकिन yield/acre बहुत अधिक होती है, इसलिए revenue अच्छा हो सकता है। फिर भी पानी, मजदूरी और 10-12 महीने की cash-cycle का जोखिम ज्यादा है।",
                "Exotic crops:",
                *[f"- {note}" for note in EXOTIC_CROP_NOTES],
                "",
                "नोट: यह planning estimate है। सटीक farm-profit के लिए आपकी जमीन, पानी, मजदूरी दर, बीज variety और खरीदी/मंडी linkage चाहिए।",
            ]
        )
        return f"समझा गया सवाल (हिंदी): {question}\n\n" + "\n".join(lines)

    def _season_matches(self, selected: str, crop_season: str) -> bool:
        s = (selected or "").lower()
        c = crop_season.lower()
        if not s or s in {"all", "any"}:
            return True
        if "annual" in c:
            return True
        if s in c:
            return True
        # If the app defaults to Rabi and user asked generic profitability,
        # still include annual sugarcane and common Western UP choices.
        return False

    def _market_price_for_crop(
        self,
        crop: str,
        base: dict,
        market_prices: dict[str, dict[str, str | float]],
    ) -> dict[str, str | float]:
        for alias in base.get("market_aliases", []):
            if alias in market_prices:
                item = dict(market_prices[alias])
                item["source"] = "Agmarknet"
                return item
        if (crop or "").lower() == "sugarcane":
            frp = get_latest_sugarcane_frp()
            if frp and frp.get("price_per_qtl"):
                return {
                    "price": float(frp["price_per_qtl"]),
                    "date": "",
                    "unit": "Rs./Quintal",
                    "source": f"CACP FRP {str(frp.get('season') or '').strip()}".strip(),
                }
        return {
            "price": float(base["fallback_price"]),
            "date": "",
            "unit": "Rs./Quintal",
            "source": "baseline/MSP-SAP estimate",
        }

    def _water_penalty(self, water_need: object) -> float:
        text = str(water_need)
        if "बहुत अधिक" in text:
            return 1.08
        if "मध्यम" in text:
            return 1.0
        if "कम" in text:
            return 0.96
        return 1.0

    def _rank_from_agmarknet_only(self, district: str, question: str) -> str | None:
        market_prices = self._load_agmarknet_prices(district)
        if not market_prices:
            return None
        scored = []
        for crop, info in market_prices.items():
            if not self._is_raw_crop_commodity(crop):
                continue
            pest_label = self._estimate_pesticide_pressure(crop)
            scored.append(
                {
                    "crop": crop,
                    "price": float(info["price"]),
                    "date": info.get("date", ""),
                    "pressure": pest_label,
                }
            )
        if not scored:
            return None
        pressure_penalty = {"कम": 1.0, "मध्यम": 1.15, "उच्च": 1.3, "अज्ञात": 1.4}
        for item in scored:
            item["score"] = item["price"] / pressure_penalty.get(item["pressure"], 1.4)
        scored = sorted(scored, key=lambda x: x["score"], reverse=True)[:5]
        lines = []
        lines.append("मानदंड: केवल कच्ची फसल/कमोडिटी, ताज़ा जिला मंडी भाव (Agmarknet), और PPQS/MUP रोग/कीट दबाव संकेत।")
        lines.append("कीमत स्रोत: Agmarknet (district market prices)")
        for i, s in enumerate(scored, start=1):
            lines.append(
                f"{i}) {s['crop']}: ताज़ा भाव ₹{int(s['price'])}/क्विंटल ({s['date']}), "
                f"रोग/कीट दबाव संकेत: {s['pressure']}"
            )
        return (
            f"समझा गया सवाल (हिंदी): {question}\n\n"
            f"जिला: {district}\n"
            "उपलब्ध मंडी भाव के आधार पर फसल-चयन संकेत:\n"
            + "\n".join(lines)
            + "\n\nनोट: यह सही शुद्ध लाभ नहीं है, क्योंकि बीज, मजदूरी, सिंचाई, उर्वरक, वास्तविक उपज और कीटनाशक की रुपये लागत उपलब्ध नहीं है। "
            "इसलिए मैंने कोई कीटनाशक लागत invent नहीं की है। गाँव/कस्बे को पहले जिला से map किया जाता है, और Agmarknet भाव जिला-स्तर पर उपलब्ध हैं।"
        )

    def _is_raw_crop_commodity(self, commodity: str) -> bool:
        c = commodity.lower().strip()
        if any(word in c for word in PROCESSED_COMMODITY_WORDS):
            return False
        if "(" in c:
            base = c.split("(", 1)[0].strip()
        else:
            base = c
        return any(raw == base or raw in c for raw in RAW_CROP_COMMODITIES)

    def _estimate_pesticide_pressure(self, crop: str) -> str:
        if not crop:
            return "अज्ञात"
        if not self.cfg.db_path:
            return "अज्ञात"
        crop_key = self._normalize_crop_for_pesticide(crop)
        try:
            conn = sqlite3.connect(self.cfg.db_path)
            count = conn.execute(
                "SELECT COUNT(*) FROM pesticide_recommendations WHERE lower(crop_name)=lower(?)",
                (crop_key,),
            ).fetchone()[0]
            conn.close()
        except Exception:
            return "अज्ञात"
        if count >= 25:
            return "उच्च"
        if count >= 8:
            return "मध्यम"
        if count > 0:
            return "कम"
        return "अज्ञात"

    def _normalize_crop_for_pesticide(self, crop: str) -> str:
        c = crop.lower()
        aliases = {
            "green chilli": "Chilli",
            "chilli green": "Chilli",
            "peas wet": "Pea",
            "peas": "Pea",
            "paddy": "Rice",
            "bengal gram": "Chickpea",
            "gram": "Chickpea",
            "arhar": "Pigeon pea",
            "tur": "Pigeon pea",
            "black gram": "Black gram",
            "green gram": "Green gram",
        }
        for key, val in aliases.items():
            if key in c:
                return val
        return crop

    def _load_agmarknet_prices(self, district: str) -> dict[str, dict[str, str | float]]:
        # Prefer SQLite market_prices if populated
        if self.cfg.db_path:
            try:
                conn = sqlite3.connect(self.cfg.db_path)
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT commodity, modal_price, arrival_date, price_unit
                    FROM market_prices
                    WHERE lower(district)=lower(?)
                    """,
                    (district,),
                ).fetchall()
                conn.close()
                if rows:
                    latest_by_commodity: dict[str, tuple[pd.Timestamp, dict[str, str | float]]] = {}
                    for r in rows:
                        try:
                            price = float(r["modal_price"])
                        except Exception:
                            continue
                        date_dt = pd.to_datetime(r["arrival_date"], errors="coerce", dayfirst=True)
                        if pd.isna(date_dt):
                            date_dt = pd.Timestamp.min
                        item = {
                            "price": price,
                            "date": r["arrival_date"] or "",
                            "unit": r["price_unit"] or "Rs./Quintal",
                        }
                        commodity = str(r["commodity"])
                        if commodity not in latest_by_commodity or date_dt > latest_by_commodity[commodity][0]:
                            latest_by_commodity[commodity] = (date_dt, item)
                    if not latest_by_commodity:
                        return {}
                    max_dt = max(dt for dt, _ in latest_by_commodity.values())
                    cutoff = max_dt - pd.Timedelta(days=365)
                    return {k: v for k, (dt, v) in latest_by_commodity.items() if dt >= cutoff}
            except Exception:
                pass

        path = Path("data/raw/live/agmarknet_report.csv")
        if not path.exists():
            return {}
        try:
            df = pd.read_csv(path)
        except Exception:
            return {}
        if {"District", "Commodity", "Arrival_Date", "Modal_Price"}.issubset(df.columns):
            df = df.rename(
                columns={
                    "District": "district",
                    "Commodity": "commodity",
                    "Arrival_Date": "arrival_date",
                    "Modal_Price": "modal_price",
                    "Price_Unit": "price_unit",
                }
            )
        elif {"district_name", "cmdt_name", "rep_date", "model_price_wt"}.issubset(df.columns):
            df = df.rename(
                columns={
                    "district_name": "district",
                    "cmdt_name": "commodity",
                    "rep_date": "arrival_date",
                    "model_price_wt": "modal_price",
                    "unit_name_price": "price_unit",
                }
            )
        else:
            return {}
        df = df[df["district"].astype(str).str.lower() == district.lower()].copy()
        if df.empty:
            return {}
        df["Arrival_Date_dt"] = pd.to_datetime(df["arrival_date"], errors="coerce", dayfirst=True)
        df = df.dropna(subset=["Arrival_Date_dt", "modal_price"])
        if df.empty:
            return {}
        latest = df.sort_values("Arrival_Date_dt").groupby("commodity", as_index=False).tail(1)
        max_dt = latest["Arrival_Date_dt"].max()
        latest = latest[latest["Arrival_Date_dt"] >= max_dt - pd.Timedelta(days=365)].copy()
        out: dict[str, dict[str, str | float]] = {}
        for _, row in latest.iterrows():
            try:
                price = float(row["modal_price"])
            except Exception:
                continue
            date = row["Arrival_Date_dt"].date().isoformat()
            out[str(row["commodity"])] = {
                "price": price,
                "date": date,
                "unit": row.get("price_unit", "Rs./Quintal"),
            }
        return out

    def _ensure_rag_components(self, load_generator: bool = True) -> None:
        if self.embedder is None:
            self.embedder = Embedder(self.cfg.embedding_model)
        if self.retriever is None:
            store = NumpyVectorStore(self.cfg.index_path, self.cfg.metadata_path)
            vectors, metadata = store.load()
            self.retriever = Retriever(vectors=vectors, metadata=metadata)
        if load_generator and self.generator is None:
            self.generator = LocalGenerator(self.cfg.generator_model)
