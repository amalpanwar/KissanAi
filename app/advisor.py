from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass, field
import json
import re
import sqlite3
from zoneinfo import ZoneInfo
import csv
from pathlib import Path
import numpy as np

from app.embeddings import Embedder
from app.generator import LocalGenerator
from app.prompting import SYSTEM_PROMPT, build_prompt
from app.retriever import Retriever
from app.vector_store import NumpyVectorStore
from app.weather import get_current_weather_hindi, get_daily_weather_forecast_hindi, get_rain_day_forecast_hindi, get_tomorrow_rain_forecast_hindi, get_weekly_weather_forecast_hindi
from app.upag_apy import load_latest_up_yield_qtl_per_acre
from app.crop_guide import CROP_ALIASES, build_crop_production_followup, build_crop_production_guide
from app.cacp import get_cacp_cost_for_crop, get_sugarcane_cost_snapshot, get_latest_sugarcane_frp
from app.msp import get_msp_for_crop
from app.web_search import is_web_search_configured, web_search
from app.agri_glossary import match_glossary_entry, format_glossary_answer, glossary_references
from app.pdf_extract import read_pdf_pages, read_pdf_text
from app.query_agent import QueryAgent, QueryPlan
from app.query_cache import QueryResponseCache
from app.symptom_matcher import (
    SYMPTOM_CANDIDATE_PATH,
    SYMPTOM_DICTIONARY_PATH,
    TRAINING_FEEDBACK_PATH,
    load_feedback_rows as load_symptom_feedback_rows,
    load_symptom_dictionary,
    normalize_symptom_text,
    save_symptom_alias_candidates,
)
import pandas as pd

from app.location_lookup import lookup_place, lookup_place_in_text

COMMODITY_ALIAS_PATH = Path("data/raw/commodity_aliases.json")
DISEASE_DICTIONARY_PATH = Path("data/processed/disease_dictionary.json")

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
    "white rust": ["white rust", "white-rust", "सफेद रतुआ"],
    "yellow rust": ["yellow rust", "stripe rust", "पीली रतुआ", "पीला रतुआ"],
    "brown rust": ["brown rust", "leaf rust", "भूरी रतुआ"],
    "black rust": ["black rust", "stem rust", "काली रतुआ"],
    "loose smut": ["loose smut", "smut", "ढीला कंडुआ", "कंडुआ"],
    "karnal bunt": ["karnal bunt", "bunt", "burnt", "करनाल बंट", "कर्नाल बंट", "बंट"],
    "powdery mildew": ["powdery mildew", "चूर्णी फफूंदी"],
    "downy mildew": ["downy mildew", "downey mildew", "डाउनी मिल्ड्यू"],
    "leaf blight": ["leaf blight", "blight", "झुलसा"],
    "alternaria blight": ["alternaria blight", "alternaria", "अल्टरनेरिया झुलसा"],
    "stem borer": ["stem borer", "तना छेदक"],
    "shoot borer": ["shoot borer", "shootborer", "early shoot borer", "earlyshootborer", "शूट बोरर"],
    "top borer": ["top borer", "topborer", "टॉप बोरर"],
    "root borer": ["root borer", "rootborer", "जड़ छेदक"],
    "white grub": ["white grub", "whitegrub", "white grubs", "whitegrubs", "white crub", "सफेद सूंडी"],
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
    "white rust": "सफेद रतुआ",
    "yellow rust": "पीली रतुआ",
    "stripe rust": "पीली रतुआ",
    "brown rust": "भूरी रतुआ",
    "leaf rust": "पत्ती/भूरी रतुआ",
    "black rust": "काली रतुआ",
    "stem rust": "तना/काली रतुआ",
    "rust": "रतुआ",
    "leaf blight": "पत्ती झुलसा",
    "blight": "झुलसा",
    "alternaria blight": "अल्टरनेरिया झुलसा",
    "loose smut": "ढीला कंडुआ",
    "karnal bunt": "कर्नाल बंट",
    "powdery mildew": "चूर्णी फफूंदी",
    "downy mildew": "डाउनी मिल्ड्यू",
    "stem borer": "तना छेदक",
    "shoot borer": "शूट बोरर",
    "top borer": "टॉप बोरर",
    "root borer": "जड़ छेदक",
    "white grub": "सफेद सूंडी",
    "insect pest": "कीट",
    "aphid": "माहू",
    "termite": "दीमक",
}

STRICT_DISEASE_QUERY_ALIASES = {
    "leaf blight": ["leaf blight", "पत्ती झुलसा"],
    "alternaria blight": ["alternaria blight", "अल्टरनेरिया झुलसा"],
    "powdery mildew": ["powdery mildew", "चूर्णी फफूंदी"],
    "downy mildew": ["downy mildew", "डाउनी मिल्ड्यू"],
}

AGRI_TERM_EXPLANATIONS = {
    "das": {
        "label": "DAS",
        "meaning": "Days After Sowing",
        "explanation": "इसका मतलब बुवाई के कितने दिन बाद है। उदाहरण: 15-20 DAS का मतलब बुवाई के 15 से 20 दिन बाद।",
    },
    "fym": {
        "label": "FYM",
        "meaning": "Farm Yard Manure",
        "explanation": "इसका मतलब गोबर की सड़ी हुई खाद है, जिसे खेत की उर्वरता बढ़ाने के लिए डाला जाता है।",
    },
    "npk": {
        "label": "NPK",
        "meaning": "Nitrogen, Phosphorus and Potash",
        "explanation": "यह मुख्य पोषक तत्वों का अनुपात बताता है। जैसे 80:40:40 NPK का मतलब नाइट्रोजन, फॉस्फोरस और पोटाश की सिफारिशी मात्रा है।",
    },
    "phi": {
        "label": "PHI",
        "meaning": "Pre-Harvest Interval",
        "explanation": "इसका मतलब दवा के आखिरी छिड़काव और फसल की कटाई के बीच जरूरी इंतजार अवधि है।",
    },
    "msp": {
        "label": "MSP",
        "meaning": "Minimum Support Price",
        "explanation": "यह सरकार द्वारा तय न्यूनतम खरीद मूल्य होता है, ताकि किसान को एक आधार भाव मिल सके।",
    },
    "frp": {
        "label": "FRP",
        "meaning": "Fair and Remunerative Price",
        "explanation": "यह गन्ने के लिए केंद्र सरकार द्वारा तय न्यूनतम भुगतान मूल्य होता है।",
    },
}


def _load_generated_disease_dictionary() -> tuple[dict[str, list[str]], dict[str, str], dict[str, dict[str, object]]]:
    if not DISEASE_DICTIONARY_PATH.exists():
        return {}, {}, {}
    try:
        data = json.loads(DISEASE_DICTIONARY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}, {}, {}

    aliases = {}
    for canonical, values in (data.get("aliases") or {}).items():
        if not canonical or not isinstance(values, list):
            continue
        cleaned = [str(value).strip() for value in values if str(value).strip()]
        if cleaned:
            aliases[str(canonical).strip().lower()] = cleaned

    hindi_terms = {}
    for canonical, value in (data.get("hindi_terms") or {}).items():
        if canonical and value:
            hindi_terms[str(canonical).strip().lower()] = str(value).strip()

    raw_display = {}
    for raw_key, payload in (data.get("raw_label_display") or {}).items():
        if not raw_key or not isinstance(payload, dict):
            continue
        raw_display[str(raw_key).strip().lower()] = payload
    return aliases, hindi_terms, raw_display


GENERATED_DISEASE_ALIASES, GENERATED_DISEASE_HINDI_TERMS, GENERATED_RAW_DISEASE_DISPLAY = _load_generated_disease_dictionary()
for canonical, aliases in GENERATED_DISEASE_ALIASES.items():
    existing = DISEASE_ALIASES.setdefault(canonical, [])
    for alias in aliases:
        if alias not in existing:
            existing.append(alias)
for canonical, hindi in GENERATED_DISEASE_HINDI_TERMS.items():
    DISEASE_HINDI_TERMS.setdefault(canonical, hindi)
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
    complex_generator_model: str | None = None
    response_cache_path: str = "data/processed/query_response_cache.json"
    response_cache_version: str = "v1"
    query_cache_ttl_sec: int = 6 * 60 * 60
    response_cache_max_entries: int = 256


def build_advisor_config(**kwargs: object) -> AdvisorConfig:
    supported = set(getattr(AdvisorConfig, "__dataclass_fields__", {}).keys())
    filtered = {key: value for key, value in kwargs.items() if key in supported}
    return AdvisorConfig(**filtered)


@dataclass
class WeatherRequest:
    intent: str = "weather"
    place: str | None = None
    action: str = "current"
    day_offset: int | None = None
    label: str | None = None
    focus: str = "weather"


@dataclass
class PesticideRequest:
    intent: str = "pesticide"
    crop: str | None = None
    crop_from_context: bool = False
    pesticide_name: str | None = None
    disease_terms: list[str] = field(default_factory=list)
    issue_mode: str = "general"
    generic_issue: bool = False
    symptom_label: str | None = None


@dataclass
class ParsedAgriIntent:
    subject: str = "unknown"
    scope: str = "unknown"
    objective: str = "unknown"
    crop: str | None = None
    season: str | None = None
    confidence: float = 0.0
    matched_label: str | None = None
    matched_phrase: str | None = None


class RAGAdvisor:
    def __init__(self, cfg: AdvisorConfig) -> None:
        self.cfg = cfg
        self.embedder: Embedder | None = None
        self.retriever: Retriever | None = None
        self.generator: LocalGenerator | None = None
        self.complex_generator: LocalGenerator | None = None
        self.top_k = cfg.top_k
        self._pdf_text_cache: dict[str, str] = {}
        self._pdf_verification_cache: dict[str, bool] = {}
        self._source_table_text_cache: dict[str, str] = {}
        self._source_table_rows_cache: dict[str, list[list[str]]] = {}
        self._source_table_record_cache: dict[str, list[dict[str, str]]] = {}
        self._processed_source_file_index: dict[str, Path] | None = None
        self._symptom_index_signature: tuple[int, int] | None = None
        self._symptom_phrase_rows: list[dict[str, str]] = []
        self._symptom_phrase_embeddings: np.ndarray | None = None
        self._intent_phrase_rows: list[dict[str, str]] = []
        self._intent_phrase_embeddings: np.ndarray | None = None
        self._intent_parse_cache: dict[str, ParsedAgriIntent] = {}
        self._feedback_symptom_candidate_signature: tuple[int, int] | None = None
        self.query_agent = QueryAgent(
            cfg.generator_model,
            complex_generator_model=cfg.complex_generator_model,
            default_top_k=cfg.top_k,
        )
        self.response_cache = QueryResponseCache(
            cfg.response_cache_path,
            version=(
                f"advisor-rag-v3::{cfg.response_cache_version}::"
                f"{cfg.embedding_model}::{cfg.generator_model}::{cfg.complex_generator_model or ''}"
            ),
            max_entries=cfg.response_cache_max_entries,
        )

    def answer(self, user_query: str) -> dict:
        import os
        from app.hindi_translation import translate_answer
        if os.getenv("KISAANAI_AGENTIC", "1").lower() in {"0", "false", "no"}:
            result = self._answer_legacy(user_query)
        else:
            from app.agent_system import build_coordinator
            context, question = self._split_context_and_question(user_query)
            result = build_coordinator(self).answer(question, context)
        return translate_answer(result)

    def _answer_legacy(self, user_query: str) -> dict:
        context_part, farmer_question = self._split_context_and_question(user_query)
        if self._is_greeting(farmer_question) and not self._has_agri_intent(farmer_question):
            return {"answer": self._time_based_greeting(), "references": [], "retrieved": [], "topic": "greeting"}

        normalized_question = self._normalize_hinglish(farmer_question)
        emergency_cost_answer = (
            self._answer_crop_cost_method_query(farmer_question, context_part)
            or self._answer_crop_cost_method_query(normalized_question, context_part)
            or self._answer_sugarcane_cost_query(farmer_question, context_part)
            or self._answer_sugarcane_cost_query(normalized_question, context_part)
            or self._answer_generic_cacp_cost_query(farmer_question, context_part)
            or self._answer_generic_cacp_cost_query(normalized_question, context_part)
        )
        if emergency_cost_answer:
            return {
                "answer": emergency_cost_answer,
                "references": ["CACP official report", "UPAG yield data"],
                "retrieved": [],
                "topic": "crop_profitability_followup",
            }
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
        agri_term_answer, agri_term_refs = self._answer_agri_term_query(farmer_question, normalized_question)
        if agri_term_answer:
            return {
                "answer": agri_term_answer,
                "references": agri_term_refs,
                "retrieved": [],
                "topic": "crop_guide_followup",
            }
        msp_answer = self._answer_msp_query(farmer_question) or self._answer_msp_query(normalized_question)
        if msp_answer:
            return {
                "answer": msp_answer,
                "references": ["PIB MSP notification"],
                "retrieved": [],
                "topic": "price",
            }
        weather_request = self._parse_weather_request(farmer_question, normalized_question)
        if weather_request:
            weather_result = self._answer_weather_request(weather_request, farmer_question, normalized_question)
            weather_result["topic"] = "weather"
            return weather_result
        parsed_intent = ParsedAgriIntent()
        if any(
            (
                self._extract_crop_from_query(normalized_question),
                self._extract_query_season(normalized_question),
                self._has_crop_method_terms(normalized_question),
                self._has_profitability_terms(normalized_question),
                self._has_crop_guide_terms(normalized_question),
            )
        ):
            parsed_intent = self._parse_agri_intent(normalized_question, context_part)
        if parsed_intent.subject == "crop_method":
            self._ensure_rag_components(load_generator=False)
            guide_followup_answer, guide_followup_sources = build_crop_production_followup(
                normalized_question,
                crop_hint=parsed_intent.crop or self._extract_preferred_crop_from_context(context_part),
                reasoning_generator=self.generator,
            )
            if guide_followup_answer:
                return {
                    "answer": guide_followup_answer,
                    "references": guide_followup_sources,
                    "retrieved": [],
                    "topic": "crop_guide_followup",
                }
        if parsed_intent.subject == "crop_guide":
            try:
                guide_answer, guide_sources = build_crop_production_guide(normalized_question)
            except Exception:
                import logging
                logging.getLogger(__name__).exception("Failed to read crop production guide")
                guide_answer, guide_sources = None, []
            if not guide_answer:
                return {
                    "answer": "इस फसल की खेती की स्रोत-आधारित गाइड अभी उपलब्ध नहीं है। "
                              "कृपया बाद में फिर प्रयास करें।",
                    "references": [], "retrieved": [], "topic": "crop_guide",
                    "status": "unavailable",
                }
            if guide_answer:
                return {
                    "answer": guide_answer,
                    "references": guide_sources,
                    "retrieved": [],
                    "topic": "crop_guide",
                }
        if parsed_intent.subject == "season_crop_list":
            season_crop_answer, season_crop_sources = self._answer_season_crop_list_query(
                context_part,
                normalized_question,
            )
            if season_crop_answer:
                return {
                    "answer": season_crop_answer,
                    "references": season_crop_sources,
                    "retrieved": [],
                    "topic": "crop_season_list",
                }
        if parsed_intent.subject == "crop_choice":
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
        if self._is_season_crop_list_intent(normalized_question):
            season_crop_answer, season_crop_sources = self._answer_season_crop_list_query(
                context_part,
                normalized_question,
            )
            if season_crop_answer:
                return {
                    "answer": season_crop_answer,
                    "references": season_crop_sources,
                    "retrieved": [],
                    "topic": "crop_season_list",
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
        pesticide_request = self._parse_pesticide_request(farmer_question, normalized_question, context_part)
        if pesticide_request:
            crop_hint = self._extract_preferred_crop_from_context(context_part)
            result = self._structured_pesticide_advice(
                normalized_question,
                crop_hint=crop_hint,
                request=pesticide_request,
            )
            result["topic"] = "pesticide"
            return result
        self._ensure_rag_components(load_generator=False)
        guide_followup_answer, guide_followup_sources = build_crop_production_followup(
            normalized_question,
            crop_hint=self._extract_preferred_crop_from_context(context_part),
            reasoning_generator=self.generator,
        )
        if guide_followup_answer:
            return {
                "answer": guide_followup_answer,
                "references": guide_followup_sources,
                "retrieved": [],
                "topic": "crop_guide_followup",
            }
        if self._looks_like_location_only(farmer_question):
            loc = (
                lookup_place_in_text(farmer_question)
                or lookup_place(farmer_question.strip())
                or lookup_place_in_text(normalized_question)
                or lookup_place(normalized_question.strip())
            )
            if loc:
                request = WeatherRequest(
                    place=str(loc.get("place") or farmer_question.strip()),
                    action="current",
                    day_offset=None,
                    label=None,
                    focus="weather",
                )
                result = self._answer_weather_request(
                    request,
                    farmer_question,
                    normalized_question,
                    place_override=str(loc.get("place") or farmer_question.strip()),
                )
                result["topic"] = "weather"
                result["weather_action"] = "current"
                return result
            web_result = self._answer_with_web_search(normalized_question, context_part)
            if web_result:
                return web_result

        normalized_query = (
            f"{context_part} किसान का प्रश्न: {normalized_question}".strip()
            if context_part
            else normalized_question
        )
        return self._answer_rag_with_agent(
            normalized_question=normalized_question,
            normalized_query=normalized_query,
            context_part=context_part,
        )

    def _get_generator_for_model(self, model_name: str | None) -> LocalGenerator | None:
        target_model = (model_name or self.cfg.generator_model).strip()
        if not target_model:
            return None
        if target_model == self.cfg.generator_model:
            self._ensure_rag_components(load_generator=True)
            return self.generator
        if self.complex_generator is None:
            try:
                self.complex_generator = LocalGenerator(target_model)
            except Exception:
                return None
        return self.complex_generator

    def _answer_rag_with_agent(
        self,
        *,
        normalized_question: str,
        normalized_query: str,
        context_part: str,
    ) -> dict:
        plan = self.query_agent.decide(normalized_question, context_part)
        cached = self.response_cache.get(
            question=normalized_question,
            context_part=context_part,
            route=plan.route,
            model_name=plan.model_name or "",
        )
        if cached:
            cached.setdefault("topic", "rag")
            cached["query_plan"] = plan.to_dict()
            return cached

        try:
            self._ensure_rag_components(load_generator=False)
        except Exception:
            web_result = self._answer_with_web_search(normalized_question, context_part) if plan.allow_web_fallback else None
            if web_result:
                web_result["query_plan"] = plan.to_dict()
                return web_result
            return {
                "answer": "अभी यह सवाल local source से नहीं निकल पाया और RAG model उपलब्ध नहीं है। कृपया सवाल में फसल/जिला साफ लिखें या थोड़ी देर बाद फिर प्रयास करें।",
                "references": [],
                "retrieved": [],
                "topic": "rag",
                "query_plan": plan.to_dict(),
                "cache_hit": False,
            }

        if self.embedder is None or self.retriever is None:
            web_result = self._answer_with_web_search(normalized_question, context_part) if plan.allow_web_fallback else None
            if web_result:
                web_result["query_plan"] = plan.to_dict()
                return web_result
            return {
                "answer": "मॉडल अभी उपलब्ध नहीं है। कृपया थोड़ी देर बाद फिर प्रयास करें।",
                "status": "unavailable",
                "references": [],
                "retrieved": [],
                "topic": "rag",
                "query_plan": plan.to_dict(),
                "cache_hit": False,
            }

        retrieved = self._retrieve_with_hyde_and_rerank(normalized_question, context_part, top_k=plan.top_k)
        response = ""
        generator = None
        if plan.route != "retrieval_only":
            generator = self._get_generator_for_model(plan.model_name)
        if generator is not None:
            prompt = build_prompt(
                normalized_query,
                retrieved,
                query_family=plan.query_family,
                max_chunks=plan.prompt_context_k,
            )
            try:
                response = generator.generate(prompt)
            except Exception:
                response = ""
        if not response or self._is_low_quality_response(response):
            if plan.allow_web_fallback:
                web_result = self._answer_with_web_search(normalized_question, context_part)
                if web_result:
                    web_result["query_plan"] = plan.to_dict()
                    return web_result
            response = self._fallback_answer(retrieved, normalized_question)

        result = {
            "answer": response,
            "references": [r.get("source_file") for r in retrieved],
            "retrieved": retrieved,
            "topic": "rag",
            "query_plan": plan.to_dict(),
            "cache_hit": False,
        }
        self.response_cache.put(
            question=normalized_question,
            context_part=context_part,
            route=plan.route,
            model_name=plan.model_name or "",
            result=result,
            ttl_sec=min(plan.cache_ttl_sec, self.cfg.query_cache_ttl_sec),
        )
        return result

    def _query_tokens(self, text: str) -> set[str]:
        tokens = re.findall(r"[a-z0-9\u0900-\u097F]+", (text or "").lower())
        return {t for t in tokens if len(t) > 1}

    def _is_price_query(self, text: str) -> bool:
        t = (text or "").lower()
        keys = [
            "price", "rate", "mandi", "bhav", "bhaav", "bhao", "daam", "dam", "keemat", "kimat", "qeemat",
            "भाव", "कीमत", "दाम", "मंडी",
            "msp", "minimum support price", "support price",
            "न्यूनतम समर्थन मूल्य", "समर्थन मूल्य", "सरकारी भाव", "सरकारी रेट",
        ]
        return any(k in t for k in keys)

    def _is_msp_query(self, text: str) -> bool:
        t = (text or "").lower()
        keys = [
            "msp",
            "minimum support price",
            "support price",
            "support rate",
            "न्यूनतम समर्थन मूल्य",
            "समर्थन मूल्य",
            "सरकारी भाव",
            "सरकारी रेट",
        ]
        return any(k in t for k in keys)

    def _answer_msp_query(self, text: str) -> str | None:
        if not self._is_msp_query(text):
            return None
        t = (text or "").lower()
        if any(cue in t for cue in ["kya hota hai", "क्या होता है", "kya hai", "क्या है", "matlab", "मतलब", "full form", "फुल फॉर्म"]):
            return None
        crop = self._extract_crop_from_query(text or "")
        if not crop:
            return "कृपया जिस फसल का MSP चाहिए उसका नाम लिखें, जैसे: गेहूं, धान, चना।"
        msp = get_msp_for_crop(crop)
        crop_label = self._crop_display_label(crop)
        if not msp:
            return f"{crop_label} के लिए अभी MSP रिकॉर्ड उपलब्ध नहीं मिला।"
        return (
            f"{crop_label} के लिए MSP (राष्ट्रीय): ₹{int(msp['msp'])}/क्विंटल.\n"
            f"स्रोत: {msp['source_url']}"
        )

    def _is_hyde_candidate(self, question: str) -> bool:
        q = (question or "").strip().lower()
        if not q or len(self._query_tokens(q)) > 10:
            return False
        if self._is_weather_intent(q) or self._is_weather_impact_intent(q):
            return False
        if self._is_price_query(q) or self._is_cost_of_production_query(q):
            return False
        vague_markers = {
            "rog", "रोग", "dawai", "दवाई", "problem", "dikat", "दिक्कत",
            "guide", "kaise", "कैसे", "konsi", "कौनसी", "kya", "क्या",
            "pest", "fungus", "symptom", "lakshan", "लक्षण",
        }
        return any(m in q for m in vague_markers)

    def _build_hyde_query(self, question: str, context_part: str) -> str:
        crop = self._extract_crop_from_query(question) or self._extract_preferred_crop_from_context(context_part) or "crop"
        disease_terms = self._extract_disease_terms_from_query(question)
        issue_mode = self._generic_issue_mode(question)
        if disease_terms:
            issue_text = ", ".join(disease_terms[:2])
            return (
                f"Farmer needs source-grounded advisory for {crop} focusing on {issue_text}, "
                "including symptoms, likely causes, recommended treatment, dose, and safety guidance."
            )
        if issue_mode == "fungal":
            return (
                f"Farmer needs source-grounded fungal disease advisory for {crop}, including likely diseases, "
                "symptoms, pesticide or seed-treatment options, dose, and precautions."
            )
        if issue_mode == "pest":
            return (
                f"Farmer needs source-grounded pest advisory for {crop}, including likely insects, symptoms, "
                "recommended control measures, pesticide options, dose, and precautions."
            )
        if self._is_crop_guide_intent(question):
            return (
                f"Farmer needs a crop cultivation guide for {crop}, including season, field preparation, seed rate, "
                "seed treatment, fertilizer, irrigation, weed management, plant protection, and harvesting."
            )
        return (
            f"Farmer needs source-grounded advisory for {crop}, with relevant agronomy, disease or input guidance, "
            "written in simple Hindi for field use."
        )

    def _rerank_retrieved(
        self,
        question: str,
        candidates: list[dict],
        top_k: int,
        *,
        keep_internal: bool = False,
    ) -> list[dict]:
        q_tokens = self._query_tokens(question)
        crop = self._extract_crop_from_query(question)
        disease_terms = [d.lower() for d in self._extract_disease_terms_from_query(question)]
        ranked: list[tuple[float, dict]] = []
        for cand in candidates:
            text = str(cand.get("text") or "")
            source = str(cand.get("source_file") or "")
            hay = f"{text}\n{source}".lower()
            c_tokens = self._query_tokens(hay)
            overlap = 0.0
            if q_tokens:
                overlap = len(q_tokens & c_tokens) / max(len(q_tokens), 1)
            crop_bonus = 0.0
            if crop and crop.lower() in hay:
                crop_bonus = 0.2
            disease_bonus = 0.0
            if disease_terms and any(term in hay for term in disease_terms):
                disease_bonus = 0.2
            hyde_bonus = 0.05 if cand.get("_from_hyde") else 0.0
            vector_score = float(cand.get("_vector_score", 0.0))
            final_score = (0.65 * vector_score) + (0.25 * overlap) + crop_bonus + disease_bonus + hyde_bonus
            item = dict(cand)
            item["_score"] = final_score
            ranked.append((final_score, item))
        ranked.sort(key=lambda x: x[0], reverse=True)
        trimmed = []
        for _score, item in ranked[:top_k]:
            if keep_internal:
                trimmed.append(item)
            else:
                trimmed.append({k: v for k, v in item.items() if not str(k).startswith("_") or k == "_score"})
        return trimmed

    def _retrieve_with_hyde_and_rerank(
        self,
        question: str,
        context_part: str,
        top_k: int | None = None,
        *,
        keep_internal: bool = False,
    ) -> list[dict]:
        if self.embedder is None or self.retriever is None:
            return []
        target_top_k = max(1, int(top_k or self.top_k))
        fetch_k = max(target_top_k * 3, 8)
        qvec = self.embedder.encode([question])[0]
        base = self.retriever.retrieve_with_scores(qvec, k=fetch_k)
        merged: dict[int, dict] = {}
        for row in base:
            merged[int(row.get("_doc_id", -1))] = dict(row)
        if self._is_hyde_candidate(question):
            hyde_query = self._build_hyde_query(question, context_part)
            hvec = self.embedder.encode([hyde_query])[0]
            hyde_rows = self.retriever.retrieve_with_scores(hvec, k=fetch_k)
            for row in hyde_rows:
                doc_id = int(row.get("_doc_id", -1))
                item = dict(row)
                item["_from_hyde"] = True
                prev = merged.get(doc_id)
                if prev is None or float(item.get("_vector_score", 0.0)) > float(prev.get("_vector_score", 0.0)):
                    if prev and prev.get("_from_hyde") is None:
                        item["_from_hyde"] = True
                    merged[doc_id] = item
                elif prev is not None:
                    prev["_from_hyde"] = prev.get("_from_hyde") or True
        return self._rerank_retrieved(question, list(merged.values()), target_top_k, keep_internal=keep_internal)

    def retrieve_debug(self, question: str, context_part: str = "", top_k: int = 5) -> list[dict]:
        self._ensure_rag_components(load_generator=False)
        if self.embedder is None or self.retriever is None:
            return []
        return self._retrieve_with_hyde_and_rerank(
            question,
            context_part,
            top_k=top_k,
            keep_internal=True,
        )

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
            "disease",
            "rog",
            "bimari",
            "lakshan",
            "symptom",
            "रोग",
            "बीमारी",
            "लक्षण",
            "dawai",
            "dawa",
            "dose",
            "dosage",
            "borer",
            "stem borer",
            "shoot borer",
            "top borer",
            "root borer",
            "white grub",
            "grub",
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
        if any(k in t for k in keys) or self._looks_like_pesticide_name_query(t):
            return True
        return bool(self._extract_disease_terms_from_query(t, allow_symptom_fallback=False))

    def _is_crop_protection_followup_intent(self, text: str) -> bool:
        t = text or ""
        return bool(
            self._is_pesticide_intent(t)
            or self._is_generic_issue_query(t)
            or self._is_disease_only_query(t)
            or self._is_fungal_query(t)
            or self._is_pest_only_query(t)
            or self._is_symptom_followup_query(t)
        )

    def _is_crop_guide_followup_intent(self, text: str) -> bool:
        t = (text or "").strip().lower()
        if not t:
            return False
        section_terms = [
            "किस्म", "kism", "kisam", "variety", "varieties", "बीज दर", "seed rate",
            "खाद", "khad", "khaad", "उर्वरक", "urvarak", "fertilizer", "fym", "compost", "जैव उर्वरक",
            "सिंचाई", "sinchai", "sichai", "sinchaai", "irrigation", "water management", "पानी", "pani", "paani", "water need", "water requirement",
            "कटाई", "katai", "katayi", "katayee", "harvest", "harvesting", "maturity",
            "खेत की तैयारी", "जुताई", "field preparation", "land preparation",
            "बुवाई", "रोपाई", "sowing", "planting", "spacing", "seed treatment",
        ]
        if any(term.lower() in t for term in section_terms):
            return True
        try:
            import difflib

            tokens = re.findall(r"[a-z\u0900-\u097f]+", t)
            for term in section_terms:
                for token in re.findall(r"[a-z\u0900-\u097f]+", term.lower()):
                    if len(token) < 4:
                        continue
                    if difflib.get_close_matches(token, tokens, n=1, cutoff=0.82):
                        return True
        except Exception:
            pass
        return False

    def _has_specific_issue_term(self, text: str) -> bool:
        t = (text or "").lower()
        specific_terms = [
            "red rot", "rust", "yellow rust", "brown rust", "black rust",
            "smut", "bunt", "blight", "mildew", "wilt", "spot", "blast",
            "stem borer", "borer", "leaf folder", "leaffolder", "hopper",
            "planthopper", "aphid", "termite", "mite", "caterpillar",
            "white grub", "grub", "shoot borer", "top borer", "root borer",
            "लाल सड़न", "रतुआ", "कंडुआ", "बंट", "झुलसा", "चूर्णी फफूंदी",
            "तना छेदक", "दीमक", "माहू", "सफेद सूंडी", "शूट बोरर", "टॉप बोरर", "जड़ छेदक",
        ]
        return any(term in t for term in specific_terms)

    def _is_generic_issue_query(self, text: str, issue_mode: str | None = None) -> bool:
        t = text or ""
        mode = issue_mode or self._generic_issue_mode(t)
        if self._has_specific_issue_term(t):
            return False
        if self._is_symptom_followup_query(t):
            return True
        if mode == "fungal":
            return self._is_fungal_query(t)
        if mode == "pest":
            return self._is_pest_only_query(t)
        if mode == "disease":
            return self._is_disease_only_query(t)
        return self._is_generic_disease_query(t) or self._is_pest_only_query(t)

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

    def _structured_pesticide_advice(
        self,
        question: str,
        crop_hint: str | None = None,
        request: PesticideRequest | None = None,
    ) -> dict:
        crop = request.crop if request and request.crop else self._extract_crop_from_query(question) or crop_hint
        pesticide_name = request.pesticide_name if request else self._extract_pesticide_name_from_query(question)
        if not pesticide_name:
            pesticide_name = self._infer_pesticide_name_from_query_tokens(question, crop=crop)
        explicit_terms = self._extract_disease_terms_from_query(question, allow_symptom_fallback=False)
        symptom_entry = None if explicit_terms else self._match_symptom_entry(question, use_semantic=True)
        disease_terms = list(request.disease_terms) if request and request.disease_terms else explicit_terms
        if not disease_terms and symptom_entry is not None:
            disease_terms = [str(t).strip().lower() for t in (symptom_entry.get("disease_candidates") or []) if str(t).strip()]
        issue_mode = request.issue_mode if request else (
            self._issue_mode_from_disease_terms(disease_terms) if disease_terms else self._generic_issue_mode(question)
        )
        symptom_label = request.symptom_label if request else None
        if not symptom_label and symptom_entry is not None:
            symptom_label = str(symptom_entry.get("label_hi") or symptom_entry.get("label_en") or "").strip() or None
        generic_issue = request.generic_issue if request else bool(crop and not pesticide_name and self._is_generic_issue_query(question, issue_mode=issue_mode))
        if crop and disease_terms:
            source_lines, source_refs = self._extract_pesticides_from_source_tables(
                crop,
                disease_terms=disease_terms,
                issue_mode=issue_mode,
                limit=3,
            )
            if source_lines:
                disease_label = self._render_symptom_or_issue_label(symptom_label, disease_terms, fallback="दिए गए रोग/कीट")
                lines = [
                    "संरचित कीटनाशक सलाह:",
                    f"- फसल: {self._crop_name_hi(crop)}",
                    f"- रोग/कीट मिलान: {disease_label}",
                    "- दवा विकल्प:",
                ]
                lines.extend(self._format_numbered_blocks(source_lines[:3]))
                lines.append("- छिड़काव/बीज उपचार से पहले उत्पाद लेबल, PHI और स्थानीय कृषि अधिकारी की सलाह जरूर मिलाएँ।")
                return {"answer": "\n".join(lines), "references": source_refs, "retrieved": []}
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
                intro.append("- यह दवा इन फसल/रोग-कीट स्थितियों में मिलती है:")
                intro.extend(self._format_numbered_blocks(chem_lines[:4]))
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
        if crop and symptom_entry is not None and disease_terms:
            db_lines, db_sources = self._extract_pesticides_from_db(
                crop,
                disease_terms=disease_terms,
                limit=8,
                symptom_text=question,
                symptom_entry=symptom_entry,
                strict_match=True,
            )
            if db_lines:
                disease_label = self._render_symptom_or_issue_label(symptom_label, disease_terms)
                lines = [
                    "संरचित कीटनाशक सलाह:",
                    f"- फसल: {self._crop_name_hi(crop)}",
                    f"- लक्षण मिलान: {disease_label}",
                    "- दवा विकल्प:",
                ]
                lines.extend(self._format_numbered_blocks(db_lines[:3]))
                lines.append("- छिड़काव/बीज उपचार से पहले उत्पाद लेबल, PHI और स्थानीय कृषि अधिकारी की सलाह जरूर मिलाएँ।")
                return {"answer": "\n".join(lines), "references": db_sources, "retrieved": []}
        if crop and not explicit_terms and self._is_symptom_followup_query(question):
            likely_terms: list[str] = []
            for term in disease_terms:
                term_l = str(term).strip().lower()
                if term_l and term_l not in likely_terms:
                    likely_terms.append(term_l)
            if not likely_terms:
                for issue in self._extract_common_crop_issues(crop, limit=3, issue_mode=issue_mode):
                    for term in self._extract_disease_terms_from_query(issue):
                        term_l = str(term).strip().lower()
                        if term_l and term_l not in likely_terms:
                            likely_terms.append(term_l)
            symptom_db_lines: list[str] = []
            symptom_sources: list[str] = []
            seen_lines: set[str] = set()
            for term in likely_terms[:4]:
                lines_for_term, refs_for_term = self._extract_pesticides_from_db(
                    crop,
                    disease_terms=[term],
                    limit=3,
                    symptom_text=question,
                    symptom_entry=symptom_entry,
                    strict_match=True,
                )
                if not lines_for_term:
                    lines_for_term, refs_for_term = self._extract_pesticides_for_issue(crop, term, limit=2)
                for line in lines_for_term:
                    key = re.sub(r"\s+", " ", str(line).strip().lower())
                    if not key or key in seen_lines:
                        continue
                    seen_lines.add(key)
                    symptom_db_lines.append(line)
                for ref in refs_for_term:
                    if ref and ref not in symptom_sources:
                        symptom_sources.append(ref)
                if len(symptom_db_lines) >= 3:
                    break
            if symptom_db_lines:
                disease_label = self._render_symptom_or_issue_label(symptom_label, disease_terms)
                lines = [
                    "संरचित कीटनाशक सलाह:",
                    f"- फसल: {self._crop_name_hi(crop)}",
                    f"- लक्षण मिलान: {disease_label}",
                    "- संभावित दवा विकल्प:",
                ]
                lines.extend(self._format_numbered_blocks(symptom_db_lines[:3]))
                lines.append("- अगर लक्षण और साफ लिखें, तो मैं सबसे सटीक विकल्प चुनकर dose और PHI और बेहतर कर दूँगा।")
                lines.append("- छिड़काव/बीज उपचार से पहले उत्पाद लेबल, PHI और स्थानीय कृषि अधिकारी की सलाह जरूर मिलाएँ।")
                return {"answer": "\n".join(lines), "references": symptom_sources, "retrieved": []}
        if crop and generic_issue:
            common_issues = self._extract_common_crop_issues(crop, limit=3, issue_mode=issue_mode)
            sample_sources: list[str] = []
            if common_issues:
                issue_lines = self._format_numbered_text_blocks(common_issues[:3])
                symptom_based = self._is_symptom_followup_query(question)
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
                issue_intro = (
                    f"- दिए गए लक्षण ({symptom_label}) के आधार पर इस फसल में ये {mode_label} संभावित लगते हैं:"
                    if symptom_based and symptom_label
                    else f"- दिए गए लक्षण के आधार पर इस फसल में ये {mode_label} संभावित लगते हैं:"
                    if symptom_based
                    else f"- इस फसल में आम तौर पर ये 2-3 {mode_label} ज़्यादा देखे जाते हैं:"
                )
                closing_prompt = (
                    "- अगर इनमें से कोई लक्षण सबसे ज्यादा मिल रहा हो, तो वही लिखें; फिर मैं उसी के हिसाब से सही दवा, dose और PHI बता दूँगा।"
                    if symptom_based
                    else "- आप इनमें से किसी एक का नाम या लक्षण लिखें, फिर मैं उसी के हिसाब से सही दवा, dose और PHI बता दूँगा।"
                )
                return {
                    "answer": (
                        "संरचित कीटनाशक सलाह:\n"
                        f"- फसल: {self._crop_name_hi(crop)}\n"
                        f"{issue_intro}\n"
                        f"{issue_lines}\n"
                        f"{symptom_prompt}\n"
                        f"{closing_prompt}"
                    ),
                    "references": sample_sources,
                    "retrieved": [],
                }
        if crop:
            db_lines, db_sources = self._extract_pesticides_from_db(crop, disease_terms=disease_terms, limit=8)
            if db_lines:
                disease_label = self._render_issue_label(disease_terms, fallback="दिए गए रोग/कीट")
                lines = [
                    "संरचित कीटनाशक सलाह:",
                    f"- फसल: {self._crop_name_hi(crop)}",
                    f"- रोग/कीट मिलान: {disease_label}",
                    "- दवा विकल्प:",
                ]
                lines.extend(self._format_numbered_blocks(db_lines[:3]))
                lines.append("- छिड़काव/बीज उपचार से पहले उत्पाद लेबल, PHI और स्थानीय कृषि अधिकारी की सलाह जरूर मिलाएँ।")
                return {"answer": "\n".join(lines), "references": db_sources, "retrieved": []}
            if disease_terms:
                source_lines, source_refs = self._extract_pesticides_from_source_tables(
                    crop,
                    disease_terms=disease_terms,
                    issue_mode=issue_mode,
                    limit=3,
                )
                if source_lines:
                    disease_label = self._render_issue_label(disease_terms, fallback="दिए गए रोग/कीट")
                    lines = [
                        "संरचित कीटनाशक सलाह:",
                        f"- फसल: {self._crop_name_hi(crop)}",
                        f"- रोग/कीट मिलान: {disease_label}",
                        "- दवा विकल्प:",
                    ]
                    lines.extend(self._format_numbered_blocks(source_lines[:3]))
                    lines.append("- यह उत्तर official PDF से निकाली गई table entries पर आधारित है।")
                    lines.append("- छिड़काव/बीज उपचार से पहले उत्पाद लेबल, PHI और स्थानीय कृषि अधिकारी की सलाह जरूर मिलाएँ।")
                    return {"answer": "\n".join(lines), "references": source_refs, "retrieved": []}
        if crop or pesticide_name or disease_terms:
            target = self._crop_name_hi(crop) if crop else "दिए गए प्रश्न"
            issue_hint = ", ".join(disease_terms[:2]) if disease_terms else "रोग/कीट"
            return {
                "answer": (
                    "संरचित कीटनाशक सलाह:\n"
                    f"- संदर्भ: {target}\n"
                    f"- {issue_hint} के लिए official MUP/PPQS PDF में साफ verified पंक्ति नहीं मिली।\n"
                    "- कृपया फसल, रोग/कीट का नाम या लक्षण थोड़ा और साफ लिखें, फिर मैं verified रिकॉर्ड से ही दवा बताऊँगा।"
                ),
                "references": [],
                "retrieved": [],
            }

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
            lines.append("- स्रोत से उदाहरण:")
            lines.extend(self._format_numbered_blocks(extra_lines[:3]))
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
            "white grub", "grub", "shoot borer", "top borer", "root borer",
            "कीट", "कीड़ा", "कीड़े", "दीमक", "माहू", "सफेद सूंडी", "शूट बोरर", "टॉप बोरर", "जड़ छेदक",
        ]
        disease_terms = ["fungus", "fungal", "fugal", "rog", "bimari", "disease", "फफूंद", "फंगस", "रोग", "बीमारी"]
        return any(k in t for k in pest_terms) and not any(k in t for k in disease_terms)

    def _is_disease_only_query(self, text: str) -> bool:
        t = text.lower()
        if self._is_fungal_query(text):
            return True
        return any(k in t for k in ["rog", "bimari", "disease", "रोग", "बीमारी"])

    def _load_symptom_entries(self) -> dict[str, dict[str, object]]:
        dict_mtime_ns = SYMPTOM_DICTIONARY_PATH.stat().st_mtime_ns if SYMPTOM_DICTIONARY_PATH.exists() else 0
        candidate_mtime_ns = SYMPTOM_CANDIDATE_PATH.stat().st_mtime_ns if SYMPTOM_CANDIDATE_PATH.exists() else 0
        return load_symptom_dictionary(dict_mtime_ns, candidate_mtime_ns)

    def _symptom_aliases_for_entry(self, entry: dict[str, object]) -> list[str]:
        aliases = [str(v).strip() for v in (entry.get("aliases") or []) if str(v).strip()]
        for extra in (entry.get("label_hi"), entry.get("label_en"), entry.get("canonical")):
            if str(extra or "").strip():
                aliases.append(str(extra).strip())
        out: list[str] = []
        seen: set[str] = set()
        for alias in aliases:
            normalized = normalize_symptom_text(alias)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            out.append(alias)
        return out

    def _lexical_symptom_match(
        self,
        text: str,
        entries: dict[str, dict[str, object]] | None = None,
    ) -> dict[str, object] | None:
        normalized = normalize_symptom_text(text)
        if not normalized:
            return None
        entries = entries or self._load_symptom_entries()
        best_entry: dict[str, object] | None = None
        best_score = 0
        for entry in entries.values():
            for alias in self._symptom_aliases_for_entry(entry):
                alias_norm = normalize_symptom_text(alias)
                if not alias_norm:
                    continue
                if alias_norm in normalized or normalized in alias_norm:
                    score = len(alias_norm)
                    if score > best_score:
                        best_score = score
                        best_entry = entry
        return best_entry

    def _is_candidate_symptom_phrase(self, text: str) -> bool:
        t = normalize_symptom_text(text)
        if not t or len(t) < 6:
            return False
        blockers = [
            "मौसम", "weather", "price", "rate", "भाव", "कीमत", "मंडी", "msp", "frp",
            "किस्म", "variety", "खाद", "fertilizer", "सिंचाई", "irrigation",
        ]
        if any(token in t for token in blockers):
            return False
        cues = [
            "लक्षण", "symptom", "धब्ब", "spot", "सड़न", "rot", "झुलसा", "blight",
            "सूख", "dry", "मुरझ", "wilt", "रंग", "yellow", "पीला", "पीली",
            "सफेद", "powder", "परत", "मुड़", "curl", "रस", "चूस", "कीड़ा", "कीड़े",
            "keeda", "kide", "kida", "छेद", "hole", "borer", "जड़",
        ]
        return any(token in t for token in cues)

    def _maybe_refresh_feedback_symptom_candidates(self) -> None:
        db_mtime_ns = 0
        if self.cfg.db_path and Path(self.cfg.db_path).exists():
            db_mtime_ns = Path(self.cfg.db_path).stat().st_mtime_ns
        feedback_mtime_ns = TRAINING_FEEDBACK_PATH.stat().st_mtime_ns if TRAINING_FEEDBACK_PATH.exists() else 0
        signature = (db_mtime_ns, feedback_mtime_ns)
        if self._feedback_symptom_candidate_signature == signature:
            return
        self._feedback_symptom_candidate_signature = signature
        if not self.cfg.db_path or (db_mtime_ns == 0 and feedback_mtime_ns == 0):
            return
        rows = load_symptom_feedback_rows(
            self.cfg.db_path,
            db_mtime_ns,
            str(TRAINING_FEEDBACK_PATH),
            feedback_mtime_ns,
        )
        if not rows:
            return
        self._ensure_rag_components(load_generator=False)
        if self.embedder is None:
            return
        base_entries = load_symptom_dictionary(
            SYMPTOM_DICTIONARY_PATH.stat().st_mtime_ns if SYMPTOM_DICTIONARY_PATH.exists() else 0,
            0,
        )
        if not base_entries:
            return
        phrase_rows: list[dict[str, str]] = []
        phrase_texts: list[str] = []
        for canonical, entry in base_entries.items():
            mode = str(entry.get("mode") or "general").strip().lower()
            for alias in self._symptom_aliases_for_entry(entry):
                alias_norm = normalize_symptom_text(alias)
                if not alias_norm:
                    continue
                phrase_rows.append({"canonical": canonical, "alias": alias_norm, "mode": mode})
                phrase_texts.append(alias_norm)
        if not phrase_texts:
            return
        phrase_vecs = np.asarray(self.embedder.encode(phrase_texts), dtype=np.float32)
        query_cache: dict[str, np.ndarray] = {}
        learned: dict[str, list[str]] = {}
        for row in rows:
            topic = str(row.get("topic") or "").strip().lower()
            if topic and topic not in {"pesticide", "crop_guide_followup", "crop_guide"}:
                continue
            raw_query = str(row.get("user_query") or "").strip()
            if not raw_query:
                continue
            normalized_query = normalize_symptom_text(self._normalize_hinglish(raw_query))
            if not self._is_candidate_symptom_phrase(normalized_query):
                continue
            if self._lexical_symptom_match(normalized_query, base_entries):
                continue
            if normalized_query not in query_cache:
                query_cache[normalized_query] = np.asarray(self.embedder.encode([normalized_query])[0], dtype=np.float32)
            query_vec = query_cache[normalized_query]
            scores = phrase_vecs @ query_vec
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            if best_score < 0.66:
                continue
            best = phrase_rows[best_idx]
            learned.setdefault(best["canonical"], []).append(normalized_query)
        aliases_payload: dict[str, list[str]] = {}
        for canonical, aliases in learned.items():
            cleaned = []
            seen: set[str] = set()
            for alias in aliases:
                alias_norm = normalize_symptom_text(alias)
                if not alias_norm or alias_norm in seen:
                    continue
                seen.add(alias_norm)
                cleaned.append(alias_norm)
            if cleaned:
                aliases_payload[canonical] = cleaned[:20]
        existing_payload: dict[str, object] = {}
        if SYMPTOM_CANDIDATE_PATH.exists():
            try:
                existing_payload = json.loads(SYMPTOM_CANDIDATE_PATH.read_text(encoding="utf-8"))
            except Exception:
                existing_payload = {}
        if aliases_payload != (existing_payload.get("aliases") or {}):
            save_symptom_alias_candidates(
                {
                    "generated_at": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
                    "source": "accepted_feedback",
                    "aliases": aliases_payload,
                }
            )
            self._symptom_index_signature = None
            self._symptom_phrase_rows = []
            self._symptom_phrase_embeddings = None

    def _ensure_symptom_semantic_index(self) -> None:
        self._maybe_refresh_feedback_symptom_candidates()
        dict_mtime_ns = SYMPTOM_DICTIONARY_PATH.stat().st_mtime_ns if SYMPTOM_DICTIONARY_PATH.exists() else 0
        candidate_mtime_ns = SYMPTOM_CANDIDATE_PATH.stat().st_mtime_ns if SYMPTOM_CANDIDATE_PATH.exists() else 0
        signature = (dict_mtime_ns, candidate_mtime_ns)
        if self._symptom_index_signature == signature and self._symptom_phrase_embeddings is not None:
            return
        self._ensure_rag_components(load_generator=False)
        if self.embedder is None:
            self._symptom_index_signature = signature
            self._symptom_phrase_rows = []
            self._symptom_phrase_embeddings = None
            return
        entries = self._load_symptom_entries()
        phrase_rows: list[dict[str, str]] = []
        phrase_texts: list[str] = []
        for canonical, entry in entries.items():
            mode = str(entry.get("mode") or "general").strip().lower()
            for alias in self._symptom_aliases_for_entry(entry):
                alias_norm = normalize_symptom_text(alias)
                if not alias_norm:
                    continue
                phrase_rows.append({"canonical": canonical, "alias": alias_norm, "mode": mode})
                phrase_texts.append(alias_norm)
        self._symptom_index_signature = signature
        self._symptom_phrase_rows = phrase_rows
        self._symptom_phrase_embeddings = (
            np.asarray(self.embedder.encode(phrase_texts), dtype=np.float32) if phrase_texts else None
        )

    def _match_symptom_entry(
        self,
        text: str,
        normalized_text: str | None = None,
        *,
        use_semantic: bool = True,
    ) -> dict[str, object] | None:
        normalized = normalize_symptom_text(normalized_text or text)
        if not normalized:
            return None
        lexical = self._lexical_symptom_match(normalized)
        if lexical is not None:
            return lexical
        if not use_semantic:
            return None
        if not self._is_candidate_symptom_phrase(normalized):
            return None
        self._ensure_symptom_semantic_index()
        if self.embedder is None or self._symptom_phrase_embeddings is None or not self._symptom_phrase_rows:
            return None
        query_vec = np.asarray(self.embedder.encode([normalized])[0], dtype=np.float32)
        scores = self._symptom_phrase_embeddings @ query_vec
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        if best_score < 0.52:
            return None
        best = self._symptom_phrase_rows[best_idx]
        return self._load_symptom_entries().get(best["canonical"])

    def _is_symptom_followup_query(self, text: str) -> bool:
        t = (text or "").strip().lower()
        if not t:
            return False
        if self._match_symptom_entry(text, use_semantic=True) is not None:
            return True
        disease_symptoms = [
            "lakshan", "symptom", "लक्षण",
            "dhab", "धब्ब", "spot",
            "sadan", "sadn", "rot", "सड़न",
            "jhulsa", "झुलसा", "blight",
            "sukh", "सूख", "dry", "drying",
            "murjha", "murja", "wilt", "मुरझा",
            "rang badal", "rang bd", "color change", "colour change",
            "रंग बदलना",
            "पीला", "पीली", "yellow", "yellowing",
            "safed parat", "white layer", "white powder", "powdery", "सफेद परत",
            "pattiyo ka rang", "pattion ka rang", "patto ka rang", "पत्तियों का रंग",
            "पत्ती मुड़ना",
        ]
        pest_symptoms = [
            "keeda dikh", "keede dikh", "kida dikh", "कीड़ा", "कीड़े",
            "छेद", "hole", "boring",
            "ras choos", "रस चूस", "रस चूसना",
            "patti kat", "leaf cut", "leaf damage",
            "jad nuksan", "जड़ नुकसान",
            "मुड़", "curl", "पत्ती मुड़",
        ]
        return any(term in t for term in disease_symptoms + pest_symptoms)

    def _generic_issue_mode(self, text: str) -> str:
        symptom_entry = self._match_symptom_entry(text, use_semantic=True)
        if symptom_entry is not None:
            mode = str(symptom_entry.get("mode") or "").strip().lower()
            if mode in {"fungal", "pest", "disease"}:
                return mode
        if self._is_fungal_query(text):
            return "fungal"
        if self._is_pest_only_query(text):
            return "pest"
        if self._is_disease_only_query(text):
            return "disease"
        t = (text or "").strip().lower()
        pest_symptoms = [
            "keeda dikh", "keede dikh", "kida dikh", "कीड़ा", "कीड़े",
            "छेद", "hole", "boring",
            "ras choos", "रस चूस",
            "patti kat", "leaf cut", "leaf damage",
            "jad nuksan", "जड़ नुकसान",
            "मुड़", "curl", "पत्ती मुड़",
        ]
        disease_symptoms = [
            "lakshan", "symptom", "लक्षण",
            "dhab", "धब्ब", "spot",
            "sadan", "sadn", "rot", "सड़न",
            "jhulsa", "झुलसा", "blight",
            "sukh", "सूख", "dry", "drying",
            "murjha", "murja", "wilt", "मुरझा",
            "rang badal", "rang bd", "color change", "colour change",
            "रंग बदलना",
            "पीला", "पीली", "yellow", "yellowing",
            "safed parat", "white layer", "white powder", "powdery", "सफेद परत",
            "pattiyo ka rang", "pattion ka rang", "patto ka rang", "पत्तियों का रंग",
            "पत्ती मुड़ना",
        ]
        if any(term in t for term in pest_symptoms):
            return "pest"
        if any(term in t for term in disease_symptoms):
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
        expanded_terms = self._expanded_disease_query_terms([issue_norm])
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
        if expanded_terms:
            filtered_rows: list[sqlite3.Row] = []
            for row in rows:
                issue_text = f"{row['disease_name_en'] or ''} {row['disease_name_hi'] or ''}".lower()
                if any(term in issue_text for term in expanded_terms):
                    filtered_rows.append(row)
            if filtered_rows:
                rows = filtered_rows
        rows = self._filter_rows_for_symptom_context(rows, symptom_text=issue)
        lines = []
        sources = []
        for r in rows[:limit]:
            if not self._verify_pesticide_row_against_pdf(r):
                continue
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
            source_ref = self._display_pesticide_source_reference(r["source_file"])
            if source_ref:
                sources.append(source_ref)
        return lines, sorted(set(sources))

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
        raw = re.sub(r"(?i)\)\s*and\s*", "), ", raw)
        raw = re.sub(r"(?<=[a-z\)])and(?=[A-Z])", ", ", raw)
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

    def _extract_disease_terms_from_query(self, text: str, allow_symptom_fallback: bool = True) -> list[str]:
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
        if not terms and allow_symptom_fallback:
            symptom_entry = self._match_symptom_entry(text, use_semantic=True)
            if symptom_entry is not None:
                for candidate in (symptom_entry.get("disease_candidates") or []):
                    cand = str(candidate).strip().lower()
                    if cand:
                        terms.append(cand)
        terms = self._dedupe_disease_terms(terms)
        return terms

    def _dedupe_disease_terms(self, terms: list[str]) -> list[str]:
        ordered: list[str] = []
        for term in terms:
            if term not in ordered:
                ordered.append(term)
        if "white rust" in ordered and "rust" in ordered:
            ordered.remove("rust")
        if any(term in ordered for term in ["yellow rust", "brown rust", "black rust"]) and "rust" in ordered:
            ordered.remove("rust")
        if "downy mildew" in ordered and "powdery mildew" in ordered and len(ordered) == 2:
            return ordered
        specific_insect_terms = {
            "fruit borer",
            "pod borer",
            "stem borer",
            "shoot borer",
            "top borer",
            "root borer",
            "shoot fly",
            "leaf folder",
            "diamondback moth",
            "aphid",
            "whitefly",
            "thrips",
            "jassid",
            "mite",
            "red spider mite",
            "yellow mite",
            "termite",
            "hopper",
            "caterpillar",
            "mealybug",
            "scale insect",
            "bollworm",
            "white grub",
        }
        if "insect pest" in ordered and any(term in ordered for term in specific_insect_terms):
            ordered.remove("insect pest")
        return ordered

    def _issue_mode_from_disease_terms(self, disease_terms: list[str]) -> str:
        pest_terms = {
            "fruit borer",
            "pod borer",
            "stem borer",
            "shoot borer",
            "top borer",
            "root borer",
            "shoot fly",
            "leaf folder",
            "diamondback moth",
            "aphid",
            "whitefly",
            "thrips",
            "jassid",
            "mite",
            "red spider mite",
            "yellow mite",
            "termite",
            "hopper",
            "caterpillar",
            "mealybug",
            "scale insect",
            "bollworm",
            "white grub",
            "insect pest",
        }
        if any(term in pest_terms for term in disease_terms):
            return "pest"
        if any(
            any(token in term for token in ("mildew", "rust", "blight", "smut", "bunt", "rot", "wilt", "spot", "blast"))
            for term in disease_terms
        ):
            return "disease"
        return "general"

    def _extract_pesticides_from_pdfs(self, crop: str) -> tuple[list[str], list[str]]:
        sources = []
        lines_out: list[str] = []
        root = Path("data/raw/all_sources")
        if not root.exists():
            return [], []
        crop_key = crop.lower()
        for pdf in root.glob("*.pdf"):
            try:
                pages = read_pdf_pages(pdf, prefer_docling=True)
            except Exception:
                continue
            sources.append(str(pdf))
            for page_text in pages[:10]:
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
        symptom_text: str | None = None,
        symptom_entry: dict[str, object] | None = None,
        strict_match: bool = False,
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
        expanded_terms = self._expanded_disease_query_terms(disease_terms, strict_match=strict_match)

        def score_row(r: sqlite3.Row) -> int:
            label_text = str(r["disease_name_en"] or r["disease_name_hi"] or "")
            text = f"{r['disease_name_en'] or ''} {r['disease_name_hi'] or ''}".lower()
            compact_label = self._compact_norm(label_text)
            compact_parts = [self._compact_norm(part) for part in self._split_issue_label(label_text)]
            compact_parts = [part for part in compact_parts if part]
            score = 0
            if disease_terms:
                exact_part_hits = 0
                for requested_term in disease_terms:
                    term_l = str(requested_term).strip().lower()
                    aliases = STRICT_DISEASE_QUERY_ALIASES.get(term_l) if strict_match else None
                    if not aliases:
                        aliases = DISEASE_ALIASES.get(term_l, [term_l])
                    alias_lowers = [str(alias).strip().lower() for alias in aliases if str(alias).strip()]
                    alias_compacts = [self._compact_norm(alias) for alias in alias_lowers if self._compact_norm(alias)]
                    if any(alias in text for alias in alias_lowers):
                        score += 10
                    if any(compact_term in compact_parts for compact_term in alias_compacts):
                        exact_part_hits += 1
                        score += 18
                    elif any(compact_term in compact_label for compact_term in alias_compacts):
                        score += 6
                if exact_part_hits:
                    # Prefer focused pest labels like "White grub" over broad combined rows.
                    score += 6
                if len(compact_parts) > 1:
                    score -= min(12, (len(compact_parts) - 1) * 4)
            if r["quality_status"] == "valid":
                score += 2
            elif r["quality_status"] == "usable":
                score += 1
            if r["pesticide_name"]:
                score += 1
            if r["ai_g"] or r["formulation"] or r["dilution"] or r["dose_text"]:
                score += 1
            return score

        scored_rows = [(score_row(r), r) for r in rows]
        if expanded_terms:
            scored_rows = [(score, r) for score, r in scored_rows if score >= 12]
        scored_rows.sort(key=lambda item: item[0], reverse=True)
        rows = [r for _, r in scored_rows]
        if disease_terms:
            focused_rows = [
                row for row in rows
                if self._is_focused_issue_row(
                    str(row["disease_name_en"] or row["disease_name_hi"] or ""),
                    disease_terms,
                    strict_match=strict_match,
                )
            ]
            if focused_rows:
                focused_keys = {
                    (
                        str(row["pesticide_name"] or ""),
                        str(row["disease_name_en"] or row["disease_name_hi"] or ""),
                        str(row["source_file"] or ""),
                    )
                    for row in focused_rows
                }
                rows = focused_rows + [
                    row for row in rows
                    if (
                        str(row["pesticide_name"] or ""),
                        str(row["disease_name_en"] or row["disease_name_hi"] or ""),
                        str(row["source_file"] or ""),
                    ) not in focused_keys
                ]
        rows = self._filter_rows_for_symptom_context(rows, symptom_text=symptom_text, symptom_entry=symptom_entry)[:limit]
        if not rows:
            return [], []
        lines = []
        sources = []
        for r in rows:
            if not self._verify_pesticide_row_against_pdf(r):
                continue
            lines.append(self._format_pesticide_record(r, include_crop=False))
            source_ref = self._display_pesticide_source_reference(r["source_file"])
            if source_ref:
                sources.append(source_ref)
        return lines, sorted(set(sources))

    def _is_focused_issue_row(
        self,
        label_text: str,
        disease_terms: list[str],
        *,
        strict_match: bool = False,
    ) -> bool:
        compact_parts = [self._compact_norm(part) for part in self._split_issue_label(label_text)]
        compact_parts = [part for part in compact_parts if part]
        if not compact_parts or len(compact_parts) > 2:
            return False
        for requested_term in disease_terms:
            term_l = str(requested_term).strip().lower()
            aliases = STRICT_DISEASE_QUERY_ALIASES.get(term_l) if strict_match else None
            if not aliases:
                aliases = DISEASE_ALIASES.get(term_l, [term_l])
            alias_compacts = [self._compact_norm(alias) for alias in aliases if self._compact_norm(alias)]
            if any(
                alias_compact and any(alias_compact in part for part in compact_parts)
                for alias_compact in alias_compacts
            ):
                return True
        return False

    def _expanded_disease_query_terms(self, disease_terms: list[str], strict_match: bool = False) -> list[str]:
        expanded_terms: list[str] = []
        for term in disease_terms:
            term_l = str(term).strip().lower()
            aliases = STRICT_DISEASE_QUERY_ALIASES.get(term_l) if strict_match else None
            if not aliases:
                aliases = DISEASE_ALIASES.get(term_l, [term_l])
            expanded_terms.extend(aliases)
        return [str(term).strip().lower() for term in expanded_terms if str(term).strip()]

    def _render_issue_label(self, disease_terms: list[str], fallback: str = "दिए गए लक्षण") -> str:
        labels: list[str] = []
        for term in disease_terms[:3]:
            cleaned = str(term).strip()
            if not cleaned:
                continue
            label = self._translate_disease_name(cleaned) or cleaned
            if label not in labels:
                labels.append(label)
        return ", ".join(labels) if labels else fallback

    def _render_symptom_or_issue_label(
        self,
        symptom_label: str | None,
        disease_terms: list[str],
        fallback: str = "दिए गए लक्षण",
    ) -> str:
        if symptom_label:
            translated = self._translate_disease_name(symptom_label)
            if translated and translated != symptom_label:
                return translated
            cleaned = str(symptom_label).strip()
            if cleaned:
                return cleaned
        return self._render_issue_label(disease_terms, fallback=fallback)

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
            if not self._verify_pesticide_row_against_pdf(r):
                continue
            disease = r["disease_name_hi"] or self._translate_disease_name(r["disease_name_en"] or "")
            crop_label = self._crop_name_hi(r["crop_name"] or crop or "")
            key = f"{crop_label}|{disease}"
            if key in seen:
                continue
            seen.add(key)
            lines.append(self._format_pesticide_record(r, include_crop=True, crop_override=crop))
            source_ref = self._display_pesticide_source_reference(r["source_file"])
            if source_ref:
                sources.append(source_ref)
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
            if not self._verify_pesticide_row_against_pdf(r):
                continue
            disease = r["disease_name_hi"] or self._translate_disease_name(r["disease_name_en"] or "")
            crop_label = self._crop_name_hi(r["crop_name"] or crop or "")
            pname = r["pesticide_name"] or "नाम उपलब्ध नहीं"
            key = f"{crop_label}|{disease}|{pname}"
            if key in seen:
                continue
            seen.add(key)
            lines.append(self._format_pesticide_record(r, include_crop=True, crop_override=crop))
            source_ref = self._display_pesticide_source_reference(r["source_file"])
            if source_ref:
                sources.append(source_ref)
            if len(lines) >= limit:
                break
        return lines, sorted(set(sources))

    def _build_hindi_dose_parts(self, row: sqlite3.Row) -> list[str]:
        parts: list[str] = []
        ai = self._format_value_with_unit(row["ai_g"], row["ai_unit"])
        formulation = self._format_value_with_unit(row["formulation"], row["formulation_unit"])
        dilution = self._format_value_with_unit(row["dilution"], row["dilution_unit"])
        dose_text = str(row["dose_text"] or "").strip()
        if ai and re.search(r"\d", ai):
            parts.append(f"खुराक (a.i.): {ai}")
        if formulation and re.search(r"\d", formulation):
            parts.append(f"फॉर्म्यूलेशन मात्रा: {formulation}")
        if dilution:
            parts.append(f"पानी/घोल: {dilution}")
        if dose_text:
            dose_text_norm = re.sub(r"\s+", " ", dose_text).strip().lower()
            rendered_norm = " ".join(part.lower() for part in parts)
            duplicate_markers = ["a.i.", "formulation", "dilution", "|"]
            if (
                dose_text_norm
                and not any(marker in dose_text_norm for marker in duplicate_markers)
                and dose_text_norm not in rendered_norm
            ):
                parts.append(f"डोज़/अतिरिक्त निर्देश: {dose_text}")
        return parts

    def _pesticide_row_text(self, row: sqlite3.Row) -> str:
        fields = [
            "crop_name",
            "disease_name_en",
            "disease_name_hi",
            "pesticide_name",
            "ai_g",
            "ai_unit",
            "formulation",
            "formulation_unit",
            "dilution",
            "dilution_unit",
            "dose_text",
        ]
        return " ".join(str(row[field] or "") for field in fields).lower()

    def _is_seed_treatment_row(self, row: sqlite3.Row) -> bool:
        text = self._pesticide_row_text(row)
        seed_markers = [
            "seed treatment",
            "seedtreatment",
            "seed dresser",
            "kg seed",
            "/kg seed",
            "of seed",
            "seeds are treated",
            "shade dry and sow",
            "seed borne",
            "slurry",
            "at the time of sowing",
        ]
        return any(marker in text for marker in seed_markers)

    def _is_leaf_symptom_context(self, symptom_text: str, symptom_entry: dict[str, object] | None = None) -> bool:
        normalized = normalize_symptom_text(symptom_text)
        if symptom_entry is not None:
            canonical = str(symptom_entry.get("canonical") or "").strip().lower()
            if canonical in {"leaf_spots", "leaf_color_change", "white_layer", "drying", "wilting", "leaf_curl", "holes"}:
                return True
            if canonical in {"root_damage", "sap_sucking", "insect_visible"}:
                return False
        leaf_markers = [
            "पत्ती",
            "पत्त",
            "leaf",
            "dhab",
            "धब्ब",
            "spot",
            "rang",
            "yellow",
            "सफेद",
            "powder",
            "सूख",
            "dry",
            "झुलसा",
            "मुड़",
        ]
        return any(marker in normalized for marker in leaf_markers)

    def _filter_rows_for_symptom_context(
        self,
        rows: list[sqlite3.Row],
        symptom_text: str | None = None,
        symptom_entry: dict[str, object] | None = None,
    ) -> list[sqlite3.Row]:
        if not rows or not symptom_text:
            return rows
        if self._is_leaf_symptom_context(symptom_text, symptom_entry=symptom_entry):
            filtered = [row for row in rows if not self._is_seed_treatment_row(row)]
            if filtered:
                return filtered
        return rows

    def _source_pdf_path(self, source_file: object) -> Path | None:
        source = str(source_file or "").strip()
        if not source:
            return None
        direct = Path(source)
        if direct.exists() and direct.suffix.lower() == ".pdf":
            return direct
        stem = source.split("_table_")[0]
        candidates = [
            Path("data/raw/all_sources") / f"{stem}.pdf",
            Path("data/raw/ppqs_pesticides") / f"{stem}.pdf",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def _source_table_path(self, source_file: object) -> Path | None:
        source = str(source_file or "").strip()
        if not source:
            return None
        direct = Path(source)
        if direct.exists() and direct.suffix.lower() in {".xlsx", ".xls", ".csv"}:
            return direct
        candidates = [
            Path("data/processed") / source,
        ]
        for path in candidates:
            if path.exists() and path.suffix.lower() in {".xlsx", ".xls", ".csv"}:
                return path
        source_key = source.lower()
        if self._processed_source_file_index is None:
            self._processed_source_file_index = {}
            for path in self._processed_source_table_inventory():
                self._processed_source_file_index.setdefault(path.name.lower(), path)
                for record in self._load_source_table_records(path):
                    source_ref = str(record.get("source_file") or "").strip().lower()
                    if source_ref:
                        self._processed_source_file_index.setdefault(source_ref, path)
        mapped = self._processed_source_file_index.get(source_key)
        if mapped and mapped.exists():
            return mapped
        return None

    def _load_pdf_text(self, pdf_path: Path | None) -> str:
        if not pdf_path:
            return ""
        key = str(pdf_path)
        cached = self._pdf_text_cache.get(key)
        if cached is not None:
            return cached
        try:
            text = read_pdf_text(pdf_path, prefer_docling=True)
        except Exception:
            text = ""
        self._pdf_text_cache[key] = text
        return text

    def _load_source_table_text(self, table_path: Path | None) -> str:
        if not table_path:
            return ""
        key = str(table_path)
        cached = self._source_table_text_cache.get(key)
        if cached is not None:
            return cached
        try:
            if table_path.suffix.lower() == ".csv":
                df = pd.read_csv(table_path, header=None)
            else:
                df = pd.read_excel(table_path, header=None, engine="openpyxl")
        except Exception:
            text = ""
        else:
            rows_text: list[str] = []
            for row in df.fillna("").itertuples(index=False):
                cells = [" ".join(str(cell).replace("\xa0", " ").split()) for cell in row if str(cell).strip()]
                if cells:
                    rows_text.append(" | ".join(cells))
            text = "\n".join(rows_text)
        self._source_table_text_cache[key] = text
        return text

    def _load_source_table_rows(self, table_path: Path | None) -> list[list[str]]:
        if not table_path:
            return []
        key = str(table_path)
        cached = self._source_table_rows_cache.get(key)
        if cached is not None:
            return cached
        try:
            if table_path.suffix.lower() == ".csv":
                df = pd.read_csv(table_path, header=None)
            else:
                df = pd.read_excel(table_path, header=None, engine="openpyxl")
        except Exception:
            rows: list[list[str]] = []
        else:
            rows = []
            for row in df.fillna("").itertuples(index=False):
                cells = [" ".join(str(cell).replace("\xa0", " ").split()) for cell in row if str(cell).strip()]
                rows.append(cells)
        self._source_table_rows_cache[key] = rows
        return rows

    def _processed_source_table_inventory(self) -> list[Path]:
        root = Path("data/processed")
        preferred_names = [
            "pesticide_recos_usable_with_autofill.xlsx",
            "Pesticides3.xlsx",
            "Pesticides1.xlsx",
            "Pesticdes 4.xlsx",
            "insecticides1.xlsx",
            "Insecticides2.xlsx",
        ]
        paths: list[Path] = []
        seen: set[str] = set()
        for name in preferred_names:
            path = root / name
            if path.exists():
                key = str(path.resolve())
                if key not in seen:
                    seen.add(key)
                    paths.append(path)
        for path in sorted(root.glob("*.xlsx")):
            if "reject" in path.name.lower():
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            paths.append(path)
        return paths

    def _normalize_source_column_name(self, name: object) -> str:
        cleaned = str(name or "").replace("\xa0", " ").strip().lower()
        cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned).strip("_")
        return cleaned

    def _load_source_table_records(self, table_path: Path | None) -> list[dict[str, str]]:
        if not table_path:
            return []
        key = str(table_path)
        cached = self._source_table_record_cache.get(key)
        if cached is not None:
            return cached
        try:
            if table_path.suffix.lower() == ".csv":
                df = pd.read_csv(table_path)
            else:
                df = pd.read_excel(table_path, engine="openpyxl")
        except Exception:
            records: list[dict[str, str]] = []
            self._source_table_record_cache[key] = records
            return records

        raw_columns = [str(col or "").strip() for col in df.columns]
        normalized_columns = [self._normalize_source_column_name(col) for col in raw_columns]
        default_source_file = next((col for col in raw_columns if col.lower().endswith(".xlsx")), "")

        def clean_value(value: object) -> str:
            text = str(value or "").replace("\xa0", " ").strip()
            if not text or text.lower() == "nan":
                return ""
            return " ".join(text.split())

        def pick(row_map: dict[str, str], names: list[str]) -> str:
            for name in names:
                value = row_map.get(name, "")
                if value:
                    return value
            return ""

        def pick_with_key(row_map: dict[str, str], names: list[str]) -> tuple[str, str]:
            for name in names:
                value = row_map.get(name, "")
                if value:
                    return value, name
            return "", ""

        records = []
        for row in df.fillna("").itertuples(index=False, name=None):
            row_map = {
                normalized_columns[idx]: clean_value(value)
                for idx, value in enumerate(row)
                if idx < len(normalized_columns)
            }
            ai_value, ai_key = pick_with_key(row_map, ["ai_g", "a_i_gm_ha", "a_i_gm", "dose", "2_a_i_mg_m"])
            formulation_value, formulation_key = pick_with_key(
                row_map,
                ["formulation", "formulation_ml_ha", "formulation_gm", "formulation_kg_ha", "dose_text"],
            )
            dilution_value, dilution_key = pick_with_key(
                row_map,
                ["dilution", "water_l_ha", "water", "dilution_in_water_liters", "surface", "exposureperiod"],
            )
            waiting_value, waiting_key = pick_with_key(
                row_map,
                ["waiting_period_day", "waiting_period_days", "waiting_period", "aeration_waiting_period"],
            )
            record = {
                "crop": pick(row_map, ["crop_name", "crop", "nameofcommodity", "commodity", "name_of_commodity"]),
                "issue": pick(
                    row_map,
                    [
                        "disease_name_en",
                        "disease_name_hi",
                        "targetpest",
                        "commonnameof_thepest",
                        "pest",
                        "nameof_insect",
                        "nameofinsect",
                        "insect_name_hi",
                        "insects_name_hi",
                    ],
                ),
                "pesticide_name": pick(
                    row_map,
                    ["pesticide_name", "pesticide", "pesticides", "pesticide_name_", "pesticide_name__"],
                ),
                "ai": ai_value,
                "ai_key": ai_key,
                "formulation": formulation_value,
                "formulation_key": formulation_key,
                "dilution": dilution_value,
                "dilution_key": dilution_key,
                "waiting": waiting_value,
                "waiting_key": waiting_key,
                "dose_text": pick(row_map, ["dose_text"]),
                "source_file": pick(row_map, ["source_file"]) or default_source_file,
                "quality_status": pick(row_map, ["quality_status"]),
                "validation_status": pick(row_map, ["validation_status"]),
            }
            row_text = " | ".join(value for value in record.values() if value)
            if not record["pesticide_name"] and not row_text:
                continue
            record["row_text"] = row_text
            records.append(record)
        self._source_table_record_cache[key] = records
        return records

    def _candidate_source_table_paths(self, issue_mode: str = "general") -> list[Path]:
        preferred_by_mode = {
            "pest": ["Pesticides3.xlsx", "pesticide_recos_usable_with_autofill.xlsx", "insecticides1.xlsx", "Insecticides2.xlsx"],
            "fungal": ["pesticide_recos_usable_with_autofill.xlsx", "Pesticides3.xlsx"],
            "disease": ["pesticide_recos_usable_with_autofill.xlsx", "Pesticides3.xlsx"],
            "general": ["pesticide_recos_usable_with_autofill.xlsx", "Pesticides3.xlsx", "insecticides1.xlsx", "Insecticides2.xlsx"],
        }
        paths: list[Path] = []
        seen: set[str] = set()
        inventory = self._processed_source_table_inventory()
        preferred_names = preferred_by_mode.get(issue_mode, preferred_by_mode["general"])
        ordered = [path for name in preferred_names for path in inventory if path.name == name]
        ordered.extend(path for path in inventory if path not in ordered)
        for path in ordered:
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            paths.append(path)
        return paths

    def _source_record_unit_from_key(self, key: str) -> str:
        normalized = str(key or "").strip().lower()
        unit_map = {
            "ai_g": "g/ha",
            "a_i_gm_ha": "g/ha",
            "a_i_gm": "g",
            "2_a_i_mg_m": "mg/m2",
            "formulation_ml_ha": "ml/ha",
            "formulation_kg_ha": "kg/ha",
            "formulation_gm": "g",
            "dilution": "L/ha",
            "water_l_ha": "L/ha",
            "dilution_in_water_liters": "L/ha",
            "waiting_period_day": "days",
            "waiting_period_days": "days",
            "waiting_period": "days",
            "aeration_waiting_period": "days",
        }
        return unit_map.get(normalized, "")

    def _infer_formulation_unit_from_pesticide_name(self, pesticide_name: str) -> str:
        text = str(pesticide_name or "").upper().replace(" ", "")
        if not text:
            return ""
        text = re.sub(r"[^A-Z.]+$", "", text)
        match = re.search(r"(WDG|WG|WP|SP|WS|SG|DP|GR|SC|EC|SL|SE|ZC|OD|ME|FS|ES|CS|EW)\.?$", text)
        if not match:
            return ""
        code = match.group(1)
        if code in {"WDG", "WG", "WP", "SP", "WS", "SG", "DP", "GR"}:
            return "g/ha"
        if code in {"SC", "EC", "SL", "SE", "ZC", "OD", "ME", "FS", "ES", "CS", "EW"}:
            return "ml/ha"
        return ""

    def _build_hindi_dose_parts_from_source_record(self, record: dict[str, str]) -> list[str]:
        parts: list[str] = []
        ai = self._format_value_with_unit(record.get("ai"), self._source_record_unit_from_key(record.get("ai_key", "")))
        formulation_unit = self._source_record_unit_from_key(record.get("formulation_key", ""))
        if not formulation_unit:
            formulation_unit = self._infer_formulation_unit_from_pesticide_name(record.get("pesticide_name", ""))
        formulation = self._format_value_with_unit(
            record.get("formulation"),
            formulation_unit,
        )
        dilution = self._format_value_with_unit(
            record.get("dilution"),
            self._source_record_unit_from_key(record.get("dilution_key", "")),
        )
        waiting = self._format_value_with_unit(
            record.get("waiting"),
            self._source_record_unit_from_key(record.get("waiting_key", "")),
        )
        dose_text = str(record.get("dose_text") or "").strip()
        if ai and re.search(r"\d", ai):
            parts.append(f"खुराक (a.i.): {ai}")
        if formulation and re.search(r"\d", formulation):
            parts.append(f"फॉर्म्यूलेशन मात्रा: {formulation}")
        if dilution:
            parts.append(f"पानी/घोल: {dilution}")
        if waiting and re.search(r"\d", waiting):
            parts.append(f"प्रतीक्षा अवधि (Waiting period): {waiting}")
        if dose_text:
            dose_text_norm = re.sub(r"\s+", " ", dose_text).strip().lower()
            rendered_norm = " ".join(part.lower() for part in parts)
            duplicate_markers = ["a.i.", "formulation", "dilution", "|"]
            if (
                dose_text_norm
                and not any(marker in dose_text_norm for marker in duplicate_markers)
                and dose_text_norm not in rendered_norm
            ):
                parts.append(f"डोज़/अतिरिक्त निर्देश: {dose_text}")
        return parts

    def _format_source_pesticide_record(self, record: dict[str, str]) -> str:
        pname = str(record.get("pesticide_name") or "नाम उपलब्ध नहीं").strip()
        issue_label = str(record.get("issue") or "").strip()
        disease = self._translate_disease_name(issue_label) or issue_label
        lines = [f"दवा: {pname}"]
        if disease:
            lines.append(f"रोग/कीट: {disease}")
        lines.extend(self._build_hindi_dose_parts_from_source_record(record))
        return "\n".join(lines).strip()

    def _looks_like_pesticide_name(self, text: str) -> bool:
        t = str(text or "").strip()
        if not t:
            return False
        return bool(
            re.search(r"(%|WG|WP|SC|GR|EC|ZC|SE|SL|WS|SP)\b", t, flags=re.IGNORECASE)
        )

    def _infer_pesticide_name_from_table_rows(self, rows: list[list[str]], row_idx: int) -> str:
        for prev_idx in range(row_idx - 1, max(-1, row_idx - 4), -1):
            prev_cells = rows[prev_idx]
            if not prev_cells:
                continue
            candidate = " ".join(prev_cells).strip()
            if self._looks_like_pesticide_name(candidate):
                return candidate
        return ""

    def _extract_pesticides_from_source_tables(
        self,
        crop: str,
        disease_terms: list[str],
        *,
        issue_mode: str = "general",
        limit: int = 5,
    ) -> tuple[list[str], list[str]]:
        crop_terms = [self._compact_norm(crop), self._compact_norm(self._crop_name_hi(crop))]
        crop_terms = [term for term in crop_terms if term]
        alias_compacts: list[str] = []
        for term in disease_terms:
            aliases = DISEASE_ALIASES.get(str(term).strip().lower(), [str(term).strip().lower()])
            for alias in aliases:
                compact = self._compact_norm(alias)
                if compact and compact not in alias_compacts:
                    alias_compacts.append(compact)
        matches: list[tuple[int, str, str]] = []
        refs: list[str] = []
        seen_lines: set[str] = set()
        for table_path in self._candidate_source_table_paths(issue_mode=issue_mode):
            records = self._load_source_table_records(table_path)
            if not records:
                continue
            for record in records:
                quality_status = str(record.get("quality_status") or "").strip().lower()
                validation_status = str(record.get("validation_status") or "").strip().lower()
                if quality_status == "reject" or validation_status == "reject":
                    continue
                row_text = str(record.get("row_text") or "").strip()
                if not row_text:
                    continue
                compact_row = self._compact_norm(row_text)
                record_crop = str(record.get("crop") or "").strip()
                record_crop_hi = self._crop_name_hi(record_crop) if record_crop else ""
                row_crop_terms = [self._compact_norm(record_crop)]
                if record_crop_hi and record_crop_hi != record_crop:
                    row_crop_terms.append(self._compact_norm(record_crop_hi))
                row_crop_terms = [term for term in row_crop_terms if term]
                crop_match = any(term in compact_row for term in crop_terms) or any(
                    crop_term in row_term or row_term in crop_term
                    for crop_term in crop_terms
                    for row_term in row_crop_terms
                )
                if crop_terms and not crop_match:
                    continue
                if alias_compacts and not any(alias in compact_row for alias in alias_compacts):
                    continue
                pesticide_name = str(record.get("pesticide_name") or "").strip()
                issue_label = str(record.get("issue") or "").strip() or ", ".join(disease_terms[:1])
                line = self._format_source_pesticide_record(record)
                line_key = re.sub(r"\s+", " ", line).strip().lower()
                if not line_key or line_key in seen_lines:
                    continue
                seen_lines.add(line_key)
                focus_bonus = 0
                split_parts = [self._compact_norm(part) for part in self._split_issue_label(issue_label)]
                split_parts = [part for part in split_parts if part]
                if self._is_focused_issue_row(issue_label, disease_terms, strict_match=True):
                    focus_bonus += 28
                if len(split_parts) <= 2 and any(alias in part for alias in alias_compacts for part in split_parts):
                    focus_bonus += 20
                if any(alias == part for alias in alias_compacts for part in split_parts):
                    focus_bonus += 20
                if len(split_parts) > 1:
                    focus_bonus -= min(14, (len(split_parts) - 1) * 4)
                if pesticide_name:
                    focus_bonus += 4
                if record_crop and any(term in self._compact_norm(record_crop) for term in crop_terms):
                    focus_bonus += 8
                if quality_status == "valid":
                    focus_bonus += 12
                elif quality_status == "usable":
                    focus_bonus += 5
                if validation_status == "valid":
                    focus_bonus += 4
                if table_path.name == "pesticide_recos_usable_with_autofill.xlsx":
                    focus_bonus += 10
                elif table_path.name == "Pesticides3.xlsx":
                    focus_bonus += 6
                source_ref = self._display_pesticide_source_reference(record.get("source_file") or table_path.name) or str(table_path)
                matches.append((focus_bonus, line, source_ref))
        matches.sort(key=lambda item: item[0], reverse=True)
        lines: list[str] = []
        for _score, line, ref in matches[:limit]:
            lines.append(line)
            if ref not in refs:
                refs.append(ref)
        return lines, refs

    def _display_pesticide_source_reference(self, source_file: object) -> str | None:
        table_path = self._source_table_path(source_file)
        pdf_path = self._source_pdf_path(source_file)
        if table_path and pdf_path:
            return f"{table_path} (verified against {pdf_path.name})"
        if table_path:
            return str(table_path)
        if pdf_path:
            return str(pdf_path)
        source = str(source_file or "").strip()
        if not source:
            return None
        if source.lower().endswith(".xlsx"):
            return None
        return source

    def _compact_norm(self, text: object) -> str:
        return re.sub(r"[^a-z0-9\u0900-\u097F]+", "", str(text or "").lower())

    def _source_chemical_tokens(self, pesticide_name: str) -> list[str]:
        stop = {"with", "plus", "min", "based", "strain", "serotype", "potency"}
        tokens = [
            tok.lower()
            for tok in re.findall(r"[A-Za-z]{5,}", pesticide_name or "")
            if tok.lower() not in stop
        ]
        unique: list[str] = []
        for tok in tokens:
            if tok not in unique:
                unique.append(tok)
        return unique

    def _source_issue_tokens(self, row: sqlite3.Row) -> list[str]:
        raw = " ".join(
            [
                str(row["disease_name_en"] or ""),
                str(row["disease_name_hi"] or ""),
            ]
        )
        stop = {
            "disease", "leaf", "brown", "black", "yellow", "rice", "wheat", "crop",
            "पत्ती", "रोग", "कीट",
        }
        tokens = [
            tok.lower()
            for tok in re.findall(r"[A-Za-z\u0900-\u097F]{3,}", raw)
            if tok.lower() not in stop
        ]
        unique: list[str] = []
        for tok in tokens:
            if tok not in unique:
                unique.append(tok)
        return unique[:8]

    def _row_matches_pesticide_source_text(
        self,
        row: sqlite3.Row,
        source_text: str,
        require_crop: bool = False,
    ) -> bool:
        if not source_text:
            return False
        source_compact = self._compact_norm(source_text)
        crop = str(row["crop_name"] or "").strip()
        crop_tokens = [self._compact_norm(crop)] if crop else []
        crop_hi = self._crop_name_hi(crop) if crop else ""
        if crop_hi and crop_hi != crop:
            crop_tokens.append(self._compact_norm(crop_hi))
        crop_ok = not crop_tokens or any(tok and tok in source_compact for tok in crop_tokens)
        pname = str(row["pesticide_name"] or "").strip()
        chem_tokens = self._source_chemical_tokens(pname)
        chem_hits = sum(1 for tok in chem_tokens if tok in source_compact)
        chem_ok = self._compact_norm(pname) in source_compact or chem_hits >= min(2, max(1, len(chem_tokens)))
        issue_tokens = self._source_issue_tokens(row)
        issue_ok = True
        if issue_tokens:
            issue_ok = any(self._compact_norm(tok) in source_compact for tok in issue_tokens)
        if require_crop:
            return bool(crop_ok and chem_ok and issue_ok)
        if chem_ok and issue_ok:
            return True
        if chem_ok and crop_ok and not issue_tokens:
            return True
        return False

    def _verify_pesticide_row_against_pdf(self, row: sqlite3.Row) -> bool:
        source_file = str(row["source_file"] or "")
        pname = str(row["pesticide_name"] or "").strip()
        crop = str(row["crop_name"] or "").strip()
        cache_key = "||".join(
            [
                source_file,
                pname,
                crop,
                str(row["disease_name_en"] or ""),
                str(row["disease_name_hi"] or ""),
            ]
        )
        cached = self._pdf_verification_cache.get(cache_key)
        if cached is not None:
            return cached
        table_path = self._source_table_path(source_file)
        table_text = self._load_source_table_text(table_path)
        if self._row_matches_pesticide_source_text(row, table_text):
            self._pdf_verification_cache[cache_key] = True
            return True
        pdf_path = self._source_pdf_path(source_file)
        pdf_text = self._load_pdf_text(pdf_path)
        verified = self._row_matches_pesticide_source_text(row, pdf_text, require_crop=True)
        self._pdf_verification_cache[cache_key] = verified
        return verified

    def _format_pesticide_record(
        self,
        row: sqlite3.Row,
        include_crop: bool = False,
        crop_override: str | None = None,
    ) -> str:
        pname = str(row["pesticide_name"] or "नाम उपलब्ध नहीं").strip()
        disease = row["disease_name_hi"] or self._translate_disease_name(row["disease_name_en"] or "")
        crop_label = self._crop_name_hi(row["crop_name"] or crop_override or "")
        dose_parts = self._build_hindi_dose_parts(row)
        waiting = self._format_value_with_unit(row["waiting_period_days"], row["waiting_period_unit"])
        lines = [f"दवा: {pname}"]
        if include_crop and crop_label:
            lines.append(f"फसल: {crop_label}")
        if disease:
            lines.append(f"रोग/कीट: {disease}")
        lines.extend(dose_parts)
        if waiting:
            lines.append(f"PHI: {waiting}")
        return "\n".join(lines).strip()

    def _format_numbered_blocks(self, blocks: list[str]) -> list[str]:
        output: list[str] = []
        for idx, block in enumerate(blocks, start=1):
            lines = [ln.strip() for ln in str(block).splitlines() if ln.strip()]
            if not lines:
                continue
            output.append(f"{idx}. {lines[0]}")
            output.extend([f"   {line}" for line in lines[1:]])
        return output

    def _format_numbered_text_blocks(self, items: list[str]) -> str:
        return "\n".join(f"{idx}. {str(item).strip()}" for idx, item in enumerate(items, start=1) if str(item).strip())

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
        raw_entry = GENERATED_RAW_DISEASE_DISPLAY.get(str(text).strip().lower())
        if raw_entry and len(self._split_issue_label(text)) <= 1:
            clean_en = str(raw_entry.get("clean_english") or "").strip()
            clean_hi = str(raw_entry.get("clean_hindi") or "").strip()
            if clean_hi and clean_en:
                return f"{clean_hi} ({self._clean_pest_label(clean_en)})"
            if clean_hi:
                return clean_hi
            if clean_en:
                text = clean_en
        lower = text.lower().replace("downey", "downy")
        compact_lower = self._compact_norm(lower)
        alias_hits: list[str] = []
        alias_hi_hits: list[str] = []
        for canonical, aliases in DISEASE_ALIASES.items():
            canonical_l = str(canonical).strip().lower()
            if any(alias and (str(alias).lower() in lower or self._compact_norm(alias) in compact_lower) for alias in aliases):
                hi = DISEASE_HINDI_TERMS.get(canonical_l, "")
                if hi and hi not in alias_hi_hits:
                    alias_hi_hits.append(hi)
                clean_label = self._clean_pest_label(canonical)
                if clean_label and clean_label not in alias_hits:
                    alias_hits.append(clean_label)
        skip_keys: set[str] = set()
        if "white rust" in lower:
            skip_keys.add("rust")
        if "alternaria blight" in lower:
            skip_keys.add("blight")
        if any(term in lower for term in ["yellow rust", "stripe rust", "brown rust", "leaf rust", "black rust", "stem rust"]):
            skip_keys.add("rust")
        hits = []
        for key, hi in DISEASE_HINDI_TERMS.items():
            if key in skip_keys:
                continue
            compact_key = self._compact_norm(key)
            if (key in lower or (compact_key and compact_key in compact_lower)) and hi not in hits:
                hits.append(hi)
        for hi in alias_hi_hits:
            if hi not in hits:
                hits.append(hi)
        if hits:
            return f"{' / '.join(hits)} ({self._clean_pest_label(text)})"
        if alias_hits:
            return self._clean_pest_label(" / ".join(alias_hits))
        return self._clean_pest_label(text)

    def _clean_pest_label(self, text: str) -> str:
        if not text:
            return ""
        cleaned = str(text).replace("\xa0", " ")
        cleaned = re.sub(r"\bdowney\b", "Downy", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"[A-Za-z]{8,}", lambda m: self._split_agri_compound(m.group(0)), cleaned)
        cleaned = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", cleaned)
        cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"(?:\s|,)+(?:and|or|&)\s*$", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip(" ,;/:-")

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

    def _web_search_include_domains(self, question: str) -> list[str]:
        q = (question or "").lower()
        if self._is_msp_query(q) or self._is_price_query(q):
            return [
                "pib.gov.in",
                "agmarknet.gov.in",
                "enam.gov.in",
                "apeda.gov.in",
                "agriexchange.apeda.gov.in",
                "indianspices.com",
            ]
        if self._is_crop_protection_followup_intent(q):
            return ["ppqs.gov.in", "icar.gov.in", "agricoop.nic.in"]
        if self._is_crop_guide_intent(q):
            return ["icar.gov.in", "tnau.ac.in", "agricoop.nic.in"]
        return ["icar.gov.in", "agricoop.nic.in", "ppqs.gov.in"]

    def _answer_agri_term_query(self, raw_question: str, normalized_question: str) -> tuple[str | None, list[str]]:
        raw = (raw_question or "").strip()
        normalized = (normalized_question or "").strip()
        haystack = f"{raw} {normalized}".lower()
        if not haystack:
            return None, []
        glossary_entry = match_glossary_entry(haystack)
        if glossary_entry:
            return format_glossary_answer(glossary_entry), glossary_references(glossary_entry)
        explanation_cues = [
            "kya hota hai",
            "क्या होता है",
            "kya hai",
            "क्या है",
            "matlab",
            "मतलब",
            "meaning",
            "full form",
            "ka full form",
            "का फुल फॉर्म",
        ]
        asks_for_meaning = any(cue in haystack for cue in explanation_cues)
        matched: dict[str, str] = {}
        for key, payload in AGRI_TERM_EXPLANATIONS.items():
            label = str(payload.get("label") or key)
            meaning = str(payload.get("meaning") or "")
            if (
                re.search(rf"\b{re.escape(key)}\b", haystack)
                or re.search(rf"\b{re.escape(label.lower())}\b", haystack)
                or (meaning and meaning.lower() in haystack)
            ):
                matched = payload
                break
        if not matched:
            return None, []
        if not asks_for_meaning and raw.strip().upper() != str(matched.get("label") or "").upper():
            return None, []
        label = str(matched.get("label") or "").strip()
        meaning = str(matched.get("meaning") or "").strip()
        explanation = str(matched.get("explanation") or "").strip()
        lines = [f"{label} का मतलब: {meaning}"]
        if explanation:
            lines.append(explanation)
        return "\n".join(lines).strip(), []

    def _extract_district_from_context(self, context_part: str) -> str | None:
        return self._extract_district(context_part)

    def _build_web_search_queries(self, question: str, context_part: str) -> list[str]:
        q = (question or "").strip()
        if not q:
            return []
        crop = self._extract_crop_from_query(q) or self._extract_preferred_crop_from_context(context_part) or ""
        district = self._extract_district_from_context(context_part) or ""
        disease_terms = self._extract_disease_terms_from_query(q)
        queries: list[str] = []
        if disease_terms and crop:
            queries.append(f"{crop} {' '.join(disease_terms[:2])} advisory India")
        if crop and self._is_msp_query(q):
            queries.append(f"{crop} MSP India official")
        if crop and self._is_price_query(q):
            if district:
                queries.append(f"{crop} mandi price {district} India")
                queries.append(f"{crop} market price {district} India")
            queries.append(f"{crop} mandi price India")
            queries.append(f"{crop} market price India")
            queries.append(f"{crop} current price India")
        if crop and district and self._is_crop_choice_intent(q):
            queries.append(f"{crop} farming economics {district} Uttar Pradesh India")
        queries.append(q)
        if crop and crop.lower() not in q.lower():
            queries.append(f"{crop} {q}")
        # preserve order but deduplicate
        seen: set[str] = set()
        out: list[str] = []
        for item in queries:
            norm = re.sub(r"\s+", " ", item).strip().lower()
            if norm and norm not in seen:
                seen.add(norm)
                out.append(item)
        return out[:4]

    def _rerank_web_results(self, question: str, results: list[dict], top_k: int = 4) -> list[dict]:
        q_tokens = self._query_tokens(question)
        crop = (self._extract_crop_from_query(question) or "").lower()
        disease_terms = [d.lower() for d in self._extract_disease_terms_from_query(question)]
        ranked: list[tuple[float, dict]] = []
        for item in results:
            hay = f"{item.get('title','')} {item.get('snippet','')} {item.get('source_file','')}".lower()
            tokens = self._query_tokens(hay)
            overlap = len(q_tokens & tokens) / max(len(q_tokens), 1) if q_tokens else 0.0
            crop_bonus = 0.25 if crop and crop in hay else 0.0
            disease_bonus = 0.2 if disease_terms and any(term in hay for term in disease_terms) else 0.0
            official_bonus = 0.15 if any(dom in hay for dom in ["icar", "ppqs", "agricoop", "pib", "agmarknet"]) else 0.0
            score = overlap + crop_bonus + disease_bonus + official_bonus
            ranked.append((score, item))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in ranked[:top_k]]

    def _answer_with_web_search(self, question: str, context_part: str) -> dict | None:
        if not is_web_search_configured():
            return None
        if self._is_weather_intent(question) or self._is_weather_impact_intent(question):
            return None
        include_domains = self._web_search_include_domains(question)
        search_results: list[dict] = []
        for query in self._build_web_search_queries(question, context_part):
            hits = web_search(query, num=5, include_domains=include_domains)
            if not hits:
                continue
            for hit in hits:
                search_results.append(
                    {
                        "source_file": hit.link,
                        "title": hit.title,
                        "text": hit.snippet,
                    }
                )
            if search_results:
                break
        if not search_results:
            return None
        ranked = self._rerank_web_results(question, search_results, top_k=4)
        evidence = "\n\n".join(
            f"[Source: {item.get('title','unknown')} | {item.get('source_file','')}]\n{item.get('text','')[:320]}"
            for item in ranked
        )
        if self.generator is not None:
            prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                "नीचे Google Programmable Search से मिले स्रोत-स्निपेट हैं। केवल इन्हीं स्रोतों के आधार पर उत्तर दें। "
                "अगर जानकारी अधूरी हो तो साफ बताएं।\n\n"
                f"स्रोत:\n{evidence}\n\n"
                f"किसान का सवाल: {question}\n\n"
                "उत्तर हिंदी में दें। संरचना रखें:\n"
                "1) सीधा उत्तर\n2) जरूरी संदर्भ/सीमा\n3) अगला सबसे उपयोगी कदम\n"
            )
            try:
                answer = self.generator.generate(prompt)
            except Exception:
                answer = ""
        else:
            answer = ""
        if not answer.strip():
            bullets = "\n".join(
                f"- {item.get('title','')}: {item.get('text','')}".strip()
                for item in ranked[:3]
            )
            answer = (
                f"समझा गया सवाल (हिंदी): {question}\n\n"
                "स्थानीय स्रोत से सीधा उत्तर नहीं मिला, इसलिए वेब स्रोतों से ये प्रासंगिक जानकारी मिली:\n"
                f"{bullets}"
            )
        return {
            "answer": answer,
            "references": [item.get("source_file", "") for item in ranked if item.get("source_file")],
            "retrieved": ranked,
            "topic": "web_search",
        }

    def _parse_weather_request(self, farmer_question: str, normalized_question: str) -> WeatherRequest | None:
        raw = (farmer_question or "").strip()
        normalized = (normalized_question or "").strip()
        if not (self._is_weather_intent(raw) or self._is_weather_intent(normalized)):
            return None
        place = self._extract_location_from_question(raw) or self._extract_location_from_question(normalized)
        forecast_target = self._extract_weather_forecast_target(normalized) or self._extract_weather_forecast_target(raw)
        rain_focus = self._is_rain_specific_query(normalized) or self._is_rain_specific_query(raw)
        action = "current"
        day_offset = None
        label = None
        if forecast_target:
            action = "daily_rain" if rain_focus else "daily"
            day_offset = int(forecast_target["day_offset"])
            label = str(forecast_target["label"])
        elif self._is_rain_day_forecast_query(normalized) or self._is_rain_day_forecast_query(raw):
            action = "rain_day"
        elif self._is_weekly_weather_query(normalized) or self._is_weekly_weather_query(raw):
            action = "rain_day" if rain_focus else "weekly"
        return WeatherRequest(
            place=place,
            action=action,
            day_offset=day_offset,
            label=label,
            focus="rain" if action in {"daily_rain", "rain_day"} else "weather",
        )

    def _resolve_weather_location(
        self,
        place: str | None,
        farmer_question: str,
        normalized_question: str,
    ) -> tuple[str | None, dict | None]:
        loc = lookup_place(place) if place else None
        if not loc:
            loc = lookup_place_in_text(farmer_question) or lookup_place_in_text(normalized_question)
        if place and not loc:
            generic_weather_tokens = {
                "baarish", "barish", "rain", "rainfall", "mausam", "weather",
                "konse", "kaunse", "kis", "din", "kab", "ki", "ka", "ke",
                "hai", "h", "hoga", "hogi", "ho", "rahega", "rahegi", "rahenge", "rahe",
                "agle", "agla", "hafta", "hafte", "hafte", "saptah", "week", "coming", "next",
                "आज", "कल", "परसों", "अगले", "अगला", "सप्ताह", "हफ्ता", "हफ्ते", "रहेगा", "रहेगी",
                "बारिश", "बारिस", "मौसम", "किस", "दिन", "कौनसे", "कौन", "कब",
                "है", "होगा", "होगी",
            }
            calendar_tokens = {
                "today", "tomorrow", "day", "after", "next", "coming",
                "aaj", "aj", "kal", "parso", "agle", "agla",
                "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
                "सोमवार", "मंगलवार", "बुधवार", "गुरुवार", "शुक्रवार", "शनिवार", "रविवार",
                "jan", "january", "feb", "february", "mar", "march", "apr", "april", "may",
                "jun", "june", "jul", "july", "aug", "august", "sep", "sept", "september",
                "oct", "october", "nov", "november", "dec", "december",
                "जनवरी", "फ़रवरी", "फरवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई",
                "अगस्त", "सितंबर", "सितम्बर", "अक्टूबर", "नवंबर", "नवम्बर", "दिसंबर", "दिसम्बर",
            }
            raw_tokens = [tok.strip(" ?!.,") for tok in re.split(r"\s+", place) if tok.strip(" ?!.,")]
            filtered_tokens = [
                tok
                for tok in raw_tokens
                if tok.lower() not in generic_weather_tokens
                and tok not in generic_weather_tokens
                and tok.lower() not in calendar_tokens
                and tok not in calendar_tokens
                and not re.fullmatch(r"\d{1,4}", tok)
            ]
            sanitized_candidates = []
            if filtered_tokens:
                sanitized_candidates.append(" ".join(filtered_tokens))
                sanitized_candidates.append(filtered_tokens[0])
            for cand in sanitized_candidates:
                loc = lookup_place(cand) or lookup_place_in_text(cand)
                if loc:
                    place = loc.get("place") or cand
                    break
            if not loc and not filtered_tokens:
                place = None
        if loc and loc.get("place"):
            place = loc.get("place")
        return place, loc

    def _answer_weather_request(
        self,
        request: WeatherRequest,
        farmer_question: str,
        normalized_question: str,
        place_override: str | None = None,
    ) -> dict:
        place, loc = self._resolve_weather_location(
            place_override or request.place,
            farmer_question,
            normalized_question,
        )
        if not place and not loc:
            return {
                "answer": "मौसम के लिए स्थान नहीं मिला। कृपया स्थान लिखें (जैसे: डोघाट/बड़ौत/मेरठ)।",
                "references": [],
                "retrieved": [],
            }
        loc = loc or (lookup_place(place) if place else None)
        district = loc.get("district") if loc else self._lookup_district_from_location(place)
        state = loc.get("state") if loc else "Uttar Pradesh"
        weather_place = place if not district else f"{place}, {district}, {state}"
        if request.action == "rain_day":
            weather = get_rain_day_forecast_hindi(weather_place, days=7)
        elif request.action == "weekly":
            weather = get_weekly_weather_forecast_hindi(weather_place, days=7)
        elif request.action in {"daily", "daily_rain"} and request.day_offset is not None:
            weather = get_daily_weather_forecast_hindi(
                weather_place,
                day_offset=request.day_offset,
                label=request.label,
                rain_focus=request.action == "daily_rain",
            )
        else:
            weather = get_current_weather_hindi(weather_place)
            if not weather:
                weather = get_current_weather_hindi(weather_place)
        if not weather:
            return {
                "answer": "अभी लाइव मौसम डेटा नहीं मिल पाया। कृपया कुछ देर बाद फिर प्रयास करें।",
                "references": [],
                "retrieved": [],
                "weather_action": request.action,
                "weather_focus": request.focus,
            }
        return {
            "answer": weather,
            "references": ["Open-Meteo API"],
            "retrieved": [],
            "weather_action": request.action,
            "weather_focus": request.focus,
        }

    def _parse_pesticide_request(
        self,
        farmer_question: str,
        normalized_question: str,
        context_part: str = "",
    ) -> PesticideRequest | None:
        raw = (farmer_question or "").strip()
        normalized = (normalized_question or "").strip()
        if not (self._is_crop_protection_followup_intent(raw) or self._is_crop_protection_followup_intent(normalized)):
            return None
        direct_crop = self._extract_crop_from_query(raw) or self._extract_crop_from_query(normalized)
        context_crop = self._extract_preferred_crop_from_context(context_part)
        crop = direct_crop or context_crop
        pesticide_name = self._extract_pesticide_name_from_query(raw) or self._extract_pesticide_name_from_query(normalized)
        if not pesticide_name:
            pesticide_name = self._infer_pesticide_name_from_query_tokens(raw, crop=crop) or self._infer_pesticide_name_from_query_tokens(normalized, crop=crop)
        disease_terms = (
            self._extract_disease_terms_from_query(normalized, allow_symptom_fallback=False)
            or self._extract_disease_terms_from_query(raw, allow_symptom_fallback=False)
        )
        symptom_entry = None if disease_terms else self._match_symptom_entry(raw, normalized, use_semantic=True)
        if not disease_terms and symptom_entry is not None:
            disease_terms = [str(t).strip().lower() for t in (symptom_entry.get("disease_candidates") or []) if str(t).strip()]
        issue_mode = (
            str(symptom_entry.get("mode") or "").strip().lower()
            if symptom_entry is not None
            else self._issue_mode_from_disease_terms(disease_terms) if disease_terms else self._generic_issue_mode(normalized or raw)
        )
        if issue_mode not in {"fungal", "pest", "disease", "general"}:
            issue_mode = self._issue_mode_from_disease_terms(disease_terms) if disease_terms else self._generic_issue_mode(normalized or raw)
        generic_issue = bool(crop and not pesticide_name and self._is_generic_issue_query(normalized or raw, issue_mode=issue_mode))
        return PesticideRequest(
            crop=crop,
            crop_from_context=bool(context_crop and not direct_crop and crop),
            pesticide_name=pesticide_name,
            disease_terms=disease_terms,
            issue_mode=issue_mode,
            generic_issue=generic_issue,
            symptom_label=(
                str(symptom_entry.get("label_hi") or symptom_entry.get("label_en") or "").strip()
                if symptom_entry is not None
                else None
            ) or None,
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
            "gehu",
            "gehun",
            "gehoo",
            "sugarcane",
            "गन्ना",
            "गन्ने",
            "ganna",
            "ganne",
            "potato",
            "आलू",
            "mustard",
            "सरसों",
            "sarso",
            "धान",
            "paddy",
            "sinchai",
            "sichai",
            "irrigation",
            "pani",
            "paani",
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
        phrase_mapping = [
            (r"\b(?:labhdayak|laabhdayak)\b", "लाभदायक"),
            (r"\b(?:jyada|zyada)\b", "ज्यादा"),
            (r"\bhow to (?:grow|cultivate)\b", "कैसे उगाएं"),
            (r"\bpatt?iyo?n?\s+(?:ka|k)\s+rang\s+badal(?:\s*r[hae]+\s*hai)?\b", "पत्तियों का रंग बदलना"),
            (r"\bpattion?\s+(?:ka|k)\s+rang\s+badal(?:\s*r[hae]+\s*hai)?\b", "पत्तियों का रंग बदलना"),
            (r"\bpatt?iyo?n?\s+pe\s+peelapan(?:\s+aa\s+r[hae]+\s*hai)?\b", "पत्तियों का रंग बदलना"),
            (r"\bsafed\s+parat(?:\s+aa\s+r[hae]+\s*hai)?\b", "सफेद परत"),
            (r"\bsafed\s+powder(?:\s+aa\s+r[hae]+\s*hai)?\b", "सफेद परत"),
            (r"\bwhite\s+(?:layer|powder)(?:\s+aa\s+r[hae]+\s*hai)?\b", "सफेद परत"),
            (r"\bpatti\s+mu[dn](?:\s*r[hae]+\s*hai)?\b", "पत्ती मुड़ना"),
            (r"\bpatti\s+mur(?:\s*r[hae]+\s*hai)?\b", "पत्ती मुड़ना"),
            (r"\bpatte\s+mur\s+r[hae]+\s*(?:hain|hai)?\b", "पत्ती मुड़ना"),
            (r"\bras\s+choos(?:\s+r[hae]+\s*hai)?\b", "रस चूसना"),
            (r"\bkeeda\s+dikh(?:\s+r[hae]+\s*hai)?\b", "कीड़ा दिखना"),
            (r"\bkeede\s+dikh(?:\s+r[hae]+\s*hai)?\b", "कीड़े दिखना"),
            (r"\bkida\s+dikh(?:\s+r[hae]+\s*hai)?\b", "कीड़ा दिखना"),
            (r"\bdhab+e?\s+aa\s+r[hae]+\s*(?:hain|hai)?\b", "धब्बे"),
        ]
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
            r"\bjankari\b": "जानकारी",
            r"\bjaankari\b": "जानकारी",
            r"\bkaise\b": "कैसे",
            r"\bkese\b": "कैसे",
            r"\bkatai\b": "कटाई",
            r"\bkatayi\b": "कटाई",
            r"\bkatayee\b": "कटाई",
            r"\bsinchai\b": "सिंचाई",
            r"\bsichai\b": "सिंचाई",
            r"\bsinchaai\b": "सिंचाई",
            r"\bkhad\b": "खाद",
            r"\bkhaad\b": "खाद",
            r"\burvarak\b": "उर्वरक",
            r"\bkism\b": "किस्म",
            r"\bkisam\b": "किस्म",
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
            r"\brog\b": "रोग",
            r"\bbimari\b": "बीमारी",
            r"\blakshan\b": "लक्षण",
            r"\bsymptom\b": "लक्षण",
            r"\bpatt?iyo?n?\b": "पत्तियों",
            r"\bpatto?n?\b": "पत्तों",
            r"\bpatti\b": "पत्ती",
            r"\brang badal(?:\s*r[hae]+\s*hai)?\b": "रंग बदलना",
            r"\bcolor change\b": "रंग बदलना",
            r"\bcolour change\b": "रंग बदलना",
            r"\bpeelapan\b": "पीला पड़ना",
            r"\bdhab+e?\b": "धब्बे",
            r"\bdaag\b": "धब्बे",
            r"\bsafed parat\b": "सफेद परत",
            r"\bsafed powder\b": "सफेद परत",
            r"\bwhite layer\b": "सफेद परत",
            r"\bwhite powder\b": "सफेद परत",
            r"\bsadan\b": "सड़न",
            r"\bsadn\b": "सड़न",
            r"\bjhulsa\b": "झुलसा",
            r"\bsukhna\b": "सूखना",
            r"\bsukh r[hae]+\b": "सूखना",
            r"\bsookh\b": "सूखना",
            r"\bmurjha(?:na)?\b": "मुरझाना",
            r"\bmurja(?:na)?\b": "मुरझाना",
            r"\bpeela\b": "पीला",
            r"\bpeeli\b": "पीली",
            r"\byellowing\b": "पीला पड़ना",
            r"\bras choos(?:na)?\b": "रस चूसना",
            r"\bpat+t[iy]?\s*mu[d]?na\b": "पत्ती मुड़ना",
            r"\bpatte\b": "पत्ते",
            r"\bcurl(?:ing)?\b": "पत्ती मुड़ना",
            r"\bkeeda dikh r[hae]+\b": "कीड़ा दिखना",
            r"\bkeede dikh r[hae]+\b": "कीड़े दिखना",
            r"\bkida dikh r[hae]+\b": "कीड़ा दिखना",
            r"\bछेद\b": "छेद",
            r"\bhole(s)?\b": "छेद",
            r"\bjad nuksan\b": "जड़ नुकसान",
            r"\bleaf blight\b": "पत्ती झुलसा",
            r"\bloose smut\b": "ढीला कंडुआ",
            r"\bdowny mildew\b": "डाउनी मिल्ड्यू",
            r"\bpowdery mildew\b": "चूर्णी फफूंदी",
            r"\bwhite rust\b": "सफेद रतुआ",
        }
        out = text
        for pattern, replacement in phrase_mapping:
            out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
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
        if re.search(r"\bweath[a-z]*\b", t):
            return True
        weather_words = [
            "weather",
            "weathe",
            "wether",
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

    def _is_tomorrow_weather_query(self, text: str) -> bool:
        t = text.strip().lower()
        day_words = ["kal", "कल", "tomorrow", "agle din", "अगले दिन"]
        rain_words = ["बारिश", "बारिस", "barish", "baarish", "rain", "rainfall", "mausam", "weather", "मौसम"]
        return any(d in t for d in day_words) and any(r in t for r in rain_words)

    def _is_rain_day_forecast_query(self, text: str) -> bool:
        t = (text or "").strip().lower()
        if not t:
            return False
        rain_words = ["बारिश", "बारिस", "barish", "baarish", "rain", "rainfall"]
        timing_words = ["किस दिन", "कौनसे दिन", "कौन से दिन", "konse din", "kaunse din", "kis din", "kab", "which day", "when"]
        return any(r in t for r in rain_words) and any(w in t for w in timing_words)

    def _is_rain_specific_query(self, text: str) -> bool:
        t = (text or "").strip().lower()
        if not t:
            return False
        rain_words = ["बारिश", "बारिस", "barish", "baarish", "rain", "rainfall", "showers", "वर्षा"]
        return any(token in t for token in rain_words)

    def _is_weekly_weather_query(self, text: str) -> bool:
        t = (text or "").strip().lower()
        if not t:
            return False
        if not self._is_weather_intent(t):
            return False
        weekly_words = [
            "agle saptah",
            "agla saptah",
            "next week",
            "coming week",
            "अगले सप्ताह",
            "अगला सप्ताह",
            "अगले हफ्ते",
            "अगला हफ्ता",
            "agle hafte",
            "agla hafta",
            "next 7 days",
            "अगले 7 दिन",
            "7 din",
        ]
        return any(w in t for w in weekly_words)

    def _extract_weather_forecast_target(self, text: str) -> dict | None:
        t = (text or "").strip().lower()
        if not t:
            return None
        if not self._is_weather_intent(t):
            return None
        month_map = {
            "jan": 1, "january": 1, "जनवरी": 1,
            "feb": 2, "february": 2, "फरवरी": 2, "फ़रवरी": 2,
            "mar": 3, "march": 3, "मार्च": 3,
            "apr": 4, "april": 4, "अप्रैल": 4,
            "may": 5, "मई": 5,
            "jun": 6, "june": 6, "जून": 6,
            "jul": 7, "july": 7, "जुलाई": 7,
            "aug": 8, "august": 8, "अगस्त": 8,
            "sep": 9, "sept": 9, "september": 9, "सितंबर": 9, "सितम्बर": 9,
            "oct": 10, "october": 10, "अक्टूबर": 10,
            "nov": 11, "november": 11, "नवंबर": 11, "नवम्बर": 11,
            "dec": 12, "december": 12, "दिसंबर": 12, "दिसम्बर": 12,
        }
        now = datetime.now(ZoneInfo("Asia/Kolkata"))

        numeric_date = re.search(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b", t)
        if numeric_date:
            try:
                day = int(numeric_date.group(1))
                month = int(numeric_date.group(2))
                year_raw = numeric_date.group(3)
                year = int(year_raw) if year_raw else now.year
                if year < 100:
                    year += 2000
                target = datetime(year, month, day).date()
                if not year_raw and target < now.date():
                    target = datetime(now.year + 1, month, day).date()
                offset = (target - now.date()).days
                if offset < 0:
                    return None
                return {"day_offset": offset, "label": target.strftime("%d-%m-%Y")}
            except Exception:
                pass

        month_names = "|".join(sorted((re.escape(k) for k in month_map.keys()), key=len, reverse=True))
        text_date = re.search(rf"\b(\d{{1,2}})\s+({month_names})(?:\s+(\d{{4}}))?\b", t)
        if text_date:
            try:
                day = int(text_date.group(1))
                month_token = text_date.group(2)
                month = month_map[month_token]
                year = int(text_date.group(3)) if text_date.group(3) else now.year
                target = datetime(year, month, day).date()
                if not text_date.group(3) and target < now.date():
                    target = datetime(now.year + 1, month, day).date()
                offset = (target - now.date()).days
                if offset < 0:
                    return None
                label = f"{day:02d}-{month:02d}-{target.year}"
                return {"day_offset": offset, "label": label}
            except Exception:
                pass

        if any(x in t for x in ["परसों", "parso", "parsō", "day after tomorrow"]):
            return {"day_offset": 2, "label": "परसों"}
        if any(x in t for x in ["कल", "kal", "tomorrow", "agle din", "अगले दिन"]):
            return {"day_offset": 1, "label": "कल"}
        m = re.search(r"(\d+)\s*(?:दिन|din)\s*(?:baad|बाद)", t)
        if m:
            days = max(0, min(int(m.group(1)), 14))
            return {"day_offset": days, "label": f"{days} दिन बाद"}
        weekday_map = {
            "monday": 0, "सोमवार": 0,
            "tuesday": 1, "मंगलवार": 1,
            "wednesday": 2, "बुधवार": 2,
            "thursday": 3, "गुरुवार": 3, "brihaspativar": 3,
            "friday": 4, "शुक्रवार": 4,
            "saturday": 5, "शनिवार": 5,
            "sunday": 6, "रविवार": 6,
        }
        for label, target_wd in weekday_map.items():
            if label in t:
                delta = (target_wd - now.weekday()) % 7
                if delta == 0 and "next" in t:
                    delta = 7
                elif delta == 0:
                    delta = 7
                pretty = label.title() if re.fullmatch(r"[a-z]+", label) else label
                return {"day_offset": delta, "label": pretty}
        return None

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
        q_lower = q.lower()
        has_weather_word = bool(re.search(r"\bweath[a-z]*\b", q_lower))
        # Fast path: strip common weather words and stopwords, keep remaining tokens as location.
        if (
            "mausam" in q.lower()
            or "मौसम" in q
            or has_weather_word
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
                "hogi",
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
                "weathe",
                "wether",
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
                "agle",
                "agla",
                "hafta",
                "hafte",
                "saptah",
                "week",
                "coming",
                "next",
                "rahega",
                "rahegi",
                "rahenge",
                "रहेगा",
                "रहेगी",
                "अगले",
                "अगला",
                "सप्ताह",
                "हफ्ता",
                "हफ्ते",
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
                "konse",
                "kaunse",
                "kis",
                "din",
                "kab",
                "दिन",
                "किस",
                "कौनसे",
                "कौन",
                "कब",
            }
            tokens = [t.strip(" ?!.,") for t in re.split(r"\s+", q) if t.strip()]
            kept = [t for t in tokens if t.strip(" ?!.,").lower() not in drop]
            if kept:
                return " ".join(kept)
        if (
            "mausam" in q_lower
            or "मौसम" in question
            or has_weather_word
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
                "hogi",
                "kesa",
                "kaisa",
                "hai",
                "h",
                "agle",
                "agla",
                "hafta",
                "hafte",
                "saptah",
                "week",
                "coming",
                "next",
                "rahega",
                "rahegi",
                "rahenge",
                "weather",
                "weathe",
                "wether",
                "barish",
                "baarish",
                "rain",
                "konse",
                "kaunse",
                "kis",
                "din",
                "kab",
            }
            for tok in tokens:
                t = tok.lower()
                if t in stop or t in {"mausam", "maussam", "mosam", "mausm", "मौसम", "weather", "बारिश", "बारिस", "क्या", "सकती", "सकता", "अभी", "है", "दिन", "किस", "कौनसे", "कौन", "कब"}:
                    continue
                return tok
        # Try explicit location phrases first
        patterns = [
            r"(?:weather in|mausam in|maussam in|mosam in)\s+([a-zA-Z\\s]+)",
            r"(?:weathe in|wether in)\s+([a-zA-Z\\s]+)",
            r"([a-zA-Z\\s]+?)\\s+(?:me|mein|में)\\s+(?:baarish|barish|rain)",
            r"([a-zA-Z\\s]+?)\\s+(?:me|mein|में)\\s+बारिश",
            r"([\\u0900-\\u097F\\s]+?)\\s+में\\s+बारिश",
            r"([a-zA-Z\\s]+?)\\s+(?:ka|ki|ke)\\s+weather",
            r"([a-zA-Z\\s]+?)\\s+(?:ka|ki|ke)\\s+(?:weathe|wether)",
            r"([a-zA-Z\\s]+?)\\s+(?:ka|ki|ke)\\s+(?:mausam|maussam|mosam|mausm|मौसम)",
            r"(?:aaj|aj)?\\s*(?:ka\\s+)?weather\\s+([a-zA-Z\\s]+?)\\s+(?:me|mein|में)",
            r"(?:aaj|aj)?\\s*(?:ka\\s+)?(?:weathe|wether)\\s+([a-zA-Z\\s]+?)\\s+(?:me|mein|में)",
            r"(?:aaj|aj)?\\s*(?:ka\\s+)?(?:mausam|maussam|mosam|mausm|मौसम)\\s+([a-zA-Z\\s]+?)\\s+(?:me|mein|में)",
            r"([a-zA-Z\\s]+?)\\s+(?:me|mein|में)\\s+(?:ka\\s+)?weather",
            r"([a-zA-Z\\s]+?)\\s+(?:me|mein|में)\\s+(?:ka\\s+)?(?:weathe|wether)",
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
            "barish",
            "baarish",
            "rain",
            "rainfall",
            "बारिश",
            "बारिस",
            "hoga",
            "hogi",
            "होगा",
            "होगी",
            "rahega",
            "rahegi",
            "rahenge",
            "agle",
            "agla",
            "hafta",
            "hafte",
            "saptah",
            "week",
            "coming",
            "next",
            "रहेगा",
            "रहेगी",
            "अगले",
            "अगला",
            "सप्ताह",
            "हफ्ता",
            "हफ्ते",
            "konse",
            "kaunse",
            "kis",
            "din",
            "kab",
            "दिन",
            "किस",
            "कौनसे",
            "कौन",
            "कब",
        }
        for idx, tok in enumerate(tokens):
            t = tok.strip(" ?!.," ).lower()
            if t in {"mausam", "maussam", "mosam", "mausm", "मौसम", "weather", "weathe", "wether"} and idx + 1 < len(tokens):
                cand = tokens[idx + 1].strip(" ?!.,")
                if cand and cand.lower() not in stop:
                    return cand
        # Fallback: token before 'mausam/मौसम'
        for idx, tok in enumerate(tokens):
            t = tok.strip(" ?!.," ).lower()
            if t in {"mausam", "maussam", "mosam", "mausm", "मौसम", "weather", "weathe", "wether"} and idx - 1 >= 0:
                cand = tokens[idx - 1].strip(" ?!.,")
                if cand and cand.lower() not in stop:
                    return cand
            # Handle "X ka mausam" -> pick token before ka/ki/ke
            if t in {"ka", "ki", "ke"} and idx + 1 < len(tokens):
                nxt = tokens[idx + 1].strip(" ?!.,").lower()
                if nxt in {"mausam", "maussam", "mosam", "mausm", "मौसम", "weather", "weathe", "wether"} and idx - 1 >= 0:
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
            "barish",
            "baarish",
            "rain",
            "rainfall",
            "बारिश",
            "बारिस",
            "hoga",
            "hogi",
            "होगा",
            "होगी",
            "konse",
            "kaunse",
            "kis",
            "din",
            "kab",
            "दिन",
            "किस",
            "कौनसे",
            "कौन",
            "कब",
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

    def _extract_area_acres(self, text: str) -> float | None:
        if not text:
            return None
        patterns = [
            (r"(\d+(?:\.\d+)?)\s*(?:acre|acres|acr|एकड़)\b", 1.0),
            (r"(\d+(?:\.\d+)?)\s*(?:hectare|hectares|ha|हेक्टेयर)\b", 2.47105),
            (r"(\d+(?:\.\d+)?)\s*(?:bigha|बीघा)\b", 0.625),
        ]
        lower_text = text.lower()
        for pattern, factor in patterns:
            m = re.search(pattern, lower_text, flags=re.IGNORECASE)
            if not m:
                continue
            try:
                return float(m.group(1)) * factor
            except ValueError:
                return None
        return None

    def _format_area_acres(self, area_acres: float | None) -> str:
        if not area_acres:
            return "1 acre"
        if abs(area_acres - round(area_acres)) < 1e-6:
            return f"{int(round(area_acres))} acre"
        return f"{area_acres:.1f} acre"

    def _extract_query_season(self, text: str) -> str | None:
        t = (text or "").strip().lower()
        if not t:
            return None
        season_aliases = {
            "Rabi": ["rabi", "रबी"],
            "Kharif": ["kharif", "खरीफ"],
            "Zaid": ["zaid", "जायद"],
            "Annual": ["annual", "वार्षिक", "सालाना"],
        }
        for season, aliases in season_aliases.items():
            if any(alias in t for alias in aliases):
                return season
        return None

    def _has_profitability_terms(self, text: str) -> bool:
        t = (text or "").strip().lower()
        if not t:
            return False
        profit_terms = [
            "profit",
            "profitable",
            "laabh",
            "labh",
            "laabhdayak",
            "labhdayak",
            "लाभ",
            "फायदे",
            "munafa",
            "मुनाफा",
            "better return",
            "best return",
            "कमाई",
            "income",
            "earnings",
            "बेहतर",
            "behtar",
            "jyada",
            "zyada",
        ]
        return any(term in t for term in profit_terms)

    def _has_crop_method_terms(self, text: str) -> bool:
        t = text.strip().lower()
        method_markers = [
            "vidhi",
            "विधि",
            "method",
            "planting method",
            "रोपाई",
            "रोपण",
            "ugane ki",
            "ugane ki konsi",
            "ugane ki kaun si",
            "kheti ki vidhi",
            "kaunsi vidhi",
            "konsi vidhi",
            "kaun si vidhi",
            "trench",
            "ssi",
            "furrow",
        ]
        cultivation_markers = [
            "ugane",
            "ugaye",
            "ugaane",
            "grow",
            "cultivate",
            "kheti",
            "खेती",
        ]
        return any(marker in t for marker in method_markers) or (
            any(marker in t for marker in cultivation_markers) and any(marker in t for marker in ("vidhi", "method", "रोपाई", "रोपण"))
        )

    def _has_crop_guide_terms(self, text: str) -> bool:
        t = text.strip().lower()
        if not t:
            return False
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
        return any(k in t for k in keys) or any(w in t for w in guide_context_words)

    def _has_crop_list_terms(self, text: str) -> bool:
        t = text.strip().lower()
        crop_list_markers = [
            "fasal",
            "फसल",
            "crop",
            "ugaye",
            "उगाएं",
            "बताये",
            "बताएं",
            "btaye",
            "batao",
            "list",
            "kaun si",
            "कौन सी",
        ]
        return any(marker in t for marker in crop_list_markers)

    def _looks_like_crop_method_lexically(self, text: str, context_part: str = "") -> bool:
        t = text.strip().lower()
        if not t:
            return False
        crop = self._extract_crop_from_query(t) or self._extract_preferred_crop_from_context(context_part)
        return bool(crop and self._has_crop_method_terms(t))

    def _looks_like_crop_choice_lexically(self, text: str) -> bool:
        t = text.strip().lower()
        if not t:
            return False
        if self._looks_like_crop_method_lexically(t):
            return False
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
            "faayde ki",
            "fayde ki",
            "faaydemand",
            "fayemand",
            "badhiya fasal",
            "acchi fasal",
            "achhi fasal",
            "best fasal",
        ]
        return any(k in t for k in keys) or (self._has_profitability_terms(t) and self._has_crop_list_terms(t))

    def _looks_like_season_crop_list_lexically(self, text: str) -> bool:
        t = text.strip().lower()
        if not t:
            return False
        if not self._extract_query_season(t):
            return False
        if self._looks_like_crop_choice_lexically(t):
            return False
        if self._has_profitability_terms(t) or any(
            marker in t for marker in ["budget", "cost", "comparison", "compare"]
        ):
            return False
        return self._has_crop_list_terms(t)

    def _looks_like_crop_guide_lexically(self, text: str) -> bool:
        t = text.strip().lower()
        if not t:
            return False
        crop_markers: list[str] = []
        for crop_name, aliases in CROP_ALIASES.items():
            crop_markers.append(str(crop_name).lower())
            for alias in aliases:
                marker = str(alias).strip().lower()
                if marker and marker not in crop_markers:
                    crop_markers.append(marker)
        return self._has_crop_guide_terms(t) and any(c in t for c in crop_markers)

    def _ensure_agri_intent_semantic_index(self) -> None:
        if self._intent_phrase_embeddings is not None:
            return
        self._ensure_rag_components(load_generator=False)
        if self.embedder is None:
            self._intent_phrase_rows = []
            self._intent_phrase_embeddings = None
            return
        rows = [
            {
                "label": "crop_choice_profitability",
                "subject": "crop_choice",
                "scope": "across_crops",
                "objective": "profitability",
                "phrase": phrase,
            }
            for phrase in [
                "कौन सी फसल सबसे ज्यादा लाभदायक है",
                "which crop is most profitable",
                "best crop for profit",
                "rabi ki konsi fasal jyada laabhdayak hai",
                "kharif me profitable crop",
            ]
        ]
        rows.extend(
            {
                "label": "season_crop_list",
                "subject": "season_crop_list",
                "scope": "season",
                "objective": "list",
                "phrase": phrase,
            }
            for phrase in [
                "kharif ki fasal btaye",
                "रबी मौसम की फसलें बताएं",
                "zaid season crops list",
                "खरीफ की फसल बताओ",
            ]
        )
        rows.extend(
            {
                "label": "crop_method_cultivation",
                "subject": "crop_method",
                "scope": "within_crop",
                "objective": "cultivation",
                "phrase": phrase,
            }
            for phrase in [
                "गन्ने की कौन सी विधि बेहतर है",
                "which planting method is best in sugarcane",
                "गेहूं उगाने की विधि",
                "crop planting method within same crop",
            ]
        )
        rows.extend(
            {
                "label": "crop_method_profitability",
                "subject": "crop_method",
                "scope": "within_crop",
                "objective": "profitability",
                "phrase": phrase,
            }
            for phrase in [
                "गन्ने में कौन सी विधि ज्यादा लाभदायक है",
                "which sugarcane method gives higher profit",
                "planting method with better yield in same crop",
                "ganne ugane ki konsi vidhi jyada profitable hai",
            ]
        )
        rows.extend(
            {
                "label": "crop_guide",
                "subject": "crop_guide",
                "scope": "within_crop",
                "objective": "cultivation",
                "phrase": phrase,
            }
            for phrase in [
                "गेहूं की खेती कैसे करें",
                "how to grow moong",
                "गन्ना की खेती guide",
                "crop production guide for wheat",
            ]
        )
        phrase_texts = [row["phrase"] for row in rows]
        self._intent_phrase_rows = rows
        self._intent_phrase_embeddings = np.asarray(self.embedder.encode(phrase_texts), dtype=np.float32)

    def _parse_agri_intent(self, text: str, context_part: str = "") -> ParsedAgriIntent:
        normalized = self._normalize_hinglish(text or "").strip()
        context_norm = (context_part or "").strip().lower()
        cache_key = f"{normalized}||{context_norm}"
        cached = self._intent_parse_cache.get(cache_key)
        if cached is not None:
            return cached

        crop = (
            self._extract_crop_from_query(text or "")
            or self._extract_crop_from_query(normalized)
            or self._extract_preferred_crop_from_context(context_part)
        )
        season = self._extract_query_season(text or "") or self._extract_query_season(normalized) or self._extract_season(context_part)
        has_crop = bool(crop)
        has_season = bool(self._extract_query_season(text or "") or self._extract_query_season(normalized))
        has_profit = self._has_profitability_terms(text) or self._has_profitability_terms(normalized)
        has_method = self._has_crop_method_terms(text) or self._has_crop_method_terms(normalized)
        has_guide = self._has_crop_guide_terms(text) or self._has_crop_guide_terms(normalized)
        has_list = self._has_crop_list_terms(text) or self._has_crop_list_terms(normalized)

        parsed = ParsedAgriIntent(crop=crop, season=season)
        if has_crop and has_method:
            parsed.subject = "crop_method"
            parsed.scope = "within_crop"
            parsed.objective = "profitability" if has_profit else "cultivation"
            parsed.confidence = 0.76
        elif has_season and has_list and not has_profit:
            parsed.subject = "season_crop_list"
            parsed.scope = "season"
            parsed.objective = "list"
            parsed.confidence = 0.74
        elif has_profit and (not has_crop or not has_method or has_season):
            parsed.subject = "crop_choice"
            parsed.scope = "across_crops"
            parsed.objective = "profitability"
            parsed.confidence = 0.72
        elif has_crop and has_guide:
            parsed.subject = "crop_guide"
            parsed.scope = "within_crop"
            parsed.objective = "cultivation"
            parsed.confidence = 0.68

        # Clear cultivation and crop-profit comparisons use structured data;
        # model downloads/inference must not precede these deterministic routes.
        clear_profit_comparison = (
            parsed.subject == "crop_choice" and has_list and
            bool(re.search(r"profit|laabh|labh|munafa|लाभ|मुनाफा|कमाई", normalized, re.I))
        )
        if parsed.subject == "crop_guide" or clear_profit_comparison:
            self._intent_parse_cache[cache_key] = parsed
            return parsed

        self._ensure_agri_intent_semantic_index()
        if self.embedder is not None and self._intent_phrase_embeddings is not None and self._intent_phrase_rows:
            try:
                query_vec = np.asarray(self.embedder.encode([normalized or (text or "").strip()])[0], dtype=np.float32)
                scores = self._intent_phrase_embeddings @ query_vec
                adjusted_scores: list[float] = []
                for idx, row in enumerate(self._intent_phrase_rows):
                    score = float(scores[idx])
                    subject = row["subject"]
                    objective = row["objective"]
                    if subject == "crop_method":
                        if has_crop:
                            score += 0.05
                        if has_method:
                            score += 0.08
                        if has_profit and objective == "profitability":
                            score += 0.06
                        if not has_crop:
                            score -= 0.10
                    elif subject == "crop_choice":
                        if has_profit:
                            score += 0.08
                        if has_method:
                            score -= 0.14
                        if has_season:
                            score += 0.04
                    elif subject == "season_crop_list":
                        if has_season:
                            score += 0.10
                        if has_list:
                            score += 0.05
                        if has_profit:
                            score -= 0.14
                    elif subject == "crop_guide":
                        if has_crop:
                            score += 0.05
                        if has_guide:
                            score += 0.08
                        if has_profit:
                            score -= 0.04
                    adjusted_scores.append(score)
                best_idx = int(np.argmax(adjusted_scores))
                best_score = float(adjusted_scores[best_idx])
                best = self._intent_phrase_rows[best_idx]
                semantic = ParsedAgriIntent(
                    subject=best["subject"],
                    scope=best["scope"],
                    objective=best["objective"],
                    crop=crop,
                    season=season,
                    confidence=best_score,
                    matched_label=best["label"],
                    matched_phrase=best["phrase"],
                )
                if best_score >= 0.48 and best_score >= (parsed.confidence - 0.02):
                    if not (
                        parsed.subject == "crop_method"
                        and semantic.subject == "crop_choice"
                        and has_method
                    ):
                        if not (semantic.subject == "season_crop_list" and not has_season):
                            parsed = semantic
            except Exception:
                pass

        if len(self._intent_parse_cache) >= 128:
            oldest_key = next(iter(self._intent_parse_cache), None)
            if oldest_key is not None:
                self._intent_parse_cache.pop(oldest_key, None)
        self._intent_parse_cache[cache_key] = parsed
        return parsed

    def _is_crop_method_intent(self, text: str) -> bool:
        parsed = self._parse_agri_intent(text)
        return parsed.subject == "crop_method" or self._looks_like_crop_method_lexically(text)

    def _is_crop_choice_intent(self, text: str) -> bool:
        parsed = self._parse_agri_intent(text)
        if parsed.subject == "crop_method":
            return False
        return parsed.subject == "crop_choice" or self._looks_like_crop_choice_lexically(text)

    def _is_season_crop_list_intent(self, text: str) -> bool:
        parsed = self._parse_agri_intent(text)
        return parsed.subject == "season_crop_list" or self._looks_like_season_crop_list_lexically(text)

    def _is_crop_guide_intent(self, text: str) -> bool:
        parsed = self._parse_agri_intent(text)
        return parsed.subject == "crop_guide" or self._looks_like_crop_guide_lexically(text)

    def _structured_crop_recommendation(
        self,
        context_part: str,
        question: str,
        district_override: str | None = None,
    ) -> tuple[str | None, list[str]]:
        if not self.cfg.db_path:
            return None, []

        district = district_override or self._extract_district(context_part) or "Meerut"
        season = self._extract_season(context_part) or self._extract_query_season(question)
        budget = self._extract_budget(question)
        area_acres = self._extract_area_acres(question)
        area_scale = area_acres or 1.0
        sources: list[str] = []
        market_prices = self._load_agmarknet_prices(district)
        if market_prices:
            sources.append("agmarknet_report.csv")

        conn = sqlite3.connect(self.cfg.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = []
            if season:
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
            baseline_answer = self._rank_from_profit_baselines(district, season or "", question, market_prices)
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
            rev = price * float(r["avg_yield_qtl_per_acre"]) * area_scale
            base_cost_min = float(r["cost_min_inr_per_acre"])
            base_cost_max = float(r["cost_max_inr_per_acre"])
            total_cost_min = base_cost_min * area_scale
            total_cost_max = base_cost_max * area_scale
            pmin = rev - total_cost_max
            pmax = rev - total_cost_min
            if budget is not None and total_cost_min > budget:
                continue
            scored.append(
                {
                    "crop": crop,
                    "cost_min": total_cost_min,
                    "cost_max": total_cost_max,
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
                f"{district} में आपके बजट के अंदर कोई स्पष्ट फसल विकल्प नहीं मिला। "
                "कृपया बजट बढ़ाएँ या फसल विकल्प बताकर फिर पूछें।"
            ), sources

        scored = sorted(scored, key=lambda x: x["profit_min"], reverse=True)[:3]
        lines = []
        lines.append("मानदंड: मंडी MSP/भाव (Agmarknet) + उपलब्ध लागत/उपज डेटा + PPQS/MUP रोग/कीट दबाव संकेत।")
        lines.append("कीमत स्रोत: Agmarknet (district market prices)")
        scope_label = f"{self._format_area_acres(area_acres)} के लिए" if area_acres else "प्रति एकड़"
        for i, s in enumerate(scored, start=1):
            lines.append(
                f"{i}) {s['crop']}: {scope_label} लागत ₹{int(s['cost_min'])}-₹{int(s['cost_max'])}, "
                f"अनुमानित आय ₹{int(s['revenue'])}, "
                f"संभावित लाभ ₹{int(s['profit_min'])}-₹{int(s['profit_max'])}"
                f"{s['price_note']} | रोग/कीट दबाव संकेत: {s['pressure']}"
            )

        if budget is not None:
            budget_line = f"कुल बजट: ₹{int(budget)}" if area_acres else f"बजट: ₹{int(budget)} प्रति एकड़"
        else:
            budget_line = "बजट: उपलब्ध नहीं"
        header_parts = [f"जिला: {district}"]
        if area_acres:
            header_parts.append(f"क्षेत्र: {self._format_area_acres(area_acres)}")
        header_parts.append(budget_line)
        return (
            f"समझा गया सवाल (हिंदी): {question}\n\n"
            + " | ".join(header_parts) + "\n"
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
        area_acres = self._extract_area_acres(question)
        area_scale = area_acres or 1.0
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
            revenue = price * yield_qtl_per_acre * area_scale
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
            total_cost_min = cost_min * area_scale
            total_cost_max = cost_max * area_scale
            if budget is not None and total_cost_min > budget:
                continue
            pest_pressure = self._estimate_pesticide_pressure(crop)
            water_penalty = self._water_penalty(base["water_need"])
            pest_penalty = {"कम": 0.98, "मध्यम": 1.0, "उच्च": 1.06, "अज्ञात": 1.03}.get(pest_pressure, 1.03)
            adjusted_cost_max = total_cost_max * water_penalty * pest_penalty
            profit_min = revenue - adjusted_cost_max
            profit_max = revenue - total_cost_min
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
                    "cost_min": total_cost_min,
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
        header_parts = [f"जिला: {district}"]
        if area_acres:
            header_parts.append(f"क्षेत्र: {self._format_area_acres(area_acres)}")
        if budget is not None:
            header_parts.append(f"कुल बजट: ₹{int(budget)}" if area_acres else f"बजट: ₹{int(budget)} प्रति एकड़")
        header = " | ".join(header_parts)
        lines = [
            header,
            "लाभ रैंकिंग उपज × भाव − लागत के आधार पर निकाली गई है।",
            "लागत में जहाँ उपलब्ध हो वहाँ CACP के अनुसार paid-out cost + family labour से लेकर पूरी लागत तक का band लिया गया है; नहीं मिलने पर baseline indicative range रखा गया है।",
            "",
            "सबसे बेहतर विकल्प:",
        ]
        scope_label = f"{self._format_area_acres(area_acres)} के लिए" if area_acres else "प्रति एकड़"
        for i, s in enumerate(scored, start=1):
            date_part = f", {s['price_date']}" if s["price_date"] else ""
            crop_label = self._crop_display_label(str(s["crop"]))
            lines.append(
                f"{i}) {crop_label}: उपज ~{s['yield']:.1f} qtl/acre"
                f"{(' (UPAG ' + str(s['yield_year']) + (', 2nd AE' if s.get('yield_note') else '') + ')') if s['yield_source']=='UPAG' and s['yield_year'] else ''}, "
                f"भाव ₹{s['price']:.0f}/qtl ({s['price_source']}{date_part}), "
                f"{scope_label} आय ~₹{int(s['revenue'])}, लागत ~₹{int(s['cost_min'])}-₹{int(s['cost_max'])} ({s['cost_source']}, {s['cost_basis']}), "
                f"लाभ ~₹{int(s['profit_min'])}-₹{int(s['profit_max'])}; पानी: {s['water_need']}, रोग/कीट दबाव: {s['pest_pressure']}"
            )
        heavy_risk_crops = [
            self._crop_display_label(str(s["crop"])).split(" (")[0]
            for s in scored
            if str(s["water_need"]) == "बहुत अधिक" or str(s["season"]).lower() == "annual"
        ]
        if heavy_risk_crops:
            crop_names = ", ".join(dict.fromkeys(heavy_risk_crops))
            lines.extend(
                [
                    "",
                    f"ध्यान दें: {crop_names} जैसी लंबी अवधि या ज्यादा पानी वाली फसलों में revenue अच्छा दिख सकता है, लेकिन पानी, मजदूरी और cash-cycle का जोखिम भी ज्यादा रहता है।",
                ]
            )
        lines.extend(
            [
                "",
                "Exotic crops:",
                *[f"- {note}" for note in EXOTIC_CROP_NOTES],
                "",
                "नोट: यह planning estimate है। सटीक farm-profit के लिए आपकी जमीन, पानी, मजदूरी दर, बीज variety और खरीदी/मंडी linkage चाहिए।",
            ]
        )
        return f"समझा गया सवाल (हिंदी): {question}\n\n" + "\n".join(lines)

    def _answer_season_crop_list_query(
        self,
        context_part: str,
        question: str,
    ) -> tuple[str | None, list[str]]:
        season = self._extract_season(context_part)
        if not season:
            q = (question or "").lower()
            if "kharif" in q or "खरीफ" in q:
                season = "Kharif"
            elif "rabi" in q or "रबी" in q:
                season = "Rabi"
            elif "zaid" in q or "जायद" in q:
                season = "Zaid"
            elif "annual" in q or "वार्षिक" in q or "सालाना" in q:
                season = "Annual"
        if not season:
            return None, []

        district = self._extract_district(context_part) or "Meerut"
        items: list[str] = []
        sources = ["UPAG yield data", f"CACP {season.title()} cost report" if season.lower() != "annual" else "CACP Sugarcane / baseline crop data"]

        for crop, base in WESTERN_UP_CROP_BASELINES.items():
            if not self._season_matches(season, str(base["season"])):
                continue
            yield_info = load_latest_up_yield_qtl_per_acre(crop, season=season)
            yield_qtl = (
                float(yield_info["yield_qtl_per_acre"])
                if yield_info and yield_info.get("yield_qtl_per_acre")
                else float(base["yield_qtl_per_acre"])
            )
            cost_info = self._official_cost_range_for_profitability(crop, yield_qtl)
            crop_label = self._crop_display_label(crop)
            if cost_info:
                item = (
                    f"{crop_label}: उपज ~{yield_qtl:.1f} qtl/acre "
                    f"(UPAG {yield_info.get('crop_year', '')}, 2nd AE), "
                    f"लागत ~₹{int(float(cost_info['cost_min']))}-₹{int(float(cost_info['cost_max']))}/acre "
                    f"({cost_info['source']}), पानी: {base['water_need']}"
                )
            else:
                item = (
                    f"{crop_label}: उपज ~{yield_qtl:.1f} qtl/acre, "
                    f"indicative लागत ~₹{int(float(base['cost_min']))}-₹{int(float(base['cost_max']))}/acre, "
                    f"पानी: {base['water_need']}"
                )
            items.append(item)

        if not items:
            return None, sources

        header = [
            f"जिला: {district}",
            f"मौसम: {season}",
            f"{season} मौसम की सामान्य फसलें:",
        ]
        answer = (
            f"समझा गया सवाल (हिंदी): {question}\n\n"
            + " | ".join(header[:2])
            + "\n"
            + header[2]
            + "\n"
            + self._format_numbered_text_blocks(items)
            + "\n\nअगर आप चाहें, तो मैं इन्हीं फसलों में से कौन सी ज्यादा लाभदायक, कम लागत वाली, या कम पानी वाली है यह भी compare कर सकता हूँ।"
        )
        return answer, sources

    def _season_matches(self, selected: str, crop_season: str) -> bool:
        s = (selected or "").lower()
        c = crop_season.lower()
        if not s or s in {"all", "any"}:
            return True
        if s == "annual":
            return "annual" in c
        if s in c:
            return True
        if s in {"rabi", "kharif", "zaid"}:
            return False
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
            try:
                self.embedder = Embedder(self.cfg.embedding_model)
            except Exception:
                self.embedder = None
                self.retriever = None
                if load_generator:
                    self.generator = None
                return
        if self.retriever is None:
            try:
                store = NumpyVectorStore(self.cfg.index_path, self.cfg.metadata_path)
                vectors, metadata = store.load()
                self.retriever = Retriever(vectors=vectors, metadata=metadata)
            except Exception:
                self.retriever = None
                if load_generator:
                    self.generator = None
                return
        if load_generator and self.generator is None:
            try:
                self.generator = LocalGenerator(self.cfg.generator_model)
            except Exception:
                self.generator = None
