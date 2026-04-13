from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass
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
import pandas as pd

from app.location_lookup import lookup_place, lookup_place_in_text


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
            return {"answer": self._time_based_greeting(), "references": [], "retrieved": []}

        normalized_question = self._normalize_hinglish(farmer_question)
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
                }
            return {"answer": weather, "references": ["Open-Meteo API"], "retrieved": []}
        if self._looks_like_location_only(farmer_question):
            return {
                "answer": "कृपया बताएं कि आप मौसम पूछ रहे हैं या भाव/कीमत?",
                "references": [],
                "retrieved": [],
            }
        if self._is_crop_choice_intent(normalized_question):
            place = self._extract_location_from_question(farmer_question)
            loc = lookup_place_in_text(farmer_question) or lookup_place_in_text(normalized_question)
            if not loc and place:
                loc = lookup_place(place)
            if loc and not place:
                place = loc.get("place")
            district_override = loc.get("district") if loc else None
            if place and not district_override:
                return {
                    "answer": f"स्थान '{place}' का जिला नहीं मिला। कृपया स्थान/जिला स्पष्ट करें।",
                    "references": [],
                    "retrieved": [],
                }
            structured, sources = self._structured_crop_recommendation(
                context_part,
                normalized_question,
                district_override=district_override,
            )
            if structured:
                retrieved = []
                if self.embedder is None or self.retriever is None or self.generator is None:
                    self._ensure_rag_components()
                if self.embedder is not None and self.retriever is not None:
                    qvec = self.embedder.encode([normalized_question])[0]
                    retrieved = self.retriever.retrieve(qvec, k=min(3, self.top_k))
                    if retrieved:
                        snippets = []
                        for r in retrieved[:3]:
                            text = str(r.get("text") or "").strip()
                            if text:
                                snippets.append(f"- {text[:180]}".rstrip() + ("..." if len(text) > 180 else ""))
                        if snippets:
                            structured += "\n\nसंदर्भ संकेत:\n" + "\n".join(snippets)
                refs = list(sources)
                refs.extend([r.get("source_file") for r in retrieved if r.get("source_file")])
                return {
                    "answer": structured,
                    "references": refs,
                    "retrieved": retrieved,
                }
        if self._is_pesticide_intent(normalized_question):
            return self._structured_pesticide_advice(normalized_question)

        normalized_query = (
            f"{context_part} किसान का प्रश्न: {normalized_question}".strip()
            if context_part
            else normalized_question
        )

        self._ensure_rag_components()
        if self.embedder is None or self.retriever is None or self.generator is None:
            return {
                "answer": "मॉडल अभी उपलब्ध नहीं है। कृपया थोड़ी देर बाद फिर प्रयास करें।",
                "references": [],
                "retrieved": [],
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
        }

    def _is_pesticide_intent(self, text: str) -> bool:
        t = text.lower()
        keys = [
            "pesticide",
            "insecticide",
            "fungicide",
            "herbicide",
            "कीटनाशक",
            "फफूंदनाशी",
            "घासनाशी",
            "दवा",
            "स्प्रे",
            "छिड़काव",
            "कीट",
            "रोग",
        ]
        return any(k in t for k in keys)

    def _structured_pesticide_advice(self, question: str) -> dict:
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
        text = "\n".join(snippets)
        pesticides = self._extract_pesticide_names(text)
        doses = self._extract_dose_lines(text)
        waiting = self._extract_waiting_period(text)

        lines = []
        lines.append("संरचित कीटनाशक सलाह:")
        if pesticides:
            lines.append(f"- सुझाई गई दवाएँ: {', '.join(pesticides[:5])}")
        if doses:
            lines.append(f"- खुराक/डोज़: {doses[0]}")
        if waiting:
            lines.append(f"- सुरक्षा अवधि (PHI): {waiting}")
        lines.append("- छिड़काव से पहले लेबल निर्देश और राज्य सलाह देखें।")

        return {
            "answer": "\n".join(lines),
            "references": [r.get("source_file") for r in retrieved],
            "retrieved": retrieved,
        }

    def _extract_pesticide_names(self, text: str) -> list[str]:
        # Simple keyword-based extraction
        names = set()
        patterns = [
            r"(?i)\\b(Chlorpyrifos|Imidacloprid|Mancozeb|Carbendazim|Metalaxyl|Copper oxychloride|Azoxystrobin|Propiconazole|Thiamethoxam|Lambda-cyhalothrin)\\b",
            r"(?i)\\b(मैनकोज़ेब|कार्बेन्डाज़िम|कॉपर ऑक्सीक्लोराइड|इमिडाक्लोप्रिड|थायमेथोक्साम)\\b",
        ]
        for pat in patterns:
            for m in re.findall(pat, text):
                names.add(m)
        return sorted(names)

    def _extract_dose_lines(self, text: str) -> list[str]:
        lines = []
        for line in text.splitlines():
            if re.search(r"(ml|g|gm|gram|लीटर|ली\\.|l/ha|kg/ha|g/l)", line, flags=re.IGNORECASE):
                lines.append(line.strip())
        return lines

    def _extract_waiting_period(self, text: str) -> str | None:
        m = re.search(r"(?:PHI|प्री-हार्वेस्ट|सुरक्षा अवधि)[^\\d]*(\\d+\\s*(?:दिन|days))", text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return None

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
            r"\baloo\b": "आलू",
            r"\bsarso\b": "सरसों",
            r"\bwhat crop should i grow\b": "मुझे कौन सी फसल उगानी चाहिए",
            r"\bgrow\b": "उगानी",
            r"\bmausam\b": "मौसम",
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
            "rain",
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
        if "mausam" in q.lower() or "मौसम" in q or "weather" in q.lower():
            drop = {
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
        if "mausam" in q_lower or "मौसम" in question or "weather" in q_lower:
            tokens = [t.strip(" ?!.,") for t in re.split(r"\s+", question) if t.strip()]
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
                "weather",
            }
            for tok in tokens:
                t = tok.lower()
                if t in stop or t in {"mausam", "maussam", "mosam", "mausm", "मौसम", "weather"}:
                    continue
                return tok
        # Try explicit location phrases first
        patterns = [
            r"(?:weather in|mausam in|maussam in|mosam in)\s+([a-zA-Z\\s]+)",
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
            "best crop",
            "which crop",
            "crop to grow",
            "फसल उगानी",
            "profitable",
            "profit",
        ]
        return any(k in t for k in keys)

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
        sources: list[str] = ["crop_economics (SQLite)"]
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
            return None, sources

        scored: list[dict] = []
        for r in rows:
            price = float(r["market_price_inr_per_qtl"])
            price_note = ""
            crop = r["crop_name"]
            if crop in market_prices:
                price_info = market_prices[crop]
                price = float(price_info["price"])
                price_note = f" (भाव: {price:.0f} Rs./Quintal, {price_info['date']})"
            rev = price * float(r["avg_yield_qtl_per_acre"])
            pmin = rev - float(r["cost_max_inr_per_acre"])
            pmax = rev - float(r["cost_min_inr_per_acre"])
            pressure_label, pressure_penalty, _ = self._estimate_pesticide_pressure(crop)
            if pressure_penalty > 0:
                pmin = pmin * (1 - pressure_penalty)
                pmax = pmax * (1 - pressure_penalty)
            if budget is not None and float(r["cost_max_inr_per_acre"]) > budget:
                continue
            scored.append(
                {
                    "crop": crop,
                    "cost_min": float(r["cost_min_inr_per_acre"]),
                    "cost_max": float(r["cost_max_inr_per_acre"]),
                    "revenue": rev,
                    "profit_min": pmin,
                    "profit_max": pmax,
                    "price_note": price_note,
                    "pressure": pressure_label,
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
        for i, s in enumerate(scored, start=1):
            lines.append(
                f"{i}) {s['crop']}: लागत ₹{int(s['cost_min'])}-₹{int(s['cost_max'])}/एकड़, "
                f"अनुमानित आय ₹{int(s['revenue'])}/एकड़, "
                f"संभावित लाभ ₹{int(s['profit_min'])}-₹{int(s['profit_max'])}/एकड़"
                f"{s['price_note']} | कीटनाशक दबाव: {s['pressure']}"
            )

        budget_line = f"बजट: ₹{int(budget)} प्रति एकड़" if budget is not None else "बजट: उपलब्ध नहीं"
        return (
            f"समझा गया सवाल (हिंदी): {question}\n\n"
            f"जिला: {district} | मौसम: {season} | {budget_line}\n"
            "उपलब्ध अर्थशास्त्रीय डेटा के आधार पर सर्वोत्तम फसल विकल्प:\n"
            + "\n".join(lines)
            + "\n\nनोट: अंतिम निर्णय से पहले स्थानीय मंडी भाव, पानी उपलब्धता और मिट्टी की स्थिति जरूर देखें।"
        ), sources

    def _estimate_pesticide_pressure(self, crop: str) -> tuple[str, float, list[str]]:
        if not crop:
            return "अज्ञात", 0.0, []
        self._ensure_rag_components()
        if self.embedder is None or self.retriever is None:
            return "अज्ञात", 0.0, []
        query = f"{crop} कीटनाशक छिड़काव मात्रा लागत"
        qvec = self.embedder.encode([query])[0]
        retrieved = self.retriever.retrieve(qvec, k=max(3, self.top_k))
        text = " ".join([str(r.get("text") or "") for r in retrieved]).lower()
        if not text.strip():
            return "अज्ञात", 0.0, []
        keywords = [
            "spray",
            "dose",
            "ml",
            "g/ha",
            "g/acre",
            "l/ha",
            "छिड़काव",
            "कीटनाशक",
            "fungicide",
            "insecticide",
            "spinosad",
            "emamectin",
        ]
        hits = sum(text.count(k) for k in keywords)
        if hits >= 18:
            return "उच्च", 0.12, [r.get("source_file") for r in retrieved if r.get("source_file")]
        if hits >= 8:
            return "मध्यम", 0.06, [r.get("source_file") for r in retrieved if r.get("source_file")]
        return "कम", 0.0, [r.get("source_file") for r in retrieved if r.get("source_file")]

    def _load_agmarknet_prices(self, district: str) -> dict[str, dict[str, str | float]]:
        path = Path("data/raw/live/agmarknet_report.csv")
        if not path.exists():
            return {}
        try:
            df = pd.read_csv(path)
        except Exception:
            return {}
        required = {"District", "Commodity", "Arrival_Date", "Modal_Price"}
        if not required.issubset(df.columns):
            return {}
        df = df[df["District"].astype(str).str.lower() == district.lower()].copy()
        if df.empty:
            return {}
        df["Arrival_Date_dt"] = pd.to_datetime(df["Arrival_Date"], errors="coerce", dayfirst=True)
        df = df.dropna(subset=["Arrival_Date_dt", "Modal_Price"])
        if df.empty:
            return {}
        latest = df.sort_values("Arrival_Date_dt").groupby("Commodity", as_index=False).tail(1)
        out: dict[str, dict[str, str | float]] = {}
        for _, row in latest.iterrows():
            try:
                price = float(row["Modal_Price"])
            except Exception:
                continue
            date = row["Arrival_Date_dt"].date().isoformat()
            out[str(row["Commodity"])] = {
                "price": price,
                "date": date,
                "unit": row.get("Price_Unit", "Rs./Quintal"),
            }
        return out

    def _ensure_rag_components(self) -> None:
        if self.embedder is None:
            self.embedder = Embedder(self.cfg.embedding_model)
        if self.retriever is None:
            store = NumpyVectorStore(self.cfg.index_path, self.cfg.metadata_path)
            vectors, metadata = store.load()
            self.retriever = Retriever(vectors=vectors, metadata=metadata)
        if self.generator is None:
            self.generator = LocalGenerator(self.cfg.generator_model)
