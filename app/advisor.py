from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass
import re
from zoneinfo import ZoneInfo

from app.embeddings import Embedder
from app.generator import LocalGenerator
from app.prompting import build_prompt
from app.retriever import Retriever
from app.vector_store import NumpyVectorStore
from app.weather import get_current_weather_hindi


@dataclass
class AdvisorConfig:
    embedding_model: str
    generator_model: str
    index_path: str
    metadata_path: str
    top_k: int


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
            district = self._extract_district(context_part) or "Meerut"
            weather = get_current_weather_hindi(district)
            return {"answer": weather, "references": ["Open-Meteo API"], "retrieved": []}

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
        greetings = ["hello", "hi", "hey", "namaste", "नमस्ते", "राम राम", "ram ram"]
        return any(g in t for g in greetings)

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

    def _ensure_rag_components(self) -> None:
        if self.embedder is None:
            self.embedder = Embedder(self.cfg.embedding_model)
        if self.retriever is None:
            store = NumpyVectorStore(self.cfg.index_path, self.cfg.metadata_path)
            vectors, metadata = store.load()
            self.retriever = Retriever(vectors=vectors, metadata=metadata)
        if self.generator is None:
            self.generator = LocalGenerator(self.cfg.generator_model)
