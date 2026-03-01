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


@dataclass
class AdvisorConfig:
    embedding_model: str
    generator_model: str
    index_path: str
    metadata_path: str
    top_k: int


class RAGAdvisor:
    def __init__(self, cfg: AdvisorConfig) -> None:
        self.embedder = Embedder(cfg.embedding_model)
        store = NumpyVectorStore(cfg.index_path, cfg.metadata_path)
        vectors, metadata = store.load()
        self.retriever = Retriever(vectors=vectors, metadata=metadata)
        self.generator = LocalGenerator(cfg.generator_model)
        self.top_k = cfg.top_k

    def answer(self, user_query: str) -> dict:
        if self._is_greeting(user_query) and not self._has_agri_intent(user_query):
            return {"answer": self._time_based_greeting(), "references": [], "retrieved": []}

        normalized_query = self._normalize_hinglish(user_query)
        qvec = self.embedder.encode([normalized_query])[0]
        retrieved = self.retriever.retrieve(qvec, k=self.top_k)
        prompt = build_prompt(normalized_query, retrieved)
        try:
            response = self.generator.generate(prompt)
            if self._is_low_quality_response(response):
                response = self._fallback_answer(retrieved, normalized_query)
        except Exception:
            response = self._fallback_answer(retrieved, normalized_query)
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
