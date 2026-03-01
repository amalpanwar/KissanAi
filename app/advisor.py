from __future__ import annotations

from dataclasses import dataclass

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
        qvec = self.embedder.encode([user_query])[0]
        retrieved = self.retriever.retrieve(qvec, k=self.top_k)
        prompt = build_prompt(user_query, retrieved)
        try:
            response = self.generator.generate(prompt)
        except Exception:
            response = self._fallback_answer(retrieved)
        return {
            "answer": response,
            "references": [r.get("source_file") for r in retrieved],
            "retrieved": retrieved,
        }

    def _fallback_answer(self, retrieved: list[dict]) -> str:
        if not retrieved:
            return (
                "मॉडल से उत्तर नहीं मिल पाया। कृपया सवाल में जिला, मौसम, बजट और फसल विकल्प जोड़कर फिर से पूछें।"
            )
        snippets = [f"- {r.get('text', '')[:140]}" for r in retrieved[:3]]
        return (
            "मॉडल अस्थायी रूप से धीमा है, इसलिए संदर्भ आधारित त्वरित सलाह दी जा रही है:\n\n"
            "1) जिले और मौसम के हिसाब से मध्यम जोखिम वाली फसल चुनें।\n"
            "2) बजट को बीज, उर्वरक, सिंचाई और रोग प्रबंधन में बांटें।\n"
            "3) बाजार कीमत और भंडारण जोखिम देखकर अंतिम निर्णय लें।\n\n"
            "संदर्भ अंश:\n"
            + "\n".join(snippets)
        )
