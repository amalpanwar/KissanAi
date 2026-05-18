from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pdf_extract import read_pdf_text


OUTPUT_PATH = Path("data/processed/agri_glossary.json")

TERM_SPECS = {
    "das": {
        "label": "DAS",
        "aliases": ["das", "days after sowing"],
        "meaning": "Days After Sowing",
        "answer_hi": "DAS का मतलब Days After Sowing होता है, यानी बुवाई के कितने दिन बाद। उदाहरण: 15-20 DAS का मतलब बुवाई के 15 से 20 दिन बाद।",
        "patterns": ["days after sowing", "15-20 das", "35-40 das", "70-75 das"],
        "sources": ["data/raw/Crop Production guide.pdf"],
    },
    "fym": {
        "label": "FYM",
        "aliases": ["fym", "farm yard manure", "farmyard manure"],
        "meaning": "Farm Yard Manure",
        "answer_hi": "FYM का मतलब Farm Yard Manure होता है, यानी गोबर की सड़ी हुई खाद। इसे खेत की उर्वरता और मिट्टी की हालत सुधारने के लिए डाला जाता है।",
        "patterns": ["farm yard manure", "farmyard manure", "fym"],
        "sources": ["data/raw/Crop Production guide.pdf"],
    },
    "npk": {
        "label": "NPK",
        "aliases": ["npk", "n:p:k", "nitrogen phosphorus potash"],
        "meaning": "Nitrogen, Phosphorus and Potash",
        "answer_hi": "NPK का मतलब Nitrogen, Phosphorus और Potash होता है। यह मुख्य पोषक तत्वों का अनुपात बताता है, जैसे 80:40:40 NPK।",
        "patterns": ["80:40:40 npk", "n:p:k", "npk fertilizer", "17-17-17 npk"],
        "sources": ["data/raw/Crop Production guide.pdf", "data/raw/official_sources/cacp_rabi_latest_en.pdf"],
    },
    "phi": {
        "label": "PHI",
        "aliases": ["phi", "pre-harvest interval", "waiting period"],
        "meaning": "Pre-Harvest Interval",
        "answer_hi": "PHI का मतलब Pre-Harvest Interval होता है, यानी दवा के आखिरी छिड़काव और फसल की कटाई के बीच जरूरी इंतजार अवधि।",
        "fallback_snippet": "MUP tables में इसे Waiting period/PHI between last application and harvest (days) के रूप में दिया गया है।",
        "patterns": [
            "Waiting period/ PHIbetween last application &harvest (days)",
            "Waiting period/ PHI",
            "pre-harvest interval",
            "phi",
        ],
        "sources": [
            "data/raw/all_sources/4._herbicides_mup_as_on_30.09.2025.pdf",
            "data/raw/all_sources/2._chemical_mup_as_on_30.09.2025.pdf",
        ],
    },
    "msp": {
        "label": "MSP",
        "aliases": ["msp", "minimum support price"],
        "meaning": "Minimum Support Price",
        "answer_hi": "MSP का मतलब Minimum Support Price होता है, यानी सरकार द्वारा तय न्यूनतम खरीद मूल्य, ताकि किसान को एक आधार भाव मिल सके।",
        "patterns": ["minimum support price", "minimum support prices", "msp"],
        "sources": [
            "data/raw/official_sources/cacp_rabi_latest_en.pdf",
            "data/raw/official_sources/cacp_kharif_latest_en.pdf",
        ],
    },
    "frp": {
        "label": "FRP",
        "aliases": ["frp", "fair and remunerative price"],
        "meaning": "Fair and Remunerative Price",
        "answer_hi": "FRP का मतलब Fair and Remunerative Price होता है। यह गन्ने के लिए सरकार द्वारा तय न्यूनतम भुगतान मूल्य होता है।",
        "patterns": ["fair and remunerative price", "frp"],
        "sources": ["data/raw/official_sources/cacp_sugarcane_latest.pdf"],
    },
}


def _read_pdf_text(path: Path) -> str:
    return read_pdf_text(path)


def _snippet_for_pattern(text: str, pattern: str, window: int = 180) -> str | None:
    match = re.search(re.escape(pattern), text, flags=re.IGNORECASE)
    if not match:
        return None
    start = max(0, match.start() - window)
    end = min(len(text), match.end() + window)
    left_boundary = max(text.rfind(".", start, match.start()), text.rfind("•", start, match.start()), text.rfind("\n", start, match.start()))
    right_candidates = [text.find(".", match.end(), end), text.find("•", match.end(), end), text.find("\n", match.end(), end)]
    right_candidates = [idx for idx in right_candidates if idx != -1]
    if left_boundary != -1:
        start = left_boundary + 1
    if right_candidates:
        end = min(right_candidates) + 1
    snippet = text[start:end].replace("\n", " ")
    snippet = re.sub(r"\s+", " ", snippet).strip()
    return snippet[:420].strip()


def build_glossary() -> dict[str, object]:
    doc_cache: dict[str, str] = {}
    terms: dict[str, dict[str, object]] = {}
    for key, spec in TERM_SPECS.items():
        snippets: list[dict[str, str]] = []
        for source_str in spec["sources"]:
            path = Path(source_str)
            if not path.exists():
                continue
            text = doc_cache.setdefault(source_str, _read_pdf_text(path))
            for pattern in spec["patterns"]:
                snippet = _snippet_for_pattern(text, pattern)
                if snippet and len(snippet) >= 20:
                    snippets.append({"source": source_str, "text": snippet})
                    break
        if not snippets and spec.get("fallback_snippet"):
            snippets.append({"source": spec["sources"][0], "text": str(spec["fallback_snippet"])})
        terms[key] = {
            "label": spec["label"],
            "aliases": spec["aliases"],
            "meaning": spec["meaning"],
            "answer_hi": spec["answer_hi"],
            "snippets": snippets,
        }
    return {
        "metadata": {
            "generated_from": "official PDFs and crop production guide",
            "extraction_strategy": "docling-preferred shared PDF extractor with safe fallback",
            "term_count": len(terms),
        },
        "terms": terms,
    }


def main() -> None:
    payload = build_glossary()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote glossary to {OUTPUT_PATH} with {payload['metadata']['term_count']} terms")


if __name__ == "__main__":
    main()
