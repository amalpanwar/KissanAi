from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from difflib import get_close_matches
from functools import lru_cache
from pathlib import Path
from typing import Any
from app.config import load_config
from app.embeddings import Embedder
from app.generator import LocalGenerator
from app.pdf_extract import read_pdf_pages, read_pdf_text


GUIDE_PDF = Path("data/raw/Crop Production guide.pdf")
ALIAS_JSON = Path("data/raw/commodity_aliases.json")
GUIDE_REVIEW_QUEUE = Path("data/processed/guide_review_queue.jsonl")
PDF_OFFSET = 12  # printed page 1 starts at PDF page 13
DEFAULT_GUIDE_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_GUIDE_GENERATOR_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

MANUAL_PAGE_MAP = {
    "Rice": 1,
    "Millets": 65,
    "Sorghum": 65,
    "Cumbu": 81,
    "Ragi": 92,
    "Maize": 103,
    "Small Millets": 117,
    "Wheat": 122,
    "Pulses": 125,
    "Redgram": 125,
    "Blackgram": 134,
    "Greengram": 145,
    "Cowpea": 154,
    "Horsegram": 159,
    "Bengalgram": 162,
    "Garden Lab lab (Avarai)": 166,
    "Field Lab lab (Mochai)": 171,
    "Soya bean": 174,
    "Sword bean": 179,
    "Oilseeds": 181,
    "Groundnut": 181,
    "Sesame": 198,
    "Castor": 206,
    "Sunflower": 216,
    "Safflower": 224,
    "Coconut": 227,
    "Oilpalm": 239,
    "Niger": 249,
    "FIBRE CROPS": 253,
    "Cotton": 253,
    "Jute": 285,
    "Sugarcane": 286,
    "Sweet Sorghum": 309,
    "Tropical Sugarbeet": 311,
    "Forage Crops": 316,
    "Fodder Cholam": 316,
    "Fodder Maize": 318,
    "Neelakolukattai": 321,
    "Guinea grass": 323,
    "Deenanath grass": 325,
    "Cumbu Napier Hybrids": 327,
    "Lucerne": 329,
    "Hedge Lucerne": 331,
    "Fodder Cowpea": 333,
    "Muyalmasal": 335,
    "Leucaena": 337,
    "Green Manure Crops": 339,
    "Daincha": 339,
    "Sunnhemp": 340,
    "Mushroom Cultivation": 341,
    "SERICULTURE": 363,
    "AGRO FORESTRY": 385,
}

CROP_ALIASES = {
    "Rice": ["rice", "धान", "paddy"],
    "Millets": ["millets", "मिलेट्स", "mota anaj", "coarse cereals"],
    "Wheat": ["wheat", "गेहूं", "gehun", "gehu"],
    "Maize": ["maize", "corn", "मक्का", "makka"],
    "Sugarcane": ["sugarcane", "गन्ना", "गन्ने", "ganne", "ganna"],
    "Groundnut": ["groundnut", "peanut", "मूंगफली", "mungfali"],
    "Sesame": ["sesame", "til", "तिल"],
    "Cotton": ["cotton", "कपास", "kapas"],
    "Sorghum": ["sorghum", "jowar", "ज्वार"],
    "Sweet Sorghum": ["sweet sorghum", "मीठा ज्वार"],
    "Ragi": ["ragi", "finger millet", "रागी", "mandua"],
    "Cumbu": ["cumbu", "pearl millet", "बाजरा", "bajra"],
    "Small Millets": ["small millets", "small millet", "लघु मिलेट्स"],
    "Pulses": ["pulses", "दलहन"],
    "Redgram": ["redgram", "red gram", "अरहर", "arhar", "pigeon pea", "tur", "tuar"],
    "Blackgram": ["blackgram", "black gram", "उड़द", "urad"],
    "Greengram": ["greengram", "green gram", "मूंग", "moong"],
    "Cowpea": ["cowpea", "लोबिया", "lobia"],
    "Horsegram": ["horsegram", "horse gram", "कुल्थी", "kulthi"],
    "Bengalgram": ["bengalgram", "bengal gram", "चना", "chana", "chickpea"],
    "Garden Lab lab (Avarai)": ["garden lab lab", "avarai", "garden lablab"],
    "Field Lab lab (Mochai)": ["field lab lab", "mochai", "field lablab"],
    "Soya bean": ["soya bean", "soybean", "सोयाबीन", "soyabean"],
    "Sword bean": ["sword bean"],
    "Oilseeds": ["oilseeds", "oil seeds", "तिलहन"],
    "Castor": ["castor", "अरंडी", "arandi"],
    "Sunflower": ["sunflower", "सूरजमुखी", "surajmukhi", "surujmukhi", "soorajmukhi", "suryamukhi"],
    "Safflower": ["safflower"],
    "Coconut": ["coconut", "नारियल"],
    "Oilpalm": ["oilpalm", "oil palm"],
    "Niger": ["niger", "ramtil"],
    "FIBRE CROPS": ["fibre crops", "fiber crops", "रेशेदार फसलें"],
    "Jute": ["jute", "जूट"],
    "Tropical Sugarbeet": ["tropical sugarbeet", "sugarbeet", "sugar beet"],
    "Forage Crops": ["forage crops", "fodder crops", "चारा फसलें"],
    "Fodder Cholam": ["fodder cholam"],
    "Fodder Maize": ["fodder maize", "चारा मक्का"],
    "Neelakolukattai": ["neelakolukattai"],
    "Guinea grass": ["guinea grass"],
    "Deenanath grass": ["deenanath grass"],
    "Cumbu Napier Hybrids": ["cumbu napier hybrids", "napier grass", "नेपियर घास"],
    "Lucerne": ["lucerne", "alfalfa", "कुडिराइमसल", "kudiraimasal"],
    "Hedge Lucerne": ["hedge lucerne", "velimasal"],
    "Fodder Cowpea": ["fodder cowpea"],
    "Muyalmasal": ["muyalmasal"],
    "Leucaena": ["leucaena", "soundal"],
    "Green Manure Crops": ["green manure crops", "हरी खाद फसलें"],
    "Daincha": ["daincha", "dhaincha", "ढैंचा"],
    "Sunnhemp": ["sunnhemp", "sunhemp", "सनहेम्प"],
    "Mushroom Cultivation": ["mushroom cultivation", "mushroom farming", "mushroom", "मशरूम"],
    "SERICULTURE": ["sericulture", "रेशम पालन"],
    "AGRO FORESTRY": ["agro forestry", "agroforestry", "वानिकी"],
}

HEADING_PATTERNS = {
    "before_sowing": [
        "CLIMATE REQUIREMENT",
        "SEASON AND VARIET",
        "DISTRICT/SEASON VARIETIES",
        "VARIETY",
        "FIELD PREPARATION",
        "APPLICATION OF FYM",
        "SEED TREATMENT",
        "NURSERY",
        "LAND PREPARATION",
    ],
    "sowing": [
        "SEED RATE",
        "SPACING",
        "FORMING BEDS",
        "FORMING RIDGES",
        "SOWING",
        "PLANTING",
        "TRANSPLANTING",
    ],
    "early_growth": [
        "WEED MANAGEMENT",
        "WATER MANAGEMENT",
        "IRRIGATION",
        "GAP FILLING",
        "THINNING",
    ],
    "mid_growth": [
        "APPLICATION OF FERTILIZERS",
        "APPLICATION OF MICRONUTRIENTS",
        "TOP DRESSING",
        "FERTIGATION",
        "EARTHING UP",
        "FOLIAR SPRAY",
        "SULPHUR",
        "BORIC ACID",
        "IMPROVING SEED SET",
        "CROP PROTECTION",
        "PLANT PROTECTION",
        "INTERCROPPING",
    ],
    "harvest": [
        "JUDGE WHEN TO HARVEST",
        "HARVESTING",
        "HARVESTING STAGE",
        "HARVEST",
        "THRESHING",
        "POST HARVEST",
    ],
}

PHASE_LABELS_HI = {
    "before_sowing": "बुवाई से पहले",
    "sowing": "बुवाई/रोपाई के समय",
    "early_growth": "शुरुआती बढ़वार",
    "mid_growth": "मध्य बढ़वार से फसल देखभाल",
    "harvest": "कटाई और बाद की तैयारी",
}

CROP_NAME_HI = {
    "Rice": "धान",
    "Wheat": "गेहूं",
    "Maize": "मक्का",
    "Sugarcane": "गन्ना",
    "Groundnut": "मूंगफली",
    "Sesame": "तिल",
    "Cotton": "कपास",
    "Sorghum": "ज्वार",
    "Sweet Sorghum": "मीठा ज्वार",
    "Ragi": "रागी",
    "Cumbu": "बाजरा",
    "Redgram": "अरहर",
    "Blackgram": "उड़द",
    "Greengram": "मूंग",
    "Cowpea": "लोबिया",
    "Horsegram": "कुल्थी",
    "Bengalgram": "चना",
    "Sunflower": "सूरजमुखी",
    "Jute": "जूट",
    "Sunnhemp": "सनहेम्प",
    "Daincha": "ढैंचा",
    "Mushroom Cultivation": "मशरूम",
}


def _pick_hinglish_alias(crop: str, aliases: list[str]) -> str | None:
    crop_norm = _norm(crop)
    for alias in aliases:
        a = str(alias).strip()
        if not a:
            continue
        if re.search(r"[\u0900-\u097f]", a):
            continue
        a_norm = _norm(a)
        if not a_norm or a_norm == crop_norm:
            continue
        if crop_norm in a_norm or a_norm in crop_norm:
            continue
        if re.fullmatch(r"[A-Za-z0-9 ()/\-]+", a):
            return a
    return None


def _crop_display_label(crop: str) -> str:
    hi = CROP_NAME_HI.get(crop, "")
    aliases = []
    shared = _shared_aliases()
    match = _match_shared_alias_key(crop, list(shared.keys()))
    if match:
        aliases = shared.get(match, [])
    hinglish = _pick_hinglish_alias(crop, aliases)
    if hi and hinglish:
        return f"{hi} ({hinglish})"
    if hi:
        return hi
    if hinglish:
        return f"{crop} ({hinglish})"
    return crop

HEADING_LABELS_HI = {
    "CLIMATE REQUIREMENT": "जलवायु",
    "SEASON AND VARIETY": "मौसम और किस्म",
    "SEASON AND VARIETIES": "मौसम और किस्म",
    "DISTRICT/SEASON VARIETIES": "मौसम और किस्म",
    "DISEASE MANAGEMENT IN NURSERY": "नर्सरी में रोग प्रबंधन",
    "FIELD PREPARATION": "खेत की तैयारी",
    "FARM LAND PREPARATION": "खेत की तैयारी",
    "APPLICATION OF FYM OR COMPOST": "गोबर की खाद/कम्पोस्ट",
    "SEED RATE": "बीज दर",
    "SEED TREATMENT": "बीज उपचार",
    "SEED TREATMENT WITH FUNGICIDES": "बीज उपचार",
    "FORMING BEDS AND CHANNEL": "क्यारियां और नालियां",
    "FORMING RIDGES AND FURROWS": "मेड़ और नालियां",
    "APPLICATION OF FERTILIZERS": "उर्वरक प्रबंधन",
    "FERTILIZER APPLICATION": "उर्वरक प्रबंधन",
    "APPLICATION OF MICRONUTRIENTS": "सूक्ष्म पोषक तत्व",
    "FOLIAR SPRAY OF NAPHTHALENE ACETIC ACID": "पत्तियों पर स्प्रे (NAA)",
    "SULPHUR FERTILIZATION": "गंधक प्रबंधन",
    "BORIC ACID": "बोरिक एसिड स्प्रे",
    "IMPROVING SEED SET BY MECHANICAL MEANS": "बीज बनने में सुधार",
    "SOWING": "बुवाई",
    "PLANTING": "रोपाई",
    "TRANSPLANTING": "रोपाई",
    "THINNING": "छंटाई",
    "WEED MANAGEMENT": "खरपतवार प्रबंधन",
    "WATER MANAGEMENT": "सिंचाई प्रबंधन",
    "IRRIGATION": "सिंचाई",
    "JUDGE WHEN TO HARVEST": "कटाई का सही समय",
    "TOP DRESSING": "ऊपरी खाद",
    "BIOFERTILIZER FOR SUGARCANE": "जैव उर्वरक",
    "PRE-HARVEST PRACTICES": "कटाई से पहले की तैयारी",
    "HARVESTING": "कटाई",
    "CROP PROTECTION": "फसल सुरक्षा",
}

TERM_REPLACEMENTS = {
    "cool and dry climate": "ठंडी और शुष्क जलवायु",
    "grown during rabi season": "रबी मौसम में उगाई जाती है",
    "wide adaptability": "कई प्रकार की परिस्थितियों में अच्छी तरह उग सकती है",
    "plough twice": "2 बार जुताई करें",
    "prepare the land to a fine tilth": "खेत को भुरभुरा और समतल तैयार करें",
    "spread": "डालें",
    "incorporate in the soil": "मिट्टी में अच्छी तरह मिला दें",
    "treat the seeds with": "बीज को",
    "24 hours before sowing": "बुवाई से 24 घंटे पहले",
    "draw the lines": "कतारें बनाएं",
    "avoid deep sowing": "बहुत गहरी बुवाई न करें",
    "one hand weeding": "एक बार हाथ से निराई",
    "two hand weedings": "दो बार हाथ से निराई",
    "the crop requires": "फसल को चाहिए",
    "water stagnation should be avoided": "पानी खड़ा नहीं होना चाहिए",
    "immediately after sowing": "बुवाई के तुरंत बाद",
    "germination phase": "अंकुरण अवस्था",
    "crown root intiation": "क्राउन रूट बनने की अवस्था",
    "active tillering stage": "टिलरिंग अवस्था",
    "tillering phase": "टिलरिंग/कल्ले बनने की अवस्था",
    "grand growth phase": "तेज बढ़वार की अवस्था",
    "flowering phase": "फूल आने की अवस्था",
    "flowering stage": "फूल आने की अवस्था",
    "grain filling stage": "दाना भरने की अवस्था",
    "grain formation stage": "दाना बनने की अवस्था",
    "maturity phase": "पकने की अवस्था",
    "pre-flowering phase": "फूल आने से पहले की अवस्था",
    "reproductive phase": "प्रजनन/फलन अवस्था",
    "vegetative phase": "शाकीय बढ़वार अवस्था",
    "pod formation stage": "फली बनने की अवस्था",
    "pod development stage": "फली विकास अवस्था",
    "pegging stage": "पेगिंग अवस्था",
    "apply remaining half of n": "बचा हुआ आधा नाइट्रोजन दें",
    "harvest the crop when": "फसल की कटाई तब करें जब",
    "thresh and winnow the grains": "मड़ाई और सफाई कर लें",
    "use a high volume sprayer and give a thorough coverage of the entire plant": "उच्च मात्रा वाले sprayer से पूरे पौधे पर समान रूप से छिड़काव करें",
    "do not use brackish water": "खारे पानी का उपयोग न करें",
    "varieties": "किस्में",
    "hybrids": "हाइब्रिड",
    "rainfed": "वर्षा आधारित",
    "irrigated": "सिंचित",
    "foliar spray": "पत्तियों पर छिड़काव",
    "micronutrient mixture": "सूक्ष्म पोषक तत्व मिश्रण",
    "biofertilizer treatment": "biofertilizer उपचार",
    "mechanical means": "यांत्रिक तरीके",
    "seed set": "बीज बनना",
    "seed filling": "बीज भराव",
    "bee hives": "मधुमक्खी के बक्से",
    "mid flowering": "मध्य फूल अवस्था",
    "head": "फूल का सिरा",
    "heads": "फूल के सिरों",
    "bracts": "पीछे की पत्तियां",
}

FOLLOWUP_SECTION_KEYWORDS = {
    "variety": ["किस्म", "kism", "kisam", "variety", "varieties", "seed rate", "बीज दर"],
    "fertilizer": ["खाद", "khad", "khaad", "उर्वरक", "urvarak", "fertilizer", "fym", "compost", "गोबर", "micronutrient", "जैव उर्वरक", "top dressing"],
    "irrigation": ["सिंचाई", "sinchai", "sichai", "sinchaai", "पानी", "pani", "paani", "irrigation", "water management", "water", "water requirement", "water need"],
    "crop_protection": ["रोग", "कीट", "disease", "pest", "fungus", "fungal", "फफूंद", "crop protection", "plant protection", "लक्षण"],
    "harvest": ["कटाई", "katai", "katayi", "katayee", "harvest", "harvesting", "maturity", "pre-harvest"],
    "field_preparation": ["खेत की तैयारी", "जुताई", "field preparation", "land preparation", "मेड़", "नालियां"],
    "sowing": ["बुवाई", "रोपाई", "sowing", "planting", "transplanting", "spacing", "seed treatment"],
    "planting_method": ["विधि", "vidhi", "method", "planting method", "planting", "रोपण", "रोपाई की विधि"],
}

MONTH_REPLACEMENTS = {
    "january": "जनवरी",
    "february": "फरवरी",
    "march": "मार्च",
    "april": "अप्रैल",
    "may": "मई",
    "june": "जून",
    "july": "जुलाई",
    "august": "अगस्त",
    "september": "सितंबर",
    "october": "अक्टूबर",
    "november": "नवंबर",
    "december": "दिसंबर",
    "first week of": "के पहले सप्ताह",
    "second week of": "के दूसरे सप्ताह",
}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


@lru_cache(maxsize=1)
def _load_pages() -> list[str]:
    if not GUIDE_PDF.exists():
        return []
    # Live app queries should avoid the heavier Docling path to keep Streamlit
    # responsive. We still use Docling in offline ingestion/build scripts.
    return read_pdf_pages(GUIDE_PDF, prefer_docling=False)


@lru_cache(maxsize=1)
def _toc_map() -> dict[str, int]:
    pages = _load_pages()
    if not pages:
        return {}
    toc_text = "\n".join(pages[2:5])
    mapping: dict[str, int] = {}
    for line in toc_text.splitlines():
        s = " ".join(line.split())
        m = re.match(r"^(?:\d+\s+)?([A-Za-z][A-Za-z \-()]+?)\s+(\d{1,3})$", s)
        if not m:
            continue
        name = m.group(1).strip()
        page = int(m.group(2))
        mapping[name] = page
    return mapping


def _resolve_crop_name(question: str) -> str | None:
    q = question.lower()
    merged_aliases = _merged_crop_aliases()
    for crop, aliases in merged_aliases.items():
        for alias in aliases:
            if alias.lower() in q:
                return crop
    # Fallback: match against TOC section names directly.
    for crop_name in _toc_map().keys():
        if crop_name.lower() in q:
            return crop_name
    return None


@lru_cache(maxsize=1)
def _shared_aliases() -> dict[str, list[str]]:
    if not ALIAS_JSON.exists():
        return {}
    try:
        data = json.loads(ALIAS_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, list[str]] = {}
    if not isinstance(data, dict):
        return out
    for key, vals in data.items():
        if isinstance(vals, list):
            out[str(key)] = [str(v) for v in vals if str(v).strip()]
    return out


def _match_shared_alias_key(crop: str, shared_keys: list[str]) -> str | None:
    target = _norm(crop)
    for key in shared_keys:
        k = _norm(key)
        if k == target or target in k or k in target:
            return key
    return None


@lru_cache(maxsize=1)
def _merged_crop_aliases() -> dict[str, list[str]]:
    shared = _shared_aliases()
    shared_keys = list(shared.keys())
    merged: dict[str, list[str]] = {}
    for crop, aliases in CROP_ALIASES.items():
        vals = set(aliases)
        vals.add(crop)
        match = _match_shared_alias_key(crop, shared_keys)
        if match:
            vals.update(shared.get(match, []))
        merged[crop] = sorted({v for v in vals if v})
    return merged


def _section_page_range(crop: str) -> tuple[int, int] | None:
    toc = {**_toc_map(), **MANUAL_PAGE_MAP}
    if not toc:
        return None
    # match crop against TOC keys, allowing compact names
    target_key = None
    for key in toc:
        if _norm(key) == _norm(crop):
            target_key = key
            break
    if target_key is None:
        for key in toc:
            if _norm(crop) in _norm(key):
                target_key = key
                break
    if target_key is None:
        return None
    start_print = toc[target_key]
    later = sorted([p for k, p in toc.items() if p > start_print])
    next_print = later[0] if later else start_print + 12
    start_pdf = start_print + PDF_OFFSET - 1
    end_pdf = next_print + PDF_OFFSET - 2
    return start_pdf, end_pdf


def _extract_section_text(crop: str) -> str:
    pages = _load_pages()
    page_range = _section_page_range(crop)
    if not pages or page_range is None:
        return ""
    start_pdf, end_pdf = page_range
    texts = []
    for i in range(max(0, start_pdf), min(len(pages), end_pdf + 1)):
        texts.append(pages[i])
    return "\n".join(texts)


def _should_stop_at_heading(crop: str, heading: str) -> bool:
    h = _heading_key(heading)
    crop_norm = _norm(crop)
    if not h:
        return False
    for other in MANUAL_PAGE_MAP:
        other_norm = _norm(other)
        if other_norm == crop_norm:
            continue
        if other_norm and other_norm in _norm(h):
            return True
    stop_markers = [
        "SWEET SORGHUM",
        "TROPICAL SUGARBEET",
        "SAFFLOWER",
        "VARIETAL SEED PRODUCTION",
        "HYBRID SEED PRODUCTION",
        "SHORT CROP",
    ]
    return any(marker in h for marker in stop_markers)


def _should_skip_heading(crop: str, heading: str) -> bool:
    h = _heading_key(heading)
    skip_markers = [
        "MORPHOLOGICAL CHARACTERS",
        "CROP PHYSIOLOGY",
        "RELATION TO MAIN FIELD PLANTING",
        "PRECAUTIONS IN MAINTAINING NURSERY CROP",
        "MANAGEMENT OF THE FIELD AFTER HARVEST OF THE PLANT CROP",
        "MANAGEMENT OF THE CROP",
        "NITROGEN SAVING",
        "IMPROVED TECHNIQUES IN BIOLOGICAL CONTROL",
        "NEMATODE MANAGEMENT",
        "EVALUATION OF FERTILIZER REQUIREMENT",
        "IMPORTANCE OF INM",
        "IMPORTANCE OF BALANCED NUTRITION",
    ]
    return any(marker in h for marker in skip_markers)


def _extract_blocks(section_text: str) -> list[tuple[str, str]]:
    if not section_text:
        return []
    lines = [ln.strip() for ln in section_text.splitlines()]
    blocks: list[tuple[str, list[str]]] = []
    current_heading = ""
    current_lines: list[str] = []
    for line in lines:
        s = " ".join(line.split())
        if not s:
            continue
        is_heading = (
            len(s) < 90
            and (
                re.match(r"^\d+\.?\s+[A-Z][A-Z /()\-]+$", s)
                or re.match(r"^[A-Z][A-Z /()\-]{4,}$", s)
                or re.match(r"^[IVX]+\.\s+[A-Z]", s)
            )
        )
        if is_heading:
            if current_heading and current_lines:
                blocks.append((current_heading, current_lines))
            current_heading = s
            current_lines = []
        else:
            current_lines.append(s)
    if current_heading and current_lines:
        blocks.append((current_heading, current_lines))
    return [(h, " ".join(ls)) for h, ls in blocks]


def _assign_phase(heading: str) -> str | None:
    upper = heading.upper()
    for phase, patterns in HEADING_PATTERNS.items():
        if any(pat in upper for pat in patterns):
            return phase
    # Broader fallback so section-specific headings are not lost.
    if any(k in upper for k in ["SPRAY", "FERTILIZ", "MICRONUTRIENT", "BORIC", "SULPHUR", "SEED SET", "POLLINATION"]):
        return "mid_growth"
    if any(k in upper for k in ["RIDGE", "FURROW", "SEED RATE", "SPACING", "SOWING"]):
        return "sowing"
    if any(k in upper for k in ["CLIMATE", "SEASON", "FIELD PREPARATION", "FYM", "SEED TREATMENT"]):
        return "before_sowing"
    if any(k in upper for k in ["THINNING", "WEED", "IRRIGATION", "WATER"]):
        return "early_growth"
    if any(k in upper for k in ["HARVEST", "POST HARVEST", "THRESH"]):
        return "harvest"
    return None


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\bkg ha-1\b", "kg/ha", text, flags=re.IGNORECASE)
    text = re.sub(r"\bt ha-1\b", "t/ha", text, flags=re.IGNORECASE)
    return text


def _heading_key(heading: str) -> str:
    upper = heading.upper()
    upper = re.sub(r"^(?:[IVX]+\.\s+|[0-9]+[.)]?\s+)", "", upper).strip()
    return upper


def _heading_label_hi(heading: str) -> str:
    upper = _heading_key(heading)
    for key, label in HEADING_LABELS_HI.items():
        if key in upper:
            return label
    return heading.title()


def _crop_type(crop: str) -> str:
    crop_norm = _norm(crop)
    if crop_norm in {_norm(x) for x in ["Wheat", "Rice", "Maize", "Sorghum", "Sweet Sorghum", "Cumbu", "Ragi", "Small Millets", "Millets"]}:
        return "grain"
    if crop_norm in {_norm(x) for x in ["Redgram", "Blackgram", "Greengram", "Cowpea", "Horsegram", "Bengalgram", "Garden Lab lab (Avarai)", "Field Lab lab (Mochai)", "Sword bean", "Soya bean"]}:
        return "pulse"
    if crop_norm in {_norm(x) for x in ["Sunflower", "Sesame", "Safflower", "Castor", "Groundnut", "Coconut", "Oilpalm", "Niger"]}:
        return "oilseed"
    if crop_norm in {_norm(x) for x in ["Sugarcane", "Tropical Sugarbeet"]}:
        return "cane"
    if crop_norm in {_norm(x) for x in ["Cotton", "Jute"]}:
        return "fibre"
    if crop_norm in {_norm(x) for x in ["Mushroom Cultivation"]}:
        return "mushroom"
    return "generic"


def _translate_terms(text: str) -> str:
    out = text
    for src, tgt in TERM_REPLACEMENTS.items():
        out = re.sub(re.escape(src), tgt, out, flags=re.IGNORECASE)
    for src, tgt in MONTH_REPLACEMENTS.items():
        out = re.sub(re.escape(src), tgt, out, flags=re.IGNORECASE)
    out = re.sub(r"\bDAS\b", "दिन बाद बुवाई", out)
    out = re.sub(r"\bt/ha\b", "टन/हेक्टेयर", out, flags=re.IGNORECASE)
    out = re.sub(r"\bkg/ha\b", "किग्रा/हेक्टेयर", out, flags=re.IGNORECASE)
    out = re.sub(r"\bhrs?\b", "घंटे", out, flags=re.IGNORECASE)
    out = re.sub(r"\blitres?\b", "लीटर", out, flags=re.IGNORECASE)
    out = re.sub(r"\bltr\.?\b", "लीटर", out, flags=re.IGNORECASE)
    out = re.sub(r"\blt\b", "लीटर", out, flags=re.IGNORECASE)
    out = re.sub(r"\b30cm\b", "30 सेमी", out, flags=re.IGNORECASE)
    out = re.sub(r"\b45cm\b", "45 सेमी", out, flags=re.IGNORECASE)
    out = re.sub(r"\b60cm\b", "60 सेमी", out, flags=re.IGNORECASE)
    out = re.sub(r"\b10th day\b", "10वें दिन", out, flags=re.IGNORECASE)
    out = re.sub(r"\biii\)\s*Treat the seeds\.?", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\bcm\b", "सेमी", out, flags=re.IGNORECASE)
    out = re.sub(r"\bmm\b", "मिमी", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _compress_sentences(text: str, limit: int = 2) -> str:
    parts = re.split(r"(?:(?<=\.)\s+|(?<=;)\s+)", text)
    parts = [p.strip(" -") for p in parts if len(p.strip()) > 10]
    return " ".join(parts[:limit]).strip()


def _cleanup_bullet_text(text: str) -> str:
    out = _translate_terms(text)
    out = re.sub(r"\b([0-9]+)(st|nd|rd|th)\b", r"\1", out, flags=re.IGNORECASE)
    out = re.sub(r"\.\s*\.", ".", out)
    out = re.sub(r"\s+\.\s*", ". ", out)
    out = re.sub(r"\s+", " ", out).strip(" .;")
    return out


def _final_phrase_cleanup(text: str) -> str:
    out = text
    replacements = [
        (r"\bfoliar spray\b", "पत्तियों पर छिड़काव"),
        (r"\bspray\b", "छिड़काव"),
        (r"\bseed set\b", "बीज बनना"),
        (r"\bseed filling\b", "बीज भराव"),
        (r"\breflective ribbon\b", "चमकीली पट्टी (ribbon)"),
        (r"\bpeak maturity\b", "पूरी परिपक्वता"),
        (r"\bray floret opening stage\b", "फूल खुलने की अवस्था"),
        (r"\bhead\b", "फूल का सिरा"),
        (r"\bbracts\b", "पीछे की पत्तियां"),
        (r"\bbiofertilizer\b", "जैव उर्वरक"),
        (r"\bphosphobacteria\b", "Phosphobacteria"),
        (r"\bhybrid\b", "हाइब्रिड"),
        (r"\bhybrids\b", "हाइब्रिड"),
        (r"\bvarieties\b", "किस्में"),
        (r"\bvariety\b", "किस्म"),
        (r"\brainfed\b", "वर्षा आधारित"),
        (r"\birrigated\b", "सिंचित"),
        (r"\bbasal\b", "बेसल"),
        (r"\bsplit dose\b", "भागों में"),
        (r"\bbud damage\b", "bud को नुकसान"),
        (r"\bmid-season\b", "मध्यम अवधि वाली"),
        (r"\bearly variety\b", "जल्दी पकने वाली किस्म"),
        (r"\bmid-season variety\b", "मध्यम अवधि वाली किस्म"),
    ]
    for pat, repl in replacements:
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)
    out = re.sub(r"\bNAA का पत्तियों पर छिड़काव\b", "NAA का छिड़काव", out)
    out = re.sub(r"\bBoric Acid\b", "Boric Acid", out)
    out = out.replace("पीछे की पीछे की पत्तियां", "पीछे की पत्तियां")
    out = out.replace("flower flower heads/capitulum", "flower heads/capitulum")
    out = out.replace("flower flower heads", "flower heads")
    out = out.replace("heads को", "flower heads को")
    out = re.sub(r"\s+\.\s*", ". ", out)
    out = re.sub(r"\.\s*।", "।", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _extract_nursery_treatments(body: str) -> list[str]:
    text = body.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    treatments: list[str] = []

    chem_pat = re.compile(
        r"(?i)(carbendazim(?:\s*\d+%[A-Z]*)?|thiram|captan|carboxin|tricyclazole|pyroquilon|pseudomonas fluorescens|mancozeb(?:\s*\d+%[A-Z]*)?)"
    )
    dose_pat = re.compile(
        r"(?i)@\s*([0-9.]+(?:\s*[-–]\s*[0-9.]+)?\s*(?:g/l/kg of seeds|g/l of water|g/kg of seed|g/kg|kg/ha|g/l|ml/l|g))"
    )

    chunks = re.split(r"(?i)\b(?:dry seed treatment|wet seed treatment|cib recommendation|seedling dip)\b", text)
    labels = re.findall(r"(?i)\b(dry seed treatment|wet seed treatment|cib recommendation|seedling dip)\b", text)
    if labels and len(chunks) > 1:
        paired = zip(labels, chunks[1:])
    else:
        paired = [("", text)]

    for label, chunk in paired:
        chems = [m.group(1).strip() for m in chem_pat.finditer(chunk)]
        doses = [m.group(1).strip() for m in dose_pat.finditer(chunk)]
        if not chems:
            continue
        chems_clean = []
        for chem in chems:
            if chem not in chems_clean:
                chems_clean.append(chem)
        prefix = ""
        ll = label.lower().strip()
        if ll == "dry seed treatment":
            prefix = "सूखा बीज उपचार"
        elif ll == "wet seed treatment":
            prefix = "गीला बीज उपचार"
        elif ll == "cib recommendation":
            prefix = "CIB सिफारिश"
        elif ll == "seedling dip":
            prefix = "पौध डुबोकर उपचार"
        line = f"{prefix}: " if prefix else ""
        line += f"{', '.join(chems_clean)} का उपयोग करें"
        if doses:
            uniq_doses = []
            for d in doses:
                if d not in uniq_doses:
                    uniq_doses.append(d)
            line += f"। खुराक: {', '.join(d.strip() for d in uniq_doses[:3])}"
        if "24 hours prior" in chunk.lower() or "24 h" in chunk.lower():
            line += "। बीज उपचार बुवाई/भिगोने से पहले करें"
        if "soak the seeds in the solution for 2 hours" in chunk.lower():
            line += "। बीज को घोल में लगभग 2 घंटे भिगोएँ"
        if "seedlings" in chunk.lower() and "soaked for 30 min" in chunk.lower():
            line += "। रोपाई से पहले पौधों को घोल/जैव एजेंट में लगभग 30 मिनट डुबोएँ"
        treatments.append(_cleanup_bullet_text(line))
    return treatments[:4]


def _summarize_chemicals(chems: list[str]) -> str:
    if not chems:
        return ""
    uniq = []
    for chem in chems:
        c = chem.strip()
        if c and c not in uniq:
            uniq.append(c)
    return ", ".join(uniq)


def _allowed_english_terms(crop: str) -> set[str]:
    allowed = {
        "NPK", "NAA", "FYM", "DAS", "ZnSO4", "MnSO4", "SSP", "Azospirillum",
        "Phosphobacteria", "Carbendazim", "Thiram", "Gypsum", "Borax",
        "Sodium", "metasilicate", "sett", "setts", "bud", "buds",
        "furrow", "furrows", "slurry", "capitulum", "sprayer", "spray",
        "biofertilizer", "hybrid", "hybrids", "variety", "varieties",
        "rainfed", "irrigated", "foliar", "micronutrient",
    }
    crop_tokens = re.findall(r"[A-Za-z]+", crop)
    allowed.update(crop_tokens)
    return {t.lower() for t in allowed}


def _english_signal_words(text: str, crop: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9/+.-]*", text)
    allowed = _allowed_english_terms(crop)
    bad = []
    for w in words:
        wl = w.lower()
        if wl in allowed:
            continue
        if len(w) <= 2:
            continue
        if re.fullmatch(r"[0-9.]+", w):
            continue
        bad.append(w)
    return bad


def _is_english_heavy(text: str, crop: str) -> bool:
    bad = _english_signal_words(text, crop)
    total = re.findall(r"\S+", text)
    if not total:
        return False
    return len(bad) >= 8 or (len(bad) >= 5 and len(bad) / max(1, len(total)) > 0.22)


def _looks_like_noisy_ocr(text: str) -> bool:
    patterns = [
        r"T_Max", r"T_Min", r"Altitude\s+m\s+MSL", r"Optimum\s+oC",
        r"\biii\)", r"\bth e\b", r"\bt o\b", r"\bhe althy\b",
        r"\bdevelop(ed)? by TNAU\b", r"\bcapitula\b", r"\bbrackish water\b",
    ]
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def _has_raw_heading_leak(text: str) -> bool:
    stripped = text.strip()
    if re.match(r"^[0-9IVXivx.\s-]*[A-Z][A-Za-z]+(?:\s+[A-Z]?[A-Za-z]+){2,}\s*:", stripped):
        return True
    if re.search(r"[A-Za-z]{4,}(?:\s+[A-Za-z]{3,}){4,}", stripped):
        return True
    return False


def _normalize_line_key(text: str) -> str:
    return re.sub(r"[^a-z0-9\u0900-\u097f]+", "", text.lower())


def _has_wrong_harvest_phrase(crop_type: str, text: str) -> bool:
    grain_phrase = "जब दाना सख्त हो जाए"
    straw_phrase = "पुआल सूखकर भुरभुरा"
    if crop_type in {"grain", "pulse"}:
        return False
    return grain_phrase in text or straw_phrase in text


def _safe_harvest_text(crop: str, crop_type: str) -> str:
    label = "कटाई"
    if crop_type == "cane":
        return f"{label}: फसल पूरी तरह पकने पर गन्ने को जमीन के पास से काटें। कटाई के बाद सूखी पत्तियां अलग करें और cane को मिल/बाजार तक जल्दी पहुंचाएं।"
    if crop_type == "oilseed":
        return f"{label}: फसल पकने पर फूल का सिरा या फलियां काटें। कटाई के बाद अच्छी तरह सुखाकर मड़ाई करें, बीज अलग करें और सूखाकर संग्रहित करें।"
    if crop_type == "fibre":
        return f"{label}: फसल पकने पर समय से कटाई करें। कटाई के बाद बंडल बनाकर आगे की सफाई/प्रसंस्करण के लिए तैयार करें।"
    if crop_type == "mushroom":
        return f"{label}: उचित आकार आने पर मशरूम को सावधानी से तोड़ें। साफ करके तुरंत उपयोग या ठंडे स्थान पर संग्रहित करें।"
    if crop_type == "pulse":
        return f"{label}: जब अधिकांश फलियां पक जाएं, तब कटाई करें। कटाई के बाद सुखाकर मड़ाई करें और दाने साफ करके संग्रहित करें।"
    return f"{label}: फसल पकने पर समय से कटाई करें। कटाई के बाद साफ-सफाई और सुखाने का काम ठीक से करें।"


def _safe_render_block(crop: str, heading: str, body: str, phase: str) -> str:
    crop_type = _crop_type(crop)
    heading_key = _heading_key(heading)
    label = _heading_label_hi(heading)
    text = _cleanup_bullet_text(_translate_terms(_compress_sentences(body, limit=3)))

    if any(token in heading_key for token in ["TIME OF SOWING", "SOWING OF SEEDS", "FERTILIZER APPLICATION"]):
        candidate = _summarize_block(crop, heading, body)
        if candidate:
            return candidate

    if "CLIMATE REQUIREMENT" in heading_key:
        lower = body.lower()
        parts = []
        if "humid" in lower:
            parts.append("यह फसल गर्म और नम जलवायु में अच्छी रहती है।")
        elif "cool and dry climate" in lower:
            parts.append("इस फसल के लिए ठंडी और शुष्क जलवायु अच्छी रहती है।")
        elif "semi arid" in lower:
            parts.append("यह फसल अर्ध-शुष्क जलवायु में अच्छी रहती है।")
        else:
            parts.append("इस फसल के लिए उपयुक्त तापमान और नमी वाली जलवायु जरूरी है।")
        if "rabi" in lower:
            parts.append("यह रबी मौसम की फसल है।")
        elif "kharif" in lower:
            parts.append("यह खरीफ मौसम में अच्छी तरह उगाई जाती है।")
        if "rainfed" in lower:
            parts.append("इसे वर्षा आधारित परिस्थितियों में भी उगाया जा सकता है।")
        return f"{label}: {' '.join(parts)}"

    if "SEASON AND VARIET" in heading_key or "DISTRICT/SEASON VARIETIES" in heading_key:
        lower = body.lower()
        parts = []
        if "kuruvai" in lower or "samba" in lower or "thaladi" in lower or "navarai" in lower:
            parts.append("इस फसल की उपयुक्त किस्में और बुवाई का समय क्षेत्र और मौसम के अनुसार बदलते हैं।")
            parts.append("अपने जिले के लिए स्थानीय कृषि विश्वविद्यालय या कृषि विभाग की सिफारिश वाली किस्म चुनें।")
        else:
            parts.append("उपयुक्त किस्म और बुवाई का समय अपने क्षेत्र, पानी की उपलब्धता और मौसम के अनुसार चुनें।")
        return f"{label}: {' '.join(parts)}"

    if "DISEASE MANAGEMENT IN NURSERY" in heading_key:
        treatments = _extract_nursery_treatments(body)
        if treatments:
            return f"{label}: " + " | ".join(treatments[:3])
        return "नर्सरी में रोग प्रबंधन: बुवाई से पहले बीज उपचार करें। उपयुक्त fungicide या bio-agent को सिफारिश के अनुसार उपयोग करें।"

    if "NURSERY MANAGEMENT" in heading_key:
        return "नर्सरी प्रबंधन: पानी के पास अच्छी जमीन चुनें, जरूरत के अनुसार बीज दर रखें और नर्सरी में बीज उपचार, समतल बुवाई तथा नमी प्रबंधन पर ध्यान दें।"

    if phase == "harvest":
        if "JUDGE WHEN TO HARVEST" in heading_key:
            if crop_type == "oilseed":
                return "कटाई का सही समय: जब फूल के पीछे की पत्तियां पीली पड़ने लगें और फूल का सिरा सख्त हो जाए, तब कटाई करें।"
            if crop_type == "cane":
                return "कटाई का सही समय: जब फसल पूरी परिपक्व हो जाए और रस/गुणवत्ता बेहतर हो, तब कटाई करें।"
        return _safe_harvest_text(crop, crop_type)

    if "FARM LAND PREPARATION" in heading_key:
        spacing = _extract_first(r"([0-9.]+\s*cm)\s*apart", body)
        if spacing:
            return f"{label}: लगभग {_translate_terms(spacing)} दूरी पर मेड़ और नालियां बनाएं।"
        return f"{label}: खेत को अच्छी तरह तैयार करके मेड़-नाली व्यवस्था बनाएं।"

    if "SEED TREATMENT" in heading_key:
        chem_pat = re.compile(
            r"(?i)(carbendazim(?:\s*\d+%[A-Z]*)?|thiram|captan|carboxin|tricyclazole|pyroquilon|pseudomonas fluorescens|mancozeb(?:\s*\d+%[A-Z]*)?)"
        )
        dose_pat = re.compile(
            r"(?i)([0-9.]+(?:\s*[-–]\s*[0-9.]+)?\s*(?:g/kg of seed|g/kg|g/l of water|g/l|ml/l|kg/ha))"
        )
        chems = _summarize_chemicals([m.group(1) for m in chem_pat.finditer(body)])
        doses = []
        for m in dose_pat.finditer(body):
            d = m.group(1).strip()
            if d not in doses:
                doses.append(d)
        parts = ["बुवाई से पहले बीज का उपचार करें।"]
        if chems:
            parts.append(f"{chems} का उपयोग सिफारिश के अनुसार करें।")
        else:
            parts.append("उपयुक्त fungicide या bio-agent का उपयोग सिफारिश के अनुसार करें।")
        if doses:
            parts.append(f"खुराक: {', '.join(doses[:3])}।")
        parts.append("उपचार के बाद बीज को सुखाकर बुवाई करें।")
        return f"{label}: {' '.join(parts)}"

    if "APPLICATION OF MICRONUTRIENTS" in heading_key:
        qty = _extract_first(r"([0-9.]+\s*kg/ha)", body)
        if qty:
            return f"{label}: सूक्ष्म पोषक तत्व मिश्रण लगभग {qty} की दर से मिट्टी/FYM के साथ मिलाकर दें।"
        return f"{label}: कमी होने पर सूक्ष्म पोषक तत्व सिफारिश के अनुसार दें।"

    if "FOLIAR SPRAY" in heading_key:
        qty = _extract_first(r"([0-9.]+\s*g\s*NAA.*?)\)", body)
        if "NAA" in body.upper():
            line = "NAA का पत्तियों पर छिड़काव 30वें और 60वें दिन करें।"
            if qty:
                line += f" उदाहरण मात्रा: {qty}।"
            return f"{label}: {line}"
        return f"{label}: जरूरत के अनुसार पत्तियों पर सिफारिश वाला छिड़काव करें।"

    if "SULPHUR" in heading_key:
        qty = _extract_first(r"([0-9.]+\s*kg/ha)", body)
        return f"{label}: गंधक लगभग {qty if qty else '20 kg/ha'} की दर से दें।"

    if "BIOFERTILIZER FOR SUGARCANE" in heading_key:
        return f"{label}: Azospirillum, Gluconacetobacter और phosphobacteria जैसे जैव उर्वरक जड़ों की वृद्धि और पोषण में मदद करते हैं। इन्हें सिफारिश के अनुसार उपयोग करें।"

    if "PRE-HARVEST PRACTICES" in heading_key:
        return f"{label}: कटाई से पहले जरूरत होने पर cane ripener का सिफारिश अनुसार उपयोग करें, ताकि गन्ने की गुणवत्ता और मिठास बेहतर रहे।"

    if "BORIC ACID" in heading_key:
        return f"{label}: फूल आने की अवस्था में 0.2% Boric Acid का छिड़काव करें, ताकि बीज भराव और बीज बनना बेहतर हो।"

    if "IMPROVING SEED SET" in heading_key:
        return f"{label}: मध्य फूल अवस्था में हल्के हाथ से परागण में मदद करें। जरूरत हो तो मधुमक्खी के बक्सों का उपयोग करें।"

    if "CROP PROTECTION" in heading_key:
        return f"{label}: प्रमुख रोग/कीट पर नियमित निगरानी रखें और जरूरत होने पर सिफारिश अनुसार दवा/बीज उपचार अपनाएं।"

    if "SUPPLEMENTAL IRRIGATION" in heading_key or "SEMI DRY RICE" in body.upper():
        return "सिंचाई: वर्षा आधारित स्थिति में नमी बचाकर रखें। जरूरत पड़ने पर महत्वपूर्ण बढ़वार और फूल/दाना बनने की अवस्था में पूरक सिंचाई दें।"

    if "IRRIGATION" in heading_key or "WATER MANAGEMENT" in heading_key:
        return "सिंचाई: फसल की अवस्था और मिट्टी की नमी के अनुसार सिंचाई दें। अंकुरण, बढ़वार, फूल आने और दाना बनने की अवस्था में पानी की कमी न होने दें।"

    if text:
        return f"{label}: {text}"
    return ""


def _validate_and_repair_guide(crop: str, entries: list[dict[str, str]]) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    crop_type = _crop_type(crop)
    phase_points: dict[str, list[str]] = {k: [] for k in PHASE_LABELS_HI}
    issues: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        point = entry["point"].strip()
        phase = entry["phase"]
        heading = entry["heading"]
        body = entry["body"]
        reasons: list[str] = []
        if _is_english_heavy(point, crop):
            reasons.append("english_heavy")
        if _looks_like_noisy_ocr(point):
            reasons.append("ocr_noise")
        if _has_raw_heading_leak(point):
            reasons.append("raw_heading_leak")
        if phase == "harvest" and _has_wrong_harvest_phrase(crop_type, point):
            reasons.append("wrong_harvest_template")
        key = _normalize_line_key(point)
        if key in seen:
            reasons.append("duplicate")
        repaired = point
        if reasons:
            repaired = _safe_render_block(crop, heading, body, phase) or point
            issues.append(
                {
                    "crop": crop,
                    "phase": phase,
                    "heading": heading,
                    "reasons": reasons,
                    "original": point,
                    "repaired": repaired,
                }
            )
        repaired = _final_phrase_cleanup(repaired)
        repaired_key = _normalize_line_key(repaired)
        if repaired_key in seen:
            continue
        seen.add(repaired_key)
        phase_points[phase].append(repaired)
    return phase_points, issues


def _prune_generic_phase_points(phase_points: dict[str, list[str]]) -> dict[str, list[str]]:
    cleaned: dict[str, list[str]] = {}
    for phase, pts in phase_points.items():
        items = list(pts or [])
        has_specific_seed_rate = any(
            p.startswith("बीज दर:") and "किस्म और सिंचाई की स्थिति के अनुसार" not in p for p in items
        )
        if has_specific_seed_rate:
            items = [
                p
                for p in items
                if not (p.startswith("बीज दर:") and "किस्म और सिंचाई की स्थिति के अनुसार" in p)
            ]
        cleaned[phase] = items
    return cleaned


def _build_guide_points_for_crop(crop: str) -> tuple[dict[str, list[str]] | None, list[dict[str, str]], list[str]]:
    section_text = _extract_section_text(crop)
    if not section_text:
        return None, [], []
    blocks = _extract_blocks(section_text)
    entries: list[dict[str, str]] = []
    for heading, body in blocks:
        if _should_stop_at_heading(crop, heading):
            break
        if _should_skip_heading(crop, heading):
            continue
        phase = _assign_phase(heading)
        if not phase:
            continue
        point = _summarize_block(crop, heading, body)
        if "IMPROVING SEED SET" in _heading_key(heading):
            point = _safe_render_block(crop, heading, body, phase)
        if point:
            entries.append(
                {
                    "phase": phase,
                    "heading": heading,
                    "body": body,
                    "point": point,
                }
            )
    phase_points, issues = _validate_and_repair_guide(crop, entries)
    _append_review_queue(crop, issues)
    phase_points = _prune_generic_phase_points(phase_points)
    if not any(phase_points.values()):
        return None, [], []
    return phase_points, entries, [str(GUIDE_PDF)]


def _append_review_queue(crop: str, issues: list[dict[str, Any]]) -> None:
    if not issues:
        return
    GUIDE_REVIEW_QUEUE.parent.mkdir(parents=True, exist_ok=True)
    with GUIDE_REVIEW_QUEUE.open("a", encoding="utf-8") as f:
        for issue in issues:
            row = {"crop": crop, **issue}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _extract_first(pattern: str, text: str) -> str | None:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    return m.group(1).strip() if m else None


def _extract_all(pattern: str, text: str) -> list[tuple[str, ...]]:
    return [tuple(g.strip() for g in m.groups()) for m in re.finditer(pattern, text, flags=re.IGNORECASE)]


def _normalize_irrigation_stage_name(name: str) -> str:
    normalized = " ".join(str(name or "").lower().split())
    normalized = normalized.replace("grandgrowth", "grand growth")
    normalized = re.sub(r"^days of irrigation interval stages\s+", "", normalized)
    normalized = re.sub(r"^stages\s+", "", normalized)
    normalized = normalized.replace("t o ", "to ")
    normalized = re.sub(r"\s+", " ", normalized).strip(" :;,-")
    return normalized


def _display_irrigation_stage_name(name: str) -> str:
    normalized = _normalize_irrigation_stage_name(name)
    display = _translate_terms(normalized)
    display = display.replace("pre-फूल आने की अवस्था", "फूल आने से पहले की अवस्था")
    return _cleanup_bullet_text(display)


def _format_water_stages(text: str) -> str:
    stages = []
    seen: set[str] = set()
    patterns = [
        r"(Immediately after sowing|[A-Za-z][A-Za-z \-/]+?(?:phase|stage))\s*:\s*([0-9\- ]+\s*DAS)",
        r"(Immediately after sowing)\b",
    ]
    for pattern in patterns:
        for stage_match in re.finditer(pattern, text, flags=re.IGNORECASE):
            raw_name = stage_match.group(1).strip()
            label = _display_irrigation_stage_name(raw_name)
            days = (stage_match.group(2) or "").strip() if stage_match.lastindex and stage_match.lastindex >= 2 else ""
            item = f"{label} ({days})" if days else label
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            stages.append(item)
    return "; ".join(stages[:5])


def _extract_irrigation_stage_ranges(text: str) -> list[dict[str, object]]:
    ranges: list[dict[str, object]] = []
    seen: set[tuple[str, int, int]] = set()
    patterns = [
        r"([A-Za-z][A-Za-z \-/]+?phase)\s*\((\d+)\s*(?:-|to)\s*(\d+)\s*days?\)\s*(\d+)\s*(\d+)",
        r"([A-Za-z][A-Za-z \-/]+?phase)\s*:\s*(\d+)\s*to\s*(\d+)\s*days",
        r"(Crown root intiation|Active tillering stage|Flowering stage|Grain filling stage)\s*:\s*(\d+)\s*-\s*(\d+)\s*DAS",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            raw_label = _normalize_irrigation_stage_name(match.group(1))
            try:
                start = int(match.group(2))
                end = int(match.group(3))
            except Exception:
                continue
            interval_min = interval_max = None
            if len(match.groups()) >= 5 and match.group(4) and match.group(5):
                try:
                    interval_min = int(match.group(4))
                    interval_max = int(match.group(5))
                except Exception:
                    interval_min = interval_max = None
            key = (raw_label, start, end)
            if key in seen:
                continue
            seen.add(key)
            ranges.append(
                {
                    "label": raw_label,
                    "start": start,
                    "end": end,
                    "interval_min": interval_min,
                    "interval_max": interval_max,
                }
            )
    return ranges


def _extract_crop_age_days(question: str) -> int | None:
    q = str(question or "").lower()
    month_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:mahine|maheene|mahina|month|months)", q)
    if month_match:
        try:
            return int(round(float(month_match.group(1)) * 30))
        except Exception:
            return None
    day_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:din|days?|dap)\b", q)
    if day_match:
        try:
            return int(round(float(day_match.group(1))))
        except Exception:
            return None
    return None


def _age_specific_irrigation_note(question: str, irrigation_text: str) -> str | None:
    age_days = _extract_crop_age_days(question)
    if age_days is None:
        return None
    stage_ranges = _extract_irrigation_stage_ranges(irrigation_text)
    for stage in stage_ranges:
        start = int(stage.get("start") or 0)
        end = int(stage.get("end") or 0)
        if start <= age_days <= end:
            label = _display_irrigation_stage_name(str(stage.get("label") or "इस अवस्था"))
            interval_min = stage.get("interval_min")
            interval_max = stage.get("interval_max")
            if interval_min and interval_max:
                return (
                    f"लगभग {age_days} दिन की फसल {label} में आती है। "
                    f"इस अवस्था में आम तौर पर हर रोज सिंचाई नहीं दी जाती; "
                    f"सामान्यतः लगभग {interval_min}-{interval_max} दिन के अंतर पर सिंचाई रखें, "
                    "लेकिन गर्मी, मिट्टी और नमी के अनुसार अंतर बदल सकता है।"
                )
            return (
                f"लगभग {age_days} दिन की फसल {label} में आती है। "
                "इस अवस्था में मिट्टी की नमी बनाए रखें और फसल को पानी की कमी न होने दें।"
            )
    if age_days <= 7:
        return "शुरुआती अवस्था में आम तौर पर हर रोज भारी सिंचाई नहीं दी जाती; हल्की नमी बनाए रखें और खेत में पानी खड़ा न होने दें।"
    return None


def _summarize_block(crop: str, heading: str, body: str) -> str:
    text = _clean_text(body)
    heading_key = _heading_key(heading)
    label = _heading_label_hi(heading)

    if any(skip in heading_key for skip in ["DESCRIPTION OF", "SEED PRODUCTION"]):
        return ""

    if "CLIMATE REQUIREMENT" in heading_key:
        optimum = _extract_first(r"Optimum\s*oC\s*([0-9 \-]+)", text)
        rainfall = _extract_first(r"Rainfall\s*mm\s*([0-9 \-]+)", text)
        parts = []
        if optimum:
            parts.append(f"उपयुक्त तापमान लगभग {optimum.strip()}°C रहता है।")
        if rainfall:
            parts.append(f"लगभग {rainfall.strip()} mm वर्षा उपयुक्त मानी जाती है।")
        if "cool and dry climate" in text.lower():
            parts.append("इस फसल के लिए ठंडी और शुष्क जलवायु अच्छी रहती है।")
        if "semi arid climate" in text.lower():
            parts.append("यह फसल अर्ध-शुष्क जलवायु में अच्छी रहती है।")
        if "rabi season" in text.lower():
            parts.append("यह रबी मौसम की फसल है।")
        if "rainfed crop" in text.lower():
            parts.append("इसे वर्षा आधारित परिस्थितियों में भी उगाया जा सकता है।")
        return f"{label}: {' '.join(parts) or _translate_terms(_compress_sentences(text))}"

    if "SEASON AND VARIETY" in heading_key or "SEASON AND VARIETIES" in heading_key or "DISTRICT/SEASON VARIETIES" in heading_key:
        if re.search(r"Zone\s+District/Season\s+Month", text, flags=re.IGNORECASE):
            return f"{label}: उपयुक्त किस्म और बुवाई का समय अपने क्षेत्र, पानी की उपलब्धता और मौसम के अनुसार चुनें।"
        sow = _extract_first(r"Ideal sowing time is\s*(.*?)(?:\.|Variety)", text)
        var = _extract_first(r"Variety\s*:\s*(.*?)(?:Morphological|$)", text)
        parts = []
        if sow:
            parts.append(f"बुवाई का सही समय: {_translate_terms(sow.strip())}.")
        if var:
            parts.append(f"उपयुक्त किस्में: {var.strip().rstrip('.')}.")
        if not parts:
            season_rows = _extract_all(r"([A-Za-z]+)\s*\(([^)]+)\)\s*All districts\s*(.*?)(?:$)", text)
            if season_rows:
                season_name, season_window, varieties = season_rows[0]
                parts.append(f"उपयुक्त मौसम: {season_name} ({season_window})।")
                if varieties:
                    parts.append(f"उपयुक्त किस्में: {varieties}।")
        return f"{label}: {' '.join(parts) or _translate_terms(_compress_sentences(text))}"

    if "FIELD PREPARATION" in heading_key:
        return f"{label}: 2-3 जुताई करके खेत को भुरभुरा और समतल तैयार करें।"

    if "APPLICATION OF FYM OR COMPOST" in heading_key:
        qty = _extract_first(r"Spread\s*([0-9.]+\s*t/ha)", text)
        if qty:
            return f"{label}: लगभग {_translate_terms(qty)} गोबर की सड़ी खाद या कम्पोस्ट खेत में डालकर मिट्टी में मिला दें।"
        qty2 = _extract_first(r"Spread\s*([0-9.]+\s*t.*?)\s+per ha", text)
        if qty2:
            return f"{label}: लगभग {_translate_terms(qty2)} गोबर की सड़ी खाद या कम्पोस्ट प्रति हेक्टेयर डालकर मिट्टी में मिला दें।"

    if "APPLICATION OF MICRONUTRIENTS" in heading_key:
        qty = _extract_first(r"Mix\s*([0-9.]+\s*kg/ha)", text)
        total = _extract_first(r"total quantity of\s*([0-9.]+\s*kg/ha)", text)
        parts = []
        if qty:
            parts.append(f"लगभग {_translate_terms(qty)} micronutrient mixture लें।")
        if total:
            parts.append(f"इसे sand/FYM के साथ मिलाकर कुल मात्रा लगभग {_translate_terms(total)} करें।")
        if "manganese deficiency" in text.lower():
            parts.append("यदि manganese deficiency हो, तो MnSO4 का foliar spray सिफारिश के अनुसार करें।")
        if "zinc deficiency" in text.lower():
            parts.append("यदि zinc deficiency हो, तो ZnSO4 basal या foliar spray सिफारिश के अनुसार दें।")
        if "borax" in text.lower() or "gypsum" in text.lower():
            parts.append("Boron/Sulphur की कमी वाली मिट्टी में Borax या Gypsum सिफारिश के अनुसार दें।")
        return f"{label}: {' '.join(parts) if parts else 'जरूरत होने पर micronutrient mixture सिफारिश के अनुसार दें।'}"

    if "SEED TREATMENT" in heading_key:
        if "soaking seeds in 2%" in text.lower():
            soak = _extract_first(r"Soaking seeds in\s*(.*?)(?:\s*for|sowing)", text)
            hours = _extract_first(r"for\s*([0-9]+\s*hrs?)", text)
            parts = []
            if soak:
                parts.append(f"बीज को {_translate_terms(soak)} घोल में उपचारित करें।")
            if hours:
                parts.append(f"उपचार का समय लगभग {_translate_terms(hours)} रखें।")
            parts.append("उपचार के बाद बीज को छाया में सुखाकर बुवाई करें।")
            return f"{label}: {' '.join(parts)}"
        dose = _extract_first(r"at\s*([0-9.]+\s*g/kg.*?)\s*(?:24 hours|$)", text)
        chems = _extract_first(r"Treat the seeds with\s*(.*?)(?:at|24 hours|$)", text)
        parts = []
        if chems:
            parts.append(f"बीज को {chems.strip().replace(' or ', ' या ')} से उपचारित करें।")
        if dose:
            parts.append(f"खुराक: {_translate_terms(dose.strip())}।")
        if "24 hours before sowing" in text.lower():
            parts.append("यह उपचार बुवाई से 24 घंटे पहले करें।")
        if "azospirillum" in text.lower() or "phosphobacteria" in text.lower():
            parts.append("जरूरत होने पर Azospirillum/Phosphobacteria जैसे biofertilizer से उपचार करके तुरंत बुवाई करें।")
        return f"{label}: {' '.join(parts) or _translate_terms(_compress_sentences(text))}"

    if "SEED RATE" in heading_key:
        setts = _extract_first(r"([0-9,]+\s*two-budded setts/ha)", text)
        if setts:
            return f"{label}: लगभग {_translate_terms(setts)} रखें।"
        if "quantity of seed required" in text.lower():
            pure_crop = _extract_first(r"(?:Pure|Sole)\s*crop\s*([0-9.]+)", text)
            mixed_crop = _extract_first(r"Mixed\s*crop\s*([0-9.]+)", text)
            rice_fallow = _extract_first(r"Rice\s*fallows?.*?([0-9.]+)\s*[-–]?\s*$", text)
            parts = []
            if pure_crop:
                parts.append(f"शुद्ध फसल में लगभग {pure_crop.strip()} किग्रा/हेक्टेयर बीज रखें।")
            if mixed_crop:
                parts.append(f"मिश्रित फसल में लगभग {mixed_crop.strip()} किग्रा/हेक्टेयर बीज रखें।")
            if rice_fallow:
                parts.append(f"rice fallow स्थिति में लगभग {rice_fallow.strip()} किग्रा/हेक्टेयर बीज रखें।")
            if parts:
                return f"{label}: {' '.join(parts)}"
        rows = _extract_all(r"(Varieties|Hybrids)\s*([0-9.]+\s*kg/ha)\s*([0-9.]+\s*kg/ha)", text)
        if rows:
            parts = []
            for kind, rainfed, irrigated in rows[:2]:
                kind_hi = "खुली परागित किस्में" if kind.lower().startswith("var") else "हाइब्रिड"
                parts.append(f"{kind_hi}: वर्षा आधारित में {_translate_terms(rainfed)}, सिंचित में {_translate_terms(irrigated)}।")
            return f"{label}: " + " ".join(parts)
        return f"{label}: बीज दर किस्म और सिंचाई की स्थिति के अनुसार रखें।"

    if "FORMING BEDS AND CHANNEL" in heading_key:
        size = _extract_first(r"Form beds size on\s*(.*?)(?:\.|The irrigation)", text)
        parts = []
        if size:
            parts.append(f"खेत में लगभग {_translate_terms(size.strip())} की क्यारियां बनाएं।")
        parts.append("सिंचाई के लिए पर्याप्त नालियां रखें।")
        return f"{label}: {' '.join(parts)}"

    if "FORMING RIDGES AND FURROWS" in heading_key:
        spacing = _extract_first(r"with\s*([0-9.]+\s*cm)\s*spacing", text)
        parts = []
        if spacing:
            parts.append(f"लगभग {_translate_terms(spacing)} दूरी पर मेड़ और नालियां बनाएं।")
        parts.append("खेत की ढाल के अनुसार सिंचाई की नालियां रखें।")
        return f"{label}: {' '.join(parts)}"

    if "FERTILIZER APPLICATION" in heading_key or "APPLICATION OF FERTILIZERS" in heading_key:
        if crop.lower() == "sugarcane":
            npk = _extract_first(r"NPK\s*@\s*([0-9:]+)\s*kg/ha", text)
            if npk:
                return f"{label}: यदि मिट्टी जांच उपलब्ध न हो तो लगभग {npk} NPK किग्रा/हेक्टेयर दें। Super phosphate furrow में डालकर मिट्टी से मिला दें।"
        rainfed = _extract_first(
            r"Rainfed\s*:\s*([0-9.]+\s*kg\s*N\s*\+\s*[0-9.]+\s*kg\s*P\s*2O5\s*\+\s*[0-9.]+\s*kg\s*K\s*2O\s*\+\s*[0-9.]+\s*kg\s*S\*?/ha)",
            text,
        )
        irrigated = _extract_first(
            r"Irrigated\s*:\s*([0-9.]+\s*kg\s*N\s*\+\s*[0-9.]+\s*kg\s*P\s*2O5\s*\+\s*[0-9.]+\s*kg\s*K\s*2O\s*\+\s*[0-9.]+\s*kg\s*S\*?/ha)",
            text,
        )
        if rainfed or irrigated:
            parts = ["यदि मिट्टी जांच उपलब्ध न हो, तो उर्वरक बुवाई से पहले बेसल रूप में दें।"]
            if rainfed:
                rainfed_txt = re.sub(r"\s+", " ", rainfed).replace("P 2O5", "P2O5").replace("K 2O", "K2O").replace("S*", "S")
                parts.append(f"वर्षा आधारित फसल में लगभग {rainfed_txt} दें।")
            if irrigated:
                irrigated_txt = re.sub(r"\s+", " ", irrigated).replace("P 2O5", "P2O5").replace("K 2O", "K2O").replace("S*", "S")
                parts.append(f"सिंचित फसल में लगभग {irrigated_txt} दें।")
            if "gypsum" in text.lower() and "single super phospate" in text.lower():
                parts.append("अगर फॉस्फोरस के लिए SSP न दें, तो गंधक gypsum के रूप में दें।")
            return f"{label}: {' '.join(parts)}"
        npk = _extract_first(r"recommendation of\s*([0-9: ]+NPK\s*kg/ha|[0-9: ]+\s*NPK\s*kg/ha|[0-9: ]+)", text)
        zns = _extract_first(r"Apply\s*([0-9.]+\s*kg\s*ZnSO\s*4.*?)\.", text)
        parts = []
        if npk:
            npk_value = re.sub(r"\s*NPK\s*kg/ha", "", npk.strip(), flags=re.IGNORECASE)
            parts.append(f"यदि मिट्टी जांच उपलब्ध न हो तो लगभग {npk_value} NPK किग्रा/हेक्टेयर दें।")
        if zns:
            parts.append(f"जिंक/सल्फर की कमी वाली मिट्टी में {_translate_terms(zns.strip())} दें।")
        parts.append("आधा नाइट्रोजन और पूरा फॉस्फोरस-पोटाश बुवाई से पहले दें।")
        return f"{label}: {' '.join(parts)}"

    if "FOLIAR SPRAY" in heading_key:
        if "naphthalene acetic acid" in text.lower() or "(naa)" in text.lower():
            return f"{label}: NAA 20 ppm की foliar spray करें। उदाहरण: लगभग 280 g NAA को 625 लीटर पानी में घोलकर 30वें और 60वें दिन छिड़काव करें। पूरे पौधे पर समान spray करें और खारे पानी का उपयोग न करें।"
        text2 = _cleanup_bullet_text(_compress_sentences(text, limit=3))
        return f"{label}: {text2}"

    if "IMPROVING SEED SET" in heading_key:
        return "बीज बनने में सुधार: मध्य फूल अवस्था में हल्के हाथ से फूल के सिरे को रगड़कर परागण में मदद करें या दो फूलों को हल्के से आमने-सामने छुआएँ। यह काम सुबह 9 से 11 बजे के बीच करें। जरूरत हो तो मधुमक्खी के बक्सों का उपयोग करें।"

    if "SULPHUR" in heading_key or "BORIC ACID" in heading_key or "IMPROVING SEED SET" in heading_key:
        if "SULPHUR" in heading_key:
            return f"{label}: लगभग 20 किग्रा/हेक्टेयर sulphur दें। इसे ammonium sulphate, single super phosphate या gypsum के रूप में दिया जा सकता है।"
        if "BORIC ACID" in heading_key:
            return f"{label}: ray floret opening stage पर 0.2% boric acid (लगभग 2 g/लीटर पानी) का spray करें, ताकि seed set और seed filling बेहतर हो।"
        text2 = _cleanup_bullet_text(_compress_sentences(text, limit=3))
        return f"{label}: {text2}"

    if "TIME OF SOWING" in heading_key:
        if "third week of january" in text.lower() and "second week of february" in text.lower():
            return f"{label}: सामान्यतः जनवरी के तीसरे सप्ताह से फरवरी के दूसरे सप्ताह तक बुवाई करें।"
        return f"{label}: {_cleanup_bullet_text(_translate_terms(_compress_sentences(text, limit=2)))}"

    if "SOWING OF SEEDS" in heading_key:
        lower = text.lower()
        if "relay cropping" in lower and "harvest of the paddy crop" in lower:
            parts = [
                "रिले फसल पद्धति में धान की कटाई से लगभग 5-10 दिन पहले खड़े खेत में उचित नमी पर बीज समान रूप से बिखेरें।",
            ]
            if "combined harvesting areas" in lower:
                parts.append("जहाँ मशीन से कटाई होती है, वहाँ धान की कटाई से पहले ही बीज का छिटकाव करें।")
            return f"{label}: {' '.join(parts)}"
        return f"{label}: {_cleanup_bullet_text(_translate_terms(_compress_sentences(text, limit=2)))}"

    if heading_key == "SOWING":
        spacing_line = _extract_first(r"Spacing\s*:\s*(.*?)(?:i\)|$)", text)
        spacing = _extract_first(r"lines\s*([0-9]+\s*cm.*?)\s*apart", text)
        depth = _extract_first(r"depth of\s*([0-9]+\s*cm)", text)
        parts = []
        if spacing_line:
            line = _translate_terms(spacing_line.strip())
            line = line.replace("Hybrids", "हाइब्रिड").replace("Varieties", "खुली परागित किस्में")
            parts.append(f"पौधों की दूरी: {line}।")
        if spacing:
            parts.append(f"कतार से कतार दूरी लगभग {_translate_terms(spacing.strip())} रखें।")
        if depth:
            parts.append(f"बीज की गहराई लगभग {_translate_terms(depth.strip())} रखें।")
        if "two seeds per hole" in text.lower():
            parts.append("प्रति जगह 2 बीज रखें, बाद में स्वस्थ पौधा छोड़ें।")
        parts.append("उर्वरक देने के बाद बुवाई करें और बहुत गहरी बुवाई से बचें।")
        return f"{label}: {' '.join(parts)}"

    if heading_key == "PLANTING" or "PREPARATION OF SETTS FOR PLANTING" in heading_key:
        parts = []
        if "take seed material" in text.lower():
            parts.append("6-7 महीने की स्वस्थ, रोग-मुक्त seed cane/sett material लें।")
        if "detrash" in text.lower():
            parts.append("sett बनाने से पहले cane की सूखी पत्तियां हाथ से हटा दें।")
        if "sharp knife" in text.lower() or "sett cutting machine" in text.lower():
            parts.append("sett काटते समय तेज चाकू या sett cutting machine का उपयोग करें, ताकि bud damage न हो।")
        if "slurry" in text.lower():
            parts.append("भारी मिट्टी में furrow में हल्की गीली slurry बनाकर sett रखें।")
        if "12 buds/metre" in text.lower():
            parts.append("लगभग 12 buds प्रति मीटर के हिसाब से setts रखें।")
        if "cover the exposed setts" in text.lower():
            parts.append("अगले दिन खुले setts को मिट्टी से ढक दें ताकि धूप से नुकसान न हो।")
        if parts:
            return f"{label}: {' '.join(parts)}"
        return f"{label}: {_cleanup_bullet_text(_compress_sentences(text, limit=3))}"

    if "THINNING" in heading_key:
        day = _extract_first(r"on the\s*([0-9a-z]+(?:th|st|nd|rd)? day)", text)
        parts = []
        if day:
            parts.append(f"बुवाई के लगभग {_cleanup_bullet_text(day)} पर छंटाई करें।")
        parts.append("हर जगह केवल 1 स्वस्थ और मजबूत पौधा रखें।")
        return f"{label}: {' '.join(parts)}"

    if "WEED MANAGEMENT" in heading_key:
        return f"{label}: 3 दिन बाद बुवाई पर प्री-इमर्जेन्स खरपतवारनाशी का छिड़काव करें। उसके बाद 20-35 दिन के बीच 1-2 बार निराई करें।"

    if "WATER MANAGEMENT" in heading_key or heading_key == "IRRIGATION":
        irrig = _extract_first(r"requires\s*([0-9 \-]+)\s*irrigations", text)
        stage_text = _format_water_stages(text)
        stage_ranges = _extract_irrigation_stage_ranges(text)
        parts = []
        if stage_ranges:
            stage_lines = []
            for stage in stage_ranges[:4]:
                stage_label = _display_irrigation_stage_name(str(stage.get("label") or "").strip())
                start = int(stage.get("start") or 0)
                end = int(stage.get("end") or 0)
                interval_min = stage.get("interval_min")
                interval_max = stage.get("interval_max")
                if interval_min and interval_max:
                    stage_lines.append(
                        f"{stage_label} ({start}-{end} दिन): सामान्यतः लगभग {interval_min}-{interval_max} दिन के अंतर पर सिंचाई रखें"
                    )
                else:
                    stage_lines.append(f"{stage_label} ({start}-{end} दिन)")
            if stage_lines:
                parts.append("अवस्था-आधारित सिंचाई: " + "; ".join(stage_lines) + "।")
        if irrig:
            parts.append(f"फसल को लगभग {irrig.strip()} सिंचाइयों की जरूरत पड़ती है।")
        if stage_text:
            parts.append(f"महत्वपूर्ण अवस्थाएं: {stage_text}।")
        interval_match = re.search(r"interval of\s*([0-9]+)\s*to\s*([0-9]+)\s*days", text, flags=re.IGNORECASE)
        if interval_match:
            parts.append(f"इसके बाद सामान्यतः लगभग {interval_match.group(1)}-{interval_match.group(2)} दिन के अंतर पर सिंचाई रखें।")
        every_match = re.search(r"once in\s*([0-9]+)\s*days", text, flags=re.IGNORECASE)
        if every_match:
            parts.append(f"लगभग हर {every_match.group(1)} दिन पर सिंचाई की जा सकती है।")
        if "immediately after sowing" in text.lower():
            parts.append("पहली सिंचाई बुवाई के तुरंत बाद करें।")
        if "life irrigation on third day" in text.lower():
            parts.append("इसके बाद तीसरे दिन life irrigation दें।")
        if "4 – 5th day" in text or "4-5th day" in text.lower():
            parts.append("दूसरी सिंचाई 4-5 दिन बाद दें।")
        if "7 to 8 day" in text.lower() or "7 t o 8 d a y s" in text.lower():
            parts.append("इसके बाद 7-8 दिन के अंतर पर सिंचाई करें।")
        if "2 to 3 cm depth of water" in text.lower():
            parts.append("शुरुआती अवस्था में 2-3 सेमी की हल्की सिंचाई दें, खासकर रेतीली मिट्टी में।")
        if "flowering and pod formation stages are critical" in text.lower():
            parts.append("फूल आने और फली बनने की अवस्था में पानी की कमी न होने दें।")
        if "pegging stage give one or two irrigations" in text.lower():
            parts.append("पेगिंग अवस्था में 1-2 सिंचाई दें।")
        if "pod development stage" in text.lower() and "2 - 3 irri" in text.lower():
            parts.append("फली विकास अवस्था में मिट्टी के अनुसार 2-3 सिंचाई दें।")
        if "water stagnation should be avoided" in text.lower():
            parts.append("अंकुरण के समय खेत में पानी खड़ा न होने दें।")
        if "sprinkle irrigation" in text.lower() or "sprinkler irrigation" in text.lower():
            parts.append("शुरुआती अवस्था या नमी प्रबंधन के लिए sprinkler irrigation उपयोगी हो सकती है।")
        if "drip irrigation" in text.lower():
            parts.append("ड्रिप सिंचाई अपनाने पर पानी की बचत और बेहतर नमी प्रबंधन मिल सकता है।")
        return f"{label}: {' '.join(parts)}" if parts else ""

    if "TOP DRESSING" in heading_key:
        if crop.lower() == "sugarcane":
            return f"{label}: नाइट्रोजन और पोटाश की ऊपरी खाद 30, 60 और 90 दिन पर भागों में दें।"
        timing = _extract_first(r"\((.*?)\)", text)
        if timing:
            return f"{label}: बचा हुआ आधा नाइट्रोजन {_translate_terms(timing)} पर दें।"
        return f"{label}: बचा हुआ आधा नाइट्रोजन शुरुआती बढ़वार पर दें।"

    if "BIOFERTILIZER FOR SUGARCANE" in heading_key:
        return "जैव उर्वरक: Azospirillum, Gluconacetobacter और Phosphobacteria जैसे जैव उर्वरक जड़ों की वृद्धि, nitrogen fixation और nutrient uptake में मदद करते हैं। इन्हें सिफारिश के अनुसार उपयोग करें।"

    if "HARVESTING" in heading_key:
        if crop.lower() == "sugarcane" or "cane" in text.lower():
            parts = []
            if "10 to 11 months" in text.lower():
                parts.append("जल्दी पकने वाली किस्म की कटाई लगभग 10-11 महीने में करें।")
            if "11 to 12 months" in text.lower():
                parts.append("मध्यम अवधि वाली किस्म की कटाई लगभग 11-12 महीने में करें।")
            parts.append("गन्ने की कटाई पूरी परिपक्वता पर करें और cane को जमीन के जितना पास हो सके उतना नीचे से काटें।")
            return f"{label}: {' '.join(parts)}"
        if crop.lower() == "sunflower":
            return f"{label}: केवल flower heads/capitulum काटें। कटाई के बाद flower heads को 2-3 दिन धूप में सुखाएं, फिर मड़ाई करके बीज अलग करें, साफ करें और दोबारा अच्छी तरह सुखाकर संग्रहित करें।"
        return f"{label}: जब दाना सख्त हो जाए और पुआल सूखकर भुरभुरा हो जाए, तब कटाई करें। कटाई के बाद मड़ाई और सफाई कर लें।"

    if "JUDGE WHEN TO HARVEST" in heading_key:
        parts = ["जब फूल के पीछे की पत्तियां नींबू-पीली हो जाएं और फूल का सिरा सख्त हो जाए, तब फसल कटाई के लिए तैयार मानी जाती है।"]
        if "bird damage" in text.lower():
            parts.append("पक्षियों से बचाव के लिए चमकीली पट्टी (ribbon) या अन्य साधन उपयोग करें।")
        return f"{label}: {' '.join(parts)}"

    if "PRE-HARVEST PRACTICES" in heading_key:
        return f"{label}: कटाई से पहले जरूरत होने पर cane ripener का उपयोग सिफारिश के अनुसार करें। उदाहरण: Sodium metasilicate 4 किग्रा/हेक्टेयर को लगभग 750 लीटर पानी में घोलकर 6वें महीने पर छिड़काव करें और जरूरत हो तो 8वें व 10वें महीने पर दोहराएँ।"

    if "CROP PROTECTION" in heading_key:
        if crop.lower() == "sugarcane":
            return "फसल सुरक्षा: गन्ने में shoot borer, termites और grassy shoot disease जैसे प्रमुख कीट/रोग पर नजर रखें। समय पर sett treatment, trash mulching, intercropping और सिफारिश अनुसार insecticide/biocontrol अपनाएँ।"
        return f"{label}: प्रमुख रोग/कीट पर नियमित निगरानी रखें और जरूरत होने पर सिफारिश अनुसार दवा/बीज उपचार अपनाएं।"

    text = _cleanup_bullet_text(_compress_sentences(text, limit=3))
    if not text:
        return ""
    return f"{label}: {text}".strip()


def build_crop_production_guide(question: str) -> tuple[str | None, list[str]]:
    crop = _resolve_crop_name(question)
    if not crop:
        return None, []
    phase_points, _entries, sources = _build_guide_points_for_crop(crop)
    if not phase_points:
        return None, []
    crop_label = _crop_display_label(crop)
    lines = [f"{crop_label} की खेती: स्टेप-बाय-स्टेप गाइड", ""]
    for phase in ["before_sowing", "sowing", "early_growth", "mid_growth", "harvest"]:
        pts = phase_points.get(phase) or []
        if not pts:
            continue
        lines.append(f"{PHASE_LABELS_HI[phase]}:")
        for point in pts:
            lines.append(f"- {point}")
        lines.append("")
    lines.append("अगर आप चाहें, तो मैं इसी फसल के लिए किस्म, खाद, सिंचाई, रोग/कीट या कटाई की और ज्यादा विस्तृत जानकारी भी अलग से बता सकता हूँ।")
    return "\n".join(lines).strip(), sources


def _detect_followup_section(question: str) -> str | None:
    q = str(question or "").lower()
    q_norm = _norm(q)
    q_tokens = re.findall(r"[a-z\u0900-\u097f]+", q)
    for section, keywords in FOLLOWUP_SECTION_KEYWORDS.items():
        for keyword in keywords:
            keyword_text = keyword.lower()
            keyword_norm = _norm(keyword_text)
            if keyword_text in q or (keyword_norm and keyword_norm in q_norm):
                return section
            keyword_tokens = re.findall(r"[a-z\u0900-\u097f]+", keyword_text)
            for token in keyword_tokens:
                if token in q_tokens:
                    return section
                if len(token) >= 4 and get_close_matches(token, q_tokens, n=1, cutoff=0.82):
                    return section
        if any(keyword.lower() in q for keyword in keywords):
            return section
    return None


def _section_matches_followup(section: str, entry: dict[str, str]) -> bool:
    heading_key = _heading_key(entry.get("heading", ""))
    label = _heading_label_hi(entry.get("heading", ""))
    point = str(entry.get("point", ""))
    blob = f"{heading_key} {label} {point}".lower()
    heading_blob = f"{heading_key} {label}".lower()
    point_blob = point.lower()
    if section == "variety":
        return any(token in heading_blob for token in ["season and variety", "season and varieties", "district/season varieties", "बीज दर", "seed rate", "variety"])
    if section == "fertilizer":
        return any(token in heading_blob for token in ["application of fertilizers", "application of micronutrients", "top dressing", "biofertilizer", "compost", "fym", "sulphur", "boric acid"]) or any(token in point_blob for token in ["ऊपरी खाद", "जैव उर्वरक", "गोबर की खाद", "कम्पोस्ट"])
    if section == "irrigation":
        return any(token in heading_blob for token in ["सिंचाई", "water management", "irrigation"])
    if section == "crop_protection":
        return any(token in heading_blob for token in ["फसल सुरक्षा", "crop protection", "plant protection"]) or any(token in point_blob for token in ["रोग", "कीट", "pest", "disease", "fungus", "fungal", "फफूंद"])
    if section == "harvest":
        return any(token in heading_blob for token in ["कटाई", "harvest", "pre-harvest", "maturity"])
    if section == "field_preparation":
        return any(token in heading_blob for token in ["खेत की तैयारी", "field preparation", "farm land preparation", "land preparation"]) or any(token in point_blob for token in ["मेड़", "नालियां"])
    if section == "sowing":
        return any(token in heading_blob for token in ["बुवाई", "रोपाई", "sowing", "planting", "transplanting", "seed treatment", "spacing", "preparation of setts", "forming ridges", "forming beds"])
    if section == "planting_method":
        return any(token in heading_blob for token in ["रोपाई", "planting", "preparation of setts", "forming ridges", "forming beds"])
    return False


@lru_cache(maxsize=1)
def _guide_embedding_model() -> str:
    try:
        cfg = load_config()
        model_name = str(cfg.embedding_model or "").strip()
        if model_name:
            return model_name
    except Exception:
        pass
    return DEFAULT_GUIDE_EMBEDDING_MODEL


@lru_cache(maxsize=1)
def _guide_generator_model() -> str:
    try:
        cfg = load_config()
        model_name = str(cfg.generator_model or "").strip()
        if model_name:
            return model_name
    except Exception:
        pass
    return DEFAULT_GUIDE_GENERATOR_MODEL


@lru_cache(maxsize=1)
def _guide_embedder() -> Embedder | None:
    try:
        return Embedder(_guide_embedding_model())
    except Exception:
        return None


@lru_cache(maxsize=1)
def _guide_translation_generator() -> LocalGenerator | None:
    enabled = os.getenv("KISAANAI_ENABLE_LLM_PDF_TRANSLATION", "").strip().lower() in {"1", "true", "yes"}
    if not enabled:
        return None
    try:
        return LocalGenerator(_guide_generator_model())
    except Exception:
        return None


@lru_cache(maxsize=1)
def _guide_chunk_size() -> int:
    try:
        cfg = load_config()
        return max(4, min(8, int(cfg.chunk_size) // 120))
    except Exception:
        return 6


@lru_cache(maxsize=1)
def _supplemental_pdf_catalog() -> list[Path]:
    raw_dir = GUIDE_PDF.parent
    docs: list[Path] = []
    for pdf_path in sorted(raw_dir.glob("*.pdf")):
        if pdf_path.name.startswith("."):
            continue
        try:
            if pdf_path.resolve() == GUIDE_PDF.resolve():
                continue
        except Exception:
            if pdf_path == GUIDE_PDF:
                continue
        docs.append(pdf_path)
    return docs


def _followup_section_title_hi(section: str) -> str:
    return {
        "variety": "किस्म और बीज दर",
        "fertilizer": "खाद और उर्वरक",
        "irrigation": "सिंचाई",
        "crop_protection": "रोग/कीट प्रबंधन",
        "harvest": "कटाई",
        "field_preparation": "खेत की तैयारी",
        "sowing": "बुवाई/रोपाई",
        "planting_method": "रोपाई/विधि",
    }.get(section, "विस्तृत जानकारी")


def _supplemental_query_mode(question: str, section: str) -> str:
    q = str(question or "").lower()
    if any(
        token in q
        for token in [
            "best", "better", "compare", "comparison", "versus", "vs", "profitable",
            "profit", "higher yield", "high yield", "jyada", "behtar", "लाभ",
            "लाभदायक", "बेहतर", "तुलना", "ज्यादा", "उपज",
        ]
    ):
        return "comparison"
    if any(
        token in q
        for token in [
            "how", "kaise", "कैसे", "steps", "step", "vidhi", "विधि", "method",
            "spacing", "distance", "process", "रोपाई",
        ]
    ):
        return "instructional"
    if section in {"sowing", "planting_method", "field_preparation"}:
        return "instructional"
    return "informational"


def _supplemental_pdf_score(pdf_path: Path, crop: str, section: str, question: str) -> float:
    stem = pdf_path.stem.replace("_", " ").replace("-", " ").lower()
    text = _supplemental_pdf_text(str(pdf_path))
    preview = " ".join(text.split()[:1600]).lower()
    blob = f"{stem} {preview}"
    score = 0.0

    merged_aliases = _merged_crop_aliases()
    aliases = [str(alias).lower() for alias in merged_aliases.get(crop, [crop]) if str(alias).strip()]
    alias_hits_name = any(alias in stem for alias in aliases if len(alias) >= 3 or re.search(r"[\u0900-\u097f]", alias))
    alias_hits_text = any(alias in preview for alias in aliases if len(alias) >= 3 or re.search(r"[\u0900-\u097f]", alias))
    if alias_hits_name:
        score += 4.0
    elif alias_hits_text:
        score += 2.0

    section_terms = _section_focus_terms(section)
    if any(term in stem for term in section_terms):
        score += 1.5
    if any(term in preview for term in section_terms):
        score += 0.8

    question_tokens = _pdf_query_tokens(question)
    if question_tokens:
        overlap = len(question_tokens & _pdf_query_tokens(blob))
        score += min(2.0, overlap * 0.15)

    mode = _supplemental_query_mode(question, section)
    if mode == "comparison":
        if any(term in preview for term in ["results", "discussion", "comparison", "yield", "cost", "productivity"]):
            score += 0.6
    elif mode == "instructional":
        if any(term in preview for term in ["methods", "method", "planting", "spacing", "distance", "seed treatment"]):
            score += 0.6
    return score


def _select_supplemental_pdf(question: str, crop: str, section: str) -> Path | None:
    scored: list[tuple[float, Path]] = []
    for pdf_path in _supplemental_pdf_catalog():
        score = _supplemental_pdf_score(pdf_path, crop, section, question)
        if score > 0:
            scored.append((score, pdf_path))
    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored:
        return None
    best_score, best_path = scored[0]
    return best_path if best_score >= 4.0 else None


def _supplemental_query_expansions(question: str, crop: str, section: str, pdf_path: Path) -> list[str]:
    section_text = section.replace("_", " ")
    stem = pdf_path.stem.replace("_", " ").replace("-", " ")
    mode = _supplemental_query_mode(question, section)
    expansions = [
        f"{crop} {question}",
        f"{crop} {section_text} {question}",
        f"{stem} {crop} {question}",
    ]
    if mode == "comparison":
        expansions.append(f"{crop} compare method yield cost benefit profit {question}")
    elif mode == "instructional":
        expansions.append(f"{crop} method steps spacing distance process {question}")
    return [item.strip() for item in expansions if item.strip()]


def _rerank_supplemental_passages(
    passages: list[dict[str, Any]],
    question: str,
    section: str,
    mode: str,
    top_k: int = 4,
) -> list[dict[str, Any]]:
    question_tokens = _pdf_query_tokens(question)
    section_terms = _section_focus_terms(section)
    rescored: list[tuple[float, dict[str, Any]]] = []
    for passage in passages:
        text = str(passage.get("text", ""))
        lower = text.lower()
        section_name = str(passage.get("section", "")).lower()
        overlap = len(question_tokens & _pdf_query_tokens(lower))
        score = float(passage.get("score") or 0.0)
        score += overlap * 0.03
        score += min(0.18, sum(0.03 for term in section_terms if term in lower))
        preview = _passage_prompt_snippet(text)
        if _fragment_is_document_noise(preview) or _looks_like_publication_metadata(preview):
            score -= 0.25
        if mode == "comparison":
            compare_terms = ["yield", "productivity", "germination", "irrigation", "water", "cost", "profit", "benefit", "comparison"]
            compare_hits = sum(1 for term in compare_terms if term in lower)
            if section_name in {"results", "conclusion"}:
                score += 0.14
            if compare_hits:
                score += min(0.18, compare_hits * 0.03)
            else:
                score -= 0.12
            if any(term in lower for term in ["increase", "decrease", "reduced", "higher", "improved"]):
                score += 0.06
        elif mode == "instructional":
            method_terms = ["apply", "plant", "spacing", "distance", "cm", "kg", "treatment", "sowing", "depth", "step"]
            method_hits = sum(1 for term in method_terms if term in lower)
            if section_name in {"methods", "body"}:
                score += 0.14
            if method_hits:
                score += min(0.18, method_hits * 0.03)
            else:
                score -= 0.08
        if re.search(r"\bet al\b|\(\d{4}\)|\b[A-Z][a-z]+\s+[A-Z]\.", text):
            score -= 0.16
        rescored.append((score, {**passage, "score": score}))
    rescored.sort(key=lambda item: item[0], reverse=True)
    ranked: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    seen_snippets: set[str] = set()
    for _, passage in rescored:
        key = (int(passage.get("start_line") or 0), int(passage.get("end_line") or 0))
        if key in seen:
            continue
        snippet_key = _normalize_line_key(_passage_prompt_snippet(str(passage.get("text", ""))))
        if snippet_key and snippet_key in seen_snippets:
            continue
        seen.add(key)
        if snippet_key:
            seen_snippets.add(snippet_key)
        ranked.append(passage)
        if len(ranked) >= top_k:
            break
    return ranked


def _retrieve_supplemental_passages(pdf_path: Path, question: str, crop: str, section: str, k: int = 4) -> list[dict[str, Any]]:
    mode = _supplemental_query_mode(question, section)
    passages = _retrieve_pdf_passages(
        pdf_path,
        question,
        crop=crop,
        section=section.replace("_", " "),
        k=max(k * 2, k),
        extra_queries=_supplemental_query_expansions(question, crop, section, pdf_path),
    )
    return _rerank_supplemental_passages(passages, question=question, section=section, mode=mode, top_k=k)


def _supplemental_passage_summary(passage: dict[str, Any], question: str, crop: str, section: str) -> str:
    text = str(passage.get("text", ""))
    summary = _summarize_pdf_passage(
        text,
        question,
        crop=crop,
        section=section.replace("_", " "),
        sentence_limit=2,
    )
    if summary and not _fragment_is_document_noise(summary):
        return summary
    return _passage_prompt_snippet(text)


def _first_supplemental_match(patterns: list[str], texts: list[str]) -> re.Match[str] | None:
    for text in texts:
        blob = str(text or "")
        if not blob:
            continue
        for pattern in patterns:
            match = re.search(pattern, blob, flags=re.IGNORECASE | re.DOTALL)
            if match:
                return match
    return None


def _extract_supplemental_facts(passages: list[dict[str, Any]]) -> dict[str, Any]:
    merged = "\n".join(str(p.get("text", "")) for p in passages)
    texts = [merged]
    facts: dict[str, Any] = {}

    germination = _first_supplemental_match(
        [
            r"germination\s*(?:from|%)\s*(\d+(?:\.\d+)?)\s*(?:to)?\s*(\d+(?:\.\d+)?)\s*(?:percent|increase|increased)?",
        ],
        texts,
    )
    if germination:
        facts["germination"] = (germination.group(1), germination.group(2))

    yield_ratio = _first_supplemental_match(
        [
            r"(\d+(?:\.\d+)?)\s*to\s*(\d+(?:\.\d+)?)\s*times\s*more(?:\s+of)?\s+.*?yield",
            r"increased\s+in\s+productivity\s+(\d+(?:\.\d+)?)\s*to\s*(\d+(?:\.\d+)?)\s*times",
        ],
        texts,
    )
    if yield_ratio:
        facts["yield_ratio"] = (yield_ratio.group(1), yield_ratio.group(2))

    avg_qha = _first_supplemental_match(
        [
            r"average(?:\s+total)?\s+productivity.*?(\d+(?:\.\d+)?)\s*q/ha",
            r"were\s*(\d+(?:\.\d+)?)\s*q/ha\s*productivity",
        ],
        texts,
    )
    if avg_qha:
        facts["avg_qha"] = avg_qha.group(1)

    higher_yield = _first_supplemental_match(
        [
            r"higher\s+cane\s+yield\s*(\d+(?:\.\d+)?)\s*and\s*(\d+(?:\.\d+)?)\s*tonnes/ha",
        ],
        texts,
    )
    if higher_yield:
        facts["higher_yield_tha"] = (higher_yield.group(1), higher_yield.group(2))

    water_saving = _first_supplemental_match(
        [
            r"save\s*(\d+)\s*%\s*percent.*?irrigation\s*water",
        ],
        texts,
    )
    if water_saving:
        facts["water_saving_pct"] = water_saving.group(1)

    irrigation_cost = _first_supplemental_match(
        [
            r"expenses\s+on\s+irrigation\s*(\d+(?:\.\d+)?)\s*(\d+(?:\.\d+)?)\s*reduced",
        ],
        texts,
    )
    if irrigation_cost:
        facts["irrigation_cost"] = (irrigation_cost.group(1), irrigation_cost.group(2))

    input_cost = _first_supplemental_match(
        [
            r"input\s+rs\.?/ha\s*(\d+(?:\.\d+)?)\s*(\d+(?:\.\d+)?)\s*increased",
        ],
        texts,
    )
    if input_cost:
        facts["input_cost"] = (input_cost.group(1), input_cost.group(2))

    if re.search(r"inter-?cropping.*?less.*?more.*?increased", merged, flags=re.IGNORECASE | re.DOTALL):
        facts["intercropping_better"] = True

    dims = _first_supplemental_match(
        [
            r"(\d+)\s*cm\s*wide\s*and\s*(\d+)\s*cm\s*depth.*?(\d+)\s*cm\s*between\s*two\s*trenches",
        ],
        texts,
    )
    if dims:
        facts["dims"] = (dims.group(1), dims.group(2), dims.group(3))

    setts = _first_supplemental_match(
        [
            r"(\d+)\s*cm\s*distance.*?between\s*two\s*sets.*?(\d+\s*-\s*\d+)\s*cm\s*soil",
        ],
        texts,
    )
    if setts:
        facts["setts"] = (setts.group(1), re.sub(r"\s+", "", setts.group(2)))

    if re.search(r"increased\s+seed\s+rate\s+and\s+fertilizer\s+cost\s+and\s+doses", merged, flags=re.IGNORECASE):
        facts["higher_input_need"] = True
    return facts


def _structured_supplemental_fallback(
    question: str,
    crop: str,
    section: str,
    guide_points: list[str],
    pdf_path: Path,
    passages: list[dict[str, Any]],
) -> str | None:
    mode = _supplemental_query_mode(question, section)
    facts = _extract_supplemental_facts(passages)
    crop_label = _crop_display_label(crop)
    section_hi = _followup_section_title_hi(section)

    if mode == "comparison":
        evidence: list[str] = []
        method_steps: list[str] = []
        cautions: list[str] = []
        benefits: list[str] = []

        if facts.get("germination"):
            before, after = facts["germination"]
            evidence.append(f"अंकुरण लगभग {before}% से बढ़कर {after}% दर्ज हुई।")
            benefits.append("अंकुरण बेहतर")
        if facts.get("yield_ratio"):
            low, high = facts["yield_ratio"]
            evidence.append(f"उपज/उत्पादकता लगभग {low} से {high} गुना अधिक दर्ज हुई।")
            benefits.append("उपज अधिक")
        if facts.get("avg_qha"):
            evidence.append(f"औसत productivity लगभग {facts['avg_qha']} q/ha दर्ज हुई।")
            if "उपज अधिक" not in benefits:
                benefits.append("उपज अधिक")
        if facts.get("higher_yield_tha"):
            low, high = facts["higher_yield_tha"]
            evidence.append(f"कुछ field trials में गन्ने की उपज लगभग {low} और {high} t/ha तक रिपोर्ट हुई।")
            if "उपज अधिक" not in benefits:
                benefits.append("उपज अधिक")
        if facts.get("water_saving_pct"):
            evidence.append(f"सिंचाई पानी की बचत लगभग {facts['water_saving_pct']}% बताई गई है।")
            benefits.append("पानी/सिंचाई खर्च कम")
        if facts.get("irrigation_cost"):
            before, after = facts["irrigation_cost"]
            evidence.append(f"सिंचाई खर्च लगभग ₹{before} से घटकर ₹{after} प्रति हेक्टेयर दिखा।")
            if "पानी/सिंचाई खर्च कम" not in benefits:
                benefits.append("पानी/सिंचाई खर्च कम")
        if facts.get("intercropping_better"):
            evidence.append("अंतरफसल की संभावना भी बेहतर बताई गई।")

        if facts.get("dims"):
            width, depth, distance = facts["dims"]
            method_steps.append(
                f"खांचे/ट्रेंच लगभग {width} सेमी चौड़े, {depth} सेमी गहरे और एक-दूसरे से लगभग {distance} सेमी दूरी पर रखें।"
            )
        if facts.get("setts"):
            spacing, cover = facts["setts"]
            method_steps.append(
                f"2-budded setts को end-to-end/stair-type ढंग से रखें, दो setts के बीच लगभग {spacing} सेमी दूरी रखें और ऊपर {cover} सेमी मिट्टी से ढकें।"
            )
        if len(method_steps) < 2 and guide_points:
            for point in guide_points[:2]:
                clean = _final_phrase_cleanup(str(point).strip())
                if clean and clean not in method_steps:
                    method_steps.append(clean)
                if len(method_steps) >= 2:
                    break

        if facts.get("input_cost"):
            before, after = facts["input_cost"]
            cautions.append(f"शुरुआती input cost लगभग ₹{before} से बढ़कर ₹{after} प्रति हेक्टेयर तक जा सकती है।")
        elif facts.get("higher_input_need"):
            cautions.append("इस विधि में बीज दर, खाद की मात्रा और शुरुआती लागत अधिक लग सकती है।")

        if evidence:
            benefit_text = " और ".join(benefits[:3]) if benefits else "कुछ मुख्य पैरामीटर बेहतर"
            lines = [f"{crop_label} में ज्यादा लाभ वाली {section_hi}:", ""]
            lines.append(f"- निष्कर्ष: उपलब्ध तुलना डेटा के आधार पर यह विधि सामान्य पद्धति की तुलना में बेहतर दिखती है, क्योंकि {benefit_text} दिखा।")
            lines.append("- आधार:")
            for item in evidence[:5]:
                lines.append(f"- {item}")
            if method_steps:
                lines.append("- कैसे करें:")
                for item in method_steps[:3]:
                    lines.append(f"- {item}")
            elif guide_points:
                lines.append("- कैसे करें:")
                for point in guide_points[:2]:
                    lines.append(f"- {point}")
            if cautions:
                lines.append("- ध्यान दें:")
                for item in cautions[:3]:
                    lines.append(f"- {item}")
            lines.append(f"- स्रोत: {pdf_path.name}")
            return "\n".join(lines).strip()

    return None


def _passage_prompt_snippet(text: str) -> str:
    cleaned = _cleanup_bullet_text(str(text or ""))
    cleaned = re.sub(
        r"\b(Abstract|Introduction|Materials(?: and| &) Methods|Results(?: and Discussion)?|Discussion|Conclusion|References)\b[:\-]?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\bTable\s+\d+\b[:\-]?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bKey words?:.*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .;:-")
    return _compress_sentences(cleaned, limit=3) or cleaned


def _fallback_supplemental_answer(
    question: str,
    crop: str,
    section: str,
    guide_points: list[str],
    pdf_path: Path,
    passages: list[dict[str, Any]],
) -> str:
    structured = _structured_supplemental_fallback(question, crop, section, guide_points, pdf_path, passages)
    if structured:
        return structured
    crop_label = _crop_display_label(crop)
    section_hi = _followup_section_title_hi(section)
    mode = _supplemental_query_mode(question, section)
    if mode == "comparison":
        header = f"{crop_label} में {section_hi} के बारे में:"
        lead = "- पूरक दस्तावेज़ से मिले तुलना के मुख्य बिंदु:"
    else:
        header = f"{crop_label} के लिए {section_hi} की जानकारी:"
        lead = "- पूरक दस्तावेज़ से मिले मुख्य बिंदु:"
    lines = [header, "", lead]
    for passage in passages[:4]:
        snippet = _supplemental_passage_summary(passage, question, crop, section)
        if snippet:
            lines.append(f"- {_translate_pdf_summary(snippet)}")
    if guide_points:
        lines.append("- मुख्य guide के सहायक बिंदु:")
        for point in guide_points[:2]:
            lines.append(f"- {point}")
    lines.append(f"- स्रोत: {pdf_path.name}")
    return "\n".join(lines).strip()


def _build_supplemental_pdf_answer(
    question: str,
    crop: str,
    section: str,
    guide_points: list[str],
    reasoning_generator: LocalGenerator | None = None,
) -> tuple[str | None, list[str]]:
    pdf_path = _select_supplemental_pdf(question, crop, section)
    if pdf_path is None:
        return None, []
    passages = _retrieve_supplemental_passages(pdf_path, question, crop, section, k=4)
    if not passages:
        return None, []

    crop_label = _crop_display_label(crop)
    section_hi = _followup_section_title_hi(section)
    mode = _supplemental_query_mode(question, section)
    evidence_blocks = []
    for idx, passage in enumerate(passages[:4], start=1):
        snippet = _supplemental_passage_summary(passage, question, crop, section)
        if snippet:
            evidence_blocks.append(f"[{idx}] {snippet}")
    if not evidence_blocks:
        return None, []

    generator = reasoning_generator
    if generator is not None:
        guide_context = "\n".join(f"- {point}" for point in guide_points[:2]) if guide_points else "नहीं"
        prompt = (
            "You are an agricultural retrieval assistant.\n"
            "Use only the supplied evidence. Do not invent facts.\n"
            "Answer in simple Hindi for farmers.\n"
            "If the question asks which method is better/profitable, compare only the methods or practices mentioned in the evidence.\n"
            "If the question asks how to do something, give clear steps from the evidence.\n"
            "Keep all numbers, units, and product/method names exactly when present.\n"
            "Do not mention chunks, retrieval, ranking, or internal analysis.\n"
            "If evidence is not enough for a firm conclusion, say so clearly.\n\n"
            f"Question: {question}\n"
            f"Crop: {crop}\n"
            f"Topic: {section_hi}\n"
            f"Guide context:\n{guide_context}\n\n"
            f"Evidence:\n" + "\n".join(evidence_blocks) + "\n\n"
            "Return a concise structured answer in Hindi using this style:\n"
            f"{crop_label} ...\n"
            "- निष्कर्ष: ...\n"
            "- आधार:\n"
            "- ...\n"
            "- कैसे करें:\n"
            "- ...\n"
            "- ध्यान दें:\n"
            "- ...\n"
            "Return only the answer.\n"
        )
        try:
            answer = generator.generate(prompt).strip()
            answer = re.sub(r"^\s*(उत्तर|Answer)\s*:\s*", "", answer, flags=re.IGNORECASE).strip()
            answer = _cleanup_bullet_text(answer)
            if answer:
                if _is_english_heavy(answer, crop):
                    answer = _translate_pdf_summary(answer)
                if answer:
                    return _final_phrase_cleanup(answer), [str(pdf_path)]
        except Exception:
            pass

    return _fallback_supplemental_answer(question, crop, section, guide_points, pdf_path, passages), [str(pdf_path)]


@lru_cache(maxsize=8)
def _supplemental_pdf_text(pdf_path_str: str) -> str:
    pdf_path = Path(pdf_path_str)
    if not pdf_path.exists():
        return ""
    try:
        text = read_pdf_text(pdf_path, prefer_docling=True)
        if text.strip():
            return text
    except Exception:
        pass
    gs_bin = shutil.which("gs")
    if gs_bin and pdf_path.exists():
        try:
            result = subprocess.run(
                [
                    gs_bin,
                    "-q",
                    "-dNOPAUSE",
                    "-dBATCH",
                    "-sDEVICE=txtwrite",
                    "-sOutputFile=-",
                    str(pdf_path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout
        except Exception:
            return ""
    return ""


def _generic_pdf_section_name(line: str) -> str | None:
    lower = str(line or "").strip().lower()
    mapping = {
        "abstract": "abstract",
        "introduction": "introduction",
        "materials and methods": "methods",
        "materials & methods": "methods",
        "methodology": "methods",
        "results and discussion": "results",
        "results": "results",
        "discussion": "results",
        "conclusion": "conclusion",
        "references": "references",
    }
    for key, value in mapping.items():
        if lower == key or lower.startswith(f"{key}:"):
            return value
        if re.search(rf"\b{re.escape(key)}\b", lower):
            return value
    return None


def _strip_inline_section_heading(line: str) -> tuple[str, str | None]:
    raw = str(line or "").strip()
    if not raw:
        return "", None
    lower = raw.lower()
    mapping = [
        ("results and discussion", "results"),
        ("materials and methods", "methods"),
        ("materials & methods", "methods"),
        ("methodology", "methods"),
        ("introduction", "introduction"),
        ("abstract", "abstract"),
        ("discussion", "results"),
        ("conclusion", "conclusion"),
        ("references", "references"),
        ("results", "results"),
    ]
    for marker, section in mapping:
        match = re.search(rf"\b{re.escape(marker)}\b[:\-]?\s*", lower)
        if not match:
            continue
        cleaned = (raw[: match.start()] + " " + raw[match.end() :]).strip(" :-")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned, section
    return raw, None


def _looks_like_publication_metadata(line: str) -> bool:
    lower = str(line or "").lower()
    if re.search(r"\b(received|acceptance|accepted|published|copyright|issn|doi)\b", lower):
        return True
    if re.search(r"\bvol\.?\b", lower) and re.search(r"\bno\.?\b", lower):
        return True
    if "journal" in lower and len(lower) < 140:
        return True
    if re.search(r"\b(key words?|keywords?|associate director|deputy cane commissioner|deputy general manager|krishi vigyan kendra|kvk)\b", lower):
        return True
    if re.search(r"\b(univ\.|university|department|factory)\b", lower) and len(lower) < 140:
        return True
    if re.fullmatch(r"[A-Z0-9 .,&()/:-]{8,}", str(line or "")) and len(str(line or "")) < 110:
        return True
    return False


def _looks_like_tabular_noise(line: str) -> bool:
    text = str(line or "")
    number_tokens = re.findall(r"\b\d+(?:\.\d+)?\b", text)
    code_tokens = re.findall(r"\b[A-Za-z]{1,8}-?\d{2,8}\b", text)
    title_tokens = re.findall(r"\b[A-Z][a-z]+\b", text)
    if len(number_tokens) >= 6:
        return True
    if len(code_tokens) >= 2 and len(number_tokens) >= 2:
        return True
    if len(title_tokens) >= 6 and len(number_tokens) >= 3:
        return True
    return False


@lru_cache(maxsize=8)
def _supplemental_pdf_chunks(pdf_path_str: str) -> list[dict[str, Any]]:
    text = _supplemental_pdf_text(pdf_path_str)
    if not text:
        return []

    lines: list[dict[str, Any]] = []
    current_section = "body"
    for raw_line in text.splitlines():
        line = " ".join(str(raw_line or "").split())
        if not line:
            continue
        line, inline_section = _strip_inline_section_heading(line)
        if not line:
            if inline_section:
                current_section = inline_section
            continue
        section_name = inline_section or _generic_pdf_section_name(line)
        if section_name:
            current_section = section_name
            line, _ = _strip_inline_section_heading(line)
            if not line:
                continue
        if re.fullmatch(r"[_\-. ]{4,}", line):
            continue
        if re.fullmatch(r"[0-9 ]{1,8}", line) or len(line) < 3:
            continue
        if _looks_like_publication_metadata(line):
            continue
        if _looks_like_tabular_noise(line):
            continue
        lines.append({"text": line, "section": current_section})

    if not lines:
        return []

    window_lines = _guide_chunk_size()
    overlap_lines = max(1, window_lines // 3)
    chunks: list[dict[str, Any]] = []
    start = 0
    while start < len(lines):
        end = min(len(lines), start + window_lines)
        parts = lines[start:end]
        if not parts:
            break
        chunk_text = re.sub(r"\s+", " ", " ".join(str(part.get("text", "")) for part in parts)).strip()
        if len(chunk_text) >= 80:
            section_counts: dict[str, int] = {}
            for part in parts:
                section = str(part.get("section") or "front_matter")
                section_counts[section] = section_counts.get(section, 0) + 1
            dominant_section = max(section_counts.items(), key=lambda item: item[1])[0]
            chunks.append(
                {
                    "text": chunk_text,
                    "start_line": start + 1,
                    "end_line": end,
                    "section": dominant_section,
                }
            )
        if end >= len(lines):
            break
        start = max(start + 1, end - overlap_lines)
    return chunks


@lru_cache(maxsize=8)
def _supplemental_pdf_vectors(pdf_path_str: str) -> Any:
    chunks = _supplemental_pdf_chunks(pdf_path_str)
    if not chunks:
        return None
    embedder = _guide_embedder()
    if embedder is None:
        return None
    try:
        return embedder.encode([str(chunk.get("text", "")) for chunk in chunks])
    except Exception:
        return None


def _pdf_query_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z\u0900-\u097f]+", str(text or "").lower())
        if len(token) >= 2
    }


def _semantic_query_variants(
    question: str,
    crop: str = "",
    section: str = "",
    extra_queries: list[str] | None = None,
) -> list[str]:
    base = str(question or "").strip()
    variants: list[str] = []
    for candidate in (
        base,
        " ".join(part for part in [crop, section, base] if part).strip(),
        " ".join(part for part in [crop, base] if part).strip(),
        " ".join(part for part in [section, base] if part).strip(),
    ):
        if candidate and candidate not in variants:
            variants.append(candidate)
    for candidate in extra_queries or []:
        text = str(candidate or "").strip()
        if text and text not in variants:
            variants.append(text)
    return variants


def _retrieve_pdf_passages(
    pdf_path: Path,
    question: str,
    crop: str = "",
    section: str = "",
    k: int = 4,
    extra_queries: list[str] | None = None,
) -> list[dict[str, Any]]:
    chunks = _supplemental_pdf_chunks(str(pdf_path))
    if not chunks:
        return []

    query_texts = _semantic_query_variants(question, crop=crop, section=section, extra_queries=extra_queries)
    vectors = _supplemental_pdf_vectors(str(pdf_path))
    embedder = _guide_embedder()
    if vectors is not None and embedder is not None:
        try:
            query_vecs = embedder.encode(query_texts)
            scored: list[tuple[float, int]] = []
            for idx in range(len(chunks)):
                score = max(float(vectors[idx] @ qvec) for qvec in query_vecs)
                score += _passage_section_bonus(str(chunks[idx].get("section", "")))
                score += _passage_focus_bonus(str(chunks[idx].get("text", "")), section)
                scored.append((score, idx))
            scored.sort(reverse=True)
            top_n = min(max(k * 2, k), len(chunks))
            return [{**chunks[idx], "score": score} for score, idx in scored[:top_n]]
        except Exception:
            pass

    query_tokens: set[str] = set()
    for query_text in query_texts:
        query_tokens.update(_pdf_query_tokens(query_text))
    scored: list[tuple[int, int, int]] = []
    for idx, chunk in enumerate(chunks):
        chunk = chunks[idx]
        chunk_text = str(chunk.get("text", ""))
        chunk_tokens = _pdf_query_tokens(chunk_text)
        overlap = len(query_tokens & chunk_tokens)
        scored.append((overlap, -len(chunk_text), idx))
    scored.sort(reverse=True)
    selected = []
    for overlap, _, idx in scored[: min(max(k * 2, k), len(scored))]:
        if overlap <= 0 and selected:
            continue
        selected.append({**chunks[idx], "score": float(overlap)})
    return selected


def _fragment_is_document_noise(fragment: str) -> bool:
    frag = str(fragment or "").strip()
    lower = frag.lower()
    if len(frag) < 20:
        return True
    if re.search(r"\b(received|acceptance|accepted|published|copyright|issn|doi|references|key words?|keywords?|abstract|agropedia)\b", lower):
        return True
    if re.search(r"\b(associate director|journal|university|kvk|department|factory)\b", lower):
        return True
    if re.search(r"\bet al\b", lower):
        return True
    letters = len(re.findall(r"[A-Za-z\u0900-\u097f]", frag))
    digits = len(re.findall(r"[0-9]", frag))
    if letters == 0:
        return True
    if digits / max(1, letters + digits) > 0.55:
        return True
    if _looks_like_tabular_noise(frag):
        return True
    if re.search(r"\([12][0-9]{3}\)", frag) and len(re.findall(r"\b[A-Z][a-z]+\b", frag)) >= 3:
        return True
    if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z](?:\.[A-Z])?\.?){1,3}", frag):
        return True
    return False


def _section_focus_terms(section: str) -> set[str]:
    raw_key = str(section or "").strip().lower().replace(" ", "_")
    key = {
        "planting_method": "planting_method",
        "planting method": "planting_method",
    }.get(raw_key, raw_key)
    terms = {
        term.strip().lower()
        for term in FOLLOWUP_SECTION_KEYWORDS.get(key, [])
        if term and (len(term.strip()) >= 3 or re.search(r"[\u0900-\u097f]", term))
    }
    for token in re.findall(r"[a-z\u0900-\u097f]+", str(section or "").lower()):
        if len(token) >= 3:
            terms.add(token)
    return terms


def _passage_section_bonus(section_name: str) -> float:
    bonuses = {
        "results": 0.18,
        "conclusion": 0.14,
        "methods": 0.08,
        "body": 0.04,
        "abstract": -0.04,
        "introduction": -0.06,
        "references": -0.25,
    }
    return bonuses.get(str(section_name or "").strip().lower(), 0.0)


def _passage_focus_bonus(text: str, section: str) -> float:
    lower = str(text or "").lower()
    focus_terms = _section_focus_terms(section)
    if not focus_terms:
        return 0.0
    hits = sum(1 for term in focus_terms if term in lower)
    noise_hits = sum(
        1
        for term in ("journal", "references", "key words", "keywords", "associate director", "agropedia")
        if term in lower
    )
    return min(0.16, hits * 0.03) - min(0.18, noise_hits * 0.09)


def _is_method_comparison_query(question: str) -> bool:
    q = str(question or "").lower()
    return any(
        token in q
        for token in [
            "best", "better", "behtar", "बेहतर", "ज्यादा उपज", "अधिक उपज",
            "jyada upaj", "high yield", "higher yield", "profitable", "profit",
            "लाभदायक", "लाभ", "laabh", "labh", "laabhdayak", "labhdayak",
            "fayda", "faayda", "फायदा", "कमाई", "income", "return", "returns",
        ]
    )


def _is_method_profit_query(question: str) -> bool:
    q = str(question or "").lower()
    return any(
        token in q
        for token in [
            "profitable", "profit", "लाभ", "लाभदायक", "laabh", "labh",
            "laabhdayak", "labhdayak", "fayda", "faayda", "फायदा",
            "कमाई", "income", "return", "returns",
        ]
    )


def _summarize_pdf_passage(text: str, question: str, crop: str = "", section: str = "", sentence_limit: int = 2) -> str:
    cleaned = _cleanup_bullet_text(text)
    cleaned = re.sub(r"\b(Abstract|Introduction|Materials(?: and| &) Methods|Results(?: and Discussion)?|Discussion|Conclusion|References)\b[:\-]?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bKey words?:.*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bTable\s+\d+\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bTable\s+\d+\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .;:-")
    fragments = [
        frag.strip(" .;:-")
        for frag in re.split(r"(?<=[.?!])\s+|(?<=:)\s+", cleaned)
        if len(frag.strip(" .;:-")) >= 20
    ]
    summary = ""
    embedder = _guide_embedder()
    if embedder is not None and fragments:
        try:
            query_text = " ".join(_semantic_query_variants(question, crop=crop, section=section)).strip()
            query_vec = embedder.encode([query_text])[0]
            fragment_vecs = embedder.encode(fragments)
            scored_fragments: list[tuple[float, int]] = []
            for idx, fragment in enumerate(fragments):
                if _fragment_is_document_noise(fragment):
                    continue
                score = float(fragment_vecs[idx] @ query_vec)
                score += _passage_focus_bonus(fragment, section)
                scored_fragments.append((score, idx))
            selected = sorted(scored_fragments, reverse=True)[: max(1, sentence_limit)]
            if selected:
                summary = ". ".join(fragments[idx] for _, idx in selected)
        except Exception:
            summary = ""
    if not summary:
        summary = _compress_sentences(cleaned, limit=sentence_limit) or cleaned
    return _final_phrase_cleanup(_translate_pdf_summary(summary))


def _translate_pdf_summary(text: str) -> str:
    source_text = str(text or "").strip()
    generator = _guide_translation_generator()
    if source_text and generator is not None:
        try:
            prompt = (
                "Translate the following agricultural advisory text into simple Hindi for farmers.\n"
                "Rules:\n"
                "- Keep all numbers, percentages, units, and technical codes unchanged.\n"
                "- Keep crop names and abbreviations like SSNM unchanged when needed.\n"
                "- Use natural Hindi, not word-for-word English.\n"
                "- Return only the translated Hindi text.\n\n"
                f"Text: {source_text}\n\nHindi:"
            )
            translated = generator.generate(prompt).strip()
            translated = re.sub(r"^\s*Hindi\s*:\s*", "", translated, flags=re.IGNORECASE).strip()
            translated = _cleanup_bullet_text(translated)
            if translated and not _is_english_heavy(translated, crop="Sugarcane"):
                return _final_phrase_cleanup(translated)
        except Exception:
            pass
    return _cleanup_bullet_text(source_text)


def _is_valid_pdf_summary(summary: str) -> bool:
    text = str(summary or "").strip()
    return len(text) >= 35 and not _fragment_is_document_noise(text)


def build_crop_production_followup(
    question: str,
    crop_hint: str | None = None,
    reasoning_generator: LocalGenerator | None = None,
) -> tuple[str | None, list[str]]:
    crop = crop_hint
    if not crop:
        try:
            crop = _resolve_crop_name(question)
        except ImportError:
            return None, []
    if not crop:
        return None, []
    section = _detect_followup_section(question)
    if not section:
        return None, []
    phase_points, entries, sources = _build_guide_points_for_crop(crop)
    if not phase_points:
        return None, []

    matched_points: list[str] = []
    for entry in entries:
        if _section_matches_followup(section, entry):
            point = _final_phrase_cleanup(str(entry.get("point", "")).strip())
            if point and point not in matched_points:
                matched_points.append(point)

    if not matched_points:
        fallback_phase = {
            "variety": "",
            "fertilizer": "mid_growth",
            "irrigation": "",
            "crop_protection": "mid_growth",
            "harvest": "harvest",
            "field_preparation": "before_sowing",
            "sowing": "sowing",
            "planting_method": "sowing",
        }.get(section, "")
        if fallback_phase:
            matched_points = [p for p in (phase_points.get(fallback_phase) or []) if p]

    supplemental_answer, supplemental_sources = _build_supplemental_pdf_answer(
        question,
        crop,
        section,
        matched_points,
        reasoning_generator=reasoning_generator,
    )
    if supplemental_answer:
        return supplemental_answer, supplemental_sources or sources

    if not matched_points:
        section_hi = _followup_section_title_hi(section)
        crop_label = _crop_display_label(crop)
        note = {
            "irrigation": "इस guide के उपलब्ध हिस्से में सिंचाई का अलग और साफ section नहीं मिला।",
            "variety": "इस guide के उपलब्ध हिस्से में किस्म का अलग section साफ नहीं मिला।",
            "fertilizer": "इस guide के उपलब्ध हिस्से में खाद/उर्वरक का साफ section नहीं मिला।",
        }.get(section, "इस guide के उपलब्ध हिस्से में इस विषय की साफ पंक्ति नहीं मिली।")
        return f"{crop_label} के लिए {section_hi} की जानकारी:\n\n- {note}", sources

    crop_label = _crop_display_label(crop)
    section_hi = _followup_section_title_hi(section)

    lines = [f"{crop_label} के लिए {section_hi} की जानकारी:", ""]
    if section == "irrigation":
        irrigation_blob = " ".join(
            str(entry.get("body", "")).strip()
            for entry in entries
            if _section_matches_followup("irrigation", entry)
        )
        age_note = _age_specific_irrigation_note(question, irrigation_blob)
        if age_note:
            lines.append(f"- {age_note}")
    for point in matched_points[:5]:
        lines.append(f"- {point}")
    if section == "crop_protection":
        lines.append("")
        lines.append("अगर आप चाहें, तो exact रोग/कीट या लक्षण लिखें; फिर मैं दवा, dose और PHI और ज्यादा साफ बता दूँगा।")
    return "\n".join(lines).strip(), sources
