from __future__ import annotations

from datetime import datetime
import json
import re
from pathlib import Path
import tempfile
from urllib.request import urlopen
from urllib.parse import quote

from pypdf import PdfReader


BASE_URL = "https://cacp.da.gov.in/"
SUGARCANE_REPORTS_URL = "https://cacp.da.gov.in/Home/sugarcanereports"
REPORT_PAGE_URLS = {
    "sugarcane": SUGARCANE_REPORTS_URL,
    "kharif": "https://cacp.da.gov.in/Home/kharifreports",
    "rabi": "https://cacp.da.gov.in/Home/rabireports",
    "jute": "https://cacp.da.gov.in/Home/jutereports",
    "copra": "https://cacp.da.gov.in/Home/coprareports",
}
REPORT_CROP_NAMES = {
    "kharif": [
        "Paddy", "Jowar", "Bajra", "Maize", "Ragi", "Arhar (Tur)", "Moong", "Urad",
        "Groundnut", "Soybean", "Sunflower", "Sesamum", "Nigerseed", "Cotton",
    ],
    "rabi": [
        "Wheat", "Barley", "Gram", "Masur", "Rapeseed & Mustard", "Safflower",
    ],
    "jute": ["Jute"],
    "copra": ["Copra"],
}
QUESTIONNAIRE_TOKENS = ("viewquestionare", "questionnaire", "questionare", "annexure")


def _current_and_previous_labels(report_kind: str, year: int) -> list[str]:
    if report_kind == "copra":
        return [str(year), str(year - 1)]
    return [f"{year}-{str(year + 1)[-2:]}", f"{year-1}-{str(year)[-2:]}"]


def _extract_report_rows(html: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for tr in re.findall(r"<tr>(.*?)</tr>", html, flags=re.IGNORECASE | re.DOTALL):
        cols = re.findall(r"<td[^>]*>(.*?)</td>", tr, flags=re.IGNORECASE | re.DOTALL)
        if len(cols) < 3:
            continue
        desc = re.sub(r"<[^>]+>", " ", cols[1])
        desc = re.sub(r"\s+", " ", desc).strip()
        links: dict[str, str] = {}
        for href, label in re.findall(r'<a[^>]+href="([^"]+\.pdf)"[^>]*>.*?</i>\s*([^<]+)\s*</a>', cols[2], flags=re.IGNORECASE | re.DOTALL):
            links[label.strip().lower()] = href
        if desc and links:
            rows.append({"description": desc, "links": links})
    return rows


def _select_links_from_rows(html: str, report_kind: str, year: int) -> dict[str, str | None]:
    rows = _extract_report_rows(html)
    family_labels = {
        "kharif": "kharif",
        "rabi": "rabi",
        "jute": "jute",
        "copra": "copra",
        "sugarcane": "sugarcane",
    }
    family = family_labels.get(report_kind, report_kind).lower()
    labels = _current_and_previous_labels(report_kind, year)

    def pick(lang: str) -> str | None:
        for season in labels:
            for row in rows:
                desc = str(row["description"]).lower()
                links = row["links"]
                if family in desc and season.lower() in desc and lang in links:
                    return str(links[lang])
        return None

    return {"english": pick("english"), "hindi": pick("hindi")}


def _season_key_from_text(text: str) -> tuple[int, int]:
    seasons = re.findall(r"(20\d{2})\s*-\s*(\d{2,4})", text)
    best = (0, 0)
    for start, end in seasons:
        s = int(start)
        e = int(str(end)[-2:])
        cand = (s, e)
        if cand > best:
            best = cand
    return best


def _find_latest_sugarcane_report_url() -> str | None:
    try:
        with urlopen(SUGARCANE_REPORTS_URL, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return None
    # First English PDF link
    m = re.search(r"Document/EnglishReports/Sugarcane_[^\"']+\.pdf", html)
    if not m:
        return None
    return BASE_URL + m.group(0)


def _normalize_pdf_url(url: str) -> str:
    if url.startswith(BASE_URL):
        rel = url[len(BASE_URL):]
        return BASE_URL + quote(rel, safe="/%._-()")
    return quote(url, safe=":/%._-()")


def _download(url: str, dest: Path) -> bool:
    try:
        with urlopen(_normalize_pdf_url(url), timeout=30) as resp:
            data = resp.read()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return True
    except Exception:
        return False


def _fetch_html(url: str) -> str | None:
    try:
        with urlopen(url, timeout=20) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return None


def _extract_pdf_links(html: str, subdir: str) -> list[str]:
    return re.findall(rf"Document/{subdir}/[^\"']+\.pdf", html)


def _candidate_report_links(paths: list[str], report_kind: str) -> list[str]:
    if not paths:
        return []
    family_tokens = {
        "sugarcane": ("sugarcane",),
        "kharif": ("kharif",),
        "rabi": ("rabi",),
        "jute": ("jute",),
        "copra": ("copra",),
    }.get(report_kind, (report_kind,))
    filtered = [p for p in paths if not any(tok in p.lower() for tok in QUESTIONNAIRE_TOKENS)]
    preferred = [p for p in filtered if any(tok in p.lower() for tok in family_tokens)]
    ordered = preferred + [p for p in filtered if p not in preferred]
    if not ordered:
        ordered = paths[:]
    return [BASE_URL + p for p in dict.fromkeys(ordered)]


def _pdf_has_report_signature(pdf_path: Path, report_kind: str, language: str) -> bool:
    try:
        reader = PdfReader(str(pdf_path))
    except Exception:
        return False
    text = ""
    for i in range(min(5, len(reader.pages))):
        try:
            text += " " + (reader.pages[i].extract_text() or "")
        except Exception:
            continue
    low = text.lower()
    if any(tok in low for tok in QUESTIONNAIRE_TOKENS):
        return False
    if language == "english":
        signatures = {
            "kharif": ["price policy for kharif", "kharif crops"],
            "rabi": ["price policy for rabi", "rabi crops"],
            "jute": ["price policy for jute", "jute"],
            "copra": ["price policy for copra", "copra"],
            "sugarcane": ["price policy for sugarcane", "sugar season"],
        }.get(report_kind, [report_kind])
        return any(sig in low for sig in signatures)
    # Hindi PDFs may extract as glyph-like Latin text depending on fonts; for them
    # trust the official page link as long as the file is a readable PDF.
    return len(low.strip()) > 20


def _pdf_preview_text(pdf_path: Path, max_pages: int = 8) -> str:
    try:
        reader = PdfReader(str(pdf_path))
    except Exception:
        return ""
    parts: list[str] = []
    for i in range(min(max_pages, len(reader.pages))):
        try:
            parts.append(reader.pages[i].extract_text() or "")
        except Exception:
            continue
    return "\n".join(parts)


def _download_validated_report(urls: list[str], dest: Path, report_kind: str, language: str) -> tuple[str | None, Path | None]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in urls:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            if not _download(url, tmp_path):
                tmp_path.unlink(missing_ok=True)
                continue
            if not _pdf_has_report_signature(tmp_path, report_kind, language):
                tmp_path.unlink(missing_ok=True)
                continue
            tmp_path.replace(dest)
            return url, dest
        except Exception:
            tmp_path.unlink(missing_ok=True)
            continue
    return None, None


def fetch_latest_cacp_report_paths(report_kind: str, official_dir: Path | str = "data/raw/official_sources") -> dict[str, str | None]:
    page_url = REPORT_PAGE_URLS.get(report_kind)
    if not page_url:
        return {"english_url": None, "hindi_url": None, "english_path": None, "hindi_path": None}
    html = _fetch_html(page_url)
    if not html:
        return {"english_url": None, "hindi_url": None, "english_path": None, "hindi_path": None}

    selected = _select_links_from_rows(html, report_kind, datetime.now().year)
    eng_urls = [BASE_URL + selected["english"]] if selected.get("english") else _candidate_report_links(_extract_pdf_links(html, "EnglishReports"), report_kind)
    hin_urls = [BASE_URL + selected["hindi"]] if selected.get("hindi") else _candidate_report_links(_extract_pdf_links(html, "HindiReports"), report_kind)

    out_dir = Path(official_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eng_path = out_dir / f"cacp_{report_kind}_latest_en.pdf" if eng_urls else None
    hin_path = out_dir / f"cacp_{report_kind}_latest_hi.pdf" if hin_urls else None
    eng_url = None
    hin_url = None
    if eng_path:
        eng_url, resolved_eng_path = _download_validated_report(eng_urls, eng_path, report_kind, "english")
        eng_path = resolved_eng_path
    if hin_path:
        hin_url, resolved_hin_path = _download_validated_report(hin_urls, hin_path, report_kind, "hindi")
        hin_path = resolved_hin_path
    return {
        "english_url": eng_url,
        "hindi_url": hin_url,
        "english_path": str(eng_path) if eng_path and eng_path.exists() else None,
        "hindi_path": str(hin_path) if hin_path and hin_path.exists() else None,
    }


def _clean_text_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        s = re.sub(r"\s+", " ", raw).strip()
        if s:
            lines.append(s)
    return lines


def _parse_generic_cop_rows(lines: list[str], report_kind: str) -> dict[str, dict[str, float]]:
    allowed = set(REPORT_CROP_NAMES.get(report_kind, []))
    if not allowed:
        return {}
    rows: dict[str, dict[str, float]] = {}
    for line in lines:
        m = re.match(r"^([A-Za-z&() /.-]{2,80}?)\s+([0-9,]+)\s+([0-9,]+)\s+([0-9,]+)$", line)
        if not m:
            continue
        crop = re.sub(r"\s+", " ", m.group(1)).strip()
        if crop not in allowed:
            continue
        rows[crop] = {
            "a2_per_qtl": float(m.group(2).replace(",", "")),
            "a2fl_per_qtl": float(m.group(3).replace(",", "")),
            "c2_per_qtl": float(m.group(4).replace(",", "")),
        }
    return rows

def _parse_rabi_projected_rows(text: str) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    crop_patterns = {
        'Wheat': r'Wheat.*?All India\s+[0-9.]+\s+([0-9,]+)\s+([0-9,]+)\s+([0-9,]+)\s+[0-9.]+',
        'Barley': r'Barley.*?All India\s+[0-9.]+\s+([0-9,]+)\s+([0-9,]+)\s+([0-9,]+)\s+[0-9.]+',
        'Gram': r'Gram.*?All India\s+[0-9.]+\s+([0-9,]+)\s+([0-9,]+)\s+([0-9,]+)\s+[0-9.]+',
        'Masur': r'Lentil.*?All India\s+[0-9.]+\s+([0-9,]+)\s+([0-9,]+)\s+([0-9,]+)\s+[0-9.]+',
        'Rapeseed & Mustard': r'Rapeseed\s*&\s*Mustard.*?All India\s+[0-9.]+\s+([0-9,]+)\s+([0-9,]+)\s+([0-9,]+)\s+[0-9.]+',
        'Safflower': r'Safflower.*?All India\s+[0-9.]+\s+([0-9,]+)\s+([0-9,]+)\s+([0-9,]+)\s+[0-9.]+',
    }
    for crop, pattern in crop_patterns.items():
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if not m:
            continue
        rows[crop] = {
            'a2_per_qtl': float(m.group(1).replace(',', '')),
            'a2fl_per_qtl': float(m.group(2).replace(',', '')),
            'c2_per_qtl': float(m.group(3).replace(',', '')),
        }
    return rows


def get_cacp_cost_table(report_kind: str, cache_path: Path | str | None = None) -> dict | None:
    if cache_path is None:
        cache_path = f"data/processed/cacp_{report_kind}_costs.json"
    cache_path = Path(cache_path)
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("rows"):
                return cached
        except Exception:
            pass
    bundle = fetch_latest_cacp_report_paths(report_kind)
    eng_path = bundle.get("english_path")
    if not eng_path:
        return None
    pdf_path = Path(str(eng_path))
    try:
        reader = PdfReader(str(pdf_path))
    except Exception:
        return None
    text_parts: list[str] = []
    for i in range(min(len(reader.pages), 260)):
        try:
            text_parts.append(reader.pages[i].extract_text() or "")
        except Exception:
            continue
    text = "\n".join(text_parts)
    if not text.strip():
        return None
    lines = _clean_text_lines(text)
    rows = _parse_generic_cop_rows(lines, report_kind)
    if not rows and report_kind == 'rabi':
        rows = _parse_rabi_projected_rows(text)
    if not rows:
        return None
    season_match = re.search(r"(20\d{2}-\d{2})", text)
    out = {
        "report_kind": report_kind,
        "season": season_match.group(1) if season_match else "",
        "rows": rows,
        "english_url": bundle.get("english_url"),
        "hindi_url": bundle.get("hindi_url"),
        "english_path": bundle.get("english_path"),
        "hindi_path": bundle.get("hindi_path"),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def get_cacp_cost_for_crop(crop_name: str) -> dict | None:
    crop_map = {
        "rice": ("kharif", "Paddy"),
        "paddy": ("kharif", "Paddy"),
        "maize": ("kharif", "Maize"),
        "jowar": ("kharif", "Jowar"),
        "bajra": ("kharif", "Bajra"),
        "ragi": ("kharif", "Ragi"),
        "pigeon pea": ("kharif", "Arhar (Tur)"),
        "arhar": ("kharif", "Arhar (Tur)"),
        "tur": ("kharif", "Arhar (Tur)"),
        "green gram": ("kharif", "Moong"),
        "moong": ("kharif", "Moong"),
        "black gram": ("kharif", "Urad"),
        "urad": ("kharif", "Urad"),
        "groundnut": ("kharif", "Groundnut"),
        "soybean": ("kharif", "Soybean"),
        "sunflower": ("kharif", "Sunflower"),
        "sesame": ("kharif", "Sesamum"),
        "sesamum": ("kharif", "Sesamum"),
        "nigerseed": ("kharif", "Nigerseed"),
        "cotton": ("kharif", "Cotton"),
        "wheat": ("rabi", "Wheat"),
        "barley": ("rabi", "Barley"),
        "chickpea": ("rabi", "Gram"),
        "gram": ("rabi", "Gram"),
        "lentil": ("rabi", "Masur"),
        "masur": ("rabi", "Masur"),
        "mustard": ("rabi", "Rapeseed & Mustard"),
        "rapeseed": ("rabi", "Rapeseed & Mustard"),
        "safflower": ("rabi", "Safflower"),
        "jute": ("jute", "Jute"),
        "copra": ("copra", "Copra"),
    }
    key = re.sub(r"[^a-z]+", " ", str(crop_name).lower()).strip()
    item = crop_map.get(key)
    if not item:
        return None
    report_kind, row_name = item
    table = get_cacp_cost_table(report_kind)
    if not table:
        return None
    row = (table.get("rows") or {}).get(row_name)
    if not row:
        return None
    out = dict(row)
    out["season"] = table.get("season", "")
    out["report_kind"] = report_kind
    out["crop_name"] = row_name
    out["english_url"] = table.get("english_url")
    out["hindi_url"] = table.get("hindi_url")
    out["source_file"] = table.get("english_path") or table.get("hindi_path")
    return out


def _extract_frp_from_pdf(pdf_path: Path) -> dict | None:
    try:
        reader = PdfReader(str(pdf_path))
    except Exception:
        return None
    # Scan first 30 pages for FRP sentence
    text = ""
    for i in range(min(30, len(reader.pages))):
        try:
            text += reader.pages[i].extract_text() or ""
        except Exception:
            continue
    if not text:
        return None
    # Example: "FRP ... 2025-26 ... 355 per quintal"
    season_match = re.search(r"SUGAR SEASON\s*(20\d{2}-\d{2})", text, re.IGNORECASE)
    season = season_match.group(1) if season_match else ""
    m = re.search(r"FRP[^\n]*?(20\d{2}-\d{2}).*?(\d{3})\s*per\s*q", text, re.IGNORECASE)
    if not m:
        m = re.search(r"FRP[^\n]*?(\d{3})\s*per\s*q", text, re.IGNORECASE)
    if not m:
        return None
    if m.lastindex and m.lastindex >= 2 and re.fullmatch(r"20\d{2}-\d{2}", m.group(1) or ""):
        season = m.group(1)
    price = m.group(2) if m.lastindex and m.lastindex >= 2 else m.group(1)
    return {
        "season": season,
        "price_per_qtl": float(price),
    }


def get_latest_sugarcane_frp(cache_path: Path | str = "data/processed/cacp_frp.json") -> dict | None:
    cache_path = Path(cache_path)
    pdf_path = Path("data/raw/official_sources/cacp_sugarcane_latest.pdf")
    if not pdf_path.exists():
        return None
    data = _extract_frp_from_pdf(pdf_path)
    if not data:
        return None
    data["source_url"] = str(pdf_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def get_sugarcane_cost_snapshot(cache_path: Path | str = "data/processed/cacp_sugarcane_cost.json") -> dict | None:
    cache_path = Path(cache_path)
    pdf_path = Path("data/raw/official_sources/cacp_sugarcane_latest.pdf")
    if not pdf_path.exists():
        return None
    try:
        reader = PdfReader(str(pdf_path))
    except Exception:
        return None

    text_parts = []
    for i in range(min(len(reader.pages), 170)):
        try:
            text_parts.append(reader.pages[i].extract_text() or "")
        except Exception:
            continue
    text = "\n".join(text_parts)
    if not text.strip():
        return None

    out: dict[str, float | str] = {}

    m = re.search(r"SUGAR SEASON\s*(20\d{2}-\d{2})", text, re.IGNORECASE)
    if m:
        out["season"] = m.group(1)

    m = re.search(r"FRP[^\n]{0,220}?([0-9]{3})\s*per\s*quintal", text, re.IGNORECASE)
    if m:
        out["frp_per_qtl"] = float(m.group(1))

    m = re.search(
        r"All-India per quintal CoP A2, A2\+FL and C2 of sugarcane at State-specific recovery rates for sugar\s*season\s*20\d{2}-\d{2}\s*are projected at\s*([0-9]+),\s*([0-9]+) and\s*([0-9]+)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        out["a2_state_specific_per_qtl"] = float(m.group(1))
        out["a2fl_state_specific_per_qtl"] = float(m.group(2))
        out["c2_state_specific_per_qtl"] = float(m.group(3))
    m = re.search(
        r"All-India per quintal CoP A2, A2\+FL and C2 at 10\.25\s*percent basic recovery rate for sugar season\s*20\d{2}-\d{2}\s*are estimated at\s*([0-9]+),\s*([0-9]+) and\s*([0-9]+)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        out["a2_basic_per_qtl"] = float(m.group(1))
        out["a2fl_basic_per_qtl"] = float(m.group(2))
        out["c2_basic_per_qtl"] = float(m.group(3))

    m = re.search(
        r"All-India modified A2, A2\+FL and C2\s*per quintal of sugarcane, inclusive of transportation cost and insurance premium, for sugar season\s*20\d{2}-\d{2}\s*at 10\.25 percent basic sugar recovery rate are estimated at\s*([0-9]+),\s*([0-9]+) and\s*([0-9]+)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        out["modified_a2_per_qtl"] = float(m.group(1))
        out["modified_a2fl_per_qtl"] = float(m.group(2))
        out["modified_c2_per_qtl"] = float(m.group(3))

    m = re.search(
        r"transportation cost and insurance charges for sugarcane have been estimated at\s*([0-9]+)\s*per quintal",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        out["transport_insurance_per_qtl"] = float(m.group(1))

    if not out:
        return None
    season = str(out.get("season") or "")
    out["source_file"] = str(pdf_path)
    out["source_name"] = f"CACP Sugarcane Price Policy {season}".strip()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
