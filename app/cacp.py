from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.request import urlopen

from pypdf import PdfReader


SUGARCANE_REPORTS_URL = "https://cacp.da.gov.in/Home/sugarcanereports"


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
    return "https://cacp.da.gov.in/" + m.group(0)


def _download(url: str, dest: Path) -> bool:
    try:
        with urlopen(url, timeout=20) as resp:
            data = resp.read()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return True
    except Exception:
        return False


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
    m = re.search(r"FRP[^\n]*?(20\d{2}-\d{2}).*?(\d{3})\s*per\s*q", text, re.IGNORECASE)
    if not m:
        m = re.search(r"FRP[^\n]*?(\d{3})\s*per\s*q", text, re.IGNORECASE)
    if not m:
        return None
    season = m.group(1) if m.lastindex and m.lastindex >= 2 else ""
    price = m.group(2) if m.lastindex and m.lastindex >= 2 else m.group(1)
    return {
        "season": season,
        "price_per_qtl": float(price),
    }


def get_latest_sugarcane_frp(cache_path: Path | str = "data/processed/cacp_frp.json") -> dict | None:
    cache_path = Path(cache_path)
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    url = _find_latest_sugarcane_report_url()
    if not url:
        return None
    pdf_path = Path("data/raw/official_sources/cacp_sugarcane_latest.pdf")
    if not pdf_path.exists():
        if not _download(url, pdf_path):
            return None
    data = _extract_frp_from_pdf(pdf_path)
    if not data:
        return None
    data["source_url"] = url
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data
