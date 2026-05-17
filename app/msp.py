from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.request import urlopen


RABI_MSP_URL = "https://www.pib.gov.in/PressReleaseIframePage.aspx?PRID=2173566&reg=3&lang=2"
KHARIF_MSP_URL = "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2260618&reg=3&lang=2"
PIB_MSP_URL = RABI_MSP_URL

STATIC_MSP_FALLBACK = {
    "Wheat": 2585,
    "Barley": 2150,
    "Gram": 5875,
    "Masur": 7000,
    "Rapeseed & Mustard": 6200,
    "Safflower": 6540,
    "Paddy (Common)": 2441,
    "Paddy (Grade A)": 2461,
    "Jowar (Hybrid)": 4023,
    "Jowar (Maldandi)": 4073,
    "Bajra": 2900,
    "Ragi": 5205,
    "Maize": 2410,
    "Arhar": 8450,
    "Moong": 8780,
    "Urad": 8200,
    "Groundnut": 7517,
    "Sunflower Seed": 8343,
    "Soyabean Yellow": 5708,
    "Soybean (Yellow)": 5708,
    "Sesamum": 10346,
    "Nigerseed": 10052,
    "Cotton (Medium Staple)": 8267,
    "Cotton (Long Staple)": 8667,
}

STATIC_MSP_SOURCE_URLS = {
    "Wheat": RABI_MSP_URL,
    "Barley": RABI_MSP_URL,
    "Gram": RABI_MSP_URL,
    "Masur": RABI_MSP_URL,
    "Rapeseed & Mustard": RABI_MSP_URL,
    "Safflower": RABI_MSP_URL,
    "Paddy (Common)": KHARIF_MSP_URL,
    "Paddy (Grade A)": KHARIF_MSP_URL,
    "Jowar (Hybrid)": KHARIF_MSP_URL,
    "Jowar (Maldandi)": KHARIF_MSP_URL,
    "Bajra": KHARIF_MSP_URL,
    "Ragi": KHARIF_MSP_URL,
    "Maize": KHARIF_MSP_URL,
    "Arhar": KHARIF_MSP_URL,
    "Moong": KHARIF_MSP_URL,
    "Urad": KHARIF_MSP_URL,
    "Groundnut": KHARIF_MSP_URL,
    "Sunflower Seed": KHARIF_MSP_URL,
    "Soyabean Yellow": KHARIF_MSP_URL,
    "Soybean (Yellow)": KHARIF_MSP_URL,
    "Sesamum": KHARIF_MSP_URL,
    "Nigerseed": KHARIF_MSP_URL,
    "Cotton (Medium Staple)": KHARIF_MSP_URL,
    "Cotton (Long Staple)": KHARIF_MSP_URL,
}


def _fetch_html(url: str) -> str | None:
    try:
        with urlopen(url, timeout=20) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return None


def _strip_tags(html: str) -> list[str]:
    text = re.sub(r"<br\\s*/?>", "\n", html, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "\n", text)
    lines = [re.sub(r"\\s+", " ", l).strip() for l in text.splitlines()]
    return [l for l in lines if l]


def _parse_msp_from_lines(lines: list[str]) -> dict[str, int]:
    crop_names = [
        "Paddy (Common)",
        "Paddy (Grade A)",
        "Jowar (Hybrid)",
        "Jowar (Maldandi)",
        "Bajra",
        "Ragi",
        "Maize",
        "Arhar",
        "Moong",
        "Urad",
        "Cotton (Medium Staple)",
        "Cotton (Long Staple)",
        "Groundnut",
        "Sunflower Seed",
        "Soyabean Yellow",
        "Sesamum",
        "Nigerseed",
        "Wheat",
        "Barley",
        "Gram",
        "Masur",
        "Rapeseed & Mustard",
        "Safflower",
        "Jute",
        "Copra (milling)",
        "Copra (ball)",
    ]
    out: dict[str, int] = {}
    # Normalize lines for matching
    norm_lines = [l.replace("’", "'").replace("–", "-") for l in lines]
    for crop in crop_names:
        for idx, line in enumerate(norm_lines):
            if crop.lower() in line.lower():
                # collect numbers from subsequent lines
                nums: list[int] = []
                for j in range(idx, min(idx + 10, len(norm_lines))):
                    nums.extend([int(n) for n in re.findall(r"\\b\\d{3,5}\\b", norm_lines[j])])
                    if len(nums) >= 3:
                        break
                if nums:
                    out[crop] = nums[-1]
                break
    return out


def get_latest_msp(cache_path: Path | str = "data/processed/msp_cache.json") -> dict | None:
    cache_path = Path(cache_path)
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    html = _fetch_html(PIB_MSP_URL)
    if not html:
        return {
            "source_url": PIB_MSP_URL,
            "source_urls": STATIC_MSP_SOURCE_URLS,
            "msp": STATIC_MSP_FALLBACK,
        }
    lines = _strip_tags(html)
    msp = _parse_msp_from_lines(lines)
    if not msp:
        return {
            "source_url": PIB_MSP_URL,
            "source_urls": STATIC_MSP_SOURCE_URLS,
            "msp": STATIC_MSP_FALLBACK,
        }
    merged_msp = dict(STATIC_MSP_FALLBACK)
    merged_msp.update(msp)
    data = {"source_url": PIB_MSP_URL, "source_urls": STATIC_MSP_SOURCE_URLS, "msp": merged_msp}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def get_msp_for_crop(crop_name: str) -> dict | None:
    data = get_latest_msp()
    if not data:
        return None
    msp_map = data.get("msp", {})
    if not msp_map:
        return None
    # Direct match
    for k, v in msp_map.items():
        if k.lower() == crop_name.lower():
            return {
                "crop": k,
                "msp": v,
                "source_url": (data.get("source_urls", {}) or {}).get(k, data.get("source_url", "")),
            }
    # Partial match
    for k, v in msp_map.items():
        if crop_name.lower() in k.lower() or k.lower() in crop_name.lower():
            return {
                "crop": k,
                "msp": v,
                "source_url": (data.get("source_urls", {}) or {}).get(k, data.get("source_url", "")),
            }
    return None
