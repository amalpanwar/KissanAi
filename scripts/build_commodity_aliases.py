from __future__ import annotations

import csv
import json
import re
from pathlib import Path


BASE_ALIASES = {
    "wheat": ["गेहूं", "गेहूँ", "गेहू", "gehu", "gehun", "gehoo"],
    "rice": ["धान", "चावल", "chawal"],
    "potato": ["आलू", "aloo"],
    "sugarcane": ["गन्ना", "ganna"],
    "mustard": ["सरसों", "sarso"],
    "maize": ["मक्का", "makka"],
    "onion": ["प्याज", "pyaz"],
    "tomato": ["टमाटर", "tamatar"],
    "sunflower": ["सूरजमुखी", "surajmukhi", "surujmukhi", "soorajmukhi", "suryamukhi"],
    "groundnut": ["मूंगफली", "mungfali", "moongfali", "peanut"],
    "sesamum(sesame,gingelly,til)": ["तिल", "til", "gingelly", "sesame"],
    "cotton": ["कपास", "kapas"],
    "jute": ["जूट"],
    "jowar(sorghum)": ["ज्वार", "jowar", "sorghum"],
    "bajra(pearl millet/cumbu)": ["बाजरा", "bajra", "cumbu", "pearl millet"],
    "ragi(finger millet)": ["रागी", "mandua", "finger millet"],
    "barley(jau)": ["जौ", "jau"],
    "red gram": ["अरहर", "arhar", "tur", "tuar", "pigeon pea"],
    "arhar(tur/red gram)(whole)": ["अरहर", "arhar", "tur", "tuar", "pigeon pea"],
    "black gram(urd beans)(whole)": ["उड़द", "urad", "urd"],
    "green gram(moong)(whole)": ["मूंग", "moong", "mung"],
    "bengal gram(gram)(whole)": ["चना", "chana", "gram", "chickpea"],
    "soyabean": ["सोयाबीन", "soyabean", "soybean"],
    "garlic": ["लहसुन", "lahsun", "lehsun"],
    "chili red": ["लाल मिर्च", "mirch", "lal mirch", "chilli", "chili"],
    "cauliflower": ["फूलगोभी", "phool gobhi", "phulgobhi", "gobhi"],
    "brinjal": ["बैंगन", "baingan", "baigan", "begun", "eggplant"],
    "turmeric": ["हल्दी", "haldi"],
    "cummin seed(jeera)": ["जीरा", "jeera"],
    "coriander(leaves)": ["धनिया", "dhania", "coriander"],
    "methi(leaves)": ["मेथी", "methi"],
    "methi seeds": ["मेथी दाना", "methi dana", "fenugreek"],
    "banana": ["केला", "kela"],
    "mango": ["आम", "aam"],
    "orange": ["संतरा", "santra"],
    "apple": ["सेब", "seb"],
    "grapes": ["अंगूर", "angoor"],
    "pineapple": ["अनानास", "ananas"],
    "safflower": ["kusum", "कुसुम"],
    "fish": ["मछली", "machhli", "machli", "machhi"],
    "egg": ["अंडा", "अंडे", "ande", "andey"],
}


def _normalize_variants(name: str) -> set[str]:
    variants = set()
    if not name:
        return variants
    variants.add(name)
    variants.add(name.lower())
    base_name = re.sub(r"\s*\(.*?\)", "", name).strip()
    if base_name:
        variants.add(base_name)
        variants.add(base_name.lower())
    norm = re.sub(r"[^a-zA-Z0-9]+", " ", base_name).strip()
    if norm:
        variants.add(norm)
        variants.add(norm.lower())
        variants.add(norm.replace(" ", ""))
        variants.add(norm.lower().replace(" ", ""))
    if "&" in name:
        v = name.replace("&", "and")
        variants.add(v)
        variants.add(v.lower())
    return {v for v in variants if v}


def main() -> None:
    csv_path = Path("data/raw/agmarknet_commodities.csv")
    out_path = Path("data/raw/commodity_aliases.json")
    if not csv_path.exists():
        raise SystemExit("Missing data/raw/agmarknet_commodities.csv")

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    aliases: dict[str, list[str]] = {}

    for row in rows:
        name = (row.get("commodity_name") or "").strip()
        if not name:
            continue
        key = name.lower()
        variants = _normalize_variants(name)
        aliases[key] = sorted(variants)

    # Merge base aliases
    for k, vals in BASE_ALIASES.items():
        if k not in aliases:
            aliases[k] = []
        merged = set(aliases[k])
        merged.update(vals)
        aliases[k] = sorted({v for v in merged if v})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(aliases, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(aliases)} commodities to {out_path}")


if __name__ == "__main__":
    main()
