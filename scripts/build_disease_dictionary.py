from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class DiseasePattern:
    canonical: str
    english: str
    hindi: str
    patterns: tuple[str, ...]


PATTERNS: tuple[DiseasePattern, ...] = (
    DiseasePattern("powdery mildew", "Powdery mildew", "चूर्णी फफूंदी", (r"\bpowdery\s+mildew\b",)),
    DiseasePattern("downy mildew", "Downy mildew", "डाउनी मिल्ड्यू", (r"\bdowny\s+mildew\b", r"\bdowney\s+mildew\b")),
    DiseasePattern("late blight", "Late blight", "लेट ब्लाइट", (r"\blate\s+blight\b",)),
    DiseasePattern("early blight", "Early blight", "अर्ली ब्लाइट", (r"\bearly\s+blight\b",)),
    DiseasePattern("alternaria blight", "Alternaria blight", "अल्टरनेरिया झुलसा", (r"\balternaria\s+blight\b",)),
    DiseasePattern("alternaria leaf spot", "Alternaria leaf spot", "अल्टरनेरिया पत्ती धब्बा", (r"\balternaria\s+leaf\s+spot\b",)),
    DiseasePattern("leaf blight", "Leaf blight", "पत्ती झुलसा", (r"\bleaf\s+blight\b",)),
    DiseasePattern("leaf spot", "Leaf spot", "पत्ती धब्बा", (r"\bleaf\s*spot\b", r"\bleafspot\b")),
    DiseasePattern("anthracnose", "Anthracnose", "एन्थ्रेक्नोज", (r"\banthracnose\b",)),
    DiseasePattern("fruit rot", "Fruit rot", "फल सड़न", (r"\bfruit\s+rot\b",)),
    DiseasePattern("root rot", "Root rot", "जड़ सड़न", (r"\broot\s+rot\b",)),
    DiseasePattern("sheath blight", "Sheath blight", "शीथ ब्लाइट", (r"\bsheath\s+blight\b",)),
    DiseasePattern("blast", "Blast", "ब्लास्ट", (r"\bblast\b",)),
    DiseasePattern("scab", "Scab", "स्कैब", (r"\bscab\b",)),
    DiseasePattern("white rust", "White rust", "सफेद रतुआ", (r"\bwhite\s+rust\b",)),
    DiseasePattern("yellow rust", "Yellow rust", "पीली रतुआ", (r"\byellow\s+rust\b", r"\bstripe\s+rust\b")),
    DiseasePattern("brown rust", "Brown rust", "भूरी रतुआ", (r"\bbrown\s+rust\b", r"\bleaf\s+rust\b")),
    DiseasePattern("black rust", "Black rust", "काली रतुआ", (r"\bblack\s+rust\b", r"\bstem\s+rust\b")),
    DiseasePattern("rust", "Rust", "रतुआ", (r"\brust\b",)),
    DiseasePattern("purple blotch", "Purple blotch", "पर्पल ब्लॉच", (r"\bpurple\s+blotch\b",)),
    DiseasePattern("loose smut", "Loose smut", "ढीला कंडुआ", (r"\bloose\s+smut\b", r"\bsmut\b")),
    DiseasePattern("karnal bunt", "Karnal bunt", "कर्नाल बंट", (r"\bkarnal\s+bunt\b", r"\bbunt\b")),
    DiseasePattern("wilt", "Wilt", "मुरझान", (r"\bwilt\b",)),
    DiseasePattern("damping off", "Damping off", "डैम्पिंग ऑफ", (r"\bdamping\s+off\b",)),
    DiseasePattern("die back", "Die back", "डाई बैक", (r"\bdie\s*back\b",)),
    DiseasePattern("seedling blight", "Seedling blight", "अंकुर झुलसा", (r"\bseedling\s+blight\b",)),
    DiseasePattern("tikka leaf spot", "Tikka leaf spot", "टिक्का पत्ती धब्बा", (r"\btikka\s+leaf\s+spot\b", r"\btikka\b")),
    DiseasePattern("aphid", "Aphid", "माहू", (r"\baphids?\b",)),
    DiseasePattern("whitefly", "Whitefly", "सफेद मक्खी", (r"\bwhite\s*fly\b", r"\bwhitefly\b")),
    DiseasePattern("thrips", "Thrips", "थ्रिप्स", (r"\bthrips\b",)),
    DiseasePattern("jassid", "Jassid", "जैसिड", (r"\bjassids?\b",)),
    DiseasePattern("red spider mite", "Red spider mite", "लाल मकड़ी किट", (r"\bred\s*spider\s*mite\b", r"\bredspidermite\b")),
    DiseasePattern("yellow mite", "Yellow mite", "पीला माइट", (r"\byellow\s*mite\b", r"\byellowmite\b")),
    DiseasePattern("mite", "Mite", "माइट", (r"\bmites?\b",)),
    DiseasePattern("bollworm", "Bollworm", "बोलवर्म", (r"\bbollworms?\b", r"\bamerican\s*bollworm\b")),
    DiseasePattern("fruit borer", "Fruit borer", "फल छेदक", (r"\bfruit\s*borer\b", r"\bfruitborer\b")),
    DiseasePattern("pod borer", "Pod borer", "फली छेदक", (r"\bpod\s*borer\b", r"\bpodborer\b")),
    DiseasePattern("stem borer", "Stem borer", "तना छेदक", (r"\bstem\s*borer\b", r"\bstemborer\b")),
    DiseasePattern("shoot borer", "Shoot borer", "शूट बोरर", (r"\bshoot\s*borer\b",)),
    DiseasePattern("shoot fly", "Shoot fly", "शूट फ्लाई", (r"\bshoot\s*fly\b", r"\bshootfly\b")),
    DiseasePattern("leaf folder", "Leaf folder", "पत्ती मोड़क", (r"\bleaf\s*folder\b", r"\bleaffolder\b")),
    DiseasePattern("diamondback moth", "Diamondback moth", "डायमंडबैक मॉथ", (r"\bdiamondback\s*moth\b", r"\bdiamondbackmoth\b")),
    DiseasePattern("termite", "Termite", "दीमक", (r"\btermites?\b",)),
    DiseasePattern("nematode", "Nematode", "सूत्रकृमि", (r"\bnematodes?\b",)),
    DiseasePattern("root-knot nematode", "Root-knot nematode", "रूट-नॉट सूत्रकृमि", (r"\broot[\s-]*knot\s+nematodes?\b",)),
    DiseasePattern("hopper", "Hopper", "हॉपर", (r"\bhoppers?\b", r"\bplanthopper\b", r"\bbrownplanthopper\b")),
    DiseasePattern("caterpillar", "Caterpillar", "इल्ली", (r"\bcaterpillar\b",)),
    DiseasePattern("mealybug", "Mealybug", "मिलीबग", (r"\bmealy\s*bug\b", r"\bmealybug\b")),
    DiseasePattern("scale insect", "Scale insect", "स्केल कीट", (r"\bscale\s+insect\b",)),
)

PATTERN_BY_CANONICAL = {pattern.canonical: pattern for pattern in PATTERNS}
DISPLAY_EN = {pattern.canonical: pattern.english for pattern in PATTERNS}
DISPLAY_HI = {pattern.canonical: pattern.hindi for pattern in PATTERNS}


def _clean_label(text: str) -> str:
    cleaned = str(text or "").replace("\xa0", " ").replace("\n", " ").strip()
    replacements = {
        "downey": "downy",
        "fruitborer": "fruit borer",
        "podborer": "pod borer",
        "stemborer": "stem borer",
        "leafspot": "leaf spot",
        "leaffolder": "leaf folder",
        "diamondbackmoth": "diamondback moth",
        "redspidermite": "red spider mite",
        "yellowmite": "yellow mite",
        "brownplanthopper": "brown planthopper",
        "shootfly": "shoot fly",
        "whitefly": "whitefly",
        "americanbollworm": "american bollworm",
    }
    for src, dest in replacements.items():
        cleaned = re.sub(src, dest, cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", cleaned)
    cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
    cleaned = re.sub(r"\s*&\s*", " & ", cleaned)
    cleaned = re.sub(r"\s*/\s*", " / ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"(?:\s|,)+(?:and|or|&)\s*$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip(" ,;/:-")


def _split_components(cleaned: str) -> list[str]:
    parts = re.split(r"\s*(?:,|&|/|\band\b|\bor\b)\s*", cleaned, flags=re.IGNORECASE)
    return [part.strip(" ,;/:-") for part in parts if part.strip(" ,;/:-")]


def _dedupe_specific_terms(canonicals: list[str]) -> list[str]:
    ordered: list[str] = []
    for name in canonicals:
        if name not in ordered:
            ordered.append(name)

    if "white rust" in ordered and "rust" in ordered:
        ordered.remove("rust")
    if any(name in ordered for name in ["yellow rust", "brown rust", "black rust"]) and "rust" in ordered:
        ordered.remove("rust")
    if "alternaria blight" in ordered and "leaf blight" in ordered:
        ordered.remove("leaf blight")
    if "alternaria leaf spot" in ordered and "leaf spot" in ordered:
        ordered.remove("leaf spot")
    if "root-knot nematode" in ordered and "nematode" in ordered:
        ordered.remove("nematode")
    if "red spider mite" in ordered and "mite" in ordered:
        ordered.remove("mite")
    if "yellow mite" in ordered and "mite" in ordered:
        ordered.remove("mite")
    return ordered


def _match_canonicals(cleaned: str) -> list[str]:
    matches: list[str] = []
    search_space = [cleaned] + _split_components(cleaned)
    for piece in search_space:
        lower = piece.lower()
        for pattern in PATTERNS:
            if any(re.search(expr, lower, flags=re.IGNORECASE) for expr in pattern.patterns):
                matches.append(pattern.canonical)
    return _dedupe_specific_terms(matches)


def _is_noise_label(cleaned: str) -> bool:
    lower = cleaned.lower()
    if lower in {"", "-", "na", "n/a", "none"}:
        return True
    noisy_headers = [
        "activeingredient",
        "dose",
        "adultmosquitoes",
        "aerial phase",
        "surface",
    ]
    return any(token in lower.replace(" ", "") for token in noisy_headers)


def _display_labels(canonicals: list[str], cleaned: str) -> tuple[str, str]:
    if not canonicals:
        return cleaned, ""
    english = ", ".join(DISPLAY_EN[name] for name in canonicals)
    hindi = " / ".join(DISPLAY_HI[name] for name in canonicals if DISPLAY_HI.get(name))
    return english, hindi


def build_dictionary(db_path: Path) -> dict:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            """
            SELECT disease_name_en, COUNT(*) AS n
            FROM pesticide_recommendations
            WHERE disease_name_en IS NOT NULL
              AND trim(disease_name_en) <> ''
            GROUP BY disease_name_en
            ORDER BY disease_name_en
            """
        ).fetchall()
    finally:
        conn.close()

    aliases: dict[str, set[str]] = defaultdict(set)
    canonical_counts: dict[str, int] = defaultdict(int)
    raw_label_display: dict[str, dict[str, object]] = {}
    unmatched: list[dict[str, object]] = []

    for raw_label, count in rows:
        cleaned = _clean_label(str(raw_label))
        if _is_noise_label(cleaned):
            continue
        canonicals = _match_canonicals(cleaned)
        english, hindi = _display_labels(canonicals, cleaned)
        raw_label_display[str(raw_label).strip().lower()] = {
            "raw": str(raw_label),
            "clean_english": english,
            "clean_hindi": hindi,
            "canonicals": canonicals,
            "count": int(count),
        }
        if not canonicals:
            unmatched.append({"raw": str(raw_label), "clean_english": cleaned, "count": int(count)})
            continue

        components = _split_components(cleaned)
        for canonical in canonicals:
            aliases[canonical].add(canonical)
            aliases[canonical].add(DISPLAY_EN[canonical])
            aliases[canonical].add(DISPLAY_HI[canonical])
            for piece in components:
                if canonical in _match_canonicals(piece):
                    aliases[canonical].add(piece)
            canonical_counts[canonical] += int(count)

    json_ready_aliases = {
        key: sorted({alias for alias in values if str(alias).strip()}, key=lambda value: (len(str(value)), str(value).lower()))
        for key, values in sorted(aliases.items())
    }
    hindi_terms = {canonical: DISPLAY_HI[canonical] for canonical in json_ready_aliases}
    summary = {
        "raw_distinct_labels": len(rows),
        "canonical_terms": len(json_ready_aliases),
        "matched_raw_labels": sum(1 for entry in raw_label_display.values() if entry["canonicals"]),
        "unmatched_raw_labels": len(unmatched),
    }

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_db": str(db_path),
        "summary": summary,
        "aliases": json_ready_aliases,
        "hindi_terms": hindi_terms,
        "canonical_counts": dict(sorted(canonical_counts.items())),
        "raw_label_display": raw_label_display,
        "unmatched_examples": sorted(unmatched, key=lambda item: (-item["count"], item["clean_english"]))[:200],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a disease/pest dictionary from structured pesticide SQLite data.")
    parser.add_argument("--db", default="data/processed/kisaanai.db")
    parser.add_argument("--output", default="data/processed/disease_dictionary.json")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"SQLite DB not found: {db_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = build_dictionary(db_path)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Saved disease dictionary to {output_path} | "
        f"{data['summary']['canonical_terms']} canonical terms | "
        f"{data['summary']['matched_raw_labels']} matched raw labels"
    )


if __name__ == "__main__":
    main()
