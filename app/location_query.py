"""Remove references to the current selection before searching for named places."""
from __future__ import annotations

import re

_RELATIVE_LOCATION = re.compile(
    r"(?<![\w\u0900-\u097f])(?:"
    r"(?:मेरे|मेरी|मेरा|अपने|अपनी|अपना)\s+(?:क्षेत्र|इलाके|इलाका|गांव|गाँव|शहर|स्थान|जिले|जिला|यहाँ|यहां)|"
    r"(?:चुने\s+हुए|चयनित)\s+(?:क्षेत्र|गांव|गाँव|स्थान|जिले|जिला)|"
    r"(?:my|our|selected|current)\s+(?:area|village|town|city|location|district)|"
    r"(?:mere|meri|mera|apne|apni|apna)\s+(?:area|gaon|gaav|ganv|village|shehar|shahar|kshetra|ilake|ilaake|yahan|yaha)|"
    r"here|यहाँ|यहां|यहीं|यहीँ|yahan|yaha"
    r")(?![\w\u0900-\u097f])", re.IGNORECASE,
)


def strip_relative_location(query: str) -> str:
    """Keep explicit names intact: 'weather here in Doghat' still names Doghat."""
    return re.sub(r"\s+", " ", _RELATIVE_LOCATION.sub(" ", str(query or ""))).strip()


def explicit_named_place(query, lookup):
    """Match complete place names; never autocorrect prose into a village."""
    from app.commodity_lookup import normalize
    text = ' ' + normalize(strip_relative_location(query)) + ' '
    candidates = []
    for column in ('place', 'sub_district', 'district'):
        if column not in lookup:
            continue
        for raw in lookup[column].dropna().astype(str).unique():
            name = normalize(raw)
            if not name or name == 'nan':
                continue
            variants = {name}
            # Common official qualifiers are optional, but individual words of
            # a multiword village name are never treated as the whole village.
            short = re.sub(r'\s+(rural|urban)$', '', name)
            if short:
                variants.add(short)
            for variant in variants:
                if len(variant) >= 3 and ' ' + variant + ' ' in text:
                    candidates.append((len(variant), len(name), raw))
    return max(candidates)[2] if candidates else None
