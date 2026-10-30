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
