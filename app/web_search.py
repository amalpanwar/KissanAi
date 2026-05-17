from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


GOOGLE_SEARCH_ENDPOINT = "https://customsearch.googleapis.com/customsearch/v1"


@dataclass
class WebSearchResult:
    title: str
    snippet: str
    link: str
    source: str


def is_google_search_configured() -> bool:
    return bool(os.getenv("GOOGLE_SEARCH_API_KEY") and os.getenv("GOOGLE_SEARCH_CX"))


def google_search(query: str, *, num: int = 5) -> list[WebSearchResult]:
    api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
    cx = os.getenv("GOOGLE_SEARCH_CX")
    if not api_key or not cx or not query.strip():
        return []
    params = urlencode(
        {
            "key": api_key,
            "cx": cx,
            "q": query.strip(),
            "num": max(1, min(int(num), 10)),
            "safe": "active",
            "hl": "hi",
        }
    )
    req = Request(
        f"{GOOGLE_SEARCH_ENDPOINT}?{params}",
        headers={"User-Agent": "KisaanAI/1.0"},
    )
    try:
        with urlopen(req, timeout=20) as resp:
            payload: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []
    items = payload.get("items") or []
    out: list[WebSearchResult] = []
    for item in items:
        title = str(item.get("title") or "").strip()
        snippet = str(item.get("snippet") or "").strip()
        link = str(item.get("link") or "").strip()
        source = str(item.get("displayLink") or "").strip()
        if not (title or snippet or link):
            continue
        out.append(WebSearchResult(title=title, snippet=snippet, link=link, source=source))
    return out
