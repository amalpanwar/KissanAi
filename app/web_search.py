from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


GOOGLE_SEARCH_ENDPOINT = "https://customsearch.googleapis.com/customsearch/v1"
TAVILY_SEARCH_ENDPOINT = "https://api.tavily.com/search"


@dataclass
class WebSearchResult:
    title: str
    snippet: str
    link: str
    source: str


def is_tavily_search_configured() -> bool:
    return bool(os.getenv("TAVILY_API_KEY"))


def is_google_search_configured() -> bool:
    return bool(os.getenv("GOOGLE_SEARCH_API_KEY") and os.getenv("GOOGLE_SEARCH_CX"))


def is_web_search_configured() -> bool:
    return is_tavily_search_configured() or is_google_search_configured()


def tavily_search(query: str, *, num: int = 5, include_domains: list[str] | None = None) -> list[WebSearchResult]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key or not query.strip():
        return []
    body: dict[str, Any] = {
        "query": query.strip(),
        "topic": "general",
        "search_depth": "basic",
        "max_results": max(1, min(int(num), 10)),
        "include_answer": False,
        "include_raw_content": False,
    }
    domains = [d.strip() for d in (include_domains or []) if str(d).strip()]
    if domains:
        body["include_domains"] = domains[:20]
    req = Request(
        TAVILY_SEARCH_ENDPOINT,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "User-Agent": "KisaanAI/1.0",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=20) as resp:
            payload: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []
    items = payload.get("results") or []
    out: list[WebSearchResult] = []
    for item in items:
        title = str(item.get("title") or "").strip()
        snippet = str(item.get("content") or item.get("snippet") or "").strip()
        link = str(item.get("url") or "").strip()
        source = ""
        if link:
            try:
                source = link.split("//", 1)[-1].split("/", 1)[0]
            except Exception:
                source = ""
        if not (title or snippet or link):
            continue
        out.append(WebSearchResult(title=title, snippet=snippet, link=link, source=source))
    return out


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


def web_search(query: str, *, num: int = 5, include_domains: list[str] | None = None) -> list[WebSearchResult]:
    if is_tavily_search_configured():
        results = tavily_search(query, num=num, include_domains=include_domains)
        if results:
            return results
    return google_search(query, num=num)
