from __future__ import annotations

import json
import os
import re
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen, Request


ALLOWED_DOMAINS = {
    "fert.nic.in",
    "fert.gov.in",
    "icar.org.in",
    "epubs.icar.org.in",
    "iari.res.in",
    "farmer.gov.in",
    "agricoop.gov.in",
    "up.gov.in",
}


def _is_allowed(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    return any(host.endswith(d) for d in ALLOWED_DOMAINS)


def _download(url: str, out_dir: Path, debug: bool = False) -> Path | None:
    if not _is_allowed(url):
        return None
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "*/*",
            "Referer": "https://www.fert.nic.in/",
        },
    )
    try:
        with urlopen(req, timeout=20) as resp:
            data = resp.read()
    except Exception:
        if debug:
            print(f"Download failed: {url}")
        return None
    name = os.path.basename(urlparse(url).path) or "document.pdf"
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    out_path = out_dir / name
    out_path.write_bytes(data)
    return out_path


def _extract_pdf_links(html: str, base_url: str) -> list[str]:
    links = re.findall(r'href=[\"\\\']([^\"\\\']+\\.pdf)[\"\\\']', html, flags=re.I)
    out = []
    for link in links:
        if link.startswith("http"):
            out.append(link)
        elif link.startswith("/"):
            parsed = urlparse(base_url)
            out.append(f"{parsed.scheme}://{parsed.netloc}{link}")
    return out


def _alt_urls(url: str) -> list[str]:
    if "fert.nic.in" not in url:
        return [url]
    alts = [url]
    alts.append(url.replace("https://www.fert.nic.in", "https://fert.nic.in"))
    alts.append(url.replace("https://fert.nic.in", "https://www.fert.nic.in"))
    alts.append(url.replace("https://", "http://"))
    alts.append(alts[-1].replace("www.fert.nic.in", "fert.nic.in"))
    # dedupe preserve order
    seen = set()
    out = []
    for u in alts:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def main() -> None:
    debug = os.getenv("OFFICIAL_SOURCES_DEBUG", "0") == "1"
    sources_path = Path("data/raw/official_sources.json")
    out_dir = Path("data/raw/official_sources")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not sources_path.exists():
        raise SystemExit("Missing data/raw/official_sources.json")

    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    if not isinstance(sources, list):
        raise SystemExit("Invalid sources file format.")

    downloaded = []
    for item in sources:
        url = item.get("url")
        if not url:
            continue
        if item.get("type") == "pdf" or url.lower().endswith(".pdf"):
            path = None
            for u in _alt_urls(url):
                path = _download(u, out_dir, debug=debug)
                if path:
                    break
            if path:
                downloaded.append(str(path))
            continue

        # If it is a page, fetch and extract PDF links
        if not _is_allowed(url):
            continue
        html = None
        for u in _alt_urls(url):
            req = Request(
                u,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "text/html,*/*",
                    "Referer": "https://www.fert.nic.in/",
                },
            )
            try:
                with urlopen(req, timeout=20) as resp:
                    html = resp.read().decode("utf-8", errors="ignore")
                    break
            except Exception:
                continue
        if html is None:
            if debug:
                print(f"Page fetch failed: {url}")
            continue
        for pdf in _extract_pdf_links(html, url):
            path = _download(pdf, out_dir, debug=debug)
            if path:
                downloaded.append(str(path))

    print(f"Downloaded {len(downloaded)} files into {out_dir}")
    for p in downloaded:
        print(p)


if __name__ == "__main__":
    main()
