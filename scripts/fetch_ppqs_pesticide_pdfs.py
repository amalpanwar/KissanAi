from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen


PDF_URLS = [
    "https://ppqs.gov.in/sites/default/files/mup_insecticide_03.10.2025.pdf",
    "https://ppqs.gov.in/sites/default/files/mup_fungicide_03.10.2025.pdf",
    "https://ppqs.gov.in/sites/default/files/mup_bio_pesticides_fungicide_03.10.2025.pdf",
    "https://ppqs.gov.in/sites/default/files/mup_herbicide_03.10.2025.pdf",
    "https://ppqs.gov.in/sites/default/files/mup_plant_growth_regulator_03.10.2025.pdf",
    "https://ppqs.gov.in/sites/default/files/mup_bio_insecticides_03.10.2025.pdf",
]


def _filename_from_url(url: str) -> str:
    name = Path(urlparse(url).path).name
    return name or "ppqs_document.pdf"


def download(url: str, out_dir: Path, timeout: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / _filename_from_url(url)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    with urlopen(url, timeout=timeout) as resp:
        data = resp.read()
    out_path.write_bytes(data)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data/raw/ppqs_pesticides")
    parser.add_argument("--timeout", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    downloaded = []
    for url in PDF_URLS:
        try:
            path = download(url, out_dir, args.timeout)
            downloaded.append(path)
            print(f"Downloaded: {path}")
        except Exception as exc:
            print(f"Failed: {url} -> {exc}", file=sys.stderr)

    if not downloaded:
        print("No files downloaded.", file=sys.stderr)
        return 1
    print(f"Total downloaded: {len(downloaded)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
