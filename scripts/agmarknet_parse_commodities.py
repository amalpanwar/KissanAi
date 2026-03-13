from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def parse_rtf(path: Path) -> dict[str, str]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    raw = re.sub(r"\\[a-zA-Z]+-?\d*", "", raw)
    raw = raw.replace("\\", "")

    pattern = re.compile(r'"id"\s*:\s*(\d+)\s*,\s*"cmdt_name"\s*:\s*"(.*?)"')
    items = pattern.findall(raw)
    seen: dict[str, str] = {}
    for id_, name in items:
        if id_ not in seen:
            seen[id_] = name.strip()
    if "99999" in seen:
        seen.pop("99999", None)
    return seen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="RTF with commodity JSON blocks")
    parser.add_argument("--out_ids", default="data/raw/agmarknet_commodity_ids.txt")
    parser.add_argument("--out_csv", default="data/raw/agmarknet_commodities.csv")
    args = parser.parse_args()

    data = parse_rtf(Path(args.input))
    out_ids = Path(args.out_ids)
    out_ids.parent.mkdir(parents=True, exist_ok=True)
    with out_ids.open("w", encoding="utf-8") as f:
        for id_ in sorted(data.keys(), key=lambda x: int(x)):
            f.write(id_ + "\n")

    out_csv = Path(args.out_csv)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["commodity_id", "commodity_name"])
        for id_ in sorted(data.keys(), key=lambda x: int(x)):
            w.writerow([id_, data[id_]])

    print(f"Parsed {len(data)} commodities")
    print(f"IDs -> {out_ids}")
    print(f"CSV -> {out_csv}")


if __name__ == "__main__":
    main()
