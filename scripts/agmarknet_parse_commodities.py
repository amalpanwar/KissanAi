from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


GROUPS = {
    "Beverages": 9,
    "Cereals": 1,
    "Drug and Narcotics": 11,
    "Dry Fruits": 8,
    "Fibre Crops": 4,
    "Flowers": 14,
    "Forest Products": 12,
    "Fruits": 5,
    "Live Stock,Poultry,Fisheries": 13,
    "Oil Seeds": 3,
    "Oils and Fats": 15,
    "Others": 10,
    "Pulses": 2,
    "Spices": 7,
    "Vegetables": 6,
}


def _clean_rtf(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    raw = re.sub(r"\\[a-zA-Z]+-?\d*", "", raw)
    raw = raw.replace("\\", "")
    return raw


def _extract_items(text: str) -> list[tuple[str, str]]:
    pattern = re.compile(r'"id"\s*:\s*(\d+)\s*,\s*"cmdt_name"\s*:\s*"(.*?)"')
    return [(i, n.strip()) for i, n in pattern.findall(text)]


def parse_rtf(path: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    raw = _clean_rtf(path)
    items = _extract_items(raw)
    seen: dict[str, str] = {}
    for id_, name in items:
        if id_ not in seen:
            seen[id_] = name
    seen.pop("99999", None)

    # Build group -> commodity ids by slicing text between group headings.
    group_ids_map: dict[str, list[str]] = {str(v): [] for v in GROUPS.values()}
    # Find section boundaries
    positions: list[tuple[int, str]] = []
    for name, gid in GROUPS.items():
        m = re.search(rf"{re.escape(name)}\s*:", raw)
        if m:
            positions.append((m.start(), str(gid)))
    positions.sort()

    for idx, (start, gid) in enumerate(positions):
        end = positions[idx + 1][0] if idx + 1 < len(positions) else len(raw)
        section = raw[start:end]
        section_items = _extract_items(section)
        ids = []
        for id_, _ in section_items:
            if id_ != "99999":
                ids.append(id_)
        # dedupe keep order
        seen_ids = []
        for i in ids:
            if i not in seen_ids:
                seen_ids.append(i)
        group_ids_map[gid] = seen_ids

    return seen, group_ids_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="RTF with commodity JSON blocks")
    parser.add_argument("--out_ids", default="data/raw/agmarknet_commodity_ids.txt")
    parser.add_argument("--out_csv", default="data/raw/agmarknet_commodities.csv")
    parser.add_argument("--out_group_json", default="data/raw/agmarknet_group_commodities.json")
    parser.add_argument("--out_group_ids", default="data/raw/agmarknet_group_ids.txt")
    args = parser.parse_args()

    data, group_map = parse_rtf(Path(args.input))
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

    out_group_json = Path(args.out_group_json)
    out_group_json.parent.mkdir(parents=True, exist_ok=True)
    out_group_json.write_text(json.dumps(group_map, indent=2), encoding="utf-8")

    out_group_ids = Path(args.out_group_ids)
    out_group_ids.write_text(
        "\n".join(str(x) for x in sorted(GROUPS.values())),
        encoding="utf-8",
    )

    print(f"Parsed {len(data)} commodities")
    print(f"IDs -> {out_ids}")
    print(f"CSV -> {out_csv}")
    print(f"Group map -> {out_group_json}")
    print(f"Group ids -> {out_group_ids}")


if __name__ == "__main__":
    main()
