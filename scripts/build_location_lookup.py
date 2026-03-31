from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def _find_col(cols: list[str], needle: str) -> str | None:
    for c in cols:
        if needle in c.lower():
            return c
    return None


def _parse_hierarchy(text: str) -> tuple[str, str, str]:
    if not text:
        return "", "", ""
    parts = [p.strip() for p in str(text).split("/") if p.strip()]
    sub_d, dist, state = "", "", ""
    for p in parts:
        m = re.match(r"(.+?)\\s*\\((.+?)\\)", p)
        if not m:
            continue
        name = m.group(1).strip()
        kind = m.group(2).strip().lower()
        if "sub" in kind:
            sub_d = name
        elif "district" in kind:
            dist = name
        elif "state" in kind:
            state = name
    return sub_d, dist, state


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def main() -> None:
    root = Path("Location")
    files = sorted(root.glob("*.xlsx"))
    if not files:
        raise SystemExit("No .xlsx files found in Location/")

    rows = []
    for f in files:
        df = pd.read_excel(f, engine="calamine")
        df.columns = [str(c).strip() for c in df.columns]
        cols = list(df.columns)
        hierarchy_col = _find_col(cols, "hierarchy")
        name_col = _find_col(cols, "name")
        for _, row in df.iterrows():
            hierarchy = str(row.get(hierarchy_col, "")).strip() if hierarchy_col else ""
            sub_d, dist, state = _parse_hierarchy(hierarchy)
            place = ""
            if name_col:
                place = str(row.get(name_col, "")).strip()
            if not place and hierarchy:
                place = hierarchy.split("/")[0].split("(")[0].strip()
            if not place:
                continue
            rows.append(
                {
                    "place": place,
                    "place_norm": _normalize(place),
                    "sub_district": sub_d,
                    "district": dist,
                    "state": state,
                    "source_file": f.name,
                }
            )

    out = pd.DataFrame(rows).drop_duplicates(subset=["place_norm", "district", "state"])
    out_path = Path("data/processed/location_lookup.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} rows to {out_path}")


if __name__ == "__main__":
    main()
