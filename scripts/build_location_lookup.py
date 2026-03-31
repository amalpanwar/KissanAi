from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def _read_excel(path: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(path, engine="calamine")
    except ImportError:
        # Fallback if calamine isn't installed in the active venv.
        try:
            return pd.read_excel(path, engine="openpyxl")
        except ImportError as exc:
            raise SystemExit(
                "Missing Excel reader engine. Install one in the active venv:\n"
                "  python -m pip install python-calamine\n"
                "or\n"
                "  python -m pip install openpyxl"
            ) from exc


def _find_col(cols: list[str], needle: str) -> str | None:
    for c in cols:
        if needle in c.lower():
            return c
    return None


def _find_any_col(cols: list[str], needles: list[str]) -> str | None:
    for n in needles:
        c = _find_col(cols, n)
        if c:
            return c
    return None


def _parse_hierarchy(text: str) -> tuple[str, str, str]:
    if not text:
        return "", "", ""
    raw = str(text)
    parts = [p.strip() for p in raw.split("/") if p.strip()]
    sub_d, dist, state = "", "", ""
    # Prefer full-string extraction (handles extra slashes/spaces reliably).
    pairs = re.findall(r"([^/]+?)\\s*\\(([^)]+)\\)", raw)
    if not pairs:
        pairs = [(p.split("(")[0].strip(), p.split("(")[-1].rstrip(")")) for p in parts if "(" in p]
    for name, kind in pairs:
        name = name.strip()
        kind_norm = re.sub(r"[^a-z]+", "", kind.lower())
        if "subdistrict" in kind_norm:
            sub_d = name
        elif "district" in kind_norm:
            dist = name
        elif "state" in kind_norm:
            state = name
    # Fallback: if no labeled pairs were found, assume ordered segments.
    if not any([sub_d, dist, state]) and parts:
        if len(parts) >= 1:
            sub_d = parts[0].split("(")[0].strip()
        if len(parts) >= 2:
            dist = parts[1].split("(")[0].strip()
        if len(parts) >= 3:
            state = parts[2].split("(")[0].strip()
    return sub_d, dist, state


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def main() -> None:
    root = Path("Location")
    files = [f for f in sorted(root.glob("*.xlsx")) if not f.name.startswith("~$")]
    if not files:
        raise SystemExit("No .xlsx files found in Location/")

    rows = []
    for f in files:
        df = _read_excel(f)
        df.columns = [str(c).strip() for c in df.columns]
        cols = list(df.columns)
        hierarchy_col = _find_any_col(cols, ["hierarchy"])
        name_col = _find_any_col(
            cols,
            [
                "name",
                "village",
                "town",
                "local body",
                "place",
                "location",
                "urban",
                "rural",
            ],
        )
        sub_col = _find_any_col(cols, ["sub-district", "sub district", "subdistrict", "tehsil", "taluka"])
        dist_col = _find_any_col(cols, ["district"])
        state_col = _find_any_col(cols, ["state"])
        for _, row in df.iterrows():
            hierarchy = str(row.get(hierarchy_col, "")).strip() if hierarchy_col else ""
            if hierarchy:
                sub_d, dist, state = _parse_hierarchy(hierarchy)
            else:
                sub_d = str(row.get(sub_col, "")).strip() if sub_col else ""
                dist = str(row.get(dist_col, "")).strip() if dist_col else ""
                state = str(row.get(state_col, "")).strip() if state_col else ""
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
