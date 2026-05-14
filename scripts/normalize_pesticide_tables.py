from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = []
    seen = {}
    for c in df.columns:
        name = str(c)
        if name in seen:
            seen[name] += 1
            name = f"{name}.{seen[name]}"
        else:
            seen[name] = 0
        cols.append(name)
    df.columns = cols
    return df


def _compose_header_from_rows(df: pd.DataFrame, row_indices: list[int]) -> list[str] | None:
    if df.empty:
        return None
    rows = []
    for i in row_indices:
        if i < 0 or i >= len(df):
            continue
        rows.append([str(x).strip() for x in df.iloc[i].values])
    if not rows:
        return None

    header = []
    num_cols = df.shape[1]
    for col_idx in range(num_cols):
        parts = []
        for r in rows:
            if col_idx >= len(r):
                continue
            val = r[col_idx]
            if not val or val == "nan":
                continue
            if val not in parts:
                parts.append(val)
        header.append(" ".join(parts).strip())
    if all(h == "" for h in header):
        return None
    for i, h in enumerate(header):
        if not h:
            header[i] = f"col_{i+1}"
        else:
            lc = h.lower()
            if ("a. i." in lc or "a.i." in lc or "a.i" in lc) and not ("dosage" in lc or "dose" in lc):
                header[i] = f"Dosage per ha {h}".strip()
            elif "formulation" in lc and not ("dosage" in lc or "dose" in lc):
                header[i] = f"Dosage per ha {h}".strip()
            elif "dilution" in lc and not ("dosage" in lc or "dose" in lc):
                header[i] = f"Dosage per ha {h}".strip()
    return header


def _table_num(path: Path) -> int:
    match = re.search(r"_table_(\d+)\.(?:csv|xlsx)$", path.name)
    return int(match.group(1)) if match else 0


def _source_stem(path: Path) -> str:
    return path.name.split("_table_")[0]


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path, header=None, engine="openpyxl")
    return pd.read_csv(path, header=None)


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".xlsx":
        df.to_excel(path, index=False)
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output file type: {path}")


def _table_files(input_dir: Path) -> list[Path]:
    selected: dict[tuple[str, int], Path] = {}
    for path in input_dir.glob("*_table_*.*"):
        if path.suffix.lower() not in {".xlsx", ".csv"}:
            continue
        key = (_source_stem(path), _table_num(path))
        current = selected.get(key)
        if current is None or (path.suffix.lower() == ".xlsx" and current.suffix.lower() != ".xlsx"):
            selected[key] = path
    return sorted(selected.values(), key=lambda p: (_source_stem(p), _table_num(p)))


def _is_pesticide_heading(value: object) -> bool:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return False
    return bool(
        re.search(
            r"\b\d+(?:\.\d+)?%|\b(?:SC|WP|EC|WDG|SL|FS|GR|WG|OD|CS|ULV|DP|DS|ME|WS)\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def _is_empty(value: object) -> bool:
    return pd.isna(value) or str(value).strip().lower() in {"", "nan", "none"}


def _text(value: object) -> str:
    return "" if _is_empty(value) else str(value).replace("\n", " ").strip()


def _looks_like_dose_value(value: object) -> bool:
    text = _text(value)
    return bool(
        re.search(
            r"\d|kg seed|ml|gm|g/|ltr|lit|ha|%|-",
            text,
            flags=re.IGNORECASE,
        )
    )


def _find_col(columns: pd.Index, predicate) -> object | None:
    for col in columns:
        if predicate(str(col).strip().lower()):
            return col
    return None


def _repair_shifted_disease_rows(df: pd.DataFrame) -> pd.DataFrame:
    crop_col = _find_col(df.columns, lambda c: "crop" in c)
    disease_col = _find_col(df.columns, lambda c: "disease" in c)
    ai_col = _find_col(df.columns, lambda c: "dosage" in c and ("a. i." in c or "a.i." in c))
    formulation_col = _find_col(df.columns, lambda c: "dosage" in c and "formulation" in c)
    dilution_col = _find_col(df.columns, lambda c: "dosage" in c and ("dilution" in c or "water" in c))
    waiting_col = _find_col(df.columns, lambda c: "waiting" in c or "harvest" in c or "phi" in c)
    filler_cols = [col for col in df.columns if re.fullmatch(r"col_\d+", str(col))]

    required = [crop_col, disease_col, ai_col, formulation_col, dilution_col, waiting_col]
    if any(col is None for col in required) or len(filler_cols) < 2:
        return df

    df = df.copy().astype("object")
    first_filler, second_filler = filler_cols[0], filler_cols[1]
    for idx, row in df.iterrows():
        if _is_empty(row.get(disease_col)) and not _is_empty(row.get(ai_col)):
            maybe_disease = row.get(ai_col)
            shifted_has_dose = any(
                _looks_like_dose_value(row.get(col))
                for col in [dilution_col, first_filler, waiting_col, second_filler]
            )
            if not _looks_like_dose_value(maybe_disease) and shifted_has_dose:
                df.at[idx, disease_col] = maybe_disease
                df.at[idx, ai_col] = row.get(dilution_col)
                df.at[idx, formulation_col] = row.get(first_filler)
                df.at[idx, dilution_col] = row.get(waiting_col)
                df.at[idx, waiting_col] = row.get(second_filler)
            elif not _looks_like_dose_value(maybe_disease):
                df.at[idx, disease_col] = maybe_disease
                df.at[idx, ai_col] = pd.NA
    return df


def _merge_disease_continuation_rows(df: pd.DataFrame) -> pd.DataFrame:
    required = {"crop_name", "disease_name_en", "pesticide_name", "ai_g", "formulation", "dilution_water_l", "waiting_period_days"}
    if not required.issubset(df.columns):
        return df

    df = df.copy()
    keep = []
    last_target_idx = None
    last_key = None
    for idx, row in df.iterrows():
        key = (_text(row.get("crop_name")).lower(), _text(row.get("pesticide_name")).lower())
        has_disease = not _is_empty(row.get("disease_name_en"))
        has_dose = any(not _is_empty(row.get(col)) for col in ["ai_g", "formulation", "dilution_water_l"])
        has_waiting = not _is_empty(row.get("waiting_period_days"))
        if has_disease and not has_dose and not has_waiting and last_target_idx is not None and key == last_key:
            current = _text(df.at[last_target_idx, "disease_name_en"])
            fragment = _text(row.get("disease_name_en"))
            if fragment and fragment.lower() not in current.lower():
                df.at[last_target_idx, "disease_name_en"] = f"{current} {fragment}".strip()
            continue
        keep.append(idx)
        if has_dose or has_waiting:
            last_target_idx = idx
            last_key = key
    return df.loc[keep].copy()


def _clean_waiting_periods(series: pd.Series) -> pd.Series:
    values = series.astype("object").tolist()
    cleaned = []
    last_full = ""
    for i, val in enumerate(values):
        text = "" if _is_empty(val) else str(val).replace("\n", " ").strip()
        next_text = ""
        if i + 1 < len(values) and not _is_empty(values[i + 1]):
            next_text = str(values[i + 1]).replace("\n", " ").strip()

        if text.lower() == "required" and last_full:
            text = f"{last_full} required"
        elif text and next_text.lower() == "required" and "required" not in text.lower():
            last_full = text
            text = f"{text} required"
        elif text and "required" not in text.lower():
            last_full = text
        elif text:
            last_full = text.replace(" required", "")
        cleaned.append(text or pd.NA)
    return pd.Series(cleaned, index=series.index)


def normalize_columns(
    df: pd.DataFrame,
    header_override: list[str] | None = None,
    drop_rows: int = 0,
    initial_pesticide: str | None = None,
) -> pd.DataFrame:
    # Apply header override derived from table 1 if present
    if header_override:
        header = header_override[:]
        if len(header) < df.shape[1]:
            for i in range(len(header), df.shape[1]):
                header.append(f"col_{i+1}")
        if len(header) > df.shape[1]:
            while len(header) > df.shape[1]:
                filler_idx = next((i for i, name in enumerate(header) if re.fullmatch(r"col_\d+", str(name))), None)
                if filler_idx is None:
                    header.pop()
                else:
                    header.pop(filler_idx)
        df = df.copy()
        df.columns = header
        df = _dedupe_columns(df)
    else:
        df = _dedupe_columns(df)
        header = _compose_header_from_rows(df, [0, 1])
        if header:
            df = df.copy()
            df.columns = header
            df = _dedupe_columns(df)
        else:
            df = _dedupe_columns(df)

    if drop_rows > 0 and len(df) > drop_rows:
        df = df.iloc[drop_rows:].copy()

    df = _repair_shifted_disease_rows(df)

    # Map columns
    col_map = {}
    for c in df.columns:
        lc = str(c).strip().lower()
        if "crop" in lc:
            col_map[c] = "crop_name"
        elif "disease" in lc or "weed" in lc or "insect" in lc or "pest" in lc:
            col_map[c] = "disease_name_en"
        elif ("a. i." in lc or "a.i." in lc or "a.i" in lc) and ("dosage" in lc or "dose" in lc):
            col_map[c] = "ai_g"
        elif "formulation" in lc and ("dosage" in lc or "dose" in lc):
            col_map[c] = "formulation"
        elif ("dilution" in lc or "water" in lc) and ("dosage" in lc or "dose" in lc):
            col_map[c] = "dilution_water_l"
        elif "waiting" in lc or "harvest" in lc or "phi" in lc:
            col_map[c] = "waiting_period_days"
    df = df.rename(columns=col_map)

    # Identify pesticide name rows (only first column filled)
    if df.columns.size > 0:
        first_col = df.columns[0]
        pesticide_name = []
        current = initial_pesticide or ""
        for _, row in df.iterrows():
            vals = [str(v).strip() for v in row.values if str(v).strip() not in ("", "nan", "None")]
            if len(vals) == 1 and row[first_col] and _is_pesticide_heading(row[first_col]):
                current = str(row[first_col]).strip()
                pesticide_name.append(current)
            else:
                pesticide_name.append(current)
        df["pesticide_name"] = pesticide_name
        df["pesticide_name"] = df["pesticide_name"].replace("", pd.NA).ffill()

    # Forward fill crop names after pesticide headings are detected.
    if "crop_name" in df.columns:
        pesticide_heading = df["crop_name"].apply(_is_pesticide_heading) & df.get("disease_name_en", pd.Series(index=df.index)).apply(_is_empty)
        df.loc[pesticide_heading, "crop_name"] = pd.NA
        df["crop_name"] = (
            df["crop_name"]
            .replace("", pd.NA)
            .replace("nan", pd.NA)
            .ffill()
            .bfill()
        )

    if "waiting_period_days" in df.columns:
        df["waiting_period_days"] = _clean_waiting_periods(df["waiting_period_days"])

    df = _merge_disease_continuation_rows(df)

    # Compose dose_text. Some bio-pesticide/herbicide tables store the complete
    # recommendation in an unmapped treatment column instead of a.i/formulation.
    mapped_output_cols = {
        "crop_name",
        "disease_name_en",
        "pesticide_name",
        "ai_g",
        "formulation",
        "dilution_water_l",
        "waiting_period_days",
    }
    extra_dose_cols = [c for c in df.columns if c not in mapped_output_cols and not str(c).startswith("Unnamed")]

    def _looks_like_recommendation(value: object) -> bool:
        text = _text(value)
        return bool(
            re.search(
                r"\b(?:gm|g/|kg|ml|lit|liter|litre|l/ha|ha|seed|soil|treat|drench|spray|apply|@)\b|\d",
                text,
                flags=re.IGNORECASE,
            )
        )

    def _compose(row):
        parts = []
        for key in ["ai_g", "formulation", "dilution_water_l"]:
            if key in row and str(row[key]).strip() not in ("", "nan", "None"):
                parts.append(str(row[key]).strip())
        for key in extra_dose_cols:
            if key in row and _looks_like_recommendation(row[key]):
                val = _text(row[key])
                if val and val not in parts:
                    parts.append(val)
        if not parts and "waiting_period_days" in row and _looks_like_recommendation(row["waiting_period_days"]):
            parts.append(_text(row["waiting_period_days"]))
        return " | ".join(parts)

    df["dose_text"] = df.apply(_compose, axis=1)

    keep = [
        "crop_name",
        "disease_name_en",
        "pesticide_name",
        "ai_g",
        "formulation",
        "dilution_water_l",
        "dose_text",
        "waiting_period_days",
    ]
    cols = [c for c in keep if c in df.columns]
    df = df.loc[:, cols].copy()
    # Drop header-only rows where crop and disease are missing
    if "crop_name" in df.columns and "disease_name_en" in df.columns:
        disease_empty = df["disease_name_en"].isna() | df["disease_name_en"].astype(str).str.strip().str.lower().isin(
            ["nan", "none", ""]
        )
        crop_empty = df["crop_name"].isna() | df["crop_name"].astype(str).str.strip().str.lower().isin(
            ["nan", "none", ""]
        )
        df = df.loc[~(crop_empty & disease_empty)].copy()
        # Drop pesticide header rows accidentally captured as crop rows
        pesticide_like = df["crop_name"].astype(str).str.contains(
            r"\b\d+(?:\.\d+)?%|\b(?:SC|WP|EC|WDG|SL|FS|GR|WG|OD|CS|ULV|DP|DS|SP|ME)\b",
            case=False,
            regex=True,
        )
        df = df.loc[~(disease_empty & pesticide_like)].copy()
        if "ai_g" in df.columns and "formulation" in df.columns and "dilution_water_l" in df.columns:
            empty_dose = (
                df["ai_g"].apply(_is_empty)
                & df["formulation"].apply(_is_empty)
                & df["dilution_water_l"].apply(_is_empty)
            )
            df = df.loc[~(disease_empty & empty_dose)].copy()
    df = df.loc[:, ~df.columns.duplicated()]
    if "pesticide_name" in df.columns:
        non_empty = df["pesticide_name"].dropna()
        df.attrs["final_pesticide"] = str(non_empty.iloc[-1]) if not non_empty.empty else initial_pesticide
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/processed/pdf_tables")
    parser.add_argument("--output_xlsx", default="data/processed/pesticide_recos_norm.xlsx")
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--disease_map", default="data/raw/disease_hindi_map.json")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_path = Path(args.output_xlsx or args.output_csv)

    if not in_dir.exists():
        print("Input dir missing.")
        return

    dfs = []
    table_files = _table_files(in_dir)
    xlsx_count = sum(1 for p in table_files if p.suffix.lower() == ".xlsx")
    csv_count = sum(1 for p in table_files if p.suffix.lower() == ".csv")
    print(f"Using {len(table_files)} table files ({xlsx_count} xlsx, {csv_count} csv fallback).")

    # Build header map from *_table_1.xlsx/csv per PDF using header + subheader rows
    header_map = {}
    header_drop_map = {}
    for table_path in [p for p in table_files if _table_num(p) == 1]:
        try:
            df = _read_table(table_path)
        except Exception:
            continue
        header_main_idx = None
        for i in range(min(3, len(df))):
            row = " ".join([str(x) for x in df.iloc[i].values if pd.notna(x)])
            row_lc = row.lower()
            if (
                "crop" in row_lc
                or "common name" in row_lc
                or "weed species" in row_lc
                or "name of insect" in row_lc
                or "dose/ha" in row_lc
            ):
                header_main_idx = i
                break
        if header_main_idx is None:
            continue
        header_rows = [header_main_idx]
        if header_main_idx + 1 < len(df):
            header_rows.append(header_main_idx + 1)
        header = _compose_header_from_rows(df, header_rows)
        if header:
            key = _source_stem(table_path)
            header_map[key] = header
            header_drop_map[key] = header_main_idx + 2

    tables_by_stem: dict[str, list[Path]] = {}
    for table_path in table_files:
        tables_by_stem.setdefault(_source_stem(table_path), []).append(table_path)

    for stem, table_paths in sorted(tables_by_stem.items()):
        current_pesticide = None
        for table_path in sorted(table_paths, key=_table_num):
            try:
                df = _read_table(table_path)
            except Exception:
                continue
            if df.empty:
                continue
            header_override = header_map.get(stem)
            drop_rows = header_drop_map.get(stem, 0) if _table_num(table_path) == 1 else 0
            df = normalize_columns(
                df,
                header_override=header_override,
                drop_rows=drop_rows,
                initial_pesticide=current_pesticide,
            )
            current_pesticide = df.attrs.get("final_pesticide", current_pesticide)
            if df.empty:
                continue
            # Final cleanup: drop pesticide-only header rows
            if "crop_name" in df.columns and "disease_name_en" in df.columns:
                disease_empty = df["disease_name_en"].isna() | df["disease_name_en"].astype(str).str.strip().str.lower().isin(
                    ["nan", "none", ""]
                )
                pesticide_like = df["crop_name"].astype(str).str.contains(
                    r"\b\d+(?:\.\d+)?%|\b(?:SC|WP|EC|WDG|SL|FS|GR|WG|OD|CS|ULV|DP|DS|SP|ME|WS)\b",
                    case=False,
                    regex=True,
                )
                df = df.loc[~(disease_empty & pesticide_like)].copy()
            df["source_file"] = table_path.name
            dfs.append(df)

    if not dfs:
        print("No tables normalized.")
        return

    out = pd.concat(dfs, ignore_index=True)
    # Final global cleanup for pesticide header rows
    if "crop_name" in out.columns and "disease_name_en" in out.columns:
        disease_empty = out["disease_name_en"].isna() | out["disease_name_en"].astype(str).str.strip().str.lower().isin(
            ["nan", "none", ""]
        )
        pesticide_like = out["crop_name"].astype(str).str.contains(
            r"\b\d+(?:\.\d+)?%|\b(?:SC|WP|EC|WDG|SL|FS|GR|WG|OD|CS|ULV|DP|DS|SP|ME|WS)\b",
            case=False,
            regex=True,
        )
        out = out.loc[~(disease_empty & pesticide_like)].copy()

    # Add disease Hindi if map exists
    disease_map_path = Path(args.disease_map)
    if disease_map_path.exists():
        try:
            disease_map = json.loads(disease_map_path.read_text(encoding="utf-8"))
        except Exception:
            disease_map = {}
    else:
        disease_map = {}
    if disease_map:
        out["disease_name_hi"] = out["disease_name_en"].map(disease_map).fillna("")
    else:
        out["disease_name_hi"] = ""

    if "dilution_water_l" in out.columns:
        out = out.rename(columns={"dilution_water_l": "dilution"})
    _write_table(out, out_path)
    print(f"Saved normalized table: {out_path}")


if __name__ == "__main__":
    main()
