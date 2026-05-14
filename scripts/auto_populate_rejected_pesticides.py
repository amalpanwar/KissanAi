from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from validate_pesticide_tables import (
    VALIDATION_COLUMNS,
    _classify_row,
    _read_table,
    _validate_row,
    _write_table,
)


CORE_COLUMNS = [
    "crop_name",
    "disease_name_en",
    "pesticide_name",
    "ai_g",
    "formulation",
    "dilution",
    "dose_text",
    "waiting_period_days",
    "source_file",
    "disease_name_hi",
]
FILL_COLUMNS = [
    "crop_name",
    "disease_name_en",
    "pesticide_name",
    "ai_g",
    "formulation",
    "dilution",
    "waiting_period_days",
    "disease_name_hi",
]


def _clean(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = str(value).replace("\xa0", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return pd.NA if text.lower() in {"", "nan", "none"} else text


def _empty(value: object) -> bool:
    return pd.isna(value) or str(_clean(value)).strip().lower() in {"", "nan", "none", "<na>"}


def _norm(value: object) -> str:
    text = "" if _empty(value) else str(_clean(value)).lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def _source_stem(value: object) -> str:
    name = "" if _empty(value) else str(_clean(value))
    return name.split("_table_")[0]


def _compose_dose(row: pd.Series) -> str:
    parts = []
    labels = [("a.i.", "ai_g"), ("formulation", "formulation"), ("dilution", "dilution")]
    for label, col in labels:
        if col in row and not _empty(row.get(col)):
            parts.append(f"{label}: {_clean(row.get(col))}")
    existing = row.get("dose_text")
    if not _empty(existing):
        text = str(_clean(existing))
        if text not in parts:
            parts.append(text)
    waiting = row.get("waiting_period_days")
    if not parts and not _empty(waiting):
        text = str(_clean(waiting))
        if re.search(r"\b(?:gm|g/|kg|ml|lit|liter|litre|l/ha|ha|seed|soil|treat|drench|spray|apply|@)\b|\d", text, re.I):
            parts.append(text)
    return " | ".join(parts)


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CORE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df.drop(columns=[c for c in VALIDATION_COLUMNS if c in df.columns], errors="ignore")
    for col in CORE_COLUMNS:
        df[col] = df[col].map(_clean)
    df["dose_text"] = df.apply(_compose_dose, axis=1)
    return df


def _load_reference(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if path.exists():
            frames.append(_clean_frame(_read_table(path)))
    if not frames:
        return pd.DataFrame(columns=CORE_COLUMNS)
    ref = pd.concat(frames, ignore_index=True)
    ref = ref.drop_duplicates(subset=[c for c in CORE_COLUMNS if c in ref.columns])
    ref = ref[~ref["pesticide_name"].map(_empty)].copy()
    ref["_pesticide_key"] = ref["pesticide_name"].map(_norm)
    ref["_source_stem"] = ref["source_file"].map(_source_stem)
    ref["_crop_key"] = ref["crop_name"].map(_norm)
    ref["_disease_key"] = ref["disease_name_en"].map(_norm)
    return ref


def _best_candidate(row: pd.Series, ref: pd.DataFrame) -> pd.Series | None:
    pesticide_key = _norm(row.get("pesticide_name"))
    if not pesticide_key:
        return None

    candidates = ref[ref["_pesticide_key"] == pesticide_key].copy()
    if candidates.empty:
        return None

    source_stem = _source_stem(row.get("source_file"))
    crop_key = _norm(row.get("crop_name"))
    disease_key = _norm(row.get("disease_name_en"))

    if source_stem:
        same_source = candidates[candidates["_source_stem"] == source_stem]
        if not same_source.empty:
            candidates = same_source

    def score(candidate: pd.Series) -> int:
        value = 0
        if source_stem and candidate.get("_source_stem") == source_stem:
            value += 8
        if crop_key and candidate.get("_crop_key") == crop_key:
            value += 6
        if disease_key and candidate.get("_disease_key") == disease_key:
            value += 6
        value += sum(1 for col in FILL_COLUMNS if not _empty(candidate.get(col)))
        return value

    candidates["_score"] = candidates.apply(score, axis=1)
    return candidates.sort_values("_score", ascending=False).iloc[0]


def _unique_value(candidates: pd.DataFrame, column: str) -> object:
    values = [v for v in candidates[column].tolist() if not _empty(v)]
    if not values:
        return pd.NA
    counts = Counter(str(_clean(v)) for v in values)
    if len(counts) == 1:
        return next(iter(counts.keys()))
    return pd.NA


def _safe_fill_from_reference(row: pd.Series, ref: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    filled = []
    pesticide_key = _norm(row.get("pesticide_name"))
    if not pesticide_key:
        return row, filled

    candidates = ref[ref["_pesticide_key"] == pesticide_key].copy()
    source_stem = _source_stem(row.get("source_file"))
    if source_stem:
        same_source = candidates[candidates["_source_stem"] == source_stem]
        if not same_source.empty:
            candidates = same_source
    if candidates.empty:
        return row, filled

    crop_key = _norm(row.get("crop_name"))
    disease_key = _norm(row.get("disease_name_en"))
    if crop_key:
        crop_candidates = candidates[candidates["_crop_key"] == crop_key]
        if not crop_candidates.empty:
            candidates = crop_candidates
    if disease_key:
        disease_candidates = candidates[candidates["_disease_key"] == disease_key]
        if not disease_candidates.empty:
            candidates = disease_candidates

    best = _best_candidate(row, candidates)
    if best is None:
        return row, filled

    for col in ["ai_g", "formulation", "dilution", "dose_text", "waiting_period_days", "disease_name_hi"]:
        if _empty(row.get(col)) and not _empty(best.get(col)):
            row[col] = best[col]
            filled.append(col)

    if _empty(row.get("crop_name")):
        value = _unique_value(candidates, "crop_name")
        if not _empty(value):
            row["crop_name"] = value
            filled.append("crop_name")

    if _empty(row.get("disease_name_en")):
        scoped = candidates
        if not _empty(row.get("crop_name")):
            crop_key = _norm(row.get("crop_name"))
            crop_candidates = scoped[scoped["_crop_key"] == crop_key]
            if not crop_candidates.empty:
                scoped = crop_candidates
        value = _unique_value(scoped, "disease_name_en")
        if not _empty(value):
            row["disease_name_en"] = value
            filled.append("disease_name_en")

    return row, filled


def _revalidate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in VALIDATION_COLUMNS if c in df.columns], errors="ignore").copy()
    checks = df.apply(_validate_row, axis=1, result_type="expand")
    checks.columns = ["validation_errors", "validation_warnings"]
    df = pd.concat([df, checks], axis=1)
    df["quality_status"] = df.apply(_classify_row, axis=1)
    df["quality_flags"] = df.apply(
        lambda row: ";".join(
            [
                x
                for x in [
                    str(row["validation_errors"]).strip(),
                    str(row["validation_warnings"]).strip(),
                    str(row.get("autofill_notes", "")).strip(),
                ]
                if x and x != "nan"
            ]
        ),
        axis=1,
    )
    df["validation_status"] = df["quality_status"]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean and safely auto-populate missing fields in rejected pesticide recommendation rows."
    )
    parser.add_argument("--reject_xlsx", default="data/processed/pesticide_recos_reject.xlsx")
    parser.add_argument(
        "--reference_xlsx",
        action="append",
        default=["data/processed/pesticide_recos_norm.xlsx", "data/processed/pesticide_recos_usable.xlsx"],
        help="Reference workbook to use for safe fills. Can be passed multiple times.",
    )
    parser.add_argument("--output_xlsx", default="data/processed/pesticide_recos_reject_autofilled.xlsx")
    parser.add_argument("--usable_xlsx", default="data/processed/pesticide_recos_reject_autofilled_usable.xlsx")
    parser.add_argument("--still_reject_xlsx", default="data/processed/pesticide_recos_reject_autofilled_still_reject.xlsx")
    parser.add_argument("--excluded_xlsx", default="data/processed/pesticide_recos_excluded_fragments.xlsx")
    parser.add_argument("--base_usable_xlsx", default="data/processed/pesticide_recos_usable.xlsx")
    parser.add_argument("--merged_usable_xlsx", default="data/processed/pesticide_recos_usable_with_autofill.xlsx")
    parser.add_argument("--summary_json", default="data/processed/pesticide_recos_reject_autofill_summary.json")
    args = parser.parse_args()

    reject_path = Path(args.reject_xlsx)
    if not reject_path.exists():
        raise FileNotFoundError(f"Reject workbook not found: {reject_path}")

    df = _clean_frame(_read_table(reject_path))
    reference_paths = [Path(p) for p in args.reference_xlsx]
    ref = _load_reference(reference_paths)

    notes_by_col: dict[str, int] = defaultdict(int)
    rows_with_fill = 0
    autofilled_rows = []
    for _, row in df.iterrows():
        row = row.copy()
        row, filled = _safe_fill_from_reference(row, ref)
        row["dose_text"] = _compose_dose(row)
        if filled:
            rows_with_fill += 1
            for col in filled:
                notes_by_col[col] += 1
            row["autofill_notes"] = "filled:" + ",".join(sorted(set(filled)))
        else:
            row["autofill_notes"] = ""
        autofilled_rows.append(row)

    out = _revalidate(pd.DataFrame(autofilled_rows))
    usable = out[out["quality_status"].isin(["valid", "usable"])].copy()
    raw_reject = out[out["quality_status"] == "reject"].copy()

    def _is_fragment_or_non_crop(row: pd.Series) -> bool:
        def cleaned_lower(col: str) -> str:
            value = _clean(row.get(col))
            return "" if _empty(value) else str(value).lower()

        crop = cleaned_lower("crop_name")
        disease = cleaned_lower("disease_name_en")
        dose = cleaned_lower("dose_text")
        source = cleaned_lower("source_file")
        text = " ".join([crop, disease, dose, source])
        if dose and dose not in {"nan", "<na>"}:
            return False
        non_crop_terms = [
            "readytouse",
            "household",
            "mosquito",
            "cockroach",
            "houseflies",
            "aedes",
            "anopheles",
            "culex",
            "habitat",
            "riverbed",
            "cement tanks",
            "non-crop",
            "noncrop",
            "non cropped",
            "aquatic weed",
            "calculated as",
        ]
        if any(term in text for term in non_crop_terms):
            return True
        errors = {x for x in str(row.get("validation_errors", "")).split(";") if x}
        if "missing_all_dose_columns" in errors and not disease:
            return True
        return False

    excluded = raw_reject[raw_reject.apply(_is_fragment_or_non_crop, axis=1)].copy()
    still_reject = raw_reject.drop(index=excluded.index).copy()
    merged_usable = usable.copy()
    base_usable_path = Path(args.base_usable_xlsx)
    if base_usable_path.exists():
        base_usable = _clean_frame(_read_table(base_usable_path))
        base_usable = _revalidate(base_usable)
        base_usable = base_usable[base_usable["quality_status"].isin(["valid", "usable"])].copy()
        merged_usable = pd.concat([base_usable, usable], ignore_index=True)
        merged_usable = merged_usable.drop_duplicates(
            subset=["crop_name", "disease_name_en", "pesticide_name", "ai_g", "formulation", "dilution", "source_file"],
            keep="first",
        )

    output_path = Path(args.output_xlsx)
    usable_path = Path(args.usable_xlsx)
    reject_out_path = Path(args.still_reject_xlsx)
    excluded_path = Path(args.excluded_xlsx)
    merged_usable_path = Path(args.merged_usable_xlsx)
    summary_path = Path(args.summary_json)

    _write_table(out, output_path)
    _write_table(usable, usable_path)
    _write_table(still_reject, reject_out_path)
    _write_table(excluded, excluded_path)
    _write_table(merged_usable, merged_usable_path)

    summary = {
        "input_rows": int(len(df)),
        "rows_with_autofill": int(rows_with_fill),
        "autofilled_columns": dict(sorted(notes_by_col.items())),
        "usable_after_autofill": int(len(usable)),
        "merged_usable_rows": int(len(merged_usable)),
        "still_reject_after_autofill": int(len(still_reject)),
        "excluded_fragment_or_non_crop_rows": int(len(excluded)),
        "reference_files": [str(p) for p in reference_paths if p.exists()],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Read rejected rows: {len(df)}")
    print(f"Rows with safe autofill: {rows_with_fill}")
    print(f"Usable after fresh validation: {len(usable)} -> {usable_path}")
    print(f"Merged usable workbook: {len(merged_usable)} -> {merged_usable_path}")
    print(f"Still rejected: {len(still_reject)} -> {reject_out_path}")
    print(f"Excluded fragments/non-crop rows: {len(excluded)} -> {excluded_path}")
    print(f"Autofilled workbook -> {output_path}")
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
