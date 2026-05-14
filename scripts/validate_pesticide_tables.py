from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


FORMULATION_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*%|\b(?:SC|WP|EC|WDG|SL|FS|GR|WG|OD|CS|ULV|DP|DS|ME|WS|FF|SP)\b",
    flags=re.IGNORECASE,
)
DOSE_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:g|gm|kg|ml|l|ltr|lit|litre|%)\b|kg seed|ha|hectare",
    flags=re.IGNORECASE,
)
WAITING_RE = re.compile(
    r"\b(?:day|days|seed treatment|seed dresser|required|harvest|phi|waiting)\b",
    flags=re.IGNORECASE,
)
WAITING_NUMBER_RE = re.compile(r"^\s*\d+(?:\.\d+)?\s*$")
VALIDATION_COLUMNS = [
    "validation_errors",
    "validation_warnings",
    "quality_status",
    "quality_flags",
    "validation_status",
]


def _clean_text(value: object) -> str:
    return str(value).replace("\xa0", " ").replace("\n", " ").strip()


def _empty(value: object) -> bool:
    return pd.isna(value) or _clean_text(value).lower() in {"", "nan", "none"}


def _text(value: object) -> str:
    return "" if _empty(value) else _clean_text(value)


def _looks_like_pesticide(value: object) -> bool:
    return bool(FORMULATION_RE.search(_text(value)))


def _looks_like_dose(value: object) -> bool:
    return bool(DOSE_RE.search(_text(value)))


def _looks_like_waiting(value: object) -> bool:
    text = _text(value)
    return bool(WAITING_RE.search(text) or WAITING_NUMBER_RE.search(text))


def _looks_like_waiting_phrase(value: object) -> bool:
    return bool(WAITING_RE.search(_text(value)))


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path, engine="openpyxl")
    return pd.read_csv(path)


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".xlsx":
        df.to_excel(path, index=False)
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output file type: {path}")


def _validate_row(row: pd.Series) -> tuple[str, str]:
    errors = []
    warnings = []

    crop = row.get("crop_name")
    disease = row.get("disease_name_en")
    pesticide = row.get("pesticide_name")
    ai_g = row.get("ai_g")
    formulation = row.get("formulation")
    dilution = row.get("dilution")
    dose_text = row.get("dose_text")
    waiting = row.get("waiting_period_days")

    if _empty(crop):
        errors.append("missing_crop")
    elif _looks_like_pesticide(crop):
        errors.append("crop_looks_like_pesticide")

    if _empty(disease):
        errors.append("missing_disease")

    if _empty(pesticide):
        errors.append("missing_pesticide")
    elif not _looks_like_pesticide(pesticide):
        warnings.append("pesticide_name_not_formula_like")

    if _empty(ai_g) and _empty(formulation) and _empty(dilution) and _empty(dose_text):
        errors.append("missing_all_dose_columns")

    if not _empty(waiting) and _looks_like_dose(waiting) and not _looks_like_waiting(waiting):
        warnings.append("waiting_period_looks_like_dose")

    if not _empty(dilution) and _looks_like_waiting_phrase(dilution):
        warnings.append("dilution_looks_like_waiting_period")

    if _empty(waiting):
        warnings.append("missing_waiting_period")

    return ";".join(errors), ";".join(warnings)


def _has_usable_core(row: pd.Series) -> bool:
    return (
        not _empty(row.get("crop_name"))
        and not _empty(row.get("pesticide_name"))
        and (
            not _empty(row.get("ai_g"))
            or not _empty(row.get("formulation"))
            or not _empty(row.get("dilution"))
            or not _empty(row.get("dose_text"))
        )
        and not _looks_like_pesticide(row.get("crop_name"))
    )


def _classify_row(row: pd.Series) -> str:
    errors = {x for x in str(row.get("validation_errors", "")).split(";") if x}
    warnings = {x for x in str(row.get("validation_warnings", "")).split(";") if x}

    hard_errors = {
        "missing_crop",
        "crop_looks_like_pesticide",
        "missing_pesticide",
        "missing_all_dose_columns",
    }
    if errors & hard_errors:
        return "reject"
    if not errors and not warnings:
        return "valid"
    if _has_usable_core(row):
        return "usable"
    return "review"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_xlsx", default="data/processed/pesticide_recos_norm.xlsx")
    parser.add_argument("--input_csv", default=None)
    parser.add_argument("--valid_xlsx", default="data/processed/pesticide_recos_valid.xlsx")
    parser.add_argument("--usable_xlsx", default="data/processed/pesticide_recos_usable.xlsx")
    parser.add_argument("--review_xlsx", default="data/processed/pesticide_recos_review.xlsx")
    parser.add_argument("--reject_xlsx", default="data/processed/pesticide_recos_reject.xlsx")
    parser.add_argument("--valid_csv", default=None)
    parser.add_argument("--usable_csv", default=None)
    parser.add_argument("--review_csv", default=None)
    parser.add_argument("--reject_csv", default=None)
    parser.add_argument("--summary_json", default="data/processed/pesticide_recos_validation_summary.json")
    args = parser.parse_args()

    input_path = Path(args.input_xlsx or args.input_csv)
    valid_path = Path(args.valid_xlsx or args.valid_csv)
    usable_path = Path(args.usable_xlsx or args.usable_csv)
    review_path = Path(args.review_xlsx or args.review_csv)
    reject_path = Path(args.reject_xlsx or args.reject_csv)
    summary_json = Path(args.summary_json)

    if not input_path.exists():
        raise FileNotFoundError(f"Input table not found: {input_path}")

    df = _read_table(input_path)
    df = df.drop(columns=[c for c in VALIDATION_COLUMNS if c in df.columns], errors="ignore")
    required = [
        "crop_name",
        "disease_name_en",
        "pesticide_name",
        "ai_g",
        "formulation",
        "dilution",
        "waiting_period_days",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    checks = df.apply(_validate_row, axis=1, result_type="expand")
    checks.columns = ["validation_errors", "validation_warnings"]
    df = pd.concat([df, checks], axis=1)
    df["quality_status"] = df.apply(_classify_row, axis=1)
    df["quality_flags"] = df.apply(
        lambda row: ";".join(
            [x for x in [str(row["validation_errors"]).strip(), str(row["validation_warnings"]).strip()] if x and x != "nan"]
        ),
        axis=1,
    )
    df["validation_status"] = df["quality_status"]

    valid = df[df["quality_status"] == "valid"].copy()
    usable = df[df["quality_status"].isin(["valid", "usable"])].copy()
    review = df[df["quality_status"] == "review"].copy()
    reject = df[df["quality_status"] == "reject"].copy()

    _write_table(valid, valid_path)
    _write_table(usable, usable_path)
    _write_table(review, review_path)
    _write_table(reject, reject_path)

    error_counts = {}
    warning_counts = {}
    for value in df["validation_errors"].dropna():
        for item in str(value).split(";"):
            if item:
                error_counts[item] = error_counts.get(item, 0) + 1
    for value in df["validation_warnings"].dropna():
        for item in str(value).split(";"):
            if item:
                warning_counts[item] = warning_counts.get(item, 0) + 1

    summary = {
        "input_rows": int(len(df)),
        "valid_rows": int(len(valid)),
        "usable_rows": int(len(usable)),
        "review_rows": int(len(review)),
        "reject_rows": int(len(reject)),
        "error_counts": error_counts,
        "warning_counts": warning_counts,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Validated {len(df)} rows")
    print(f"Valid rows -> {valid_path} ({len(valid)})")
    print(f"Usable rows -> {usable_path} ({len(usable)})")
    print(f"Review rows -> {review_path} ({len(review)})")
    print(f"Reject rows -> {reject_path} ({len(reject)})")
    print(f"Summary -> {summary_json}")


if __name__ == "__main__":
    main()
