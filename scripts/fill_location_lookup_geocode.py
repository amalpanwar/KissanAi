from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError, HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


DEFAULT_IN = "data/processed/location_lookup.csv"
DEFAULT_CACHE = "data/processed/location_geocode_cache.json"


def load_env_file(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception:
        return


def load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_cache(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def geocode_mapsco(address: str, api_key: str, cache: dict[str, Any], sleep_sec: float) -> dict[str, Any] | None:
    key = f"mapsco::{address.strip()}"
    if not address.strip():
        return None
    if key in cache:
        return cache[key]
    params = urlencode({"q": address, "api_key": api_key})
    req = Request(f"https://geocode.maps.co/search?{params}", headers={"User-Agent": "KisaanAi/1.0"})
    with urlopen(req, timeout=20) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    result: dict[str, Any]
    if payload:
        top = payload[0]
        result = {
            "lat": top.get("lat"),
            "lon": top.get("lon"),
            "formatted_address": top.get("display_name", address),
            "place_id": str(top.get("place_id", "")),
            "types": [top.get("type", "")],
            "status": "OK",
        }
    else:
        result = {"lat": None, "lon": None, "formatted_address": "", "place_id": "", "types": [], "status": "ZERO_RESULTS"}
    cache[key] = result
    if sleep_sec > 0:
        time.sleep(sleep_sec)
    return result


def geocode_google(address: str, api_key: str, cache: dict[str, Any], sleep_sec: float) -> dict[str, Any] | None:
    key = f"google::{address.strip()}"
    if not address.strip():
        return None
    if key in cache:
        return cache[key]
    params = urlencode({"address": address, "key": api_key, "region": "in"})
    url = f"https://maps.googleapis.com/maps/api/geocode/json?{params}"
    with urlopen(url, timeout=20) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    status = payload.get("status")
    result: dict[str, Any]
    if status == "OK" and payload.get("results"):
        top = payload["results"][0]
        loc = ((top.get("geometry") or {}).get("location") or {})
        result = {
            "lat": loc.get("lat"),
            "lon": loc.get("lng"),
            "formatted_address": top.get("formatted_address", address),
            "place_id": top.get("place_id", ""),
            "types": top.get("types", []),
            "status": status,
        }
    else:
        result = {"lat": None, "lon": None, "formatted_address": "", "place_id": "", "types": [], "status": status}
    cache[key] = result
    if sleep_sec > 0:
        time.sleep(sleep_sec)
    return result


def geocode_with_retries(
    address: str,
    provider: str,
    api_key: str,
    cache: dict[str, Any],
    sleep_sec: float,
    retries: int,
) -> dict[str, Any] | None:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return geocode_address(address, provider, api_key, cache, sleep_sec)
        except (TimeoutError, URLError, HTTPError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(max(2.0, sleep_sec * (attempt + 2)))
    if last_error:
        raise last_error
    return None


def geocode_address(address: str, provider: str, api_key: str, cache: dict[str, Any], sleep_sec: float) -> dict[str, Any] | None:
    if provider == "mapsco":
        return geocode_mapsco(address, api_key, cache, sleep_sec)
    if provider == "google":
        return geocode_google(address, api_key, cache, sleep_sec)
    raise ValueError(f"Unsupported provider: {provider}")


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip().lower()
    return text in {"", "nan", "none"}


def to_float_or_none(value: Any) -> float | None:
    try:
        if is_missing(value):
            return None
        return float(value)
    except Exception:
        return None


def save_progress(df: pd.DataFrame, out_path: Path, cache_path: Path, cache: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)
    save_cache(cache_path, cache)


def unique_nonempty(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        val = str(item or "").strip()
        if not val or val in seen:
            continue
        seen.add(val)
        out.append(val)
    return out


def build_place_queries(row: pd.Series) -> list[str]:
    place = str(row.get("place") or "").strip()
    sub_district = str(row.get("sub_district") or "").strip()
    district = str(row.get("district") or "").strip()
    state = str(row.get("state") or "").strip() or "Uttar Pradesh"
    sub_fmt = str(row.get("sub_district_formatted_address") or "").strip()
    dist_fmt = str(row.get("district_formatted_address") or "").strip()
    return unique_nonempty(
        [
            f"{place}, {sub_fmt}" if place and sub_fmt else "",
            f"{place}, {sub_district}, {district}, {state}, India" if place else "",
            f"{place}, {sub_district}, {state}, India" if place and sub_district else "",
            f"{place}, {district}, {state}, India" if place and district else "",
            f"{place}, {dist_fmt}" if place and dist_fmt else "",
            f"{place}, {state}, India" if place else "",
        ]
    )


def first_geocode_result(
    queries: list[str],
    provider: str,
    api_key: str,
    cache: dict[str, Any],
    sleep_sec: float,
    retries: int,
) -> dict[str, Any] | None:
    for query in queries:
        try:
            result = geocode_with_retries(query, provider, api_key, cache, sleep_sec, retries)
        except Exception:
            continue
        if result and result.get("lat") is not None:
            return result
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill village, sub-district, and district coordinates using a geocoding provider.")
    parser.add_argument("--input", default=DEFAULT_IN)
    parser.add_argument("--output", default="")
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--provider", choices=["mapsco", "google"], default="mapsco")
    parser.add_argument("--api_key", default="")
    parser.add_argument("--sleep_sec", type=float, default=1.1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--save_every", type=int, default=25)
    parser.add_argument("--levels", default="place,sub_district,district", help="Comma-separated: place,sub_district,district")
    parser.add_argument("--backfill_from_subdistrict", action="store_true", help="Copy sub_district_lat/lon into empty place lat/lon")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    load_env_file()
    env_key_name = "MAPSCO_API_KEY" if args.provider == "mapsco" else "GOOGLE_GEOCODING_API_KEY"
    api_key = args.api_key or os.environ.get(env_key_name, "").strip()
    if not api_key:
        raise SystemExit(f"Provide --api_key or set {env_key_name} in environment/.env.")

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path
    cache_path = Path(args.cache)
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")

    df = pd.read_csv(in_path)
    required_cols = ["place", "sub_district", "district", "state"]
    for col in required_cols:
        if col not in df.columns:
            raise SystemExit(f"Missing required column: {col}")

    for col in [
        "lat",
        "lon",
        "sub_district_lat",
        "sub_district_lon",
        "district_lat",
        "district_lon",
        "place_geocode_source",
        "sub_district_geocode_source",
        "district_geocode_source",
        "place_formatted_address",
        "sub_district_formatted_address",
        "district_formatted_address",
    ]:
        if col not in df.columns:
            df[col] = ""
    for col in ["lat", "lon", "sub_district_lat", "sub_district_lon", "district_lat", "district_lon"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    cache = load_cache(cache_path)
    touched = 0
    processed = 0
    enabled_levels = {part.strip() for part in str(args.levels).split(",") if part.strip()}

    for idx, row in df.iterrows():
        state = str(row.get("state") or "").strip() or "Uttar Pradesh"
        district = str(row.get("district") or "").strip()
        sub_district = str(row.get("sub_district") or "").strip()
        place = str(row.get("place") or "").strip()

        if args.backfill_from_subdistrict and place and (is_missing(row.get("lat")) or is_missing(row.get("lon"))):
            sub_lat = to_float_or_none(row.get("sub_district_lat"))
            sub_lon = to_float_or_none(row.get("sub_district_lon"))
            if sub_lat is not None and sub_lon is not None:
                df.at[idx, "lat"] = sub_lat
                df.at[idx, "lon"] = sub_lon
                if not str(row.get("place_geocode_source") or "").strip():
                    df.at[idx, "place_geocode_source"] = "sub_district_fallback"
                if not str(row.get("place_formatted_address") or "").strip():
                    df.at[idx, "place_formatted_address"] = str(row.get("sub_district_formatted_address") or "")
                touched += 1
                processed += 1
                if args.save_every and processed % args.save_every == 0:
                    save_progress(df, out_path, cache_path, cache)
                if args.limit and processed >= args.limit:
                    break
                continue

        if "place" in enabled_levels and place and (args.force or is_missing(row.get("lat")) or is_missing(row.get("lon"))):
            place_result = first_geocode_result(
                build_place_queries(row),
                args.provider,
                api_key,
                cache,
                args.sleep_sec,
                args.retries,
            )
            if place_result and place_result.get("lat") is not None:
                df.at[idx, "lat"] = to_float_or_none(place_result["lat"])
                df.at[idx, "lon"] = to_float_or_none(place_result["lon"])
                df.at[idx, "place_geocode_source"] = args.provider
                df.at[idx, "place_formatted_address"] = place_result.get("formatted_address", "")
                touched += 1

        if "sub_district" in enabled_levels and sub_district and (args.force or is_missing(row.get("sub_district_lat")) or is_missing(row.get("sub_district_lon"))):
            addr_parts = [sub_district, district, state, "India"]
            sub_result = geocode_with_retries(
                ", ".join([p for p in addr_parts if p]),
                args.provider,
                api_key,
                cache,
                args.sleep_sec,
                args.retries,
            )
            if sub_result and sub_result.get("lat") is not None:
                df.at[idx, "sub_district_lat"] = to_float_or_none(sub_result["lat"])
                df.at[idx, "sub_district_lon"] = to_float_or_none(sub_result["lon"])
                df.at[idx, "sub_district_geocode_source"] = args.provider
                df.at[idx, "sub_district_formatted_address"] = sub_result.get("formatted_address", "")
                touched += 1

        if "district" in enabled_levels and district and (args.force or is_missing(row.get("district_lat")) or is_missing(row.get("district_lon"))):
            addr_parts = [district, state, "India"]
            district_result = geocode_with_retries(
                ", ".join([p for p in addr_parts if p]),
                args.provider,
                api_key,
                cache,
                args.sleep_sec,
                args.retries,
            )
            if district_result and district_result.get("lat") is not None:
                df.at[idx, "district_lat"] = to_float_or_none(district_result["lat"])
                df.at[idx, "district_lon"] = to_float_or_none(district_result["lon"])
                df.at[idx, "district_geocode_source"] = args.provider
                df.at[idx, "district_formatted_address"] = district_result.get("formatted_address", "")
                touched += 1

        processed += 1
        if args.save_every and processed % args.save_every == 0:
            save_progress(df, out_path, cache_path, cache)
        if args.limit and processed >= args.limit:
            break

    save_progress(df, out_path, cache_path, cache)
    print(f"Updated lookup file: {out_path}")
    print(f"Provider: {args.provider}")
    print(f"Rows processed: {processed}")
    print(f"Coordinate writes: {touched}")


if __name__ == "__main__":
    main()
