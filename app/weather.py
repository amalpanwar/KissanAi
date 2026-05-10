from __future__ import annotations

import json
import math
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from app.location_lookup import lookup_place_in_text, lookup_place, resolve_location_hierarchy


def _weather_code_hi(code: int) -> str:
    mapping = {
        0: "आसमान साफ",
        1: "मुख्यतः साफ",
        2: "आंशिक बादल",
        3: "बादल छाए",
        45: "कोहरा",
        48: "घना कोहरा",
        51: "हल्की फुहार",
        53: "मध्यम फुहार",
        55: "तेज फुहार",
        61: "हल्की बारिश",
        63: "मध्यम बारिश",
        65: "तेज बारिश",
        71: "हल्की बर्फबारी",
        80: "बारिश के छिटपुट दौर",
        95: "आंधी/तूफान",
    }
    return mapping.get(code, "मौसम सामान्य")


def _geocode_free(place: str) -> tuple[float, float, str] | None:
    if not place:
        return None
    loc = resolve_location_hierarchy(place) or lookup_place(place) or lookup_place_in_text(place)
    base_place = place
    if loc and loc.get("place"):
        base_place = loc["place"]
    # If lookup has coordinates, use them directly.
    if loc:
        try:
            coord_candidates = [
                (loc.get("lat"), loc.get("lon"), loc.get("place")),
                (loc.get("sub_district_lat"), loc.get("sub_district_lon"), loc.get("sub_district")),
                (loc.get("district_lat"), loc.get("district_lon"), loc.get("district")),
            ]
            for lat_raw, lon_raw, label in coord_candidates:
                lat = float(lat_raw or 0)
                lon = float(lon_raw or 0)
                if lat and lon and not (math.isnan(lat) or math.isnan(lon)):
                    display = ", ".join(
                        [p for p in [label or loc.get("place"), loc.get("district"), loc.get("state"), "India"] if p]
                    )
                    return lat, lon, display
        except Exception:
            pass

    candidates = [
        base_place,
        f"{base_place}, Uttar Pradesh",
        f"{base_place}, Uttar Pradesh, India",
        f"{base_place}, India",
    ]
    if loc and loc.get("district"):
        candidates.insert(0, f"{base_place}, {loc['district']}, Uttar Pradesh, India")
    if loc and loc.get("sub_district") and loc.get("district"):
        candidates.insert(0, f"{base_place}, {loc['sub_district']}, {loc['district']}, Uttar Pradesh, India")
    if loc and loc.get("sub_district"):
        candidates.insert(0, f"{base_place}, {loc['sub_district']}, Uttar Pradesh, India")

    seen = set()
    candidates = [c for c in candidates if c and not (c in seen or seen.add(c))]

    cache_path = "data/processed/weather_geocode_cache.json"
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
    except Exception:
        cache = {}

    for cand in candidates[:4]:
        if cand in cache:
            cached = cache[cand]
            return cached["lat"], cached["lon"], cached["display"]
        # OSM/Nominatim first (free website geocoder)
        try:
            osm_params = urlencode({"q": cand, "format": "json", "limit": 1, "addressdetails": 1})
            osm_url = f"https://nominatim.openstreetmap.org/search?{osm_params}"
            req = Request(osm_url, headers={"User-Agent": "KisaanAi/1.0"})
            with urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if data:
                top = data[0]
                if not _result_matches_lookup(top, loc):
                    raise ValueError("Geocoder result does not match lookup region")
                lat = float(top.get("lat"))
                lon = float(top.get("lon"))
                display = top.get("display_name") or cand
                cache[cand] = {"lat": lat, "lon": lon, "display": display, "ts": time.time()}
                try:
                    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(cache, f, ensure_ascii=False)
                except Exception:
                    pass
                return lat, lon, display
        except Exception:
            pass

        # Open-Meteo geocoder fallback
        params = urlencode({"name": cand, "count": 1, "language": "hi", "format": "json"})
        url = f"https://geocoding-api.open-meteo.com/v1/search?{params}"
        try:
            with urlopen(url, timeout=4) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            results = payload.get("results") or []
            if results:
                top = results[0]
                if not _openmeteo_result_matches_lookup(top, loc):
                    raise ValueError("Open-Meteo result does not match lookup region")
                lat = float(top.get("latitude"))
                lon = float(top.get("longitude"))
                name_parts = [top.get("name"), top.get("admin1"), top.get("country")]
                display = ", ".join([p for p in name_parts if p])
                cache[cand] = {"lat": lat, "lon": lon, "display": display, "ts": time.time()}
                try:
                    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(cache, f, ensure_ascii=False)
                except Exception:
                    pass
                return lat, lon, display
        except Exception:
            pass
    return None


def _result_matches_lookup(result: dict, loc: dict | None) -> bool:
    if not loc:
        return True
    address = result.get("address") or {}
    country = (address.get("country") or "").lower()
    country_code = (address.get("country_code") or "").lower()
    if country_code and country_code != "in":
        return False
    if country and "india" not in country:
        return False
    district = (loc.get("district") or "").lower().strip()
    state = (loc.get("state") or "").lower().strip()
    sub_district = (loc.get("sub_district") or "").lower().strip()
    hay = " ".join(
        [
            str(result.get("display_name") or ""),
            str(address.get("state") or ""),
            str(address.get("state_district") or ""),
            str(address.get("county") or ""),
            str(address.get("city_district") or ""),
            str(address.get("suburb") or ""),
            str(address.get("village") or ""),
            str(address.get("town") or ""),
            str(address.get("city") or ""),
        ]
    ).lower()
    if state and state not in hay:
        return False
    if district and district not in hay:
        return False
    if sub_district and sub_district not in hay:
        # sub-district mismatch is softer than district/state mismatch
        return district in hay if district else True
    return True


def _openmeteo_result_matches_lookup(result: dict, loc: dict | None) -> bool:
    if not loc:
        return True
    country = str(result.get("country") or "").lower()
    admin1 = str(result.get("admin1") or "").lower()
    admin2 = str(result.get("admin2") or "").lower()
    admin3 = str(result.get("admin3") or "").lower()
    name = str(result.get("name") or "").lower()
    state = (loc.get("state") or "").lower().strip()
    district = (loc.get("district") or "").lower().strip()
    sub_district = (loc.get("sub_district") or "").lower().strip()
    if country and "india" not in country:
        return False
    if state and state not in " ".join([admin1, admin2, admin3, name]):
        return False
    if district and district not in " ".join([admin1, admin2, admin3, name]):
        return False
    if sub_district and sub_district not in " ".join([admin1, admin2, admin3, name]):
        return district in " ".join([admin1, admin2, admin3, name]) if district else True
    return True


def get_current_weather_hindi(place: str) -> str:
    geo = _geocode_free(place.strip())
    if not geo:
        return "अभी लाइव मौसम डेटा नहीं मिल पाया। कृपया कुछ देर बाद फिर प्रयास करें।"
    lat, lon, resolved_name = geo
    # Primary: Open-Meteo
    params = urlencode(
        {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,rain,weather_code",
            "hourly": "precipitation_probability,rain,showers,weather_code,temperature_2m",
            "forecast_hours": 24,
            "timezone": "Asia/Kolkata",
        }
    )
    url = f"https://api.open-meteo.com/v1/forecast?{params}"
    try:
        with urlopen(url, timeout=6) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        payload = None

    if not payload:
        # Fallback: wttr.in (free)
        try:
            wttr_url = f"https://wttr.in/{resolved_name}?format=j1"
            with urlopen(wttr_url, timeout=6) as resp:
                wdata = json.loads(resp.read().decode("utf-8"))
            current = (wdata.get("current_condition") or [{}])[0]
            temp = current.get("temp_C", "NA")
            humidity = current.get("humidity", "NA")
            wind = current.get("windspeedKmph", "NA")
            desc = (current.get("weatherDesc") or [{}])[0].get("value", "मौसम सामान्य")
            return (
                f"आज का मौसम ({resolved_name}):\n"
                f"- स्थिति: {desc}\n"
                f"- तापमान: {temp}°C\n"
                f"- आर्द्रता: {humidity}%\n"
                f"- हवा की गति: {wind} km/h\n\n"
                "कृषि सुझाव: अगर वर्षा/हवा अधिक हो तो सिंचाई और स्प्रे शेड्यूल समायोजित करें।"
            )
        except Exception:
            return "अभी लाइव मौसम डेटा नहीं मिल पाया। कृपया कुछ देर बाद फिर प्रयास करें।"

    current = payload.get("current", {})
    hourly = payload.get("hourly", {})
    hourly_times = hourly.get("time", []) or []
    hourly_rain = hourly.get("rain", []) or []
    hourly_showers = hourly.get("showers", []) or []
    hourly_prob = hourly.get("precipitation_probability", []) or []

    temp = current.get("temperature_2m", "NA")
    humidity = current.get("relative_humidity_2m", "NA")
    wind = current.get("wind_speed_10m", "NA")
    rain = current.get("rain", "NA")
    code = int(current.get("weather_code", 0))
    summary = _weather_code_hi(code)

    outlook_lines = []
    try:
        if current.get("time") and hourly_times:
            now_idx = 0
            for i, t in enumerate(hourly_times):
                if t >= current["time"]:
                    now_idx = i
                    break
            next6 = slice(now_idx, min(now_idx + 6, len(hourly_times)))
            next24 = slice(now_idx, min(now_idx + 24, len(hourly_times)))
            if hourly_prob:
                max_p6 = max(hourly_prob[next6], default=0)
                max_p24 = max(hourly_prob[next24], default=0)
                outlook_lines.append(f"- अगले 6 घंटे में बारिश की अधिकतम संभावना: {max_p6}%")
                outlook_lines.append(f"- अगले 24 घंटे में बारिश की अधिकतम संभावना: {max_p24}%")
            if hourly_rain or hourly_showers:
                rain6 = sum((hourly_rain[next6] if hourly_rain else [])) + sum(
                    (hourly_showers[next6] if hourly_showers else [])
                )
                if rain6:
                    outlook_lines.append(f"- अगले 6 घंटे में अनुमानित वर्षा: {rain6:.1f} mm")
    except Exception:
        pass

    return (
        f"आज का मौसम ({resolved_name}):\n"
        f"- स्थिति: {summary}\n"
        f"- तापमान: {temp}°C\n"
        f"- आर्द्रता: {humidity}%\n"
        f"- हवा की गति: {wind} km/h\n"
        f"- वर्षा: {rain} mm\n\n"
        "घंटावार अनुमान:\n"
        + ("\n".join(outlook_lines) if outlook_lines else "- उपलब्ध नहीं\n")
        + "\n\nकृषि सुझाव: अगर वर्षा/हवा अधिक हो तो सिंचाई और स्प्रे शेड्यूल समायोजित करें।"
    )
