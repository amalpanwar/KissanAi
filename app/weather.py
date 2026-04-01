from __future__ import annotations

import json
from urllib.parse import urlencode
from urllib.request import urlopen


DISTRICT_COORDS = {
    "meerut": (28.9845, 77.7064),
    "muzaffarnagar": (29.4727, 77.7085),
    "baghpat": (28.9446, 77.2187),
    "saharanpur": (29.9680, 77.5552),
    "shamli": (29.4497, 77.3153),
    "bulandshahr": (28.4069, 77.8498),
    "baraut": (29.1028, 77.2587),
}

LOCATION_ALIASES = {
    "meerut": ["meerut", "मेरठ"],
    "muzaffarnagar": ["muzaffarnagar", "मुज़फ्फरनगर", "मुजफ्फरनगर"],
    "baghpat": ["baghpat", "बागपत"],
    "saharanpur": ["saharanpur", "सहारनपुर"],
    "shamli": ["shamli", "शामली"],
    "bulandshahr": ["bulandshahr", "बुलंदशहर"],
    "baraut": ["baraut", "बड़ौत", "बरौत", "barot"],
}


def resolve_location_name(text: str) -> str | None:
    if not text:
        return None
    t = text.lower()
    for key, aliases in LOCATION_ALIASES.items():
        for alias in aliases:
            if alias.lower() in t:
                return key
    return None


def _geocode_location(name: str) -> tuple[float, float, str] | None:
    if not name:
        return None
    candidates = [
        name,
        f"{name}, Uttar Pradesh",
        f"{name}, Uttar Pradesh, India",
    ]
    for cand in candidates:
        params = urlencode(
            {
                "name": cand,
                "count": 1,
                "language": "hi",
                "format": "json",
            }
        )
        url = f"https://geocoding-api.open-meteo.com/v1/search?{params}"
        try:
            with urlopen(url, timeout=8) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception:
            continue
        results = payload.get("results") or []
        if not results:
            continue
        top = results[0]
        try:
            lat = float(top.get("latitude"))
            lon = float(top.get("longitude"))
        except Exception:
            continue
        name_parts = [top.get("name"), top.get("admin1"), top.get("country")]
        display = ", ".join([p for p in name_parts if p])
        return lat, lon, display
    return None


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


def get_current_weather_hindi(location: str) -> str:
    # Prefer geocoding for specific places (village/town + district).
    resolved_name = location.strip()
    geo = _geocode_location(location.strip())
    if geo:
        lat, lon, resolved_name = geo
    else:
        key = resolve_location_name(location) or location.strip().lower()
        lat_lon = DISTRICT_COORDS.get(key)
        if not lat_lon:
            return ""
        lat, lon = lat_lon
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
        with urlopen(url, timeout=8) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return ""

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
