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
}


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


def get_current_weather_hindi(district: str) -> str:
    key = district.strip().lower()
    lat_lon = DISTRICT_COORDS.get(key)
    if not lat_lon:
        return (
            "इस जिले के लिए मौसम डेटा उपलब्ध नहीं है। "
            "कृपया Meerut, Muzaffarnagar, Baghpat, Saharanpur, Shamli या Bulandshahr चुनें।"
        )

    lat, lon = lat_lon
    params = urlencode(
        {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,rain,weather_code",
            "timezone": "Asia/Kolkata",
        }
    )
    url = f"https://api.open-meteo.com/v1/forecast?{params}"
    try:
        with urlopen(url, timeout=8) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return "अभी लाइव मौसम डेटा नहीं मिल पाया। कृपया कुछ देर बाद फिर प्रयास करें।"

    current = payload.get("current", {})
    temp = current.get("temperature_2m", "NA")
    humidity = current.get("relative_humidity_2m", "NA")
    wind = current.get("wind_speed_10m", "NA")
    rain = current.get("rain", "NA")
    code = int(current.get("weather_code", 0))
    summary = _weather_code_hi(code)

    return (
        f"आज का मौसम ({district.title()}):\n"
        f"- स्थिति: {summary}\n"
        f"- तापमान: {temp}°C\n"
        f"- आर्द्रता: {humidity}%\n"
        f"- हवा की गति: {wind} km/h\n"
        f"- वर्षा: {rain} mm\n\n"
        "कृषि सुझाव: अगर वर्षा/हवा अधिक हो तो सिंचाई और स्प्रे शेड्यूल समायोजित करें।"
    )
