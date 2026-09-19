"""Validated hierarchy selections shared by the UI and specialist agents."""
from __future__ import annotations

import json
import re
from typing import Mapping

import pandas as pd

from app.location_lookup import _get_lookup, _normalize_place, _row_to_result


def scoped_places(state: str, district: str, lookup: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = _get_lookup() if lookup is None else lookup
    if frame.empty or not {'state', 'district', 'place'}.issubset(frame.columns):
        return pd.DataFrame()
    mask = (frame.state.fillna('').map(_normalize_place) == _normalize_place(state)) & (
        frame.district.fillna('').map(_normalize_place) == _normalize_place(district))
    return frame[mask].fillna('').copy()


def place_options(state: str, district: str, lookup: pd.DataFrame | None = None) -> dict[str, dict]:
    frame = scoped_places(state, district, lookup)
    if frame.empty:
        return {}
    result = {}
    for row in frame.sort_values(['place', 'sub_district'], key=lambda col: col.str.casefold()).to_dict('records'):
        if not row['place'].strip():
            continue
        key = json.dumps([row['state'], row['district'], row.get('sub_district', ''), row['place']], ensure_ascii=False)
        loc = _row_to_result(row, row['place'])
        loc['match_level'] = 'place'
        result.setdefault(key, loc)
    return result


def place_label(location: Mapping) -> str:
    place = str(location.get('place') or '')
    tehsil = str(location.get('sub_district') or '')
    return f'{place} — {tehsil}' if tehsil else place


def qualified_place(location: Mapping) -> str:
    return ', '.join(dict.fromkeys(str(location.get(k) or '').strip() for k in
                                 ['place', 'sub_district', 'district', 'state'] if location.get(k)))


def location_context(location: Mapping) -> str:
    def safe(key):
        return str(location.get(key) or '').replace('|', ' ').replace('\n', ' ').strip()
    return f"State: {safe('state')} | District: {safe('district')} | Subdistrict: {safe('sub_district')} | Place: {safe('place')} | "


def location_from_context(context: str) -> dict:
    values = {}
    aliases = {'state': 'state', 'राज्य': 'state', 'district': 'district', 'जिला': 'district',
               'subdistrict': 'sub_district', 'tehsil': 'sub_district', 'place': 'place', 'स्थान': 'place'}
    for piece in context.split('|'):
        label, separator, value = piece.partition(':')
        key = aliases.get(label.strip().lower())
        if key and separator:
            values[key] = value.strip()
    state = values.get('state') or 'Uttar Pradesh'
    district = values.get('district', '')
    if not district:
        return {}
    if values.get('place'):
        choices = place_options(state, district)
        matches = [v for v in choices.values() if _normalize_place(v['place']) == _normalize_place(values['place'])
                   and (not values.get('sub_district') or _normalize_place(v['sub_district']) == _normalize_place(values['sub_district']))]
        if len(matches) == 1:
            return matches[0]
        # A selected place must match exactly within its hierarchy.
        return {'state': state, 'district': district, 'place': values['place'], 'invalid_selection': True}
    return {'state': state, 'district': district, 'place': '', 'sub_district': '', 'match_level': 'district'}


def scope_market_rows(frame: pd.DataFrame, location: Mapping) -> tuple[pd.DataFrame, str]:
    """Use same-name town markets, otherwise explicitly use district markets.

    Agmarknet does not provide a village-to-nearest-market mapping here, so no
    nearest-market or village-level observation is inferred.
    """
    if frame.empty:
        return frame.copy(), 'unavailable'
    district_col = 'district_name' if 'district_name' in frame else 'District'
    state_col = 'state_name' if 'state_name' in frame else 'State'
    if district_col not in frame or state_col not in frame:
        return frame.iloc[0:0].copy(), 'unavailable'
    out = frame[(frame[district_col].fillna('').map(_normalize_place) == _normalize_place(str(location.get('district', '')))) &
                (frame[state_col].fillna('').map(_normalize_place) == _normalize_place(str(location.get('state', ''))))].copy()
    market_col = 'market_name' if 'market_name' in out else 'Market'
    place = str(location.get('place') or '')
    if place and market_col in out:
        def market_key(value):
            value = re.sub(r'\([^)]*\)', '', str(value))
            value = re.sub(r'\b(apmc|mandi|market|rural|urban)\b', '', value, flags=re.I)
            return _normalize_place(value)
        exact = out[out[market_col].map(market_key) == market_key(place)]
        if not exact.empty:
            return exact, 'town_market'
    return out, 'district_markets' if not out.empty else 'unavailable'


def market_scope_caption(location: Mapping, scope: str) -> str:
    place = str(location.get('place') or '')
    district = str(location.get('district') or '')
    if not place:
        return f'{district} जिले की उपलब्ध मंडियों के भाव।'
    if scope == 'town_market':
        return f'{place}: इसी नाम की मंडी के उपलब्ध भाव; गांव के खेत पर बिक्री का भाव नहीं।'
    if scope == 'unavailable':
        return f'{place}, {district}: चयनित क्षेत्र के लिए मंडी डेटा उपलब्ध नहीं है।'
    return f'{place} के लिए अलग मंडी रिकॉर्ड नहीं मिला; {district} जिले की उपलब्ध मंडियों के भाव दिखाए गए हैं।'
