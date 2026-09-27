import ast
import difflib
import os
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from app.location_query import strip_relative_location
from app.location_selection import location_context, place_options
from app.agent_tools import AdvisorTools
from app.weather import _geocode_free


def load_ui_location_functions():
    """Exercise production parsers without starting the Streamlit application."""
    wanted = {'_normalize_text', '_normalize_district_name', '_filter_location_lookup_scope',
              '_build_location_alias_map', '_match_place_from_lookup', 'extract_place_from_query',
              '_lookup_district_from_location', '_extract_explicit_district_from_query',
              '_extract_location_search_hint', '_suggest_locations_within_scope',
              '_resolve_query_location_with_selection'}
    tree = ast.parse(Path('streamlit_app.py').read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    lookup = pd.read_csv('data/processed/location_lookup.csv').fillna('')
    scope = dict(pd=pd, re=re, difflib=difflib, Path=Path, strip_relative_location=strip_relative_location,
                 load_location_lookup=lambda *a: lookup, load_commodity_aliases=lambda *a: {})
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'streamlit_app.py', 'exec'), scope)
    return scope, lookup


class RelativeLocationTests(unittest.TestCase):
    def test_named_places_survive_relative_word_removal(self):
        self.assertIn('Doghat Rural', strip_relative_location('weather here in Doghat Rural'))
        self.assertIn('Baghpat', strip_relative_location('my area Baghpat weather'))
        self.assertEqual(strip_relative_location('Hereford weather'), 'Hereford weather')

    def test_selection_relative_questions_do_not_trigger_location_clarification(self):
        scope, _ = load_ui_location_functions()
        if '_resolve_query_location_with_selection' not in scope:
            if os.environ.get('CI'):
                self.fail('Scoped location parser is missing from the deployed app')
            self.skipTest('Local checkout predates scoped parser; CI checks main-derived parser')
        resolve = scope['_resolve_query_location_with_selection']
        for query in ['आज मेरे क्षेत्र में मौसम कैसा रहेगा?', 'weather in my area',
                      'What is the weather like here?', 'mere gaon me mausam kaisa hai',
                      'यहाँ मौसम कैसा है?', 'weather in selected location']:
            with self.subTest(query=query):
                result = resolve(query, selected_state='Uttar Pradesh', selected_district='Baghpat', strict_on_hint=True)
                self.assertEqual(result['status'], 'none')
        named = resolve('weather in Doghat Rural', selected_state='Uttar Pradesh', selected_district='Baghpat', strict_on_hint=True)
        self.assertEqual(named['place'], 'Doghat Rural')
        invalid = resolve('weather in Xyzunknownplace', selected_state='Uttar Pradesh', selected_district='Baghpat', strict_on_hint=True)
        self.assertEqual(invalid['status'], 'suggest')

    def test_doghat_weather_uses_village_coordinates_and_district_only_falls_back(self):
        _, lookup = load_ui_location_functions()
        selected = next(v for v in place_options('Uttar Pradesh', 'Baghpat', lookup).values() if v['place'] == 'Doghat Rural')
        advisor = SimpleNamespace(_normalize_hinglish=lambda q: q,
            _parse_weather_request=lambda *a: SimpleNamespace(place=None, action='current', day_offset=None, label=None))
        with patch('app.location_selection._get_lookup', return_value=lookup), patch('app.weather.get_current_weather_hindi', return_value='मौसम') as weather:
            result = AdvisorTools(advisor).weather('weather', {'question': 'आज मेरे क्षेत्र में मौसम कैसा रहेगा?', 'context': location_context(selected)})
            weather.assert_called_once_with('Doghat Rural, Baraut, Baghpat, Uttar Pradesh')
            self.assertEqual(result.evidence['location']['place'], 'Doghat Rural')
        with patch('app.location_lookup._get_lookup', return_value=lookup):
            lat, lon, label = _geocode_free('Doghat Rural, Baraut, Baghpat, Uttar Pradesh')
        self.assertAlmostEqual(lat, float(selected['lat']))
        self.assertAlmostEqual(lon, float(selected['lon']))
        self.assertIn('Doghat Rural', label)
        self.assertNotIn('क्षेत्र का मौसम', label)
        with patch('app.location_selection._get_lookup', return_value=lookup), patch('app.weather.get_current_weather_hindi', return_value='मौसम') as weather:
            AdvisorTools(advisor).weather('weather', {'question': 'weather in my area', 'context': location_context({'state': 'Uttar Pradesh', 'district': 'Baghpat'})})
            weather.assert_called_once_with('Baghpat, Uttar Pradesh')


if __name__ == '__main__':
    unittest.main()
