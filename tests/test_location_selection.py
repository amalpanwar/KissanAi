import unittest
from unittest.mock import patch
import pandas as pd

from app.location_selection import (place_options, location_context, location_from_context,
                                    scope_market_rows, market_scope_caption, qualified_place)
from app.weather import _lookup_weather_coordinates, _geocode_free


def hierarchy():
    return pd.DataFrame([
        dict(place='Rampur', place_norm='rampur', sub_district='Sardhana', district='Meerut', state='Uttar Pradesh', lat=29.2, lon=77.7, sub_district_lat=29.1, sub_district_lon=77.6, district_lat=28.9, district_lon=77.7),
        dict(place='Rampur', place_norm='rampur', sub_district='Mawana', district='Meerut', state='Uttar Pradesh', lat=29.3, lon=77.8, sub_district_lat=29.2, sub_district_lon=77.9, district_lat=28.9, district_lon=77.7),
        dict(place='Rampur', place_norm='rampur', sub_district='Khekada', district='Baghpat', state='Uttar Pradesh', lat=28.9, lon=77.3, sub_district_lat=28.8, sub_district_lon=77.2, district_lat=29.0, district_lon=77.3),
    ])


class HierarchyTests(unittest.TestCase):
    def test_duplicate_names_have_distinct_hierarchy_keys(self):
        options=place_options('Uttar Pradesh','Meerut',hierarchy())
        self.assertEqual(len(options),2)
        self.assertEqual({v['sub_district'] for v in options.values()},{'Sardhana','Mawana'})
        self.assertTrue(all(v['district']=='Meerut' for v in options.values()))

    def test_context_round_trip_preserves_tehsil(self):
        selected=next(v for v in place_options('Uttar Pradesh','Meerut',hierarchy()).values() if v['sub_district']=='Mawana')
        with patch('app.location_selection._get_lookup',return_value=hierarchy()):
            result=location_from_context(location_context(selected))
        self.assertEqual(result['sub_district'],'Mawana')
        self.assertEqual(result['lat'],29.3)

    def test_ambiguous_or_cross_district_place_is_not_guessed(self):
        with patch('app.location_selection._get_lookup',return_value=hierarchy()):
            result=location_from_context('State: Uttar Pradesh | District: Meerut | Place: Rampur')
            self.assertTrue(result['invalid_selection'])
            result=location_from_context('State: Uttar Pradesh | District: Meerut | Subdistrict: Khekada | Place: Rampur')
            self.assertTrue(result['invalid_selection'])

    def test_town_market_selected_within_state_and_district(self):
        frame=pd.DataFrame([
            dict(State='Uttar Pradesh',District='Meerut',Market='Rampur APMC'),
            dict(State='Uttar Pradesh',District='Baghpat',Market='Rampur APMC'),
            dict(State='Uttar Pradesh',District='Meerut',Market='Mawana APMC')])
        selected={'state':'Uttar Pradesh','district':'Meerut','place':'Rampur'}
        result,scope=scope_market_rows(frame,selected)
        self.assertEqual(scope,'town_market');self.assertEqual(len(result),1)
        self.assertEqual(result.iloc[0]['District'],'Meerut')
        selected['place']='Unknown village'
        result,scope=scope_market_rows(frame,selected)
        self.assertEqual(scope,'district_markets');self.assertEqual(len(result),2)
        self.assertIn('अलग मंडी रिकॉर्ड नहीं',market_scope_caption(selected,scope))

    def test_invalid_geocode_uses_disclosed_regional_fallback(self):
        loc=hierarchy().iloc[0].to_dict()
        loc['place_formatted_address']='Rampur, Moradabad, Uttar Pradesh, India'
        lat,lon,label=_lookup_weather_coordinates(loc)
        self.assertEqual((lat,lon),(29.1,77.6))
        self.assertIn('क्षेत्र का मौसम',label)

    def test_inherited_coordinates_are_not_labelled_village_weather(self):
        loc=hierarchy().iloc[0].to_dict();loc.update(lat=29.1,lon=77.6)
        self.assertIn('गांव के अलग निर्देशांक उपलब्ध नहीं',_lookup_weather_coordinates(loc)[2])

    def test_full_qualification_resolves_correct_duplicate(self):
        with patch('app.location_lookup._get_lookup',return_value=hierarchy()):
            lat,lon,label=_geocode_free('Rampur, Mawana, Meerut, Uttar Pradesh')
        self.assertEqual((lat,lon),(29.3,77.8))

    def test_unknown_district_has_no_villages(self):
        self.assertEqual(place_options('Uttar Pradesh','Unknown',hierarchy()),{})


class AgentLocationTests(unittest.TestCase):
    def test_weather_agent_keeps_selected_hierarchy(self):
        from app.agent_tools import AdvisorTools
        from types import SimpleNamespace
        advisor=SimpleNamespace(_normalize_hinglish=lambda q:q,
                   _parse_weather_request=lambda *args:SimpleNamespace(place=None,action='current',day_offset=None,label=None))
        location=hierarchy().iloc[1].to_dict()
        with patch('app.location_selection._get_lookup',return_value=hierarchy()), patch('app.weather.get_current_weather_hindi',return_value='मौसम ठीक है') as weather:
            result=AdvisorTools(advisor).weather('weather',{'question':'weather today','context':location_context(location)})
        weather.assert_called_once_with('Rampur, Mawana, Meerut, Uttar Pradesh')
        self.assertEqual(result.evidence['location']['sub_district'],'Mawana')
