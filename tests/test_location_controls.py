import json
import unittest
from streamlit.testing.v1 import AppTest

APP = '''
import pandas as pd
import streamlit as st
from unittest.mock import patch
from app.location_controls import render_place_selector
from app.location_selection import location_context
rows = pd.DataFrame([
 {'state':'Uttar Pradesh','district':'Meerut','sub_district':'Mawana','place':'Rampur'},
 {'state':'Uttar Pradesh','district':'Meerut','sub_district':'Sardhana','place':'Rampur'},
 {'state':'Uttar Pradesh','district':'Baghpat','sub_district':'Khekada','place':'Rampur'},
])
state=st.selectbox('State',['Uttar Pradesh','Other state'],key='state')
district=st.selectbox('District',['Meerut','Baghpat','Unsupported'],key='district')
with patch('app.location_selection._get_lookup',return_value=rows):
    loc=render_place_selector(state,district)
st.session_state['selection_result']=loc
st.text(location_context(loc))
'''


class LocationControlsTests(unittest.TestCase):
    def test_select_place_then_change_district_clears_old_village(self):
        app=AppTest.from_string(APP).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox(key='fc_place').options,['Whole district','Rampur — Mawana','Rampur — Sardhana'])
        key=json.dumps(['Uttar Pradesh','Meerut','Mawana','Rampur'],ensure_ascii=False)
        app.selectbox(key='fc_place').set_value(key).run()
        self.assertEqual(app.session_state['selection_result']['sub_district'],'Mawana')
        app.selectbox(key='district').select('Baghpat').run()
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox(key='fc_place').value,'')
        self.assertEqual(app.session_state['selection_result']['district'],'Baghpat')
        self.assertEqual(app.selectbox(key='fc_place').options,['Whole district','Rampur — Khekada'])

    def test_state_change_clears_selection_and_disables_missing_coverage(self):
        app=AppTest.from_string(APP).run()
        key=json.dumps(['Uttar Pradesh','Meerut','Mawana','Rampur'],ensure_ascii=False)
        app.selectbox(key='fc_place').set_value(key).run()
        app.selectbox(key='state').select('Other state').run()
        self.assertFalse(app.exception)
        self.assertTrue(app.selectbox(key='fc_place').disabled)
        self.assertEqual(app.selectbox(key='fc_place').value,'')
        self.assertIn('not yet available',app.caption[0].value)
