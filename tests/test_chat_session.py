import unittest
from app.chat_session import sync_chat_location

LOCATION = dict(state='Uttar Pradesh', district='Baghpat', sub_district='Baraut', place='Doghat Rural')

class ChatSessionTests(unittest.TestCase):
    def test_initial_and_unchanged_selection_preserve_chat(self):
        state = {'chat_history': ['answer'], 'session_id': 'original'}
        self.assertFalse(sync_chat_location(state, LOCATION))
        self.assertFalse(sync_chat_location(state, dict(LOCATION, lat=29.2)))
        self.assertFalse(sync_chat_location(state, dict(LOCATION, place=' Doghat Rural ')))
        self.assertEqual(state['chat_history'], ['answer'])
        self.assertEqual(state['session_id'], 'original')

    def test_every_hierarchy_change_clears_chat_and_followup_context(self):
        for field, value in [('state', 'Haryana'), ('district', 'Meerut'), ('sub_district', 'Other'), ('place', 'Other village'), ('place', '')]:
            with self.subTest(field=field, value=value):
                state = {'auth_user': {'id': 7}}
                sync_chat_location(state, LOCATION)
                state.update(chat_history=['old answer'], last_structured_context={'preferred_crop': 'Wheat'},
                             pending_chat_items=['old response'], pending_weather_location={'query': 'old'},
                             auto_market_meta={'old': True}, last_chat_submission='old-id')
                self.assertTrue(sync_chat_location(state, dict(LOCATION, **{field: value})))
                self.assertEqual(state['chat_history'], [])
                for key in ['last_structured_context', 'pending_chat_items', 'pending_weather_location', 'auto_market_meta', 'last_chat_submission']:
                    self.assertNotIn(key, state)
                self.assertEqual(state['chat_location_epoch'], 1)
                self.assertEqual(state['auth_user'], {'id': 7})
                self.assertFalse(sync_chat_location(state, dict(LOCATION, **{field: value})))
                self.assertTrue(sync_chat_location(state, LOCATION))
                self.assertEqual(state['chat_history'], [])
                self.assertEqual(state['chat_location_epoch'], 2)

if __name__ == '__main__':
    unittest.main()
