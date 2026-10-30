import unittest
from unittest.mock import patch

from app.hindi_translation import translate_answer, _translate, _protect, TOKEN


class NamePreservationTests(unittest.TestCase):
    def setUp(self):
        _translate.cache_clear()
        patcher = patch('app.hindi_translation._setting', side_effect=lambda name, default='': 'test-token' if name == 'HF_TOKEN' else default)
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def hindi_with_tokens(text, token=''):
        return ' '.join(TOKEN.findall(text)) + ' में आसमान साफ है।'

    def test_weather_location_stays_in_english_and_hindi_is_kept(self):
        answer = 'आज का मौसम (Doghat Rural, Baghpat, Uttar Pradesh):\n- तापमान: 32.3°C\n- स्थिति: आसमान साफ'
        with patch('app.hindi_translation._translate') as translate:
            result = translate_answer({'answer': answer, 'topic': 'weather'})
        self.assertEqual(result['answer'], answer)
        self.assertEqual(result['translation']['status'], 'not_needed')
        translate.assert_not_called()

    def test_location_and_person_names_survive_translation(self):
        source = {'answer': 'Ram Singh: Clear sky in Doghat Rural, Baghpat, Uttar Pradesh.',
                  'protected_names': ['Ram Singh'],
                  'agent_trace': {'messages': [{'payload': {'evidence': {'location': {
                      'place': 'Doghat Rural', 'district': 'Baghpat', 'state': 'Uttar Pradesh'}}}}]}}
        with patch('app.hindi_translation._translate', side_effect=self.hindi_with_tokens) as translate:
            result = translate_answer(source)
        self.assertEqual(result['translation']['status'], 'translated')
        self.assertEqual(translate.call_count, 1)
        for name in ['Ram Singh', 'Doghat Rural', 'Baghpat', 'Uttar Pradesh']:
            self.assertIn(name, result['answer'])
        self.assertNotIn('Clear sky', result['answer'])

    def test_context_protects_location_in_mixed_language_answer(self):
        source = {'answer': 'Doghat Rural and Baghpat गेहूं की खेती।', 'agent_trace': {'messages': [
            {'payload': {'context': 'State: Uttar Pradesh | District: Baghpat | Place: Doghat Rural | '}}]}}
        with patch('app.hindi_translation._translate', side_effect=self.hindi_with_tokens) as translate:
            result = translate_answer(source)
        for name in ['Doghat Rural', 'Baghpat']:
            self.assertIn(name, result['answer'])
        self.assertEqual(result['translation']['status'], 'translated')

    def test_longest_name_is_preserved_as_a_unit(self):
        text, values = _protect('Doghat Rural, Doghat', ['Doghat Rural', 'Doghat'])
        self.assertEqual(list(values.values()), ['Doghat Rural', 'Doghat'])
        self.assertEqual(len(TOKEN.findall(text)), 2)

    def test_do_not_guess_prose_is_a_name(self):
        from app.hindi_translation import _protected_names
        self.assertEqual(_protected_names({'answer': 'Apply Fertilizer'}), [])


if __name__ == '__main__':
    unittest.main()
