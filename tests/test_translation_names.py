import json
import unittest
from unittest.mock import patch, MagicMock
from app.hindi_translation import translate_answer, _translate, _protect, TOKEN


class NamePreservationTests(unittest.TestCase):
    def setUp(self):
        _translate.cache_clear()
        p = patch('app.hindi_translation._setting', side_effect=lambda n, default='': 'test-key' if n == 'SARVAM_API_KEY' else default)
        p.start()
        self.addCleanup(p.stop)

    def test_doghat_weather_heading_never_sent_to_translation(self):
        answer = 'आज का मौसम (Doghat Rural, Baghpat, Uttar Pradesh):\n- तापमान: 32.3°C\n- स्थिति: आसमान साफ'
        with patch('app.hindi_translation.urlopen') as call:
            result = translate_answer({'answer': answer, 'topic': 'weather'})
        self.assertEqual(result['answer'], answer)
        self.assertEqual(result['translation']['status'], 'not_needed')
        call.assert_not_called()

    def test_location_and_person_names_survive_translated_prose(self):
        source = {'answer': 'Ram Singh: Clear sky in Doghat Rural, Baghpat, Uttar Pradesh.',
                  'protected_names': ['Ram Singh'],
                  'agent_trace': {'messages': [{'payload': {'evidence': {'location': {
                      'place': 'Doghat Rural', 'district': 'Baghpat', 'state': 'Uttar Pradesh'}}}}]}}
        def server(request, timeout):
            text = json.loads(request.data)['input']
            for name in ['Ram Singh', 'Doghat Rural', 'Baghpat', 'Uttar Pradesh']:
                self.assertNotIn(name, text)
            tokens = TOKEN.findall(text)
            response = MagicMock()
            response.__enter__.return_value.read.return_value = json.dumps({
                'translated_text': f'{tokens[0]}: {tokens[1]}, {tokens[2]}, {tokens[3]} में आसमान साफ है।'
            }).encode()
            return response
        with patch('app.hindi_translation.urlopen', side_effect=server):
            result = translate_answer(source)
        self.assertEqual(result['translation']['status'], 'translated')
        for name in ['Ram Singh', 'Doghat Rural', 'Baghpat', 'Uttar Pradesh']:
            self.assertIn(name, result['answer'])
        self.assertNotIn('Clear sky', result['answer'])

    def test_context_protects_location_for_agronomy_answer(self):
        source = {'answer': 'Doghat Rural और Baghpat में गेहूं की खेती।', 'agent_trace': {'messages': [
            {'payload': {'context': 'State: Uttar Pradesh | District: Baghpat | Place: Doghat Rural | '}}]}}
        with patch('app.hindi_translation.urlopen') as call:
            result = translate_answer(source)
        call.assert_not_called()
        self.assertEqual(result['answer'], source['answer'])

    def test_longest_name_is_preserved_as_a_unit(self):
        text, values = _protect('Doghat Rural, Doghat', ['Doghat Rural', 'Doghat'])
        self.assertEqual(list(values.values()), ['Doghat Rural', 'Doghat'])
        self.assertEqual(len(TOKEN.findall(text)), 2)

    def test_do_not_guess_prose_is_a_name(self):
        from app.hindi_translation import _protected_names
        self.assertEqual(_protected_names({'answer': 'Apply Fertilizer'}), [])


if __name__ == '__main__':
    unittest.main()
