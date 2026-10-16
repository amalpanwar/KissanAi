import json
import unittest
from unittest.mock import patch, MagicMock
from app.sarvam_generation import generate_guide


class SarvamGenerationTests(unittest.TestCase):
    def setUp(self):
        self.source = {'answer': 'Apply 750 kg FYM before sowing.', 'references': ['guide.pdf'],
                       'agent_status': 'ok', 'requires_hindi': True}
        self.settings = patch('app.sarvam_generation._setting', side_effect=lambda name, default='': 'test-secret' if name == 'SARVAM_API_KEY' else default)
        self.settings.start(); self.addCleanup(self.settings.stop)

    def reply(self, content, finish='stop'):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({'choices': [
            {'finish_reason': finish, 'message': {'content': content}}]}).encode()
        return response

    def test_hindi_generation_uses_selected_evidence_and_preserves_references(self):
        answer = 'बुवाई से पहले 750 kg गोबर की खाद मिट्टी में मिलाएँ।'
        with patch('app.sarvam_generation.urlopen', return_value=self.reply(answer)) as call:
            result = generate_guide(self.source, 'Bajra ki kheti kaise kare')
        self.assertEqual(result['answer'], answer)
        self.assertEqual(result['references'], ['guide.pdf'])
        self.assertEqual(result['generation']['status'], 'generated')
        request = call.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload['model'], 'sarvam-105b')
        self.assertIn('Apply 750 kg FYM', payload['messages'][0]['content'])
        self.assertEqual(call.call_args.kwargs['timeout'], 25)

    def test_invalid_or_truncated_generation_falls_back_to_translation_input(self):
        for answer, finish in [('Apply 750 kg FYM.', 'stop'), ('खाद की मात्रा 999 किग्रा रखें।', 'stop'),
                               ('खाद का उपयोग करें।', 'length'), ('', 'stop')]:
            with self.subTest(answer=answer), patch('app.sarvam_generation.urlopen', return_value=self.reply(answer, finish)):
                result = generate_guide(self.source, 'Bajra')
            self.assertEqual(result['answer'], self.source['answer'])
            self.assertEqual(result['generation']['status'], 'fallback')
            self.assertNotIn('test-secret', str(result))

    def test_timeout_falls_back_without_exposing_exception(self):
        with patch('app.sarvam_generation.urlopen', side_effect=TimeoutError('test-secret')):
            result = generate_guide(self.source, 'Bajra')
        self.assertEqual(result['generation']['reason'], 'TimeoutError')
        self.assertNotIn('test-secret', str(result))

    def test_missing_key_skips_network(self):
        with patch('app.sarvam_generation._setting', side_effect=lambda name, default='': '' if name == 'SARVAM_API_KEY' else default), patch('app.sarvam_generation.urlopen') as call:
            result = generate_guide(self.source, 'Bajra')
        call.assert_not_called()
        self.assertEqual(result['generation']['reason'], 'not_configured')

if __name__ == '__main__': unittest.main()
