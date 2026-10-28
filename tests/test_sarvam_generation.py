import json
import unittest
from unittest.mock import patch, MagicMock
from app.sarvam_generation import generate_text


class SarvamGenerationTests(unittest.TestCase):
    def setUp(self):
        settings = patch('app.sarvam_generation._setting', side_effect=lambda name, default='': 'test-secret' if name == 'SARVAM_API_KEY' else default)
        settings.start(); self.addCleanup(settings.stop)

    def reply(self, content, finish='stop'):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({'choices': [
            {'finish_reason': finish, 'message': {'content': content}}]}).encode()
        return response

    def test_generates_english_from_evidence_prompt(self):
        prompt = 'Answer in English using evidence: Plant in trenches 30 cm deep. [1]'
        with patch('app.sarvam_generation.urlopen', return_value=self.reply('Plant 30 cm deep. [1]')) as call:
            answer = generate_text(prompt)
        self.assertEqual(answer, 'Plant 30 cm deep. [1]')
        payload = json.loads(call.call_args.args[0].data)
        self.assertEqual(payload['messages'][0]['content'], prompt)
        self.assertEqual(payload['model'], 'sarvam-105b')
        self.assertEqual(payload['max_tokens'], 1800)
        self.assertNotIn('response_format', payload)
        self.assertEqual(call.call_args.kwargs['timeout'], 25)

    def test_structured_translation_options_are_sent_only_when_requested(self):
        with patch('app.sarvam_generation.urlopen', return_value=self.reply('{"translations": {"0": "खाद"}}')) as call:
            generate_text('Translate fragments as JSON', timeout=15, max_tokens=3500,
                          response_format={'type': 'json_object'})
        payload = json.loads(call.call_args.args[0].data)
        self.assertEqual(payload['response_format'], {'type': 'json_object'})
        self.assertEqual(payload['max_tokens'], 3500)
        self.assertEqual(call.call_args.kwargs['timeout'], 15)

    def test_empty_or_truncated_generation_is_rejected(self):
        for text, finish in [('', 'stop'), ('partial answer', 'length')]:
            with self.subTest(text=text), patch('app.sarvam_generation.urlopen', return_value=self.reply(text, finish)):
                with self.assertRaises(ValueError): generate_text('evidence')

    def test_missing_key_never_attempts_network(self):
        with patch('app.sarvam_generation._setting', side_effect=lambda name, default='': '' if name == 'SARVAM_API_KEY' else default), patch('app.sarvam_generation.urlopen') as call:
            with self.assertRaises(RuntimeError): generate_text('evidence')
        call.assert_not_called()

    def test_timeout_does_not_fall_back_to_local_model(self):
        with patch('app.sarvam_generation.urlopen', side_effect=TimeoutError):
            with self.assertRaises(TimeoutError): generate_text('evidence')

if __name__ == '__main__': unittest.main()
