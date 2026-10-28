import io
import json
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError
from app.sarvam_diagnostics import run_diagnostics, _cached_probe, SAMPLE
from app.translation_status import render_translation_status


class SarvamDiagnosticTests(unittest.TestCase):
    def setUp(self):
        _cached_probe.cache_clear()
        settings = patch('app.sarvam_diagnostics._setting', side_effect=lambda k, d='': 'test-secret' if k == 'SARVAM_API_KEY' else d)
        settings.start(); self.addCleanup(settings.stop)

    def reply(self, payload):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(payload).encode()
        return response

    def server(self, request, timeout):
        self.assertEqual(timeout, 8)
        payload = json.loads(request.data)
        self.assertNotIn('test-secret', str(payload))
        if request.full_url.endswith('/translate'):
            self.assertEqual(payload['input'], SAMPLE)
            return self.reply({'translated_text': 'यह जाँच है। Urea 50 kg'})
        self.assertIn(SAMPLE, payload['messages'][0]['content'])
        return self.reply({'choices': [{'message': {'content': 'यह अनुवाद जाँच है।'}}]})

    def test_both_raw_endpoints_are_checked_without_application_translation_guards(self):
        with patch('app.sarvam_diagnostics.urlopen', side_effect=self.server) as call:
            result = run_diagnostics()
        self.assertEqual(call.call_count, 2)
        self.assertEqual(result['translate']['status'], 'responded')
        self.assertTrue(result['translate']['hindi_present'])
        self.assertTrue(result['chat']['hindi_present'])
        self.assertIn('Urea 50 kg', result['translate']['output'])
        self.assertNotIn('test-secret', str(result))

    def test_one_endpoint_failure_does_not_hide_the_other_and_no_error_body_is_exposed(self):
        def server(request, timeout):
            if request.full_url.endswith('/translate'):
                raise HTTPError(request.full_url, 403, 'test-secret private body', None, io.BytesIO(b'test-secret'))
            return self.reply({'choices': [{'message': {'content': 'हिंदी test-secret'}}]})
        with patch('app.sarvam_diagnostics.urlopen', side_effect=server):
            result = run_diagnostics()
        self.assertEqual(result['translate']['reason'], 'http_403')
        self.assertEqual(result['chat']['status'], 'responded')
        self.assertNotIn('test-secret', str(result))
        self.assertNotIn('private body', str(result))

    def test_non_hindi_response_is_visible_for_diagnosis_not_suppressed(self):
        with patch('app.sarvam_diagnostics.urlopen', return_value=self.reply({'translated_text': 'English only', 'choices': [{'message': {'content': 'English only'}}]})):
            result = run_diagnostics()
        self.assertEqual(result['translate']['output'], 'English only')
        self.assertFalse(result['translate']['hindi_present'])
        self.assertEqual(result['translate']['status'], 'responded')

    def test_repeated_checks_are_cached_and_expire(self):
        with patch('app.sarvam_diagnostics.urlopen', side_effect=self.server) as call:
            with patch('app.sarvam_diagnostics.time.time', return_value=100):
                run_diagnostics(); run_diagnostics()
            self.assertEqual(call.call_count, 2)
            with patch('app.sarvam_diagnostics.time.time', return_value=401):
                run_diagnostics()
            self.assertEqual(call.call_count, 4)

    def test_missing_key_never_calls_provider(self):
        with patch('app.sarvam_diagnostics._setting', return_value=''), patch('app.sarvam_diagnostics.urlopen') as call:
            result = run_diagnostics()
        self.assertEqual(result['status'], 'not_configured')
        call.assert_not_called()

    def test_probe_requires_explicit_button_click_and_uses_session_result(self):
        st = MagicMock(); st.session_state = {}; st.button.return_value = False
        with patch('app.sarvam_diagnostics.run_diagnostics', return_value={'sample': SAMPLE}) as probe:
            render_translation_status(st, {'status': 'fallback'}, probe_key=123)
            probe.assert_not_called()
            st.button.return_value = True
            render_translation_status(st, {'status': 'fallback'}, probe_key=123)
            probe.assert_called_once()
        self.assertEqual(st.session_state['kisaanai_sarvam_probe'], {'sample': SAMPLE})
        self.assertIn('sarvam_probe_123', [c.kwargs['key'] for c in st.button.call_args_list])
