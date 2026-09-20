import json
import unittest
from unittest.mock import patch, MagicMock

from app.hindi_translation import translate_answer, _translate, TOKEN, _chunks


class TranslationTests(unittest.TestCase):
    def setUp(self):
        from app.hindi_translation import _setting
        self._real_setting = _setting
        _translate.cache_clear()
        settings = {'SARVAM_API_KEY': 'test-key', 'KISAANAI_HINDI_TRANSLATION': '1'}
        p = patch('app.hindi_translation._setting', side_effect=lambda k, d='': settings.get(k, d))
        p.start()
        self.addCleanup(p.stop)

    def response(self, text):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({'translated_text': text}).encode()
        return response

    def test_real_example_preserves_quantities_and_evidence(self):
        source = 'जिंक/सल्फर की कमी वाली मिट्टी में 37.5 kg ZnSO 4, 40 kg S basally for soils having Zn and S deficiencies दें।'
        original = {'answer': source, 'references': ['https://example.org/source'],
                    'agent_status': 'ok', 'agent_trace': {'goal': 'private farmer context'}}
        def server(request, timeout):
            payload = json.loads(request.data)
            self.assertEqual(payload['model'], 'sarvam-translate:v1')
            self.assertEqual(payload['source_language_code'], 'en-IN')
            self.assertEqual(payload['target_language_code'], 'hi-IN')
            self.assertLessEqual(len(payload['input']), 2000)
            self.assertNotIn('private farmer context', payload['input'])
            tokens = TOKEN.findall(payload['input'])
            self.assertEqual(len(tokens), 4)
            return self.response(f'{tokens[0]}, {tokens[1]} बुवाई के समय उन मिट्टियों में डालें जिनमें {tokens[2]} और {tokens[3]} की कमी है')
        with patch('app.hindi_translation.urlopen', side_effect=server) as call:
            out = translate_answer(original)
            again = translate_answer(original)
        self.assertEqual(out['translation']['status'], 'translated')
        self.assertNotIn('basally', out['answer'])
        for value in ['37.5 kg ZnSO 4', '40 kg S']:
            self.assertIn(value, out['answer'])
        self.assertEqual(out['references'], original['references'])
        self.assertEqual(out['agent_trace'], original['agent_trace'])
        self.assertEqual(again, out)
        self.assertEqual(call.call_count, 1)

    def test_hindi_and_formula_only_skip_api(self):
        with patch('app.hindi_translation.urlopen') as call:
            out = translate_answer({'answer': 'गेहूं में 80:40:40 NPK किग्रा/हेक्टेयर।'})
        call.assert_not_called()
        self.assertEqual(out['translation']['status'], 'not_needed')

    def test_lost_changed_reordered_or_duplicated_numbers_rejected(self):
        for text in ['80 किग्रा डालें', 'ZXQBQXZ फिर ZXQAQXZ डालें',
                     'ZXQAQXZ और ZXQAQXZ डालें', 'ZXQAQXZ और ZXQBQXZ 999 डालें']:
            with self.subTest(text=text), patch('app.hindi_translation.urlopen', return_value=self.response(text)):
                source = {'answer': 'Apply 40 kg and 20 kg.'}
                out = translate_answer(source)
                self.assertEqual(out['answer'], source['answer'])
                self.assertEqual(out['translation']['status'], 'fallback')

    def test_english_output_and_empty_output_rejected(self):
        for text in ['Apply fertiliser', '']:
            with patch('app.hindi_translation.urlopen', return_value=self.response(text)):
                self.assertEqual(translate_answer({'answer': 'Apply fertiliser'})['translation']['status'], 'fallback')

    def test_api_failure_is_atomic_and_does_not_leak_error(self):
        with patch('app.hindi_translation.urlopen', side_effect=TimeoutError('SECRET token')):
            out = translate_answer({'answer': 'Apply fertiliser'})
        self.assertEqual(out['answer'], 'Apply fertiliser')
        self.assertEqual(out['translation']['reason'], 'TimeoutError')
        self.assertNotIn('SECRET', str(out))

    def test_missing_key_and_disabled_do_not_call_service(self):
        for setting, expected in [('SARVAM_API_KEY', 'not_configured'), ('KISAANAI_HINDI_TRANSLATION', 'disabled')]:
            with patch('app.hindi_translation._setting', side_effect=lambda k, d='': ('0' if k == 'KISAANAI_HINDI_TRANSLATION' else '') if k == setting else d), patch('app.hindi_translation.urlopen') as call:
                out = translate_answer({'answer': 'Apply fertiliser'})
                self.assertEqual(out['translation']['status'], expected)
                call.assert_not_called()

    def test_long_input_chunks_and_budget(self):
        self.assertTrue(all(len(s) <= 1800 for s in _chunks('word ' * 1000)))
        with patch('app.hindi_translation.urlopen', side_effect=lambda *a, **kw: self.response('उर्वरक डालें')) as call:
            source = {'answer': '\n'.join(['Apply fertiliser'] * 25)}
            # Distinct fragments defeat the success cache, exercising request budget.
            source['answer'] = '\n'.join('Apply ' + chr(97+i) + ' fertiliser' for i in range(25))
            out = translate_answer(source)
        self.assertEqual(out['translation']['status'], 'fallback')
        self.assertEqual(out['answer'], source['answer'])
        self.assertEqual(call.call_count, 24)

    def test_links_code_and_markdown_list_retained(self):
        with patch('app.hindi_translation.urlopen', side_effect=lambda *a, **kw: self.response('स्रोत')):
            out = translate_answer({'answer': '- Source\nhttps://example.org/123\n`abc123`'})
        self.assertEqual(out['answer'], '- स्रोत\nhttps://example.org/123\n`abc123`')

    def test_streamlit_secrets_are_supported(self):
        # Access the real implementation, independently of setUp's settings stub.
        real = self._real_setting
        st = MagicMock()
        st.secrets = {'SARVAM_API_KEY': 'from-secrets'}
        with patch.dict('os.environ', {'SARVAM_API_KEY': ''}), patch.dict('sys.modules', {'streamlit': st}):
            self.assertEqual(real('SARVAM_API_KEY'), 'from-secrets')

    def test_advisor_finalizes_legacy_response(self):
        from app.advisor import RAGAdvisor
        advisor = object.__new__(RAGAdvisor)
        with patch.dict('os.environ', {'KISAANAI_AGENTIC': '0'}), patch.object(advisor, '_answer_legacy', return_value={'answer': 'Apply fertiliser'}), patch('app.hindi_translation.urlopen', return_value=self.response('उर्वरक डालें')):
            self.assertEqual(advisor.answer('question')['answer'], 'उर्वरक डालें')


if __name__ == '__main__':
    unittest.main()
