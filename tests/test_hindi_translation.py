import unittest
from unittest.mock import patch

from app.hindi_translation import translate_answer, _translate, TOKEN, _chunks, TranslationValidationError


class TranslationTests(unittest.TestCase):
    def setUp(self):
        from app.hindi_translation import _setting
        self.real_setting = _setting
        _translate.cache_clear()
        self.settings = {'HF_TOKEN': 'hf-test-token', 'KISAANAI_HINDI_TRANSLATION': '1'}
        setting = patch('app.hindi_translation._setting', side_effect=lambda key, default='': self.settings.get(key, default))
        setting.start()
        self.addCleanup(setting.stop)

    @staticmethod
    def translated_fragment(text, token=''):
        tokens = TOKEN.findall(text)
        return (' '.join(tokens) + ' ' if tokens else '') + 'हिंदी में अनुवादित वाक्य।'

    def test_english_fragments_use_local_model_and_preserve_source_metadata(self):
        original = {'answer': 'Apply 37.5 kg ZnSO 4 basally before sowing [PDF page 135].',
                    'references': ['Crop Production Guide, page 135'],
                    'agent_trace': {'goal': 'private farmer context'}}
        with patch('app.hindi_translation._translate', side_effect=self.translated_fragment) as translate:
            result = translate_answer(original)
        self.assertEqual(result['translation']['status'], 'translated')
        self.assertEqual(result['translation']['provider'], 'local')
        self.assertEqual(result['translation']['model'], 'ai4bharat/indictrans2-en-indic-dist-200M')
        self.assertIn('37.5 kg ZnSO 4', result['answer'])
        self.assertNotIn('Apply', result['answer'])
        self.assertEqual(result['references'], original['references'])
        self.assertEqual(result['agent_trace'], original['agent_trace'])
        for call in translate.call_args_list:
            self.assertNotRegex(call.args[0], r'37\.5|ZnSO|PDF page')
        self.assertEqual(translate.call_count, 2)

    def test_partial_translation_keeps_success_and_reports_untranslated_fragment(self):
        source = {'answer': 'Prepare the field.\nApply Carbendazim.', 'topic': 'crop_guide'}
        with patch('app.hindi_translation._translate',
                   side_effect=['खेत तैयार करें।', TranslationValidationError('hindi_missing')]) as translate:
            result = translate_answer(source)
        self.assertEqual(result['translation']['status'], 'partial')
        self.assertIn('खेत तैयार करें।', result['answer'])
        self.assertIn('Apply Carbendazim.', result['answer'])
        self.assertIn('Apply', result['translation']['untranslated_fragments'][0]['text'])
        self.assertEqual(translate.call_count, 2)

    def test_hindi_and_formula_only_skip_model(self):
        with patch('app.hindi_translation._translate') as translate:
            result = translate_answer({'answer': 'गेहूं में 80:40:40 NPK किग्रा/हेक्टेयर।'})
        translate.assert_not_called()
        self.assertEqual(result['translation']['status'], 'not_needed')

    def test_bad_model_output_falls_back_without_losing_answer(self):
        source = {'answer': 'Apply 40 kg and 20 kg fertiliser.'}
        for reason in ['protected_values_changed', 'untranslated_text_or_new_numbers', 'empty_response']:
            with self.subTest(reason=reason), patch('app.hindi_translation._translate', side_effect=TranslationValidationError(reason)):
                result = translate_answer(source)
                self.assertEqual(result['answer'], source['answer'])
                self.assertEqual(result['translation']['status'], 'fallback')

    def test_model_exception_is_atomic_and_does_not_leak_details(self):
        with patch('app.hindi_translation._translate', side_effect=RuntimeError('secret detail')):
            result = translate_answer({'answer': 'Apply fertiliser'})
        self.assertEqual(result['answer'], 'Apply fertiliser')
        self.assertEqual(result['translation']['reason'], 'RuntimeError')
        self.assertNotIn('secret detail', str(result))

    def test_missing_token_and_disabled(self):
        for key, value, expected in [
            ('HF_TOKEN', '', 'not_configured'),
            ('KISAANAI_HINDI_TRANSLATION', '0', 'disabled'),
        ]:
            self.settings[key] = value
            with patch('app.hindi_translation._translate') as translate:
                result = translate_answer({'answer': 'Apply fertiliser'})
            self.assertEqual(result['translation']['status'], expected)
            translate.assert_not_called()
            self.settings[key] = 'hf-test-token' if key == 'HF_TOKEN' else '1'

    def test_links_code_and_markdown_are_retained(self):
        with patch('app.hindi_translation._translate', side_effect=self.translated_fragment):
            result = translate_answer({'answer': '- Source\nhttps://example.org/123\n`abc123`'})
        self.assertIn('- ', result['answer'])
        self.assertIn('https://example.org/123', result['answer'])
        self.assertIn('`abc123`', result['answer'])

    def test_streamlit_secret_setting_reads_hf_token(self):
        st = type('StreamlitStub', (), {'secrets': {'HF_TOKEN': 'from-secrets'}})()
        with patch.dict('os.environ', {'HF_TOKEN': ''}), patch.dict('sys.modules', {'streamlit': st}):
            self.assertEqual(self.real_setting('HF_TOKEN'), 'from-secrets')

    def test_chunk_boundaries(self):
        self.assertTrue(all(len(chunk) <= 1800 for chunk in _chunks('word ' * 1000)))


if __name__ == '__main__':
    unittest.main()
