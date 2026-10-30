import unittest
from unittest.mock import MagicMock, patch

from app.hindi_translation import (
    TOKEN, TranslationValidationError, _local_translation_generator,
    _translate, translate_answer, translation_stream_callback,
)


class TranslationTests(unittest.TestCase):
    def setUp(self):
        _translate.cache_clear()
        _local_translation_generator.cache_clear()
        settings = {'KISAANAI_HINDI_TRANSLATION': '1'}
        patcher = patch('app.hindi_translation._setting',
                        side_effect=lambda key, default='': settings.get(key, default))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_local_model_translates_and_keeps_quantities_terms_and_references(self):
        def local(text, key='', timeout=30, model='local-instruct'):
            tokens = TOKEN.findall(text)
            return 'बुवाई से पहले ' + ' '.join(tokens) + ' मिट्टी में डालें।'
        original = {
            'answer': 'Apply 750 kg FYM before sowing.',
            'references': ['Crop Production Guide, page 135'],
            'agent_trace': {'private': 'must stay in the app'},
        }
        with patch('app.hindi_translation._request_translation', side_effect=local) as request:
            result = translate_answer(original)
        self.assertEqual(result['translation']['provider'], 'local')
        self.assertEqual(result['translation']['status'], 'translated')
        self.assertIn('750 kg FYM', result['answer'])
        self.assertNotIn('Apply', result['answer'])
        self.assertEqual(result['references'], original['references'])
        self.assertEqual(result['agent_trace'], original['agent_trace'])
        request.assert_called()

    def test_local_prompt_requests_hindi_and_never_calls_sarvam_generation(self):
        model = MagicMock()
        model.generate.return_value = 'खेत तैयार करें।'
        with patch('app.hindi_translation._local_translation_generator', return_value=model), patch(
                'app.sarvam_generation.generate_text', side_effect=AssertionError('Sarvam called')):
            from app.hindi_translation import _request_translation
            translated = _request_translation('Prepare the field.')
        self.assertEqual(translated, 'खेत तैयार करें।')
        prompt = model.generate.call_args.args[0]
        self.assertIn('Hindi', prompt)
        self.assertIn('Prepare the field.', prompt)

    def test_hindi_and_formula_only_skip_local_model(self):
        with patch('app.hindi_translation._request_translation') as call:
            result = translate_answer({'answer': 'गेहूं में 80:40:40 NPK किग्रा/हेक्टेयर।'})
        call.assert_not_called()
        self.assertEqual(result['translation']['status'], 'not_needed')

    def test_new_numbers_are_rejected(self):
        with patch('app.hindi_translation._request_translation', return_value='999 किलो डालें।'):
            result = translate_answer({'answer': 'Apply 750 kg.'})
        self.assertEqual(result['translation']['status'], 'fallback')
        self.assertEqual(result['translation']['reason'], 'new_numbers')
        self.assertEqual(result['answer'], 'Apply 750 kg.')

    def test_changed_placeholders_retry_only_unmasked_local_prose(self):
        seen = []
        def local(text, key='', timeout=30, model='local-instruct'):
            seen.append(text)
            if TOKEN.search(text):
                return 'बदले हुए placeholder'
            return 'खेत में खाद डालें।' if 'Apply' in text else 'बुवाई से पहले।'
        with patch('app.hindi_translation._request_translation', side_effect=local):
            result = translate_answer({'answer': 'Apply 750 kg FYM before sowing.'})
        self.assertEqual(result['translation']['status'], 'translated')
        self.assertIn('750 kg FYM', result['answer'])
        self.assertTrue(any(TOKEN.search(text) for text in seen))
        self.assertTrue(any(not TOKEN.search(text) for text in seen[1:]))

    def test_local_model_failure_keeps_source_grounded_english_answer_visible(self):
        with patch('app.hindi_translation._request_translation', side_effect=TimeoutError('internal detail')):
            result = translate_answer({'answer': 'Prepare the field.'})
        self.assertEqual(result['translation']['status'], 'fallback')
        self.assertEqual(result['answer'], 'Prepare the field.')
        self.assertNotIn('internal detail', str(result['translation']))

    def test_partial_translation_streams_successful_fragments(self):
        updates = []
        def local(text, key='', timeout=30, model='local-instruct'):
            if 'Apply compost' in text:
                raise TranslationValidationError('local_model_error')
            return 'नियमित सिंचाई करें।'
        with patch('app.hindi_translation._request_translation', side_effect=local):
            with translation_stream_callback(updates.append):
                result = translate_answer({'answer': 'Apply compost.\nसिंचाई करें।\nIrrigate regularly.'})
        self.assertEqual(result['translation']['status'], 'partial')
        self.assertIn('Apply compost.', result['answer'])
        self.assertIn('नियमित सिंचाई करें।', result['answer'])
        self.assertTrue(updates)
        self.assertEqual(updates[-1], result['answer'])

    def test_translation_can_be_disabled_without_loading_local_model(self):
        with patch('app.hindi_translation._setting', return_value='0'), patch(
                'app.hindi_translation._local_translation_generator') as generator:
            result = translate_answer({'answer': 'Prepare the field.'})
        self.assertEqual(result['translation']['status'], 'disabled')
        generator.assert_not_called()


if __name__ == '__main__':
    unittest.main()
