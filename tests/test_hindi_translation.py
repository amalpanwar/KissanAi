import json
import unittest
import threading
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
            self.assertNotIn('numerals_format', payload)
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

    def test_written_number_equivalents_are_not_invented_quantities(self):
        for source, output in [('Apply two equal doses.', '२ समान मात्रा में डालें।'),
                               ('Use half the nitrogen.', '0.5 नाइट्रोजन का उपयोग करें।'),
                               ('Wait twenty-five days.', '25 दिन प्रतीक्षा करें।')]:
            with self.subTest(source=source), patch('app.hindi_translation.urlopen', return_value=self.response(output)):
                out = translate_answer({'answer': source})
            self.assertEqual(out['translation']['status'], 'translated')

    def test_duplicate_reordered_and_negative_written_quantities_are_rejected(self):
        for output in ['2 और 2 मात्रा डालें।', '-2 मात्रा डालें।', '3 और 2 मात्रा डालें।']:
            with self.subTest(output=output), patch('app.hindi_translation.urlopen', return_value=self.response(output)):
                out = translate_answer({'answer': 'Apply two doses then three doses.'})
            self.assertEqual(out['translation']['status'], 'fallback')
            self.assertEqual(out['translation']['reason'], 'new_numbers')

    def test_remaining_english_retries_native_sarvam_model_without_values(self):
        def server(request, timeout):
            payload = json.loads(request.data)
            if payload['model'] == 'sarvam-translate:v1':
                tokens = TOKEN.findall(payload['input'])
                return self.response('बुवाई से पहले compost डालें ' + ' '.join(tokens))
            self.assertEqual(payload['model'], 'mayura:v1')
            self.assertEqual(payload['output_script'], 'fully-native')
            self.assertLessEqual(len(payload['input']), 1000)
            self.assertNotIn('750', payload['input'])
            self.assertFalse(TOKEN.search(payload['input']))
            return self.response('गोबर की खाद मिट्टी में मिलाएँ')
        with patch('app.hindi_translation.urlopen', side_effect=server):
            out = translate_answer({'answer': 'Apply 750 kg FYM before sowing.', 'references': ['guide.pdf']})
        self.assertEqual(out['translation']['status'], 'translated')
        self.assertEqual(out['translation']['retry_model'], 'mayura:v1')
        self.assertIn('750 kg FYM', out['answer'])
        self.assertNotIn('compost', out['answer'])
        self.assertEqual(out['references'], ['guide.pdf'])

    def test_changed_placeholders_retry_prose_without_sending_values(self):
        inputs = []
        def server(request, timeout):
            text = json.loads(request.data)['input']
            inputs.append(text)
            if TOKEN.search(text):
                return self.response('प्रदाता ने प्लेसहोल्डर बदल दिए')
            return self.response('गोबर की खाद डालें')
        with patch('app.hindi_translation.urlopen', side_effect=server):
            out = translate_answer({'answer': 'Apply 750 kg FYM before sowing.', 'references': ['guide.pdf']})
        self.assertEqual(out['translation']['status'], 'translated')
        self.assertIn('750 kg FYM', out['answer'])
        self.assertEqual(out['references'], ['guide.pdf'])
        self.assertGreater(len(inputs), 1)
        self.assertTrue(all(not TOKEN.search(text) for text in inputs[1:]))
        self.assertTrue(all('750' not in text for text in inputs))

    def test_prose_retry_rejects_new_numbers(self):
        def server(request, timeout):
            text = json.loads(request.data)['input']
            return self.response('बदले हुए प्लेसहोल्डर' if TOKEN.search(text) else '999 किग्रा डालें')
        with patch('app.hindi_translation.urlopen', side_effect=server):
            out = translate_answer({'answer': 'Apply 750 kg FYM.'})
        self.assertEqual(out['translation']['status'], 'fallback')
        self.assertNotIn('999', out['answer'])

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
        with patch('app.hindi_translation.urlopen') as call:
            source = {'answer': 'Apply fertiliser. ' * 3000}
            out = translate_answer(source)
        self.assertEqual(out['translation']['status'], 'fallback')
        self.assertEqual(out['answer'], source['answer'])
        self.assertEqual(call.call_count, 0)
        self.assertGreater(out['translation']['request_chunks'], 24)
        self.assertEqual(out['translation']['reason'], 'request_limit')

    def test_many_formatted_lines_are_batched_not_counted_as_requests(self):
        source = '# Wheat cultivation\n\n' + '\n'.join(f'- Apply {i} kg FYM.' for i in range(1, 36))
        requests = []
        def server(request, timeout):
            text = json.loads(request.data)['input']
            requests.append(text)
            self.assertLessEqual(len(text), 1800)
            return self.response(text.replace('Wheat cultivation', 'गेहूं की खेती').replace('Apply', 'डालें'))
        with patch('app.hindi_translation.urlopen', side_effect=server):
            out = translate_answer({'answer': source, 'references': ['guide.pdf']})
        self.assertEqual(out['translation']['status'], 'translated')
        self.assertEqual(out['translation']['request_chunks'], len(requests))
        self.assertLess(len(requests), 4)
        self.assertNotIn('Apply', out['answer'])
        self.assertFalse(TOKEN.search(out['answer']))
        for i in range(1, 36):
            self.assertIn(f'{i} kg FYM', out['answer'])
        self.assertEqual(out['references'], ['guide.pdf'])

    def test_primary_latency_does_not_consume_native_retry_budget(self):
        clock = [0.0]
        def server(text, key, timeout, model='sarvam-translate:v1'):
            if model == 'sarvam-translate:v1':
                self.assertEqual(timeout, 15.0)
                clock[0] += 12.0
                return 'खाद compost डालें'
            self.assertEqual(model, 'mayura:v1')
            self.assertEqual(timeout, 15.0)
            clock[0] += 10.0
            return 'खाद डालें'
        with patch('app.hindi_translation.time.monotonic', side_effect=lambda: clock[0]), patch('app.hindi_translation._request_translation', side_effect=server):
            out = translate_answer({'answer': 'Apply compost'})
        self.assertEqual(out['translation']['status'], 'translated')
        self.assertEqual(out['translation']['version'], 'sarvam-hindi-v13')
        self.assertEqual(out['translation']['retry_model'], 'mayura:v1')

    def test_chunks_run_concurrently_and_preserve_source_order(self):
        barrier = threading.Barrier(2)
        def server(text, key, timeout, model='sarvam-translate:v1'):
            barrier.wait(timeout=2)
            return {'Apply compost': 'खाद डालें', 'Irrigate regularly': 'नियमित सिंचाई करें'}[text]
        with patch('app.hindi_translation._request_translation', side_effect=server):
            out = translate_answer({'answer': '- Apply compost\nसिंचाई\n- Irrigate regularly'})
        self.assertEqual(out['translation']['status'], 'translated')
        self.assertEqual(out['answer'], '- खाद डालें\nसिंचाई\n- नियमित सिंचाई करें')

    def test_queued_chunks_cannot_start_after_answer_deadline(self):
        clock = [0.0]
        def server(text, key, timeout, model='sarvam-translate:v1'):
            clock[0] += 61.0
            return 'खाद डालें'
        with patch('app.hindi_translation.time.monotonic', side_effect=lambda: clock[0]), patch('app.hindi_translation.TRANSLATION_WORKERS', 1), patch('app.hindi_translation._request_translation', side_effect=server) as call:
            out = translate_answer({'answer': 'Apply compost\nसिंचाई\nIrrigate regularly'})
        self.assertEqual(call.call_count, 1)
        self.assertEqual(out['translation']['reason'], 'translation_deadline')
        self.assertEqual(out['answer'], 'Apply compost\nसिंचाई\nIrrigate regularly')

    def test_quantity_rich_guide_uses_one_validated_chat_translation(self):
        source = 'Apply 10 kg before sowing, add 20 kg after sowing, irrigate at 30 days, weed at 40 days, harvest at 50 days.'
        def chat(prompt, **kwargs):
            fragments = json.loads(prompt.split('\n', 1)[1])['fragments']
            self.assertNotIn('10 kg', prompt)
            self.assertFalse(TOKEN.search(prompt))
            self.assertEqual(kwargs['response_format'], {'type': 'json_object'})
            return json.dumps({'translations': {key: 'खाद डालें' for key in fragments}})
        with patch('app.hindi_translation._request_translation', return_value='प्रदाता ने प्लेसहोल्डर बदल दिए') as primary, patch(
                'app.sarvam_generation.generate_text', side_effect=chat) as fallback:
            out = translate_answer({'answer': source, 'references': ['guide.pdf']})
        self.assertEqual(out['translation']['status'], 'translated')
        self.assertEqual(out['translation']['retry_model'], 'sarvam-105b')
        self.assertEqual(primary.call_count, 1)
        self.assertEqual(fallback.call_count, 1)
        for value in ['10 kg', '20 kg', '30', '40', '50']:
            self.assertIn(value, out['answer'])
        self.assertEqual(out['references'], ['guide.pdf'])

    def test_quantity_rich_chat_translation_still_rejects_changed_values(self):
        source = 'Apply 10 kg then 20 kg then 30 kg then 40 kg then 50 kg.'
        with patch('app.hindi_translation._request_translation', return_value='बदले हुए टोकन'), patch(
                'app.sarvam_generation.generate_text', return_value='999 किग्रा डालें'):
            out = translate_answer({'answer': source})
        self.assertEqual(out['translation']['status'], 'fallback')
        self.assertEqual(out['translation']['reason'], 'invalid_translation_structure')
        self.assertEqual(out['answer'], source)

    def test_chat_retry_rejects_new_numbers_and_remaining_english(self):
        source = 'Apply 10 kg then 20 kg then 30 kg then 40 kg then 50 kg.'
        for extra, reason in [('999', 'new_numbers'), ('compost', 'untranslated_text')]:
            def chat(prompt, **kwargs):
                fragments = json.loads(prompt.split('\n', 1)[1])['fragments']
                return json.dumps({'translations': {key: 'खाद डालें ' + extra for key in fragments}})
            with self.subTest(reason=reason), patch('app.hindi_translation._request_translation', return_value='बदले हुए टोकन'), patch(
                    'app.sarvam_generation.generate_text', side_effect=chat):
                out = translate_answer({'answer': source})
            self.assertEqual(out['translation']['status'], 'fallback')
            self.assertEqual(out['translation']['reason'], reason)
            self.assertEqual(out['answer'], source)

    def test_structured_fallback_keeps_names_and_values_entirely_local(self):
        source = 'Apply Urea at 50 kg/ha before sowing, use Carbendazim 50% WP at 2 g/kg seed, and select COW(W)1.'
        def chat(prompt, **kwargs):
            fragments = json.loads(prompt.split('\n', 1)[1])['fragments']
            for value in ['Urea', 'Carbendazim', '50 kg/ha', '2 g/kg', 'COW(W)1']:
                self.assertNotIn(value, prompt)
            self.assertFalse(TOKEN.search(prompt))
            # JSON key ordering must not change original name/dose associations.
            return json.dumps({'translations': {key: 'स्रोत के अनुसार उपयोग करें' for key in reversed(fragments)}})
        with patch('app.hindi_translation._request_translation', return_value='सभी प्लेसहोल्डर बदल गए'), patch(
                'app.sarvam_generation.generate_text', side_effect=chat):
            out = translate_answer({'answer': source, 'retrieved': [{'text': 'Variety COW(W)1'}]})
        self.assertEqual(out['translation']['status'], 'translated')
        last = -1
        for value in ['Urea', '50 kg/ha', 'Carbendazim', '50%', 'WP', '2 g/kg', 'COW(W)1']:
            position = out['answer'].index(value, last + 1)
            self.assertGreater(position, last)
            last = position

    def test_structured_fallback_rejects_missing_unknown_duplicate_and_empty_fragments(self):
        source = 'Apply 10 kg then 20 kg then 30 kg then 40 kg then 50 kg.'
        for kind in ['missing', 'unknown', 'duplicate', 'empty', 'extra_field', 'malformed']:
            def chat(prompt, **kwargs):
                fragments = json.loads(prompt.split('\n', 1)[1])['fragments']
                values = {key: 'खाद डालें' for key in fragments}
                if kind == 'missing': values.pop(next(iter(values)))
                if kind == 'unknown': values['unknown'] = 'खाद डालें'
                if kind == 'empty': values[next(iter(values))] = ''
                if kind == 'duplicate': return '{"translations":{"0":"खाद", "0":"खाद"}}'
                if kind == 'malformed': return 'खाद डालें'
                data = {'translations': values}
                if kind == 'extra_field': data['notes'] = 'खाद'
                return json.dumps(data)
            with self.subTest(kind=kind), patch('app.hindi_translation._request_translation', return_value='बदले हुए टोकन'), patch(
                    'app.sarvam_generation.generate_text', side_effect=chat):
                out = translate_answer({'answer': source})
            self.assertEqual(out['translation']['status'], 'fallback')
            self.assertEqual(out['translation']['reason'], 'invalid_translation_structure')
            self.assertEqual(out['answer'], source)

    def test_protected_only_chunks_skip_api_without_needing_hindi(self):
        with patch('app.hindi_translation._request_translation') as call:
            value = _translate('ZXQAQXZ, ZXQBQXZ.', 'test-key', 30)
        self.assertEqual(value, 'ZXQAQXZ, ZXQBQXZ.')
        call.assert_not_called()

    def test_long_technical_tail_does_not_block_hindi_prose(self):
        source = 'Apply compost. ' + ' '.join(f'{i} kg FYM' for i in range(1, 401))
        def server(request, timeout):
            text = json.loads(request.data)['input']
            return self.response('खाद डालें। ' + ' '.join(TOKEN.findall(text)))
        with patch('app.hindi_translation.urlopen', side_effect=server) as call:
            out = translate_answer({'answer': source})
        self.assertEqual(out['translation']['status'], 'translated')
        self.assertGreater(out['translation']['request_chunks'], 1)
        self.assertEqual(call.call_count, 1)
        self.assertIn('400 kg FYM', out['answer'])

    def test_native_connector_symbol_does_not_need_its_own_hindi_letters(self):
        def server(request, timeout):
            payload = json.loads(request.data)
            if payload['model'] == 'sarvam-translate:v1':
                return self.response('बदले हुए प्लेसहोल्डर')
            return self.response('&' if payload['input'].strip() == 'and' else 'उपयोग करें')
        with patch('app.hindi_translation.urlopen', side_effect=server):
            out = translate_answer({'answer': 'Apply Urea and DAP.'})
        self.assertEqual(out['translation']['status'], 'translated')
        self.assertIn('Urea & DAP', out['answer'])

    def test_structured_connector_and_written_number_fragments_can_be_non_hindi(self):
        source = 'Apply Urea two DAP and SSP before sowing ZnSO4 after mixing FYM.'
        def chat(prompt, **kwargs):
            fragments = json.loads(prompt.split('\n', 1)[1])['fragments']
            return json.dumps({'translations': {key: '2' if text.strip() == 'two' else '&' if text.strip() == 'and' else 'स्रोत के अनुसार डालें' for key, text in fragments.items()}})
        with patch('app.hindi_translation._request_translation', return_value='बदले हुए प्लेसहोल्डर'), patch(
                'app.sarvam_generation.generate_text', side_effect=chat):
            out = translate_answer({'answer': source})
        self.assertEqual(out['translation']['status'], 'translated')
        self.assertIn('Urea 2 DAP & SSP', out['answer'])

    def test_punctuation_cannot_replace_a_meaningful_instruction(self):
        with patch('app.hindi_translation.urlopen', return_value=self.response('।')):
            out = translate_answer({'answer': 'Apply fertiliser before sowing.'})
        self.assertEqual(out['translation']['status'], 'fallback')
        self.assertEqual(out['translation']['reason'], 'hindi_missing')
        self.assertEqual(out['translation']['failure_scope'], 'chunk')
        self.assertEqual(out['translation']['source_fragment_kind'], 'prose')

    def test_numeric_fragment_may_include_sentence_punctuation(self):
        with patch('app.hindi_translation.urlopen', return_value=self.response('2.')):
            out = translate_answer({'answer': 'दो मात्रा में से two.'})
        self.assertEqual(out['translation']['status'], 'translated')
        self.assertEqual(out['answer'], 'दो मात्रा में से 2.')

    def test_connector_only_output_cannot_be_a_complete_hindi_answer(self):
        with patch('app.hindi_translation.urlopen', return_value=self.response('&')):
            out = translate_answer({'answer': 'and'})
        self.assertEqual(out['translation']['status'], 'fallback')
        self.assertEqual(out['translation']['reason'], 'hindi_missing')
        self.assertEqual(out['translation']['failure_scope'], 'answer')

    def test_untranslated_diagnostics_only_show_public_guide_words(self):
        from app.production_guide_evidence import GUIDE
        with patch('app.hindi_translation.urlopen', return_value=self.response('rabi PrivateFarmerName')):
            out = translate_answer({'answer': 'Prepare the field.',
                'retrieved': [{'source_file': str(GUIDE), 'text': 'rabi sowing irrigation'}]})
        self.assertEqual(out['translation']['reason'], 'untranslated_text')
        self.assertEqual(out['translation']['failure_scope'], 'native_fragment')
        self.assertEqual(out['translation']['remaining_latin_count'], 2)
        self.assertEqual(out['translation']['remaining_latin_words'], ['rabi'])
        self.assertNotIn('PrivateFarmerName', str(out['translation']))

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
