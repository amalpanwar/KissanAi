from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from app.hindi_translation import translate_answer, _translate, TOKEN, _protect, _protected_names
from app.translation_terms import _glossary, _pesticides


class TechnicalTranslationTests(unittest.TestCase):
    def setUp(self):
        _translate.cache_clear()
        settings = patch('app.hindi_translation._setting', side_effect=lambda k, d='': 'test-key' if k == 'SARVAM_API_KEY' else d)
        settings.start(); self.addCleanup(settings.stop)



    def test_hindi_sentences_keep_english_names_varieties_and_quantities(self):
        source = 'Use Urea at 50 kg/ha and Carbendazim 50% WP at 2 g/kg seed for COW(W)1.'
        result = {'answer': source, 'retrieved': [{'text': 'Variety: COW(W)1'}], 'references': ['guide.pdf']}
        def local(text, *args):
            self.assertNotIn('Carbendazim', text)
            self.assertNotIn('Urea', text)
            self.assertNotIn('COW(W)1', text)
            return 'इनका उपयोग करें: ' + ' '.join(TOKEN.findall(text)) + '। बीज उपचार स्रोत के अनुसार करें।'
        with patch('app.hindi_translation._request_translation', side_effect=local) as call:
            out = translate_answer(result)
        self.assertEqual(out['translation']['status'], 'translated')
        self.assertEqual(call.call_count, 1)
        for value in ['Urea', 'Carbendazim', 'WP', 'COW(W)1', '50 kg/ha', '2 g/kg']:
            self.assertIn(value, out['answer'])
        self.assertNotIn('Use', out['answer'])
        self.assertEqual(out['references'], ['guide.pdf'])

    def test_actual_wheat_variety_name_is_preserved_as_one_term(self):
        name = 'TNAU Samba Wheat COW 2'
        result = {'answer': 'Select ' + name + '.', 'retrieved': [{'text': 'Variety: ' + name}]}
        masked, values = _protect(result['answer'], _protected_names(result))
        self.assertIn(name, values.values())
        self.assertNotIn('Samba', masked)

    def test_names_from_actual_wheat_guide_are_protected(self):
        from app.production_guide_evidence import lookup, GUIDE
        from types import SimpleNamespace
        if not GUIDE.exists():
            self.skipTest('Guide absent')
        evidence = lookup(SimpleNamespace(_extract_preferred_crop_from_context=lambda c: None),
                          {'question': 'How to grow wheat?'})
        result = {'answer': 'Use TNAU Samba Wheat COW 2 and Urea.',
                  'retrieved': evidence.evidence['records']}
        masked, values = _protect(result['answer'], _protected_names(result))
        self.assertIn('TNAU Samba Wheat COW 2', values.values())
        self.assertIn('Urea', values.values())
        self.assertIn('Use', masked)
        self.assertNotIn('Samba', masked)

    def test_technical_names_do_not_allow_english_instructions(self):
        def local(text, *args):
            if TOKEN.search(text):
                return 'Apply ' + ' '.join(TOKEN.findall(text)) + ' before sowing'
            return text
        with patch('app.hindi_translation._request_translation', side_effect=local):
            out = translate_answer({'answer': 'Apply Urea before sowing.'})
        self.assertEqual(out['translation']['status'], 'fallback')
        self.assertEqual(out['translation']['reason'], 'untranslated_text')

    def test_new_english_chemical_not_in_original_still_rejected(self):
        with patch('app.hindi_translation._request_translation', return_value='Carbendazim डालें'):
            out = translate_answer({'answer': 'Prepare the field.'})
        self.assertEqual(out['translation']['status'], 'fallback')

    def test_explicit_technical_metadata_and_case_are_preserved(self):
        result = {'answer': 'Use customBIO material.', 'protected_terms': ['CustomBio']}
        def local(text, *args):
            return 'इसका उपयोग करें ' + ' '.join(TOKEN.findall(text))
        with patch('app.hindi_translation._request_translation', side_effect=local):
            out = translate_answer(result)
        self.assertEqual(out['translation']['status'], 'translated')
        self.assertIn('customBIO', out['answer'])

    def test_changed_quantity_is_rejected_even_with_chemical_names(self):
        def local(text, *args):
            tokens = TOKEN.findall(text)
            return 'डालें ' + ' '.join(tokens) + ' 999'
        with patch('app.hindi_translation._request_translation', side_effect=local):
            out = translate_answer({'answer': 'Apply Urea at 50 kg/ha.'})
        self.assertEqual(out['translation']['status'], 'fallback')
        self.assertEqual(out['translation']['reason'], 'new_numbers')

    def test_formulation_percentage_and_seed_dose_are_whole_protected_values(self):
        result = {'answer': 'Use Carbendazim 50%WP at 2 g/kg seed.'}
        _, values = _protect(result['answer'], _protected_names(result))
        self.assertIn('50%', values.values())
        self.assertIn('2 g/kg', values.values())

    def test_dictionary_and_pesticide_registry_are_data_driven(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'terms.json'
            path.write_text(json.dumps({'chemicals': ['NewChemical']}))
            self.assertIn('NewChemical', _glossary(str(path), path.stat().st_mtime_ns))
            dbpath = Path(folder) / 'pesticides.db'
            with sqlite3.connect(dbpath) as db:
                db.execute('CREATE TABLE pesticide_recommendations (pesticide_name TEXT)')
                db.executemany('INSERT INTO pesticide_recommendations VALUES (?)', [('NewChemical 20% EC',), ('Apply fertiliser before sowing',)])
            terms = _pesticides(str(dbpath), dbpath.stat().st_mtime_ns)
            self.assertIn('NewChemical', terms)
            self.assertNotIn('Apply fertiliser before sowing', terms)
