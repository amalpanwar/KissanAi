import json
import unittest
from pathlib import Path
from app.commodity_lookup import expand_english_query, resolve_commodities

class DocumentAliasTests(unittest.TestCase):
    def setUp(self):
        self.aliases = json.loads(Path('data/raw/commodity_aliases.json').read_text())

    def test_bajra_forms_expand_to_pearl_millet_and_cumbu_from_dictionary(self):
        for name in ['Bajra', 'बाजरा', 'बाजरे', 'bajre', 'Pearl Millet', 'pearl-millet', 'cumbu']:
            with self.subTest(name=name):
                expanded = expand_english_query('How to cultivate this crop?', name, aliases=self.aliases)
                self.assertIn('pearl millet', expanded.lower())
                self.assertIn('cumbu', expanded.lower())
                self.assertEqual(len(resolve_commodities(name, aliases=self.aliases)), 1)

    def test_expansion_is_data_driven_and_avoids_substring_matches(self):
        aliases = {'example': ['custom crop', 'new synonym']}
        self.assertIn('new synonym', expand_english_query('custom crop cultivation', aliases=aliases))
        self.assertEqual(expand_english_query('custom cropper cultivation', aliases=aliases), 'custom cropper cultivation')
