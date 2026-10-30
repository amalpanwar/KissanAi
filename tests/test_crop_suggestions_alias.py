import json
import unittest
from pathlib import Path

from app.chat_controls import SUGGESTIONS


class CropSuggestionAliasTests(unittest.TestCase):
    def test_suggestions_include_sugarcane_and_pearl_millet(self):
        entries = [(item['question'], item['keywords'].casefold()) for item in SUGGESTIONS]
        sugarcane = next((q for q, keys in entries if 'sugarcane' in keys), '')
        pearl_millet = next((q for q, keys in entries if 'pearl millets' in keys), '')
        self.assertIn('गन्ना', sugarcane)
        self.assertIn('बाजरा', pearl_millet)

    def test_pearl_millets_alias_is_in_production_dictionary(self):
        path = Path('data/raw/commodity_aliases.json')
        aliases = json.loads(path.read_text(encoding='utf-8'))
        bajra = {value.casefold() for value in aliases['bajra(pearl millet/cumbu)']}
        self.assertIn('pearl millets', bajra)


if __name__ == '__main__':
    unittest.main()
