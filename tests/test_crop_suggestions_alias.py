import unittest

from app.chat_controls import SUGGESTIONS
from app.commodity_lookup import load_aliases, normalize, resolve_commodities


class CropSuggestionAliasTests(unittest.TestCase):
    def test_suggestions_include_sugarcane_and_pearl_millet(self):
        entries = [(item['question'], item['keywords'].casefold()) for item in SUGGESTIONS]
        sugarcane = next((q for q, keys in entries if 'sugarcane' in keys), '')
        pearl_millet = next((q for q, keys in entries if 'pearl millets' in keys), '')
        self.assertIn('गन्ना', sugarcane)
        self.assertIn('बाजरा', pearl_millet)

    def test_pearl_millets_resolves_to_bajra(self):
        aliases = load_aliases()
        resolved = resolve_commodities('pearl millets price', aliases=aliases)
        self.assertIn('bajra(pearl millet/cumbu)', {normalize(item) for item in resolved})


if __name__ == '__main__':
    unittest.main()
