import re
import unittest
from unittest.mock import patch, MagicMock
from app.crop_guide import _summarize_block, _build_guide_points_for_crop
from app.translation_status import render_translation_status

SOURCE = ('If soil test is not available follow blanket recommendation of 80:40:40 NPK kg/ha. '
          'Apply 37.5 kg ZnSO 4, 40 kg S basally for soils having Zn and S deficiencies. '
          'Apply half nitrogen and full phosphorus and potassium basally.')


class CropHindiTests(unittest.TestCase):
    def test_exact_fertilizer_passage_is_hindi_without_api(self):
        answer = _summarize_block('Wheat', 'FERTILIZER APPLICATION', SOURCE)
        self.assertNotIn('basally', answer)
        self.assertNotIn('deficiencies', answer)
        self.assertIn('37.5 किग्रा जिंक सल्फेट', answer)
        self.assertIn('40 किग्रा गंधक बुवाई के समय', answer)
        self.assertIn('80:40:40', answer)
        self.assertFalse(re.search('[A-Za-z]', answer.replace('NPK', '')))

    def test_quantities_are_extracted_not_hardcoded(self):
        answer = _summarize_block('Wheat', 'FERTILIZER APPLICATION', SOURCE.replace('37.5', '21.7').replace('40 kg S', '32 kg S'))
        self.assertIn('21.7 किग्रा जिंक सल्फेट', answer)
        self.assertIn('32 किग्रा गंधक', answer)
        self.assertNotIn('37.5', answer)

    def test_wheat_basal_fertilizer_is_in_sowing_section(self):
        with patch('app.crop_guide._extract_section_text', return_value=SOURCE), \
             patch('app.crop_guide._extract_blocks', return_value=[('FERTILIZER APPLICATION', SOURCE)]), \
             patch('app.crop_guide._append_review_queue'):
            _, entries, _ = _build_guide_points_for_crop('Wheat')
        self.assertEqual(entries[0]['phase'], 'sowing')

    def test_fallback_reason_survives_history_roundtrip(self):
        import json
        item = {'role': 'assistant', 'translation': {'status': 'fallback', 'reason': 'http_403', 'version': 'sarvam-hindi-v2'}}
        restored = json.loads(json.dumps(item))
        st = MagicMock()
        render_translation_status(st, restored['translation'])
        st.caption.assert_called_once()
        self.assertEqual(st.json.call_args.args[0]['reason'], 'http_403')

    def test_missing_key_and_old_answers_have_visible_status(self):
        for info in [None, {'status': 'not_configured'}]:
            st = MagicMock()
            render_translation_status(st, info)
            st.caption.assert_called_once()


if __name__ == '__main__':
    unittest.main()
