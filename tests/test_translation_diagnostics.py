import unittest
from unittest.mock import MagicMock
from app.translation_status import render_translation_status


class TranslationDiagnosticTests(unittest.TestCase):
    def test_failure_caption_never_claims_original_is_displayed(self):
        st = MagicMock()
        render_translation_status(st, {'status': 'fallback', 'reason': 'TimeoutError'})
        caption = st.caption.call_args.args[0]
        self.assertNotIn('मूल उत्तर दिखाया', caption)
        self.assertIn('समय पर', caption)

    def test_budget_and_api_timeout_have_distinct_explanations(self):
        captions = []
        for reason in ['request_limit', 'translation_deadline', 'TimeoutError', 'http_403']:
            st = MagicMock()
            render_translation_status(st, {'status': 'fallback', 'reason': reason})
            captions.append(st.caption.call_args.args[0])
            self.assertEqual(st.json.call_args.args[0]['reason'], reason)
        self.assertEqual(len(set(captions)), 4)
