import json
import unittest
from unittest.mock import MagicMock, patch
from app.query_translation import translate_query
from app.chat_controls import consume_submission


class QueryTranslationTests(unittest.TestCase):
    def test_mixed_query_uses_auto_and_preserves_name(self):
        response=MagicMock();response.__enter__.return_value.read.return_value=json.dumps({'translated_text':'ZXQAQXZ में धान की किस्म बताएं'}).encode()
        with patch('app.query_translation._setting',return_value='test-key'), patch('app.query_translation.urlopen',return_value=response) as request:
            result=translate_query('Doghat Rural mein rice variety btaye',['Doghat Rural'])
        self.assertEqual(result['text'],'Doghat Rural में धान की किस्म बताएं')
        body=json.loads(request.call_args.args[0].data)
        self.assertEqual(body['source_language_code'],'auto')
        self.assertEqual(body['model'],'mayura:v1')
        self.assertNotIn('Doghat Rural',body['input'])

    def test_hindi_no_network_and_failed_translation_keeps_original(self):
        with patch('app.query_translation.urlopen',side_effect=TimeoutError()) as request:
            self.assertEqual(translate_query('धान की किस्म बताएं')['status'],'unchanged')
            request.assert_not_called()
            with patch('app.query_translation._setting',return_value='key'):
                result=translate_query('rice varieties')
                self.assertEqual(result,{'status':'unavailable','text':'rice varieties'})

    def test_draft_is_never_a_chat_submission(self):
        state={}
        self.assertIsNone(consume_submission({'kind':'draft','text':'rice','id':'draft1'},state))
        self.assertEqual(state,{})
        self.assertEqual(consume_submission({'text':'धान की किस्म बताएं','id':'sent1'},state),'धान की किस्म बताएं')
        self.assertIsNone(consume_submission({'text':'धान की किस्म बताएं','id':'sent1'},state))
