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
        self.assertEqual(body['output_script'],'fully-native')
        self.assertNotIn('Doghat Rural',body['input'])

    def test_hindi_no_network_and_failed_translation_keeps_original(self):
        with patch('app.query_translation.urlopen',side_effect=TimeoutError()) as request:
            self.assertEqual(translate_query('धान की किस्म बताएं')['status'],'unchanged')
            request.assert_not_called()
            with patch('app.query_translation._setting',return_value='key'):
                result=translate_query('rice varieties')
                self.assertEqual(result,{'status':'timeout','text':'rice varieties'})

    def test_draft_is_never_a_chat_submission(self):
        state={}
        self.assertIsNone(consume_submission({'kind':'draft','text':'rice','id':'draft1'},state))
        self.assertEqual(state,{})
        self.assertEqual(consume_submission({'text':'धान की किस्म बताएं','id':'sent1'},state),'धान की किस्म बताएं')
        self.assertIsNone(consume_submission({'text':'धान की किस्म बताएं','id':'sent1'},state))


class TranslationFailureTests(unittest.TestCase):
    def test_http_failures_are_actionable_and_do_not_expose_secrets(self):
        from urllib.error import HTTPError
        for code, expected in [(401,'authentication_failed'),(403,'authentication_failed'),
                               (422,'invalid_request'),(429,'rate_limited'),(500,'service_error')]:
            with self.subTest(code=code), patch('app.query_translation._setting',return_value='secret'), patch('app.query_translation.urlopen',side_effect=HTTPError('https://api.sarvam.ai/translate',code,'sensitive body',{},None)):
                result=translate_query('rice varieties')
                self.assertEqual(result,{'status':expected,'text':'rice varieties'})

    def test_missing_key_and_changed_entities_are_distinct(self):
        with patch('app.query_translation._setting',return_value=''):
            self.assertEqual(translate_query('rice varieties')['status'],'not_configured')
        response=MagicMock();response.__enter__.return_value.read.return_value=json.dumps({'translated_text':'डोगहट में धान की किस्म'}).encode()
        with patch('app.query_translation._setting',return_value='key'), patch('app.query_translation.urlopen',return_value=response):
            self.assertEqual(translate_query('Doghat rice variety',['Doghat'])['status'],'protected_values_changed')
