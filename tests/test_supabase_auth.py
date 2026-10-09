import io
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError
from app import supabase_auth as auth


class AuthTimeoutTests(unittest.TestCase):
    def setUp(self):
        self.cfg = auth.SupabaseConfig('https://example.supabase.co','test-key')

    def signup(self):
        return auth.sign_up(self.cfg,email='test@example.com',password='test-password',
                            username='test',display_name='Test')

    def test_connection_and_wrapped_timeouts_are_not_retried(self):
        for exc in [TimeoutError('The read operation timed out'), URLError(TimeoutError('timed out'))]:
            with self.subTest(error=type(exc).__name__), patch.object(auth,'urlopen',side_effect=exc) as request:
                ok,payload=self.signup()
                self.assertFalse(ok)
                self.assertEqual(payload['code'],'client_timeout')
                self.assertIn('could not confirm whether your account was created',payload['error_description'])
                request.assert_called_once()

    def test_response_body_timeout_is_handled(self):
        response=MagicMock()
        response.__enter__.return_value.read.side_effect=TimeoutError('read')
        with patch.object(auth,'urlopen',return_value=response) as request:
            self.assertEqual(self.signup()[1]['code'],'client_timeout')
            request.assert_called_once()

    def test_http_error_body_timeout_does_not_escape(self):
        body=MagicMock(); body.read.side_effect=TimeoutError('read')
        error=HTTPError('https://example.supabase.co',504,'timeout',{},body)
        with patch.object(auth,'urlopen',side_effect=error):
            self.assertEqual(self.signup()[1]['code'],'client_timeout')

    def test_email_actions_describe_uncertain_delivery(self):
        for operation in [auth.resend_signup_email,auth.send_password_reset_email]:
            with patch.object(auth,'urlopen',side_effect=TimeoutError()) as request:
                ok,payload=operation(self.cfg,'test@example.com')
                self.assertFalse(ok)
                self.assertIn('email delivery could not be confirmed',payload['error_description'])
                request.assert_called_once()

    def test_rate_limit_and_success_are_preserved(self):
        error=HTTPError('https://example.supabase.co',429,'limit',{},
                        io.BytesIO(b'{"code":"over_email_send_rate_limit","msg":"email rate limit exceeded"}'))
        with patch.object(auth,'urlopen',side_effect=error):
            ok,payload=self.signup()
            self.assertFalse(ok)
            self.assertEqual(payload['code'],'over_email_send_rate_limit')
        response=MagicMock()
        response.__enter__.return_value.read.return_value=b'{"user":{"id":"123"}}'
        with patch.object(auth,'urlopen',return_value=response):
            self.assertEqual(self.signup(),(True,{'user':{'id':'123'}}))


if __name__=='__main__':
    unittest.main()
