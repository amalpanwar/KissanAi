# Signup timeout troubleshooting

The app's Supabase HTTP client waits up to 25 seconds for a response. A read timeout does not establish whether signup completed on the server. The app now reports that uncertainty and asks the user to check their inbox before submitting again. Requests are never automatically retried, including verification resends and password resets. The signup form remains in Create Account mode on failure.

For an administrator:

1. Inspect Supabase Auth logs for the signup timestamp. This distinguishes SMTP delivery errors, hook failures, server/database failures and a client network timeout.
2. Check Authentication > Users to see whether the attempted account exists before attempting signup again.
3. For Gmail custom SMTP, confirm host `smtp.gmail.com`, the same Gmail address as sender/username, and a Google App Password. Both 465 (implicit TLS) and 587 (STARTTLS) are supported; trying the other port is diagnostic, not a guaranteed repair. Do not confuse this host with the Google Workspace relay.
4. If an unintended Send Email Hook is enabled, inspect that configuration: an enabled hook handles email instead of SMTP.
5. After correcting the underlying problem, test one signup or resend and inspect its Auth log and inbox.

This code change improves timeout handling; it cannot repair hosted Supabase SMTP settings or confirm delivery. Do not solve delivery failures by bypassing email confirmation.

Sources:
- https://supabase.com/docs/guides/troubleshooting/using-google-smtp-with-supabase-custom-smtp-ZZzU4Y
- https://supabase.com/docs/guides/troubleshooting/not-receiving-auth-emails-from-the-supabase-project-OFSNzw
- https://supabase.com/docs/guides/auth/auth-hooks/send-email-hook
