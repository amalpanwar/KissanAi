# Sarvam Hindi translation

The app now uses Sarvam's `/translate` API with `sarvam-translate:v1`, English source, Hindi target and formal mode. It translates remaining English fragments after agent/legacy answers are assembled, keeping existing Hindi. It does not replace retrieval, planning, the local LLM, or source evidence.

## Enable on Streamlit Community Cloud

Open the app's Settings → Secrets and add these top-level TOML entries (above any `[section]` headings):

```toml
SARVAM_API_KEY = "your-sarvam-api-key"
KISAANAI_HINDI_TRANSLATION = "1"
```

Save and restart the app if it does not reload. The code must be deployed before the key takes effect. Get a key from https://dashboard.sarvam.ai/ and ensure the account has API access/credits.

For local Streamlit, add the same entries to the ignored `.streamlit/secrets.toml`. For another host or CLI, set `SARVAM_API_KEY` in its environment. Merely editing `.env` does not export environment variables; your launcher must load it.

A GitHub Actions secret is available only to workflows that explicitly use it. It does not automatically configure Streamlit Community Cloud. Never commit the key in `.env.example`, Python code, or a tracked secrets file.

## Behavior and limits

- Only English fragments of the final answer are submitted; Hindi, question/context, structured references and agent traces are kept local. English names in the answer can be sent.
- Numbers together with adjacent recognized units/formulae, URLs and inline/fenced code are masked. Scientific notation such as `NPK`, `ZnSO4`, and `kg` within quantities intentionally remains unchanged.
- The provider must return every mask exactly once in its original order, without introducing numbers or leaving English prose. Failed checks retain the complete original answer; no partly translated answer is published.
- Each request stays below the API's 2,000-character limit. A response allows at most 24 chunks and a 20-second request budget, with an 8-second timeout per call. Up to 256 successful chunk translations are cached in process memory; failures are not cached.
- `result.translation` records `translated`, `not_needed`, `not_configured`, `disabled`, or `fallback`. The UI displays a Hindi fallback notice when translation is unconfigured or fails. Source answer status and references remain intact.
- Set `KISAANAI_HINDI_TRANSLATION=0` to disable external translation.
- Numeric/formula validation cannot prove semantic correctness (including negation, timing or applicability). Review representative agricultural answers with a Hindi speaker before relying on translation quality. This feature does not repair incorrect source facts or crop-phase headings.

## Validation

Run `python -m unittest discover -s tests -v`. Tests use mocked API responses; they require no key and make no billable calls. Once a key is configured, ask `gehu ki kheti kese kare` and check the fertilizer paragraph. A live translation has not been verified without a configured key.

API contract: https://docs.sarvam.ai/api-reference/text/translate-text

Streamlit secret setup: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management

## Troubleshooting a mixed-language answer

For each newly generated answer, expand **हिंदी अनुवाद की स्थिति**. This diagnostic is stored with chat history, so it survives Streamlit reruns. `version: sarvam-hindi-v2` identifies this implementation. Existing chat replies are not rewritten; ask the question again after deployment.

- `not_configured`: the running app cannot read SARVAM_API_KEY. Put it at the top level of Streamlit Secrets, then restart. A GitHub Actions secret does not configure the app.
- `disabled`: KISAANAI_HINDI_TRANSLATION disables translation.
- `http_401` / `http_403`: check provider credentials/access.
- `http_429`: check the provider's rate limits and account quota.
- `protected_values_changed`: the provider changed or reordered protected placeholders; the translation was rejected.
- `untranslated_text_or_new_numbers`: the response retained English or introduced a number; it was rejected.
- `TimeoutError`: the service timeout or total translation budget was exceeded.
- `not_needed`: there was no remaining English prose requiring Sarvam.

The known wheat zinc/sulphur basal-fertilizer sentence is now rendered completely in Hindi from its extracted source quantities, even without an API key. It is placed in the sowing section. This correction does not prove that the external Sarvam service is configured or functioning.

## Preserve names

Selected village/town, tehsil, district, and state names, resolved weather-header locations, and explicit `protected_names` are masked before translation and restored with their exact source spelling. For example, `Doghat Rural, Baghpat, Uttar Pradesh` stays unchanged while weather descriptions remain Hindi. Recognized names are not submitted in plaintext to Sarvam. Other agents can supply person or organization names through `protected_names`; this is not general named-entity recognition for arbitrary prose. Existing chat replies are unchanged; ask again after deployment. Version: `sarvam-hindi-v3`.

