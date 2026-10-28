# Bilingual production-guide response cache

Static guide answers are cached on demand after the normal source lookup, reranking, English generation and Hindi translation flow. Each entry stores the source-grounded English answer, successful Hindi answer, translation metadata and selected PDF page evidence. The chat shows an optional English answer expander and PDF page references for fresh and saved replies.

## Reuse and invalidation

Common generic Hindi/Hinglish/English cultivation questions share a key after resolving crop aliases to the same PDF section. Other phrasings use the normalized full question. Extra qualifiers, focused questions, context/location and different crops keep distinct keys. Weather, market, news, pesticide-tool routes and time-relative questions (for example today/this week) bypass this cache. Answers containing references outside the guide are not stored.

The English key fingerprints PDF contents, commodity aliases, relevant generation/retrieval implementation files and the generation model. PDF/alias/generation changes cause a fresh lookup and answer. Hindi has a separate fingerprint for translation code/version, technical glossary, pesticide-name vocabulary, translation settings and model. A Hindi-only change keeps the saved English answer and rebuilds its translation. Updating unrelated rows in the main application database does not change the vocabulary fingerprint.

A failed translation never becomes a reusable Hindi entry. Its valid English answer can be retained; the next request reuses English and retries only translation. Generation failures and mixed web/news evidence are not saved. Hindi reuse requires successful translation metadata and Hindi prose. These checks do not establish linguistic accuracy or replace human review.

The cache returns source/page evidence rather than replaying a previous caller's agent trace. Raw questions, context strings and agent traces are not stored by this cache; questions/context participate only in a SHA-256 key. Generated answers can still contain details relevant to the question. Runtime SQLite files are ignored by Git and created with owner-only permissions.

## Storage and controls

No new package or API credentials are required. Defaults:

```toml
KISAANAI_GUIDE_RESPONSE_CACHE = "1"
# Optional writable SQLite file; otherwise data/processed/guide_response_cache.sqlite3
# KISAANAI_GUIDE_CACHE_PATH = "/persistent/storage/guide_response_cache.sqlite3"
```

Settings can be environment variables or existing Streamlit secrets. Set the enable flag to `0` to bypass the cache. Keep custom cache files outside source control. Remove saved entries locally with:

```sh
python -m app.guide_response_cache --clear
```

The file stores up to 256 entries and evicts the least recently accessed entries. SQLite writes are atomic. A digest check prevents concurrent requests from attaching a Hindi translation to different English text. Concurrent identical requests in one process share a lock; separate processes can still perform duplicate cold provider calls. Storage errors/corrupt files fall back to normal answer generation rather than failing the chat.

Disk storage survives reruns and process restarts only while that filesystem is retained. Streamlit Cloud container replacement/redeployment can erase local files. For retention across those events, supply a genuinely persistent writable path; external database/object-storage integration is not included in this change. This feature does not ship pretranslated Hindi crop data and does not make a cold first translation independent of Sarvam. It starts helping after valid English/Hindi output has populated an entry.

## Verification

Regression tests cover real-guide alias/section keys, bilingual replay with no repeated model calls, page references and English display, focused/context separation, source and language invalidation, translation-only retry, dynamic-route exclusion, corrupt/unwritable storage, concurrent requests/writes, bounded size and matching English/Hindi digests. Provider outputs are mocked; the Sarvam key is configured only in Streamlit Cloud, so live first-fill output remains unverified.

## Cloud-side provider check

After a translation failure, expand **हिंदी अनुवाद की स्थिति** and select **Test Sarvam Hindi services**. This explicitly runs two small fixed-sample checks against Sarvam's `/translate` and `/v1/chat/completions` endpoints using the existing Cloud key. The sample is a test label, not farming advice. No farmer question, actual answer or retrieved evidence is sent in these probes. Results include the raw fixed-sample output, Hindi-presence flag, status and elapsed time; error bodies/headers/credentials are not displayed. Identical checks are reused for up to five minutes per process to limit extra calls. These probes can consume API quota.

A successful sample test does not prove that a longer crop answer will translate correctly. It distinguishes basic endpoint/key availability from the application's masking and validation behavior. The regular failure diagnostic can also show a bounded set of remaining English words only when those words are attested in the public production-guide evidence; other output text and farmer-specific words are not exposed. This diagnostic change does not bypass translation validation or claim to resolve live Hindi output.
