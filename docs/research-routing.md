# Evidence lookup for unfamiliar questions

The document research path now uses the configured multilingual embedding model and saved index rather than the earlier bilingual keyword-overlap filter. See [multilingual-document-search.md](multilingual-document-search.md) for the query-language pipeline, score threshold, index requirements, and limitations.

Document agents search first. If no document passage qualifies, the web agent checks official sources and ranks snippets with the same encoder. The response uses cited source excerpts and the existing Hindi output translator. Missing indexes, failed translation, and unavailable providers are diagnosed separately from no matching evidence; weak-match topic suggestions are not used.

Location extraction still requires complete known names. Translation, agent activity, and citations remain below the answer. Conversation storage and training are separate future work. The ten-minute reconnect window still does not persist sessions across server restarts.
