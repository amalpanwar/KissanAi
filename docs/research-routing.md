# Evidence lookup for unfamiliar questions

Unclassified questions now route to a research agent without downloading a local planner model. The research agent uses the existing bounded message bus to ask a document agent and a web-search agent. Both are checked before the system declines the question. The general RAG fallback uses the same pipeline.

The document agent scans the indexed shared-document metadata CSV, keeping a bounded shortlist. It does not re-ingest raw uploads: ingest/index a new document to make it searchable. The scan has a five-second/100,000-row limit; an incomplete or unavailable index is reported. The web agent prefers Tavily, with the existing Google search option when Tavily is not configured. Keys can come from environment variables or top-level Streamlit Secrets. It uses authoritative agricultural domains and bounded, cited snippets. No credentials or conversation history are sent to search; the current question is sent.

Matching uses lexical overlap with basic Hindi/Hinglish soil-testing and crop aliases. This is a relevance heuristic, not a semantic guarantee. Strong matches return explicitly labelled source excerpts, not an uncited generated answer. Related-only results list sources and ask whether to proceed. A subsequent explicit yes shows the saved excerpts; a new question discards that offer, and location changes clear it. Missing credentials, failed search, unavailable documents and no relevant evidence have distinct tool outcomes. The user-facing answer says which source check could not complete.

Location lookup now requires complete known names (with optional Rural/Urban suffix), so arbitrary words such as `kare` cannot be expanded into `Kareempur`. Fuzzy location suggestions remain available for explicit weather clarification; they are not automatic evidence of a place in a general agriculture question. Known named places outside the selected district still receive a scope warning.

## Deployment

Set `TAVILY_API_KEY` in Streamlit Secrets to enable Tavily. No key was available for live testing in this change; API behavior is tested with mocks. Document retrieval uses `metadata_store` from `configs/pipeline.yaml`. The app no longer blocks all chat merely because a vector index or document index is missing.

## Connection diagnosis and limits

`server.disconnectedSessionTTL = 600` retains existing sessions for up to ten minutes after a temporary WebSocket disconnect, if the process remains alive and the browser reconnects to the session. This is not durable chat persistence and does not survive a process restart or a fresh browser session.

The existing auth persistence attempts to assign `st.context.cookies`, a read-only request interface. It does not reliably set a browser cookie. This change does not redesign authentication persistence. To diagnose the reported connection error, inspect Streamlit Cloud logs around the event for restarts or memory errors and compare with browser connection behavior. The deployed URL/logs were requested but not yet available. No root cause of the live disconnect is asserted.

Conversation collection, durable conversation storage and training remain a separate task. This change does not feed conversations into model training.
