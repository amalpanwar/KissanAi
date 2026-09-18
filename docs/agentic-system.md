# Agent workflow and market refresh

Every `RAGAdvisor.answer` call now creates a goal and a bounded plan, dispatches specialist tasks, evaluates their results, and returns an answer plus `agent_trace`. Streamlit and the CLI use this entry point. The trace exposes task decisions and evidence, not hidden chain-of-thought.

The coordinator uses validated model-generated JSON for ambiguous or mixed requests when the configured model is available. Simple tool queries use deterministic plans. Invalid model output falls back to rules; it cannot introduce arbitrary tools or omit explicitly detected domains. `KISAANAI_LLM_PLANNER=0` disables the model planning step. `KISAANAI_AGENTIC=0` restores the previous application routing.

Specialists:
- Weather: existing location-aware Open-Meteo tools, including the existing cached-data disclosure.
- Prices: dated Agmarknet observations; asks the market-data agent to check freshness before answering. Missing commodities or locations produce clarification requests. Prices are never invented by the language model.
- Market data: bounded official refresh with a 30-minute failed-attempt cooldown and an inter-process lock.
- Pesticides: existing structured pesticide retrieval. Spray queries request weather evidence directly from the weather agent; unavailable weather does not produce a spray-time recommendation.
- Agronomy: existing crop guidance and RAG functionality.

Messages include sender, recipient, task goal, payload, reply ID and result status. Communication is in-process Python, not the external A2A network protocol. Eight tool-agent calls per request and cycle detection bound execution. Mixed answers preserve the specialists' evidence rather than allowing an LLM to rewrite numerical prices or pesticide dosages.

## Refresh

Run from the project root with its installed Python environment:

```sh
python scripts/agmarknet_daily_refresh.py
```

The default `auto` mode tries the dashboard before the report endpoint. Refreshes merge existing history, preserve up to two years, atomically replace the CSV, rebuild the catalog and synchronize SQLite. Partial pagination, request failures and stale report dates are not treated as fresh success. HTTP 400 remains an error; 429 uses bounded backoff. The default geography is the project's configured Western UP district set. `AGMARKNET_ALL_DISTRICTS=1` includes all districts of configured states in dashboard mode. Report mode still uses the configured district filter.

A daily GitHub Actions workflow runs at 06:30 IST once merged into the default branch and enabled by the repository. It needs repository Actions write permissions. It commits validated data and the market database. To generate a macOS schedule for a local environment, run `python scripts/install_agmarknet_scheduler.py`; then install the resulting plist using the printed command. Generate it with the Python environment that has project dependencies installed.

Latest refresh performed during this change: September 17–18, 2026; 1,032 new rows, including 462 dated September 18. The historical GitHub CSV is preserved, giving 221,971 total rows. This refresh does not backfill the entire gap since June. Each answer shows its report date; different commodities may have different latest dates.

## Model assessment

The default remains `Qwen/Qwen2.5-0.5B-Instruct`; complex-model selection was previously unset. The loader now applies the official chat template, supports Apple MPS as well as CUDA/CPU, and uses a configurable 512-token output budget instead of 160. These fix inference integration; they do not establish model accuracy.

The copied `.venv` points to a missing Homebrew Python, and the available test runtime has no Torch/Transformers weights. Therefore live model generation has not been verified. Run:

```sh
python scripts/check_model.py
python scripts/check_model.py --generate
```

For production planning, evaluate a larger instruction model such as `Qwen/Qwen2.5-7B-Instruct` using `KISAANAI_COMPLEX_GENERATOR_MODEL`. Keep the small model as a low-resource fallback only after testing Hindi/Hinglish routing, JSON validity, grounded pesticide answers, tool failures and latency. A model change alone cannot fix missing source documents or stale APIs. Do not assume a larger model is validated just because it loads.

The agentic chat path returns dated price text; the existing separate forecasting controls remain available. Legacy price-chat charts can be restored with `KISAANAI_AGENTIC=0`.

## Validation

```sh
python -m unittest discover -s tests -v
python -m compileall -q app scripts streamlit_app.py
```

Tests cover mixed routing, specialist-to-specialist messages, model-plan validation, bounded cycles/calls, weather failure during spray advice, preservation of existing CSVs after failed/incomplete pagination, corrected-price deduplication, mixed historical/dashboard SQLite import, and chat-template formatting. These are workflow regressions, not a model-quality benchmark.
