# Commodity price resolution

Price tools and the Streamlit commodity selectors share `app/commodity_lookup.py`. It reads the existing `data/raw/commodity_aliases.json` dictionary (564 commodity entries), the official commodity catalog, and names present in the current market CSV. Adding a new catalog name does not require a new crop-specific code branch. Translated/local names still require a maintained alias.

Matching uses normalized complete token phrases, retaining Hindi vowel marks. Longer overlapping phrases win: `green chilli` does not become red chilli, `pineapple` does not become apple, and `price` does not become rice. There is no fuzzy auto-selection that can substitute a different commodity. Unknown names request clarification. Missing records return an explicit unavailable result. Multiple recognized crops are kept separately, with any missing commodity named in the answer.

Market rows are filtered to exact resolved commodity names and the selected state/district/place scope. A missing mandi name is labelled as unavailable, not discarded. Non-positive/non-finite prices, invalid dates, and future-dated rows are excluded. Each stale observation is labelled so an older crop is not silently removed merely because another requested crop has newer data. Unreadable, empty, or malformed CSV files produce a data-unavailable response.

Sugarcane's official purchase notice path is separate from daily mandi observations; see `sugarcane-prices.md` for season limitations and sources. The catalog resolver does not create missing market records or guarantee fresh prices at village level.

Regression tests include Hindi/Hinglish/English tomato queries through the coordinator, dynamic catalog additions, all canonical names in the production dictionary, overlapping names, multi-crop queries, corrupt data, and stale/missing records. Runtime deployment errors still require their actual traceback to diagnose; no unobserved deployment error is claimed fixed by these tests.
