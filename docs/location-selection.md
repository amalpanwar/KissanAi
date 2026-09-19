# Town and village selection

The location controls now provide State → District → Town / Village. Entries come from `data/processed/location_lookup.csv` and include a tehsil label to distinguish duplicate names. The current table covers 8,785 rows across 10 Uttar Pradesh districts, not the whole state. Town selection is disabled where the table has no coverage; district-level queries remain available.

Selections use the complete state/district/tehsil/place identity. Changing state or district clears any incompatible place. The full hierarchy is passed to the specialist agents, preventing a same-name village elsewhere from replacing the selected location. An invalid or ambiguous hierarchy prompts reselection rather than guessing.

Weather uses the selected place's coordinates where available. Coordinates copied from tehsil/district centres are labelled as regional weather. Geocoder addresses that contradict the selected state/district are rejected, with a labelled regional fallback if available. This does not independently verify every coordinate in the source table.

Agmarknet prices are market observations. The price agent and price panel use a matching town-market name inside the selected state/district where available. Otherwise they explicitly show district markets. The app does not invent village-level prices or claim that a market is geographically nearest. Crop-price queries filter the commodity before choosing a town-market or district fallback. Advanced forecasts use the same location scope.

Validation includes Streamlit AppTest selection changes, duplicate village names across districts and tehsils, scoped weather requests, coordinate fallback disclosure, and town-market/district-market selection.
