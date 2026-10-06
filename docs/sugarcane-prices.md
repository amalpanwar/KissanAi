# Sugarcane price answers

The price specialist recognizes the crop using whole Latin words: `price` must not match `rice`. A sugarcane request uses reviewed official purchase notices in `app/sugarcane_prices.py`, even when the Agmarknet CSV is absent. These are mill purchase policy rates, not a village or mandi transaction quote.

Sources verified on 6 October 2026:

- Uttar Pradesh government cabinet notice dated 14 November 2025, page 3: 2025–26 SAP ₹400/quintal (early), ₹390 (general), ₹355 (unsuitable).
- PIB release 2258142, 5 May 2026: 2026–27 FRP ₹365/quintal at 10.25% basic recovery, applicable October 2026–September 2027. Actual FRP depends on recovery.

The reply includes source links, seasons and the fixed verification date. The 2025–26 UP rate is explicitly historical in 2026–27; the app states that its verified records do not contain the new UP SAP notification. Central FRP is not substituted for UP SAP. Other states do not receive UP rates. Once a stored season expires, its rate is labelled historical rather than today's price.

These reviewed notices do not automatically refresh. Update the module and tests after verifying a new official notification; retain source and season labels. This change does not alter profitability estimates or the legacy non-agentic UI fallback.

For other crops, missing mandi names no longer discard valid district price rows. Such rows retain their dates and show that the mandi name is unavailable.
