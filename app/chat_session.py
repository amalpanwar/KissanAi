"""Start a fresh conversation only when the selected location changes."""
from uuid import uuid4


def sync_chat_location(state, location):
    signature = tuple(str(location.get(key) or '').strip().casefold()
                      for key in ('state', 'district', 'sub_district', 'place'))
    previous = state.get('location_selection_signature')
    changed = previous is not None and tuple(str(value or '').strip().casefold() for value in previous) != signature
    if changed:
        state['chat_history'] = []
        for key in ('last_structured_topic', 'last_structured_context', 'pending_weather_location',
                    'pending_research_offer', 'pending_chat_items', 'pending_selection', 'need_location_correction',
                    'auto_chart', 'auto_forecast_table', 'auto_forecast_caption', 'auto_market_meta',
                    'fc_commodity_override', 'last_chat_submission', 'corr_place', 'corr_district'):
            state.pop(key, None)
        state['show_local_prices_panel'] = False
        state['chat_location_epoch'] = int(state.get('chat_location_epoch', 0)) + 1
        state['session_id'] = str(uuid4())
    if previous is None or changed:
        state['last_location_context'] = dict(location)
    state['location_selection_signature'] = signature
    return changed
