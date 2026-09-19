"""Town/village selection, independently testable without the full application."""
from app.location_selection import place_options, place_label


def render_place_selector(state: str, district: str, *, key: str = 'fc_place') -> dict:
    import streamlit as st

    choices = place_options(state, district)
    options = [''] + list(choices)
    if st.session_state.get(key, '') not in options:
        st.session_state[key] = ''
    selected = st.selectbox(
        'Town / Village', options, key=key,
        format_func=lambda value: place_label(choices[value]) if value else 'Whole district',
        disabled=not choices,
        help='Places are scoped to the selected state and district. The second name is the tehsil.',
    )
    if not choices:
        st.caption('Town/village coverage is not yet available for this district.')
    return choices.get(selected) or {'state': state, 'district': district, 'place': '', 'sub_district': '', 'match_level': 'district'}
